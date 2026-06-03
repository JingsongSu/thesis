from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_input_embeddings(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        emb = model.base_model.get_input_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens

    raise ValueError("Cannot locate input embeddings on base model.")


def _get_output_embeddings(model: nn.Module) -> Optional[nn.Module]:
    if hasattr(model, "get_output_embeddings"):
        emb = model.get_output_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "base_model") and hasattr(model.base_model, "get_output_embeddings"):
        emb = model.base_model.get_output_embeddings()
        if emb is not None:
            return emb

    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head

    return None


def _get_active_adapter_name(module: nn.Module):
    active = getattr(module, "active_adapter", None)

    if callable(active):
        try:
            active = active()
        except TypeError:
            pass

    if isinstance(active, (list, tuple)):
        if len(active) == 0:
            return None
        return active[0]

    return active


def _get_wrapped_module_weight(module: nn.Module) -> torch.Tensor:
    """
    Return `.weight` from normal modules and PEFT ModulesToSaveWrapper.

    With LoRA + modules_to_save=["embed_tokens", "lm_head"], PEFT may replace
    the embedding/lm_head with ModulesToSaveWrapper. That wrapper can be called
    in forward, but it does not expose `.weight` directly.
    """
    if hasattr(module, "weight"):
        weight = getattr(module, "weight")
        if isinstance(weight, torch.Tensor):
            return weight
        if isinstance(weight, nn.Parameter):
            return weight

    modules_to_save = getattr(module, "modules_to_save", None)

    if modules_to_save is not None:
        active = _get_active_adapter_name(module)

        if active is not None and active in modules_to_save:
            sub = modules_to_save[active]
            if hasattr(sub, "weight"):
                return sub.weight

        if "default" in modules_to_save:
            sub = modules_to_save["default"]
            if hasattr(sub, "weight"):
                return sub.weight

        try:
            for _, sub in modules_to_save.items():
                if hasattr(sub, "weight"):
                    return sub.weight
        except Exception:
            pass

    original_module = getattr(module, "original_module", None)
    if original_module is not None and hasattr(original_module, "weight"):
        return original_module.weight

    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None and hasattr(base_layer, "weight"):
        return base_layer.weight

    raise AttributeError(
        f"Cannot locate `.weight` inside module type {type(module).__name__}. "
        "This is usually caused by an unsupported PEFT wrapper around embeddings."
    )


def _last_non_pad_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Return logits at the last real prompt token for each row.

    Works for both left padding and right padding.
    """
    seq_len = attention_mask.size(1)
    pos = torch.arange(seq_len, device=attention_mask.device).view(1, -1)
    pos = pos.expand_as(attention_mask)
    last_pos = pos.masked_fill(attention_mask.eq(0), -1).max(dim=1).values
    last_pos = last_pos.clamp(min=0)

    gather_idx = last_pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    return logits.gather(dim=1, index=gather_idx).squeeze(1)


class InterleavedLatentQwen(nn.Module):
    """
    Prompt-only implicit HoloRec-Qwen wrapper.

    Visible sequence:
        prompt + fine tokens

    Hidden transition per item-code position:
        state_t
        -> coarse_logits_t over position-specific coarse codebook
        -> soft latent coarse embedding_t
        -> fine_logits_t
        -> gold fine embedding_t during training
        -> state_{t+1}

    Coarse tokens are never decoded as visible tokens.
    """

    def __init__(
        self,
        base_model: nn.Module,
        pad_token_id: int,
        temperature: float = 0.8,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        coarse_align_weight: float = 0.0,
        use_train_cache: bool = True,
        **unused_kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.pad_token_id = int(pad_token_id)
        self.temperature = float(temperature)
        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)
        self.coarse_align_weight = float(coarse_align_weight)
        self.use_train_cache = bool(use_train_cache)
        self.config = getattr(base_model, "config", None)

        self.coarse_position_token_ids: List[torch.Tensor] = []
        self.coarse_id_to_local: List[torch.Tensor] = []

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            base_model = self.__dict__.get("_modules", {}).get("base_model", None)
            if base_model is not None and hasattr(base_model, name):
                return getattr(base_model, name)
            raise exc

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """
        Keep disabled.

        This wrapper calls the same base/PEFT model multiple times in one forward.
        Gradient checkpointing plus ZeRO can produce repeated reduction errors,
        and gradient checkpointing is also incompatible with cache-based training.
        """
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return _get_input_embeddings(self.base_model)

    def get_output_embeddings(self):
        return _get_output_embeddings(self.base_model)

    def get_input_embedding_weight(self) -> torch.Tensor:
        return _get_wrapped_module_weight(self.get_input_embeddings())

    def get_output_embedding_weight(self) -> Optional[torch.Tensor]:
        out = self.get_output_embeddings()
        if out is None:
            return None
        return _get_wrapped_module_weight(out)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Use module forward instead of direct F.embedding so PEFT wrappers remain valid.
        """
        return self.get_input_embeddings()(input_ids)

    def save_pretrained(self, *args, **kwargs):
        if hasattr(self.base_model, "save_pretrained"):
            return self.base_model.save_pretrained(*args, **kwargs)
        raise AttributeError("base_model has no save_pretrained")

    def set_codebooks(self, coarse_position_token_ids: Sequence[Sequence[int]]):
        """
        Register per-position coarse candidate ids.

        coarse_position_token_ids[pos] is the valid coarse-token id list for
        position pos. Buffers are non-persistent, so checkpoints stay clean.
        """
        for name in list(self._buffers.keys()):
            if name.startswith("coarse_ids_") or name.startswith("coarse_map_"):
                del self._buffers[name]

        self.coarse_position_token_ids = []
        self.coarse_id_to_local = []

        vocab_size = int(getattr(getattr(self.base_model, "config", None), "vocab_size", 0) or 0)

        if vocab_size <= 0:
            out_weight = self.get_output_embedding_weight()
            if out_weight is not None:
                vocab_size = int(out_weight.size(0))
            else:
                vocab_size = int(self.get_input_embedding_weight().size(0))

        for pos, ids in enumerate(coarse_position_token_ids):
            ids_tensor = torch.tensor([int(x) for x in ids], dtype=torch.long)

            if ids_tensor.numel() == 0:
                raise ValueError(f"Empty coarse codebook at position {pos}.")

            max_id = int(ids_tensor.max().item())
            if max_id >= vocab_size:
                vocab_size = max_id + 1

            map_tensor = torch.full((vocab_size,), fill_value=-1, dtype=torch.long)
            map_tensor[ids_tensor] = torch.arange(ids_tensor.numel(), dtype=torch.long)

            self.register_buffer(f"coarse_ids_{pos}", ids_tensor, persistent=False)
            self.register_buffer(f"coarse_map_{pos}", map_tensor, persistent=False)

            self.coarse_position_token_ids.append(getattr(self, f"coarse_ids_{pos}"))
            self.coarse_id_to_local.append(getattr(self, f"coarse_map_{pos}"))

    def _call_base(
        self,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
    ):
        if use_cache is None:
            use_cache = self.use_train_cache

        return self.base_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=bool(use_cache),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

    def _soft_coarse_embedding(
        self,
        coarse_logits: torch.Tensor,
        pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        code_pos = min(pos, len(self.coarse_position_token_ids) - 1)
        candidate_ids = self.coarse_position_token_ids[code_pos].to(coarse_logits.device)

        candidate_logits = coarse_logits.index_select(dim=-1, index=candidate_ids)

        probs = F.softmax(
            candidate_logits.float() / max(self.temperature, 1e-6),
            dim=-1,
        ).to(coarse_logits.dtype)

        emb_weight = self.get_input_embedding_weight().to(coarse_logits.device)

        candidate_emb = emb_weight.index_select(dim=0, index=candidate_ids)
        candidate_emb = candidate_emb.to(probs.dtype)

        coarse_emb = probs @ candidate_emb

        return coarse_emb, candidate_logits, candidate_ids, code_pos

    def _coarse_ce(
        self,
        candidate_logits: torch.Tensor,
        code_pos: int,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.to(candidate_logits.device)
        local = torch.full_like(labels, fill_value=IGNORE_INDEX)

        valid = labels.ne(IGNORE_INDEX)

        if valid.any():
            lookup = self.coarse_id_to_local[code_pos].to(labels.device)

            safe_labels = labels[valid]
            in_range = (safe_labels >= 0) & (safe_labels < lookup.numel())

            mapped = torch.full_like(safe_labels, fill_value=IGNORE_INDEX)

            if in_range.any():
                mapped_in_range = lookup.index_select(0, safe_labels[in_range])
                mapped[in_range] = mapped_in_range.masked_fill(mapped_in_range.lt(0), IGNORE_INDEX)

            local[valid] = mapped

        return F.cross_entropy(
            candidate_logits.float(),
            local,
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fine_labels: Optional[torch.Tensor] = None,
        coarse_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if fine_labels is None or coarse_labels is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        if len(self.coarse_position_token_ids) == 0:
            raise ValueError("Call InterleavedLatentQwen.set_codebooks(...) before training.")

        if self.use_train_cache:
            return self._forward_with_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                fine_labels=fine_labels,
                coarse_labels=coarse_labels,
            )

        return self._forward_recompute(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
        )

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ):
        device = _model_device(self.base_model)
        input_ids = input_ids.to(device)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        attention_mask = attention_mask.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)

        outputs = self._call_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # Important:
        # Do NOT convert this to legacy tuple.
        # Qwen3 expects a Cache/DynamicCache object and calls get_seq_length().
        past = outputs.past_key_values

        coarse_logits = _last_non_pad_logits(outputs.logits, attention_mask)
        cur_attention = attention_mask

        total_loss = coarse_logits.new_zeros(())
        total_weight = 0.0
        fine_logits_for_return = []
        code_len = fine_labels.size(1)

        for pos in range(code_len):
            gold_coarse = coarse_labels[:, pos]
            gold_fine = fine_labels[:, pos]

            active = gold_fine.ne(IGNORE_INDEX) & gold_coarse.ne(IGNORE_INDEX)
            if not active.any():
                break

            coarse_emb, candidate_logits, _candidate_ids, code_pos = self._soft_coarse_embedding(
                coarse_logits=coarse_logits,
                pos=pos,
            )

            coarse_loss = self._coarse_ce(
                candidate_logits=candidate_logits,
                code_pos=code_pos,
                labels=gold_coarse,
            )

            coarse_w = self.coarse_loss_weight + self.coarse_align_weight

            if coarse_w != 0.0:
                total_loss = total_loss + coarse_w * coarse_loss
                total_weight += abs(coarse_w)

            one = torch.ones(
                (input_ids.size(0), 1),
                dtype=cur_attention.dtype,
                device=cur_attention.device,
            )

            attention_after_coarse = torch.cat([cur_attention, one], dim=1)

            out_after_coarse = self._call_base(
                inputs_embeds=coarse_emb.unsqueeze(1),
                attention_mask=attention_after_coarse,
                past_key_values=past,
                use_cache=True,
            )

            # Keep Cache/DynamicCache object.
            past_after_coarse = out_after_coarse.past_key_values

            fine_logits = out_after_coarse.logits[:, -1, :]
            fine_logits_for_return.append(fine_logits.unsqueeze(1))

            fine_loss = F.cross_entropy(
                fine_logits.float(),
                gold_fine,
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

            if self.fine_loss_weight != 0.0:
                total_loss = total_loss + self.fine_loss_weight * fine_loss
                total_weight += abs(self.fine_loss_weight)

            safe_fine = gold_fine.masked_fill(
                gold_fine.eq(IGNORE_INDEX),
                self.pad_token_id,
            )

            fine_emb = self.embed_input_ids(safe_fine).unsqueeze(1)

            attention_after_fine = torch.cat([attention_after_coarse, one], dim=1)

            out_after_fine = self._call_base(
                inputs_embeds=fine_emb,
                attention_mask=attention_after_fine,
                past_key_values=past_after_coarse,
                use_cache=True,
            )

            # Keep Cache/DynamicCache object.
            past = out_after_fine.past_key_values

            coarse_logits = out_after_fine.logits[:, -1, :]
            cur_attention = attention_after_fine

        if total_weight == 0.0:
            total_loss = total_loss + outputs.logits.sum() * 0.0

        if fine_logits_for_return:
            logits = torch.cat(fine_logits_for_return, dim=1)
        else:
            logits = outputs.logits[:, -1:, :]

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
        )

    def _forward_recompute(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ):
        """
        Correct but slower no-cache fallback.

        This exists only for compatibility with scripts that pass
        use_train_cache=False. For speed, use_train_cache=True is recommended.
        """
        device = _model_device(self.base_model)
        input_ids = input_ids.to(device)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        attention_mask = attention_mask.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)

        seq_embeds = self.embed_input_ids(input_ids)
        cur_attention = attention_mask

        outputs = self._call_base(
            inputs_embeds=seq_embeds,
            attention_mask=cur_attention,
            use_cache=False,
        )

        coarse_logits = _last_non_pad_logits(outputs.logits, cur_attention)

        total_loss = coarse_logits.new_zeros(())
        total_weight = 0.0
        fine_logits_for_return = []
        code_len = fine_labels.size(1)

        for pos in range(code_len):
            gold_coarse = coarse_labels[:, pos]
            gold_fine = fine_labels[:, pos]

            active = gold_fine.ne(IGNORE_INDEX) & gold_coarse.ne(IGNORE_INDEX)
            if not active.any():
                break

            coarse_emb, candidate_logits, _candidate_ids, code_pos = self._soft_coarse_embedding(
                coarse_logits=coarse_logits,
                pos=pos,
            )

            coarse_loss = self._coarse_ce(
                candidate_logits=candidate_logits,
                code_pos=code_pos,
                labels=gold_coarse,
            )

            coarse_w = self.coarse_loss_weight + self.coarse_align_weight

            if coarse_w != 0.0:
                total_loss = total_loss + coarse_w * coarse_loss
                total_weight += abs(coarse_w)

            one = torch.ones(
                (input_ids.size(0), 1),
                dtype=cur_attention.dtype,
                device=cur_attention.device,
            )

            seq_embeds = torch.cat([seq_embeds, coarse_emb.unsqueeze(1)], dim=1)
            cur_attention = torch.cat([cur_attention, one], dim=1)

            out_after_coarse = self._call_base(
                inputs_embeds=seq_embeds,
                attention_mask=cur_attention,
                use_cache=False,
            )

            fine_logits = out_after_coarse.logits[:, -1, :]
            fine_logits_for_return.append(fine_logits.unsqueeze(1))

            fine_loss = F.cross_entropy(
                fine_logits.float(),
                gold_fine,
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

            if self.fine_loss_weight != 0.0:
                total_loss = total_loss + self.fine_loss_weight * fine_loss
                total_weight += abs(self.fine_loss_weight)

            safe_fine = gold_fine.masked_fill(
                gold_fine.eq(IGNORE_INDEX),
                self.pad_token_id,
            )

            fine_emb = self.embed_input_ids(safe_fine).unsqueeze(1)

            seq_embeds = torch.cat([seq_embeds, fine_emb], dim=1)
            cur_attention = torch.cat([cur_attention, one], dim=1)

            if pos + 1 < code_len:
                out_after_fine = self._call_base(
                    inputs_embeds=seq_embeds,
                    attention_mask=cur_attention,
                    use_cache=False,
                )

                coarse_logits = out_after_fine.logits[:, -1, :]

        if total_weight == 0.0:
            total_loss = total_loss + outputs.logits.sum() * 0.0

        if fine_logits_for_return:
            logits = torch.cat(fine_logits_for_return, dim=1)
        else:
            logits = outputs.logits[:, -1:, :]

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=None,
        )
