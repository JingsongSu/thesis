from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

IGNORE_INDEX = -100


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


class InterleavedLatentQwen(nn.Module):
    """
    Fast HoloRec-Qwen latent wrapper.

    Goal:
        Keep the HoloRec-style core design:
            1. predicted coarse embedding aligns to gold coarse embedding
            2. direct fine-code loss
            3. coarse-to-fine loss

        But avoid repeated Qwen forward calls.

    Old slow exact path:
        context -> Qwen -> coarse soft emb
        context + coarse soft emb -> Qwen again -> fine
        repeated for each code position

    This fast path:
        prompt + gold previous fine tokens -> Qwen once
        hidden_i -> coarse logits_i -> coarse soft emb_i
        hidden_i -> direct fine logits_i
        hidden_i + coarse soft emb_i -> lm_head -> coarse-to-fine logits_i

    Important:
        - gold coarse embedding is NEVER inserted into the Qwen context.
        - gold coarse embedding is only used as alignment target.
        - training and inference use the same latent formula:
              hidden -> predicted coarse emb -> hidden + coarse emb -> lm_head
        - Qwen itself is run only once per training batch.

    Loss:
        loss =
            fine_loss_weight    * fine_loss
          + coarse_loss_weight  * coarse_to_fine_loss
          + coarse_align_weight * coarse_align_loss

    For compatibility with existing lora_finetune.py:
        coarse_loss_weight is reused as coarse_to_fine_loss weight.
    """

    def __init__(
        self,
        base_model,
        pad_token_id: int,
        temperature: float = 1.0,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        coarse_align_weight: float = 2.0,
        use_train_cache: bool = False,
        align_loss_type: str = "mse_cosine",
        coarse_fusion_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.base_model = base_model
        self.pad_token_id = int(pad_token_id)

        self.temperature = float(temperature)
        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)
        self.coarse_align_weight = float(coarse_align_weight)

        self.coarse_to_fine_loss_weight = self.coarse_loss_weight
        self.align_loss_type = str(align_loss_type)
        self.coarse_fusion_scale = float(coarse_fusion_scale)

        # Kept only for compatibility with existing lora_finetune.py.
        # This fast version does not use KV cache during training.
        self.use_train_cache = False

        self.config = getattr(base_model, "config", None)
        self.coarse_position_token_ids: Optional[List[List[int]]] = None
        self._coarse_codebook_cache = {}

        self.last_loss_dict = {}

    def set_codebooks(self, coarse_position_token_ids: List[List[int]]):
        if coarse_position_token_ids is None or len(coarse_position_token_ids) == 0:
            raise ValueError("coarse_position_token_ids must be non-empty.")

        self.coarse_position_token_ids = [
            [int(x) for x in ids] for ids in coarse_position_token_ids
        ]
        self._coarse_codebook_cache = {}

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        if hasattr(self.base_model, "get_output_embeddings"):
            return self.base_model.get_output_embeddings()
        return None

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # This wrapper depends on capturing lm_head input.
        # Gradient checkpointing can make debugging much harder and is not needed
        # for this single-forward version.
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()
        return None

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def _unwrap_modules_to_save_embedding(self, module):
        """
        PEFT may wrap embed_tokens with ModulesToSaveWrapper when using:
            modules_to_save=["embed_tokens", "lm_head"]

        This helper returns the real embedding module.
        """
        if module is None:
            raise AttributeError("Input embedding module is None.")

        if hasattr(module, "weight") and callable(getattr(module, "forward", None)):
            return module

        modules_to_save = getattr(module, "modules_to_save", None)
        if modules_to_save is not None:
            active_names = []

            if hasattr(module, "active_adapter"):
                active_names.extend(_to_list(getattr(module, "active_adapter")))

            if hasattr(module, "active_adapters"):
                active_adapters = getattr(module, "active_adapters")
                if callable(active_adapters):
                    try:
                        active_adapters = active_adapters()
                    except TypeError:
                        active_adapters = None
                active_names.extend(_to_list(active_adapters))

            if hasattr(module, "_active_adapter"):
                active_names.extend(_to_list(getattr(module, "_active_adapter")))

            active_names.append("default")

            seen = set()
            clean_names = []
            for name in active_names:
                if name is None:
                    continue
                name = str(name)
                if name in seen:
                    continue
                seen.add(name)
                clean_names.append(name)

            for name in clean_names:
                if name in modules_to_save:
                    candidate = modules_to_save[name]
                    if hasattr(candidate, "weight"):
                        return candidate

            for _, candidate in modules_to_save.items():
                if hasattr(candidate, "weight"):
                    return candidate

        original_module = getattr(module, "original_module", None)
        if original_module is not None and hasattr(original_module, "weight"):
            return original_module

        child = getattr(module, "module", None)
        if child is not None and child is not module:
            return self._unwrap_modules_to_save_embedding(child)

        raise AttributeError(
            "Cannot find real embedding weight from input embeddings. "
            f"Got module type: {type(module).__name__}."
        )

    def _get_embedding_module(self):
        embedding_module = self.get_input_embeddings()
        return self._unwrap_modules_to_save_embedding(embedding_module)

    def _get_embedding_weight(self):
        return self._get_embedding_module().weight

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_module()(token_ids)

    def _get_lm_head_module(self):
        lm_head = self.get_output_embeddings()
        if lm_head is not None:
            return lm_head

        # Fallback for some Qwen-like model structures.
        if hasattr(self.base_model, "lm_head"):
            return self.base_model.lm_head

        if hasattr(self.base_model, "base_model") and hasattr(self.base_model.base_model, "lm_head"):
            return self.base_model.base_model.lm_head

        raise AttributeError("Cannot locate lm_head / output embeddings module.")

    def _apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm_head = self._get_lm_head_module()
        return lm_head(hidden_states)

    def _get_codebook_tensor(self, pos: int, device: torch.device) -> torch.Tensor:
        if self.coarse_position_token_ids is None:
            raise ValueError(
                "coarse_position_token_ids is None. "
                "Call model.set_codebooks(...) before training."
            )

        if pos >= len(self.coarse_position_token_ids):
            raise ValueError(
                f"Requested coarse codebook position {pos}, "
                f"but only {len(self.coarse_position_token_ids)} positions exist."
            )

        key = (int(pos), str(device))
        if key not in self._coarse_codebook_cache:
            ids = self.coarse_position_token_ids[pos]
            if len(ids) == 0:
                raise ValueError(f"Empty coarse codebook at position {pos}.")

            self._coarse_codebook_cache[key] = torch.tensor(
                ids,
                dtype=torch.long,
                device=device,
            )

        return self._coarse_codebook_cache[key]

    def _build_fine_teacher_forcing_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Qwen input sequence:

            prompt, gold_fine_1, gold_fine_2, ..., gold_fine_L

        No coarse embedding is inserted.

        Logits positions:
            prompt last hidden state predicts fine_1 / coarse_1
            fine_1 hidden state predicts fine_2 / coarse_2
            ...
        """
        valid_code_mask = fine_labels.ne(IGNORE_INDEX) & coarse_labels.ne(IGNORE_INDEX)

        safe_fine_ids = fine_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )

        prompt_embeds = self._embed_token_ids(input_ids)
        fine_embeds = self._embed_token_ids(safe_fine_ids)

        fine_attention = valid_code_mask.to(dtype=attention_mask.dtype)

        inputs_embeds = torch.cat([prompt_embeds, fine_embeds], dim=1)
        full_attention_mask = torch.cat([attention_mask, fine_attention], dim=1)

        return inputs_embeds, full_attention_mask, valid_code_mask

    def _forward_base_and_capture_lm_hidden(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Run base_model once and capture the hidden states that are fed into lm_head.

        This avoids output_hidden_states=True, which would store all layer outputs
        and noticeably increase memory/time.
        """
        lm_head = self._get_lm_head_module()
        captured = {}

        def _pre_hook(module, inputs):
            if len(inputs) > 0:
                captured["hidden"] = inputs[0]

        handle = lm_head.register_forward_pre_hook(_pre_hook)

        try:
            outputs = self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError(
                "Failed to capture lm_head input hidden states. "
                "Please check whether get_output_embeddings() returns the lm_head "
                "actually used by your Qwen model."
            )

        return outputs, captured["hidden"]

    def _gather_position_states(
        self,
        tensor: torch.Tensor,
        prompt_len: int,
        code_len: int,
    ) -> torch.Tensor:
        """
        For sequence:
            prompt, fine_1, fine_2, ...

        Causal LM position t predicts t+1.

        Prediction positions:
            fine/coarse_1: prompt_len - 1
            fine/coarse_2: prompt_len
            fine/coarse_3: prompt_len + 1
            ...
        """
        device = tensor.device
        positions = (prompt_len - 1) + torch.arange(
            code_len,
            dtype=torch.long,
            device=device,
        )
        return tensor.index_select(dim=1, index=positions)

    def _masked_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: [B, L, V]
        labels: [B, L]
        """
        valid = labels.ne(IGNORE_INDEX)
        count = valid.sum().to(device=logits.device, dtype=logits.dtype)

        if not valid.any():
            return logits.new_zeros(()), count

        ce = F.cross_entropy(
            logits[valid].float(),
            labels[valid],
            reduction="mean",
        )
        return ce.to(dtype=logits.dtype), count

    def _coarse_soft_embedding_one_pos(
        self,
        coarse_logits: torch.Tensor,
        pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coarse_logits: [B, V]

        Returns:
            soft_emb:       [B, H]
            codebook_logits:[B, K]
            codebook_ids:   [K]
        """
        codebook_ids = self._get_codebook_tensor(pos, coarse_logits.device)

        codebook_logits = coarse_logits.index_select(
            dim=-1,
            index=codebook_ids,
        )

        probs = F.softmax(
            codebook_logits.float() / max(self.temperature, 1e-6),
            dim=-1,
        )
        probs = probs.to(dtype=self._get_embedding_weight().dtype)

        codebook_emb = self._get_embedding_weight().index_select(
            dim=0,
            index=codebook_ids,
        )

        soft_emb = probs @ codebook_emb
        return soft_emb, codebook_logits, codebook_ids

    def _gold_coarse_embedding(
        self,
        coarse_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_ids = coarse_ids.masked_fill(~valid_mask, self.pad_token_id)
        return self._embed_token_ids(safe_ids)

    def _coarse_alignment_loss_one_pos(
        self,
        pred_emb: torch.Tensor,
        gold_emb: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = valid_mask.to(device=pred_emb.device, dtype=torch.bool)

        if not valid_mask.any():
            zero = pred_emb.new_zeros(())
            return zero, zero

        pred = pred_emb[valid_mask].float()

        # Detach target embedding so this loss mainly teaches the coarse predictor.
        # Fine CE / c2f CE can still update embeddings through normal lm_head path.
        gold = gold_emb[valid_mask].float().detach()

        if self.align_loss_type == "mse":
            per_item = F.mse_loss(pred, gold, reduction="none").mean(dim=-1)
        elif self.align_loss_type == "cosine":
            per_item = 1.0 - F.cosine_similarity(pred, gold, dim=-1)
        else:
            mse = F.mse_loss(pred, gold, reduction="none").mean(dim=-1)
            cosine = 1.0 - F.cosine_similarity(pred, gold, dim=-1)
            per_item = mse + cosine

        loss_sum = per_item.sum().to(dtype=pred_emb.dtype)
        count = valid_mask.sum().to(device=pred_emb.device, dtype=pred_emb.dtype)

        return loss_sum, count

    def _build_predicted_coarse_embeddings_and_align_loss(
        self,
        coarse_source_logits: torch.Tensor,
        coarse_labels: torch.Tensor,
        valid_code_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        coarse_source_logits: [B, L, V]
        coarse_labels:       [B, L]
        valid_code_mask:     [B, L]

        Returns:
            coarse_embeds:      [B, L, H]
            align_loss:         scalar
            align_count:        scalar
        """
        batch_size, code_len, _ = coarse_source_logits.shape

        coarse_embeds = []
        align_loss_sum = coarse_source_logits.new_zeros(())
        align_count = coarse_source_logits.new_zeros(())

        for pos in range(code_len):
            valid_mask = valid_code_mask[:, pos]

            pred_emb_t, _, _ = self._coarse_soft_embedding_one_pos(
                coarse_logits=coarse_source_logits[:, pos, :],
                pos=pos,
            )

            gold_emb_t = self._gold_coarse_embedding(
                coarse_ids=coarse_labels[:, pos],
                valid_mask=valid_mask,
            )

            align_sum_t, align_count_t = self._coarse_alignment_loss_one_pos(
                pred_emb=pred_emb_t,
                gold_emb=gold_emb_t,
                valid_mask=valid_mask,
            )

            align_loss_sum = align_loss_sum + align_sum_t
            align_count = align_count + align_count_t

            # For padded code positions, replace by pad embedding.
            pad_ids = torch.full(
                (batch_size,),
                fill_value=self.pad_token_id,
                dtype=torch.long,
                device=pred_emb_t.device,
            )
            pad_emb = self._embed_token_ids(pad_ids).to(dtype=pred_emb_t.dtype)

            pred_emb_t = torch.where(
                valid_mask.unsqueeze(-1),
                pred_emb_t,
                pad_emb,
            )

            coarse_embeds.append(pred_emb_t)

        coarse_embeds = torch.stack(coarse_embeds, dim=1)

        align_loss = align_loss_sum.float() / align_count.float().clamp_min(1.0)

        return coarse_embeds, align_loss, align_count

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        fine_labels=None,
        coarse_labels=None,
        labels=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        if fine_labels is None and labels is not None:
            fine_labels = labels

        if fine_labels is None or coarse_labels is None:
            raise ValueError(
                "InterleavedLatentQwen requires `fine_labels` and `coarse_labels`."
            )

        if fine_labels.size() != coarse_labels.size():
            raise ValueError(
                "fine_labels and coarse_labels must have the same shape, got "
                f"{tuple(fine_labels.size())} vs {tuple(coarse_labels.size())}."
            )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        prompt_len = input_ids.size(1)
        code_len = fine_labels.size(1)

        inputs_embeds, full_attention_mask, valid_code_mask = (
            self._build_fine_teacher_forcing_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                fine_labels=fine_labels,
                coarse_labels=coarse_labels,
            )
        )

        outputs, lm_hidden = self._forward_base_and_capture_lm_hidden(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )

        # [B, L, V], direct fine prediction from Qwen context.
        direct_fine_logits = self._gather_position_states(
            tensor=outputs.logits,
            prompt_len=prompt_len,
            code_len=code_len,
        )

        # [B, L, H], hidden states at the same prediction positions.
        context_hidden = self._gather_position_states(
            tensor=lm_hidden,
            prompt_len=prompt_len,
            code_len=code_len,
        )

        # Predict coarse soft embedding from the same source logits.
        predicted_coarse_embeds, coarse_align_loss, coarse_align_count = (
            self._build_predicted_coarse_embeddings_and_align_loss(
                coarse_source_logits=direct_fine_logits,
                coarse_labels=coarse_labels,
                valid_code_mask=valid_code_mask,
            )
        )

        # Coarse -> fine path.
        # No second Qwen forward. The predicted coarse embedding is fused into
        # the current Qwen hidden state, then scored by the same lm_head.
        fused_hidden = context_hidden + self.coarse_fusion_scale * predicted_coarse_embeds
        coarse_to_fine_logits = self._apply_lm_head(fused_hidden)

        fine_loss, fine_count = self._masked_ce_loss(
            logits=direct_fine_logits,
            labels=fine_labels,
        )

        coarse_to_fine_loss, coarse_to_fine_count = self._masked_ce_loss(
            logits=coarse_to_fine_logits,
            labels=fine_labels,
        )

        loss = (
            self.fine_loss_weight * fine_loss.float()
            + self.coarse_to_fine_loss_weight * coarse_to_fine_loss.float()
            + self.coarse_align_weight * coarse_align_loss.float()
        )

        self.last_loss_dict = {
            "fine_loss": float(fine_loss.detach().float().cpu()),
            "coarse_to_fine_loss": float(coarse_to_fine_loss.detach().float().cpu()),
            "coarse_align_loss": float(coarse_align_loss.detach().float().cpu()),
            "fine_count": float(fine_count.detach().float().cpu()),
            "coarse_to_fine_count": float(
                coarse_to_fine_count.detach().float().cpu()
            ),
            "coarse_align_count": float(coarse_align_count.detach().float().cpu()),
            "fast_single_qwen_forward": 1.0,
        }

        return CausalLMOutputWithPast(
            loss=loss,
            logits=coarse_to_fine_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )