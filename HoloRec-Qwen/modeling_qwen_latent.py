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
    TIGER-style latent coarse-to-fine wrapper for HoloRec-Qwen.

    Goal:
    - coarse_loss: context -> coarse token CE
    - fine_loss: context -> predicted soft coarse embedding -> fine token CE
    - coarse_align_loss: predicted soft coarse embedding aligns to gold coarse embedding

    Total loss:
        loss = coarse_loss_weight * coarse_loss
             + fine_loss_weight * fine_loss
             + coarse_align_weight * coarse_align_loss
    """

    def __init__(
        self,
        base_model,
        pad_token_id: int,
        temperature: float = 1.0,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        coarse_align_weight: float = 1.0,
        use_train_cache: bool = False,
        align_loss_type: str = "mse_cosine",
        **kwargs,
    ):
        super().__init__()

        self.base_model = base_model
        self.pad_token_id = int(pad_token_id)
        self.temperature = float(temperature)

        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)
        self.coarse_align_weight = float(coarse_align_weight)

        self.align_loss_type = str(align_loss_type)

        # Kept only for compatibility with lora_finetune.py.
        # This wrapper intentionally does not use KV cache during training.
        self.use_train_cache = False

        self.config = getattr(base_model, "config", None)
        self.coarse_position_token_ids: Optional[List[List[int]]] = None
        self._coarse_codebook_cache = {}

        # Trainer reads this dict after every training forward.
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
        # This wrapper runs the inner model twice in one forward.
        # Gradient checkpointing can cause repeated-reduction issues with DDP/ZeRO.
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

        ModulesToSaveWrapper itself may not expose .weight directly.
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
                    if hasattr(candidate, "weight") and callable(
                        getattr(candidate, "forward", None)
                    ):
                        return candidate

            for _, candidate in modules_to_save.items():
                if hasattr(candidate, "weight") and callable(
                    getattr(candidate, "forward", None)
                ):
                    return candidate

        original_module = getattr(module, "original_module", None)
        if original_module is not None and hasattr(original_module, "weight"):
            return original_module

        base_layer = getattr(module, "base_layer", None)
        if base_layer is not None and hasattr(base_layer, "weight"):
            return base_layer

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

    def _base_forward_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return outputs.logits

    def _build_fine_only_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build pass-1 teacher-forced input:
            prompt, gold_fine_1, gold_fine_2, ...

        This pass predicts coarse tokens.
        """
        valid_code_mask = fine_labels.ne(IGNORE_INDEX) & coarse_labels.ne(IGNORE_INDEX)

        safe_fine_ids = fine_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )

        prompt_embeds = self._embed_token_ids(input_ids)
        fine_embeds = self._embed_token_ids(safe_fine_ids)

        pieces = [prompt_embeds]
        attn_pieces = [attention_mask]

        code_len = fine_labels.size(1)
        for pos in range(code_len):
            valid = valid_code_mask[:, pos : pos + 1].to(dtype=attention_mask.dtype)
            pieces.append(fine_embeds[:, pos : pos + 1, :])
            attn_pieces.append(valid)

        inputs_embeds = torch.cat(pieces, dim=1)
        full_attention_mask = torch.cat(attn_pieces, dim=1)

        return inputs_embeds, full_attention_mask, valid_code_mask

    def _gather_coarse_logits_from_fine_only(
        self,
        logits: torch.Tensor,
        prompt_len: int,
        code_len: int,
    ) -> torch.Tensor:
        """
        For pass-1 sequence:
            prompt, f1, f2, f3, ...

        Coarse prediction positions:
            coarse_1 predicted at prompt last: prompt_len - 1
            coarse_2 predicted at f1:          prompt_len
            coarse_3 predicted at f2:          prompt_len + 1
        """
        device = logits.device
        pos = torch.arange(code_len, dtype=torch.long, device=device)
        coarse_predict_positions = (prompt_len - 1) + pos

        coarse_logits = logits.index_select(
            dim=1,
            index=coarse_predict_positions,
        )
        return coarse_logits

    def _build_predicted_interleaved_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        predicted_coarse_embeds: torch.Tensor,
        fine_labels: torch.Tensor,
        valid_code_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build pass-2 input:
            prompt, predicted_soft_coarse_1_emb, gold_fine_1_emb,
            predicted_soft_coarse_2_emb, gold_fine_2_emb, ...
        """
        prompt_embeds = self._embed_token_ids(input_ids)

        safe_fine_ids = fine_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )
        fine_embeds = self._embed_token_ids(safe_fine_ids)

        pieces = [prompt_embeds]
        attn_pieces = [attention_mask]

        code_len = fine_labels.size(1)
        for pos in range(code_len):
            valid = valid_code_mask[:, pos : pos + 1].to(dtype=attention_mask.dtype)

            pieces.append(predicted_coarse_embeds[:, pos : pos + 1, :])
            attn_pieces.append(valid)

            pieces.append(fine_embeds[:, pos : pos + 1, :])
            attn_pieces.append(valid)

        inputs_embeds = torch.cat(pieces, dim=1)
        full_attention_mask = torch.cat(attn_pieces, dim=1)

        return inputs_embeds, full_attention_mask

    def _gather_fine_logits_from_interleaved(
        self,
        logits: torch.Tensor,
        prompt_len: int,
        code_len: int,
    ) -> torch.Tensor:
        """
        For pass-2 sequence:
            prompt, c1, f1, c2, f2, ...

        Causal LM logits at position t predict position t+1.
        fine_i is predicted at c_i position:
            c1 position = prompt_len
            c2 position = prompt_len + 2
            c3 position = prompt_len + 4
        """
        device = logits.device
        pos = torch.arange(code_len, dtype=torch.long, device=device)
        fine_predict_positions = prompt_len + 2 * pos

        fine_logits = logits.index_select(
            dim=1,
            index=fine_predict_positions,
        )
        return fine_logits

    def _coarse_loss(
        self,
        coarse_logits: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarse loss is restricted to the coarse codebook at each position.
        """
        loss_sum = coarse_logits.new_zeros(())
        count = coarse_logits.new_zeros(())

        _, code_len, _ = coarse_logits.shape

        for pos in range(code_len):
            target_ids = coarse_labels[:, pos]
            valid = target_ids.ne(IGNORE_INDEX)
            if not valid.any():
                continue

            codebook_ids = self._get_codebook_tensor(pos, coarse_logits.device)
            pos_logits = coarse_logits[:, pos, :]
            codebook_logits = pos_logits.index_select(
                dim=-1,
                index=codebook_ids,
            )

            matches = codebook_ids.unsqueeze(0).eq(target_ids.unsqueeze(1))
            in_codebook = matches.any(dim=1)

            valid = valid & in_codebook
            if not valid.any():
                continue

            target_pos = matches.long().argmax(dim=1)

            ce = F.cross_entropy(
                codebook_logits[valid].float(),
                target_pos[valid],
                reduction="sum",
            )

            loss_sum = loss_sum + ce.to(dtype=coarse_logits.dtype)
            count = count + valid.sum().to(dtype=coarse_logits.dtype)

        return loss_sum, count

    def _fine_loss(
        self,
        fine_logits: torch.Tensor,
        fine_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fine token loss is normal full-vocabulary CE.
        """
        valid = fine_labels.ne(IGNORE_INDEX)
        if not valid.any():
            return fine_logits.new_zeros(()), fine_logits.new_zeros(())

        ce = F.cross_entropy(
            fine_logits[valid].float(),
            fine_labels[valid],
            reduction="sum",
        )

        return ce.to(dtype=fine_logits.dtype), valid.sum().to(dtype=fine_logits.dtype)

    def _predicted_soft_coarse_embeddings(
        self,
        coarse_logits: torch.Tensor,
        valid_code_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert predicted coarse logits to soft coarse embeddings.
        """
        batch_size, code_len, _ = coarse_logits.shape

        embedding_weight = self._get_embedding_weight()

        pad_ids = torch.full(
            (batch_size,),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=coarse_logits.device,
        )
        pad_emb = self._embed_token_ids(pad_ids).to(dtype=embedding_weight.dtype)

        predicted_embeds = []

        for pos in range(code_len):
            codebook_ids = self._get_codebook_tensor(pos, coarse_logits.device)
            codebook_logits = coarse_logits[:, pos, :].index_select(
                dim=-1,
                index=codebook_ids,
            )

            probs = F.softmax(
                codebook_logits.float() / max(float(self.temperature), 1e-6),
                dim=-1,
            ).to(dtype=embedding_weight.dtype)

            codebook_embeds = embedding_weight.index_select(
                dim=0,
                index=codebook_ids,
            ).to(dtype=probs.dtype)

            soft_coarse_emb = probs @ codebook_embeds

            valid_pos = valid_code_mask[:, pos].to(dtype=torch.bool)
            soft_coarse_emb = torch.where(
                valid_pos.unsqueeze(-1),
                soft_coarse_emb,
                pad_emb,
            )

            predicted_embeds.append(soft_coarse_emb)

        return torch.stack(predicted_embeds, dim=1)

    def _gold_coarse_embeddings(
        self,
        coarse_labels: torch.Tensor,
        valid_code_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_coarse_ids = coarse_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )
        return self._embed_token_ids(safe_coarse_ids)

    def _coarse_align_loss(
        self,
        predicted_coarse_embeds: torch.Tensor,
        gold_coarse_embeds: torch.Tensor,
        valid_code_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align predicted soft coarse embeddings to gold coarse embeddings.
        """
        valid = valid_code_mask.to(dtype=torch.bool)

        if not valid.any():
            return (
                predicted_coarse_embeds.new_zeros(()),
                predicted_coarse_embeds.new_zeros(()),
            )

        pred = predicted_coarse_embeds[valid].float()
        gold = gold_coarse_embeds[valid].float()

        loss_sum = predicted_coarse_embeds.new_zeros(())
        align_type = self.align_loss_type.lower()

        if align_type in ["mse", "mse_only"]:
            mse = F.mse_loss(pred, gold, reduction="none").mean(dim=-1)
            loss_sum = mse.sum().to(dtype=predicted_coarse_embeds.dtype)

        elif align_type in ["cos", "cosine", "cosine_only"]:
            cos_loss = 1.0 - F.cosine_similarity(pred, gold, dim=-1)
            loss_sum = cos_loss.sum().to(dtype=predicted_coarse_embeds.dtype)

        elif align_type in ["mse_cosine", "cosine_mse", "mse+cosine"]:
            mse = F.mse_loss(pred, gold, reduction="none").mean(dim=-1)
            cos_loss = 1.0 - F.cosine_similarity(pred, gold, dim=-1)
            loss_sum = (mse + cos_loss).sum().to(dtype=predicted_coarse_embeds.dtype)

        else:
            raise ValueError(
                f"Unknown align_loss_type: {self.align_loss_type}. "
                "Supported: mse, cosine, mse_cosine."
            )

        count = valid.sum().to(dtype=predicted_coarse_embeds.dtype)
        return loss_sum, count

    @staticmethod
    def _scalar_to_float(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return float(x.detach().float().cpu())
        return float(x)

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

        # ------------------------------------------------------------------
        # Pass 1:
        #   prompt + gold_fine
        #   -> coarse logits
        #   -> predicted soft coarse embeddings
        #   -> coarse CE + coarse alignment
        # ------------------------------------------------------------------
        (
            fine_only_inputs_embeds,
            fine_only_attention_mask,
            valid_code_mask,
        ) = self._build_fine_only_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
        )

        fine_only_logits = self._base_forward_logits(
            inputs_embeds=fine_only_inputs_embeds,
            attention_mask=fine_only_attention_mask,
        )

        coarse_logits = self._gather_coarse_logits_from_fine_only(
            logits=fine_only_logits,
            prompt_len=prompt_len,
            code_len=code_len,
        )

        coarse_loss_sum, coarse_count = self._coarse_loss(
            coarse_logits=coarse_logits,
            coarse_labels=coarse_labels,
        )

        predicted_coarse_embeds = self._predicted_soft_coarse_embeddings(
            coarse_logits=coarse_logits,
            valid_code_mask=valid_code_mask,
        )

        gold_coarse_embeds = self._gold_coarse_embeddings(
            coarse_labels=coarse_labels,
            valid_code_mask=valid_code_mask,
        )

        coarse_align_loss_sum, coarse_align_count = self._coarse_align_loss(
            predicted_coarse_embeds=predicted_coarse_embeds,
            gold_coarse_embeds=gold_coarse_embeds,
            valid_code_mask=valid_code_mask,
        )

        # ------------------------------------------------------------------
        # Pass 2:
        #   prompt + predicted_soft_coarse + gold_fine
        #   -> fine logits
        #   -> fine CE
        # ------------------------------------------------------------------
        pred_inputs_embeds, pred_attention_mask = self._build_predicted_interleaved_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            predicted_coarse_embeds=predicted_coarse_embeds,
            fine_labels=fine_labels,
            valid_code_mask=valid_code_mask,
        )

        pred_logits = self._base_forward_logits(
            inputs_embeds=pred_inputs_embeds,
            attention_mask=pred_attention_mask,
        )

        fine_logits = self._gather_fine_logits_from_interleaved(
            logits=pred_logits,
            prompt_len=prompt_len,
            code_len=code_len,
        )

        fine_loss_sum, fine_count = self._fine_loss(
            fine_logits=fine_logits,
            fine_labels=fine_labels,
        )

        coarse_loss = coarse_loss_sum.float() / coarse_count.float().clamp_min(1.0)
        fine_loss = fine_loss_sum.float() / fine_count.float().clamp_min(1.0)
        coarse_align_loss = (
            coarse_align_loss_sum.float()
            / coarse_align_count.float().clamp_min(1.0)
        )

        weighted_coarse_loss = self.coarse_loss_weight * coarse_loss
        weighted_fine_loss = self.fine_loss_weight * fine_loss
        weighted_coarse_align_loss = self.coarse_align_weight * coarse_align_loss

        loss = weighted_coarse_loss + weighted_fine_loss + weighted_coarse_align_loss

        # This dict is read by HoloRecTrainer and logged every logging_step.
        # The *_weighted values are the coefficient-multiplied values.
        self.last_loss_dict = {
            "coarse_loss": self._scalar_to_float(coarse_loss),
            "fine_loss": self._scalar_to_float(fine_loss),
            "coarse_align_loss": self._scalar_to_float(coarse_align_loss),

            "weighted_coarse_loss": self._scalar_to_float(weighted_coarse_loss),
            "weighted_fine_loss": self._scalar_to_float(weighted_fine_loss),
            "weighted_coarse_align_loss": self._scalar_to_float(
                weighted_coarse_align_loss
            ),
            "weighted_loss_sum": self._scalar_to_float(loss),

            "coarse_count": self._scalar_to_float(coarse_count),
            "fine_count": self._scalar_to_float(fine_count),
            "coarse_align_count": self._scalar_to_float(coarse_align_count),

            "coarse_loss_weight": float(self.coarse_loss_weight),
            "fine_loss_weight": float(self.fine_loss_weight),
            "coarse_align_weight": float(self.coarse_align_weight),
            "temperature": float(self.temperature),
            "align_loss_type": self.align_loss_type,
            "mode": "tiger_style_qwen_two_pass",
        }

        return CausalLMOutputWithPast(
            loss=loss,
            logits=fine_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
