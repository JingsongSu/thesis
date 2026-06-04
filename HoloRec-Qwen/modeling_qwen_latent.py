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
    Simple teacher-forced implicit interleaved wrapper for HoloRec-Qwen.

    Core design:

    Visible sequence:
        fine_1, fine_2, fine_3, ...

    Hidden training input embedding sequence:
        prompt,
        gold_coarse_1_emb, gold_fine_1_emb,
        gold_coarse_2_emb, gold_fine_2_emb,
        ...

    Loss positions:
        prompt last hidden state predicts coarse_1
        coarse_1 position predicts fine_1
        fine_1 position predicts coarse_2
        coarse_2 position predicts fine_2
        ...

    Important:
        - coarse tokens are never visible text tokens.
        - coarse tokens are inserted only as embeddings.
        - fine tokens are the explicit recommendation code tokens.
        - training uses gold coarse embedding, like teacher forcing.
        - inference should use predicted soft coarse embedding.
    """

    def __init__(
        self,
        base_model,
        pad_token_id: int,
        temperature: float = 1.0,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        coarse_align_weight: float = 0.0,
        use_train_cache: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.base_model = base_model
        self.pad_token_id = int(pad_token_id)

        # temperature is kept for compatibility with old lora_finetune.py.
        # It is mainly used at inference when converting coarse logits to soft emb.
        self.temperature = float(temperature)

        self.coarse_loss_weight = float(coarse_loss_weight)
        self.fine_loss_weight = float(fine_loss_weight)

        # In this simplified teacher-forced version we do not need alignment loss,
        # because training directly inserts gold coarse embeddings.
        self.coarse_align_weight = float(coarse_align_weight)

        # Kept only for compatibility with current lora_finetune.py.
        # This wrapper intentionally does not use KV cache during training.
        self.use_train_cache = False

        self.config = getattr(base_model, "config", None)

        self.coarse_position_token_ids: Optional[List[List[int]]] = None
        self._coarse_codebook_cache = {}

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
        # For this wrapper, gradient checkpointing is not necessary and can make
        # repeated embedding-input logic harder to debug. Keep it disabled.
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

        if hasattr(module, "weight"):
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

    def _build_interleaved_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build teacher-forced implicit interleaved embedding sequence.

        Input:
            input_ids:      [B, P]
            attention_mask: [B, P]
            fine_labels:    [B, L]
            coarse_labels:  [B, L]

        Output:
            inputs_embeds:
                [B, P + 2L, H]
                prompt + coarse_1_emb + fine_1_emb + ...

            full_attention_mask:
                [B, P + 2L]

            valid_code_mask:
                [B, L]
        """
        batch_size, code_len = fine_labels.shape

        prompt_embeds = self._embed_token_ids(input_ids)

        valid_code_mask = (
            fine_labels.ne(IGNORE_INDEX)
            & coarse_labels.ne(IGNORE_INDEX)
        )

        safe_fine_ids = fine_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )
        safe_coarse_ids = coarse_labels.masked_fill(
            ~valid_code_mask,
            self.pad_token_id,
        )

        fine_embeds = self._embed_token_ids(safe_fine_ids)
        coarse_embeds = self._embed_token_ids(safe_coarse_ids)

        pieces = [prompt_embeds]
        attn_pieces = [attention_mask]

        for pos in range(code_len):
            valid = valid_code_mask[:, pos:pos + 1].to(dtype=attention_mask.dtype)

            # coarse is implicit: inserted as embedding only
            pieces.append(coarse_embeds[:, pos:pos + 1, :])
            attn_pieces.append(valid)

            # fine is explicit token, but during training we feed its embedding
            # as normal teacher forcing.
            pieces.append(fine_embeds[:, pos:pos + 1, :])
            attn_pieces.append(valid)

        inputs_embeds = torch.cat(pieces, dim=1)
        full_attention_mask = torch.cat(attn_pieces, dim=1)

        return inputs_embeds, full_attention_mask, valid_code_mask

    def _gather_interleaved_logits(
        self,
        logits: torch.Tensor,
        prompt_len: int,
        code_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For sequence:
            prompt, c1, f1, c2, f2, ...

        Causal LM logits at position t predict position t+1.

        Therefore:
            coarse_i is predicted at:
                i=0: prompt_len - 1
                i>0: previous fine position

            fine_i is predicted at:
                coarse_i position
        """
        device = logits.device
        pos = torch.arange(code_len, dtype=torch.long, device=device)

        coarse_predict_positions = (prompt_len - 1) + 2 * pos
        fine_predict_positions = prompt_len + 2 * pos

        coarse_logits = logits.index_select(
            dim=1,
            index=coarse_predict_positions,
        )
        fine_logits = logits.index_select(
            dim=1,
            index=fine_predict_positions,
        )

        return coarse_logits, fine_logits

    def _coarse_loss(
        self,
        coarse_logits: torch.Tensor,
        coarse_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarse loss is restricted to the coarse codebook at each position.

        coarse_logits: [B, L, V]
        coarse_labels: [B, L]
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

            target_pos = matches.float().argmax(dim=1).long()

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

        fine_logits: [B, L, V]
        fine_labels: [B, L]
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

        inputs_embeds, full_attention_mask, valid_code_mask = self._build_interleaved_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            fine_labels=fine_labels,
            coarse_labels=coarse_labels,
        )

        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        coarse_logits, fine_logits = self._gather_interleaved_logits(
            logits=outputs.logits,
            prompt_len=prompt_len,
            code_len=code_len,
        )

        coarse_loss_sum, coarse_count = self._coarse_loss(
            coarse_logits=coarse_logits,
            coarse_labels=coarse_labels,
        )

        fine_loss_sum, fine_count = self._fine_loss(
            fine_logits=fine_logits,
            fine_labels=fine_labels,
        )

        loss = coarse_loss_sum.float() * 0.0

        loss = loss + self.coarse_loss_weight * (
            coarse_loss_sum.float() / coarse_count.float().clamp_min(1.0)
        )

        loss = loss + self.fine_loss_weight * (
            fine_loss_sum.float() / fine_count.float().clamp_min(1.0)
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=fine_logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )



