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
    Implicit interleaved latent wrapper for HoloRec-Qwen.

    Visible decoded sequence:
        fine_1 fine_2 fine_3 ...

    Hidden latent procedure:
        context -> coarse logits
        coarse logits -> soft coarse embedding
        context + soft coarse embedding -> fine logits
        context + soft coarse embedding + gold fine -> next latent coarse step

    Coarse tokens are not placed into visible input_ids and are not decoded.
    They only supervise the latent coarse step.
    """

    def __init__(
        self,
        base_model,
        pad_token_id: int,
        temperature: float = 1.0,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        coarse_align_weight: float = 2.0,
        use_train_cache: bool = True,
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

        self.coarse_position_token_ids: Optional[List[List[int]]] = None
        self._coarse_codebook_cache = {}

    def set_codebooks(self, coarse_position_token_ids: List[List[int]]):
        if coarse_position_token_ids is None or len(coarse_position_token_ids) == 0:
            raise ValueError("coarse_position_token_ids must be non-empty.")

        self.coarse_position_token_ids = [
            [int(x) for x in ids]
            for ids in coarse_position_token_ids
        ]

        self._coarse_codebook_cache = {}

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        if hasattr(self.base_model, "get_output_embeddings"):
            return self.base_model.get_output_embeddings()
        return None

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            return self.base_model.gradient_checkpointing_disable()

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)

    def _unwrap_modules_to_save_embedding(self, module):
        """
        PEFT may wrap embed_tokens with ModulesToSaveWrapper when using:

            modules_to_save=["embed_tokens", "lm_head"]

        ModulesToSaveWrapper itself has no `.weight`, but its active adapter
        module does. This helper returns the real nn.Embedding module.
        """

        if hasattr(module, "weight"):
            return module

        modules_to_save = getattr(module, "modules_to_save", None)

        if modules_to_save is not None:
            active_names = []

            if hasattr(module, "active_adapter"):
                active_adapter = getattr(module, "active_adapter")
                active_names.extend(_to_list(active_adapter))

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

            seen = set()
            active_names = [
                str(x) for x in active_names
                if x is not None and str(x) not in seen and not seen.add(str(x))
            ]

            for name in active_names:
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

        raise AttributeError(
            "Cannot find real embedding weight from input embeddings. "
            f"Got module type: {type(module).__name__}. "
            "If this is a PEFT ModulesToSaveWrapper, please make sure it contains "
            "an active embedding module in `modules_to_save` or `original_module`."
        )

    def _get_embedding_module(self):
        embedding_module = self.get_input_embeddings()
        return self._unwrap_modules_to_save_embedding(embedding_module)

    def _get_embedding_weight(self):
        return self._get_embedding_module().weight

    def _embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_module()(token_ids)

    def _append_attention(self, attention_mask: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(
            (attention_mask.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        return torch.cat([attention_mask, ones], dim=1)

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

    def _target_indices_in_codebook(
        self,
        codebook_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = target_ids.ne(IGNORE_INDEX)

        matches = codebook_ids.unsqueeze(0).eq(target_ids.unsqueeze(1))
        in_codebook = matches.any(dim=1)
        valid = valid & in_codebook

        target_pos = matches.float().argmax(dim=1).long()

        return target_pos, valid

    def _coarse_step(
        self,
        coarse_logits: torch.Tensor,
        target_coarse_ids: torch.Tensor,
        pos: int,
    ):
        """
        coarse_logits:
            [batch_size, vocab_size]

        target_coarse_ids:
            [batch_size]

        Returns:
            coarse_loss_sum
            coarse_count
            align_loss_sum
            align_count
            soft_coarse_embedding
        """

        device = coarse_logits.device

        codebook_ids = self._get_codebook_tensor(pos, device)
        codebook_logits = coarse_logits.index_select(dim=-1, index=codebook_ids)

        embedding_weight = self._get_embedding_weight()
        codebook_embeds = embedding_weight.index_select(0, codebook_ids)

        if self.temperature > 0:
            codebook_logits_for_prob = codebook_logits.float() / self.temperature
        else:
            codebook_logits_for_prob = codebook_logits.float()

        codebook_probs = F.softmax(codebook_logits_for_prob, dim=-1).to(
            dtype=codebook_embeds.dtype
        )

        soft_coarse_embeds = torch.matmul(codebook_probs, codebook_embeds)

        target_pos, valid = self._target_indices_in_codebook(
            codebook_ids=codebook_ids,
            target_ids=target_coarse_ids,
        )

        if valid.any():
            valid_logits = codebook_logits[valid].float()
            valid_target_pos = target_pos[valid]

            coarse_loss_vec = F.cross_entropy(
                valid_logits,
                valid_target_pos,
                reduction="none",
            )

            coarse_loss_sum = coarse_loss_vec.sum().to(dtype=coarse_logits.dtype)
            coarse_count = valid.sum().to(dtype=coarse_logits.dtype)

            safe_target_coarse_ids = target_coarse_ids.masked_fill(
                target_coarse_ids.eq(IGNORE_INDEX),
                self.pad_token_id,
            )

            gold_coarse_embeds = self._embed_token_ids(safe_target_coarse_ids)

            pred = soft_coarse_embeds[valid].float()
            gold = gold_coarse_embeds[valid].float()

            align_loss_vec = 1.0 - F.cosine_similarity(pred, gold, dim=-1)
            align_loss_sum = align_loss_vec.sum().to(dtype=coarse_logits.dtype)
            align_count = coarse_count
        else:
            coarse_loss_sum = coarse_logits.new_zeros(())
            coarse_count = coarse_logits.new_zeros(())
            align_loss_sum = coarse_logits.new_zeros(())
            align_count = coarse_logits.new_zeros(())

        return (
            coarse_loss_sum,
            coarse_count,
            align_loss_sum,
            align_count,
            soft_coarse_embeds,
        )

    def _fine_step(
        self,
        fine_logits: torch.Tensor,
        target_fine_ids: torch.Tensor,
    ):
        valid = target_fine_ids.ne(IGNORE_INDEX)

        if not valid.any():
            return fine_logits.new_zeros(()), fine_logits.new_zeros(())

        loss_vec = F.cross_entropy(
            fine_logits[valid].float(),
            target_fine_ids[valid],
            reduction="none",
        )

        return (
            loss_vec.sum().to(dtype=fine_logits.dtype),
            valid.sum().to(dtype=fine_logits.dtype),
        )

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ):
        code_len = fine_labels.size(1)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        cur_attention_mask = attention_mask

        coarse_loss_sum = next_logits.new_zeros(())
        coarse_count = next_logits.new_zeros(())
        fine_loss_sum = next_logits.new_zeros(())
        fine_count = next_logits.new_zeros(())
        align_loss_sum = next_logits.new_zeros(())
        align_count = next_logits.new_zeros(())

        fine_logits_for_output = []

        for pos in range(code_len):
            target_coarse = coarse_labels[:, pos]
            target_fine = fine_labels[:, pos]

            (
                c_loss_sum,
                c_count,
                a_loss_sum,
                a_count,
                soft_coarse_embeds,
            ) = self._coarse_step(
                coarse_logits=next_logits,
                target_coarse_ids=target_coarse,
                pos=pos,
            )

            coarse_loss_sum = coarse_loss_sum + c_loss_sum
            coarse_count = coarse_count + c_count
            align_loss_sum = align_loss_sum + a_loss_sum
            align_count = align_count + a_count

            cur_attention_mask = self._append_attention(cur_attention_mask)

            outputs = self.base_model(
                inputs_embeds=soft_coarse_embeds.unsqueeze(1),
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            fine_logits = outputs.logits[:, -1, :]
            fine_logits_for_output.append(fine_logits)

            f_loss_sum, f_count = self._fine_step(
                fine_logits=fine_logits,
                target_fine_ids=target_fine,
            )

            fine_loss_sum = fine_loss_sum + f_loss_sum
            fine_count = fine_count + f_count

            if pos + 1 < code_len:
                safe_fine_ids = target_fine.masked_fill(
                    target_fine.eq(IGNORE_INDEX),
                    self.pad_token_id,
                )

                fine_embeds = self._embed_token_ids(safe_fine_ids)

                cur_attention_mask = self._append_attention(cur_attention_mask)

                outputs = self.base_model(
                    inputs_embeds=fine_embeds.unsqueeze(1),
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]

        return (
            coarse_loss_sum,
            coarse_count,
            fine_loss_sum,
            fine_count,
            align_loss_sum,
            align_count,
            fine_logits_for_output,
        )

    def _forward_no_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        fine_labels: torch.Tensor,
        coarse_labels: torch.Tensor,
    ):
        code_len = fine_labels.size(1)

        cur_embeds = self._embed_token_ids(input_ids)
        cur_attention_mask = attention_mask

        coarse_loss_sum = cur_embeds.new_zeros(())
        coarse_count = cur_embeds.new_zeros(())
        fine_loss_sum = cur_embeds.new_zeros(())
        fine_count = cur_embeds.new_zeros(())
        align_loss_sum = cur_embeds.new_zeros(())
        align_count = cur_embeds.new_zeros(())

        fine_logits_for_output = []

        for pos in range(code_len):
            outputs = self.base_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attention_mask,
                use_cache=False,
                return_dict=True,
            )

            coarse_logits = outputs.logits[:, -1, :]
            target_coarse = coarse_labels[:, pos]
            target_fine = fine_labels[:, pos]

            (
                c_loss_sum,
                c_count,
                a_loss_sum,
                a_count,
                soft_coarse_embeds,
            ) = self._coarse_step(
                coarse_logits=coarse_logits,
                target_coarse_ids=target_coarse,
                pos=pos,
            )

            coarse_loss_sum = coarse_loss_sum + c_loss_sum
            coarse_count = coarse_count + c_count
            align_loss_sum = align_loss_sum + a_loss_sum
            align_count = align_count + a_count

            cur_embeds = torch.cat(
                [cur_embeds, soft_coarse_embeds.unsqueeze(1)],
                dim=1,
            )
            cur_attention_mask = self._append_attention(cur_attention_mask)

            outputs = self.base_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attention_mask,
                use_cache=False,
                return_dict=True,
            )

            fine_logits = outputs.logits[:, -1, :]
            fine_logits_for_output.append(fine_logits)

            f_loss_sum, f_count = self._fine_step(
                fine_logits=fine_logits,
                target_fine_ids=target_fine,
            )

            fine_loss_sum = fine_loss_sum + f_loss_sum
            fine_count = fine_count + f_count

            if pos + 1 < code_len:
                safe_fine_ids = target_fine.masked_fill(
                    target_fine.eq(IGNORE_INDEX),
                    self.pad_token_id,
                )

                fine_embeds = self._embed_token_ids(safe_fine_ids)

                cur_embeds = torch.cat(
                    [cur_embeds, fine_embeds.unsqueeze(1)],
                    dim=1,
                )
                cur_attention_mask = self._append_attention(cur_attention_mask)

        return (
            coarse_loss_sum,
            coarse_count,
            fine_loss_sum,
            fine_count,
            align_loss_sum,
            align_count,
            fine_logits_for_output,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        fine_labels=None,
        coarse_labels=None,
        labels=None,
        **kwargs,
    ):
        if fine_labels is None or coarse_labels is None:
            raise ValueError(
                "InterleavedLatentQwen requires `fine_labels` and `coarse_labels`. "
                "Do not pass explicit interleaved labels."
            )

        if input_ids is None:
            raise ValueError("input_ids must be provided.")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        if fine_labels.size() != coarse_labels.size():
            raise ValueError(
                f"fine_labels and coarse_labels must have same shape, got "
                f"{tuple(fine_labels.size())} vs {tuple(coarse_labels.size())}"
            )

        if self.use_train_cache:
            try:
                (
                    coarse_loss_sum,
                    coarse_count,
                    fine_loss_sum,
                    fine_count,
                    align_loss_sum,
                    align_count,
                    fine_logits_for_output,
                ) = self._forward_with_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    fine_labels=fine_labels,
                    coarse_labels=coarse_labels,
                )
            except (RuntimeError, TypeError, ValueError) as e:
                print(
                    "[InterleavedLatentQwen] cached latent forward failed; "
                    f"fallback to no-cache forward. Error: {repr(e)}"
                )

                (
                    coarse_loss_sum,
                    coarse_count,
                    fine_loss_sum,
                    fine_count,
                    align_loss_sum,
                    align_count,
                    fine_logits_for_output,
                ) = self._forward_no_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    fine_labels=fine_labels,
                    coarse_labels=coarse_labels,
                )
        else:
            (
                coarse_loss_sum,
                coarse_count,
                fine_loss_sum,
                fine_count,
                align_loss_sum,
                align_count,
                fine_logits_for_output,
            ) = self._forward_no_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                fine_labels=fine_labels,
                coarse_labels=coarse_labels,
            )

        loss = coarse_loss_sum.float() * 0.0

        loss = loss + self.coarse_loss_weight * (
            coarse_loss_sum.float() / coarse_count.float().clamp_min(1.0)
        )

        loss = loss + self.fine_loss_weight * (
            fine_loss_sum.float() / fine_count.float().clamp_min(1.0)
        )

        if self.coarse_align_weight > 0:
            loss = loss + self.coarse_align_weight * (
                align_loss_sum.float() / align_count.float().clamp_min(1.0)
            )

        if len(fine_logits_for_output) > 0:
            logits = torch.stack(fine_logits_for_output, dim=1)
        else:
            vocab_size = self._get_embedding_weight().size(0)
            logits = input_ids.new_zeros(
                (input_ids.size(0), 0, vocab_size),
                dtype=self._get_embedding_weight().dtype,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
