from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class LETTER(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)

        self.temperature = 1.0
        self.coarse_loss_weight = 1.0
        self.fine_loss_weight = 1.0
        self.coarse_align_weight = 2.0
        self.curriculum_warmup_steps = 10000

        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long), persistent=False)

        d_model = config.d_model
        vocab_size = config.vocab_size

        # coarse / fine heads
        self.coarse_head = nn.Linear(d_model, vocab_size, bias=False)
        self.fine_head = nn.Linear(d_model, vocab_size, bias=False)

        # coarse latent 注入：gate + residual
        self.coarse_proj = nn.Linear(d_model, d_model, bias=False)
        self.coarse_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.coarse_inject_ln = nn.LayerNorm(d_model)

        self._init_new_heads_from_lm_head()
        self._init_injection_layers()

    def _init_new_heads_from_lm_head(self):
        with torch.no_grad():
            self.coarse_head.weight.copy_(self.lm_head.weight)
            self.fine_head.weight.copy_(self.lm_head.weight)

    def _init_injection_layers(self):
        nn.init.xavier_uniform_(self.coarse_proj.weight)
        nn.init.xavier_uniform_(self.coarse_gate.weight)
        nn.init.zeros_(self.coarse_gate.bias)

    def resize_token_embeddings_and_heads(self, new_num_tokens):
        old_coarse_weight = self.coarse_head.weight.data.clone()
        old_fine_weight = self.fine_head.weight.data.clone()

        super().resize_token_embeddings(new_num_tokens)

        d_model = self.config.d_model
        old_vocab = old_coarse_weight.size(0)
        device = self.shared.weight.device
        dtype = self.shared.weight.dtype

        self.coarse_head = nn.Linear(d_model, new_num_tokens, bias=False).to(device=device, dtype=dtype)
        self.fine_head = nn.Linear(d_model, new_num_tokens, bias=False).to(device=device, dtype=dtype)

        with torch.no_grad():
            self.coarse_head.weight[:old_vocab].copy_(old_coarse_weight.to(device=device, dtype=dtype))
            self.fine_head.weight[:old_vocab].copy_(old_fine_weight.to(device=device, dtype=dtype))

            if new_num_tokens > old_vocab:
                self.coarse_head.weight[old_vocab:].copy_(self.lm_head.weight[old_vocab:].to(device=device, dtype=dtype))
                self.fine_head.weight[old_vocab:].copy_(self.lm_head.weight[old_vocab:].to(device=device, dtype=dtype))

        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def set_hyper(
        self,
        temperature=1.0,
        coarse_loss_weight=1.0,
        fine_loss_weight=1.0,
        coarse_align_weight=2.0,
        curriculum_warmup_steps=10000,
    ):
        self.temperature = temperature
        self.coarse_loss_weight = coarse_loss_weight
        self.fine_loss_weight = fine_loss_weight
        self.coarse_align_weight = coarse_align_weight
        self.curriculum_warmup_steps = curriculum_warmup_steps

    def _masked_ce_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(logits.device)
        return loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    def _masked_mse_loss(self, pred, target, mask):
        """
        pred/target: [B, D]
        mask: [B] bool
        """
        if mask is None or mask.sum().item() == 0:
            return pred.new_tensor(0.0)
        per_token = ((pred - target) ** 2).mean(dim=-1)  # [B]
        return per_token.masked_select(mask).mean()

    def _maybe_scale_hidden(self, hidden):
        if self.config.tie_word_embeddings:
            hidden = hidden * (self.model_dim ** -0.5)
        return hidden

    def _soft_coarse_embedding(self, coarse_logits):
        probs = F.softmax(coarse_logits / self.temperature, dim=-1)   # [B, V]
        coarse_emb = torch.matmul(probs, self.shared.weight)          # [B, D]
        return coarse_emb

    def _gold_coarse_embedding(self, coarse_ids):
        return self.shared(coarse_ids)

    def _curriculum_pred_ratio(self):
        """
        课程学习：从 gold coarse -> pred coarse
        step=0 时 pred_ratio=0，warmup 结束后 pred_ratio=1
        """
        if (not self.training) or self.curriculum_warmup_steps <= 0:
            return 1.0
        step = int(self._train_step.item())
        return min(1.0, float(step) / float(self.curriculum_warmup_steps))

    def _blend_coarse_embedding(self, pred_emb, gold_emb, active_mask=None):
        """
        训练时：gold -> pred 逐步过渡
        """
        pred_ratio = self._curriculum_pred_ratio()
        mixed = gold_emb * (1.0 - pred_ratio) + pred_emb * pred_ratio

        if active_mask is not None:
            pad_id = self.config.pad_token_id
            if pad_id is None:
                pad_id = 0
            pad_emb = self.shared.weight[pad_id].view(1, -1).to(device=mixed.device, dtype=mixed.dtype)
            mixed = torch.where(active_mask.unsqueeze(-1), mixed, pad_emb)

        return mixed

    def _inject_coarse(self, hidden_after_coarse, coarse_emb):
        """
        gate + residual 注入 coarse latent
        """
        gate = torch.sigmoid(self.coarse_gate(torch.cat([hidden_after_coarse, coarse_emb], dim=-1)))
        coarse_delta = self.coarse_proj(coarse_emb)
        fine_hidden = hidden_after_coarse + gate * coarse_delta
        fine_hidden = self.coarse_inject_ln(fine_hidden)
        return fine_hidden

    def _decoder_prefill(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        decoder_input_ids,
    ):
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        last_hidden = self._maybe_scale_hidden(last_hidden)
        return last_hidden, decoder_outputs.past_key_values

    def _decoder_step(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_values,
        input_ids=None,
        inputs_embeds=None,
    ):
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        last_hidden = decoder_outputs.last_hidden_state[:, -1, :]
        last_hidden = self._maybe_scale_hidden(last_hidden)
        return last_hidden, decoder_outputs.past_key_values

    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,               # 兼容 Trainer；这里把 labels 当 fine_labels
        coarse_labels=None,
        fine_labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,
        return_hidden_state=False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Trainer 默认会传 labels，这里直接当 fine_labels
        if fine_labels is None and labels is not None:
            fine_labels = labels

        if fine_labels is None:
            raise ValueError("fine_labels (or labels) must be provided.")
        if coarse_labels is None:
            raise ValueError("coarse_labels must be provided.")

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # encoder
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs.last_hidden_state

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)

        batch_size = hidden_states.size(0)
        device = hidden_states.device

        tgt_len = fine_labels.size(1)

        decoder_start_token_id = self.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = 0

        start_ids = torch.full(
            (batch_size, 1),
            fill_value=decoder_start_token_id,
            dtype=torch.long,
            device=device
        )

        # prefill
        last_hidden, past_key_values = self._decoder_prefill(
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=start_ids,
        )

        all_coarse_logits = []
        all_fine_logits = []
        align_losses = []

        for t in range(tgt_len):
            active_mask = fine_labels[:, t].ne(-100)

            # 1) coarse prediction
            coarse_logits_t = self.coarse_head(last_hidden)                 # [B, V]
            pred_coarse_emb_t = self._soft_coarse_embedding(coarse_logits_t)  # [B, D]

            # 2) gold coarse embedding
            gold_coarse_ids_t = coarse_labels[:, t].clone()
            gold_coarse_ids_t[gold_coarse_ids_t == -100] = self.config.pad_token_id
            gold_coarse_emb_t = self._gold_coarse_embedding(gold_coarse_ids_t)  # [B, D]

            # 3) curriculum mix: easy -> hard
            if self.training:
                coarse_emb_t = self._blend_coarse_embedding(
                    pred_emb=pred_coarse_emb_t,
                    gold_emb=gold_coarse_emb_t,
                    active_mask=active_mask,
                )
            else:
                pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
                pad_emb = self.shared.weight[pad_id].view(1, -1).to(device=device, dtype=pred_coarse_emb_t.dtype)
                coarse_emb_t = torch.where(active_mask.unsqueeze(-1), pred_coarse_emb_t, pad_emb)

            # 4) coarse latent -> decoder step
            hidden_after_coarse, past_key_values = self._decoder_step(
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                input_ids=None,
                inputs_embeds=coarse_emb_t.unsqueeze(1),
            )

            # 5) gate inject coarse
            fine_hidden_t = self._inject_coarse(hidden_after_coarse, coarse_emb_t)
            fine_logits_t = self.fine_head(fine_hidden_t)  # [B, V]

            all_coarse_logits.append(coarse_logits_t)
            all_fine_logits.append(fine_logits_t)

            # 6) coarse embedding MSE 对齐
            align_t = self._masked_mse_loss(pred_coarse_emb_t, gold_coarse_emb_t, active_mask)
            align_losses.append(align_t)

            # 7) teacher forcing 下一步的 fine token
            gold_fine_t = fine_labels[:, t].clone()
            gold_fine_t[gold_fine_t == -100] = self.config.pad_token_id

            if t < tgt_len - 1:
                last_hidden, past_key_values = self._decoder_step(
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    input_ids=gold_fine_t.unsqueeze(1),
                    inputs_embeds=None,
                )

        coarse_logits = torch.stack(all_coarse_logits, dim=1)  # [B, T, V]
        fine_logits = torch.stack(all_fine_logits, dim=1)      # [B, T, V]

        coarse_loss = self._masked_ce_loss(coarse_logits, coarse_labels)
        fine_loss = self._masked_ce_loss(fine_logits, fine_labels)
        coarse_align_loss = torch.stack(align_losses).mean() if len(align_losses) > 0 else fine_logits.new_tensor(0.0)

        loss = (
            self.coarse_loss_weight * coarse_loss
            + self.fine_loss_weight * fine_loss
            + self.coarse_align_weight * coarse_align_loss
        )

        if self.training:
            with torch.no_grad():
                self._train_step += 1

        if not return_dict:
            return (loss, fine_logits)

        output = Seq2SeqLMOutput(
            loss=loss,
            logits=fine_logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        output.coarse_loss = coarse_loss
        output.fine_loss = fine_loss
        output.coarse_align_loss = coarse_align_loss
        output.coarse_logits = coarse_logits
        output.fine_logits = fine_logits
        return output
