# import transformers
# import copy
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# import torch.nn.functional as F
# from typing import List, Optional, Tuple, Union

# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
# from transformers.utils import add_start_docstrings_to_model_forward, add_start_docstrings,replace_return_docstrings
# # , logging, replace_return_docstrings, add_start_docstrings



# _CONFIG_FOR_DOC = "LlamaConfig"

# # @add_start_docstrings(
# #     "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
# #     LLAMA_START_DOCSTRING,
# # )
# class LETTER(LlamaForCausalLM):
#     _tied_weights_keys = ["lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)

#         self.temperature = 1.0
#     def set_hyper(self, temperature):
#         self.temperature = temperature
    
#     def ranking_loss(self, shift_logits, shift_labels):
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits /self.temperature, shift_labels)
#         return loss

#     def total_loss(self, shift_logits, shift_labels):
#         gen_loss = self.ranking_loss(shift_logits, shift_labels)
#         loss = gen_loss

#         return loss

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
#         >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         hidden_states = outputs[0]

#         if self.config.pretraining_tp > 1:
#             lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
#             logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
#             logits = torch.cat(logits, dim=-1)
#         else:
#             logits = self.lm_head(hidden_states)
#         logits = logits.float()
#         # print(logits)
#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()

#             loss = self.total_loss(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class LETTER(nn.Module):
    """
    通用 CausalLM 包装器：
    - 内部加载 AutoModelForCausalLM（可用 Qwen3 / LLaMA / Mistral 等）
    - forward 输出保持 transformers 的 CausalLMOutputWithPast
    - loss 使用你原来的 temperature CE（ranking_loss）
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.temperature = 1.0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        让你的 train.py 里还能继续写：
            model = LETTER.from_pretrained(...)
        """
        base = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            trust_remote_code=True,  # Qwen3 常需要
        )
        return cls(base)

    def save_pretrained(self, *args, **kwargs):
        # Trainer.save_model 需要这个
        return self.model.save_pretrained(*args, **kwargs)

    def resize_token_embeddings(self, new_num_tokens: int):
        return self.model.resize_token_embeddings(new_num_tokens)

    def set_hyper(self, temperature: float):
        self.temperature = float(temperature)

    def ranking_loss(self, shift_logits, shift_labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fct(shift_logits / self.temperature, shift_labels)
        return loss

    def total_loss(self, shift_logits, shift_labels):
        return self.ranking_loss(shift_logits, shift_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 强制拿到 logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.total_loss(shift_logits, shift_labels)

        if not return_dict:
            # 保持兼容 tuple 输出
            out = (logits,)
            if outputs.past_key_values is not None:
                out += (outputs.past_key_values,)
            if output_hidden_states:
                out += (outputs.hidden_states,)
            if output_attentions:
                out += (outputs.attentions,)
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def __getattr__(self, name):
        """
        关键：把没找到的属性委托给内部 base model，
        例如：model.config / model.generate / model.print_trainable_parameters 等。
        """
        if name in ["model", "config", "temperature"]:
            return super().__getattr__(name)
        return getattr(self.model, name)
