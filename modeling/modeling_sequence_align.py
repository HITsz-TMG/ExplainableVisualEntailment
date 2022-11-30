from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn.utils.rnn as rnn_utils

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)

import numpy as np


class BaseLine(nn.Module):
    def __init__(self, oscar, gpt, gpt_toker, beam_size, max_hypo):
        super(BaseLine, self).__init__()
        self.global_enc = oscar
        self.gpt_toker = gpt_toker
        self.dec = gpt

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.gpt_toker.pad_token_id)
        # 因为乘了个矩阵
        self.dropout = nn.Dropout(self.global_enc.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.max_hypo = max_hypo
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                syn_labels_ids=None):

        # global_IMG=img_feat[:,:1]
        # global_mask=input_mask[:,:self.max_hypo+1]
        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        sequence_output = outputs[0]
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        loss_fct = CrossEntropyLoss()
        loss_cls_0 = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=sequence_output,
                           encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 syn_labels_ids=None):
        # global_IMG = img_feat[:, :1]
        # global_mask = input_mask[:, :self.max_hypo + 1]
        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        sequence_output = outputs[0]
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        outputs = torch.full((expl_ids.size(0), self.max_gen_len), fill_value=self.gpt_toker.pad_token_id,
                             dtype=int).cuda()

        past_key_values = None
        cur_unfinished = outputs.new(outputs.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = expl_ids[:, 0]
        for index in range(self.max_gen_len - 1):
            gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=sequence_output,
                               encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            outputs[:, index] = tokens_to_add
            cur_len += 1
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
        if cur_len == self.max_gen_len:
            outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        return outputs, matched_0, pre


class BaseLine_cls(nn.Module):
    def __init__(self, oscar, beam_size, max_hypo):
        super(BaseLine_cls, self).__init__()
        self.oscar = oscar
        # 因为乘了个矩阵
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_hypo = max_hypo
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3

    def forward(self, input_ids, img_feat, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                syn_labels_ids=None):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        loss_fct = CrossEntropyLoss()
        loss_cls_0 = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 syn_labels_ids=None):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)

        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return matched_0, pre

    def save_heat(self, input_ids, img_feat, input_mask=None, label=None, attn_mask=None,
                  token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None):
        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)

        oscar_CLS = outputs[1]
        hypo_len=input_ids.size(-1)
        logits = self.classifier(oscar_CLS)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return matched_0, pre, torch.stack(outputs[2])[:, :, :, :hypo_len, hypo_len:].squeeze(1)


class BaseLine_freeze(nn.Module):
    def __init__(self, oscar, gpt, gpt_toker, beam_size, max_hypo):
        super(BaseLine_freeze, self).__init__()
        self.oscar = oscar
        self.dec_toker = gpt_toker
        self.dec = gpt

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        # 因为乘了个矩阵
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_hypo = max_hypo
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                syn_labels_ids=None):
        hypo_len = input_ids.size(1)

        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        sequence_output = outputs[0]
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        loss_fct = CrossEntropyLoss()
        loss_cls_0 = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label
        encoder_hs = sequence_output[:, 1:hypo_len, :].contiguous().detach()
        encoder_mask = input_mask[:, 1:hypo_len].contiguous()
        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 syn_labels_ids=None):
        hypo_len = input_ids.size(1)

        outputs = self.oscar(input_ids, img_feats=img_feat, attention_mask=input_mask,
                             position_ids=position_ids, token_type_ids=token_type_ids,
                             head_mask=head_mask,
                             encoder_history_states=encoder_history_states)
        sequence_output = outputs[0]
        oscar_CLS = outputs[1]

        logits = self.classifier(oscar_CLS)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label
        encoder_hs = sequence_output[:, 1:hypo_len].contiguous()
        encoder_mask = input_mask[:, 1:hypo_len].contiguous()
        outputs = torch.full((expl_ids.size(0), self.max_gen_len), fill_value=self.dec_toker.pad_token_id,
                             dtype=int).cuda()

        for i in range(matched_0.size(0)):
            if matched_0[i] == torch.tensor(True):
                decoder_input = expl_ids[i]
                b_rtnl_index = torch.nonzero((decoder_input == self.b_rtnl.unsqueeze(0)).to(torch.int64))[0, -1]
                prompt_decoder_input = decoder_input[:b_rtnl_index]
                gpt_out = self.dec(input_ids=prompt_decoder_input.unsqueeze(0),
                                   encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                   encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
                                   past_key_values=None)
                past_key_values = gpt_out[1]
                cur_unfinished = outputs.new(1).fill_(1)
                cur_len = 0
                tokens_to_add = decoder_input[b_rtnl_index].unsqueeze(-1)
                for index in range(self.max_gen_len - 1):
                    gpt_out = self.dec(input_ids=tokens_to_add.unsqueeze(0),
                                       encoder_hidden_states=sequence_output[i].unsqueeze(0),
                                       encoder_attention_mask=input_mask[i].unsqueeze(0), use_cache=True,
                                       past_key_values=past_key_values)
                    past_key_values = gpt_out[1]
                    gpt_out = gpt_out[0][:, -1:, :]
                    # 只取最后一个作为当前步的输出
                    lm_logits = self.lm_head(gpt_out)
                    gen_label = torch.argmax(lm_logits, dim=-1).squeeze(-1)
                    tokens_to_add = gen_label * cur_unfinished + self.dec_toker.pad_token_id * (1 - cur_unfinished)
                    outputs[i, index] = tokens_to_add
                    cur_len += 1
                    cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.dec_toker.eos_token_id).long())
                    if cur_unfinished.max() == 0:
                        break
                if cur_len == self.max_gen_len:
                    outputs[i, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.dec_toker.eos_token_id)
        return outputs, matched_0, pre
