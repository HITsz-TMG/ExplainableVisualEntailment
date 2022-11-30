from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from oscar.transformers.modeling_bert import (BertEmbeddings,
                                              BertSelfAttention, BertAttention, BertEncoder, BertLayer,
                                              BertSelfOutput, BertIntermediate, BertOutput,
                                              BertPooler, BertPreTrainedModel)
from torch.nn.utils.rnn import pad_sequence

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        if do_chunk_cross == True:
            mixed_query_layer_new = mixed_query_layer.clone()
            # 用chunk的平均值作为查询
            for ba, offset in enumerate(offsets):
                sent_len = gather_index[ba].size(0)
                chunk = torch.zeros((len(offset), self.hidden_size)).cuda()
                chunk_hidden = mixed_query_layer[ba, 1:sent_len + 1]
                chunk = torch.index_add(chunk, 0, gather_index[ba], chunk_hidden)
                chunk_len = torch.tensor([len(item) for item in offset]).cuda()
                chunk_mean = chunk / chunk_len.unsqueeze(-1)
                mixed_query_layer_new[ba, 1:sent_len + 1] = torch.gather(chunk_mean, 0,
                                                                         gather_index[ba].unsqueeze(-1).repeat(1,
                                                                                                               self.hidden_size))
            mixed_query_layer = mixed_query_layer_new
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class CaptionBertAttention(BertAttention):
    """
    Modified from BertAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask=head_mask, history_state=history_state,
                                 do_chunk_cross=do_chunk_cross, offsets=offsets, gather_index=gather_index)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None, do_chunk_cross=False, offsets=None, gather_index=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask=head_mask, history_state=history_state,
                                           do_chunk_cross=do_chunk_cross, offsets=offsets, gather_index=gather_index)
        attention_output = attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.num_hidden_layers = config.num_hidden_layers
        self.add_residual = config.add_residual
        self.add_local_residual = config.add_local_residual
        self.chunk_attention_layers = [0, 1, 2]
        self.cross_chunk_attention_layers = [3, 4, 5, 6, 7, 8, 9, 10, 11]

    def forward(self, hidden_states, chunk_attention_mask, gather_index, img_mask, input_mask, hypo_len,
                img_len,
                head_mask=None, encoder_history_states=None,
                offsets=None):
        all_hidden_states = ()
        all_attentions = ()
        do_chunk_cross = False
        # 初始attention，只看到chunk内部和image
        attention_mask = input_mask.repeat(1, 1, hypo_len + img_len, 1)
        attention_mask[:, :, :hypo_len, :hypo_len] = chunk_attention_mask
        # image只看到image之间
        attention_mask[:, :, hypo_len:, :hypo_len] = -10000.0
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            if i in self.cross_chunk_attention_layers:
                attention_mask = input_mask
                do_chunk_cross = True

            layer_outputs = layer_module(
                hidden_states, attention_mask,
                head_mask=head_mask[i], history_state=history_state, do_chunk_cross=do_chunk_cross, offsets=offsets,
                gather_index=gather_index)

            hidden_states = layer_outputs[0]
            if i == 5:
                # 预训练时最后六层在对齐
                chunk_hidden_states = hidden_states

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, chunk_hidden_states  # outputs, (hidden states), (attentions)


class SeqBertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """

    def __init__(self, config):
        super(SeqBertImgModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.init_weights()
        self.max_hypo = config.max_hypo
        self.edge_dense = nn.Embedding(1, config.hidden_size)

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_mask=None,
                position_ids=None, head_mask=None, img_feats=None, img_mask=None,
                encoder_history_states=None, offsets=None, gather_index=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if img_mask.dim() == 2:
            extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_img_mask = img_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        if input_mask.dim() == 2:
            extended_input_mask = input_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_input_mask = input_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_input_mask = (1.0 - extended_input_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        img_embedding_output = self.img_embedding(img_feats)
        if self.use_img_layernorm:
            img_embedding_output = self.LayerNorm(img_embedding_output)
        # add dropout on image embedding
        img_embedding_output = self.dropout(img_embedding_output)

        embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
        hypo_len = input_ids.size(1)
        img_len = img_feats.size(1)
        encoder_outputs, chunk_hidden_states = self.encoder(embedding_output, extended_attention_mask,
                                                            gather_index=gather_index,
                                                            hypo_len=hypo_len, img_len=img_len,
                                                            img_mask=extended_img_mask,
                                                            input_mask=extended_input_mask,
                                                            head_mask=head_mask, offsets=offsets,
                                                            encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs, chunk_hidden_states


class ClsLayer(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(ClsLayer, self).__init__(config)
        self.cls_q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.align_k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, self_chunk_align, cls, word_mask):
        cls_q = self.cls_q_proj(cls.unsqueeze(1))
        self_chunk_align_k = self.align_k_proj(self_chunk_align)
        self_chunk_align_v = self_chunk_align_k.clone()
        self_chunk_align_k = self_chunk_align_k.permute(0, 2, 1)

        attn_weight = torch.matmul(cls_q, self_chunk_align_k)
        attn_weight = attn_weight + word_mask
        attn_weight = nn.Softmax(dim=-1)(attn_weight)
        attn_weight = self.dropout(attn_weight)
        cls_attn_output = torch.matmul(attn_weight, self_chunk_align_v)
        cls_with_align = cls + cls_attn_output.squeeze(1)
        return cls_with_align


class ChunkAlign_CLS_enc3(nn.Module):
    def __init__(self, global_enc, seq_enc):
        super(ChunkAlign_CLS_enc3, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc

        self.attention_thresh = 0.6
        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, expl_ids=None, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids=None, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return matched_0, pre


class ClsLayer2(BertLayer):
    """
    Modified from BertLayer to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(ClsLayer2, self).__init__(config)
        self.cls_q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.align_k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, self_chunk_align, cls, word_mask):
        cls_q = self.cls_q_proj(cls.unsqueeze(1))
        self_chunk_align_k = self.align_k_proj(self_chunk_align)
        self_chunk_align_v = self_chunk_align_k.clone()
        self_chunk_align_k = self_chunk_align_k.permute(0, 2, 1)

        attn_weight = torch.matmul(cls_q, self_chunk_align_k)
        attn_weight = attn_weight + word_mask
        attn_weight = nn.Softmax(dim=-1)(attn_weight)
        attn_weight = self.dropout(attn_weight)
        cls_attn_output = torch.matmul(attn_weight, self_chunk_align_v).squeeze(1)
        cls_attn_output = self.dense(cls_attn_output)
        cls_attn_output = self.dropout(cls_attn_output)
        cls_with_align = self.LayerNorm(cls_attn_output + cls)
        intermediate_output = self.intermediate(cls_with_align)
        layer_output = self.output(intermediate_output, cls_with_align)
        return layer_output


class ChunkAlign_CLS_enc4(nn.Module):
    def __init__(self, global_enc, seq_enc):
        super(ChunkAlign_CLS_enc4, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc

        self.attention_thresh = 0.6
        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, img_feat, expl_ids=None, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        img_len = img_feat.size(1)
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids=None, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)
        logits = self.classifier(CLS_ensem)
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        return matched_0, pre


    def cal_attention(self, input_ids, img_feat, expl_ids=None, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        global_attn_weight = torch.stack(outputs[2][-6:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        global_attn_weight = global_attn_weight.masked_fill(global_attn_weight == 0, -1e5)
        global_attn_weight = nn.Softmax(dim=-1)(global_attn_weight)

        seq_attn_weight = torch.stack(seq_outputs[2][-6:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        seq_attn_weight = seq_attn_weight.masked_fill(seq_attn_weight == 0, -1e5)
        seq_attn_weight = nn.Softmax(dim=-1)(seq_attn_weight)


        return matched_0, pre


class ChunkAlign_dec4(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec4, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = self_chunk_align.detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = self_chunk_align.detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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


class ChunkAlign_dec4_4(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec4_4, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat((seq_outputs[0], global_output, chunk_hidden_states), 1).detach()
        encoder_mask = torch.cat((input_mask, input_mask, input_mask), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat((seq_outputs[0], global_output, chunk_hidden_states), 1)
        encoder_mask = torch.cat((input_mask, input_mask, input_mask), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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


class ChunkAlign_dec4_5(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec4_5, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states[:, 1:hypo_len]),
            1).detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states[:, 1:hypo_len]), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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


class ChunkAlign_dec5_4(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec5_4, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states),
            1).detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len], chunk_hidden_states), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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


class ChunkAlign_dec5_3(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec5_3, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states_hypo = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states_hypo), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat((seq_outputs[0], global_output, chunk_hidden_states), 1).detach()
        encoder_mask = torch.cat((input_mask, input_mask, input_mask), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states_hypo = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states_hypo), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0], global_output, chunk_hidden_states), 1)
        encoder_mask = torch.cat((input_mask, input_mask, input_mask), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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


class ChunkAlign_dec5_2(nn.Module):
    def __init__(self, global_enc, dec, seq_enc, dec_toker, beam_size, max_hypo):
        super(ChunkAlign_dec5_2, self).__init__()
        self.global_enc = global_enc
        self.seq_enc = seq_enc
        self.dec_toker = dec_toker
        self.dec = dec

        self.gate_weight = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.cls_ensemble = nn.Linear(self.global_enc.config.hidden_size * 2, self.global_enc.config.hidden_size)
        self.cls_layer_num = 3
        self.cls_layer = nn.ModuleList([ClsLayer2(self.global_enc.config) for _ in range(self.cls_layer_num)])
        self.num_labels = 3
        self.classifier = nn.Linear(self.global_enc.config.hidden_size, self.num_labels)
        self.cls_loss_fct = CrossEntropyLoss()

        self.vocab_num = self.dec.vocab_size
        self.lm_head = nn.Linear(self.dec.config.n_embd, self.dec.vocab_size, bias=False)
        self.gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=self.dec_toker.pad_token_id)
        self.max_len = 50
        self.beam_size = beam_size
        self.max_gen_len = 100
        self.repeat_penalty = 5.3
        self.b_rtnl = torch.tensor(self.dec_toker.encode("<|b_rtnl|>")).cuda()

    def forward(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states_hypo = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states_hypo), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)
        loss_cls_0 = self.cls_loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat(
            (seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len]), 1).detach()
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)

        gpt_out = self.dec(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs,
                           encoder_attention_mask=encoder_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return gen_loss, loss_cls_0, matched_0

    def evaluate(self, input_ids, img_feat, expl_ids, input_mask=None, label=None, attn_mask=None,
                 token_type_ids=None, position_ids=None, head_mask=None, encoder_history_states=None,
                 offsets=None, chunk_attention_mask=None, syn_labels_ids=None, gather_index=None):
        hypo_len = input_ids.size(1)

        outputs = self.global_enc(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        global_output = outputs[0]
        global_CLS = outputs[1]

        img_mask = input_mask[:, hypo_len:]
        seq_outputs, chunk_hidden_states = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask,
                                                        input_mask=input_mask,
                                                        attention_mask=chunk_attention_mask,
                                                        position_ids=position_ids, token_type_ids=token_type_ids,
                                                        head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        chunk_CLS = seq_outputs[1]
        CLS_ensem = self.cls_ensemble(torch.cat((global_CLS, chunk_CLS), -1))

        chunk_align = seq_outputs[0][:, 1:hypo_len]
        global_hypo = global_output[:, 1:hypo_len]
        chunk_hidden_states_hypo = chunk_hidden_states[:, 1:hypo_len]
        self_chunk_align = torch.cat((global_hypo, chunk_align, chunk_hidden_states_hypo), dim=1)

        word_mask = input_mask[:, 1:hypo_len].unsqueeze(1)
        word_mask = (1.0 - word_mask) * -10000.0
        word_mask = torch.cat((word_mask, word_mask, word_mask), -1)  # batch,word*2,hidden
        for i, layer_module in enumerate(self.cls_layer):
            CLS_ensem = layer_module(self_chunk_align, CLS_ensem, word_mask)

        logits = self.classifier(CLS_ensem)

        pre = logits.max(dim=-1)[1]
        matched_0 = pre == label

        encoder_hs = torch.cat((seq_outputs[0][:, 1:hypo_len], global_output[:, 1:hypo_len]), 1)
        encoder_mask = torch.cat((input_mask[:, 1:hypo_len], input_mask[:, 1:hypo_len]), 1)
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
                                       encoder_hidden_states=encoder_hs[i].unsqueeze(0),
                                       encoder_attention_mask=encoder_mask[i].unsqueeze(0), use_cache=True,
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
