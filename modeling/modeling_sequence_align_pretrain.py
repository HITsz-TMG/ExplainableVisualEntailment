from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from oscar.transformers.modeling_bert import (BertEmbeddings, BertOnlyMLMHead,
                                              BertSelfAttention, BertAttention, BertEncoder, BertLayer,
                                              BertSelfOutput, BertIntermediate, BertOutput,
                                              BertPooler, BertPreTrainedModel)
from torch.nn.utils.rnn import pad_sequence

BertLayerNorm = torch.nn.LayerNorm
logger = logging.getLogger(__name__)


class BertCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                                 k=int(loss.shape[0] * (1 - self.drop_worst_ratio)),
                                 largest=False)

        loss = loss.mean()

        return loss


class CaptionBertSelfAttention(BertSelfAttention):
    """
    Modified from BertSelfAttention to add support for output_hidden_states.
    """

    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, attention_mask, head_mask=None, gather_index=None,
                history_state=None, do_chunk_cross=False, offsets=None):
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
                chunk_hidden = mixed_query_layer[ba, 2:sent_len + 2]
                chunk = torch.index_add(chunk, 0, gather_index[ba], chunk_hidden)
                chunk_len = torch.tensor([len(item) for item in offset]).cuda()
                chunk_mean = chunk / chunk_len.unsqueeze(-1)
                mixed_query_layer_new[ba, 2:sent_len + 2] = torch.gather(chunk_mean, 0,
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

    def forward(self, input_tensor, attention_mask, head_mask=None, gather_index=None,
                history_state=None, do_chunk_cross=False, offsets=None):
        self_outputs = self.self(input_tensor, attention_mask, gather_index=gather_index, head_mask=head_mask,
                                 history_state=history_state,
                                 do_chunk_cross=do_chunk_cross, offsets=offsets)
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

    def forward(self, hidden_states, attention_mask, head_mask=None, gather_index=None,
                history_state=None, do_chunk_cross=False, offsets=None):
        attention_outputs = self.attention(hidden_states, attention_mask, gather_index=gather_index,
                                           head_mask=head_mask, history_state=history_state,
                                           do_chunk_cross=do_chunk_cross, offsets=offsets)
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

        self.chunk_attention_layers = [0, 1, 2]
        self.cross_chunk_attention_layers = [3, 4, 5, 6, 7, 8]
        self.cross_modal_layers = [9, 10, 11]

    def forward(self, hidden_states, chunk_attention_mask, img_mask, input_mask, hypo_len,
                img_len, gather_index,
                head_mask=None, encoder_history_states=None,
                offsets=None):
        all_hidden_states = ()
        all_attentions = ()
        do_chunk_cross = False
        # 初始attention，只看到chunk内部和image
        attention_mask = input_mask.clone()
        attention_mask[:, :, :hypo_len, :hypo_len] = chunk_attention_mask

        # image只看到image之间
        attention_mask[:, :, hypo_len:, :hypo_len] = -10000.0
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]

            if i == self.cross_chunk_attention_layers[0]:
                attention_mask = input_mask
                # do_chunk_cross = True
            elif i in self.cross_modal_layers:
                # cross modal时取mean chunk
                do_chunk_cross = True
                if i == self.cross_modal_layers[0]:
                    # 图片部分只看到自己
                    img_mask = torch.eye(img_len)
                    img_mask = img_mask.unsqueeze(0).repeat(attention_mask.size(0), 1, 1)
                    img_mask = torch.cat((torch.zeros(attention_mask.size(0), img_len, hypo_len), img_mask), -1)
                    img_mask = (1.0 - img_mask) * -10000.0
                    attention_mask[:, :, hypo_len:, :] = img_mask.unsqueeze(1)
                    # 文本部分仅可见chunk 内部 和image
                    attention_mask[:, :, :hypo_len, :hypo_len] = chunk_attention_mask

            layer_outputs = layer_module(
                hidden_states, attention_mask, gather_index=gather_index,
                head_mask=head_mask[i], history_state=history_state, do_chunk_cross=do_chunk_cross, offsets=offsets)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)


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
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, gather_index=gather_index,
                                       hypo_len=hypo_len, img_len=img_len, img_mask=extended_img_mask,
                                       input_mask=extended_input_mask,
                                       head_mask=head_mask, offsets=offsets,
                                       encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs


class SeqAlign_pretrain_v2(nn.Module):
    def __init__(self, seq_enc, config):
        super(SeqAlign_pretrain_v2, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.seq_enc = seq_enc
        self.cls_loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.cls = BertOnlyMLMHead(config)
        self.loss = BertCaptioningLoss(config)

    def forward(self, input_ids, img_feat, total_label, img_mask, input_mask,
                token_type_ids=None, position_ids=None, head_mask=None,
                offsets=None, chunk_attention_mask=None, masked_token=None, masked_pos=None, gather_index=None,
                align_pos=None, example_id=None):

        hypo_len = input_ids.size(2)
        batch_size = input_ids.size(0)
        input_ids_flatten = input_ids.reshape(batch_size * 2, -1)
        img_feat_flatten = img_feat.unsqueeze(1).repeat(1, 2, 1, 1).reshape(batch_size * 2, -1, img_feat.size(-1))
        img_mask_flatten = img_mask.unsqueeze(1).repeat(1, 2, 1).reshape(batch_size * 2, -1)
        input_mask_flatten = input_mask.unsqueeze(1).repeat(1, 2, 1, 1).reshape(batch_size * 2, input_mask.size(-1),
                                                                                input_mask.size(-1))
        chunk_attention_mask_flatten = chunk_attention_mask.unsqueeze(1).repeat(1, 2, 1, 1).reshape(batch_size * 2,
                                                                                                    chunk_attention_mask.size(
                                                                                                        -1),
                                                                                                    chunk_attention_mask.size(
                                                                                                        -1))
        token_type_ids_flatten = token_type_ids.unsqueeze(1).repeat(1, 2, 1).reshape(batch_size * 2, -1)
        offsets_flatten = []
        for item in offsets:
            offsets_flatten.append(item)
            offsets_flatten.append(item)
        gather_index_flatten = []
        for item in gather_index:
            gather_index_flatten.append(item)
            gather_index_flatten.append(item)
        seq_outputs = self.seq_enc(input_ids_flatten, img_feats=img_feat_flatten, img_mask=img_mask_flatten,
                                   input_mask=input_mask_flatten,
                                   attention_mask=chunk_attention_mask_flatten,
                                   position_ids=position_ids, token_type_ids=token_type_ids_flatten,
                                   head_mask=head_mask, offsets=offsets_flatten, gather_index=gather_index_flatten)
        attn_weight = torch.stack(seq_outputs[2][-3:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        attn_weight = attn_weight.masked_fill(attn_weight == 0, -1e5)
        attn_weight = nn.Softmax(dim=-1)(attn_weight)

        align_pos_flatten = align_pos.unsqueeze(1).repeat(1, 2, 1).reshape(batch_size * 2, -1)
        total_label_flatten = total_label.unsqueeze(1).repeat(1, 2, 1).reshape(batch_size * 2, -1)
        total_label_align = total_label_flatten[align_pos_flatten == 1].to(dtype=torch.int64)
        attn_weight_align = attn_weight[align_pos_flatten == 1, :]
        align_loss = self.cls_loss_fct(attn_weight_align, total_label_align)

        sequence_output = seq_outputs[0]
        sequence_output = self.dropout(sequence_output)
        text_cls_hidden = sequence_output[:, 1, :].reshape(batch_size, 2, sequence_output.size(-1))
        origin_input = text_cls_hidden[:, 0, :]
        mask_input = text_cls_hidden[:, 1, :]
        # origin_input = F.relu(origin_input)
        origin_input = F.normalize(origin_input, p=2, dim=-1)
        # mask_input = F.relu(mask_input)
        mask_input = F.normalize(mask_input, p=2, dim=-1)

        similarities = torch.matmul(origin_input, mask_input.permute(1, 0))
        similarities = similarities / 0.05
        object_label = torch.arange(0, similarities.size(0), dtype=torch.long).cuda()
        loss_fct = CrossEntropyLoss()
        object_contras_loss = loss_fct(similarities, object_label)
        pre = similarities.max(dim=-1)[1]
        matched = pre == object_label

        sequence_output_masked = seq_outputs[0].reshape(batch_size, 2, -1, sequence_output.size(-1))[:, 1,
                                 :hypo_len - 1][masked_pos == 1, :]
        class_logits = self.cls(sequence_output_masked)
        masked_ids = masked_token[masked_token != 0]  # remove padding masks
        masked_loss = self.loss(class_logits.float(), masked_ids)
        return align_loss, object_contras_loss, matched, masked_loss

    def evaluate(self, input_ids, img_feat, total_label, img_mask, input_mask,
                 token_type_ids=None, position_ids=None, head_mask=None,
                 offsets=None, chunk_attention_mask=None, gather_index=None,
                 align_pos=None):
        hypo_len = input_ids.size(1)

        seq_outputs = self.seq_enc(input_ids, img_feats=img_feat, img_mask=img_mask, input_mask=input_mask,
                                   attention_mask=chunk_attention_mask,
                                   position_ids=position_ids, token_type_ids=token_type_ids,
                                   head_mask=head_mask, offsets=offsets, gather_index=gather_index)
        attn_weight = torch.stack(seq_outputs[2][-3:], dim=1).sum(1).sum(1)[:, :hypo_len, hypo_len:]
        attn_weight = attn_weight.masked_fill(attn_weight == 0, -1e5)
        attn_weight = nn.Softmax(dim=-1)(attn_weight)
        correct = 0
        total_sum = 0

        total_label_align = total_label[align_pos == 1].to(dtype=torch.int64)
        attn_weight_align = attn_weight[align_pos == 1, :]
        total_sum += total_label_align.size(0)
        correct += (torch.argmax(attn_weight_align, -1) == total_label_align).sum().item()

        return correct, total_sum
