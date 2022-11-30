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
        BertPooler, BertPreTrainedModel,
		BertPredictionHeadTransform, BertOnlyMLMHead, BertLMPredictionHead,
        BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        load_tf_weights_in_bert)
BertLayerNorm = torch.nn.LayerNorm
import random

from .modeling_utils import CaptionPreTrainedModel, ImgPreTrainedModel
from .modeling_bert import CaptionBertAttention
from oscar.transformers.modeling_gpt2 import GPT2PreTrainedModel,PARALLELIZE_DOCSTRING,DEPARALLELIZE_DOCSTRING,GPT2_INPUTS_DOCSTRING,_TOKENIZER_FOR_DOC,_CONFIG_FOR_DOC,_CHECKPOINT_FOR_DOC,BaseModelOutputWithPastAndCrossAttentions
from transformers.activations import ACT2FN
from oscar.transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM,LogitsProcessorList,MinLengthLogitsProcessor,BeamScorer)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from collections import UserDict
logger = logging.getLogger(__name__)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
import numpy as np
class BeamSearchScorer(BeamScorer):

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty= 1.0,
        do_early_stopping= False,
        num_beam_hyps_to_keep = 1,
        num_beam_groups= 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id = None,
        eos_token_id = None,
    ) :
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), f"Batch can only be done if at least {self.num_beams} beams have been generated"
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id= None,
        eos_token_id= None,
    ) :
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)
        sorted_ids=[]
        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            score_list=[x[0] for x in beam_hyp.beams]
            sorted_id = sorted(range(len(score_list)), key=lambda k: score_list[k])[0]
            sorted_ids.append(sorted_id)
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
                "sorted_ids":sorted_ids
            }
        )
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()

        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer(
            "bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_head, self.split_size // self.n_head, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            mask = self.bias[:, :, ns - nd : ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = (torch.matmul(w, v),)
        if output_attentions:
            outputs += (w,)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            assert hasattr(
                self, "q_attn"
            ), "If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`."
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key.transpose(-2, -1), value)  # transpose to have same shapes
        else:
            present = None

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return (a, present) + attn_outputs[1:]  # a, present, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        attn_outputs = self.attn(
            self.ln_1(hidden_states),
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attn_outputs = self.crossattention(
                self.ln_cross_attn(hidden_states),
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        # residual connection
        hidden_states = hidden_states + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


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
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

    def attention_cal(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                                           head_mask, history_state)
        attention_output = attention_outputs[0]
        return attention_output

    def forward_ffn(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # outputs = (layer_output,)
        return layer_output

class CaptionBertEncoder(BertEncoder):
    """
    Modified from BertEncoder to add support for output_hidden_states.
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)

    def forward_layer(self,index, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        layer_module = self.layer[index]
        layer_outputs = layer_module(
            hidden_states, attention_mask, head_mask,
            history_state)
        hidden_states = layer_outputs[0]
        return hidden_states

    def attention_layer(self,index, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        layer_module=self.layer[index]
        attention_output=layer_module.attention_cal(
                    hidden_states, attention_mask, head_mask,
                    history_state)
        return attention_output

    def ffn_layer(self,index,attention_output):
        layer_module = self.layer[index]
        outputs=layer_module.forward_ffn(attention_output)
        return outputs

class BertImgModel_add(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel_add, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)
        self.linear_1=nn.Linear(config.hidden_size*2,config.hidden_size,bias=False)
        self.linear_2 = nn.Linear(config.hidden_size*2, config.hidden_size, bias=False)
        self.sig = nn.Sigmoid()
        self.img_dim = config.img_feature_dim
        logger.info('BertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        # self.apply(self.init_weights)
        self.init_weights()

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None,region_id=None):
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

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)
            embedding_output_new=embedding_output.clone()
            # concatenate two embeddings
            for index in range(len(input_ids)):
                region = region_id[index]
                for i in range(len(region)):
                    if region[i] != 0:
                        # 加上对应位置的图片向量
                        # 注意需要-1，因为起始位置不同
                        alpha = torch.cat((embedding_output[index][i], img_embedding_output[index][region[i] - 1]), dim=-1)
                        alpha = self.linear_1(alpha)
                        alpha = self.sig(alpha)
                        embedding_output_new[index][i] = (1 - alpha).mul(embedding_output[index][i]) + alpha.mul(
                            img_embedding_output[index][region[i] - 1])
            embedding_output = torch.cat((embedding_output_new, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def cal_emb(self,input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):
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

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
        return embedding_output,head_mask,extended_attention_mask

class BertImgModel(BertPreTrainedModel):
    """ Expand from BertModel to handle image region features as input
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__(config)
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

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        # self.apply(self.init_weights)
        self.init_weights()

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):
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

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def cal_emb(self,input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):
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

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

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
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)
        if encoder_history_states:
            assert img_feats is None, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t': # transpose
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale': # left scaled
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            else:
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
        return embedding_output,head_mask,extended_attention_mask

class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config.add_cross_attention=True
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2Model_guid(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config.add_cross_attention=True
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None



    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        encoder_pooler=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        if encoder_pooler is not None:
            hidden_states = torch.cat((encoder_pooler, hidden_states), dim=1)
            if attention_mask is not None:
                encoder_attention_mask =attention_mask[:,:,:,:1]
                attention_mask=torch.cat((encoder_attention_mask,attention_mask),dim=-1)
        # 直接拼接到最前面
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        if encoder_pooler is not None:
            output_shape=hidden_states.size()
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertForImageCaptioningAndCls(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """
    def __init__(self, config):
        super(BertForImageCaptioningAndCls, self).__init__(config)
        self.config = config
        self.bert = BertImgModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        # self.apply(self.init_weights)
        self.tie_weights()
        self.bert_gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, input_ids_cls, input_ids_gen, img_feat, expl_ids, input_mask=None, attention_mask=None,
                label=None, attn_mask=None,
                token_type_ids_cls=None, token_type_ids_gen=None, position_ids=None, head_mask=None,
                encoder_history_states=None, attention_mask_cross=None, input_mask_enc=None):
        outputs = self.bert(input_ids_gen, img_feats=img_feat, attention_mask=attention_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids_gen,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        cap_label = input_ids_gen[:, :40]
        label = cap_label[:, 1:]
        sequence_output = outputs[0][:, :40]
        sequence_output = sequence_output[:, :-1]
        sequence_output = self.cls(sequence_output)
        loss_bert_gen = self.bert_gen_criterion(
            sequence_output.reshape(sequence_output.size(0) * sequence_output.size(1), -1),
            label.reshape(-1))
        return loss_bert_gen

class BertForImageCaptioningAndCls_add(CaptionPreTrainedModel):
    """
    Bert for Image Captioning.
    """
    def __init__(self, config):
        super(BertForImageCaptioningAndCls_add, self).__init__(config)
        self.config = config
        self.bert = BertImgModel_add(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
        # self.apply(self.init_weights)
        self.tie_weights()
        self.bert_gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.bert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.bert.embeddings.word_embeddings.weight.requires_grad = not freeze

class Oscar_cls(nn.Module):
    def __init__(self, oscar):
        super(Oscar_cls, self).__init__()
        self.oscar = oscar
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier =nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )

    def forward(self, input_ids, img_feat,input_mask=None,label=None,
            token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=None)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        return loss_cls, matched

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        return matched, pre

class Oscar_GPT(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=gpt_toker.pad_token_id)
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier =nn.Linear(self.oscar.config.hidden_size , self.num_labels)

        self.max_len=50
        self.max_gen_len=50
        self.beam_size=beam_size

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=None)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs=self.bert_gpt_proj(outputs[0])
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hs,encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return gen_loss,loss_cls, matched

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]
        attn_prob=outputs[-1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = torch.full((expl_ids.size(0), self.max_gen_len), fill_value=self.gpt_toker.pad_token_id,
                             dtype=int).cuda()

        past_key_values = None
        cur_unfinished = outputs.new(outputs.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = expl_ids[:, 0]
        gpt_attn_prob=[]
        for index in range(self.max_gen_len - 1):
            gpt_out = self.gpt(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values,output_attentions=True)
            past_key_values = gpt_out.past_key_values
            gpt_attn_prob.append(torch.stack(list(gpt_out.cross_attentions),dim=1))
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
        gpt_attn_prob=torch.stack(gpt_attn_prob,dim=1)
        return outputs, matched, pre,attn_prob,gpt_attn_prob
    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        batch_size = input_ids.size(0)
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.beam_size,
            max_length=self.max_gen_len,
            device=torch.device('cuda'))

        logits_processor = LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=4.3), NoRepeatNGramLogitsProcessor(5),
             MinLengthLogitsProcessor(5, eos_token_id=self.gpt_toker.eos_token_id)])
        # instantiate logits processors
        logits_warper = LogitsProcessorList([TopPLogitsWarper(0.3,min_tokens_to_keep=50), TemperatureLogitsWarper(0.7)])

        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = self.beam_sample(input_ids=input_ids, beam_scorer=beam_scorer, encoder_hidden_states=encoder_hs,
                                   encoder_mask=input_mask, logits_processor=logits_processor,
                                   logits_warper=logits_warper)
        # outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return outputs, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        encoder_hidden_states,
        encoder_mask,
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id  = None,
        eos_token_id = None,
        output_attentions  = None,
        output_hidden_states = None,
        output_scores = None,
        return_dict_in_generate  = None,
        **model_kwargs,
    )  :
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = self.max_gen_len
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.gpt_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex=encoder_hidden_states.unsqueeze(1).repeat(1,num_beams,1,1)
        encoder_hidden_states_ex=encoder_hidden_states_ex.reshape((batch_size*num_beams,-1,encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # outputs = self(
            #     **model_inputs,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )
            outputs = self.gpt(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len, max_length=max_length
            # )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]

class Oscar_GPT_kl(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT_kl, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier =nn.Linear(self.oscar.config.hidden_size , self.num_labels)
        self.dec_proj=nn.Linear(self.gpt.config.hidden_size , self.oscar.config.hidden_size)
        # self.dec_classifier =nn.Linear(self.gpt.config.hidden_size , self.num_labels)
        self.max_len=50
        self.max_gen_len=50
        self.beam_size=beam_size

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,eos_index=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=None)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs=self.bert_gpt_proj(outputs[0])
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hs,encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        # gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        gen_loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.gpt_toker.pad_token_id)
        gen_loss =gen_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(-1,shift_logits.size(1))
        gen_loss=gen_loss.sum(1)/attn_mask.sum(1)
        loss_weight=torch.ones_like(matched.float())
        false_weight=torch.full_like(loss_weight,0.5)
        loss_weight = torch.where(matched.int() ==0, false_weight, loss_weight)
        gen_loss=gen_loss*loss_weight

        eos_hidden=torch.gather(gpt_out, 1, eos_index.unsqueeze(-1).repeat(1, 1, gpt_out.size(-1)))
        eos_hidden=self.dec_proj(eos_hidden.squeeze())
        dec_cls_logits = self.classifier(eos_hidden).squeeze()
        dec_loss_cls = loss_fct(dec_cls_logits.view(-1, self.num_labels), label.view(-1))
        dec_pre = dec_cls_logits.max(dim=-1)[1]
        dec_matched = dec_pre == label
        loss_kl = F.kl_div(
            input=F.log_softmax(dec_cls_logits / 1.0, dim=-1),
            target=F.softmax(logits / 1.0, dim=-1),
            reduction="batchmean",
        ) * (1.0 ** 2)
        # relation计算

        return gen_loss,loss_cls, matched,dec_loss_cls,dec_matched,loss_kl

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]
        attn_prob=outputs[-1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = torch.full((expl_ids.size(0), self.max_gen_len), fill_value=self.gpt_toker.pad_token_id,
                             dtype=int).cuda()

        past_key_values = None
        cur_unfinished = outputs.new(outputs.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = expl_ids[:, 0]
        gpt_attn_prob=[]
        for index in range(self.max_gen_len - 1):
            gpt_out = self.gpt(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values,output_attentions=True)
            past_key_values = gpt_out.past_key_values
            gpt_attn_prob.append(torch.stack(list(gpt_out.cross_attentions),dim=1))
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
        gpt_attn_prob=torch.stack(gpt_attn_prob,dim=1)
        return outputs, matched, pre,attn_prob,gpt_attn_prob
    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        batch_size = input_ids.size(0)
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.beam_size,
            max_length=self.max_gen_len,
            device=torch.device('cuda'))

        logits_processor = LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=4.3), NoRepeatNGramLogitsProcessor(5),
             MinLengthLogitsProcessor(5, eos_token_id=self.gpt_toker.eos_token_id)])
        # instantiate logits processors
        logits_warper = LogitsProcessorList([TopPLogitsWarper(0.3,min_tokens_to_keep=50), TemperatureLogitsWarper(0.7)])

        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = self.beam_sample(input_ids=input_ids, beam_scorer=beam_scorer, encoder_hidden_states=encoder_hs,
                                   encoder_mask=input_mask, logits_processor=logits_processor,
                                   logits_warper=logits_warper)
        # outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return outputs, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        encoder_hidden_states,
        encoder_mask,
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id  = None,
        eos_token_id = None,
        output_attentions  = None,
        output_hidden_states = None,
        output_scores = None,
        return_dict_in_generate  = None,
        **model_kwargs,
    )  :
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = self.max_gen_len
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.gpt_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex=encoder_hidden_states.unsqueeze(1).repeat(1,num_beams,1,1)
        encoder_hidden_states_ex=encoder_hidden_states_ex.reshape((batch_size*num_beams,-1,encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # outputs = self(
            #     **model_inputs,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )
            outputs = self.gpt(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len, max_length=max_length
            # )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]

class Oscar_GPT_cls(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT_cls, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=gpt_toker.pad_token_id)
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        self.max_len=50
        self.max_gen_len=50
        self.beam_size=beam_size

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=None)
        pooled_output = outputs[1]
        sequence_ouput = outputs[0].mean(dim=1)
        cls_output = torch.cat((pooled_output, sequence_ouput), -1)
        logits = self.classifier(cls_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs=self.bert_gpt_proj(outputs[0])
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hs,encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return gen_loss,loss_cls, matched

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]
        attn_prob=outputs[-1]

        sequence_ouput = outputs[0].mean(dim=1)
        cls_output = torch.cat((pooled_output, sequence_ouput), -1)
        logits = self.classifier(cls_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = torch.full((expl_ids.size(0), self.max_gen_len), fill_value=self.gpt_toker.pad_token_id,
                             dtype=int).cuda()

        past_key_values = None
        cur_unfinished = outputs.new(outputs.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = expl_ids[:, 0]
        gpt_attn_prob=[]
        for index in range(self.max_gen_len - 1):
            gpt_out = self.gpt(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values,output_attentions=True)
            past_key_values = gpt_out.past_key_values
            gpt_attn_prob.append(torch.stack(list(gpt_out.cross_attentions),dim=1))
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
        gpt_attn_prob=torch.stack(gpt_attn_prob,dim=1)
        return outputs, matched, pre,attn_prob,gpt_attn_prob
    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        batch_size = input_ids.size(0)
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.beam_size,
            max_length=self.max_gen_len,
            device=torch.device('cuda'))

        logits_processor = LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=4.3), NoRepeatNGramLogitsProcessor(5),
             MinLengthLogitsProcessor(5, eos_token_id=self.gpt_toker.eos_token_id)])
        # instantiate logits processors
        logits_warper = LogitsProcessorList([TopPLogitsWarper(0.3,min_tokens_to_keep=50), TemperatureLogitsWarper(0.7)])

        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs = self.beam_sample(input_ids=input_ids, beam_scorer=beam_scorer, encoder_hidden_states=encoder_hs,
                                   encoder_mask=input_mask, logits_processor=logits_processor,
                                   logits_warper=logits_warper)
        # outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return outputs, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer,
        encoder_hidden_states,
        encoder_mask,
        logits_processor = None,
        stopping_criteria = None,
        logits_warper = None,
        max_length = None,
        pad_token_id  = None,
        eos_token_id = None,
        output_attentions  = None,
        output_hidden_states = None,
        output_scores = None,
        return_dict_in_generate  = None,
        **model_kwargs,
    )  :
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = self.max_gen_len
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.gpt_toker.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.eos_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.gpt_toker.convert_tokens_to_ids('<|e_exp|>')
        output_scores = True
        output_attentions = False
        output_hidden_states = (output_hidden_states)
        return_dict_in_generate = (return_dict_in_generate)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
        #     encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        #     encoder_hidden_states = (
        #         model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        #     )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))
        encoder_hidden_states_ex=encoder_hidden_states.unsqueeze(1).repeat(1,num_beams,1,1)
        encoder_hidden_states_ex=encoder_hidden_states_ex.reshape((batch_size*num_beams,-1,encoder_hidden_states_ex.size(-1)))
        encoder_mask_ex = encoder_mask.unsqueeze(1).repeat(1, num_beams, 1)
        encoder_mask_ex = encoder_mask_ex.reshape(
            (batch_size * num_beams, -1))

        while cur_len < max_length:
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # outputs = self(
            #     **model_inputs,
            #     return_dict=True,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            # )
            outputs = self.gpt(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states_ex,
                               encoder_attention_mask=encoder_mask_ex)

            next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len, max_length=max_length
            # )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = F.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            # if model_kwargs["past"] is not None:
            #     model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        return sequence_outputs["sequences"]

class Oscar_GPT_scst(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT_scst, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=gpt_toker.pad_token_id)
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier =nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.max_len=50
        self.beam_size=beam_size

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,sc_train_sample_n=2):

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=None)
        pooled_output = outputs[1].detach()

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs=self.bert_gpt_proj(outputs[0].detach())
        predict_his = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id,
                                 dtype=int).cuda()
        greedy_outputs = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        predict_his[:, 0] = expl_ids[:, 0]
        past_key_values = None
        cur_unfinished = predict_his.new(predict_his.size(0)).fill_(1)
        self.eval()
        cur_len = 0
        with torch.no_grad():
            for index in range(0, self.max_len):
                gpt_out = self.gpt(input_ids=predict_his[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)
                past_key_values = gpt_out.past_key_values
                gpt_out = gpt_out[0][:, -1:, :]
                # 只取最后一个作为当前步的输出
                lm_logits = self.lm_head(gpt_out)
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
                tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
                if index < self.max_len - 1:
                    predict_his[:, index + 1] = tokens_to_add
                greedy_outputs[:, index] = tokens_to_add
                cur_len+=1
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break

        if cur_len == self.max_len:
            greedy_outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)


        self.train()
        predict_his = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id,
                                 dtype=int).cuda()
        predict_his[:, 0] = expl_ids[:, 0]
        predict_his=predict_his.unsqueeze(1).repeat(1,sc_train_sample_n,1)
        predict_his=predict_his.reshape(-1,predict_his.size(-1))
        encoder_hs_exp=encoder_hs.unsqueeze(1).repeat(1,sc_train_sample_n,1,1)
        encoder_hs_exp = encoder_hs_exp.reshape(encoder_hs_exp.size(0)*sc_train_sample_n,-1, encoder_hs_exp.size(-1))
        input_mask_exp=input_mask.unsqueeze(1).repeat(1,sc_train_sample_n,1)
        input_mask_exp = input_mask_exp.reshape(-1, input_mask_exp.size(-1))
        sample_outputs=torch.full(predict_his.size(), fill_value=self.gpt_toker.pad_token_id,dtype=int).cuda()
        unfinished_sents = []
        cur_unfinished = input_ids.new(predict_his.size(0)).fill_(1)
        logprobs = []
        # log of scores for each sentence in the batch
        past_key_values = None
        cur_len=0
        tokens_to_add=predict_his[:, 0]
        for index in range(0, self.max_len):
            gpt_out = self.gpt(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=input_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            lm_logits = top_k_top_p_filtering(lm_logits, top_k=40, top_p=1.0)
            next_token = torch.multinomial(F.softmax(lm_logits, dim=-1), num_samples=1).squeeze(1)

            _scores = F.log_softmax(lm_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)
            tokens_to_add = next_token * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            # if index<self.max_len-1:
            #     predict_his[:, index + 1] = tokens_to_add
            sample_outputs[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            cur_len+=1
            if cur_unfinished.max() == 0:
                break
        # add eos_token_ids to unfinished sentences
        if cur_len == self.max_len:
            sample_outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)
        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        return matched,greedy_outputs,pre,sample_outputs,logprobs.unsqueeze(1),loss_cls

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his = torch.full((expl_ids.size(0),self.max_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        outputs=torch.full((expl_ids.size(0),self.max_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        predict_his[:, 0] =expl_ids[:, 0]
        past_key_values = None
        for index in range(0,self.max_len-1):
            gpt_out = self.gpt(input_ids=predict_his[:,index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask,use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            #只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his[:, index + 1] = gen_label
            outputs[:, index] = gen_label
        return matched,outputs,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return matched,outputs,pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.max_exp_len = max_a_seq_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size * 4, self.oscar.config.hidden_size * 4),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 4, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 4, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        self.cls_embedding=nn.Embedding(self.num_labels,self.gpt_cap.config.n_embd)
        # self.init_weight()
        self.cls_template = 'The answer is'
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]
        #阻断梯度
        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算

        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)


        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]
        encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp,encoder_pooler=encoder_pooler)
        lm_logits = self.lm_head(gpt_out[0][..., 1:, :])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_caption_len - 1):
            if index==0:
                gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values,encoder_pooler=encoder_pooler)
            else:
                gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_cap(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_cap, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.max_exp_len = 50
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]

        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds

        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)

        raw_output = outputs[1]
        # logits操作
        cls_logits =self.cls_output(raw_output)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp)
        lm_logits = self.lm_head(gpt_out[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None
        cur_unfinished = predict_his_cap.new(predict_his_cap.size(0)).fill_(1)
        cur_len = 0
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            if index < self.max_len - 1:
                predict_his_cap[:, index + 1] = tokens_to_add
            output_cap[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_cap[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()

        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)

        cls_logits =self.cls_output(outputs[1])
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask

        predict_his_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 0] = expl_ids[:, 0]
        past_key_values=None
        cur_unfinished = input_ids.new(predict_his_exp.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = predict_his_exp[:, 0]
        for index in range(0, self.max_seq_len - 1):
            gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)

            output_exp[:,index]=tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_exp[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])

        output_cap = self.cap_batch_predict_beam(encoder_hs, input_mask)
        output_cap=output_cap[:, 1:-1]
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()

        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)

        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)

        cls_logits =self.cls_output(outputs[1])
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        output_exp = self.batch_predict_beam(encoder_hs_exp, encoder_mask_exp)

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_seq_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))

        while cur_len < self.max_exp_len:
            out = self.gpt_exp(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            # out = self.gpt_exp(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
            #                        encoder_attention_mask=attn_masks_ex,encoder_pooler=encoder_pooler_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_exp_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_exp_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

    def cap_batch_predict_beam(self, sequence_output, attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_caption_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len = 1
        sequence_output = sequence_output.unsqueeze(1)
        sequence_output_ex = sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),
                                                    sequence_output.size(3))
        sequence_output_ex = sequence_output_ex.reshape(-1, sequence_output.size(2), sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_caption_len:
            out = self.gpt_cap(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                               encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out = out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        # 还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_caption_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_caption_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_ensemble(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_ensemble, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output_0 = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.cls_output_1 = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.max_exp_len = 50
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size * 2, self.oscar.config.hidden_size, bias=False)
        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size * 4, self.oscar.config.hidden_size * 4),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 4, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 4, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        self.cls_embedding=nn.Embedding(self.num_labels,self.gpt_cap.config.n_embd)
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]
        #阻断梯度
        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds


        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)

        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        raw_output = outputs[1]
        # 门控
        # beta = torch.cat((raw_output, pooled_output), dim=-1)
        # beta = self.oscar_gate(beta)
        # beta = self.sig(beta)
        # oscar_ensemble = (1 - beta).mul(raw_output) + beta.mul(pooled_output)
        # cls_logits = self.cls_output(oscar_ensemble)

        # logits操作
        cls_logits_0 =self.cls_output_0(pooled_output)
        cls_logits_1 =self.cls_output_1(raw_output)
        cls_logits=cls_logits_0+cls_logits_1

        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        # relation计算
        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation_0 = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist_0 = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        # relation计算
        hypo_vis_cap_rel = self.relation_linear(raw_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation_1 = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(raw_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist_1 = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        loss_relation=loss_relation_0+loss_relation_1
        loss_dist=loss_dist_0+loss_dist_1

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        # cls_logits = cls_logits / 0.1
        # # 让softmax输出接近one-hot
        # soft_output = F.softmax(cls_logits, dim=-1)
        # encoder_pooler= torch.matmul(soft_output.unsqueeze(1), self.cls_embedding.weight)

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp)
        lm_logits = self.lm_head(gpt_out[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None
        cur_unfinished = predict_his_cap.new(predict_his_cap.size(0)).fill_(1)
        cur_len = 0
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            if index < self.max_len - 1:
                predict_his_cap[:, index + 1] = tokens_to_add
            output_cap[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_cap[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()


        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        # beta = torch.cat((outputs[1], pooled_output), dim=-1)
        # beta = self.oscar_gate(beta)
        # beta = self.sig(beta)
        # oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(pooled_output)
        # cls_logits = self.cls_output(oscar_ensemble)
        cls_logits_0 =self.cls_output_0(pooled_output)
        cls_logits_1 =self.cls_output_1(outputs[1])
        cls_logits=cls_logits_0+cls_logits_1

        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        # encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)
        # encoder_pooler=self.cls_embedding(pre).unsqueeze(1)

        predict_his_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 0] = expl_ids[:, 0]
        past_key_values=None
        cur_unfinished = input_ids.new(predict_his_exp.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = predict_his_exp[:, 0]
        for index in range(0, self.max_seq_len - 1):
            # if index==0:
            #     gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
            #                    encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values,encoder_pooler=encoder_pooler)
            # else:
            gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)

            output_exp[:,index]=tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_exp[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])

        output_cap = self.cap_batch_predict_beam(encoder_hs, input_mask)
        output_cap=output_cap[:, 1:-1]
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()

        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)

        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)

        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        # beta = torch.cat((outputs[1], pooled_output), dim=-1)
        # beta = self.oscar_gate(beta)
        # beta = self.sig(beta)
        # oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(pooled_output)
        # cls_logits = self.cls_output(oscar_ensemble)
        cls_logits_0 =self.cls_output_0(pooled_output)
        cls_logits_1 =self.cls_output_1(outputs[1])
        cls_logits=cls_logits_0+cls_logits_1
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        output_exp = self.batch_predict_beam(encoder_hs_exp, encoder_mask_exp)

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_seq_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))

        while cur_len < self.max_exp_len:
            out = self.gpt_exp(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)

            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_exp_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_exp_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

    def cap_batch_predict_beam(self, sequence_output, attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_caption_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len = 1
        sequence_output = sequence_output.unsqueeze(1)
        sequence_output_ex = sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),
                                                    sequence_output.size(3))
        sequence_output_ex = sequence_output_ex.reshape(-1, sequence_output.size(2), sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_caption_len:
            out = self.gpt_cap(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                               encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out = out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        # 还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_caption_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_caption_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_ensemble_concat(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_ensemble_concat, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size* 2, self.oscar.config.hidden_size * 4),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 4, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 4, self.num_labels)
        )
        self.max_exp_len = 50
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size* 2, self.oscar.config.hidden_size* 2)
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size * 2, self.oscar.config.hidden_size, bias=False)
        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size * 8, self.oscar.config.hidden_size * 16),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 16, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 16, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size* 2, self.oscar.config.hidden_size* 2)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        self.cls_embedding=nn.Embedding(self.num_labels,self.gpt_cap.config.n_embd)
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]
        #阻断梯度
        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds


        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)

        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        raw_output = outputs[1]
        oscar_ensemble =  torch.cat((raw_output, pooled_output), dim=-1)
        cls_logits = self.cls_output(oscar_ensemble)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算

        hypo_vis_cap_rel = self.relation_linear(oscar_ensemble)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(oscar_ensemble)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)


        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        # cls_logits = cls_logits / 0.1
        # # 让softmax输出接近one-hot
        # soft_output = F.softmax(cls_logits, dim=-1)
        # encoder_pooler= torch.matmul(soft_output.unsqueeze(1), self.cls_embedding.weight)

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp)
        lm_logits = self.lm_head(gpt_out[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None
        cur_unfinished = predict_his_cap.new(predict_his_cap.size(0)).fill_(1)
        cur_len = 0
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            if index < self.max_len - 1:
                predict_his_cap[:, index + 1] = tokens_to_add
            output_cap[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_cap[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()


        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        oscar_ensemble = torch.cat((outputs[1], pooled_output), dim=-1)
        cls_logits = self.cls_output(oscar_ensemble)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        # encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)
        encoder_pooler=self.cls_embedding(pre).unsqueeze(1)

        predict_his_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 0] = expl_ids[:, 0]
        past_key_values=None
        cur_unfinished = input_ids.new(predict_his_exp.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = predict_his_exp[:, 0]
        for index in range(0, self.max_seq_len - 1):
            # if index==0:
            #     gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
            #                    encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values,encoder_pooler=encoder_pooler)
            # else:
            gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)

            output_exp[:,index]=tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_exp[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])

        output_cap = self.cap_batch_predict_beam(encoder_hs, input_mask)
        output_cap=output_cap[:, 1:-1]
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()

        hypo_vis_cap = torch.cat(( outputs[0], cap_soft_hidden), dim=1)

        hypo_vis_cap_mask = torch.cat(( input_mask, attn_mask_cap), dim=1)

        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        oscar_ensemble = torch.cat((outputs[1], pooled_output), dim=-1)
        cls_logits = self.cls_output(oscar_ensemble)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp=hypo_vis_cap_mask
        output_exp = self.batch_predict_beam(encoder_hs_exp, encoder_mask_exp)

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_seq_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))

        while cur_len < self.max_exp_len:
            out = self.gpt_exp(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_exp_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_exp_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

    def cap_batch_predict_beam(self, sequence_output, attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_caption_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len = 1
        sequence_output = sequence_output.unsqueeze(1)
        sequence_output_ex = sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),
                                                    sequence_output.size(3))
        sequence_output_ex = sequence_output_ex.reshape(-1, sequence_output.size(2), sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_caption_len:
            out = self.gpt_cap(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                               encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out = out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        # 还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_caption_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_caption_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_ensemble_noobj(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_ensemble_noobj, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.hypo_len=max_a_seq_len
        self.max_exp_len = 50
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size * 2, self.oscar.config.hidden_size, bias=False)
        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size * 4, self.oscar.config.hidden_size * 4),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 4, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 4, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        self.cls_embedding=nn.Embedding(self.num_labels,self.gpt_cap.config.n_embd)
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]
        #阻断梯度
        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        raw_output = outputs[1]
        beta = torch.cat((raw_output, pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(raw_output) + beta.mul(pooled_output)
        cls_logits = self.cls_output(oscar_ensemble)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算

        hypo_vis_cap_rel = self.relation_linear(oscar_ensemble)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(oscar_ensemble)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        hypo_vis_cap_wo_obj=torch.cat((hypo_vis_cap[:,:self.hypo_len],hypo_vis_cap[:,self.max_seq_len:]),dim=1)
        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap_wo_obj)
        encoder_mask_exp=torch.cat((hypo_vis_cap_mask[:,:self.hypo_len],hypo_vis_cap_mask[:,self.max_seq_len:]),dim=1)
        encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp,encoder_pooler=encoder_pooler)
        lm_logits = self.lm_head(gpt_out[0][..., 1:, :])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None
        cur_unfinished = predict_his_cap.new(predict_his_cap.size(0)).fill_(1)
        cur_len = 0
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            if index < self.max_len - 1:
                predict_his_cap[:, index + 1] = tokens_to_add
            output_cap[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_cap[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool),self.gpt_toker.eos_token_id)
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(pooled_output)

        cls_logits = self.cls_output(oscar_ensemble)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        hypo_vis_cap_wo_obj = torch.cat((hypo_vis_cap[:, :self.hypo_len], hypo_vis_cap[:, self.max_seq_len:]), dim=1)
        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap_wo_obj)
        encoder_mask_exp = torch.cat((hypo_vis_cap_mask[:, :self.hypo_len], hypo_vis_cap_mask[:, self.max_seq_len:]),dim=1)
        encoder_pooler = torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        predict_his_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_seq_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 0] = expl_ids[:, 0]
        past_key_values=None
        cur_unfinished = input_ids.new(predict_his_exp.size(0)).fill_(1)
        cur_len = 0
        tokens_to_add = predict_his_exp[:, 0]
        for index in range(0, self.max_seq_len - 1):
            if index==0:
                gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values,encoder_pooler=encoder_pooler)
            else:
                gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)

            output_exp[:,index]=tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            if cur_unfinished.max() == 0:
                break
            cur_len += 1
        if cur_len == self.max_len:
            output_exp[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])

        output_cap = self.cap_batch_predict_beam(encoder_hs, input_mask)
        output_cap=output_cap[:, 1:-1]
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(pooled_output)

        cls_logits = self.cls_output(oscar_ensemble)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        hypo_vis_cap_wo_obj = torch.cat((hypo_vis_cap[:, :self.hypo_len], hypo_vis_cap[:, self.max_seq_len:]), dim=1)
        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap_wo_obj)
        encoder_mask_exp = torch.cat((hypo_vis_cap_mask[:, :self.hypo_len], hypo_vis_cap_mask[:, self.max_seq_len:]),dim=1)
        encoder_pooler = torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)
        output_exp = self.batch_predict_beam(encoder_hs_exp, encoder_mask_exp,encoder_pooler)

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks,encoder_pooler):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_seq_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        encoder_pooler_ex=encoder_pooler.unsqueeze(1).repeat(1,self.beam_size,1,1)
        encoder_pooler_ex=encoder_pooler_ex.reshape(-1,encoder_pooler.size(1),encoder_pooler.size(2))
        while cur_len < self.max_exp_len:
            out = self.gpt_exp(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex,encoder_pooler=encoder_pooler_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_exp_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_exp_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

    def cap_batch_predict_beam(self, sequence_output, attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_caption_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len = 1
        sequence_output = sequence_output.unsqueeze(1)
        sequence_output_ex = sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),
                                                    sequence_output.size(3))
        sequence_output_ex = sequence_output_ex.reshape(-1, sequence_output.size(2), sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_caption_len:
            out = self.gpt_cap(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                               encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out = out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        # 还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_caption_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_caption_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.max_exp_len = max_a_seq_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size * 4, self.oscar.config.hidden_size * 4),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 4, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 4, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        self.cls_embedding=nn.Embedding(self.num_labels,self.gpt_cap.config.n_embd)
        # self.init_weight()
        self.cls_template = 'The answer is'
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence=outputs[0]
        #阻断梯度
        encoder_hs=self.bert_gpt_proj(oscar_sequence)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        hypo_vis_cap=torch.cat((oscar_sequence,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算

        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)


        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]
        encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp,encoder_pooler=encoder_pooler)
        lm_logits = self.lm_head(gpt_out[0][..., 1:, :])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))
        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings.word_embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler= torch.matmul(cls_logits.unsqueeze(1), self.cls_embedding.weight)

        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_caption_len - 1):
            if index==0:
                gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values,encoder_pooler=encoder_pooler)
            else:
                gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_cls_sep(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_cls_sep, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.cls_hidden=self.oscar.config.hidden_size+self.gpt_cap.config.n_embd
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size*2, self.oscar.config.hidden_size, bias=False)
        self.cls_gate = nn.Linear(self.cls_hidden, self.num_labels, bias=False)
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=max_a_seq_len
        self.beam_size=beam_size

        self.oscar_cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size,self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size* 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.gpt_cls_output = nn.Sequential(
            nn.Linear(self.gpt_cap.config.n_embd,self.gpt_cap.config.n_embd * 2),
            nn.ReLU(),
            nn.LayerNorm(self.gpt_cap.config.n_embd* 2, eps=1e-12),
            nn.Linear(self.gpt_cap.config.n_embd * 2, self.num_labels)
        )
        self.max_exp_len = max_a_seq_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.cls_hidden, self.cls_hidden)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.cls_hidden * 4, self.cls_hidden * 8),
            nn.ReLU(),
            nn.LayerNorm(self.cls_hidden * 8, eps=1e-12),
            nn.Linear(self.cls_hidden * 8, 2)
        )
        self.dist_linear = nn.Linear(self.cls_hidden, self.cls_hidden)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        sequence_output=outputs[0]

        encoder_hs=self.bert_gpt_proj(sequence_output)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids=  torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds=self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden+=cap_position_embeds
        hypo_vis_cap=torch.cat((sequence_output,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp)
        lm_logits = self.lm_head(gpt_out[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))

        end_index = np.argwhere(expl_ids.cpu().numpy() == self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(gpt_out[0].size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(gpt_out[0][i,item[1]-1])
                    #前一个时间步的输出才是eos
                    break
        raw_output=outputs[1]
        exp_pooled_out=torch.stack(exp_pooled_out,dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta=torch.cat((raw_output,oscar_pooled_output),dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble= (1 - beta).mul(raw_output) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算
        pooled_output = torch.cat((exp_pooled_out, oscar_ensemble), dim=1)
        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)



        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        exp_hidden=torch.zeros((expl_ids.size(0),self.max_caption_len,self.gpt_cap.config.n_embd)).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_exp_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            exp_hidden[:,index:index+1]=gpt_out
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        end_index = np.argwhere(output_exp.cpu().numpy() ==self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(output_exp.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(exp_hidden[i,item[1]])
                    break
                if j+1==len(end_index):
                    #没有找到对应位置免责直接拿最后一维
                    exp_pooled_out.append(exp_hidden[i, -2])

        exp_pooled_out = torch.stack(exp_pooled_out, dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], oscar_pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_cls_sep_sf(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len,max_exp_len=50):
        super(Oscar_GPT_gen_cls_sep_sf, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.cls_hidden=self.oscar.config.hidden_size+self.gpt_cap.config.n_embd
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size*2, self.oscar.config.hidden_size, bias=False)
        self.cls_gate = nn.Linear(self.cls_hidden, self.num_labels, bias=False)
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=max_a_seq_len
        self.beam_size=beam_size

        self.oscar_cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size,self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size* 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.gpt_cls_output = nn.Sequential(
            nn.Linear(self.gpt_cap.config.n_embd,self.gpt_cap.config.n_embd * 2),
            nn.ReLU(),
            nn.LayerNorm(self.gpt_cap.config.n_embd* 2, eps=1e-12),
            nn.Linear(self.gpt_cap.config.n_embd * 2, self.num_labels)
        )
        self.max_exp_len = max_exp_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.cls_hidden, self.cls_hidden)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.cls_hidden * 4, self.cls_hidden * 8),
            nn.ReLU(),
            nn.LayerNorm(self.cls_hidden * 8, eps=1e-12),
            nn.Linear(self.cls_hidden * 8, 2)
        )
        self.dist_linear = nn.Linear(self.cls_hidden, self.cls_hidden)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,sample_prob,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        sequence_output=outputs[0]

        encoder_hs=self.bert_gpt_proj(sequence_output)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids=  torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds=self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden+=cap_position_embeds
        hypo_vis_cap=torch.cat((sequence_output,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]

        predict_his = torch.full((expl_ids.size(0), expl_ids.size(1)), fill_value=self.gpt_toker.pad_token_id,
                                 dtype=int).cuda()
        greedy_outputs = torch.full((expl_ids.size(0),expl_ids.size(1)), fill_value=self.gpt_toker.pad_token_id,
                                    dtype=int).cuda()
        # predict_his[:, 0] = expl_ids[:, 0]
        past_key_values = None
        cur_unfinished = predict_his.new(predict_his.size(0)).fill_(1)

        tokens_to_add = expl_ids[:, 0]
        lm_logits_list=[]
        gpt_hidden_list=[]
        end_token=self.gpt_toker.convert_tokens_to_ids("<|e_exp|>")
        for index in range(0, expl_ids.size(1)-1):
            gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out[1]
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            lm_logits_list.append(lm_logits)
            gpt_hidden_list.append(gpt_out)
            gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
            prob=random.random()
            if end_token in expl_ids[:,index+1] or prob <sample_prob:
                #保证classifier token的存在
                tokens_to_add = expl_ids[:,index+1]
            else:
                tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)

            greedy_outputs[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())

        lm_logits=torch.stack(lm_logits_list,dim=1).squeeze()
        gpt_hidden=torch.stack(gpt_hidden_list,dim=1).squeeze()
        # lm_logits = self.lm_head(gpt_out[0])
        # shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(lm_logits.view(-1, lm_logits.size(-1)),
                                          shift_labels.view(-1))

        end_index = np.argwhere(greedy_outputs.cpu().numpy() == self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(gpt_hidden.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(gpt_hidden[i,item[1]])
                    break
                if j ==len(end_index)-1:
                    exp_pooled_out.append(gpt_hidden[i, -1])
        raw_output=outputs[1]
        exp_pooled_out=torch.stack(exp_pooled_out,dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta=torch.cat((raw_output,oscar_pooled_output),dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble= (1 - beta).mul(raw_output) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算
        pooled_output = torch.cat((exp_pooled_out, oscar_ensemble), dim=1)
        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)



        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        exp_hidden=torch.zeros((expl_ids.size(0),self.max_caption_len,self.gpt_cap.config.n_embd)).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_exp_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            exp_hidden[:,index:index+1]=gpt_out
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        end_index = np.argwhere(output_exp.cpu().numpy() ==self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(output_exp.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(exp_hidden[i,item[1]])
                    break
                if j+1==len(end_index):
                    #没有找到对应位置免责直接拿最后一维
                    exp_pooled_out.append(exp_hidden[i, -2])

        exp_pooled_out = torch.stack(exp_pooled_out, dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], oscar_pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_cls_sep_scst(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_cls_sep_scst, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.cls_hidden=self.oscar.config.hidden_size+self.gpt_cap.config.n_embd
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size*2, self.oscar.config.hidden_size, bias=False)
        self.cls_gate = nn.Linear(self.cls_hidden, self.num_labels, bias=False)
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=max_a_seq_len
        self.beam_size=beam_size

        self.oscar_cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size,self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size* 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.gpt_cls_output = nn.Sequential(
            nn.Linear(self.gpt_cap.config.n_embd,self.gpt_cap.config.n_embd * 2),
            nn.ReLU(),
            nn.LayerNorm(self.gpt_cap.config.n_embd* 2, eps=1e-12),
            nn.Linear(self.gpt_cap.config.n_embd * 2, self.num_labels)
        )
        self.max_exp_len = max_a_seq_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.cls_hidden, self.cls_hidden)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.cls_hidden * 4, self.cls_hidden * 8),
            nn.ReLU(),
            nn.LayerNorm(self.cls_hidden * 8, eps=1e-12),
            nn.Linear(self.cls_hidden * 8, 2)
        )
        self.dist_linear = nn.Linear(self.cls_hidden, self.cls_hidden)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,sc_train_sample_n=1):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        sequence_output=outputs[0].detach()

        encoder_hs=self.bert_gpt_proj(sequence_output)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids=  torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds=self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden+=cap_position_embeds
        hypo_vis_cap=torch.cat((sequence_output,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]

        predict_his = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id,
                                 dtype=int).cuda()
        greedy_outputs = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id,
                                    dtype=int).cuda()
        predict_his[:, 0] = expl_ids[:, 0]
        past_key_values = None
        cur_unfinished = predict_his.new(predict_his.size(0)).fill_(1)
        self.eval()
        cur_len = 0
        with torch.no_grad():
            for index in range(0, self.max_len ):
                gpt_out = self.gpt_exp(input_ids=predict_his[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
                past_key_values = gpt_out.past_key_values
                gpt_out = gpt_out[0][:, -1:, :]
                # 只取最后一个作为当前步的输出
                lm_logits = self.lm_head(gpt_out)
                gen_label = torch.argmax(lm_logits, dim=-1).squeeze()
                tokens_to_add = gen_label * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
                if index < self.max_len - 1:
                    predict_his[:, index + 1] = tokens_to_add
                greedy_outputs[:, index] = tokens_to_add
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
                if cur_unfinished.max() == 0:
                    break
        if cur_len == self.max_len:
            greedy_outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)

        self.train()
        predict_his = torch.full((expl_ids.size(0), self.max_len), fill_value=self.gpt_toker.pad_token_id,dtype=int).cuda()
        predict_his[:, 0] = expl_ids[:, 0]
        predict_his = predict_his.unsqueeze(1).repeat(1, sc_train_sample_n, 1)
        predict_his = predict_his.reshape(-1, predict_his.size(-1))
        encoder_hs_exp = encoder_hs_exp.unsqueeze(1).repeat(1, sc_train_sample_n, 1, 1)
        encoder_hs_exp = encoder_hs_exp.reshape(encoder_hs_exp.size(0) * sc_train_sample_n, -1, encoder_hs_exp.size(-1))
        encoder_mask_exp = encoder_mask_exp.unsqueeze(1).repeat(1, sc_train_sample_n, 1)
        encoder_mask_exp = encoder_mask_exp.reshape(-1, encoder_mask_exp.size(-1))
        sample_outputs = torch.full(predict_his.size(), fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        unfinished_sents = []
        cur_unfinished = input_ids.new(predict_his.size(0)).fill_(1)
        logprobs = []
        # log of scores for each sentence in the batch
        past_key_values = None
        cur_len = 0
        tokens_to_add = predict_his[:, 0]
        sample_hidden=[]
        for index in range(0, self.max_len):
            gpt_out = self.gpt_exp(input_ids=tokens_to_add.unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            next_token = torch.multinomial(F.softmax(lm_logits, dim=-1), num_samples=1).squeeze(1)
            sample_hidden.append(gpt_out)
            _scores = F.log_softmax(lm_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)
            tokens_to_add = next_token * cur_unfinished + self.gpt_toker.pad_token_id * (1 - cur_unfinished)
            # if index<self.max_len-1:
            #     predict_his[:, index + 1] = tokens_to_add
            sample_outputs[:, index] = tokens_to_add
            cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(self.gpt_toker.eos_token_id).long())
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            cur_len += 1
            if cur_unfinished.max() == 0:
                break
        # add eos_token_ids to unfinished sentences
        if cur_len == self.max_len:
            sample_outputs[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), self.gpt_toker.eos_token_id)
        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)
        sample_hidden=torch.stack(sample_hidden,dim=1)
        end_index = np.argwhere(sample_outputs.cpu().numpy() == self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(sample_hidden.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(sample_hidden[i,item[1]])
                    break
                if j ==len(end_index)-1:
                    exp_pooled_out.append(sample_hidden[i, -1])
        raw_output=outputs[1]
        exp_pooled_out=torch.stack(exp_pooled_out,dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta=torch.cat((raw_output,oscar_pooled_output),dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble= (1 - beta).mul(raw_output) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        loss_cls = self.cls_criterion(cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label
        # relation计算
        pooled_output = torch.cat((exp_pooled_out, oscar_ensemble), dim=1)
        hypo_vis_cap_rel = self.relation_linear(pooled_output)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(pooled_output)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        return loss_cls,gen_cap_loss,matched,loss_relation,loss_dist,greedy_outputs,sample_outputs,logprobs.unsqueeze(1)

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)



        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        exp_hidden=torch.zeros((expl_ids.size(0),self.max_caption_len,self.gpt_cap.config.n_embd)).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_exp_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            exp_hidden[:,index:index+1]=gpt_out
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        end_index = np.argwhere(output_exp.cpu().numpy() ==self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(output_exp.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(exp_hidden[i,item[1]])
                    break
                if j+1==len(end_index):
                    #没有找到对应位置免责直接拿最后一维
                    exp_pooled_out.append(exp_hidden[i, -2])

        exp_pooled_out = torch.stack(exp_pooled_out, dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], oscar_pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)
        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        alpha = torch.cat((oscar_pooled_output, exp_pooled_out), dim=-1)
        alpha = self.cls_gate(alpha)
        alpha = self.sig(alpha)
        cls_logits = (1 - alpha).mul(oscar_cls_logits) + alpha.mul(gpt_cls_logits)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        return output_cap,output_exp,matched,pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_cls_sep_kl(nn.Module):
    def __init__(self, oscar,cls_oscar,gpt_cap,gpt_exp,gpt_toker,beam_size,max_seq_len,max_a_seq_len,max_caption_len):
        super(Oscar_GPT_gen_cls_sep_kl, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt_cap = gpt_cap
        self.gpt_exp = gpt_exp
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt_cap.config.n_embd)
        self.vocab_num=self.gpt_cap.vocab_size
        self.lm_head = nn.Linear(self.gpt_cap.config.n_embd, self.gpt_cap.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=self.gpt_toker.pad_token_id)
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.cls_hidden=self.oscar.config.hidden_size+self.gpt_cap.config.n_embd
        self.oscar_gate = nn.Linear(self.oscar.config.hidden_size * 2, self.oscar.config.hidden_size, bias=False)
        self.oscar_cls_output = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size,self.oscar.config.hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size* 2, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 2, self.num_labels)
        )
        self.gpt_cls_output = nn.Sequential(
            nn.Linear(self.gpt_cap.config.n_embd,self.gpt_cap.config.n_embd * 2),
            nn.ReLU(),
            nn.LayerNorm(self.gpt_cap.config.n_embd* 2, eps=1e-12),
            nn.Linear(self.gpt_cap.config.n_embd * 2, self.num_labels)
        )
        self.max_exp_len = max_a_seq_len
        self.max_seq_len = max_seq_len
        self.max_caption_len = max_caption_len
        self.cls_num = 4
        self.cls_oscar = cls_oscar.bert.encoder
        self.cls_oscar.layer = self.cls_oscar.layer[:self.cls_num]
        self.pooler = BertPooler(self.oscar.config)
        self.oscar.config.n_head = 16
        self.oscar.config.attn_pdrop = 0.1
        self.oscar.config.resid_pdrop = 0.1
        self.relation_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)

        self.relation_cls = nn.Sequential(
            nn.Linear(self.oscar.config.hidden_size* 4, self.oscar.config.hidden_size* 8),
            nn.ReLU(),
            nn.LayerNorm(self.oscar.config.hidden_size * 8, eps=1e-12),
            nn.Linear(self.oscar.config.hidden_size * 8, 2)
        )
        self.dist_linear = nn.Linear(self.oscar.config.hidden_size, self.oscar.config.hidden_size)
        self.margin_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2, eps=1e-06, swap=False, size_average=None,
                                                      reduce=None, reduction='mean')
        # self.init_weight()
        self.sig = nn.Sigmoid()
        self.id2label = {0: '<neutral>',
                         1: '<contradiction>',
                         2: '<entailment>'}


    def forward(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        input_mask = input_mask.float()
        attn_mask = attn_mask.float()
        attn_mask_cap = attn_mask_cap.float()

        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states)
        oscar_sequence_out=outputs[0].detach()

        encoder_hs=self.bert_gpt_proj(oscar_sequence_out)
        gpt_out_cap = self.gpt_cap(input_ids=cap_ids, attention_mask=attn_mask_cap, encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True)
        lm_logits_cap = self.lm_head(gpt_out_cap[0])
        shift_logits_cap = lm_logits_cap[..., :-1, :].contiguous()
        shift_labels_cap = cap_ids[..., 1:].contiguous()
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cap.size(-1)),
                                          shift_labels_cap.view(-1))
        gpt_cap_logits = shift_logits_cap / 0.1
        # 让softmax输出接近one-hot
        soft_output = F.softmax(gpt_cap_logits, dim=-1)
        cap_soft_hidden = torch.matmul(soft_output, self.oscar.bert.embeddings.word_embeddings.weight)
        cap_position_ids = torch.arange(0, cap_soft_hidden.size(1), dtype=torch.long).cuda()
        cap_position_embeds = self.oscar.bert.embeddings.position_embeddings(cap_position_ids)
        cap_soft_hidden += cap_position_embeds
        hypo_vis_cap=torch.cat((oscar_sequence_out,cap_soft_hidden),dim=1)
        hypo_vis_cap_mask=torch.cat((input_mask,attn_mask_cap[:,1:]),dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        encoder_hs_exp=self.bert_gpt_proj(hypo_vis_cap[:,1:])
        encoder_mask_exp=hypo_vis_cap_mask[:,1:]

        gpt_out = self.gpt_exp(input_ids=expl_ids, attention_mask=attn_mask, encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp)
        lm_logits = self.lm_head(gpt_out[0])
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        gen_exp_loss = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1))

        end_index = np.argwhere(expl_ids.cpu().numpy() == self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(gpt_out[0].size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(gpt_out[0][i,item[1]-1])
                    #前一个时间步的输出才是eos
                    break
        exp_pooled_out=torch.stack(exp_pooled_out,dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        raw_output = outputs[1].detach()
        beta = torch.cat((raw_output, oscar_pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(raw_output) + beta.mul(oscar_pooled_output)

        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)

        loss_cls = self.cls_criterion(oscar_cls_logits.view(-1, self.num_labels), label.view(-1))
        pre = oscar_cls_logits.max(dim=-1)[1]
        matched = pre == label

        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)
        gpt_loss_cls = self.cls_criterion(gpt_cls_logits.view(-1, self.num_labels), label.view(-1))
        gpt_pre = gpt_cls_logits.max(dim=-1)[1]
        gpt_matched = gpt_pre == label

        loss_kl = F.kl_div(
            input=F.log_softmax(oscar_cls_logits / 1.0, dim=-1),
            target=F.softmax(gpt_cls_logits / 1.0, dim=-1),
            reduction="batchmean",
        ) * (1.0 ** 2)
        # relation计算

        hypo_vis_cap_rel = self.relation_linear(oscar_ensemble)
        hypo_vis_cap_rel = F.relu(hypo_vis_cap_rel)
        anchor_pair = hypo_vis_cap_rel[1].unsqueeze(0).repeat(2, 1)
        pos_neg_pairs = torch.cat((hypo_vis_cap_rel[0].unsqueeze(0), hypo_vis_cap_rel[-1].unsqueeze(0)), dim=0)
        pairs_mul = anchor_pair.mul(pos_neg_pairs)
        pairs_minus = anchor_pair - pos_neg_pairs
        relation_pairs = torch.cat((anchor_pair, pos_neg_pairs, pairs_mul, pairs_minus), dim=-1)
        logits = self.relation_cls(relation_pairs)
        relation_label = torch.tensor(([label[0] == label[1], label[-1] == label[1]]), dtype=label.dtype).cuda()
        loss_relation = self.cls_criterion(logits.view(-1, 2), relation_label.view(-1))
        # dis计算
        hypo_vis_cap_dist = self.dist_linear(oscar_ensemble)
        hypo_vis_cap_dist = F.relu(hypo_vis_cap_dist)
        anchor_pair = hypo_vis_cap_dist[1].unsqueeze(0)
        pos_neg_pairs = torch.cat((hypo_vis_cap_dist[0].unsqueeze(0), hypo_vis_cap_dist[-1].unsqueeze(0)), dim=0)
        neg_pair = pos_neg_pairs[torch.where(relation_label == 0)]
        pos_pair = pos_neg_pairs[torch.where(relation_label == 1)]
        loss_dist = self.margin_loss(anchor_pair, pos_pair, neg_pair)

        return loss_cls,gen_exp_loss,gen_cap_loss,matched,loss_relation,loss_dist,gpt_loss_cls,gpt_matched,loss_kl

    def evaluate(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)



        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        exp_hidden=torch.zeros((expl_ids.size(0),self.max_caption_len,self.gpt_cap.config.n_embd)).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values=None
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs_exp,
                               encoder_attention_mask=encoder_mask_exp, use_cache=True, past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            exp_hidden[:,index:index+1]=gpt_out
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:,index+1]=gen_label
            output_exp[:,index]=gen_label

        end_index = np.argwhere(output_exp.cpu().numpy() ==  self.gpt_toker.convert_tokens_to_ids("<classifier>"))
        exp_pooled_out=[]
        for i in range(output_exp.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    exp_pooled_out.append(exp_hidden[i,item[1]])
                    break

        exp_pooled_out = torch.stack(exp_pooled_out, dim=0)
        oscar_pooled_output = self.pooler(hypo_vis_cap)
        beta = torch.cat((outputs[1], oscar_pooled_output), dim=-1)
        beta = self.oscar_gate(beta)
        beta = self.sig(beta)
        oscar_ensemble = (1 - beta).mul(outputs[1]) + beta.mul(oscar_pooled_output)
        oscar_cls_logits = self.oscar_cls_output(oscar_ensemble)


        pre = oscar_cls_logits.max(dim=-1)[1]
        matched = pre == label

        gpt_cls_logits = self.gpt_cls_output(exp_pooled_out)

        gpt_pre = gpt_cls_logits.max(dim=-1)[1]
        gpt_matched = gpt_pre == label

        return output_cap,output_exp,matched,pre,gpt_matched,gpt_pre

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,cap_ids,input_mask=None,label=None,attn_mask=None,attn_mask_cap=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states)
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_cap = torch.full((cap_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_cap[:, :1] = cap_ids[:, :1]
        past_key_values = None

        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_cap(input_ids=predict_his_cap[:, index].unsqueeze(-1), encoder_hidden_states=encoder_hs,
                                   encoder_attention_mask=input_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = gpt_out.past_key_values
            gpt_out_logits = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out_logits)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_cap[:, index + 1] = gen_label
            output_cap[:, index] = gen_label
        cap_soft_hidden = self.oscar.bert.embeddings(output_cap[:, :-1])
        attn_mask_cap = torch.ones((cap_soft_hidden.size(0), cap_soft_hidden.size(1)), dtype=torch.float32).cuda()
        end_index = np.argwhere(output_cap.cpu().numpy() == self.gpt_toker.eos_token_id)
        for i in range(attn_mask_cap.size(0)):
            for j in range(len(end_index)):
                item = end_index[j]
                if item[0] == i:
                    attn_mask_cap[i, item[-1] + 1:] = 0
                    break
        attn_mask_cap = attn_mask_cap.float()
        hypo_vis_cap = torch.cat((outputs[0], cap_soft_hidden), dim=1)
        hypo_vis_cap_mask = torch.cat((input_mask, attn_mask_cap[:, 1:]), dim=1)
        extended_hypo_vis_cap_mask = hypo_vis_cap_mask.unsqueeze(1).unsqueeze(2)
        extended_hypo_vis_cap_mask = (1.0 - extended_hypo_vis_cap_mask) * -10000.0

        for i in range(self.cls_num):
            hypo_vis_cap = self.cls_oscar.forward_layer(i, hypo_vis_cap, extended_hypo_vis_cap_mask)

        pooled_output = self.pooler(hypo_vis_cap)
        cls_logits = self.cls_output(pooled_output)
        pre = cls_logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs_exp = self.bert_gpt_proj(hypo_vis_cap)
        encoder_mask_exp = hypo_vis_cap_mask
        encoder_pooler = self.bert_gpt_proj(pooled_output.unsqueeze(1))
        cls_gpt = self.gpt_exp(inputs_embeds=encoder_pooler, use_cache=True)
        predict_his_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                     dtype=int).cuda()
        output_exp = torch.full((expl_ids.size(0), self.max_caption_len), fill_value=self.gpt_toker.pad_token_id,
                                dtype=int).cuda()
        predict_his_exp[:, 1] = expl_ids[:, 1]
        past_key_values = cls_gpt.past_key_values
        for index in range(0, self.max_caption_len - 1):
            gpt_out = self.gpt_exp(input_ids=predict_his_exp[:, index].unsqueeze(-1),
                                   encoder_hidden_states=encoder_hs_exp,
                                   encoder_attention_mask=encoder_mask_exp, use_cache=True,
                                   past_key_values=past_key_values)
            past_key_values = gpt_out.past_key_values
            gpt_out = gpt_out[0][:, -1:, :]
            # 只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            predict_his_exp[:, index + 1] = gen_label
            output_exp[:, index] = gen_label

        return output_cap, output_exp, matched, pre

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded


class Oscar_GPT_gen_add(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT_gen_add, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        #因为乘了个矩阵
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.max_gen_len=70

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None,region_id=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states,region_id=region_id)


        encoder_hs=self.bert_gpt_proj(outputs[0])
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hs,encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        shift_logits_cap = shift_logits.mul(
            gpt_martrix_cap[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_cap = shift_labels.mul(gpt_martrix_cap[..., 1:].contiguous())
        shift_logits_cls=shift_logits.mul(gpt_martrix_cls[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_cls=shift_labels.mul(gpt_martrix_cls[..., 1:].contiguous())
        shift_logits_exp = shift_logits.mul(
            gpt_martrix_expl[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_exp = shift_labels.mul(gpt_martrix_expl[..., 1:].contiguous())
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cls.size(-1)),
                                          shift_labels_cap.view(-1))
        gen_cls_loss = self.gen_criterion(shift_logits_cls.view(-1, shift_logits_cls.size(-1)), shift_labels_cls.view(-1))
        gen_exp_loss = self.gen_criterion(shift_logits_exp.view(-1, shift_logits_exp.size(-1)),
                                          shift_labels_exp.view(-1))

        return gen_cls_loss,gen_exp_loss,gen_cap_loss

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None,region_id=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states,region_id=region_id)

        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his = torch.full((expl_ids.size(0),self.max_gen_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        outputs=torch.full((expl_ids.size(0),self.max_gen_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        predict_his[:, 0] =expl_ids[:, 0]
        for index in range(self.max_gen_len-1):
            gpt_out = self.gpt(input_ids=predict_his[:,:index+1], encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask)[0]
            gpt_out=gpt_out[:,-1:,:]
            #只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            for i in range(expl_ids.size(0)):
                predict_his[i, index + 1] = gen_label[i]
                outputs[i, index] = gen_label[i]
        return outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,region_id=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states,region_id=region_id)

        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return outputs

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class Oscar_GPT_gen_add_cls(nn.Module):
    def __init__(self, oscar,gpt,gpt_toker,beam_size):
        super(Oscar_GPT_gen_add_cls, self).__init__()
        self.oscar = oscar
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar.config.hidden_size, self.gpt.config.n_embd)
        self.vocab_num=self.gpt.vocab_size
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        #因为乘了个矩阵
        self.dropout = nn.Dropout(self.oscar.config.hidden_dropout_prob)
        self.num_labels = 3
        self.classifier = nn.Linear(self.oscar.config.hidden_size, self.num_labels)
        self.max_len=50
        self.beam_size=beam_size
        self.max_gen_len=70

    def forward(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None,region_id=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                            position_ids=position_ids, token_type_ids=token_type_ids,
                            head_mask=head_mask,
                            encoder_history_states=encoder_history_states,region_id=region_id)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        encoder_hs=self.bert_gpt_proj(outputs[0])
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hs,encoder_attention_mask=input_mask)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        shift_logits_cap = shift_logits.mul(
            gpt_martrix_cap[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_cap = shift_labels.mul(gpt_martrix_cap[..., 1:].contiguous())
        shift_logits_cls=shift_logits.mul(gpt_martrix_cls[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_cls=shift_labels.mul(gpt_martrix_cls[..., 1:].contiguous())
        shift_logits_exp = shift_logits.mul(
            gpt_martrix_expl[..., 1:].contiguous().unsqueeze(-1).expand(shift_logits.size()).float())
        shift_labels_exp = shift_labels.mul(gpt_martrix_expl[..., 1:].contiguous())
        gen_cap_loss = self.gen_criterion(shift_logits_cap.view(-1, shift_logits_cls.size(-1)),
                                          shift_labels_cap.view(-1))
        gen_cls_loss = self.gen_criterion(shift_logits_cls.view(-1, shift_logits_cls.size(-1)), shift_labels_cls.view(-1))
        gen_exp_loss = self.gen_criterion(shift_logits_exp.view(-1, shift_logits_exp.size(-1)),
                                          shift_labels_exp.view(-1))

        return gen_cls_loss,gen_exp_loss,gen_cap_loss,loss_cls,matched

    def evaluate(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None,region_id=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states,region_id=region_id)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        encoder_hs = self.bert_gpt_proj(outputs[0])
        predict_his = torch.full((expl_ids.size(0),self.max_gen_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        outputs=torch.full((expl_ids.size(0),self.max_gen_len),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        predict_his[:, 0] =expl_ids[:, 0]
        for index in range(self.max_gen_len-1):
            gpt_out = self.gpt(input_ids=predict_his[:,:index+1], encoder_hidden_states=encoder_hs,
                               encoder_attention_mask=input_mask)[0]
            gpt_out=gpt_out[:,-1:,:]
            #只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            gen_label = self.choose_top_word(lm_logits)
            for i in range(expl_ids.size(0)):
                predict_his[i, index + 1] = gen_label[i]
                outputs[i, index] = gen_label[i]
        return outputs,matched

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def pred(self, input_ids, img_feat,expl_ids,input_mask=None,label=None,attn_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,encoder_history_states=None,region_id=None,gpt_martrix_cls=None,gpt_martrix_expl=None,gpt_martrix_cap=None):
        outputs = self.oscar.bert(input_ids, img_feats=img_feat, attention_mask=input_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids,
                                  head_mask=head_mask,
                                  encoder_history_states=encoder_history_states,region_id=region_id)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label
        encoder_hs = self.bert_gpt_proj(outputs[0])
        outputs=self.batch_predict_beam(encoder_hs,input_mask)
        return outputs,matched

    def batch_predict_beam(self,sequence_output,attn_masks):
        batch_size = sequence_output.size(0)
        beam_scores = torch.zeros((batch_size, self.beam_size))  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        beam_scores = beam_scores.view(-1)  # 展开为(batch_size * num_beams)
        done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
        generated_hyps = [
            BeamHypotheses(self.beam_size, self.max_gen_len, length_penalty=0.7)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例
        # 初始输入: （batch_size * num_beams, 1）个sos token
        input_ids = torch.full((batch_size * self.beam_size, 1), self.gpt_toker.bos_token_id, dtype=torch.long).cuda()
        cur_len=1
        sequence_output=sequence_output.unsqueeze(1)
        sequence_output_ex=sequence_output.expand(batch_size, self.beam_size, sequence_output.size(2),sequence_output.size(3))
        sequence_output_ex=sequence_output_ex.reshape(-1,sequence_output.size(2),sequence_output.size(3))
        attn_masks = attn_masks.unsqueeze(1)
        attn_masks_ex = attn_masks.expand(batch_size, self.beam_size, attn_masks.size(2))
        attn_masks_ex = attn_masks_ex.reshape(-1, sequence_output.size(2))
        while cur_len < self.max_gen_len:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=sequence_output_ex,
                                   encoder_attention_mask=attn_masks_ex)
            out = out.last_hidden_state
            out = out[:, -1:, :]
            out = self.lm_head(out)
            out=out.squeeze(1)
            scores = F.log_softmax(out, dim=-1)  # log_softmax
            next_scores = scores + beam_scores[:, None].expand_as(scores).cuda()  # 累加上以前的scores
            next_scores = next_scores.view(
                batch_size, self.beam_size * self.vocab_num
            )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
            # 取topk
            # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
            next_scores, next_tokens = torch.topk(next_scores, self.beam_size, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    next_batch_beam.extend([(0, 0, 0)] * self.beam_size)  # pad the batch
                    continue
                next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // self.vocab_num  # 1
                    token_id = beam_token_id % self.vocab_num  # 1
                    # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                    # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                    # batch_idx=1时，真实beam_id如下式计算为4或5
                    effective_beam_id = batch_idx * self.beam_size + beam_id
                    # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                    if token_id.item() == self.gpt_toker.eos_token_id:
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                        #还是需要继续加入，不然和encoder维度不统一
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    else:
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == self.beam_size:
                        break
                        # 当前batch是否解码完所有句子
                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )  # 注意这里取当前batch的所有log_prob的最大值
                    # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                    # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                next_batch_beam.extend(next_sent_beam)
                # 如果batch中每个句子的beam search都完成了，则停止
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
            # 因为有些beam id对应的句子已经解码完了
            input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
            # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(self.beam_size):
                # 对于每个batch_idx的每句beam，都执行加入add
                # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                effective_beam_id = batch_idx * self.beam_size + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
            # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
            # 下面选择若干最好的序列输出
            # 每个样本返回几个句子
        output_num_return_sequences_per_batch = 1
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        # 记录每个返回句子的长度，用于后面pad
        sent_lengths = input_ids.new(output_batch_size)
        best = []

    # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            # x: (score, hyp), x[0]: score
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, self.max_gen_len)
            # fill pad
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.gpt_toker.pad_token_id)

            # 填充内容
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < self.max_gen_len:
                    decoded[i, sent_lengths[i]] = self.gpt_toker.pad_token_id
        else:
            # 否则直接堆叠起来
            decoded = torch.stack(best).type(torch.long)
            # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
        return decoded

class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

class CrossAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        # self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.c_attn = Conv1D(self.embed_dim*2, self.embed_dim)
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim)


        self.c_proj = Conv1D( self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # if self.scale_attn_weights:
        #     attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):

        query = self.q_attn(hidden_states)
        key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
        attention_mask = encoder_attention_mask

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class Biencoder_gpt(nn.Module):
    def __init__(self, oscar_gen,oscar_cls,gpt,gpt_toker,oscar_toker):
        super(Biencoder_gpt, self).__init__()
        self.oscar_gen = oscar_gen
        self.oscar_cls = oscar_cls
        self.oscar_toker=oscar_toker
        self.gpt_toker=gpt_toker
        self.gpt = gpt
        self.bert_gpt_proj = nn.Linear(self.oscar_gen.config.hidden_size, self.gpt.config.n_embd)
        self.lm_head = nn.Linear(self.gpt.config.n_embd, self.gpt.vocab_size, bias=False)
        self.gen_criterion=torch.nn.CrossEntropyLoss(reduction='mean',ignore_index=gpt_toker.pad_token_id)
        self.bert_gen_criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.dropout = nn.Dropout(self.oscar_gen.config.hidden_dropout_prob)
        self.num_labels = 3
        self.max_seq_len = 40
        self.max_expl_len = 50
        self.classifier = nn.Linear(self.oscar_gen.config.hidden_size, self.num_labels)
        self.ln_cross_attn = BertLayerNorm(self.oscar_gen.config.hidden_size, eps=1e-12)
        self.crossattention = nn.ModuleList([CrossAttention(self.oscar_gen.config, is_cross_attention=True) for _ in
                                             range(self.oscar_gen.config.num_hidden_layers)])
        self.bert=CaptionBertLayer(oscar_cls.config)

    def forward(self, input_ids_cls,input_ids_gen, img_feat,expl_ids,input_mask=None,attention_mask=None,label=None,attn_mask=None,
            token_type_ids_cls=None,token_type_ids_gen=None, position_ids=None, head_mask=None,encoder_history_states=None,attention_mask_cross=None,input_mask_enc=None):
        gen_embedding_output, gen_head_mask, gen_extended_attention_mask = self.oscar_gen.bert.cal_emb(input_ids_gen,
                                                                                           img_feats=img_feat,
                                                                                           attention_mask=attention_mask,
                                                                                           position_ids=position_ids,
                                                                                           token_type_ids=token_type_ids_gen,
                                                                                           head_mask=head_mask,
                                                                                           encoder_history_states=encoder_history_states)
        cls_embedding_output, cls_head_mask, cls_extended_attention_mask = self.oscar_cls.bert.cal_emb(input_ids_cls,
                                                                                           img_feats=img_feat,
                                                                                           attention_mask=input_mask,
                                                                                           position_ids=position_ids,
                                                                                           token_type_ids=token_type_ids_cls,
                                                                                           head_mask=head_mask,
                                                                                           encoder_history_states=encoder_history_states)

        if attention_mask_cross.dim() == 2:
            extended_attention_mask_cross = attention_mask_cross.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask_cross = attention_mask_cross.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attention_mask_cross = extended_attention_mask_cross.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask_cross = (1.0 - extended_attention_mask_cross) * -10000.0

        gen_hidden_states=gen_embedding_output
        cls_hidden_states=cls_embedding_output
        for i in range(self.oscar_gen.config.num_hidden_layers):
            gen_hidden_states=self.oscar_gen.bert.encoder.forward_layer(i,gen_hidden_states,gen_extended_attention_mask)

            cls_hidden_states = self.oscar_cls.bert.encoder.forward_layer(i, cls_hidden_states,
                                                                          cls_extended_attention_mask)
            #cross_attention
            # cls_attention_output=self.oscar_cls.bert.encoder.attention_layer(i,cls_hidden_states,cls_extended_attention_mask)
            # residual = cls_attention_output
            # cls_attention_output = self.ln_cross_attn(cls_attention_output)
            # cross_layer = self.crossattention[i]
            # cross_attn_outputs = cross_layer(hidden_states=cls_attention_output, attention_mask=cls_extended_attention_mask,
            #                                  encoder_hidden_states=gen_hidden_states.detach(),
            #                                  encoder_attention_mask=extended_attention_mask_cross)
            # cls_attention_output = cross_attn_outputs[0]
            # cls_attention_output = residual + cls_attention_output
            # cls_hidden_states = self.oscar_cls.bert.encoder.ffn_layer(i,cls_attention_output)

        #改为在最后的输出cat并过一层bert
        if input_mask_enc.dim() == 2:
            extended_input_mask_enc = input_mask_enc.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_input_mask_enc = input_mask_enc.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_input_mask_enc = extended_input_mask_enc.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_input_mask_enc = (1.0 - extended_input_mask_enc) * -10000.0

        cls_hidden_states=torch.cat((cls_hidden_states,gen_hidden_states[:,:self.max_seq_len].detach()),dim=1)
        cls_hidden_states=self.bert(cls_hidden_states,extended_input_mask_enc)[0]

        pooled_output = self.oscar_cls.bert.pooler(cls_hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss_cls = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
        pre = logits.max(dim=-1)[1]
        matched = pre == label

        #计算bert的caption生成
        cap_label=input_ids_gen[:,:self.max_seq_len]
        label = cap_label[:, 1:]
        sequence_output = gen_hidden_states[:, :self.max_seq_len]
        sequence_output=sequence_output[:,:-1]
        sequence_output = self.oscar_gen.cls(sequence_output)
        loss_bert_gen = self.bert_gen_criterion(sequence_output.reshape(sequence_output.size(0) * sequence_output.size(1), -1),
                                      label.reshape(-1))


        #mask需要拼接
        encoder_hidden_states=self.bert_gpt_proj(cls_hidden_states)
        #注意这里需要input_mask_enc
        gpt_out=self.gpt(input_ids=expl_ids,attention_mask=attn_mask,encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=input_mask_enc)[0]
        lm_logits = self.lm_head(gpt_out)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = expl_ids[..., 1:].contiguous()
        loss_gen = self.gen_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss_gen,loss_cls,loss_bert_gen, matched

    def evaluate(self, input_ids_cls,input_ids_gen, img_feat,expl_ids,input_mask=None,attention_mask=None,label=None,attn_mask=None,
            token_type_ids_cls=None,token_type_ids_gen=None, position_ids=None, head_mask=None,encoder_history_states=None,attention_mask_cross=None,input_mask_enc=None):
        outputs = self.generate_gen(input_ids=input_ids_gen, img_feats=img_feat, attention_mask=attention_mask,
                                  position_ids=position_ids, token_type_ids=token_type_ids_gen,
                                  head_mask=head_mask,num_beams=1)
        input_ids, logprobs, gen_logits=outputs
        input_ids_gen_eval=input_ids_gen.clone()
        input_ids_gen_eval[:,:self.max_seq_len]=input_ids.squeeze(1)
        #用的是生成的caption

        gen_embedding_output, gen_head_mask, gen_extended_attention_mask = self.oscar_gen.bert.cal_emb(input_ids_gen,
                                                                                                       img_feats=img_feat,
                                                                                                       attention_mask=attention_mask,
                                                                                                       position_ids=position_ids,
                                                                                                       token_type_ids=token_type_ids_gen,
                                                                                                       head_mask=head_mask,
                                                                                                       encoder_history_states=encoder_history_states)
        cls_embedding_output, cls_head_mask, cls_extended_attention_mask = self.oscar_cls.bert.cal_emb(input_ids_cls,
                                                                                                       img_feats=img_feat,
                                                                                                       attention_mask=input_mask,
                                                                                                       position_ids=position_ids,
                                                                                                       token_type_ids=token_type_ids_cls,
                                                                                                       head_mask=head_mask,
                                                                                                       encoder_history_states=encoder_history_states)

        if attention_mask_cross.dim() == 2:
            extended_attention_mask_cross = attention_mask_cross.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask_cross = attention_mask_cross.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attention_mask_cross = extended_attention_mask_cross.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask_cross = (1.0 - extended_attention_mask_cross) * -10000.0

        gen_hidden_states = gen_embedding_output
        cls_hidden_states = cls_embedding_output
        for i in range(self.oscar_gen.config.num_hidden_layers):
            gen_hidden_states = self.oscar_gen.bert.encoder.forward_layer(i, gen_hidden_states,
                                                                          gen_extended_attention_mask)

            cls_hidden_states = self.oscar_cls.bert.encoder.forward_layer(i, cls_hidden_states,
                                                                          cls_extended_attention_mask)

            # cls_attention_output = self.oscar_cls.bert.encoder.attention_layer(i, cls_hidden_states,
            #                                                                    cls_extended_attention_mask)
            # residual = cls_attention_output
            # cls_attention_output = self.ln_cross_attn(cls_attention_output)
            # cross_layer = self.crossattention[i]
            # cross_attn_outputs = cross_layer(hidden_states=cls_attention_output,
            #                                  attention_mask=cls_extended_attention_mask,
            #                                  encoder_hidden_states=gen_hidden_states.detach(),
            #                                  encoder_attention_mask=extended_attention_mask_cross)
            # cls_attention_output = cross_attn_outputs[0]
            # cls_attention_output = residual + cls_attention_output
            # cls_hidden_states = self.oscar_cls.bert.encoder.ffn_layer(i, cls_attention_output)

        # 改为在最后的输出cat并过一层bert
        if input_mask_enc.dim() == 2:
            extended_input_mask_enc = input_mask_enc.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_input_mask_enc = input_mask_enc.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_input_mask_enc = extended_input_mask_enc.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_input_mask_enc = (1.0 - extended_input_mask_enc) * -10000.0

        cls_hidden_states = torch.cat((cls_hidden_states, gen_hidden_states[:, :self.max_seq_len].detach()), dim=1)
        cls_hidden_states = self.bert(cls_hidden_states, extended_input_mask_enc)[0]

        pooled_output = self.oscar_cls.bert.pooler(cls_hidden_states)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        pre = logits.max(dim=-1)[1]
        matched = pre == label


        # mask需要拼接
        encoder_hidden_states = self.bert_gpt_proj(cls_hidden_states)
        predict_his = torch.full(expl_ids.size(),fill_value=self.gpt_toker.pad_token_id, dtype=int).cuda()
        outputs=torch.full(expl_ids.size(), fill_value=self.gpt_toker.pad_token_id,dtype=int).cuda()
        predict_his[:, 0] =expl_ids[:, 0]
        for index in range(expl_ids.size(1)-1):
            gpt_out = self.gpt(input_ids=predict_his[:,:index+1], encoder_hidden_states=encoder_hidden_states,
                               encoder_attention_mask=input_mask_enc)[0]
            gpt_out=gpt_out[:,-1:,:]
            #只取最后一个作为当前步的输出
            lm_logits = self.lm_head(gpt_out)
            label = self.choose_top_word(lm_logits)
            for i in range(expl_ids.size(0)):
                predict_his[i, index + 1] = label[i]
                outputs[i, index] = label[i]

        return input_ids.squeeze(1),matched,outputs

    def choose_top_word(self, prob):
        label = np.argmax(prob.cpu().numpy(), axis=2)
        label.resize(prob.size(0))
        label = torch.from_numpy(label)
        return label

    def generate_gen(self, img_feats, attention_mask, img_token_type_ids=None, img_pos_feat=None, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None,
            do_sample=False, num_beams=1, temperature=None, top_k=0, top_p=1,
            repetition_penalty=None,
            eos_token_ids=None,  length_penalty=None,
            num_return_sequences=1,
            num_keep_best=1,
            add_od_labels=True,
            use_cbs=False, fsm=None
            ):
        """ Generates captions given image features
        """
        bos_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
            self.oscar_toker.convert_tokens_to_ids([self.oscar_toker.cls_token, self.oscar_toker.sep_token,
                                                    self.oscar_toker.pad_token, self.oscar_toker.mask_token, '.'])
        eos_token_ids=[sep_token_id]
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.oscar_gen.config.vocab_size
        if not use_cbs:
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b==batch_size and v==vocab_size and f1==num_fsm_states

        self.add_od_labels = add_od_labels
        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = self.max_seq_len
        if self.add_od_labels:
            # get od labels part from input_ids
            assert input_ids.shape[0] == batch_size
            od_label_ids = input_ids[:, self.max_seq_len:]
            self.od_labels_len = input_ids.shape[1] - self.max_seq_len
            input_ids = None
        else:
            self.od_labels_len = 0
            od_label_ids = None
            assert input_ids.shape == (batch_size, self.max_seq_len)
            input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=img_feats.device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        if  num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                        self.od_labels_start_posid,
                        self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        num_expand = num_beams * num_fsm_states * num_return_sequences
        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand)
        self.img_feats = self._expand_for_beams(img_feats, num_expand)
        self.img_token_type_ids=self._expand_for_beams(img_token_type_ids,num_expand)
        self.img_pos_feat=self._expand_for_beams(img_pos_feat,num_expand)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)

        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)


        if num_beams > 1:
            output = self._generate_beam_search_gen(
                input_ids,
                cur_len,
                self.max_seq_len,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search_gen(
                input_ids,
                cur_len,
                self.max_seq_len,
                do_sample,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                pad_token_id,
                eos_token_ids,
                effective_batch_size,
            )

        return output

    def _generate_no_beam_search_gen(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        assert self.num_keep_best == 1, 'cannot generate >1 sentences in greedy search'
        # current position / max lengths / length of generated sentences / unfinished sentences
        unfinished_sents = []
        cur_unfinished = input_ids.new(batch_size).fill_(1)

        # log of scores for each sentence in the batch
        logprobs = []

        past = None


        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation_gen(input_ids, past=past)
            outputs = self.oscar_gen.bert(**model_inputs)

            if cur_len == 1:
                token_len = 1 + self.od_labels_len
                next_token_idx = 0
            else:
                assert cur_len > 1
                if not self._do_output_past(outputs):
                    token_len = cur_len +  self.od_labels_len
                    next_token_idx = cur_len-1
                else:
                    token_len = 1
                    next_token_idx = 0
            # outputs[0]=outputs[0][:,:token_len]
            # assert outputs[0].shape[1] == token_len
            next_token_logits = outputs[0][:, next_token_idx, :]
            next_token_logits = self.oscar_gen.cls(next_token_logits)
            if cur_len == 1:
                gen_logits=next_token_logits.unsqueeze(1)
            else:
                gen_logits =torch.cat([gen_logits, next_token_logits.unsqueeze(1)], dim=1)
            # if model has past, then set the past variable to speed up decoding
            if self._do_output_past(outputs):
                past = outputs[1]

            # # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            # if repetition_penalty != 1.0:
            #     for i in range(batch_size):
            #         for previous_token in set(input_ids[i].tolist()):
            #             # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            #             if next_token_logits[i, previous_token] < 0:
            #                 next_token_logits[i, previous_token] *= repetition_penalty
            #             else:
            #                 next_token_logits[i, previous_token] /= repetition_penalty


                # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

            # Compute scores
            _scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size, vocab_size)
            _scores = torch.gather(_scores, -1, next_token.unsqueeze(-1))  # (batch_size, 1)
            logprobs.append(_scores)  # (batch_size, 1)
            unfinished_sents.append(cur_unfinished)

            # update generations and finished sentences
            tokens_to_add = next_token * cur_unfinished + pad_token_id * (1 - cur_unfinished)
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            #for t in input_ids:
                #print(self.tokenizer.convert_ids_to_tokens(t.tolist()))

            for eos_token_id in eos_token_ids:
                cur_unfinished = cur_unfinished.mul(tokens_to_add.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if cur_unfinished.max() == 0:
                break

        # add eos_token_ids to unfinished sentences
        if cur_len == max_length:
            input_ids[:, -1].masked_fill_(cur_unfinished.to(dtype=torch.bool), eos_token_ids[0])

        logprobs = torch.cat(logprobs, dim=1)
        unfinished_sents = torch.stack(unfinished_sents, dim=1).float()
        sum_logprobs = (logprobs * unfinished_sents).sum(dim=1)
        # return logprobs to keep consistent with beam search output
        logprobs = sum_logprobs / unfinished_sents.sum(dim=1)

        # pad to the same length, otherwise DataParallel will give error
        pad_len = max_length - input_ids.shape[1]
        if pad_len > 0:
            padding_ids = input_ids.new(batch_size, pad_len).fill_(pad_token_id)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)

        # (batch_size, n_best, max_len), (batch_size, n_best)
        return input_ids.unsqueeze(1), logprobs.unsqueeze(1),gen_logits

    def prepare_inputs_for_generation_gen(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        batch_size = curr_ids.shape[0]

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            # input_ids = torch.cat([curr_ids, mask_ids], dim=1)
            input_ids=curr_ids

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                    full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1]-row_end+row_start,
                        t.shape[2]-col_end+col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                    seq_end, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            # input_ids = torch.cat([last_token, mask_ids], dim=1)
            input_ids=last_token
            start_pos = curr_ids.shape[1]-1
            end_pos = start_pos + input_ids.shape[1]

            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            img_token_type_ids=None
            img_pos_feat=None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 1 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                #没有mask_id
                self.prev_encoded_layers = [
                        torch.cat([x[:, 1:, :], x[:, :start_pos,:]], dim=1)
                        for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                        :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                        self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                        :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                        self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                        [torch.cat([i2i, i2s], dim=2),
                        torch.cat([s2i, s2s], dim=2)],
                        dim=1)
            else:
                assert start_pos > 1
                # assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :, :]], dim=1)
                        for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                self.od_labels_len+self.img_seq_len+start_pos: self.od_labels_len+self.img_seq_len+end_pos,
                :self.od_labels_len+self.img_seq_len+end_pos]

        return {'input_ids': input_ids, 'img_feats': img_feats,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'encoder_history_states': self.prev_encoded_layers}

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.oscar_gen.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.oscar_gen.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False