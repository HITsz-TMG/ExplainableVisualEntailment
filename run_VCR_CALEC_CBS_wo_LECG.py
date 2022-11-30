# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import sys

sys.path.append('../')
import argparse
import os
import os.path as op
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import random, time, json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from utils.misc import (mkdir)
from modeling.modeling_transfomres import BertImgModel, GPT2Model
from modeling.modeling_vcr_chunkalign_v10 import ChunkAlign_CLS_dec5_4_beam, SeqBertImgModel

from transformers import BertTokenizerFast, BertConfig

from Data.VCRChunkAlign import VCR_ChunkAlign_prefix_Dataset
from progressbar import ProgressBar
from transformers import GPT2Tokenizer, GPT2Config
import xlwt

from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.spice.spice import Spice


def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True, batch_size=opts.per_gpu_train_batch_size * opts.num_gpus,
                                num_workers=0,
                                shuffle=True, collate_fn=dataset.SNLIGPT_gen_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate)
    return dataloader


def test(args, test_dataloader, model, tokenizer):
    time_meter = 0
    result_dict = {}
    n_examples = 0
    n_correct_qa_0 = 0

    score_1_exp_all = 0.0
    meteor_exp = 0.0
    rouge_exp = 0.0
    spice_exp = 0.0

    # model = model.module if hasattr(model, 'module') else model
    model.eval()
    cider_exp = 0.0
    scorers = [
        (Bleu(1), "Bleu_1"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    pbar = ProgressBar(n_total=len(test_dataloader), desc='testing')
    gts_exp = {}
    res_exp = {}
    index = 0
    b_rtnl = model.dec_toker.encode("<|b_rtnl|>")[0]
    e_rtnl = model.dec_toker.encode("<|e_rtnl|>")[0]

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'input_ids': batch['input_ids'], 'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'expl_ids': batch['expl_ids'], 'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask']}
            outputs, matched_0, pres = model.test_beam(**inputs)
            n_correct_qa_0 += matched_0.sum().item()
            img_keys = batch['img_id']
            img_keys = [img_keys[i:i + 4] for i in range(0, len(img_keys), 4)]
            n_examples += pres.size(0)
            ques = batch['ques_str']
            ques = [ques[i:i + 4] for i in range(0, len(ques), 4)]

            ans = batch['ans_str']
            ans = [ans[i:i + 4] for i in range(0, len(ans), 4)]

            label = batch['label'].reshape(-1, 4)
            label = torch.argmax(label, -1)
            golden_expls = inputs['expl_ids'].reshape(matched_0.size(0), 4, -1)[:, 0, :].cpu()
            for img_key, que, an, la, oscar_match, pre, gen_expl, golden_expl in zip(img_keys, ques, ans, label,
                                                                                     matched_0, pres, outputs,
                                                                                     golden_expls):
                tmp_res = []
                golden_expl = golden_expl.tolist()
                b_rtnl_index = golden_expl.index(b_rtnl)
                e_rtnl_index = golden_expl.index(e_rtnl)
                golden_expl_str = model.dec_toker.decode(golden_expl[b_rtnl_index + 1:e_rtnl_index],
                                                         skip_special_tokens=False).replace("<|det", "").replace("|>",
                                                                                                                 "")

                if oscar_match.item() == True:
                    gen_expl = gen_expl.tolist()
                    try:
                        e_rtnl_index = gen_expl.index(e_rtnl)
                        gen_expl_str = model.dec_toker.decode(gen_expl[:e_rtnl_index],
                                                              skip_special_tokens=False).replace("<|det", "").replace(
                            "|>", "")
                    except:
                        gen_expl_str = model.dec_toker.decode(gen_expl, skip_special_tokens=False).replace("<|det",
                                                                                                           "").replace(
                            "|>", "")
                    gts_exp[index] = [golden_expl_str]
                    res_exp[index] = [gen_expl_str]
                    index += 1
                    tmp_res.append(
                        {'que': que[0], 'ans_list': an, 'oscar_match': str(oscar_match.item()),
                         'golden_rel': int(la), 'pre': int(pre), 'golden_expl': golden_expl_str,
                         'output_expl': gen_expl_str})
                else:
                    tmp_res.append(
                        {'que': que[0], 'ans_list': an, 'oscar_match': str(oscar_match.item()),
                         'golden_rel': int(la), 'pre': int(pre), 'golden_expl': golden_expl_str,
                         'output_expl': ''})

                if img_key[0] in result_dict.keys():
                    result_dict[img_key[0]].append(tmp_res[0])
                else:
                    result_dict[img_key[0]] = tmp_res
            pbar(step=step,
                 info={'acc_0': n_correct_qa_0 / n_examples})
        with open('./Oscar/oscar/output/VCR_dec5_4/reault_str', 'w') as json_file:
            json_file.write(json.dumps(result_dict))

        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts_exp, res_exp)
            if method == 'Bleu_1':
                score_1_exp_all += score[0]
            elif method == 'METEOR':
                meteor_exp += score
            elif method == 'ROUGE_L':
                rouge_exp += score
            elif method == 'SPICE':
                spice_exp += score
            elif method == 'CIDEr':
                cider_exp += score
        pbar(step=step,
             info={'acc_0': n_correct_qa_0 / n_examples, 'score_1_exp_all': score_1_exp_all,
                   'meteor_exp': meteor_exp,
                   'rouge_exp': rouge_exp, 'spice_exp': spice_exp, 'cider_exp': cider_exp})
        result_str = json.dumps(result_dict)
        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))
    return n_correct_qa_0 / n_examples, result_str


def eval_nlp_scores(pred, gt, verbose=False):
    """
    evaluates the nlp scores bleu1-bleu4, meteor, rouge-l, cider, spice
    Args:
        pred (List): List of predictions
        gt (List): List of ground truths
    """
    if len(pred) == len(gt) == 0:
        return {}

    gts = {}
    res = {}
    # 原文不做tokenzie
    for imgId in range(len(pred)):
        gts[imgId] = gt[imgId]
        res[imgId] = pred[imgId]

    # Set up scorers
    if verbose:
        print("Setting up scorers...")
    results = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE_L"]),
        (Cider(), ["CIDEr"]),
        (Spice(), ["SPICE"]),  # NOTE: SPICE is VERY slow
    ]
    # Compute scores
    for scorer, method in scorers:
        if verbose:
            print("Computing %s score..." % (scorer.method()))

        # NOTE: may crash when run with very little training
        corpus_score, sentence_scores = scorer.compute_score(gts, res)

        # iterate (for bleu)
        for ind in range(len(method)):
            cs, ss = corpus_score, sentence_scores
            if isinstance(corpus_score, list):
                cs, ss = corpus_score[ind], sentence_scores[ind]

            results[method[ind]] = cs, ss

            if verbose:
                print("%s: %0.3f" % (method[ind], cs))

    return results


def restore_training_settings(args):
    checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logger.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
            max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
                       'max_img_seq_length']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                                                                              test_v, train_v))
                setattr(args, param, train_v)
    return args


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def ensure_init_process_group(local_rank=None, port=12345):
    # init with env
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    if world_size > 1 and not dist.is_initialized():
        assert local_rank is not None
        print("Init distributed training on local rank {}".format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl', init_method='env://'
        )
    return local_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcr_example_file_test",
                        default="./Oscar/datasets/VCR_UNITER_feat/test_example_data.pkl",
                        type=str)
    parser.add_argument("--vcr_feat_file_test",
                        default="./Oscar/datasets/VCR_UNITER_feat/test_image_data_vilvl.pkl",
                        type=str)
    parser.add_argument("--vcr_chunk_mask_test",
                        default="./Oscar/datasets/VCR_UNITER_feat/ChunkMaskTest_v4.pkl",
                        type=str)
    parser.add_argument("--num_gpus", default=1, type=int, help="Workers in dataloader.")
    parser.add_argument("--train_yaml", default='train.yaml', type=str, required=False,
                        help="yaml file for training.")
    parser.add_argument("--test_yaml", default='test.yaml', type=str, required=False,
                        help="yaml file for testing.")
    parser.add_argument("--val_yaml", default='val.yaml', type=str, required=False,
                        help="yaml file used for validation during training.")
    parser.add_argument("--gpt_model_name_or_path", default='./GPT2', type=str,
                        required=False,
                        help="Path to GPT model.")
    parser.add_argument("--model_name_or_path",
                        default='./Oscar/pretrained_models/image_captioning/pretrained_base/checkpoint-2000000/',
                        type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--seq_model_name_or_path",
                        default='./Oscar/pretrained_models/image_captioning/pretrained_base/checkpoint-2000000/',
                        type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--output_dir", default='./output/VCR_dec5_4', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--loss_type", default='sfmx', type=str,
                        help="Loss function types: support kl, x2, sfmx")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=140, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded.")
    parser.add_argument("--max_hypo_len", default=50, type=int,
                        help="The maximum sequence length for hypothesis.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_local_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--wo_gate", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mask_prob", default=0.0, type=float,
                        help="Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--add_od_labels", default=False, action='store_true',
                        help="Whether to add object detection labels or not")
    parser.add_argument("--drop_out", default=0.3, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=150, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int,
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--tie_weights", default=False, action='store_true',
                        help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--freeze_embedding", default=False, action='store_true',
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                        help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                        help=".")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear or")
    parser.add_argument("--num_workers", default=0, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=100, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    # for self-critical sequence training
    parser.add_argument('--scst', action='store_true', help='Self-critical sequence training')
    parser.add_argument('--sc_train_sample_n', type=int, default=2,
                        help="number of sampled captions for sc training")
    parser.add_argument('--sc_baseline_type', type=str, default='greedy',
                        help="baseline tyep of REINFORCE algorithm")
    parser.add_argument('--beam_size', type=int, default=5,
                        help="beam size for scst training")
    parser.add_argument('--cider_cached_tokens', type=str, default='coco-train-words.p',
                        help="path to cached cPickle file used to calculate CIDEr scores")
    # for generation
    parser.add_argument("--eval_model_dir", type=str,
                        default='./Oscar/oscar/output/VCR_dec5_4/checkpoint-4-56000-score0.1400-nlg_score-0.1919-acc-0.7293-blue-0.0525-meteor-0.1254-rouge-0.2369-cider_exp-0.4067-spice-0.1380-/',
                        help="Model directory for evaluation.")
    parser.add_argument('--max_gen_length', type=int, default=40,
                        help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', action='store_true',
                        help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1, help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                        help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                        help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                        help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                        help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                        help="repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)")
    # for Constrained Beam Search
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument("--epoch_begin", default=14, type=int)
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation begin")
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")
    parser.add_argument(
        "--repeat_penalty", default=1.0, type=float,
        help="")
    parser.add_argument(
        "--length_penalty", default=1.0, type=float,
        help="")
    parser.add_argument(
        "--constrained", default=1.0, type=float,
        help="")
    parser.add_argument(
        "--stop_words_index", default=5, type=int,
        help="")
    args = parser.parse_args()

    global logger
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    # args.num_gpus = get_world_size()
    args.distributed = False
    args.device = torch.device('cuda')
    # args.device = torch.device('cpu')
    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)

    args = restore_training_settings(args)

    gpt_config_class, gpt_model_class, gpt_tokenizer_class = GPT2Config, GPT2Model, GPT2Tokenizer
    gpt_tokenizer = gpt_tokenizer_class.from_pretrained(args.gpt_model_name_or_path, bos_token='[CLS]',
                                                        eos_token='[SEP]', pad_token='[PAD]')
    det_tokens = ["<|det%d|>" % i for i in range(45)]
    gpt_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|b_qn|>", "<|e_qn|>", "<|b_ans|>", "<|e_ans|>", "<|b_rtnl|>",
                                       "<|e_rtnl|>"] + det_tokens})
    gpt_config = gpt_config_class.from_pretrained(args.gpt_model_name_or_path)
    gpt_model = gpt_model_class.from_pretrained(args.gpt_model_name_or_path, config=gpt_config)
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))

    assert args.model_name_or_path is not None
    config_class, model_class, tokenizer_class = BertConfig, BertImgModel, BertTokenizerFast
    config = config_class.from_pretrained(args.config_name if args.config_name else \
                                              args.model_name_or_path, num_labels=args.num_labels,
                                          finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                                                    else args.model_name_or_path, do_lower_case=args.do_lower_case)

    tokenizer.add_special_tokens({"additional_special_tokens": det_tokens})

    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    config.output_attentions = True
    oscar_model = model_class.from_pretrained(args.model_name_or_path,
                                              from_tf=False, config=config)

    oscar_model.resize_token_embeddings(len(tokenizer))

    seq_config_class, seq_model_class = BertConfig, SeqBertImgModel
    seq_config = seq_config_class.from_pretrained(args.seq_model_name_or_path, num_labels=args.num_labels,
                                                  finetuning_task='image_captioning')

    seq_config.img_feature_dim = args.img_feature_dim
    seq_config.img_feature_type = args.img_feature_type
    seq_config.hidden_dropout_prob = args.drop_out
    seq_config.loss_type = args.loss_type
    seq_config.tie_weights = args.tie_weights
    seq_config.freeze_embedding = args.freeze_embedding
    seq_config.label_smoothing = args.label_smoothing
    seq_config.drop_worst_ratio = args.drop_worst_ratio
    seq_config.drop_worst_after = args.drop_worst_after
    seq_config.max_hypo = args.max_hypo_len
    seq_config.output_attentions = True
    seq_config.add_residual = args.add_residual
    seq_config.add_local_residual = args.add_local_residual
    seq_model = seq_model_class.from_pretrained(args.seq_model_name_or_path,
                                                from_tf=False, config=seq_config)
    seq_model.resize_token_embeddings(len(tokenizer))
    model = ChunkAlign_CLS_dec5_4_beam(oscar_model, seq_model, gpt_model, tokenizer, gpt_tokenizer, num_labels=4,
                                       repeat_penalty=args.repeat_penalty,
                                       length_penalty=args.length_penalty, constrained=args.constrained)

    model_file = os.path.join(args.eval_model_dir, 'model.pth')
    model.load_state_dict(torch.load(model_file))

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    test_dataset = VCR_ChunkAlign_prefix_Dataset(tokenizer, gpt_tokenizer, args.vcr_example_file_test,
                                                 args.vcr_chunk_mask_test,
                                                 args.vcr_feat_file_test)
    test_dataloader = build_dataloader(test_dataset, False, args)
    acc, result_str = test(args, test_dataloader, model, tokenizer)

    result_dic = json.loads(result_str)

    pred = []
    gt = []
    for imgId in result_dic.keys():
        for i in range(len(result_dic[imgId])):
            if result_dic[imgId][i]['oscar_match'] == 'True':
                pred.append(result_dic[imgId][i]['output_expl'])
                gt.append(result_dic[imgId][i]['golden_expl'])

    file_name = 'gen-rep-{}-len-{}-con-{}-stop-{}.txt'.format(args.repeat_penalty, args.length_penalty,
                                                              args.constrained, args.stop_words_index)
    with open(os.path.join(args.eval_model_dir, file_name), 'w') as f:
        for line in pred:
            f.write(line)
            f.write('\n')

    file_name = 'gt-rep-{}-len-{}-con-{}-stop-{}.txt'.format(args.repeat_penalty, args.length_penalty, args.constrained,
                                                             args.stop_words_index)
    with open(os.path.join(args.eval_model_dir, file_name), 'w') as f:
        for line in gt:
            f.write(str(line))
            f.write('\n')

    pred = []
    gt = []
    for imgId in result_dic.keys():
        for i in range(len(result_dic[imgId])):
            if result_dic[imgId][i]['oscar_match'] == 'True':
                pred.append([result_dic[imgId][i]['output_expl']])
                gt.append([result_dic[imgId][i]['golden_expl']])

    score = eval_nlp_scores(pred, gt)

    print('no-tokenize')
    for key, item in score.items():
        print('{}:{:.4f}'.format(key, item[0]))

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('SNLI')
    index = 0
    sheet.write(index, 0, 'B1')
    sheet.write(index, 1, 'B2')
    sheet.write(index, 2, 'B3')
    sheet.write(index, 3, 'B4')
    sheet.write(index, 4, 'R-L')
    sheet.write(index, 5, 'MET')
    sheet.write(index, 6, 'CIDEr')
    sheet.write(index, 7, 'SPICE')

    sheet.write(1, 0, round(score['Bleu_1'][0] * 100, 2))
    sheet.write(1, 1, round(score['Bleu_2'][0] * 100, 2))
    sheet.write(1, 2, round(score['Bleu_3'][0] * 100, 2))
    sheet.write(1, 3, round(score['Bleu_4'][0] * 100, 2))
    sheet.write(1, 4, round(score['ROUGE_L'][0] * 100, 2))
    sheet.write(1, 5, round(score['METEOR'][0] * 100, 2))
    sheet.write(1, 6, round(score['CIDEr'][0] * 100, 2))
    sheet.write(1, 7, round(score['SPICE'][0] * 100, 2))
    file_name = 'NLG_metrix_rep-{}-len-{}-con-{}-stop-{}.xls'.format(args.repeat_penalty, args.length_penalty,
                                                                     args.constrained, args.stop_words_index)
    workbook.save(os.path.join(args.eval_model_dir, file_name))
    nlg_score = (score['Bleu_4'][0] + score['ROUGE_L'][0] + score['METEOR'][0] + score['CIDEr'][0] + score['SPICE'][
        0]) * 100
    file_name = 'VCR-acc-{:.4f}-nlg_score-{:.4f}-bs-{}-rep-{}-len-{}-con-{}-stop-{}.json'.format(acc, nlg_score,
                                                                                                 args.beam_size,
                                                                                                 args.repeat_penalty,
                                                                                                 args.length_penalty,
                                                                                                 args.constrained,
                                                                                                 args.stop_words_index)
    with open(os.path.join(args.eval_model_dir, file_name), 'w') as json_file:
        json_file.write(result_str)

    with open(os.path.join(args.eval_model_dir, 'total_result.json'), 'a') as json_file:
        info = 'rep-' + str(args.repeat_penalty) + '-len-' + str(args.length_penalty) + '-con-' + str(
            args.constrained) + '-stop-' + str(args.stop_words_index) + '\n'
        score_str = 'Bleu_4-' + str(score['Bleu_4'][0] * 100) + '-ROUGE_L-' + str(
            score['ROUGE_L'][0] * 100) + '-METEOR-' + str(score['METEOR'][0] * 100) + '-CIDEr-' + str(
            score['CIDEr'][0] * 100) + '-SPICE-' + str(score['SPICE'][0] * 100) + '\n'
        total_str = 'total-' + str(nlg_score) + '\n'
        json_file.write(info)
        json_file.write(score_str)
        json_file.write(total_str)
        json_file.write('\n')

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('SNLI_GPT_wo_cap')
    index = 0
    sheet.write(index, 0, 'key')
    sheet.write(index, 1, 'que')
    sheet.write(index, 2, 'a')
    sheet.write(index, 3, 'b')
    sheet.write(index, 4, 'c')
    sheet.write(index, 5, 'd')
    sheet.write(index, 6, 'label')
    sheet.write(index, 7, 'pre')
    sheet.write(index, 8, 'match')
    for key in result_dic.keys():
        for j in range(len(result_dic[key])):
            index += 1
            hypo = result_dic[key][j]['que']
            a = result_dic[key][j]['ans_list'][0]
            b = result_dic[key][j]['ans_list'][1]
            c = result_dic[key][j]['ans_list'][2]
            d = result_dic[key][j]['ans_list'][3]
            golden_rel = result_dic[key][j]['golden_rel']
            pre = result_dic[key][j]['pre']
            match = result_dic[key][j]['oscar_match']
            sheet.write(index, 0, key)
            sheet.write(index, 1, hypo)
            sheet.write(index, 2, a)
            sheet.write(index, 3, b)
            sheet.write(index, 4, c)
            sheet.write(index, 5, d)
            sheet.write(index, 6, golden_rel)
            sheet.write(index, 7, pre)
            sheet.write(index, 8, match)
    file_name = 'VCR-acc-{:.4f}-nlg_score-{:.4f}-bs-{}-rep-{}-len-{}-con-{}-stop-{}.xls'.format(acc, nlg_score,
                                                                                                args.beam_size,
                                                                                                args.repeat_penalty,
                                                                                                args.length_penalty,
                                                                                                args.constrained,
                                                                                                args.stop_words_index)
    workbook.save(os.path.join(args.eval_model_dir, file_name))
    print(args.eval_model_dir)
    print('写入完成')


if __name__ == "__main__":
    main()
