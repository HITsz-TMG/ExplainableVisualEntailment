# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
import sys

sys.path.append('../')
import argparse
import pickle
import os
import os.path as op

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from utils.logger import setup_logger
from utils.misc import (mkdir)
from modeling.modeling_transfomres import GPT2Model, BertImgModel
from modeling.modeling_sequence_align_encoder1 import ChunkAlign_CLS_enc4_wo_reasoning, SeqBertImgModel
from transformers import BertTokenizerFast, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from Data.SNLISeqAlign import SNLISeqAlignChunkDataset_v7
from progressbar import ProgressBar
from transformers import GPT2Tokenizer, GPT2Config
import xlwt

label2id = {0: 'neutral',
            1: 'contradiction',
            2: 'entailment'}


def build_dataloader(dataset, is_train, opts):
    if is_train:
        dataloader = DataLoader(dataset, drop_last=True, batch_size=opts.per_gpu_train_batch_size * opts.num_gpus,
                                num_workers=0,
                                shuffle=True, collate_fn=dataset.SNLIGPT_gen_collate)
    else:
        dataloader = DataLoader(dataset, batch_size=opts.per_gpu_eval_batch_size,
                                num_workers=0, shuffle=False, collate_fn=dataset.SNLIGPT_gen_collate)
    return dataloader


def save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-latest')
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)
            # model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir


def save_checkpoint(model, tokenizer, args, epoch, iteration, optimizer, scheduler, num_trial=10, acc=0.0,
                    res_str=None):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}-acc-{:.4f}'.format(epoch, iteration, acc))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            checkpoint_dir_model = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model_to_save.state_dict(), checkpoint_dir_model)

            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_dir_op = os.path.join(checkpoint_dir, "optimizer.pth")
            torch.save(optimizer.state_dict(), checkpoint_dir_op)
            checkpoint_dir_sc = os.path.join(checkpoint_dir, "scheduler.pth")
            torch.save(scheduler.state_dict(), checkpoint_dir_sc)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    if res_str is not None:
        file_name = 'SNLI-acc-{:.4f}.json'.format(acc)
        with open(os.path.join(checkpoint_dir, file_name), 'w') as json_file:
            json_file.write(res_str)
        print('写入完成')
    return checkpoint_dir


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    scores = logits == labels
    return scores


def train(args, train_dataloader, val_dataset, model, tokenizer):
    # model = nn.DataParallel(model)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                                                   args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and scheduler
    # seq align
    global_enc = ['global_enc']
    seq_align = ['seq_enc']
    grouped_parameters = [
        # 推理层
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in seq_align + global_enc)],
         'lr': args.learning_rate},
        # decoder端未预训练
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in global_enc)],
         'lr': args.learning_rate},
        # seqalign端
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in seq_align)],
         'lr': args.learning_rate * 0.1},

    ]
    global_step = args.global_step
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    if global_step > 0:
        model_file = os.path.join(args.eval_model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_file))
        optimizer.load_state_dict(torch.load(op.join(args.eval_model_dir, 'optimizer.pth')))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()  # an optimizer.cuda() method for this operation would be nice
        scheduler.load_state_dict(torch.load(op.join(args.eval_model_dir, 'scheduler.pth')))
        logger.info("  Resume from %s", args.eval_model_dir)

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                args.per_gpu_train_batch_size * args.num_gpus * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Total examples = %d", len(train_dataloader) * args.per_gpu_train_batch_size * args.num_gpus)

    n_correct_qa_0 = 0

    global_loss = 0.0
    global_cls_loss_0 = 0.0
    model.zero_grad()
    model.train()
    n_examples = 0
    new_step = 0
    pbar_len = len(train_dataloader) // args.gradient_accumulation_steps
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader) // args.gradient_accumulation_steps, desc='training')
        for step, batch in enumerate(train_dataloader):

            inputs = {'input_ids': batch['input_ids'], 'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'expl_ids': batch['expl_ids'], 'attn_mask': batch['attn_mask'],
                      'syn_labels_ids': batch['syn_labels_ids'], 'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask']}
            loss_cls_0, matched_0 = model(**inputs)

            n_correct_qa_0 += matched_0.sum().item()
            loss = loss_cls_0
            n_examples += args.per_gpu_train_batch_size * args.num_gpus
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss_cls_0 = loss_cls_0 / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            global_loss += loss.item()
            global_cls_loss_0 += loss_cls_0.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                new_step += 1

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                pbar(step=new_step % pbar_len,
                     info={'Epoch': epoch, 'oscar_match_0': n_correct_qa_0 / n_examples, 'loss': global_loss / new_step,
                           'cls_loss': global_cls_loss_0 / new_step})
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                                               optimizer.param_groups[
                                                                                                   0]["lr"], loss,
                                                                                               global_loss / global_step)
                    )

                if (args.save_steps > 0 and global_step % args.save_steps == 0) or \
                        global_step == t_total:
                    checkpoint_dir = save_latest_checkpoint(model, tokenizer, args, optimizer, scheduler)
                    # evaluation
                if global_step % args.valid_steps == 0 and epoch >= args.epoch_begin:
                    logger.info("Perform evaluation at step: %d" % (global_step))
                    acc, result_str = eval(args, val_dataset, model, tokenizer)
                    model.train()
                    checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, global_step, optimizer, scheduler,
                                                     acc=acc, res_str=result_str)
    return checkpoint_dir


def eval(args, test_dataloader, model, tokenizer):
    time_meter = 0
    result_dict = {}
    n_examples = 0
    n_correct_qa_0 = 0

    model.eval()

    pbar = ProgressBar(n_total=len(test_dataloader), desc='testing')

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'input_ids': batch['input_ids'], 'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'expl_ids': batch['expl_ids'], 'attn_mask': batch['attn_mask'],
                      'syn_labels_ids': batch['syn_labels_ids'], 'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask']}
            matched_0, pres = model.evaluate(**inputs)
            n_correct_qa_0 += matched_0.sum().item()
            img_keys = batch['img_id']
            n_examples += img_keys.size(0)
            hypos = inputs['input_ids'][:, :args.max_gen_length].cpu()

            for img_key, hypo, la, oscar_match, pre in zip(img_keys, hypos, batch['label'], matched_0, pres):
                tmp_res = []
                hypo_str = tokenizer.decode(hypo.tolist(), skip_special_tokens=True)
                tmp_res.append(
                    {'hypo': hypo_str, 'oscar_match': str(oscar_match.item()),
                     'golden_rel': label2id[int(la)], 'pre': label2id[int(pre)]})
                if int(img_key) in result_dict.keys():
                    result_dict[int(img_key)].append(tmp_res[0])
                else:
                    result_dict[int(img_key)] = tmp_res
            pbar(step=step,
                 info={'acc_0': n_correct_qa_0 / n_examples})
        pbar(step=step,
             info={'acc_0': n_correct_qa_0 / n_examples})
        result_str = json.dumps(result_dict)
        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))
        return n_correct_qa_0 / n_examples, result_str


def test(args, test_dataloader, model, tokenizer):
    time_meter = 0
    result_dict = {}
    n_examples = 0
    n_correct_qa_0 = 0

    model.eval()

    pbar = ProgressBar(n_total=len(test_dataloader), desc='testing')

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'input_ids': batch['input_ids'], 'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask']}
            matched_0, pres = model.evaluate(**inputs)
            n_correct_qa_0 += matched_0.sum().item()
            img_keys = batch['img_id']
            n_examples += img_keys.size(0)
            hypos = inputs['input_ids'][:, :args.max_gen_length].cpu()

            for img_key, hypo, la, oscar_match, pre in zip(img_keys, hypos, batch['label'], matched_0, pres):
                tmp_res = []
                hypo_str = tokenizer.decode(hypo.tolist(), skip_special_tokens=True)
                tmp_res.append(
                    {'hypo': hypo_str, 'oscar_match': str(oscar_match.item()),
                     'golden_rel': label2id[int(la)], 'pre': label2id[int(pre)]})
                if int(img_key) in result_dict.keys():
                    result_dict[int(img_key)].append(tmp_res[0])
                else:
                    result_dict[int(img_key)] = tmp_res
            pbar(step=step,
                 info={'acc_0': n_correct_qa_0 / n_examples})
        pbar(step=step,
             info={'acc_0': n_correct_qa_0 / n_examples})
        result_str = json.dumps(result_dict)
        logger.info("Inference model computing time: {} seconds per batch".format(time_meter / (step + 1)))
        return n_correct_qa_0 / n_examples, result_str


def save_heat(args, test_dataloader, model, tokenizer):
    model.eval()

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            inputs = {'input_ids': batch['input_ids'], 'token_type_ids': batch['token_type_ids'],
                      'input_mask': batch['input_mask'], 'img_feat': batch['img_feat'], 'label': batch['label'],
                      'gather_index': batch['gather_index'],
                      'offsets': batch['offsets'], 'chunk_attention_mask': batch['chunk_attention_mask']}
            matched_0, pres, attn = model.save_heat(**inputs)
        return attn


def restore_training_settings(args):
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:

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
    parser.add_argument("--SNLI_example_file_train",
                        default="./e-ViL-main/train_example_data.pkl", type=str)
    parser.add_argument("--SNLI_example_file_dev",
                        default="./e-ViL-main/val_example_data_csv.pkl", type=str)
    parser.add_argument("--SNLI_example_file_test",
                        default="./e-ViL-main/test_example_data_csv.pkl",
                        type=str)
    parser.add_argument("--flickr_feat_file_train",
                        default="./e-ViL-main/train_image_data.pkl", type=str)
    parser.add_argument("--flickr_feat_file_dev",
                        default="./e-ViL-main/val_image_data_csv.pkl", type=str)
    parser.add_argument("--flickr_feat_file_test",
                        default="./e-ViL-main/test_image_data_csv.pkl",
                        type=str)
    parser.add_argument("--SNLI_chunk_mask_train",
                        default="./e-ViL-main/ChunkMaskTrain_v4.pkl", type=str)
    parser.add_argument("--SNLI_chunk_mask_dev",
                        default="./e-ViL-main/ChunkMaskDev_v4.pkl", type=str)
    parser.add_argument("--SNLI_chunk_mask_test",
                        default="./e-ViL-main/ChunkMaskTest_v4.pkl",
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
    parser.add_argument("--output_dir", default='./output/SNLI_ChunkAlign_CLS_enc4_wo_reasoning', type=str, required=False,
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
    parser.add_argument("--max_hypo_len", default=40, type=int,
                        help="The maximum sequence length for hypothesis.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--add_local_residual", action='store_true', help="Whether to run evaluation.")
    parser.add_argument("--load_DecP", action='store_true', help="Whether to load pretrained decoder.")

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
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before backward.")
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
    parser.add_argument('--save_steps', type=int, default=500,
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
    parser.add_argument("--eval_model_dir", type=str, default='',
                        help="Model directory for evaluation.")
    parser.add_argument("--seq_pretrain_model_dir", type=str,
                        default='./Oscar/oscar/output/SeqAlign_pretrain_v2_align_only/checkpoint-13-12800-acc-0.7960-/',
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
    parser.add_argument('--length_penalty', type=int, default=1,
                        help="beam search length penalty")
    # for Constrained Beam Search
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--use_cbs', action='store_true',
                        help='Use constrained beam search for decoding')
    parser.add_argument('--min_constraints_to_satisfy', type=int, default=2,
                        help="minimum number of constraints to satisfy")
    parser.add_argument("--epoch_begin", default=2, type=int)
    parser.add_argument("--valid_steps", default=500, type=int,
                        help="Run validation begin")
    parser.add_argument(
        "--global_step", default=0, type=int,
        help="")
    parser.add_argument(
        "--example_index", default=None, type=int,
        help="")
    args = parser.parse_args()

    global logger

    # Setup CUDA, GPU & distributed training
    local_rank = ensure_init_process_group(local_rank=args.local_rank)
    args.local_rank = local_rank
    # args.num_gpus = get_world_size()
    args.distributed = False
    args.device = torch.device('cuda')

    synchronize()

    output_dir = args.output_dir
    mkdir(output_dir)

    logger = setup_logger("vlpretrain", output_dir, args.local_rank)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.num_gpus)
    # set_seed(args.seed, args.num_gpus)
    args = restore_training_settings(args)

    # Load pretrained model and tokenizer

    gpt_config_class, gpt_model_class, gpt_tokenizer_class = GPT2Config, GPT2Model, GPT2Tokenizer
    gpt_tokenizer = gpt_tokenizer_class.from_pretrained(args.gpt_model_name_or_path, bos_token='[CLS]',
                                                        eos_token='[SEP]', pad_token='[PAD]')

    assert args.model_name_or_path is not None
    config_class, model_class, tokenizer_class = BertConfig, BertImgModel, BertTokenizerFast
    config = config_class.from_pretrained(args.config_name if args.config_name else \
                                              args.model_name_or_path, num_labels=args.num_labels,
                                          finetuning_task='image_captioning')

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                                                    else args.model_name_or_path, do_lower_case=args.do_lower_case)

    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = args.img_feature_type
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = args.loss_type
    config.tie_weights = args.tie_weights
    config.freeze_embedding = args.freeze_embedding
    config.label_smoothing = args.label_smoothing
    config.drop_worst_ratio = args.drop_worst_ratio
    config.drop_worst_after = args.drop_worst_after
    config.max_hypo = args.max_hypo_len
    config.output_attentions = True
    oscar_model = model_class.from_pretrained(args.model_name_or_path,
                                              from_tf=False, config=config)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<self>"]})
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
    # 加载预训练seqAlign
    model_file = os.path.join(args.seq_pretrain_model_dir, 'model.pth')
    pretrained_dict = torch.load(model_file)
    renamed_dict = {}
    for k, v in pretrained_dict.items():
        if 'seq_enc' in k:
            k = '.'.join(k.split('.')[1:])
            renamed_dict[k] = v
    seq_model.load_state_dict(renamed_dict)
    logger.info("load pretrained ChunkAlign from %s", args.seq_pretrain_model_dir)
    model = ChunkAlign_CLS_enc4_wo_reasoning(oscar_model, seq_model)

    if args.do_test:
        model_file = os.path.join(args.eval_model_dir, 'model.pth')
        model.load_state_dict(torch.load(model_file))

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = SNLISeqAlignChunkDataset_v7(tokenizer, gpt_tokenizer, args.SNLI_example_file_train,
                                                    args.flickr_feat_file_train,
                                                    args.SNLI_chunk_mask_train,
                                                    max_hypo_len=args.max_hypo_len, max_seq_length=args.max_seq_length)
        train_dataloader = build_dataloader(train_dataset, True, args)
        val_dataset = SNLISeqAlignChunkDataset_v7(tokenizer, gpt_tokenizer, args.SNLI_example_file_dev,
                                                  args.flickr_feat_file_dev,
                                                  args.SNLI_chunk_mask_dev,
                                                  max_hypo_len=args.max_hypo_len, max_seq_length=args.max_seq_length)
        val_dataloader = build_dataloader(val_dataset,
                                          False, args)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)


    # inference and evaluation
    elif args.do_test and args.example_index is None:
        test_dataset = SNLISeqAlignChunkDataset_v7(tokenizer, gpt_tokenizer, args.SNLI_example_file_test,
                                                   args.flickr_feat_file_test,
                                                   args.SNLI_chunk_mask_test,
                                                   max_hypo_len=args.max_hypo_len, max_seq_length=args.max_seq_length)
        test_dataloader = build_dataloader(test_dataset,
                                           False, args)

        acc, result_str = test(args, test_dataloader, model, tokenizer)

        result_dic = json.loads(result_str)

        file_name = 'SNLI-acc-{:.4f}.json'.format(acc)
        with open(os.path.join(args.eval_model_dir, file_name), 'w') as json_file:
            json_file.write(result_str)

        workbook = xlwt.Workbook()
        sheet = workbook.add_sheet('SNLI_GPT_wo_cap')
        index = 0
        sheet.write(index, 0, 'key')
        sheet.write(index, 1, 'hypo')
        sheet.write(index, 2, 'golden_rel')
        sheet.write(index, 3, 'pre')
        sheet.write(index, 4, 'match')
        for key in result_dic.keys():
            for j in range(len(result_dic[key])):
                index += 1
                hypo = result_dic[key][j]['hypo']
                golden_rel = result_dic[key][j]['golden_rel']
                pre = result_dic[key][j]['pre']
                match = result_dic[key][j]['oscar_match']
                sheet.write(index, 0, key)
                sheet.write(index, 1, hypo)
                sheet.write(index, 2, golden_rel)
                sheet.write(index, 3, pre)
                sheet.write(index, 4, match)
        file_name = 'SNLI-acc-{:.4f}.xls'.format(acc)
        workbook.save(os.path.join(args.eval_model_dir, file_name))
        print(args.eval_model_dir)
        print('写入完成')
    else:
        # 保存热力图
        test_dataset = SNLISeqAlignChunkDataset_v7(tokenizer, gpt_tokenizer, args.SNLI_example_file_test,
                                                   args.flickr_feat_file_test,
                                                   args.SNLI_chunk_mask_test,
                                                   max_hypo_len=args.max_hypo_len, max_seq_length=args.max_seq_length,
                                                   heat_index=args.example_index)
        test_dataloader = build_dataloader(test_dataset,
                                           False, args)
        attn = save_heat(args, test_dataloader, model, tokenizer)
        pickle.dump(attn, open(os.path.join(args.eval_model_dir, str(args.example_index) + '.pkl'), 'wb'))
        print(args.eval_model_dir)
        print('写入完成')


if __name__ == "__main__":
    main()
