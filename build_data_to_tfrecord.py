#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import random

import numpy as np
import tensorflow as tf

import util
from bert.tokenization import FullTokenizer

"""
Desc:
suppose origin input_tokens are :
['[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 
'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 
'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 
'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 
'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 
'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.'] 
IF sliding window size is 50. 

Args:
doc_idx: a string: cctv/bn/0001
sentence_map: 
    e.g. [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7]
subtoken_map: 
    e.g. [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 53, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 97, 98, 99, 99, 99, 100, 101, 102, 103]
flattened_window_input_ids: [num-window, window-size]
    e.g. before bert_tokenizer convert subtokens into ids:
    [['[CLS]', '[unused19]', 'speaker', '#', '1', '[unused73]', '-', '-', 'basically', ',', 'it', 'was', 'unanimously', 'agreed', 'upon', 'by', 'the', 'various', 'relevant', 'parties', '.', 'To', 'express', 'its', 'determination', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', '[SEP]'],
    ['[CLS]', ',', 'the', 'Chinese', 'securities', 'regulatory', 'department', 'compares', 'this', 'stock', 'reform', 'to', 'a', 'die', 'that', 'has', 'been', 'cast', '.', 'It', 'takes', 'time', 'to', 'prove', 'whether', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '[SEP]'],
    ['[CLS]', 'the', 'stock', 'reform', 'can', 'really', 'meet', 'expectations', ',', 'and', 'whether', 'any', 'de', '##viation', '##s', 'that', 'arise', 'during', 'the', 'stock', 'reform', 'can', 'be', 'promptly', 'corrected', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', '[SEP]'],
    ['[CLS]', '.', '[unused19]', 'Xu', '_', 'l', '##i', '[unused73]', 'Dear', 'viewers', ',', 'the', 'China', 'News', 'program', 'will', 'end', 'here', '.', 'This', 'is', 'Xu', 'Li', '.', 'Thank', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '[SEP]'],
    ['[CLS]', 'you', 'everyone', 'for', 'watching', '.', 'Coming', 'up', 'is', 'the', 'Focus', 'Today', 'program', 'hosted', 'by', 'Wang', 'Shi', '##lin', '.', 'Good', '-', 'bye', ',', 'dear', 'viewers', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']] 
flattened_window_masked_ids: [num-window, window-size]
    e.g.: before bert_tokenizer ids:
    [[-3, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, -1, -1, -1, -1, -1, -1, 68, 69, 70, 71, 72, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -3],
    [-3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, -3, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4]]
span_start: 
    e.g.: mention start indices in the original document 
    [17, 20, 26, 43, 60, 85, 86]
span_end:
    e.g.: mention end indices in the original document 
cluster_ids: 
    e.g.: cluster ids for the (span_start, span_end) pairs
    [1, 1, 2, 2, 2, 3, 3] 
span_mention:
    mention of [0, 1] 
check the mention in the subword list: 
    1. ['its']
    1. ['the', 'Chinese', 'securities', 'regulatory', 'department']
    2. ['this', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    2. ['the', 'stock', 'reform']
    3. ['you']
    3. ['everyone']
"""

subtoken_maps = {}
gold = {}


def prepare_training_data(input_data_dir, output_data_dir, input_filename, output_filename, language, config, \
                          vocab_file, sliding_window_size, demo=False):
    generate_fake = True
    if generate_fake:
        output_filename = "fake_small_test"
        dir = os.path.join(output_data_dir, "{}.{}.tfrecord".format(output_filename, language))
        print("output_filename:", dir)

    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    writer = tf.python_io.TFRecordWriter(
        os.path.join(output_data_dir, "{}.{}.tfrecord".format(output_filename, language)))

    data_file_path = os.path.join(input_data_dir, input_filename)
    with open(data_file_path, "r") as f:
        documents = [json.loads(jsonline) for jsonline in f.readlines()]
    doc_map = {}

    if generate_fake:
        # 输入生成的假数据的个数
        samples_num = 2
        output_filename = "fake_small_test"
        none_empty_documents = []
        for doc_idx, document in enumerate(documents):
            if len(document["clusters"]) > 0:
                none_empty_documents.append((doc_idx, len(document["subtoken_map"]), document))
        sorted_documents = sorted(none_empty_documents, key=lambda t: t[1])
        new_documents = [item[2] for item in sorted_documents]
        documents = new_documents[:samples_num]

    for doc_idx, document in enumerate(documents):
        doc_key = document["doc_key"]
        tensorized = tensorize_example(document, config, tokenizer, is_training=True)
        if type(tensorized) is not tuple:
            tensorized = tuple(tensorized)
        write_instance_to_example_file(writer, tensorized, doc_key, config)
        doc_map[doc_idx] = doc_key
        if demo and doc_idx > 5:
            break
    with open(os.path.join(output_data_dir, "{}.{}.map".format(output_filename, language)), 'w') as fo:
        json.dump(doc_map, fo, indent=2)


def write_instance_to_example_file(writer, instance, doc_key, config):
    input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = instance
    input_id_windows = input_ids
    mask_windows = input_mask
    flattened_input_ids = [i for j in input_id_windows for i in j]
    flattened_input_mask = [i for j in mask_windows for i in j]
    cluster_ids = [int(tmp) for tmp in cluster_ids]

    max_sequence_len = int(config["max_training_sentences"])
    max_seg_len = int(config["max_segment_len"])
    before_pad_start = gold_starts
    before_pad_end = gold_ends
    before_text_len = text_len

    sentence_map = clip_or_pad(sentence_map, max_sequence_len * max_seg_len, pad_idx=-1)
    text_len = clip_or_pad(text_len, max_sequence_len, pad_idx=-1)
    tmp_subtoken_maps = clip_or_pad(subtoken_maps[doc_key], max_sequence_len * max_seg_len, pad_idx=-1)

    tmp_speaker_ids = clip_or_pad(speaker_ids[0], max_sequence_len * max_seg_len, pad_idx=-1)

    flattened_input_ids = clip_or_pad(flattened_input_ids, max_sequence_len * max_seg_len, pad_idx=-1)
    flattened_input_mask = clip_or_pad(flattened_input_mask, max_sequence_len * max_seg_len, pad_idx=-1)
    # genre = clip_or_pad(genre, )
    gold_starts = clip_or_pad(gold_starts, config["max_cluster_num"], pad_idx=-1)
    gold_ends = clip_or_pad(gold_ends, config["max_cluster_num"], pad_idx=-1)
    cluster_ids = clip_or_pad(cluster_ids, config["max_cluster_num"], pad_idx=-1)

    span_mention = pad_span_mention(before_text_len, config, before_pad_start, before_pad_end)

    features = {
        'sentence_map': create_int_feature(sentence_map),
        'text_len': create_int_feature(text_len),
        'subtoken_map': create_int_feature(tmp_subtoken_maps),
        'speaker_ids': create_int_feature(tmp_speaker_ids),
        'flattened_input_ids': create_int_feature(flattened_input_ids),
        'flattened_input_mask': create_int_feature(flattened_input_mask),
        'genre': create_int_feature([genre]),
        'span_starts': create_int_feature(gold_starts),
        'span_ends': create_int_feature(gold_ends),
        'cluster_ids': create_int_feature(cluster_ids),
        'span_mention': create_int_feature(span_mention)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def clip_or_pad(var, max_var_len, pad_idx=-1):
    if len(var) >= max_var_len:
        return var[:max_var_len]
    else:
        pad_var = (max_var_len - len(var)) * [pad_idx]
        var = list(var) + list(pad_var)
        return var


def pad_span_mention(text_len_lst, config, before_pad_start, before_pad_end):
    span_mention = np.zeros((config["max_training_sentences"], config["max_segment_len"], config["max_segment_len"]),
                            dtype=int)

    for idx, (tmp_s, tmp_e) in enumerate(zip(before_pad_start, before_pad_end)):
        start_seg = int(tmp_s // config["max_segment_len"])
        end_seg = int(tmp_s // config["max_segment_len"])
        if start_seg != end_seg:
            continue
        try:
            sent_idx = int(tmp_s // config["max_segment_len"]) + 1 if tmp_s % config["max_segment_len"] != 0 else int(
                tmp_s // config["max_segment_len"])
            start_offset = tmp_s % config["max_segment_len"]
            end_offset = tmp_e % config["max_segment_len"]
            span_mention[sent_idx, start_offset, end_offset] = 1
        except:
            continue

    flatten_span_mention = np.reshape(span_mention, (1, -1))
    flatten_span_mention = flatten_span_mention.tolist()
    flatten_span_mention = [j for j in flatten_span_mention]

    return flatten_span_mention[0]


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def tensorize_example(example, config, tokenizer, is_training):
    """
    把一篇文章的所有原始特征信息，分片转成tensor
    :param example:
    :param is_training:
    :return:
    """
    clusters = example["clusters"]
    genres = {g: i for i, g in enumerate(config["genres"])}

    gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}  # 每个mention的token span对应的mention id
    cluster_ids = np.zeros(len(gold_mentions))  # 每个mention_id对应的cluster_id
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]  # 多少个滑动窗口
    ### num_words = sum(len(s) for s in sentences)
    speakers = example["speakers"]
    speaker_dict = get_speaker_dict(util.flatten(speakers), config)
    sentence_map = example['sentence_map']  # 每个token_id对应的sentence_id，标出句子的边界，防止出现跨句的candidate_span
    max_sentence_length = config["max_segment_len"]
    text_len = np.array([len(s) for s in sentences])  # 滑动窗口每个滑块的长度

    input_ids, input_mask, speaker_ids = [], [], []
    for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
        sent_input_ids = tokenizer.convert_tokens_to_ids(sentence)
        sent_input_mask = [1] * len(sent_input_ids)
        sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
        while len(sent_input_ids) < max_sentence_length:
            sent_input_ids.append(0)
            sent_input_mask.append(0)
            sent_speaker_ids.append(0)
        input_ids.append(sent_input_ids)
        speaker_ids.append(sent_speaker_ids)
        input_mask.append(sent_input_mask)
    input_ids = np.array(input_ids)  # 多个滑动窗口，每个滑动窗口有一个list的tokens
    input_mask = np.array(input_mask)
    speaker_ids = np.array(speaker_ids)

    doc_key = example["doc_key"]  # mention span是以sub-token为基准的，但评测时是以token为基准的
    subtoken_maps[doc_key] = example.get("subtoken_map", None)  # sub-token对回原来是第几个单词
    gold[doc_key] = example["clusters"]  #
    genre = genres.get(doc_key[:2], 0)

    gold_starts, gold_ends = tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                       cluster_ids, sentence_map)

    if len(sentences) > config["max_training_sentences"]:
        # if config['single_example']:
        # return truncate_example(config, *example_tensors)
        # else:
        offsets = range(config['max_training_sentences'], len(sentences), config['max_training_sentences'])
        tensor_list = [truncate_example(config, *(example_tensors + (offset,))) for offset in offsets]
        #         input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = tensor_list[0]
        # xiaoya : to do tensor_list[0], tensor_list[i]
        input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = \
            tensor_list[0]

        return tensor_list[0]
    else:
        return example_tensors


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []
    return np.array(starts), np.array(ends)


def truncate_example(config, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                     cluster_ids, sentence_map, sentence_offset=None):
    """因为显存装不下，训练时对于一篇文章的所有128的doc_span，随机选其中连续的n个做训练，mentions也做相应的截取"""

    max_training_sentences = config["max_training_sentences"]
    num_sentences = input_ids.shape[0]
    assert num_sentences > max_training_sentences

    sentence_offset = random.randint(0,
                                     num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
    word_offset = text_len[:sentence_offset].sum()
    num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
    input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
    speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    sentence_map = sentence_map[word_offset: word_offset + num_words]
    gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map


def get_speaker_dict(speakers, config):
    speaker_dict = {'UNK': 0, '[SPL]': 1}
    for s in speakers:
        if s not in speaker_dict and len(speaker_dict) < config['max_num_speakers']:
            speaker_dict[s] = len(speaker_dict)
    return speaker_dict


if __name__ == '__main__':
    # python3 build_data_to_tfrecord.py
    data_sign = "test"
    for sliding_window_size in [128]:
        print("=*=" * 20)
        print("current sliding window size is : {}".format(str(sliding_window_size)))
        print("=*=" * 20)
        for data_sign in ["train", "dev", "test"]:
            print("%*%" * 20)
            print(data_sign)
            print("%*%" * 20)
            config = util.initialize_from_env(use_tpu=False)
            language = "english"
            vocab_file = "/xiaoya/pretrain_ckpt/spanbert_base_cased/vocab.txt"
            input_data_dir = "/xiaoya/data"

            input_filename = "{}.english.{}.jsonlines".format(data_sign, str(sliding_window_size))

            output_data_dir = "/xiaoya/data/zijun/mention_proposal_data/span_all_{}_{}".format(str(sliding_window_size),
                                                                                               str(
                                                                                                   config[
                                                                                                       "max_training_sentences"]))
            os.makedirs(output_data_dir, exist_ok=True)
            output_filename = "{}.english.jsonlines".format(data_sign)
            print("$^$" * 30)
            print(output_data_dir, output_filename)
            print("$^$" * 30)
            prepare_training_data(input_data_dir, output_data_dir, input_filename, output_filename, language, config,
                                  vocab_file, sliding_window_size)

    # prepare demo dataset
    # config = util.initialize_from_env(use_tpu=False)
    # vocab_file = "/xiaoya/pretrain_ckpt/spanbert_base_cased/vocab.txt"
    # input_data_dir = "/xiaoya/data"
    # language = "english"
    # input_filename = "{}.english.128.jsonlines".format(data_sign)
    # sliding_window_size = 128
    # output_data_dir = "/xiaoya/tpu_data/mention_proposal/test_span"
    # output_data_dir = "/xiaoya/tpu_data/mention_proposal/demo_128_{}".format(str(config["max_training_sentences"]))
    # os.makedirs(output_data_dir, exist_ok=True)
    # output_filename = "{}.english.jsonlines".format(data_sign)

    # prepare_training_data(input_data_dir, output_data_dir, input_filename, output_filename, language, config, vocab_file, \
    #     sliding_window_size, demo=True)
