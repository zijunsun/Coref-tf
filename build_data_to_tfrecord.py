#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



import json
import os
import random 
import numpy as np 
import tensorflow as tf
import util 
from bert.tokenization import FullTokenizer

subtoken_maps = {}
gold = {}


def prepare_training_data(data_dir, language, filename, config, vocab_file, sliding_window_size):
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    # for dataset in ['train', 'dev', 'test']:
    writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, "{}.{}.tfrecord".format(filename, language)))

    data_file_path = os.path.join(data_dir, filename)
    with open(data_file_path, "r") as f:
        documents = [json.loads(jsonline) for jsonline in f.readlines()]
    doc_map = {}
    for doc_idx, document in enumerate(documents):
        doc_key = document["doc_key"]
        tensorized = tensorize_example(document, config, tokenizer, is_training=True)
        if type(tensorized) is not tuple:
            tensorized = tuple(tensorized)
            # input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map
            # 
        write_instance_to_example_file(writer, tensorized, doc_key)
    with open(os.path.join(data_dir, "{}.{}.map".format(filename, language)), 'w') as fo:
        json.dump(doc_map, fo, indent=2)



def write_instance_to_example_file(writer, instance, doc_key):
    # input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map
    input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map = instance 
    # doc_idx, sentence_map, subtoken_map, input_id_windows, mask_windows, span_starts, span_ends, cluster_ids = instance
    input_id_windows = input_ids 
    mask_windows = input_mask 
    flattened_input_ids = [i for j in input_id_windows for i in j]
    flattened_input_mask = [i for j in mask_windows for i in j]
    cluster_ids = [int(tmp) for tmp in cluster_ids]
    # print(cluster_ids )
    # print("check cluster ids ")
    # exit()
    # input_ids, input_mask,   genre,  
    # print(subtoken_maps)
    # print("check subtoken map")
    # exit()
    features = {
        'sentence_map': create_int_feature(sentence_map), # 
        'text_len': create_int_feature(text_len), # 
        'subtoken_map': create_int_feature(subtoken_maps[doc_key]),  # 
        'speaker_ids': create_int_feature(speaker_ids[0]), # 
        'flattened_input_ids': create_int_feature(flattened_input_ids),
        'flattened_input_mask': create_int_feature(flattened_input_mask),
        'genre': create_int_feature([genre]),
        'span_starts': create_int_feature(gold_starts), # 
        'span_ends': create_int_feature(gold_ends), # 
        'cluster_ids': create_int_feature(cluster_ids), # 
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())



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

    sentences = example["sentences"] # 多少个滑动窗口
    ### num_words = sum(len(s) for s in sentences)
    speakers = example["speakers"]
    speaker_dict = get_speaker_dict(util.flatten(speakers), config)
    sentence_map = example['sentence_map'] # 每个token_id对应的sentence_id，标出句子的边界，防止出现跨句的candidate_span
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
    input_ids = np.array(input_ids)   # 多个滑动窗口，每个滑动窗口有一个list的tokens
    input_mask = np.array(input_mask)
    speaker_ids = np.array(speaker_ids)

    doc_key = example["doc_key"] # mention span是以sub-token为基准的，但评测时是以token为基准的
    subtoken_maps[doc_key] = example.get("subtoken_map", None)   # sub-token对回原来是第几个单词
    gold[doc_key] = example["clusters"]  #
    genre = genres.get(doc_key[:2], 0)

    gold_starts, gold_ends = tensorize_mentions(gold_mentions)
    example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                           cluster_ids, sentence_map)

    if  len(sentences) > config["max_training_sentences"]:
        # if config['single_example']:
        return truncate_example(config, *example_tensors)
        # else:
        #     offsets = range(config['max_training_sentences'], len(sentences), config['max_training_sentences'])
        #     tensor_list = [truncate_example(config, *(example_tensors + (offset,))) for offset in offsets]
        #     return tensor_list
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

    sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
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
    # python3 build_data_to_tfrecord.py train_spanbert_base
    config = util.initialize_from_env()
    data_dir = "/xiaoya/data"
    language = "english"
    vocab_file = "/xiaoya/pretrain_ckpt/spanbert_base_cased/vocab.txt"
    filename = "test.english.128.jsonlines"
    # filename = "dev.english.128.jsonlines"
    sliding_window_size = 128
    prepare_training_data(data_dir, language, filename, config, vocab_file, sliding_window_size)





