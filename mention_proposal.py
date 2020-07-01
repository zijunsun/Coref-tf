#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import json
import os
import sys 
import random
import threading

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


import numpy as np
import tensorflow as tf

import util
from bert import modeling
from bert import tokenization



class MentionProposalModel(object):
    def __init__(self, config):
        self.config = config
        self.max_segment_len = config['max_segment_len']
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.dropout = None
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.bert_config.hidden_dropout_prob = self.config["dropout_rate"]
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config['vocab_file'], do_lower_case=False)
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()

    def restore(self, session):
        """在evaluate和predict阶段加载整个模型"""
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables()]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)


    def get_dropout(self, dropout_rate, is_training):  # is_training为True时keep=1-drop, 为False时keep=1
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def get_mention_proposal_and_loss(
        self, input_ids, input_mask, text_len, speaker_ids, genre, is_training,
        gold_starts, gold_ends, cluster_ids, sentence_map, span_mention
    ):
        """get mention proposals"""

        input_ids = tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=input_ids, y=tf.zeros_like(input_ids)) 
        input_mask = tf.where(tf.cast(tf.math.greater_equal(input_mask, tf.zeros_like(input_mask)), tf.bool), x=input_mask, y=tf.zeros_like(input_mask)) 
        text_len = tf.where(tf.cast(tf.math.greater_equal(text_len, tf.zeros_like(text_len)), tf.bool), x= text_len, y=tf.zeros_like(text_len)) 
        speaker_ids = tf.where(tf.cast(tf.math.greater_equal(speaker_ids, tf.zeros_like(speaker_ids)),tf.bool), x=speaker_ids, y=tf.zeros_like(speaker_ids)) 
        gold_starts = tf.where(tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts)),tf.bool), x=gold_starts, y=tf.zeros_like(gold_starts)) 
        gold_ends = tf.where(tf.cast(tf.math.greater_equal(gold_ends, tf.zeros_like(gold_ends)),tf.bool), x=gold_ends, y=tf.zeros_like(gold_ends) ) 
        cluster_ids = tf.where(tf.cast(tf.math.greater_equal(cluster_ids, tf.zeros_like(cluster_ids)),tf.bool), x=cluster_ids, y=tf.zeros_like(cluster_ids)) 
        sentence_map = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 


        input_ids = tf.reshape(input_ids, [-1, self.config["max_segment_len"]])
        input_mask  = tf.reshape(input_mask, [-1, self.config["max_segment_len"]])
        text_len = tf.reshape(text_len, [-1])
        speaker_ids = tf.reshape(speaker_ids, [-1, self.config["max_segment_len"]])
        sentence_map = tf.reshape(sentence_map, [-1])
        cluster_ids = tf.reshape(cluster_ids, [-1]) 
        gold_starts = tf.reshape(gold_starts, [-1]) 
        gold_ends = tf.reshape(gold_ends, [-1]) 
        span_mention = tf.reshape(span_mention, [self.config["max_training_sentences"] * self.config["max_segment_len"], self.config["max_training_sentences"] * self.config["max_segment_len"]])


        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        mention_doc = model.get_sequence_output()  # (batch_size, seq_len, hidden)
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)  # (b, s, e) -> (b*s, e) 取出有效token的emb
        num_words = util.shape(mention_doc, 0)  # b*s

        with tf.variable_scope("start_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            start_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)
        with tf.variable_scope("end_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            end_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)

        gold_start_label = tf.reshape(gold_starts, [-1, 1])
        start_value = tf.reshape(tf.ones_like(gold_starts), [-1])
        start_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_start_label = tf.cast(tf.scatter_nd(gold_start_label, start_value, start_shape),tf.float64) 
        # gold_start_label = tf.boolean_mask(gold_start_label, tf.reshape(input_mask, [-1]))

        gold_end_label = tf.reshape(gold_ends, [-1, 1])
        end_value = tf.reshape(tf.ones_like(gold_ends), [-1])
        end_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_end_label = tf.cast(tf.scatter_nd(gold_end_label, end_value, end_shape), tf.float64)
        # gold_end_label = tf.boolean_mask(gold_end_label, tf.reshape(input_mask, [-1]))
        start_scores = tf.cast(tf.reshape(tf.sigmoid(start_scores), [-1]),tf.float64)
        end_scores = tf.cast(tf.reshape(tf.sigmoid(end_scores), [-1]),tf.float64)
            
        # loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.cast(tf.reshape(tf.sigmoid(start_scores), [-1]),tf.float32), labels=tf.reshape(gold_start_label, [-1])))
        # loss +=  tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.cast(tf.reshape(tf.sigmoid(end_scores), [-1]),tf.float32), labels=tf.reshape(gold_end_label, [-1])) )

        loss = self.bce_loss(y_pred=start_scores, y_true=tf.cast(tf.reshape(gold_start_label, [-1]), tf.float64))
        loss += self.bce_loss(y_pred=end_scores, y_true=tf.cast(tf.reshape(gold_end_label, [-1]), tf.float64))

        return loss, start_scores, end_scores


    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return flattened_emb 
        ##### return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


    def load_eval_data(self):  # lazy加载，真正需要evaluate的时候加载，加载一次常驻内存
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            # num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data) 所有token数
            print("Loaded {} eval examples.".format(len(self.eval_data)))


    def evaluate_mention_proposal(self, session, official_stdout=False, eval_mode=False):
        self.load_eval_data()
        summary_dict = {}
        tp = 0
        fp = 0
        fn = 0
        epsilon = 1e-10
        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            pred_labels, gold_labels = session.run([self.pred_mention_labels, self.gold_mention_labels],
                                                   feed_dict=feed_dict)

            tp += np.logical_and(pred_labels, gold_labels).sum()
            fp += np.logical_and(pred_labels, np.logical_not(gold_labels)).sum()
            fn += np.logical_and(np.logical_not(pred_labels), gold_labels).sum()

            if (example_num + 1) % 100 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        p = tp / (tp+fp+epsilon)
        r = tp / (tp+fn+epsilon)
        f = 2*p*r/(p+r+epsilon)
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(self.eval_data)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util.make_summary(summary_dict), f


    def mention_proposal_loss(self, mention_span_score, gold_mention_span):
        """
        Desc:
            caluate mention score
        """
        mention_span_score = tf.reshape(mention_span_score, [-1])
        gold_mention_span = tf.reshape(gold_mention_span, [-1])
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=gold_mention_span, logits=mention_span_score) 
