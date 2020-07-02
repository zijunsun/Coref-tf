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
import metrics 
import optimization 
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

        input_props = []
        input_props.append((tf.int32, [None, None]))  # input_ids. (batch_size, seq_len)
        input_props.append((tf.int32, [None, None]))  # input_mask (batch_size, seq_len)
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None, None]))  # Speaker IDs.  (batch_size, seq_len)
        input_props.append((tf.int32, []))  # Genre.  能确保整个batch都是同主题，能因为一篇文章的多段放在一个batch里
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts. 一个instance只有一个start?是整篇文章的所有mention的start
        input_props.append((tf.int32, [None]))  # Gold ends. 整篇文章的所有mention的end
        input_props.append((tf.int32, [None]))  # Cluster ids. 整篇文章的所有mention的id
        input_props.append((tf.int32, [None]))  # Sentence Map 整篇文章的每个token属于哪个句子

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)  # 10是batch_size?
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()  # self.queue_input_tensors 不一样？

        if self.config["run"] == "session":
            self.loss, self.pred_start_scores, self.pred_end_scores = self.get_mention_proposal_and_loss(*self.input_tensors)
            tvars = tf.trainable_variables()
            # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
            # Get the assignment map from the tensorflow checkpoint.
            # Depending on the extension, use TF/Pytorch to load weights.
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config['tf_checkpoint'])
            init_from_checkpoint = tf.train.init_from_checkpoint  
            init_from_checkpoint(config['init_checkpoint'], assignment_map)
            print("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
                    print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

            num_train_steps = int(self.config['num_docs'] * self.config['num_epochs'])  # 文章数 * 训练轮数
            num_warmup_steps = int(num_train_steps * 0.1)  # 前1/10做warm_up
            self.global_step = tf.train.get_or_create_global_step()  # 根据不同的model得到不同的optimizer
            self.train_op = optimization.create_custom_optimizer(tvars, self.loss, self.config['bert_learning_rate'],
                                                            self.config['task_learning_rate'],
                                                            num_train_steps, num_warmup_steps, False, self.global_step,
                                                            freeze=-1, task_opt=self.config['task_optimizer'],
                                                            eps=config['adam_eps'])

        # else:
        #    pass 
            # self.loss, self.pred_start_scores, self.pred_end_scores, self.pred_mention_scores = self.get_mention_proposal_and_loss(*self.input_tensors)

        
        self.coref_evaluator = metrics.CorefEvaluator()

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

    def get_mention_proposal_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training,
        gold_starts, gold_ends, cluster_ids, sentence_map, span_mention=None):
        """get mention proposals"""

        start_end_loss_mask = tf.cast(tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=tf.ones_like(input_ids), y=tf.zeros_like(input_ids)), tf.float32) 
        input_ids = tf.where(tf.cast(tf.math.greater_equal(input_ids, tf.zeros_like(input_ids)),tf.bool), x=input_ids, y=tf.zeros_like(input_ids)) 
        input_mask = tf.where(tf.cast(tf.math.greater_equal(input_mask, tf.zeros_like(input_mask)), tf.bool), x=input_mask, y=tf.zeros_like(input_mask)) 
        text_len = tf.where(tf.cast(tf.math.greater_equal(text_len, tf.zeros_like(text_len)), tf.bool), x= text_len, y=tf.zeros_like(text_len)) 
        speaker_ids = tf.where(tf.cast(tf.math.greater_equal(speaker_ids, tf.zeros_like(speaker_ids)),tf.bool), x=speaker_ids, y=tf.zeros_like(speaker_ids)) 
        gold_starts = tf.where(tf.cast(tf.math.greater_equal(gold_starts, tf.zeros_like(gold_starts)),tf.bool), x=gold_starts, y=tf.zeros_like(gold_starts)) 
        gold_ends = tf.where(tf.cast(tf.math.greater_equal(gold_ends, tf.zeros_like(gold_ends)),tf.bool), x=gold_ends, y=tf.zeros_like(gold_ends) ) 
        cluster_ids = tf.where(tf.cast(tf.math.greater_equal(cluster_ids, tf.zeros_like(cluster_ids)),tf.bool), x=cluster_ids, y=tf.zeros_like(cluster_ids)) 
        sentence_map = tf.where(tf.cast(tf.math.greater_equal(sentence_map, tf.zeros_like(sentence_map)),tf.bool), x=sentence_map, y=tf.zeros_like(sentence_map)) 
        span_mention = tf.where(tf.cast(tf.math.greater_equal(span_mention, tf.zeros_like(span_mention)),tf.bool), x=span_mention, y=tf.zeros_like(span_mention)) 
        span_mention_loss_mask = tf.cast(tf.where(tf.cast(tf.math.greater_equal(span_mention, tf.zeros_like(span_mention)),tf.bool), x=tf.ones_like(span_mention), y=tf.zeros_like(span_mention)) , tf.float32)
        # span

        # gold_starts -> [1, 3, 5, 8, -1, -1, -1, -1] -> [1, 3, 5, 8, 0, 0, 0, 0]

        input_ids = tf.reshape(input_ids, [-1, self.config["max_segment_len"]])    # (max_train_sent, max_segment_len) 
        input_mask  = tf.reshape(input_mask, [-1, self.config["max_segment_len"]])   # (max_train_sent, max_segment_len)
        text_len = tf.reshape(text_len, [-1])  # (max_train_sent)
        speaker_ids = tf.reshape(speaker_ids, [-1, self.config["max_segment_len"]])  # (max_train_sent, max_segment_len) 
        sentence_map = tf.reshape(sentence_map, [-1])   # (max_train_sent * max_segment_len) 
        cluster_ids = tf.reshape(cluster_ids, [-1])     # (max_train_sent * max_segment_len) 
        gold_starts = tf.reshape(gold_starts, [-1])     # (max_train_sent * max_segment_len) 
        gold_ends = tf.reshape(gold_ends, [-1])         # (max_train_sent * max_segment_len) 
        span_mention = tf.reshape(span_mention, [self.config["max_training_sentences"], self.config["max_segment_len"] * self.config["max_segment_len"]])
        # span_mention : (max_train_sent, max_segment_len, max_segment_len)

        model = modeling.BertModel(config=self.bert_config, is_training=is_training, input_ids=input_ids,
            input_mask=input_mask, use_one_hot_embeddings=False, scope='bert')

        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        mention_doc = model.get_sequence_output()  # (max_train_sent, max_segment_len, hidden)
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)  # (max_train_sent, max_segment_len, emb) -> (max_train_sent * max_segment_len, e) 取出有效token的emb
        num_words = util.shape(mention_doc, 0)  # max_train_sent * max_segment_len

        seg_mention_doc = tf.reshape(mention_doc, [self.config["max_training_sentences"], self.config["max_segment_len"], -1]) # (max_train_sent, max_segment_len, embed)
        start_seg_mention_doc = tf.stack([seg_mention_doc] * self.config["max_segment_len"], axis=1) # (max_train_sent, 1, max_segment_len, embed) -> (max_train_sent, max_segment_len, max_segment_len, embed)
        end_seg_mention_doc = tf.stack([seg_mention_doc, ] * self.config["max_segment_len"], axis=2) # (max_train_sent, max_segment_len, 1, embed) -> (max_train_sent, max_segment_len, max_segment_len, embed)
        span_mention_doc = tf.concat([start_seg_mention_doc, end_seg_mention_doc], axis=-1) # (max_train_sent, max_segment_len, max_segment_len, embed * 2)
        span_mention_doc = tf.reshape(span_mention_doc, (self.config["max_training_sentences"]*self.config["max_segment_len"]*self.config["max_segment_len"], -1))
        # # (max_train_sent * max_segment_len * max_segment_len, embed * 2)

        with tf.variable_scope("span_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            span_scores = util.ffnn(span_mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"]*2, 1, self.dropout) # (max_train_sent, max_segment_len, 1)
        with tf.variable_scope("start_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            start_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # (max_train_sent, max_segment_len, 1) 
        with tf.variable_scope("end_scores", reuse=tf.AUTO_REUSE):  # [k, 1] 每个候选span的得分
            end_scores = util.ffnn(mention_doc, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # (max_train_sent, max_segment_len, 1)

        gold_start_label = tf.reshape(gold_starts, [-1, 1])  
        # gold_starts -> [1, 3, 5, 8, -1, -1, -1, -1]
        start_value = tf.reshape(tf.ones_like(gold_starts), [-1])
        start_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_start_label = tf.cast(tf.scatter_nd(gold_start_label, start_value, start_shape), tf.int32)
        # gold_start_label = tf.boolean_mask(gold_start_label, tf.reshape(input_mask, [-1]))

        gold_end_label = tf.reshape(gold_ends, [-1, 1])
        end_value = tf.reshape(tf.ones_like(gold_ends), [-1])
        end_shape = tf.constant([self.config["max_training_sentences"] * self.config["max_segment_len"]])
        gold_end_label = tf.cast(tf.scatter_nd(gold_end_label, end_value, end_shape), tf.int32)
        # gold_end_label = tf.boolean_mask(gold_end_label, tf.reshape(input_mask, [-1]))
        start_scores = tf.cast(tf.reshape(tf.sigmoid(start_scores), [-1]),tf.float32)
        end_scores = tf.cast(tf.reshape(tf.sigmoid(end_scores), [-1]),tf.float32)
        span_scores = tf.cast(tf.reshape(tf.sigmoid(span_scores), [-1]), tf.float32)
        # span_mention = tf.cast(span_mention, tf.float32)

        start_scores = tf.stack([(1 - start_scores), start_scores], axis=-1) 
        end_scores = tf.stack([(1 - end_scores), end_scores], axis=-1) 
        span_scores = tf.stack([(1 - span_scores), span_scores], axis=-1)

        gold_start_label = tf.cast(tf.one_hot(tf.reshape(gold_start_label, [-1]), 2, axis=-1), tf.float32)
        gold_end_label = tf.cast(tf.one_hot(tf.reshape(gold_end_label, [-1]), 2, axis=-1), tf.float32)
        span_mention = tf.cast(tf.one_hot(tf.reshape(span_mention, [-1]), 2, axis=-1),tf.float32)

        start_end_loss_mask = tf.reshape(start_end_loss_mask, [-1])
        # true, pred 
        start_loss = tf.keras.losses.binary_crossentropy(gold_start_label, start_scores,)
        end_loss = tf.keras.losses.binary_crossentropy(gold_end_label, end_scores)
        span_loss = tf.keras.losses.binary_crossentropy(span_mention, span_scores,)

        start_loss = tf.reduce_mean(tf.multiply(start_loss, tf.cast(start_end_loss_mask, tf.float32))) 
        end_loss = tf.reduce_mean(tf.multiply(end_loss, tf.cast(start_end_loss_mask, tf.float32))) 
        span_loss = tf.reduce_mean(tf.multiply(span_loss, tf.cast(span_mention_loss_mask, tf.float32))) 


        if span_mention is None :
            loss = self.config["start_ratio"] * start_loss + self.config["end_ratio"] * end_loss 
            return loss, start_scores, end_scores
        else:
            loss = self.config["start_ratio"] * start_loss + self.config["end_ratio"] * end_loss +self.config["mention_ratio"] * span_loss 
            return loss, start_scores, end_scores, span_scores 


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


    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:  # 每个例子是一篇文章，同一篇文章的所有段落一起做session run
                random.shuffle(train_examples)
                if self.config['single_example']:
                    for example in train_examples:
                        tensorized_example = self.tensorize_example(example, is_training=True)
                        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)
                else:  
                    examples = []
                    for example in train_examples:
                        tensorized = self.tensorize_example(example, is_training=True)
                        if type(tensorized) is not list:
                            tensorized = [tensorized]
                        examples += tensorized
                    random.shuffle(examples)
                    print('num examples', len(examples))
                    for example in examples:
                        feed_dict = dict(zip(self.queue_input_tensors, example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()
