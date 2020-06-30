#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import tensorflow as tf
import util
from radam import RAdam
from input_builder import file_based_input_fn_builder


format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


tf.app.flags.DEFINE_string('f', '', 'kernel')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", "data", "The output directory of the model training.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("slide_window_size", 156, "size of sliding window.")
flags.DEFINE_integer("max_seq_length", 200, "Max sequence length for the input sequence.")
flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name "
                       "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("tpu_zone", None, "[Optional] GCE zone where the Cloud TPU is located in. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not "
                       "specified, we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 1, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
#FLAGS = tf.flags.FLAGS


def model_fn_builder(config):
    # init_checkpoint = config.init_checkpoint

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        config = util.initialize_from_env()

        tmp_features = {}
        input_ids = tf.boolean_mask(features["flattened_input_ids"], tf.math.greater_equal(features["flattened_input_ids"], tf.zeros_like(features["flattened_input_ids"]))) 
        input_mask = tf.boolean_mask(features["flattened_input_mask"], tf.math.greater_equal(features["flattened_input_mask"], tf.zeros_like(features["flattened_input_mask"]))) 
        text_len = tf.boolean_mask(features["text_len"], tf.math.greater_equal(features["text_len"], tf.zeros_like(features["text_len"]))) 

        speaker_ids = tf.boolean_mask(features["speaker_ids"], tf.math.greater_equal(features["speaker_ids"], tf.zeros_like(features["speaker_ids"]))) 
        genre = features["genre"] 
        gold_starts = tf.boolean_mask(features["span_starts"], tf.math.greater_equal(features["span_starts"], tf.zeros_like(features["span_starts"]))) 
        gold_ends = tf.boolean_mask(features["span_ends"], tf.math.greater_equal(features["span_ends"], tf.zeros_like(features["span_ends"]))) 
        cluster_ids = tf.boolean_mask(features["cluster_ids"], tf.math.greater_equal(features["cluster_ids"], tf.zeros_like(features["cluster_ids"]))) 
        sentence_map = tf.boolean_mask(features["sentence_map"], tf.math.greater_equal(features["sentence_map"], tf.zeros_like(features["sentence_map"]))) 

        input_ids = tf.reshape(input_ids, [-1, config["max_segment_len"]])
        input_mask  = tf.reshape(input_mask, [-1, config["max_segment_len"]])
        text_len = tf.reshape(text_len, [-1])
        speaker_ids = tf.reshape(features["speaker_ids"], [-1, config["max_segment_len"]])
        sentence_map = tf.reshape(sentence_map, [-1])
        cluster_ids = tf.reshape(cluster_ids, [-1]) 
        gold_starts = tf.reshape(gold_starts, [-1]) 
        gold_ends = tf.reshape(gold_ends, [-1]) 
        
        # input_ids =  tf.reshape(features["flattened_input_ids"], [-1, config["max_segment_len"]])
        # input_mask  = tf.reshape(features["flattened_input_mask"], [-1, config["max_segment_len"]])
        # text_len = tf.reshape(features["text_len"], [-1])
        # speaker_ids = tf.reshape(features["speaker_ids"], [-1, config["max_segment_len"]])
        # genre = features["genre"]
        # gold_starts = tf.reshape(features["span_starts"], [-1])
        # gold_ends = tf.reshape(features["span_ends"], [-1])
        # cluster_ids = tf.reshape(features["cluster_ids"], [-1])
        # sentence_map = tf.reshape(features["sentence_map"], [-1, config["max_segment_len"]])
        # sentence_map = tf.reshape(features["sentence_map"], [-1, config["max_segment_len"]])
        tmp_features["input_ids"] = input_ids 
        tmp_features["input_mask"] = input_mask 
        tmp_features["text_len"] = text_len 
        tmp_features["speaker_ids"] = speaker_ids
        tmp_features["genre"] = genre
        tmp_features["gold_starts"] = gold_starts
        tmp_features["gold_ends"] =  gold_ends
        tmp_features["speaker_ids"] = speaker_ids
        tmp_features["cluster_ids"] = cluster_ids
        tmp_features["sentence_map"] = sentence_map


        tf.logging.info("*** Features ***")
        for name in sorted(tmp_features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, tmp_features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="mention_proposal")
  
        if FLAGS.use_tpu:
            def tpu_scaffold():
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            scaffold_fn = None 
           
        # if mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("**** Trainable Variables ****")
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['bert_learning_rate'])
        total_loss, pred_mention_labels, gold_mention_labels = model.get_mention_proposal_and_loss(input_ids, input_mask, \
                text_len, speaker_ids, genre, is_training, gold_starts,
                gold_ends, cluster_ids, sentence_map)

        if config["tpu"]:
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=config['bert_learning_rate'])
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            # predictions, total_loss = model.get_predictions_and_loss(input_ids, input_mask, \
            #     text_len, speaker_ids, genre, is_training, gold_starts,
            #     gold_ends, cluster_ids, sentence_map)
            train_op = optimizer.minimize(total_loss, tf.train.get_global_step()) 
            # train_op = model.train_op 
            # prediction, total_loss = model.get_predictions_and_loss() 
        else:
            optimizer = RAdam(learning_rate=config['bert_learning_rate'], epsilon=1e-8, beta1=0.9, beta2=0.999)
            train_op = optimizer.minimize(total_loss, tf.train.get_global_step())

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        # else:
            # is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            # def metric_fn(loss):
            #    summary_dict, average_f1 =  model.evaluate(features, is_training)
            #     return summary_dict, average_f1
            # eval_metrics = (metric_fn, [total_loss])
            # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            #     mode=mode,
            #     loss=total_loss,
            #     eval_metrics=eval_metrics,
            #     scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def main(_):
    config = util.initialize_from_env()

    tf.logging.set_verbosity(tf.logging.INFO)


    num_train_steps = config["num_docs"] * config["num_epochs"]
    # use_tpu = FLAGS.use_tpu
    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=config["save_checkpoints_steps"],
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(config)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=1,
        predict_batch_size=1)

    seq_length = config["max_segment_len"] * config["max_training_sentences"]

    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(config["train_path"], seq_length, config, is_training=True, drop_remainder=True), max_steps=20000)


if __name__ == '__main__':
    # main()
    tf.app.run()






