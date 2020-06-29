#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
import tensorflow as tf
import util
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
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")
#FLAGS = tf.flags.FLAGS


def model_fn_builder(config):
    # init_checkpoint = config.init_checkpoint

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        config = util.initialize_from_env()

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = util.get_model(config, model_sign="mention_proposal")
  
        if FLAGS.use_tpu:
            def tpu_scaffold():
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            scaffold_fn = None 
           
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("**** Trainable Variables ****")

            train_op = model.train_op 
            total_loss = model.loss 
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        else:
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            def metric_fn(loss):
                summary_dict, average_f1 =  model.evaluate(features, is_training)
                return summary_dict, average_f1
            eval_metrics = (metric_fn, [total_loss])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def main():
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

    if FLAGS.do_train:
        estimator.train(input_fn=file_based_input_fn_builder(config["train_path"] ,config, is_training=True, drop_remainder=True), max_steps=20000)


if __name__ == '__main__':
    main()
    # tf.app.run()






