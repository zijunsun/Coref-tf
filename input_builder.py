import tensorflow as tf


def file_based_input_fn_builder(input_file, seq_length, config, is_training, drop_remainder, preprocess=True):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn_from_tfrecord(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat() 
            d = d.shuffle(buffer_size=100)

        name_to_features = {
            'sentence_map': tf.FixedLenFeature([seq_length], tf.int64),
            'text_len': tf.FixedLenFeature([seq_length], tf.int64),
            'subtoken_map': tf.FixedLenFeature([seq_length], tf.int64),
            'speaker_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'flattened_input_ids': tf.FixedLenFeature([seq_length], tf.int64),
            'flattened_input_mask': tf.FixedLenFeature([seq_length], tf.int64),
            'genre': tf.FixedLenFeature([seq_length], tf.int64),
            'span_starts': tf.FixedLenFeature([seq_length], tf.int64),
            'span_ends': tf.FixedLenFeature([seq_length], tf.int64), 
            'cluster_ids': tf.FixedLenFeature([seq_length], tf.int64) }

        def _decode_features(record):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        d = d.map(_decode_features)
        d = d.apply(tf.contrib.data.batch_and_drop_remainder(
            batch_size))

        return d

    return input_fn_from_tfrecord 