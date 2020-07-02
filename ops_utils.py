#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: operations for running experiments on TPU device 

VERY_LARGE_NEGATIVE_VALUE = -1e12


def exp_mask(logits, mask, mask_is_length=True):
    """Exponential mask for logits.
    Logits cannot be masked with 0 (i.e. multiplying boolean mask)
    because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
    to `False` portion of `mask` so that the portion is effectively ignored
    when exponentiated, e.g. softmaxed.
    Args:
        logits: Arbitrary-rank logits tensor to be masked.
        mask: `boolean` type mask tensor.
            Could be same shape as logits (`mask_is_length=False`)
            or could be length tensor of the logits (`mask_is_length=True`).
    mask_is_length: `bool` value. whether `mask` is boolean mask.
  Returns:
    Masked logits with the same shape of `logits`.
  """
  if mask_is_length:
    mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
  return logits + (1.0 - tf.cast(mask, 'float')) * VERY_LARGE_NEGATIVE_VALUE