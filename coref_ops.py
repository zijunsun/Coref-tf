from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os 
import sys 

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-1])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)


import tensorflow as tf

coref_op_library = tf.load_op_library(os.path.join(repo_path, "coref_kernels.so"))

extract_spans = coref_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")
