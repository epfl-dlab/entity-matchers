import math
import random

import numpy as np
import tensorflow as tf
from sklearn import preprocessing


def init_embeddings(shape, name, init, is_l2_norm, dtype=tf.float32):
    embeds = None
    if init == 'xavier':
        embeds = xavier_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'normal':
        # BootEA uses this initialization, is_l2_norm = True, shape = (num_entities/num_relation, 100)
        embeds = truncated_normal_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'uniform':
        embeds = random_uniform_init(shape, name, is_l2_norm, dtype=dtype)
    elif init == 'unit':
        embeds = random_unit_init(shape, name, is_l2_norm, dtype=dtype)
    return embeds


def xavier_init(shape, name, is_l2_norm, dtype=None):
    with tf.name_scope('xavier_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def truncated_normal_init(shape, name, is_l2_norm, dtype=None):
    with tf.name_scope('truncated_normal'):  # Define a name scope in which all the variables here will be
        std = 1.0 / math.sqrt(shape[1])
        # create a new variable with name = name initialized from a truncated normal dist (values more than 2 std are discarded)
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.truncated_normal(stddev=std))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings  # return values as l2 normalized


def random_uniform_init(shape, name, is_l2_norm, minval=0, maxval=None, dtype=None):
    with tf.name_scope('random_uniform'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype,
                                     initializer=tf.initializers.random_uniform(minval=minval, maxval=maxval))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def random_unit_init(shape, name, is_l2_norm, dtype=None):
    with tf.name_scope('random_unit_init'):
        vectors = list()
        for i in range(shape[0]):
            vectors.append([random.gauss(0, 1) for j in range(shape[1])])
    embeddings = tf.Variable(preprocessing.normalize(np.matrix(vectors)), name=name, dtype=dtype)
    return tf.nn.l2_normalize(embeddings, 1) if is_l2_norm else embeddings


def orthogonal_init(shape, name, dtype=None):
    with tf.name_scope('orthogonal_init'):
        embeddings = tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.initializers.orthogonal())
    return embeddings
