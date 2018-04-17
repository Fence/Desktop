#-*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):

    def __init__(self, rnn_size, layer_size, enc_vsize, 
        dec_vsize, word_dim, grad_clip, predict=False):
        # define inputs
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

        # define embedding layer
        with tf.variable_scope('embedding'):
            enc_emb = tf.Variable(tf.truncated_normal(shape=[enc_vsize, word_dim], stddev=0.1), 
                name='enc_emb')
            dec_emb = tf.Variable(tf.truncated_normal(shape=[dec_vsize, word_dim], stddev=0.1),
                name='dec_emb')

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(rnn_size, layer_size)

        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(enc_emb, self.input_x)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)

        # define helper for decoder
        if predict:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(dec_emb, self.start_tokens, self.end_token)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            with tf.device('/cpu:0'):
                target_embeddeds = tf.nn.embedding_lookup(dec_emb, self.target_ids)
            helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(dec_vsize)
            decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        if not predict:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, dec_vsize])
            print 'shape logits_flat:{}'.format(logits_flat.shape)
            print 'shape logits:{}'.format(logits.rnn_output.shape) 

            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)