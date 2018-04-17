import ipdb
import numpy as np
import tensorflow as tf
from tf.contrib.seq2seq import *

class ActFinder(object):
    """docstring for ActFinder"""
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.enc_vsize = args.enc_vsize
        self.dec_vsize = args.dec_vsize
        self.word_dim = args.word_dim
        self.rnn_num = args.rnn_num
        self.layer_num = args.layer_num
        self.grad_clip = args.grad_clip


    def build_model(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')

        # define embedding layer
        with tf.variable_scope('embedding'):
            enc_emb = tf.Variable(
                tf.truncated_normal(shape=[self.enc_vsize, self.word_dim], stddev=0.1), 
                name='enc_emb')
            dec_emb = enc_emb

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(self.self.rnn_num, self.layer_num)

        input_x_emb = tf.nn.embedding_lookup(enc_emb, self.input_x)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_emb, dtype=tf.float32)

        # define helper for decoder
        if predict:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(dec_emb, self.start_tokens, self.end_token)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[None, None], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[None], name='batch_seq_length')
            target_emb = tf.nn.embedding_lookup(dec_emb, self.target_ids)
            helper = TrainingHelper(target_emb, self.decoder_seq_length)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(self.dec_vsize)
            decoder_cell = self._get_simple_lstm(self.self.rnn_num, self.layer_num)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        if not predict:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, self.dec_vsize])
            print 'shape logits_flat:{}'.format(logits_flat.shape)
            print 'shape logits:{}'.format(logits.rnn_output.shape) 

            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)


    def _get_simple_lstm(self, self.self.rnn_num, self.layer_num):
        lstm_layers = [tf.contrib.rnn.LSTMCell(self.self.rnn_num) for _ in xrange(self.layer_num)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


    def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.dec_vsize)