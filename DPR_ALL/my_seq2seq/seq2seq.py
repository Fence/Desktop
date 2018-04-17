import ipdb
import numpy as np
import tensorflow as tf
from tf.contrib.seq2seq import *

class Seq2Seq(object):
    """docstring for Seq2Seq"""
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.enc_vsize = self.dec_vsize = args.vocab_size
        self.enc_len = args.enc_len
        self.dec_len = args.dec_len
        self.word_dim = args.word_dim
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num
        self.grad_clip = args.grad_clip


    def build_model(self, train_flag):
        self.input_x = tf.placeholder(tf.int32, [None, self.enc_len], 'input_ids')

        # define embedding layer
        with tf.variable_scope('embedding'):
            initializer = tf.truncated_normal([self.enc_vsize, self.word_dim], stddev=0.1)
            enc_emb = tf.Variable(initializer, 'enc_emb')
            dec_emb = enc_emb

        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm()

        input_x_emb = tf.nn.embedding_lookup(enc_emb, self.input_x)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_emb, dtype=tf.float32)

        # define helper for decoder
        if train_flag:
            self.target_ids = tf.placeholder(tf.int32, [None, self.dec_len], 'target_ids')
            self.dec_seq_lens = tf.placeholder(tf.int32, [self.dec_len], 'batch_seq_length')
            target_emb = tf.nn.embedding_lookup(dec_emb, self.target_ids)
            helper = TrainingHelper(target_emb, self.dec_seq_lens)
        else:
            self.start_tokens = tf.placeholder(tf.int32, [None], 'start_tokens')
            self.end_token = tf.placeholder(tf.int32, 'end_token')
            helper = GreedyEmbeddingHelper(dec_emb, self.start_tokens, self.end_token)

        with tf.variable_scope('decoder'):
            fc_layer = Dense(self.dec_vsize)
            decoder_cell = self._get_simple_lstm()
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)

        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)

        if train_flag:
            targets = tf.reshape(self.target_ids, [-1])
            logits_flat = tf.reshape(logits.rnn_output, [-1, self.dec_vsize])
            print('shape logits_flat:{}'.format(logits_flat.shape))
            print('shape logits:{}'.format(logits.rnn_output.shape))

            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.grad_clip)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            self.prob = tf.nn.softmax(logits)


    def _get_simple_lstm(self):
        lstm_layers = [tf.contrib.rnn.LSTMCell(self.hidden_size) for _ in xrange(self.layer_num)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


    def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.dec_vsize)