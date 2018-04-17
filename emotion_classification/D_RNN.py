from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import numpy as np
class d_bi_RNN:
    # ==========
    #   MODEL
    # ==========

    # Parameters
    learning_rate = 0.5
    training_iters = 4000000
    vocabulary_size =40000
    # batch_size = 64
    display_step = 100
    checkpoint_step = 1000
    # Network Parameters
    input_len = 20
    embed_dim = 100 
    # embed_dim = 50
    n_hidden = 64 # hidden layer num of features
    n_classes = 6 # linear sequence or not

    # Define weights
    weights = {
        'out': tf.Variable(tf.truncated_normal([2*n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.truncated_normal([n_classes]))
    }

    def __init__(self,batch_size,learning_rate,learning_rate_decay_factor):
        # tf Graph input
        # self.x_input = tf.placeholder(shape = [None, self.input_len, self.seq_max_len],dtype= tf.float32)
        # self.y_input = tf.placeholder(dtype= tf.float32, shape = [None, self.n_classes])

        # self.x_ = tf.transpose(self.x_input, perm= [1,0,2])
        # self.x1 = tf.reshape(self.x_, shape= [-1,self.seq_max_len])
        # self.x = array_ops.split(self.x1,self.input_len, 0)
        self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.batch_size=batch_size
        self.x_input = tf.placeholder(shape = [None, self.input_len],dtype= tf.int32)
        self.y_input = tf.placeholder(dtype= tf.int32, shape = [None, self.n_classes])
        self.rescaling=tf.placeholder(dtype= tf.float32, shape = [None, self.n_classes])
        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[self.vocabulary_size,self.embed_dim],dtype=tf.float32)
            self.x_tmp=tf.nn.embedding_lookup(embedding,self.x_input)
        self.x_ = tf.transpose(self.x_tmp, perm= [1,0,2])
        self.x1 = tf.reshape(self.x_, shape= [-1,self.embed_dim])
        self.x = array_ops.split(0,self.input_len,self.x1)
        # Define a lstm cell with tensorflow
        self.lstm_fw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0,
                                                         input_size= [None, self.embed_dim])
        self.lstm_bw_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0,
                                                         input_size= [None, self.embed_dim])

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        self.outputs ,_,_=rnn.bidirectional_rnn(cell_fw= self.lstm_fw_cell,
                                               cell_bw= self.lstm_bw_cell,
                                               inputs= self.x,
                                               dtype= tf.float32)
                                               # sequence_length= self.seq_max_len)

        self.pred = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']

        # rescaling=tf.constant([[22.33, 13.15,14.79, 9.40, 16.88, 0.49] for i in range(self.batch_size)])

        # self.pred=tf.multiply(pred,self.rescaling)
        # import ipdb;ipdb.set_trace()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input,logits=self.pred))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta').minimize(self.cost)
        self.m=tf.argmax(self.pred, 1)
        self.n=tf.argmax(self.y_input, 1)
        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.correct_num= tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
