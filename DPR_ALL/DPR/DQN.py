import os
import ipdb
import numpy as np
import tensorflow as tf
from utils import save_pkl, load_pkl
from tensorflow.contrib.layers.python.layers import initializers

class DeepQLearner:
    def __init__(self, args, sess):
        print('Initializing the DQN...')
        self.sess = sess
        self.cnn_format = args.cnn_format
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate
        self.decay_rate = args.decay_rate
        self.optimizer = args.optimizer
        self.momentum = args.momentum
        self.epsilon = args.epsilon

        self.target_output = args.target_output
        self.char_emb_flag = args.char_emb_flag
        self.num_actions = args.num_actions
        self.words_num = args.words_num
        self.char_dim = args.char_dim
        self.emb_dim = args.emb_dim
        self.nchars = args.nchars
        self.word_dim = args.word_dim
        self.build_dqn()


    def build_model(self):
        self.enc_input = tf.placeholder(tf.int32, 
                    [self.batch_size, self.enc_len, self.dec_len, self.word_dim], 'enc_input')
        self.dec_target = tf.placeholder(tf.int32, 
                    [self.batch_size, self.dec_len, self.word_dim], 'dec_target')

        # build action encoder
        enc_act_cell = tf.contrib.rnn.BasicLSTMCell(self.word_dim)
        if self.keep_prob < 1: # dropout
            enc_act_cell = tf.contrib.rnn.DropoutWrapper(
                enc_act_cell, output_keep_prob=self.keep_prob)
        enc_act_lstm = tf.contrib.rnn.MultiRNNCell([enc_act_cell] * self.enc_layers)

        self.init_enc_act_state = enc_act_lstm.zero_state(self.batch_size, tf.float32)

        # encoder computation
        self.enc_act_outputs = []
        state = self.init_enc_act_state # the states of a batch samples
        for act_step in range(self.enc_len):
            #tmp_enc_outputs = []
            with tf.variable_scope('enc_act%d' % act_step):
                for time_step in range(self.dec_len):
                    if time_step > 0: 
                        tf.get_variable_scope().reuse_variables()
                    # enc_cell_out: [batch, word_dim]
                    (enc_cell_output, state) = enc_act_lstm(self.enc_input[:, act_step, time_step, :], state)
                    #tmp_enc_outputs.append(tf.reshape(enc_cell_output, [-1, 1, self.word_dim]))
                    # self.enc_act_outputs: [self.enc_len, self.batch_size, 1, self.word_dim]
                    if time_step == self.dec_len - 1: 
                        self.enc_act_outputs.append(tf.reshape(enc_cell_output, [-1, 1, self.word_dim]))
        # self.enc_act_outputs: [self.batch_size, self.enc_len, self.word_dim]
        self.enc_act_outputs = tf.transpose(self.enc_act_outputs, perm = [1, 0, 2, 3])
        self.enc_act_outputs = tf.reshape(self.enc_act_outputs, [-1, self.enc_len, self.word_dim])

        # build plan encoder 
        enc_plan_cell = tf.contrib.rnn.BasicLSTMCell(self.word_dim)
        if self.keep_prob < 1: # dropout
            enc_plan_cell = tf.contrib.rnn.DropoutWrapper(
                enc_plan_cell, output_keep_prob=self.keep_prob)
        enc_plan_lstm = tf.contrib.rnn.MultiRNNCell([enc_plan_cell] * self.enc_layers) 
        self.init_enc_plan_state = enc_plan_lstm.zero_state(self.batch_size, tf.float32)

        # encoder computation
        self.enc_plan_outputs = []
        state = self.init_enc_plan_state # the states of a batch samples
        with tf.variable_scope('enc_plan'):
            for time_step in range(self.enc_len):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                # enc_cell_out: [batch, word_dim]
                (enc_cell_output, state) = enc_plan_lstm(self.enc_act_outputs[:, time_step, :], state)
                self.enc_plan_outputs.append(tf.reshape(enc_cell_output, [-1, 1, self.word_dim]))
        # enc_output.shape = [batch*enc_len, word_dim]
        #enc_output = tf.reshape(tf.concat(self.enc_plan_outputs, 1), [-1, self.word_dim])

        # build decoder
        dec_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.word_dim)
        if self.keep_prob < 1: # dropout
            dec_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                dec_lstm_cell, output_keep_prob=self.keep_prob)
        dec_lstm = tf.contrib.rnn.MultiRNNCell([dec_lstm_cell] * self.dec_layers)
        
        # decoder computation
        self.dec_outputs = []
        #state = tf.reshape(self.enc_plan_outputs[-1], [-1, self.word_dim])
        with tf.variable_scope('dec'):
            for time_step in range(self.dec_len):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (dec_cell_output, state) = dec_lstm(self.dec_target[:, time_step, :], state)
                self.dec_outputs.append(tf.reshape(dec_cell_output, [-1, 1, self.word_dim]))
        dec_output = tf.reshape(tf.concat(self.dec_outputs, 1), [-1, self.word_dim])

        # sampled softmax
        w = tf.get_variable('w', [self.word_dim, self.vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('b', [self.vocab_size])
        self.z = tf.matmul(dec_output, w) + b
        labels = tf.reshape(self.dec_target, [-1, 1])
        #labels = tf.reshape(self.dec_target, [-1])
        #ipdb.set_trace()
        #self.cost = tf.reduce_mean(
        #    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.z))
        self.cost = tf.reduce_sum(tf.nn.sampled_softmax_loss(
                                w_t, b, labels, dec_output, self.num_samples, self.vocab_size))
        #ipdb.set_trace()
        self.lr = tf.Variable(self.current_lr)
        if self.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adam':    
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        else: 
            self.train_op = tf.train.RMSPropOptimizer(
                0.0025, decay=self.decay, momentum=0.1, epsilon=1e-10).minimize(self.cost)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer,
        activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        with tf.variable_scope(name):
            if self.cnn_format == 'NCHW':
                stride = [1, 1, stride[0], stride[1]]
                kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
            elif self.cnn_format == 'NHWC':
                stride = [1, stride[0], stride[1], 1]
                kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

            w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding, data_format=self.cnn_format)

            b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b, self.cnn_format)
        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def max_pooling(self, x, kernel_size, stride, name='max_pool'):
        with tf.variable_scope(name):
            if self.cnn_format == 'NCHW':
                stride_shape = [1, 1, stride[0], stride[1]]
                kernel_shape = [1, 1, kernel_size[0], kernel_size[1]]
                return tf.nn.max_pool(x, ksize=kernel_shape, strides=stride_shape, padding="VALID")
            elif self.cnn_format == 'NHWC':
                stride_shape = [1, stride[0], stride[1], 1]
                kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
                return tf.nn.max_pool(x, ksize=kernel_shape, strides=stride_shape, padding="VALID")


    def linear(self, input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(name):
            w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
                #tf.truncated_normal_initializer(0, stddev))
                
            b = tf.get_variable('bias', [output_size],
                initializer=tf.constant_initializer(bias_start))

            out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b


    def build_dqn(self):
        fw = self.emb_dim - 1  #filter width
        ccs = 1  #convolution column stride
        fn = 32  #filter num
        self.w = {}
        self.t_w = {}

        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        #initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
            print 'Initializing main network...'
            self.act = tf.placeholder(tf.float32, [None, self.words_num, self.word_dim], 'act')
            self.obj = tf.placeholder(tf.float32, [None, self.objs_num, self.word_dim], 'obj')

            self.w['obj_w'] = tf.get_variable('obj_w', [self.objs_num, 1])
            self.w['obj_b'] = tf.get_variable('obj_b', [1])
            tmp_obj = tf.reshape(tf.transpose(self.obj, perm=[0, 2, 1]), [-1, self.objs_num])
            self.proj_obj = tf.matmul(tmp_obj, self.w['obj_w']) + self.w['obj_b']
            self.proj_obj = tf.reshape(self.proj_obj, [-1, self.word_dim])

            self.s_t = tf.concat([self.act, self.proj_obj], 1)
            if self.cnn_format == 'NHWC':  #CPU only
                self.s_t = tf.reshape(self.s_t, [None, self.words_num, self.emb_dim, 1])
            else:
                self.s_t = tf.reshape(self.s_t, [None, 1, self.words_num, self.emb_dim])
            
            self.l1, self.w['l1_w'], self.w['l1_b'] = self.conv2d(self.s_t,
                fn, [2, fw], [1, ccs], initializer, activation_fn, name='l1')
            self.l2 = self.max_pooling(self.l1, kernel_size = [self.words_num-1, 1], stride = [1, 1], name='l2')
            
            self.l3, self.w['l3_w'], self.w['l3_b'] = self.conv2d(self.s_t,
                fn, [3, fw], [1, ccs], initializer, activation_fn, name='l3')
            self.l4 = self.max_pooling(self.l3, kernel_size = [self.words_num-2, 1], stride = [1, 1], name='l4')

            self.l5, self.w['l5_w'], self.w['l5_b'] = self.conv2d(self.s_t,
                fn, [4, fw], [1, ccs], initializer, activation_fn, name='l5')
            self.l6 = self.max_pooling(self.l5, kernel_size = [self.words_num-3, 1], stride = [1, 1], name='l6')

            self.l7, self.w['l7_w'], self.w['l7_b'] = self.conv2d(self.s_t,
                fn, [5, fw], [1, ccs], initializer, activation_fn, name='l7')
            self.l8 = self.max_pooling(self.l7, kernel_size = [self.words_num-4, 1], stride = [1, 1], name='l8')

            #ipdb.set_trace()
            self.l9 = tf.concat([self.l2, self.l4, self.l6, self.l8], 3)
            l9_shape = self.l9.get_shape().as_list()
            self.l9_flat = tf.reshape(self.l9, [-1, reduce(lambda x, y: x * y, l9_shape[1:])])
            
            self.l10, self.w['l10_w'], self.w['l10_b'] = self.linear(
                self.l9_flat, 256, activation_fn=activation_fn, name='l10')
            self.q, self.w['q_w'], self.w['q_b'] = self.linear(
                self.l10, self.target_output, name='q')
            

        # target network
        with tf.variable_scope('target'):
            print 'Initializing target network...'
            self.target_act = tf.placeholder(tf.float32, [None, self.words_num, self.word_dim], 'act')
            self.target_obj = tf.placeholder(tf.float32, [None, self.objs_num, self.word_dim], 'obj')

            self.t_w['obj_w'] = tf.get_variable('obj_w', [self.objs_num, 1])
            self.t_w['obj_b'] = tf.get_variable('obj_b', [1])
            tmp_target_obj = tf.reshape(tf.transpose(self.target_obj, perm=[0, 2, 1]), [-1, self.objs_num])
            self.target_proj_obj = tf.matmul(tmp_target_obj, self.t_w['obj_w']) + self.t_w['obj_b']
            self.target_proj_obj = tf.reshape(self.target_proj_obj, [-1, self.word_dim])

            self.target_s_t = tf.concat([self.target_act, self.target_proj_obj], 1)
            if self.cnn_format == 'NHWC':  #CPU only
                self.target_s_t = tf.reshape(self.target_s_t, [None, self.words_num, self.emb_dim, 1])
            else:
                self.target_s_t = tf.reshape(self.target_s_t, [None, 1, self.words_num, self.emb_dim])

            self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = self.conv2d(self.target_s_t,
                fn, [2, fw], [1, ccs], initializer, activation_fn, name='l1')
            self.target_l2 = self.max_pooling(self.target_l1, 
                kernel_size = [self.words_num-1, 1], stride = [1, 1], name='l2')

            self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = self.conv2d(self.target_s_t,
                fn, [3, fw], [1, ccs], initializer, activation_fn, name='l3')
            self.target_l4 = self.max_pooling(self.target_l3, 
                kernel_size = [self.words_num-2, 1], stride = [1, 1], name='l4')

            self.target_l5, self.t_w['l5_w'], self.t_w['l5_b'] = self.conv2d(self.target_s_t,
                fn, [4, fw], [1, ccs], initializer, activation_fn, name='l5')
            self.target_l6 = self.max_pooling(self.target_l5, 
                kernel_size = [self.words_num-3, 1], stride = [1, 1], name='l6')

            self.target_l7, self.t_w['l7_w'], self.t_w['l7_b'] = self.conv2d(self.target_s_t,
                fn, [5, fw], [1, ccs], initializer, activation_fn, name='l7')
            self.target_l8 = self.max_pooling(self.target_l7, 
                kernel_size = [self.words_num-4, 1], stride = [1, 1], name='l8')

            self.target_l9 = tf.concat([self.target_l2, self.target_l4, self.target_l6, self.target_l8], 3)
            self.target_l9_flat = tf.reshape(self.target_l9, [-1, reduce(lambda x, y: x * y, l9_shape[1:])])

            self.target_l10, self.t_w['l10_w'], self.t_w['l10_b'] = self.linear(
                self.target_l9_flat, 256, activation_fn=activation_fn, name='l10')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = self.linear(
                self.target_l10, self.target_output, name='q')


        with tf.variable_scope('pred_to_target'):
            print 'Initializing pred_to_target...'
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])


        # optimizer
        with tf.variable_scope('optimizer'):
            print 'Initializing optimizer...'
            self.target_q_t = tf.placeholder('float32', [self.batch_size, self.target_output], name='target_q_t')
            self.delta = self.target_q_t - self.q
            #self.loss = tf.reduce_mean(tf.square(self.delta), name='loss')
            self.loss = tf.reduce_sum(tf.square(self.delta), name='loss')
            if self.optimizer == 'sgd':
                self.optim = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer == 'adam':
                self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            elif self.optimizer == 'adadelta':
                self.optim = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
            else:
                self.optim = tf.train.RMSPropOptimizer(
                self.learning_rate, decay=self.decay_rate, momentum=self.momentum, epsilon=self.epsilon).minimize(self.loss)
            
        tf.global_variables_initializer().run()


    def update_target_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})


    def train(self, minibatch, epoch):
        # expand components of minibatch
        prestates, actions, rewards, poststates, terminals = minibatch
        
        if self.cnn_format == 'NHWC':
            post_input = np.transpose(poststates, axes=(0, 2, 3, 1))
        else:
            post_input = poststates
        postq = self.target_q.eval({self.target_s_t: post_input})
        assert postq.shape == (self.batch_size, self.target_output)
        
        # calculate max Q-value for each poststate  
        maxpostq = np.max(postq, axis=1)
        assert maxpostq.shape == (self.batch_size,)
        
        if self.cnn_format == "NHWC":
            pre_input = np.transpose(prestates, axes=(0, 2, 3, 1))
        else:
            pre_input = prestates
        preq = self.q.eval({self.s_t: pre_input})
        assert preq.shape == (self.batch_size, self.target_output)
        
        # make copy of prestate Q-values as targets  
        targets = preq.copy()

        # update Q-value targets for actions taken  
        for i, action in enumerate(actions):
            if terminals[i]:  
                targets[i, action%2] = float(rewards[i])
            else:  
                targets[i, action%2] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        _, q_t, delta, loss = self.sess.run([self.optim, self.q, self.delta, self.loss], {
            self.target_q_t: targets, self.s_t: pre_input,})     


    def predict(self, current_state):
        if self.cnn_format == 'NHWC':
            state_input = np.reshape(current_state, [1, self.words_num, self.emb_dim, 1])
        else:
            state_input = np.reshape(current_state, [1, 1, self.words_num, self.emb_dim])

        qvalues = self.q.eval({self.s_t: state_input})

        return qvalues



    def save_weights(self, weight_dir):
        print('Saving weights to %s ...' % weight_dir)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(weight_dir, "%s.pkl" % name))


    def load_weights(self, weight_dir, cpu_mode=False):
        print('Loading weights from %s ...' % weight_dir)
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}

            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(weight_dir, "%s.pkl" % name))})

        self.update_target_network()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_actions", type=int, default=1000, help="Total actions of this task.")
    parser.add_argument("--words_num", type=int, default=500, help="Total words of an input text.")
    parser.add_argument("--word_dim", type=int, default=100, help="Size of word vector.")
    parser.add_argument("--learning_rate", type=float, default=0.0025, help="Learning rate.")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
    parser.add_argument("--discount_rate", type=float, default=0.9, help="Discount rate for future rewards.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for neural network.")
    parser.add_argument("--target_output", type=int, default=2, help="Output dimension of DQN.")
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        net = DeepQLearner(args, sess)
