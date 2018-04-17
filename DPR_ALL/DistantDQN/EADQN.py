import os
import ipdb
import numpy as np
import tensorflow as tf
from utils import save_pkl, load_pkl
from tensorflow.contrib.layers.python.layers import initializers

class DeepQLearner:
    def __init__(self, args, sess, agent_mode):
        print('Initializing the DQN...')
        self.sess = sess
        self.cnn_format = 'NCHW' # for GPU
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate
        self.decay_rate = args.decay_rate
        self.optimizer = args.optimizer
        self.momentum = args.momentum
        self.epsilon = args.epsilon

        self.target_output = 2**args.batch_act_num
        if agent_mode == 'act':
            self.words_num = args.words_num
            self.emb_dim = args.emb_dim
        else: # agent_mode == 'obj'
            self.words_num = args.context_len * 2 + 1
            self.emb_dim = args.obj_emb_dim
        self.word_dim = args.word_dim
        self.char_dim = args.char_dim
        self.num_k = args.num_k
        self.multi_cnn = args.multi_cnn
        self.use_k_max_pool = args.use_k_max_pool
        self.add_linear = args.add_linear
        with tf.variable_scope(agent_mode):
            self.build_dqn()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer,
        activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        with tf.variable_scope(name):
            #self.cnn_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]

            w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding, data_format=self.cnn_format)

            b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b, self.cnn_format)
        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def max_pooling(self, x, kernel_size, stride, name='max_pool'):
        with tf.variable_scope(name):
            stride_shape = [1, 1, stride[0], stride[1]]
            kernel_shape = [1, 1, kernel_size[0], kernel_size[1]]
            return tf.nn.max_pool(x, ksize=kernel_shape, strides=stride_shape, padding="VALID")


    def k_max_pooling(self, x, k=2, name='k_max_pool'):
        with tf.variable_scope(name):
            # self.cnn_format == 'NCHW'
            values, indices = tf.nn.top_k(tf.transpose(x, perm=[0,1,3,2]), k=k)
            return values


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
        ccs =  1  #convolution column stride
        fn = 32  #filter num
        num_k = self.num_k #8
        self.w = {}
        self.t_w = {}

        #ipdb.set_trace()
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        #initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        #self.cnn_format == 'NCHW'
        self.s_t = tf.placeholder('float32', [None, 1, self.words_num, self.emb_dim], name='s_t')
        self.target_s_t = tf.placeholder('float32', [None, 1, self.words_num, self.emb_dim], name='s_t')
        # training network
        def build_nn(name, weight, s_t):
            with tf.variable_scope(name):
                print('Initializing %s network...' % name)
                self.l1, weight['l1_w'], weight['l1_b'] = self.conv2d(s_t,
                    fn, [2, fw], [1, ccs], initializer, activation_fn, name='l1')
                self.l3, weight['l3_w'], weight['l3_b'] = self.conv2d(s_t,
                    fn, [3, fw], [1, ccs], initializer, activation_fn, name='l3')
                self.l5, weight['l5_w'], weight['l5_b'] = self.conv2d(s_t,
                    fn, [4, fw], [1, ccs], initializer, activation_fn, name='l5')
                self.l7, weight['l7_w'], weight['l7_b'] = self.conv2d(s_t,
                    fn, [5, fw], [1, ccs], initializer, activation_fn, name='l7')

                if self.multi_cnn:
                    tmpl1, weight['tmp_l1_w'], weight['tmp_l1_b'] = self.conv2d(self.l1,
                        fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l1')
                    tmpl3, weight['tmp_l3_w'], weight['tmp_l3_b'] = self.conv2d(self.l3,
                        fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l3')
                    tmpl5, weight['tmp_l5_w'], weight['tmp_l5_b'] = self.conv2d(self.l5,
                        fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l5')
                    tmpl7, weight['tmp_l7_w'], weight['tmp_l7_b'] = self.conv2d(self.l7,
                        fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l7')
                else:
                    tmpl1 = self.l1
                    tmpl3 = self.l3
                    tmpl5 = self.l5
                    tmpl7 = self.l7

                if self.use_k_max_pool:
                    self.l2 = self.k_max_pooling(tmpl1, k=num_k)
                    self.l4 = self.k_max_pooling(tmpl3, k=num_k)
                    self.l6 = self.k_max_pooling(tmpl5, k=num_k)
                    self.l8 = self.k_max_pooling(tmpl7, k=num_k)
                else:
                    self.l2 = self.max_pooling(
                        tmpl1, kernel_size = [self.words_num-1, 1], stride = [1, 1], name='l2')
                    self.l4 = self.max_pooling(
                        tmpl3, kernel_size = [self.words_num-2, 1], stride = [1, 1], name='l4')
                    self.l6 = self.max_pooling(
                        tmpl5, kernel_size = [self.words_num-3, 1], stride = [1, 1], name='l6')
                    self.l8 = self.max_pooling(
                        tmpl7, kernel_size = [self.words_num-4, 1], stride = [1, 1], name='l8')

                self.l9 = tf.concat([self.l2, self.l4, self.l6, self.l8], 3)
                l9_shape = self.l9.get_shape().as_list()
                self.l9_flat = tf.reshape(self.l9, [-1, reduce(lambda x, y: x * y, l9_shape[1:])])
                
                if self.add_linear:
                    self.l10, weight['l10_w'], weight['l10_b'] = self.linear(
                                        self.l9_flat, 256, activation_fn=activation_fn, name='l10')
                else:
                    self.l10 = self.l9_flat
                out_layer, weight['q_w'], weight['q_b'] = self.linear(self.l10, self.target_output, name='q')

                return out_layer
        self.q = build_nn('prediction', self.w, self.s_t)
        self.target_q = build_nn('target', self.t_w, self.target_s_t)


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


    def train(self, minibatch):
        # expand components of minibatch
        prestates, actions, rewards, poststates, terminals = minibatch
        
        post_input = poststates
        postq = self.target_q.eval({self.target_s_t: post_input})
        assert postq.shape == (self.batch_size, self.target_output)
        
        # calculate max Q-value for each poststate  
        maxpostq = np.max(postq, axis=1)
        assert maxpostq.shape == (self.batch_size,)
        
        pre_input = prestates
        preq = self.q.eval({self.s_t: pre_input})
        assert preq.shape == (self.batch_size, self.target_output)
        
        # make copy of prestate Q-values as targets  
        targets = preq.copy()

        # update Q-value targets for actions taken  
        for i, action in enumerate(actions):
            if terminals[i]:  
                targets[i, action] = float(rewards[i])
            else:  
                targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        _, q_t, delta, loss = self.sess.run([self.optim, self.q, self.delta, self.loss], {
            self.target_q_t: targets, self.s_t: pre_input,})     


    def predict(self, current_state):
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
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--discount_rate", type=float, default=0.9, help="")
    parser.add_argument("--learning_rate", type=float, default=0.0025, help="")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="")
    parser.add_argument("--optimizer", type=str, default='rmsprop', help="")
    parser.add_argument("--momentum", type=float, default=0.8, help="")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="")
    parser.add_argument("--target_output", type=int, default=2, help="")
    parser.add_argument("--words_num", type=int, default=500, help="")
    parser.add_argument("--word_dim", type=int, default=50, help="")
    parser.add_argument("--char_dim", type=int, default=50, help="")
    parser.add_argument("--emb_dim", type=int, default=100, help="")

    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        net = DeepQLearner(args, sess)
