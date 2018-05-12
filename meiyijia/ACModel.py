# coding: utf-8
import ipdb
import argparse
import numpy as np
import tensorflow as tf
from collections import deque


class ActorCritic(object):
    """docstring for ActorCritic"""
    def __init__(self, env, sess, args):
        self.env = env
        self.sess = sess
        self.size = args.size
        self.n_items = args.n_items
        self.n_stores = args.n_stores
        self.keep_prob = args.keep_prob
        self.lstm_layers = args.lstm_layers
        self.learning_rate = args.learning_rate

        self.memory = deque(maxlen=args.mem_size)
        ipdb.set_trace()
        # =================================================================== #
        #                               Actor Model                           #
        # =================================================================== #
        self.actor_state_input, self.actor_y_mask, self.actor_y, self.actor_output = self.build_actor_network()
        _, __, self.target_actor_y, self.target_actor_ouput = self.build_actor_network(name='target_actor')


        self.actor_critic_grad = tf.placeholder(tf.float32, [1]) # a = actor(s)
        actor_weights = [v for v in tf.trainable_variables() if 'actor' in v.name]
        self.actor_grads = tf.gradients(self.actor_output, actor_weights, # dC/dA
                                        -self.actor_critic_grad) # de/dC
        grads = zip(self.actor_grads, actor_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # =================================================================== #
        #                              Critic Model                           #
        # =================================================================== #
        self.critic_state_input, self.critic_action_input, self.critic_output = self.build_critic_network()
        _, __, self.target_critic_output = self.build_critic_network(name='target_critic')

        # de/dC, that is d(error)/d(action)
        self.critic_grads = tf.gradients(self.critic_output, self.critic_action_input) 

        self.sess.run(tf.global_variables_initializer())


    def build_actor_network(self, name='actor'):
        time_steps = self.n_items
        batch_size = self.n_stores
        with tf.variable_scope(name):
            state_input = tf.placeholder(tf.float32, [batch_size, time_steps, self.size])
            y_mask = tf.placeholder(tf.float32, [batch_size, time_steps])

            lstm = tf.contrib.rnn.BasicLSTMCell(self.size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell([drop]*self.lstm_layers)

            init_state = cell.zero_state(state_input.shape[0], tf.float32)
            # outputs.shape = [batch_size, time_steps, self.size]
            outputs, hidden_state = tf.nn.dynamic_rnn(cell, state_input, sequence_length=None, initial_state=init_state, dtype=tf.float32)

            output = tf.reshape(outputs, [-1, self.size])
            w = tf.get_variable('w', [self.size, 1], dtype=tf.float32)
            b = tf.get_variable('b', [1], dtype=tf.float32)
            z = tf.matmul(output, w) + b
            y = tf.reshape(z, [-1, time_steps])

            order = tf.reduce_sum(y*y_mask)

            return state_input, y_mask, y, order


    def build_critic_network(self, name='critic'):
        time_steps = self.n_items
        batch_size = self.n_stores
        with tf.variable_scope(name):
            state_input = tf.placeholder(tf.float32, [batch_size, time_steps, self.size])
            action_input = tf.placeholder(tf.float32, [batch_size, 1])

            lstm = tf.contrib.rnn.BasicLSTMCell(self.size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell([drop]*self.lstm_layers)

            init_state = cell.zero_state(state_input.shape[0], tf.float32)
            # outputs.shape = [batch_size, time_steps, self.size]
            outputs, hidden_state = tf.nn.dynamic_rnn(cell, state_input, sequence_length=None, initial_state=init_state, dtype=tf.float32)

            output = tf.reshape(outputs, [-1, time_steps*self.size])
            state_w = tf.get_variable('state_w', [time_steps*self.size, 1], dtype=tf.float32)
            state_b = tf.get_variable('state_b', [1], dtype=tf.float32)
            # state_z.shape = [batch_size, 1]
            state_z = tf.matmul(output, state_w) + state_b
            # state_action.shape = [batch_size, 2]
            state_action = tf.concat([state_z, action_input], axis=1)
            critic_input = tf.reshape(state_action, [1, -1])
            critic_w = tf.get_variable('critic_w', [batch_size*2, 1], dtype=tf.float32)
            critic_b = tf.get_variable('critic_b', [1], dtype=tf.float32)
            critic_z = tf.matmul(critic_input, critic_w) + critic_b

            return state_input, action_input, critic_z

    # ===================================================================== #
    #                              Model Training                           #
    # ===================================================================== #
    def remember(self, y_mask, cur_state, action, reward, new_state, done):
        self.memory.append([y_mask, cur_state, action, reward, new_state, done])


    def _train_actor(self, samples):
        for sample in samples:
            y_mask, cur_state, action, reward, new_state, _ = sample
            predicted_action = self.sess.run(self.actor_output, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_y_mask: y_mask
                })



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-size', type=int, default=30)
    parser.add_argument('-n_items', type=int, default=50)
    parser.add_argument('-n_stores', type=int, default=100)
    parser.add_argument('-mem_size', type=int, default=1000)
    parser.add_argument('-lstm_layers', type=int, default=2)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-keep_prob', type=float, default=0.5)
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = ActorCritic('', sess, args)


if __name__ == '__main__':
    main()