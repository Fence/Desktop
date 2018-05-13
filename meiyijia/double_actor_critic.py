# coding: utf-8
import ipdb
import time
import random
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from collections import deque

from environment import Environment


class ActorCritic(object):
    """docstring for ActorCritic"""
    def __init__(self, env, sess, args):
        self.env = env
        self.sess = sess
        self.emb_dim = args.emb_dim
        self.n_items = args.n_items
        self.n_stores = args.n_stores
        self.batch_size = args.batch_size
        self.conv_layers = args.conv_layers

        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.max_random_action = args.max_random_action

        self.memory = {'cur_state': deque(maxlen=args.mem_size),
                       'action':    deque(maxlen=args.mem_size),
                       'reward':    deque(maxlen=args.mem_size),
                       'new_state': deque(maxlen=args.mem_size),
                       'terminal':  deque(maxlen=args.mem_size)}
        self.actor_state_input, self.actor_model = self.build_actor_network()
        _, self.target_actor_model = self.build_actor_network()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, 1]) 
        # where we will feed de/dC (from critic)
        
        #ipdb.set_trace()
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #       

        self.critic_state_input, self.critic_action_input, \
            self.critic_model = self.build_critic_network()
        _, _, self.target_critic_model = self.build_critic_network()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())


    def build_actor_network(self, name='actor'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        inputs = state_input
        for i in range(self.conv_layers):
            conv = Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu')(inputs)
            mp = MaxPooling2D((100, 1), strides=(100, 1), padding='valid')(conv)
            inputs = mp
        #state_h1 = Dense(8, activation='relu')(state_input)
        state_h1 = Flatten()(mp)
        output = Dense(1)(state_h1)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return state_input, model


    def build_critic_network(self, name='critic'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        inputs = state_input
        for i in range(self.conv_layers):
            conv = Conv2D(32, (5, 5), strides=(1, 1), padding='valid', activation='relu')(inputs)
            mp = MaxPooling2D((100, 1), strides=(100, 1), padding='valid')(conv)
            inputs = mp
        #state_h1 = Dense(8, activation='relu')(state_input)
        flat = Flatten()(mp)
        state_h1 = Dense(1, activation='relu')(flat)

        action_input = Input(shape=[1])
        action_h1 = Dense(1, activation='relu')(action_input)

        merged = Add()([state_h1, action_h1])
        output = Dense(1)(merged)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return state_input, action_input, model

    # ===================================================================== #
    #                              Model Training                           #
    # ===================================================================== #
    def remember(self, cur_state, action, reward, new_state, terminal):
        self.memory['cur_state'].append(cur_state)
        self.memory['action'].append(action)
        self.memory['reward'].append(reward)
        self.memory['new_state'].append(new_state)
        self.memory['terminal'].append(terminal)

    def _train_actor(self, cur_states, actions, rewards, new_states, terminals):
        # 训练actor网络，朝着策略梯度的方向修正
        predicted_actions = self.actor_model.predict(cur_states)
        # 输入s_t到当前目标actor得到动作a_t，再将s_t, a_t输入critic网络，计算得到Q(s_t, a_t)
        # 然后计算梯度d Q(s_t, a_t) / d a_t，即de/dC，C这里可以看做是动作a_t
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  cur_states,
            self.critic_action_input: predicted_actions
        })[0]
        # 接着在de/dC的基础上，利用de/dA = de/dC * dC/dA即可求得actor网络的梯度，即策略梯度
        # 因为actor.output就是a_t，所以d actor.output / d actor_model_weights就是dC/dA，
        # 也就是d a_t/d theta，其中theta是policy网络的参数
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: cur_states,
            self.actor_critic_grad: grads
        })
            
    def _train_critic(self, cur_states, actions, rewards, new_states, terminals):
        # 训练critic网络；  输入：s_t, a_t；  输出：累积回报的期望 Q(s_t, a_t)
        # s_{t+1}输入actor得到a_{t+1}，再将s_{t+1},a_{t+1}输入critic计算R_{t+1:infty}
        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states, target_actions])#[0][0]
        for i in range(len(terminals)):
            if not terminals[i]:
                # R_t = r_t + gamma * R_{t+1:infty}
                rewards[i] += self.gamma * future_rewards[i][0]
        self.critic_model.fit([cur_states, actions], rewards, verbose=0)
        
    def train(self):
        if len(self.memory['cur_state']) < self.batch_size:
            return

        #ipdb.set_trace()
        rewards = []
        indexes = random.sample(range(len(self.memory['cur_state'])), self.batch_size)
        cur_states = np.array(self.memory['cur_state'], dtype=np.float32)[indexes]
        actions = np.array(self.memory['action'], dtype=np.int32)[indexes]
        rewards = np.array(self.memory['reward'], dtype=np.float32)[indexes]
        new_states = np.array(self.memory['new_state'], dtype=np.float32)[indexes]
        terminals = np.array(self.memory['terminal'], dtype=np.bool)[indexes]
        self._train_critic(cur_states, actions, rewards, new_states, terminals)
        self._train_actor(cur_states, actions, rewards, new_states, terminals)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)     

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            while True:
                action = np.random.randn(1)[0]*self.max_random_action + self.max_random_action
                if action > 0:
                    return int(action)
        return int(self.actor_model.predict(cur_state)[0][0])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_dim', type=int, default=30)
    parser.add_argument('-n_items', type=int, default=688)
    parser.add_argument('-n_stores', type=int, default=2000)
    parser.add_argument('-mem_size', type=int, default=5000)
    parser.add_argument('-conv_layers', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gamma', type=float, default=0.95)
    parser.add_argument('-epsilon', type=float, default=1.0)
    parser.add_argument('-epsilon_decay', type=float, default=0.995)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-max_random_action', type=int, default=50)
    parser.add_argument('-max_stores_per_warehouse', type=int, default=2000)
    parser.add_argument('-gpu_rate', type=float, default=0.2)
    parser.add_argument('-result_dir', type=str, default='results/test')
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    with open('%s.txt'%args.result_dir,'w') as args.logger:
        for (k,v) in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
            args.logger.write('{}: {}\n'.format(k, v))
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            #ipdb.set_trace()
            start = time.time()
            env = Environment(args)
            model = ActorCritic(env, sess, args)

            total_reward = epoch = 0
            for i in range(10000):
                if i % 100 == 0:
                    print('step: %d' % i)
                cur_state = env.get_state()
                tmp_state = np.reshape(cur_state, [1, cur_state.shape[0], cur_state.shape[1], 1])
                action = model.act(tmp_state)
                new_state, reward, terminal = env.step(action)
                cur_state = np.reshape(cur_state, [cur_state.shape[0], cur_state.shape[1], 1])
                new_state = np.reshape(new_state, [new_state.shape[0], new_state.shape[1], 1])

                model.remember(cur_state, action, reward, new_state, terminal)
                model.train()

                if not terminal:
                    total_reward += reward
                else:
                    args.logger.write(('epoch {}, reward: {:.4f}\n'.format(epoch, total_reward)))
                    print('epoch {}, reward: {:.4f}\n'.format(epoch, total_reward))
                    epoch += 1
                    total_reward = 0

        args.logger.write('\nmax_action: {}\nsorted seen actions:\n'.format(env.max_action))
        for k,v in sorted(self.seen_actions.iteritems(), key=lambda x:x[1], reverse=True):
            args.write('{}: {}\n'.format(k*10, v))
        end = time.time()
        args.logger.write('\nTime cost: %.2fs\n' % (end - start))
        print('\nTime cost: %.2fs\n' % (end - start))



if __name__ == '__main__':
    main()