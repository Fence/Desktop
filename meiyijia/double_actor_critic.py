# coding: utf-8
import ipdb
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
from collections import deque

from data_processor import DateProcessing, timeit
from environment import Environment


class ActorCritic(object):
    """docstring for ActorCritic"""
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess
        self.emb_dim = args.emb_dim
        self.n_stores = args.n_stores
        self.dense_dim = args.dense_dim
        self.batch_size = args.batch_size
        self.conv_layers = args.conv_layers
        self.maxp_dim = args.maxp_dim

        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.max_random_action = args.max_random_action

        self.memory = {'cur_state': deque(maxlen=args.mem_size),
                       'action':    deque(maxlen=args.mem_size),
                       'reward':    deque(maxlen=args.mem_size),
                       'new_state': deque(maxlen=args.mem_size),
                       'terminal':  deque(maxlen=args.mem_size)}
        self.actor_state_input, self.actor_model = self.build_actor_network()
        _, self.actor_target_model = self.build_actor_network()
        self.actor_model.summary()

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

        self.critic_state_input, self.critic_action_input, self.critic_model = self.build_critic_network()
        _, _, self.critic_target_model = self.build_critic_network()
        self.critic_model.summary()

        self.critic_grads = tf.gradients(self.critic_model.output, 
            self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())


    def build_actor_network(self, name='actor'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        if self.conv_layers > 1:
            inputs = state_input
            for i in xrange(self.conv_layers):
                conv = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
                mp = MaxPooling2D((10, 1), strides=(10, 1), padding='valid')(conv)
                inputs = mp
        else:
            conv = Conv2D(32, (5, 5), strides=(2, 2), padding='valid', activation='relu')(state_input)
            mp = MaxPooling2D((self.maxp_dim, 1), strides=(self.maxp_dim, 1), padding='valid')(conv)
        flat = Flatten()(mp)
        state_h1 = Dense(self.dense_dim, activation='relu')(flat)
        output = Dense(1)(state_h1)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        #model.summary()
        return state_input, model


    def build_critic_network(self, name='critic'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        if self.conv_layers > 1:
            inputs = state_input
            for i in xrange(self.conv_layers):
                conv = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
                mp = MaxPooling2D((10, 1), strides=(10, 1), padding='valid')(conv)
                inputs = mp
        else:
            conv = Conv2D(32, (5, 5), strides=(2, 2), padding='valid', activation='relu')(state_input)
            mp = MaxPooling2D((self.maxp_dim, 1), strides=(self.maxp_dim, 1), padding='valid')(conv)
        flat = Flatten()(mp)
        state_h1 = Dense(self.dense_dim, activation='relu')(flat)
        state_h2 = Dense(1, activation='relu')(state_h1)

        action_input = Input(shape=[1])
        action_h1 = Dense(1, activation='relu')(action_input)

        merged = Add()([state_h2, action_h1])
        output = Dense(1)(merged)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        #model.summary()
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
        target_actions = self.actor_target_model.predict(new_states)
        future_rewards = self.critic_target_model.predict([new_states, target_actions])#[0][0]
        for i in xrange(len(terminals)):
            if not terminals[i]:
                # R_t = r_t + gamma * R_{t+1:infty}
                rewards[i] += self.gamma * future_rewards[i][0]
        self.critic_model.fit([cur_states, actions], rewards, verbose=0)
        
    def train(self):
        if len(self.memory['cur_state']) < self.batch_size:
            return

        rewards = []
        indexes = random.sample(xrange(len(self.memory['cur_state'])), self.batch_size)
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
        actor_target_weights = self.actor_target_model.get_weights()
        
        for i in xrange(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.actor_target_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in xrange(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)     

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            while True:
                action = np.random.randn(1)[0]*self.max_random_action + self.max_random_action
                if action > 0:
                    return int(action)
        return int(self.actor_model.predict(cur_state)[0][0])


    def move(self, is_train):
        cur_state = self.env.get_state()
        tmp_state = np.reshape(cur_state, [1, cur_state.shape[0], cur_state.shape[1], 1])
        action = self.act(tmp_state)
        new_state, reward, terminal = self.env.step(action, is_train)
        cur_state = np.reshape(cur_state, [cur_state.shape[0], cur_state.shape[1], 1])
        new_state = np.reshape(new_state, [new_state.shape[0], new_state.shape[1], 1])
        return cur_state, action, reward, new_state, terminal

    @timeit
    def test(self, epochs):
        print('Testing ...')
        self.env.restart(is_train=False)
        test_step = test_reward = 0
        for epoch in xrange(epochs):
            terminal = False
            while not terminal:
                cur_state, action, reward, new_state, terminal = self.move(is_train=False)
                test_reward += reward
                test_step += 1
            #print('test epoch: {}/{} \t test step: {}'.format(epoch+1, epochs, test_step))
        #print('avg_test_reward:', test_reward/test_step)
        return test_step, test_reward, test_reward/test_step



class A2CNets(object):
    """docstring for A2CNets"""
    def __init__(self, scope_name):
        self.emb_dim = args.emb_dim
        self.n_stores = args.n_stores
        self.dense_dim = args.dense_dim
        self.batch_size = args.batch_size
        self.conv_layers = args.conv_layers
        self.maxp_dim = args.maxp_dim
        self.learning_rate = args.learning_rate

        with tf.variable_scope(scope_name):
            self.actor_state_input, self.actor_model = self.build_actor_network()
            self.critic_state_input, self.critic_action_input, self.critic_model = self.build_critic_network()

            critic_model_weights = self.critic_model.trainable_weights
            self.critic_action_grads = tf.gradients(self.critic_model.output, self.critic_action_input)
            self.critic_weight_grads = tf.gradients(self.critic_model.output, critic_model_weights)
            #self.grads = zip(self.critic_weight_grads, critic_model_weights)

            self.actor_critic_grad = tf.placeholder(tf.float32, [None, 1]) 
            actor_model_weights = self.actor_model.trainable_weights
            self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, self.actor_critic_grad)#-self.critic_action_grads[-1])
            #self.grads.extend(zip(self.actor_grads, actor_model_weights))
            self.grads = zip(self.actor_grads, actor_model_weights)

            self.optimize = tf.train.AdamOptimizer(args.learning_rate).apply_gradients(self.grads)
            


    def build_actor_network(self, name='actor'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        if self.conv_layers > 1:
            inputs = state_input
            for i in xrange(self.conv_layers):
                conv = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
                mp = MaxPooling2D((10, 1), strides=(10, 1), padding='valid')(conv)
                inputs = mp
        else:
            conv = Conv2D(32, (5, 5), strides=(2, 2), padding='valid', activation='relu')(state_input)
            mp = MaxPooling2D((self.maxp_dim, 1), strides=(self.maxp_dim, 1), padding='valid')(conv)
        flat = Flatten()(mp)
        state_h1 = Dense(self.dense_dim, activation='relu')(flat)
        output = Dense(1)(state_h1)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, model


    def build_critic_network(self, name='critic'):
        state_input = Input(shape=[self.n_stores, self.emb_dim, 1])
        if self.conv_layers > 1:
            inputs = state_input
            for i in xrange(self.conv_layers):
                conv = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
                mp = MaxPooling2D((10, 1), strides=(10, 1), padding='valid')(conv)
                inputs = mp
        else:
            conv = Conv2D(32, (5, 5), strides=(2, 2), padding='valid', activation='relu')(state_input)
            mp = MaxPooling2D((self.maxp_dim, 1), strides=(self.maxp_dim, 1), padding='valid')(conv)
        flat = Flatten()(mp)
        state_h1 = Dense(self.dense_dim, activation='relu')(flat)
        state_h2 = Dense(1, activation='relu')(state_h1)

        action_input = Input(shape=[1])
        action_h1 = Dense(1, activation='relu')(action_input)

        merged = Add()([state_h2, action_h1])
        output = Dense(1)(merged)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, action_input, model
        


class A2CSingleThread(object):
    """docstring for A2CSingleThread"""
    def __init__(self, data, thread_id, master):
        self.thread_id = thread_id
        self.env = Environment(args, data)
        self.master = master
        self.local_net = A2CNets('local_net%d'%thread_id)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.max_random_action = args.max_random_action
        #self.master.sess.run(tf.global_variables_initializer())
        


    def sync_network(self, sess, source_net):
        source_actor_weights  = source_net.actor_model.get_weights()
        local_actor_weights = self.local_net.actor_model.trainable_weights #get_weights()
        ops = []
        for i in xrange(len(source_actor_weights)):
            ops.append(tf.assign(local_actor_weights[i], source_actor_weights[i]))
        sess.run(ops)
        
        # for i in xrange(len(source_actor_weights)):
        #     local_actor_weights[i] = source_actor_weights[i]
        # self.local_net.actor_model.set_weights(local_actor_weights)

        source_critic_weights  = source_net.critic_model.get_weights()
        local_critic_weights = self.local_net.critic_model.trainable_weights #get_weights()
        ops = []
        for i in xrange(len(source_critic_weights)):
            ops.append(tf.assign(local_critic_weights[i], source_critic_weights[i]))
        sess.run(ops)
        
        # for i in xrange(len(source_actor_weights)):
        #     local_critic_weights[i] = source_critic_weights[i]
        # self.local_net.critic_model.set_weights(local_critic_weights)


    def act(self, cur_state):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon or args.cur_train_epoch == 0:
            while True:
                action = np.random.randn(1)[0]*self.max_random_action + self.max_random_action
                if action > 0:
                    return int(action)
        return int(self.local_net.actor_model.predict(cur_state)[0][0])


    def forward_explore(self):
        terminal = False
        train_step = 0
        states, actions, rewards = [], [], []
        while not terminal and train_step <= self.batch_size:
            cur_state = self.env.get_state()
            tmp_state = np.reshape(cur_state, [1, cur_state.shape[0], cur_state.shape[1], 1])
            action = self.act(tmp_state)
            _, reward, terminal = self.env.step(action, is_train=True)
            cur_state = np.reshape(cur_state, [cur_state.shape[0], cur_state.shape[1], 1])
            states.append(cur_state)
            actions.append(action)
            rewards.append(reward)
            train_step += 1
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        return states, actions, rewards, train_step-1 


    def train(self, sess, g):
        self.env.restart(is_train=True)
        while args.train_step <= args.max_train_steps:
            if args.cur_train_epoch > 0:
                self.sync_network(sess, self.master.shared_net)

            states, actions, rewards, train_step = self.forward_explore() # states[1:] are new states
            cur_states = states[:-1]
            new_states = states[1:]
            target_actions = self.master.shared_net.actor_model.predict(new_states)
            future_rewards = self.master.shared_net.critic_model.predict([new_states, target_actions])#[0][0]
            tmp_rewards = rewards[:-1]
            tmp_rewards += self.gamma * future_rewards[:, 0]
            while True:
                if not args.is_update:
                    args.is_update = True
                    break
            self.master.shared_net.critic_model.fit([new_states, actions[:-1]], tmp_rewards, verbose=0)

            predicted_actions = self.local_net.actor_model.predict(cur_states)
            grads = sess.run(self.local_net.critic_action_grads, feed_dict={
                                        self.local_net.critic_state_input:  cur_states,
                                        self.local_net.critic_action_input: predicted_actions,})[0]
                                        #self.local_net.actor_state_input:   cur_states})

            _, step = sess.run([self.master.shared_net.optimize, self.master.global_step], feed_dict={
                                self.master.shared_net.actor_state_input: cur_states,
                                self.master.shared_net.actor_critic_grad: grads})
            args.is_update = False

            args.train_step += train_step
            args.cur_train_epoch += 1
            print('Global epoch: {} Thread: {}  step: {}'.format(args.cur_train_epoch, self.thread_id, args.train_step))
            if args.cur_train_epoch % args.test_per_n_epochs == 0:
                test_step, test_reward, avg_test_reward = self.master.test(args.test_epochs)
                print('test_epochs {} \t test_steps {} \t avg_test_reward {}'.format(args.test_epochs, test_step, avg_test_reward))

    

class AsyActorCritic(object):
    """docstring for AsyActorCritic"""
    def __init__(self, sess, data):
        self.shared_net = A2CNets('global_net')
        self.env = Environment(args, data)
        self.sess = sess
        self.gamma = args.gamma
        self.epsilon = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.max_random_action = args.max_random_action
        self.jobs = []

        for thread_id in xrange(args.n_jobs):
            self.jobs.append(A2CSingleThread(data, thread_id, self))
        
        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
        self.sess.run(tf.global_variables_initializer())

    
    def act(self, cur_state):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            while True:
                action = np.random.randn(1)[0]*self.max_random_action + self.max_random_action
                if action > 0:
                    return int(action)
        return int(self.shared_net.actor_model.predict(cur_state)[0][0])


    def move(self, is_train):
        cur_state = self.env.get_state()
        tmp_state = np.reshape(cur_state, [1, cur_state.shape[0], cur_state.shape[1], 1])
        action = self.act(tmp_state)
        new_state, reward, terminal = self.env.step(action, is_train)
        cur_state = np.reshape(cur_state, [cur_state.shape[0], cur_state.shape[1], 1])
        new_state = np.reshape(new_state, [new_state.shape[0], new_state.shape[1], 1])
        return cur_state, action, reward, new_state, terminal

    @timeit
    def test(self, epochs):
        print('Testing ...')
        self.env.restart(is_train=False)
        test_step = test_reward = 0
        for epoch in xrange(epochs):
            terminal = False
            while not terminal:
                cur_state, action, reward, new_state, terminal = self.move(is_train=False)
                test_reward += reward
                test_step += 1
            #print('test epoch: {}/{} \t test step: {}'.format(epoch+1, epochs, test_step))
        #print('avg_test_reward:', test_reward/test_step)
        return test_step, test_reward, test_reward/test_step


    def _train(self, thread_id, g, sess):
        with g.as_default():
            self.jobs[thread_id].train(sess, g)


    def train(self):
        import threading
        args.is_update = False
        args.train_step = 0
        args.cur_train_epoch = 0
        # self.jobs[0].train()
        # self.jobs[1].train()
        g = tf.get_default_graph()
        threads = [threading.Thread(target=self._train, args=(i, g, self.sess)) for i in xrange(len(self.jobs))]
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads:
            thread.join()

@timeit
def test_Asyn():
    data = DateProcessing()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #ipdb.set_trace()
        K.set_session(sess)
        K.set_image_data_format('channels_last')
        try:
            model = AsyActorCritic(sess, data)
            model.train()
        except KeyboardInterrupt:
            print('\nManually stop the program !\n')

 
# tf.app.args.DEFINE_integer("max_train_steps", 1e4, "train max time step")
# tf.app.args.DEFINE_integer("test_per_n_epochs", 10, "train max time step")
# tf.app.args.DEFINE_integer("cur_train_epoch", 0, "train max time step")
# tf.app.args.DEFINE_integer("is_update", False, "train max time step")
# tf.app.args.DEFINE_integer("train_step", 0, "train step. unchanged")
# args = tf.app.args.FLAGS
def arg_init():
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument('-train_start_date',  type=str, default='2017-01-01')
    parser.add_argument('-train_end_date',    type=str, default='2017-12-31')
    parser.add_argument('-test_start_date',   type=str, default='2018-01-01')
    parser.add_argument('-test_end_date',     type=str, default='2018-03-31')
    parser.add_argument('-min_dates',         type=int, default=100)
    parser.add_argument('-min_stores',        type=int, default=10)
    parser.add_argument('-use_padding',       type=int, default=1)
    parser.add_argument('-random',            type=int, default=1)
    # network arguments
    parser.add_argument('-emb_dim',           type=int, default=73)
    parser.add_argument('-n_stores',          type=int, default=500)
    parser.add_argument('-conv_layers',       type=int, default=1)
    parser.add_argument('-dense_dim',         type=int, default=64)
    parser.add_argument('-maxp_dim',          type=int, default=50)
    parser.add_argument('-batch_size',        type=int, default=32)
    # agent arguments
    parser.add_argument('-gamma',             type=float, default=0.95)
    parser.add_argument('-epsilon_start',     type=float, default=1.0)
    parser.add_argument('-epsilon_end',       type=float, default=0.1)
    parser.add_argument('-epsilon_decay',     type=float, default=0.9975)
    parser.add_argument('-learning_rate',     type=float, default=0.0025)
    parser.add_argument('-mem_size',          type=int, default=5000)
    parser.add_argument('-max_random_action', type=int, default=70)
    # main arguments
    parser.add_argument('-n_jobs',            type=int, default=4)
    parser.add_argument('-target_steps',      type=int, default=100)
    parser.add_argument('-test_epochs',       type=int, default=50)
    parser.add_argument('-test_per_n_epochs', type=int, default=2)
    parser.add_argument('-train_step',        type=int, default=0)
    parser.add_argument('-max_train_steps',   type=int, default=1000000)
    parser.add_argument('-result_dir',        type=str, default='results/bs64_ly1_ms10_ns500_random1_test50_tep2_pad1_abs_reward')
    parser.add_argument('-gpu_rate',          type=float, default=0.25)
    return parser.parse_args()


def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
    with open('%s.txt'%args.result_dir,'w') as args.logger:
        for (k,v) in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
            args.logger.write('{}: {}\n'.format(k, v))
            print('{}: {}'.format(k, v))

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            #ipdb.set_trace()
            start = time.time()
            data = DateProcessing()
            env = Environment(args, data)
            model = ActorCritic(env, sess)

            try: # training
                log_steps, log_rewards, log_test_epochs, log_test_rewards = [], [], [], []
                total_reward = avg_reward = epoch = last_step = 0
                env.restart(is_train=True)
                for i in xrange(1, args.max_train_steps+1):
                    if i % 100 == 0:
                        print('total training step: %d' % i)
                    
                    cur_state, action, reward, new_state, terminal = model.move(is_train=True)
                    model.remember(cur_state, action, reward, new_state, terminal)
                    model.train()
                    total_reward += reward

                    if i % args.target_steps == 0:
                        model.update_target()
                        avg_reward = total_reward / args.target_steps
                        total_reward = 0
                        log_steps.append(i)
                        log_rewards.append(avg_reward)
                    
                    if terminal:
                        steps = i - last_step
                        last_step = i
                        epoch += 1
                        most_action = sorted(env.seen_actions.iteritems(), key=lambda x:x[1])[-1]
                        most_order = sorted(env.target_orders.iteritems(), key=lambda x:x[1])[-1]
                        #args.logger.write('epoch {} steps {} avg_reward {} most_order {} {} most_action {} {}\n'.format(
                        #                    epoch, steps, avg_reward, most_order[0], most_order[1], most_action[0], most_action[1]))
                        print('most_order: {}\tmost_action: {}'.format(most_order, most_action))
                        print('epoch {}\tsteps {}\tavg_reward: {:.2f}\n'.format(epoch, steps, avg_reward))

                        if epoch % args.test_per_n_epochs == 0:
                            test_step, test_reward, avg_test_reward = model.test(epochs=args.test_epochs)
                            print('test_epochs {} \t test_steps {} \t avg_test_reward {}'.format(args.test_epochs, test_step, avg_test_reward))
                            args.logger.write(
                                'epoch {:<4} train_step {:<6} test_step {:<6} avg_test_reward {:<10.2f} most_order {:<5} {:<5} most_action {:<5} {:<5}\n'.format(
                                epoch, i, test_step, avg_test_reward, most_order[0], most_order[1], most_action[0], most_action[1]))
                            log_test_epochs.append(epoch/args.test_per_n_epochs)
                            log_test_rewards.append(avg_test_reward)
                        # reset the environment
                        env.restart(is_train=True)
           
            except KeyboardInterrupt:
                print('\nManually stop the program !\n')

        actions = sorted(env.seen_actions.iteritems(), key=lambda x:x[1], reverse=True)[:100]
        orders = sorted(env.target_orders.iteritems(), key=lambda x:x[1], reverse=True)[:100]
        args.logger.write('\nsorted target actions and seen actions:\n')
        num = len(actions) if len(actions) <= len(orders) else len(orders)
        for i in range(num):
            args.logger.write('{:<5}: {:<5}\t{:<5}: {:<5}\n'.format(orders[i][0], orders[i][1], actions[i][0], actions[i][1]))
        plt.switch_backend('pdf')
        plt.subplot(211)
        plt.plot(log_steps, log_rewards)
        plt.subplot(212)
        plt.plot(log_test_epochs, log_test_rewards)
        plt.savefig('%s.pdf'%args.result_dir, format='pdf')
        end = time.time()
        args.logger.write('\nTime cost: %.2fs\n' % (end - start))
        print('\nTime cost: %.2fs\n' % (end - start))



if __name__ == '__main__':
    global args
    args = arg_init()
    main()
    #test_Asyn()