import ipdb
import keras
import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Bidirectional, LSTM, Masking
from keras.models import Model


class DeepQLearner:
    def __init__(self, args, agent_mode, embedding=[]):
        print('Initializing the DQN...')
        self.optimizer = args.optimizer
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate
        self.target_output = args.act_vocab_size + args.obj_vocab_size
        self.words_num = args.words_num
        self.word_dim = args.word_dim
        self.emb_dim = args.emb_dim
        self.dqn_mode = args.dqn_mode
        self.agent_mode = agent_mode
        self.reward_base = args.reward_base
        if agent_mode == 'dv':
            self.embedding = embedding
            self.emb_dim = self.target_output = embedding.shape[1]
            self.build_dqn()
        if self.dqn_mode == 'cnn':
            self.build_dqn()
        else:
            self.build_lstm()


    def build_lstm(self):
        hidden_size = [64, 32]
        dropout = [0,25, 0,25]
        inputs = Input(shape=(self.words_num, self.emb_dim))

        masked_inputs = inputs #Masking(mask_value=0.0)(inputs) # filter zero-value time steps
        lstm1 = Bidirectional(LSTM(hidden_size[0], activation='tanh', return_sequences=False, 
                        dropout_W=dropout[0], dropout_U=dropout[1]))(masked_inputs)
        #lstm2 = LSTM(hidden_size[1], activation='tanh', return_sequences=True, 
        #                dropout_W=dropout[0], dropout_U=dropout[1])(lstm1)
        out = Dense(self.target_output, kernel_initializer='truncated_normal')(lstm1)

        self.model = Model(inputs, out)
        self.target_model = Model(inputs, out)

        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=0.5, momentum=0.9, decay=0.9, nesterov=True)
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        else:
            opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)

        #def myloss(y_true, y_pred):
        #    return tf.reduce_sum(tf.square(y_true - y_pred))
        myloss = 'mse'
        self.model.compile(optimizer=opt, loss=myloss, metrics=['mse', 'acc'])
        self.target_model.compile(optimizer=opt, loss=myloss, metrics=['mse', 'acc'])

        print(self.model.summary())


    def build_dqn(self):
        # ipdb.set_trace()
        fw = self.emb_dim - 1  #filter width
        fn = 32  #filter num
        # try: height = self.words_num = actions amount of an action sequence, e.g. 20
        # try: width = self.emb_dim = act_dim + obj_dim, e.g. 10
        inputs = Input(shape=(self.words_num, self.emb_dim, 1))

        bi_gram = Conv2D(fn, (2, fw), padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        bi_gram = MaxPooling2D((self.words_num - 1, 1), strides=(1, 1), padding='valid')(bi_gram)

        tri_gram = Conv2D(fn, (3, fw), padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        tri_gram = MaxPooling2D((self.words_num - 2, 1), strides=(1, 1), padding='valid')(tri_gram)

        four_gram = Conv2D(fn, (4, fw), padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        four_gram = MaxPooling2D((self.words_num - 3, 1), strides=(1, 1), padding='valid')(four_gram)

        five_gram = Conv2D(fn, (5, fw), padding='valid', activation='relu', kernel_initializer='glorot_normal')(inputs)
        five_gram = MaxPooling2D((self.words_num - 4, 1), strides=(1, 1), padding='valid')(five_gram)

        # concates.shape = [None, 1, 8, 32]
        concate = kl.concatenate([bi_gram, tri_gram, four_gram, five_gram], axis=2)
        flat = Flatten()(concate)

        full_con = Dense(256, activation='relu', kernel_initializer='truncated_normal')(flat)
        out = Dense(self.target_output, kernel_initializer='truncated_normal')(full_con)

        self.model = Model(inputs, out)
        self.target_model = Model(inputs, out)

        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=0.5, momentum=0.9, decay=0.9, nesterov=True)
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        else:
            opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06)

        #def myloss(y_true, y_pred):
        #    return tf.reduce_sum(tf.square(y_true - y_pred))
        myloss = 'mse'
        self.model.compile(optimizer=opt, loss=myloss, metrics=['mse', 'acc'])
        self.target_model.compile(optimizer=opt, loss=myloss, metrics=['mse', 'acc'])

        print(self.model.summary()) 


    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self, minibatch):
        # expand components of minibatch
        # channel_first
        prestates, actions, rewards, poststates, terminals = minibatch
        
        if self.agent_mode == 'dv':
            prestates = np.matmul(prestates, self.embedding)
            poststates = np.matmul(poststates, self.embedding)
        if self.dqn_mode == 'cnn':
            post_input = np.reshape(poststates, [-1, self.words_num, self.emb_dim, 1])
        postq = self.target_model.predict_on_batch(post_input)
        assert postq.shape == (self.batch_size, self.target_output)
        
        # calculate max Q-value for each poststate  
        maxpostq = np.max(postq, axis=1)
        assert maxpostq.shape == (self.batch_size,)
        
        if self.dqn_mode == 'cnn':
            pre_input = np.reshape(prestates, [-1, self.words_num, self.emb_dim, 1])
        preq = self.model.predict_on_batch(pre_input)
        assert preq.shape == (self.batch_size, self.target_output)
        
        # make copy of prestate Q-values as targets  
        targets = preq.copy()

        #ipdb.set_trace()
        # update Q-value targets for actions taken  
        for i, action in enumerate(actions):
            # minimize all negative samples probability
            #if rewards[i] >= self.reward_base:
                #ipdb.set_trace()
            #    targets[i, :] = np.min(targets[i])
            if terminals[i]:  
                targets[i, action] = float(rewards[i])
            else:  
                targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        self.model.train_on_batch(pre_input, targets)


    def predict(self, current_state):
        if self.agent_mode == 'dv':
            current_state = np.matmul(current_state, self.embedding)
        if self.dqn_mode == 'cnn':
            state_input = np.reshape(current_state, [1, self.words_num, self.emb_dim, 1])

        qvalues = self.model.predict_on_batch(state_input)
        #ipdb.set_trace()
        return qvalues


    def save_weights(self, weight_dir):
        self.model.save_weights(weight_dir+'.h5')
        print('Saved weights to %s ...' % weight_dir)


    def load_weights(self, weight_dir):
        self.model.load_weights(weight_dir+'.h5')
        print('Loaded weights from %s ...' % weight_dir)




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

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    #set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    net = DeepQLearner(args)














