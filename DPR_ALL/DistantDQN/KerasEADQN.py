import ipdb
import keras
import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.models import Model
from keras.layers.normalization import BatchNormalization


class DeepQLearner:
    def __init__(self, args, agent_name):
        print('Initializing the DQN...')
        self.word_dim = args.word_dim
        self.gram_num = args.gram_num
        self.optimizer = args.optimizer
        self.dropout = args.dropout
        self.filter_num = args.filter_num
        self.dense_dim = args.dense_dim
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.learning_rate = args.learning_rate
        self.target_output = args.target_output
        
        self.dqn_mode = args.dqn_mode
        self.agent_mode = args.agent_mode
        if agent_name == 'act':
            self.words_num = args.words_num
            self.emb_dim = args.emb_dim
        elif agent_name == 'obj':
            self.words_num = args.context_len
            self.emb_dim = args.emb_dim
            # if args.append_distance:
            #     self.emb_dim += args.dis_dim
        if self.dqn_mode == 'lstm':
            self.build_lstm()
        elif self.dqn_mode == 'cnn2':
            self.build_cnn2()
        else:
            self.build_dqn()


    def build_lstm(self):
        merge_first = 1
        tmp_tag_dim = 50
        hidden_size = [64, 32]
        dropout = [0,25, 0,25]
        #inputs = Input(shape=(self.words_num, self.emb_dim))
        input_words = Input(shape=(self.words_num, self.word_dim))
        input_tags = Input(shape=(self.words_num, ))
        input_idx = Input(shape=(1, ))

        #masked_inputs = Masking(mask_value=0.0)(inputs) # filter zero-value time steps
        embedded_tags = Embedding(input_dim=3, output_dim=tmp_tag_dim, trainable=True)(input_tags)
        if merge_first:
            merged = concatenate([input_words, embedded_tags], axis=-1)
            lstm = Bidirectional(LSTM(hidden_size[0], return_sequences=True, 
                        dropout_W=dropout[0], dropout_U=dropout[1]))(merged)
            out = TimeDistributed(Dense(self.target_output, activation=None))(lstm)
            #out = Dense(self.target_output)(concatenate(sliced_lstm, axis=0))
        else:
            lstm1 = Bidirectional(LSTM(tmp_tag_dim, return_sequences=True, 
                            dropout_W=dropout[0], dropout_U=dropout[1]))(embedded_tags)
            lstm2 = Bidirectional(LSTM(hidden_size[1], return_sequences=True, 
                            dropout_W=dropout[0], dropout_U=dropout[1]))(input_words)
            merged = concatenate([lstm2, lstm1], axis=-1)
            out = TimeDistributed(Dense(self.target_output, activation=None))(merged)

        self.model = Model([input_words, input_tags], out) #Model(inputs, out)
        self.target_model = Model([input_words, input_tags], out) #Model(inputs, out)
        self.compile_model()


    def build_cnn2(self):
        #ipdb.set_trace()
        fw = self.emb_dim - 1  #filter width
        fn = self.filter_num  #filter num
        ngrams = []
        inputs = Input(shape=(self.words_num, self.emb_dim, 1))

        for i in xrange(2, self.gram_num+1):
            conv = Conv2D(fn, (i, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
            #conv = BatchNormalization()(conv)
            atv = Activation(activation='relu')(conv)
            tmp_gram = MaxPooling2D((self.words_num-i+1, 1), strides=(1, 1), padding='valid')(atv)
            #tmp_gram = Dropout(self.dropout)(tmp_gram)
            #tmp_gram = Conv2D(fn, (i + 1, fw), activation='relu', kernel_initializer='glorot_normal')(inputs)
            #tmp_gram = MaxPooling2D((self.words_num - i, 1), strides=(1, 1))(tmp_gram)
            ngrams.append(tmp_gram)

        # concates.shape = [None, 1, 2*num, fn]
        concate = kl.concatenate(ngrams, axis=2)
        flat = Flatten()(concate)
        flat = Dropout(self.dropout)(flat)

        full_con = Dense(self.dense_dim, kernel_initializer='truncated_normal')(flat)
        #full_con = BatchNormalization()(full_con)
        atv_full = Activation(activation='relu')(full_con)
        #atv_full = Dropout(self.dropout)(atv_full)
        out = Dense(self.target_output, kernel_initializer='truncated_normal')(atv_full)

        self.model = Model(inputs, out)
        self.target_model = Model(inputs, out)
        self.compile_model()


    def build_dqn(self):
        #ipdb.set_trace()
        fw = self.emb_dim - 1  #filter width
        fn = self.filter_num  #filter num
        inputs = Input(shape=(self.words_num, self.emb_dim, 1))

        bi_gram = Conv2D(fn, (2, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #bi_gram = BatchNormalization()(bi_gram)
        bi_gram = Activation(activation='relu')(bi_gram)
        bi_gram = MaxPooling2D((self.words_num - 1, 1), strides=(1, 1), padding='valid')(bi_gram)

        tri_gram = Conv2D(fn, (3, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #tri_gram = BatchNormalization()(tri_gram)
        tri_gram = Activation(activation='relu')(tri_gram)
        tri_gram = MaxPooling2D((self.words_num - 2, 1), strides=(1, 1), padding='valid')(tri_gram)

        four_gram = Conv2D(fn, (4, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #four_gram = BatchNormalization()(four_gram)
        four_gram = Activation(activation='relu')(four_gram)
        four_gram = MaxPooling2D((self.words_num - 3, 1), strides=(1, 1), padding='valid')(four_gram)

        five_gram = Conv2D(fn, (5, fw), padding='valid', kernel_initializer='glorot_normal')(inputs)
        #five_gram = BatchNormalization()(five_gram)
        five_gram = Activation(activation='relu')(five_gram)
        five_gram = MaxPooling2D((self.words_num - 4, 1), strides=(1, 1), padding='valid')(five_gram)

        # concates.shape = [None, 1, 8, 32]
        concate = kl.concatenate([bi_gram, tri_gram, four_gram, five_gram], axis=2)
        flat = Flatten()(concate)

        full_con = Dense(self.dense_dim, activation='relu', kernel_initializer='truncated_normal')(flat)
        out = Dense(self.target_output, kernel_initializer='truncated_normal')(full_con)

        self.model = Model(inputs, out)
        self.target_model = Model(inputs, out)
        self.compile_model()


    def compile_model(self):
        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.9, nesterov=True)
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
        self.model.compile(optimizer=opt, loss=myloss)
        self.target_model.compile(optimizer=opt, loss=myloss)

        if self.agent_mode != 'multi':
            print(self.model.summary())


    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())


    def train(self, minibatch):
        # expand components of minibatch
        # channel_first
        prestates, actions, rewards, poststates, terminals = minibatch
        
        if self.dqn_mode == 'lstm':
            post_input_words = poststates[:, :, :self.word_dim]
            post_input_tags = poststates[:, :, -1]
            post_input = [post_input_words, post_input_tags]
            postq = self.target_model.predict_on_batch(post_input)
            #assert postq.shape == (self.batch_size, self.words_num, self.target_output)
        
            pre_input_words = prestates[:, :, :self.word_dim]
            pre_input_tags = prestates[:, :, -1]
            pre_input = [pre_input_words, pre_input_tags]
            preq = self.model.predict_on_batch(pre_input)
            #assert preq.shape == (self.batch_size, self.words_num, self.target_output)
 
            targets = preq.copy()
            maxpostq = np.max(postq, axis=2)
            for i, action in enumerate(actions):
                word_idx = action / 2
                act_idx = action % 2
                if terminals[i]:  
                    targets[i, word_idx, act_idx] = float(rewards[i])
                else:  
                    targets[i, word_idx, act_idx] = float(rewards[i]) + self.discount_rate * maxpostq[i, word_idx]
        
        else:
            post_input = np.reshape(poststates, [-1, self.words_num, self.emb_dim, 1])
            postq = self.target_model.predict_on_batch(post_input)
            #assert postq.shape == (self.batch_size, self.target_output)
        
            pre_input = np.reshape(prestates, [-1, self.words_num, self.emb_dim, 1])
            preq = self.model.predict_on_batch(pre_input)
            #assert preq.shape == (self.batch_size, self.target_output)
        
            # make copy of prestate Q-values as targets  
            targets = preq.copy()
            # calculate max Q-value for each poststate  
            maxpostq = np.max(postq, axis=1)
            #assert maxpostq.shape == (self.batch_size,)
            #ipdb.set_trace()
            # update Q-value targets for actions taken  
            for i, action in enumerate(actions):
                if terminals[i]:  
                    targets[i, action] = float(rewards[i])
                else:  
                    targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        self.model.train_on_batch(pre_input, targets)


    def predict(self, current_state):
        if self.dqn_mode == 'lstm':
            current_state = np.reshape(current_state, [1, self.words_num, self.emb_dim])
            state_input_words = current_state[:, :, :self.word_dim]
            state_input_tags = current_state[:, :, -1]
            state_input = [state_input_words, state_input_tags]
        else:
            state_input = np.reshape(current_state, [1, self.words_num, self.emb_dim, 1])

        qvalues = self.model.predict_on_batch(state_input)
        #ipdb.set_trace()
        return qvalues


    def save_weights(self, weight_dir):
        self.model.save_weights(weight_dir)
        print('Saved weights to %s ...' % weight_dir)


    def load_weights(self, weight_dir):
        self.model.load_weights(weight_dir)
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
    parser.add_argument("--dqn_mode", type=str, default='lstm', help="")
    parser.add_argument("--agent_mode", type=str, default='act', help="")

    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    net = DeepQLearner(args, 'act')