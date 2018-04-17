import os
import time
import ipdb
import keras
import pickle
import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Bidirectional, LSTM, Masking
from keras.callbacks import EarlyStopping
from keras.models import Model
from gensim.models import KeyedVectors
from tqdm import tqdm
from utils import ten_fold_split_idx, index2data

class DynamicVocab:
    def __init__(self, args, sess):
        print('Initializing the DynamicVocab...')
        self.sess = sess
        self.cnn_format = 'NCHW' # for GPU
        self.multi_cnn = args.multi_cnn
        self.add_linear = args.add_linear
        self.use_k_max_pool = args.use_k_max_pool
        self.fold_id = args.fold_id
        self.weight_dir = args.weight_dir
        self.words_num = args.words_num
        self.word_dim = args.word_dim
        self.optimizer = args.optimizer
        self.current_lr = args.learning_rate
        self.decay_rate = args.decay_rate
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.verbose = args.verbose
        self.data_name = args.data_name
        self.word2vec = args.word2vec
        self.ten_fold_indices = args.ten_fold_indices
        #self.data, self.targets = self.read_data()
        #self.build_cnn()
        self.folds = self.read_data()
        self.build_tf_cnn()


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



    def build_tf_cnn(self):
        fw = self.emb_dim - 1  #filter width
        ccs =  1  #convolution column stride
        fn = 32  #filter num
        self.w = {}
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        #initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        self.inputs = tf.placeholder(tf.float32, [None, self.words_num, self.word_dim], name='input')
        self.embeddings = tf.placeholder(tf.float32, [self.word_dim, self.emb_dim], name='embeddings')
        self.targets = tf.placeholder(tf.int32, [None, self.emb_dim], name='target')

        self.att_inputs = tf.matmul(tf.reshape(self.inputs, [-1, self.word_dim]), self.embeddings)
        self.att_inputs = tf.reshape(self.att_inputs, [-1, 1, self.words_num, self.emb_dim])

        self.l1, self.w['l1_w'], self.w['l1_b'] = self.conv2d(self.att_inputs,
            fn, [2, fw], [1, ccs], initializer, activation_fn, name='l1')
        self.l3, self.w['l3_w'], self.w['l3_b'] = self.conv2d(self.att_inputs,
            fn, [3, fw], [1, ccs], initializer, activation_fn, name='l3')
        self.l5, self.w['l5_w'], self.w['l5_b'] = self.conv2d(self.att_inputs,
            fn, [4, fw], [1, ccs], initializer, activation_fn, name='l5')
        self.l7, self.w['l7_w'], self.w['l7_b'] = self.conv2d(self.att_inputs,
            fn, [5, fw], [1, ccs], initializer, activation_fn, name='l7')

        if self.multi_cnn:
            tmpl1, self.w['tmp_l1_w'], self.w['tmp_l1_b'] = self.conv2d(self.l1,
                fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l1')
            tmpl3, self.w['tmp_l3_w'], self.w['tmp_l3_b'] = self.conv2d(self.l3,
                fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l3')
            tmpl5, self.w['tmp_l5_w'], self.w['tmp_l5_b'] = self.conv2d(self.l5,
                fn, [2, fw], [1, ccs], initializer, activation_fn, name='tmp_l5')
            tmpl7, self.w['tmp_l7_w'], self.w['tmp_l7_b'] = self.conv2d(self.l7,
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
            self.l10, self.w['l10_w'], self.w['l10_b'] = self.linear(
                                self.l9_flat, 256, activation_fn=activation_fn, name='l10')
        else:
            self.l10 = self.l9_flat
        self.output, self.w['q_w'], self.w['q_b'] = self.linear(self.l10, self.target_output, name='output')
        self.z = tf.nn.softmax(self.output)

        #ipdb.set_trace()
        self.loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.targets))
        self.learning_rate = tf.Variable(self.current_lr)
        if self.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'adam':
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'adadelta':
            self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)
        else:
            self.train_op = tf.train.RMSPropOptimizer(
                self.learning_rate, decay=self.decay_rate, momentum=self.momentum).minimize(self.loss)
    
        tf.global_variables_initializer().run()



    def build_cnn(self):
        # ipdb.set_trace()
        fw = self.emb_dim - 1  #filter width
        fn = 32  #filter num

        input_words = Input(shape=(self.words_num, self.emb_dim, 1))

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
        out = Dense(self.target_output, activation='softmax', kernel_initializer='truncated_normal')(full_con)

        if self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=0.5, momentum=0.9, decay=0.9, nesterov=True)
        elif self.optimizer == 'adam':
            opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        else:
            opt = keras.optimizers.RMSprop(lr=0.0025, rho=0.9, epsilon=1e-06)

        self.model = Model(inputs, out)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())


    def train(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(self.data, self.targets, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                        callbacks=[early_stop], validation_split=0.2, shuffle=True, initial_epoch=0)


    def run_epoch(self, data, train_flag, outfile):
        ipdb.set_trace()
        N = len(data['inputs']) / self.batch_size
        M = len(data['inputs']) % self.batch_size
        if M > 0:
            N += 1

        rec = pre = f1 = losses = 0.0
        right = tagged = 0
        for idx in xrange(N):
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            x = data['inputs'][start_idx: end_idx]
            y = data['targets'][start_idx: end_idx]

            if train_flag:
                _, loss, pred = self.sess.run([self.train_op, self.loss, self.z],
                                            feed_dict={
                                                self.embeddings: self.embedding,
                                                self.inputs: x,
                                                self.targets: y})

            else:
                loss, pred = self.sess.run([self.loss, self.z],
                                            feed_dict={
                                                self.embeddings: self.embedding,
                                                self.inputs: x,
                                                self.targets: y})
            losses += loss
            total += sum(targets)
            for b in xrange(self.batch_size):
                for i in xrange(self.emb_dim):
                    if pred[b][i] == 1:
                        tagged += 1
                        if y[b][i] == 1:
                            right += 1
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2 * rec * pre / (rec + pre)
        return f1, losses / N


    def run(self, outfile):
        last_f1 = 0.0
        best_f1 = 0.0
        best_epo = -1
        log_idx = 0
        log_f1 = []
        self.log_loss = []
        train_data = self.folds['train'][self.fold_id]
        valid_data = self.folds['valid'][self.fold_id]
        for idx in range(self.epochs):
            train_f1, train_loss = self.run_epoch(train_data, 1, outfile)
            valid_f1, valid_loss = self.run_epoch(valid_data, 0, outfile)

            if best_f1 < valid_f1:
                best_f1 = valid_f1
                best_epo = idx
                log_idx += 1
                print('\nTry to save model, log_idx: %d\n' % log_idx)
                self.save_weights(self.weight_dir)
                last_f1 = best_f1
            self.log_loss.append(valid_loss)
            log_f1.append(valid_f1)
            display = {
                'epoch:': idx,
                'train_loss': train_loss,
                'train_f1': train_f1,
                'valid_loss': valid_loss,
                'valid_f1': valid_f1,
                'learning_rate': self.current_lr
            }
            print('{}\n'.format(display))
            outfile.write('\n{}\n'.format(display))
            if self.optimizer == 'sgd' and idx > self.start_decay_epoch:
                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx] > self.log_loss[idx-1] * 0.9999:
                    if self.current_lr > self.stop_lr:
                        self.current_lr *= self.decay #self.current_lr / 1.5
                        self.lr.assign(self.current_lr).eval()
            # stop training if no improvement in self.stop_epoch_gap epoch 
            if idx - best_epo > self.stop_epoch_gap and idx > 200: #train_loss < 1: # 
                break


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


    def read_data(self):
        print('Preparing datasets...')
        with open(self.data_name, 'r') as f:
            indata = pickle.load(f)
        
        #ipdb.set_trace()
        # build vocabulary
        self.all_act_words = []
        self.all_obj_words = []
        sents = []
        f = open('dv_sents.txt', 'w')
        for i in xrange(len(indata)):
            sent = []
            for act, obj_line in indata[i]['act_seq']:
                self.all_act_words.append(act)
                objs = obj_line.split('_')
                for obj in objs:
                    self.all_obj_words.append(obj)
                sent.append(act)
                sent.extend(objs)
            try:
                f.write('Sentence {}:\n{}\n'.format(i, ' '.join(sent)))
            except Exception as e:
                pass
            sents.append(sent)
        f.close()

        self.all_act_words = list(set(self.all_act_words))
        self.all_obj_words = list(set(self.all_obj_words))
        self.word2index = {}
        for w in self.all_act_words + self.all_obj_words:
            if w not in self.word2index:
                self.word2index[w] = len(self.word2index)
        self.index2word = {v:k for k,v in self.word2index.iteritems()}

        self.embedding = []
        for word in self.word2index:
            if word in self.word2vec.vocab:
                self.embedding.append(self.word2vec[word])
            else:
                self.embedding.append(np.zeros(self.word_dim))
        self.embedding = np.array(self.embedding).T
        self.emb_dim = self.target_output = self.embedding.shape[1]

        #data_matrix = []
        #targets = []
        data = {'inputs': [], 'targets': []}
        for i in xrange(len(sents)):
            tmp_vec = []
            tmp_trg = np.zeros(len(self.word2index), dtype=np.int32)
            for word in sents[i]:
                tmp_trg[self.word2index[word]] = 1
                if word in self.word2vec.vocab:
                    tmp_vec.append(self.word2vec[word])
                else:
                    tmp_vec.append(np.zeros(self.word_dim))
            if len(tmp_vec) < self.words_num:
                for j in xrange(self.words_num - len(tmp_vec)):
                    tmp_vec.append(np.zeros(self.word_dim))
            else:
                tmp_vec = tmp_vec[: self.words_num]
            #att_vec = np.matmul(np.array(tmp_vec), self.embedding)
            #data_matrix.append(tmp_vec)
            #targets.append(tmp_trg)
            data['inputs'].append(np.array(tmp_vec))
            data['targets'].append(np.array(tmp_trg))

        indices = ten_fold_split_idx(len(data['inputs']), self.ten_fold_indices)
        folds = index2data(indices, data)
        return folds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--learning_rate", type=float, default=0.0025, help="")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="")
    parser.add_argument("--optimizer", type=str, default='rmsprop', help="")
    parser.add_argument("--momentum", type=float, default=0.8, help="")
    parser.add_argument("--words_num", type=int, default=200, help="")
    parser.add_argument("--word_dim", type=int, default=50, help="")
    parser.add_argument("--verbose", type=int, default=2, help="")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument("--data_name", type=str, default='/home/fengwf/Documents/DRL_data/wikihow/wikihow_act_seq_15.pkl', help='')
    parser.add_argument("--weight_dir", type=str, default='', help="")
    parser.add_argument("--gpu_rate", type=float, default=0.13, help="")
    parser.add_argument("--fold_id", type=int, default=0, help="")
    parser.add_argument("--stop_epoch_gap", type=int, default=10, help="")
    parser.add_argument("--ten_fold_indices", type=str, default='data/test_dv_ten_fold_indices.pkl', help="")
    parser.add_argument("--num_k", type=int, default=2, help="")
    parser.add_argument("--multi_cnn", type=int, default=0, help="")
    parser.add_argument("--use_k_max_pool", type=int, default=0, help="")
    parser.add_argument("--add_linear", type=int, default=1, help="")

    args = parser.parse_args()
    model_dir = '/home/fengwf/Documents/DRL_data/wikihow/wikihow_model_50_5'
    args.word2vec = KeyedVectors.load_word2vec_format(model_dir, binary=True)

    with open('results/test_dv_result.txt', 'w') as outfile:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = DynamicVocab(args, sess)
            model.run(outfile)