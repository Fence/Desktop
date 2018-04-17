import ipdb
import time
import argparse
import numpy as np
import tensorflow as tf
import keras.layers as kl
from keras.layers import Input, Embedding, Conv2D, MaxPooling2D
from keras.layers import Lambda, Flatten, Dense, Dropout, Activation
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

from data_generator import prepare_data
from disan import disan



class EmotionClassifier(object):
    """docstring for EmotionClassifier"""
    def __init__(self, args):
        self.sent_len = args.sent_len
        self.word_dim = args.word_dim
        self.vocab_size = args.vocab_size
        self.outfile = args.outfile
        self.dropout = args.dropout
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.dense_dim = args.dense_dim
        self.model_type = args.model_type
        self.use_emb = args.use_emb
        self.epochs = args.epochs
        self.patience = args.patience
        self.mini_count = args.mini_count
        self.num_sample = args.num_sample
        self.num_filter = args.num_filter
        self.num_gram = args.num_gram


    def build_cnn(self):
        #ipdb.set_trace()
        fw = self.word_dim
        fn = self.num_filter
        if self.use_emb:
            inputs = Input(shape=(self.sent_len, self.word_dim), dtype='float32')
            word_emb = kl.core.Reshape((self.sent_len, self.word_dim, 1))(inputs)
        else:
            inputs = Input(shape=(self.sent_len, ), dtype='int32')
            word_emb = Embedding(input_dim=self.vocab_size, output_dim=self.word_dim, 
                                    input_length=self.sent_len)(inputs)
            word_emb = kl.core.Reshape((self.sent_len, self.word_dim, 1))(word_emb)
        n_gram_list = []
        for i in range(2, self.num_gram+1):
            #n_gram = Conv2D(fn, (i, fw), padding='valid', activation='relu')(word_emb)
            conv = Conv2D(fn, (i, fw), padding='valid')(word_emb)
            bn = BatchNormalization()(conv)
            atv = Activation(activation='relu')(bn)
            n_gram = MaxPooling2D((self.sent_len-i+1, 1), strides=(1, 1), padding='valid')(atv)
            n_gram = Dropout(self.dropout)(n_gram)
            n_gram_list.append(n_gram)

        concate = kl.Concatenate(axis=2)(n_gram_list)
        flat = Flatten()(concate)

        full_con = Dense(self.dense_dim)(flat)
        bn_full = BatchNormalization()(full_con)
        atv_full = Activation(activation='relu')(bn_full)
        atv_full = Dropout(self.dropout)(atv_full)
        out = Dense(6, activation='softmax')(atv_full)

        self.model = Model(inputs, out)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary() 


    def build_lstm(self):
        #ipdb.set_trace()
        if self.use_emb:
            inputs = Input(shape=(self.sent_len, self.word_dim), dtype='float32')
            word_emb = inputs
        else:
            inputs = Input(shape=(self.sent_len,), dtype='int32')
            word_emb = Embedding(input_dim=self.vocab_size, output_dim=self.word_dim, 
                                input_length=self.sent_len)(inputs)

        #bi_lstm = Bidirectional(LSTM(self.word_dim, dropout=self.dropout))(word_emb)
        lstm = LSTM(self.word_dim, dropout=self.dropout)(word_emb)
        out = Dense(6, activation='softmax')(lstm)

        self.model = Model(inputs, out)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()


    def build_disan(self):
        if self.use_emb:
            inputs = Input(shape=(self.sent_len, self.word_dim), dtype='float32')
            word_emb = inputs
        else:
            inputs = Input(shape=(self.sent_len,), dtype='int32')
            word_emb = Embedding(input_dim=self.vocab_size, output_dim=self.word_dim, 
                                input_length=self.sent_len)(inputs)
        
        sent_token = Input(shape=(self.sent_len, ), dtype='int32')
        sent_mask = tf.cast(sent_token, tf.bool)
        sent_rep = Lambda(disan(word_emb, sent_mask, scope='DiSAN', keep_prob=self.dropout, is_train=True, wd=0.,
                activation='elu', tensor_dict=None, name=''))
        out = Dense(6, activation='softmax')(sent_rep)

        ipdb.set_trace()
        self.model = Model([inputs, sent_token], out)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.model.summary()



    def train(self):
        #ipdb.set_trace()
        model_name = '../weibo_data/model_%d_%d' % (self.mini_count, self.word_dim) 
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        vocab_size, sents, labels, lens, tsents, tlabels, tlens = prepare_data("./data/original_train_data0306", 
            self.sent_len, self.use_emb, model_name, self.num_sample, "./data/vocabulary", 40000, None) 

        self.vocab_size = vocab_size
        #self.print_obj()
        if self.model_type == 'cnn':
            self.build_cnn()
        elif self.model_type == 'disan':
            self.build_disan()
        else:
            self.build_lstm()
        if self.use_emb:
            sents = np.array(sents)
            tsents = np.array(tsents)
            lens = np.array(lens)
            tlens = np.array(tlens)
        if self.model_type == 'disan':
            hist = self.model.fit([sents, lens], labels, batch_size=self.batch_size, epochs=self.epochs,
                verbose=1, callbacks=[early_stopping], validation_split=0.2, shuffle=False)
            pred = self.model.predict([tsents, tlens], batch_size=self.batch_size, verbose=2)
        else:
            hist = self.model.fit(sents, labels, batch_size=self.batch_size, epochs=self.epochs,
                verbose=1, callbacks=[early_stopping], validation_split=0.2, shuffle=False)

            pred = self.model.predict(tsents, batch_size=self.batch_size, verbose=2)
        #ipdb.set_trace()
        #print(pred.shape)
        indexes = np.argmax(pred, axis=1)
        right = [0 for _ in range(6)]
        total = sum(tlabels)
        tagged = [0 for _ in range(6)]
        for i in range(len(tlabels)):
            if tlabels[i][indexes[i]] == 1:
                right[indexes[i]] += 1
            tagged[indexes[i]] += 1
        self.outfile.write('\nright\ttotal\ttagged\n')
        for i in range(6):
            rec, pre, f1 = self.compute_f1(right[i], total[i], tagged[i])
            self.outfile.write('{}\t{}\t{}\n'.format(right[i], total[i], tagged[i]))
            self.outfile.write('{}\t{}\t{}\n'.format(rec, pre, f1))
            print(right[i], total[i], tagged[i])
            print(rec, pre, f1)
        print('\n{}\n'.format(sum(right)/sum(total)))
        self.outfile.write('\n{}\n'.format(sum(right)/sum(total)))

    def print_obj(self):
        print('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))


    def compute_f1(self, right, total, tagged):
        f1 = pre = rec = 0
        if(total > 0):
            rec = right / total
        if(tagged > 0):
            pre = right / tagged
        if(pre + rec > 0):
            f1 = 2*pre*rec / (pre + rec)
        return rec, pre, f1






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini_count", type=int, default=10, help="")
    parser.add_argument("--use_emb", type=bool, default=True, help="")

    parser.add_argument("--sent_len", type=int, default=100, help="")
    parser.add_argument("--word_dim", type=int, default=200, help="")
    parser.add_argument("--vocab_size", type=int, default=40000, help="")
    parser.add_argument("--num_sample", type=int, default=4000, help="")

    parser.add_argument("--num_gram", type=int, default=5, help="")
    parser.add_argument("--num_filter", type=int, default=32, help="")
    parser.add_argument("--dense_dim", type=int, default=64, help="")
    parser.add_argument("--dropout", type=float, default=0.5, help="")
    
    parser.add_argument("--patience", type=int, default=10, help="")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--optimizer", type=str, default='adam', help="")
    parser.add_argument("--model_type", type=str, default='cnn', help="")
    args = parser.parse_args()
    start = time.time()
    #for var in ['adam', 'sgd', 'rmsprop']:
    for dp in [0.25, 0.5]:
        for ns in [3000, 4000]:
            #args.batch_size = bs
            args.dropout = dp
            args.num_sample = ns
            with open('results/test200_nf%d_ng%d_sf1_ns%d_bs%d_drop%.2f_%s_len%d_dim%s.txt'%(args.num_filter, args.num_gram, args.num_sample, 
                args.batch_size, args.dropout, args.model_type, args.sent_len, args.word_dim), 'w') as args.outfile:
                for k in args.__dict__:
                    print(k, args.__dict__[k])
                    args.outfile.write('{}:{}\n'.format(k, args.__dict__[k]))
                model = EmotionClassifier(args)
                #model.build_lstm()     
                model.train() 
                #tf.reset_default_graph()    
    end = time.time()
    print('Total time cost: %.2fs\n' % (end - start))     

