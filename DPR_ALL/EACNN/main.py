#coding:utf-8
import re, os, time, ipdb
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gensim.models import KeyedVectors
from keras import backend as K
from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Dense, Flatten, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from utils import ten_fold_split_idx, index2data

class ActionClassifier:
    def __init__(self, args):
        self.emb_dim = args.emb_dim
        self.output_dim = args.output_dim
        self.words_num = args.words_num
        self.epochs = args.epochs
        self.verbose = args.verbose
        self.optimizer = args.optimizer
        self.valid_rate = args.valid_rate
        self.batch_size = args.batch_size
        self.context_num = args.context_num
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=1)
      

    def build_model_cnn(self, jj):
        print '\n-----Building model %d...' % jj
        n_gram_flag = 0
        model_shapes = []
        input_shape = (1, self.words_num, self.emb_dim)
        model = Sequential()
        if n_gram_flag:
            model.add(Convolution2D(32, 2, self.emb_dim, border_mode='valid', input_shape=input_shape))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 4, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 5, 1, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(MaxPooling2D(pool_size=(self.words_num - 10, 1)))
            model_shapes.append(model.output_shape)
        else:
            model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 3, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(Convolution2D(32, 3, 3, border_mode='valid'))
            model.add(Activation('relu'))
            model_shapes.append(model.output_shape)
            model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
            model_shapes.append(model.output_shape)

        model.add(Flatten())
        model_shapes.append(model.output_shape)

        model.add(Dense(256, activation='relu'))
        model_shapes.append(model.output_shape)

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['acc'])

        if jj == 0:
            for s in model_shapes:
                print s
        self.model =  model


    def build_merged_cnn(self, jj):
        print '\n-----Building model %d...\n' % jj
        self.filter_num = 32
        input_texts = (1, self.words_num, self.emb_dim)
        bi_gram = Sequential()
        bi_gram.add(Convolution2D(self.filter_num, 2, self.emb_dim - 1, 
                        border_mode='valid', input_shape=input_texts))
        bi_gram.add(Activation('relu'))
        bi_gram.add(MaxPooling2D(pool_size=(self.words_num - 1, 1)))

        tri_gram = Sequential()
        tri_gram.add(Convolution2D(self.filter_num, 3, self.emb_dim - 1, 
                        border_mode='valid', input_shape=input_texts))
        tri_gram.add(Activation('relu'))
        tri_gram.add(MaxPooling2D(pool_size=(self.words_num - 2, 1)))

        four_gram = Sequential()
        four_gram.add(Convolution2D(self.filter_num, 4, self.emb_dim - 1, 
                        border_mode='valid', input_shape=input_texts))
        four_gram.add(Activation('relu'))
        four_gram.add(MaxPooling2D(pool_size=(self.words_num - 3, 1)))

        five_gram = Sequential()
        five_gram.add(Convolution2D(self.filter_num, 5, self.emb_dim - 1, 
                        border_mode='valid', input_shape=input_texts))
        five_gram.add(Activation('relu'))
        five_gram.add(MaxPooling2D(pool_size=(self.words_num - 4, 1)))

        merged = Merge([bi_gram, tri_gram, four_gram, five_gram], mode='concat')

        model = Sequential()
        model.add(merged)
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['acc'])

        self.model = model


    def build_model_mlp(self, jj):
        #print('\n-----Building model %d...' % jj)
        model = Sequential()
        model.add(Dense(self.emb_dim, input_dim=self.emb_dim, init='uniform', activation='relu'))
        model.add(Dense(self.output_dim))
        model.add(Activation('sigmoid'))
        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
        if jj == 0:
            print(model.summary())
        self.model = model


    def train_one(self, x, label):
        return self.model.fit(x, label, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, 
            callbacks=[self.early_stopping], validation_split=self.valid_rate, shuffle=True)



    def test_one(self, test_x):
        return self.model.predict_classes(test_x, batch_size=self.batch_size, verbose=self.verbose)


    def save_weights(self, weight_dir):
        self.model.save_weights(weight_dir)
        #print('Saved weights to %s ...' % weight_dir)


    def load_weights(self, weight_dir):
        self.model.load_weights(weight_dir)
        #print('Loaded weights from %s ...' % weight_dir)

 
    
def record_history(hist, outfile, pred, tokens, labels):
    for i in xrange(len(hist)):
        outfile.write('\n\nclassifier %d\n' % i)
        for h in hist[i].history:
            outfile.write('{}:\n'.format(h))
            for e in hist[i].history[h]:
                outfile.write('{}\n'.format(e))
            outfile.write('\n')

    for i in xrange(len(pred)):
        for j in xrange(len(pred[i])):
            outfile.write('%s  %d  %d\n' % (tokens[i][j], labels[i][j], pred[i][j])) 


def compute_result(args, outfile, pred, valid_data):
    outfile.write('\n\npredicted results:\n')
    total = right = tagged = 0
    total_ecs = right_ecs = tagged_ecs = 0
    pre = rec = f1 = 0

    text_len = [len(d) for d in valid_data['tokens']]
    labels = valid_data['tags']
    outfile.write('\ntext_len: {}\n'.format(text_len))
    #ipdb.set_trace()
    for i in xrange(len(pred)):
        for j in xrange(len(pred[i])):#xrange(text_len[i]):#
            if j >= text_len[i]:
                break
            label = np.argmax(labels[i][j])
            if pred[i][j] >= 1:
                tagged += 1
                if pred[i][j] == label:
                    right += 1
                if pred[i][j] == args.output_dim - 1: # exclussive
                    tagged_ecs += 1
                    if pred[i][j] == label:
                        right_ecs += 1
            if label >= 1:
                total += 1
                if label == args.output_dim - 1: # exclussive
                    total_ecs += 1

    results = {'rec': [], 'pre': [], 'f1': []}
    basic_f1(total_ecs, right_ecs, tagged_ecs, results)
    basic_f1(total, right, tagged, results)
    dev_end = time.time()
    for k, v in results.iteritems():
        print(k, v)
        outfile.write('{}: {}\n'.format(k, v))
    return results['rec'][-1], results['pre'][-1], results['f1'][-1]


def basic_f1(total, right, tagged, results):
    rec = pre = f1 = 0.0
    if total > 0:
        rec = right / float(total)
    if tagged > 0:
        pre = right / float(tagged)
    if rec + pre > 0:
        f1 = 2 * pre * rec / (pre + rec)
    results['rec'].append(rec)
    results['pre'].append(pre)
    results['f1'].append(f1)


def read_eas_texts(args):
    with open(args.data_name, 'r') as f:
        indata = pickle.load(f)
        if args.actionDB[0] == 'wikihow': # for wikihow data
            indata = indata[:118]
    
    data = {'tokens': [], 'tags': [], 'text_vecs': [],
            'sents': [],  'word2sent': [], 'sent_acts': []}
    cn = args.context_num
    args.emb_dim = cn * args.word_dim
    max_words_num = 0
    for i in xrange(len(indata)):
        if len(indata[i]['words']) > max_words_num:
            max_words_num = len(indata[i]['words'])
    print('max_words_num: %d\n' % max_words_num)
    # args.words_num = max_words_num

    for i in xrange(len(indata)):
        if len(indata[i]['words']) == 0:
            continue
        tmp_tags = np.zeros(args.words_num, dtype=np.int32)
        for acts in indata[i]['acts']:
            if acts['act_idx'] < args.words_num:
                tmp_tags[acts['act_idx']] = acts['act_type']
        tmp_vec = np.zeros([cn - 1 + args.words_num, args.word_dim])
        text_vec = np.zeros([args.words_num, args.emb_dim])
        #ipdb.set_trace()
        for idx, w in enumerate(indata[i]['words']):
            if idx >= args.words_num:
                break
            if w in args.word2vec.vocab:
                tmp_vec[cn - 1 + idx] = args.word2vec[w]
            text_vec[idx] = np.reshape(tmp_vec[idx: idx + cn], (1, -1))
        data['text_vecs'].append(text_vec)
        data['tags'].append(one_hot(tmp_tags, args.output_dim))
        data['tokens'].append(indata[i]['words'])
        data['sents'].append(indata[i]['sents'])
        data['word2sent'].append(indata[i]['word2sent'])
        data['sent_acts'].append(indata[i]['sent_acts'])

    indices = ten_fold_split_idx(len(data['tags']), args.k_fold_indices, args.k_fold)
    folds = index2data(indices, data)
    print('Total eas texts: %d' % len(data['tags']))
    return folds


def eas2af_sents(args, pred, data):
    sents = data['sents'][data_idx]
    sent_acts = data['sent_acts'][data_idx]
    word2sent = data['word2sent'][data_idx]
    sent_idx = word2sent[act_idx]
    word_ids = []
    if sent_idx > 0: # use the former sentence and current one
        words = sents[sent_idx - 1] + sents[sent_idx] + ['UNKNOWN_TOKEN']
        for k, v in word2sent.iteritems():
            if v == sent_idx or v == sent_idx - 1:
                word_ids.append(k)
    else:
        words = sents[sent_idx] + ['UNKNOWN_TOKEN']
        for k, v in word2sent.iteritems():
            if v == sent_idx:
                word_ids.append(k)
    end_idx = max(word_ids) # the last index of words of these two sents
    start_idx = min(word_ids)
    sent_len = len(words)
    acts = sent_acts[sent_idx]
    
    af_tags = np.ones(af_words_num, dtype=np.int32)
    af_sents = {}
    act_idxs = []
    obj_idxs = [[], []]
    for act in acts: # this is a tagged right action
        if act['act_idx'] < af_words_num:
            act_idxs.append(act['act_idx'])
        if act['act_idx'] != act_idx - start_idx:
            continue
        obj_idxs = act['obj_idxs']
        if len(obj_idxs[1]) == 0:
            for oi in obj_idxs[0]:
                if oi < af_words_num:
                    af_tags[oi] = 1
        else:
            for oi in obj_idxs[0] + obj_idxs[1]:
                if oi < af_words_num:
                    af_tags[oi] = 2
        break
    position = np.zeros(af_words_num)
    position.fill(act_idx - start_idx)
    distance = np.abs(np.arange(af_words_num) - position)
    tmp_vec = np.zeros([cn - 1 + af_words_num, args.word_dim])
    text_vec = np.zeros([af_words_num, args.emb_dim])
    #ipdb.set_trace()
    for idx, w in enumerate(words):
        if idx >= af_words_num:
            break
        if w in args.word2vec.vocab:
            tmp_vec[cn - 1 + idx] = args.word2vec[w]
        text_vec[idx, : cont_dim] = np.reshape(tmp_vec[idx: idx + cn], (1, -1))
        text_vec[idx, cont_dim: ] = distance[idx]
    af_sents['tokens'].append(words)
    af_sents['tags'].append(one_hot(af_tags, af_out_dim))
    af_sents['text_vecs'].append(text_vec)
    return af_sents


def read_af_sents(args):
    with open(args.data_name, 'rb') as f:
        indata = pickle.load(f)[-1]
        if args.actionDB == 'wikihow':
            indata = indata[:118]

    data = {'tokens': [], 'tags': [], 'text_vecs': []}
    cn = args.context_num
    cont_dim = cn * args.word_dim
    args.emb_dim = cont_dim + args.dis_dim
    max_words_num = 0
    for i in xrange(len(indata)):
        for j in xrange(len(indata[i])):
            if len(indata[i][j]) == 0:
                continue
            words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK']
            if len(words) > max_words_num:
                max_words_num = len(words)
    print('max_words_num: %d\n' % max_words_num)
    # args.words_num = max_words_num

    for i in xrange(len(indata)):
        for j in xrange(len(indata[i])):
            if len(indata[i][j]) == 0:
                continue
            words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK'] 
            sent_len = len(words)
            for k in xrange(len(indata[i][j]['acts'])):
                act_idx = indata[i][j]['acts'][k]['act_idx']
                obj_idxs = indata[i][j]['acts'][k]['obj_idxs']
                af_tags = np.zeros(args.words_num, dtype=np.int32)
                if len(obj_idxs[1]) == 0:
                    for oi in obj_idxs[0]:
                        if oi < args.words_num:
                            af_tags[oi] = 1
                else:
                    for oi in obj_idxs[0] + obj_idxs[1]:
                        if oi < args.words_num:
                            af_tags[oi] = 2
                position = np.zeros(args.words_num)
                position.fill(act_idx)
                distance = np.abs(np.arange(args.words_num) - position)
                
                tmp_vec = np.zeros([cn - 1 + args.words_num, args.word_dim])
                text_vec = np.zeros([args.words_num, args.emb_dim])
                #ipdb.set_trace()
                for idx, w in enumerate(words):
                    if idx >= args.words_num:
                        break
                    if w in args.word2vec.vocab:
                        tmp_vec[cn - 1 + idx] = args.word2vec[w]
                    text_vec[idx, : cont_dim] = np.reshape(tmp_vec[idx: idx + cn], (1, -1))
                    text_vec[idx, cont_dim: ] = distance[idx]
                data['tokens'].append(words)
                data['tags'].append(one_hot(af_tags, args.output_dim))
                data['text_vecs'].append(text_vec)

    indices = ten_fold_split_idx(len(data['tags']), args.k_fold_indices, args.k_fold)
    folds = index2data(indices, data)
    print('Total af sents: %d' % len(data['tags']))
    return folds


def one_hot(tags, dim):
    #ipdb.set_trace()
    tmp_tags = np.zeros([tags.shape[0], dim])
    for i, t in enumerate(tags):
        tmp_tags[i][t] = 1
    return tmp_tags


def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--words_num", type=int, default=500, help='')
    parser.add_argument("--word_dim", type=int, default=50, help='')
    parser.add_argument("--dis_dim", type=int, default=50, help='')
    parser.add_argument("--output_dim", type=int, default=4, help='')
    parser.add_argument("--batch_size", type=int, default=8, help='')
    parser.add_argument("--verbose", type=int, default=0, help='')
    parser.add_argument("--valid_rate", type=float, default=0.2, help='')
    parser.add_argument("--model_type", default='mlp', help='')
    parser.add_argument("--context_num", type=int, default=1, help='')
    parser.add_argument("--optimizer", default='rmsprop', help='')
    parser.add_argument("--epochs", type=int, default=100, help='')

    parser.add_argument("--gpu_rate", type=float, default=0.24, help='')
    parser.add_argument("--k_fold", type=int, default=5, help="")
    parser.add_argument("--fold_id", type=int, default=0, help="")
    parser.add_argument("--start_fold", type=int, default=0, help='')
    parser.add_argument("--end_fold", type=int, default=0, help='')

    parser.add_argument("--is_test", type=int, default=0, help='')
    parser.add_argument("--save_weight", type=int, default=1, help='')
    parser.add_argument("--actionDB", type=str, default='cooking', help='')
    parser.add_argument("--agent_mode", type=str, default='eas', help='')
    parser.add_argument("--result_dir", type=str, default='fixed_words_num', help='')

    args = parser.parse_args()
    K.set_image_data_format('channels_first')
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    if args.agent_mode == 'af':
        args.words_num = 100
        args.output_dim = 3
        args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        args.k_fold_indices = 'data/indices/%s_af_%d_fold_indices.pkl' % (args.actionDB, args.k_fold)
    else:
        args.data_name = 'data/%s_labeled_text_data.pkl' % args.actionDB 
        args.k_fold_indices = 'data/indices/%s_eas_%d_fold_indices.pkl' % (args.actionDB, args.k_fold)
    args.result_dir = 'results/%s/%s/%s.txt' % (args.actionDB, args.agent_mode, args.result_dir)
    if args.end_fold == 0:
        args.end_fold = args.k_fold
    return args


def main(args):
    start = time.time()
    if args.agent_mode == 'af':
        folds = read_af_sents(args)
    else:
        folds = read_eas_texts(args)

    #ipdb.set_trace()
    fold_result = {'rec': [], 'pre': [], 'f1': []}
    for fi in xrange(args.start_fold, args.end_fold):
        fold_start = time.time()
        train_data = folds['train'][fi]
        valid_data = folds['valid'][fi]
        train_vecs = np.array(train_data['text_vecs'])
        valid_vecs = np.array(valid_data['text_vecs'])
        train_labels = np.array(train_data['tags'])
        train_x = np.reshape(train_vecs, (-1, 1, args.words_num, args.emb_dim))
        valid_x = np.reshape(valid_vecs, (-1, 1, args.words_num, args.emb_dim))
        hist = []
        pred = []
        print('Train data: %d  Test data: %d' % (len(train_data['tags']), len(valid_data['tags'])))
        with open("%s_fold%d.txt" % (args.result_dir, fi), 'w') as outfile:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = args.gpu_rate
            set_session(tf.Session(config=config))
            for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            AC = ActionClassifier(args)
            for jj in tqdm(xrange(args.words_num)):
                AC.build_model_mlp(jj)
                if args.is_test == 0:
                    temp_hist = AC.train_one(train_x[:, 0, jj], train_labels[:,jj])
                    hist.append(temp_hist)
                else:
                    filename = 'weights/%s/%s/new_fold%d_model%d.h5' % (args.actionDB, args.agent_mode, args.fold_id, jj) 
                    AC.load_weights(filename)
                temp_pr = AC.test_one(valid_x[:, 0, jj])
                #ipdb.set_trace()
                if len(pred) == 0:
                    pred = temp_pr.reshape(-1, 1) 
                else:
                    pred =  np.concatenate([pred, temp_pr.reshape(-1, 1)],axis=1)
                if args.is_test == 0 and args.save_weight:
                    filename = 'weights/%s/%s/new_fold%d_model%d.h5' % (args.actionDB, args.agent_mode, args.fold_id, jj) 
                    AC.save_weights(filename)
            tf.reset_default_graph()
            rec, pre, f1 = compute_result(args, outfile, pred, valid_data)
            fold_result['rec'].append(rec)
            fold_result['pre'].append(pre)
            fold_result['f1'].append(f1)
            max_f1 = max(fold_result['f1'])
            avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
            fold_end = time.time()
            outfile.write('\n{}\n'.format(fold_result))
            outfile.write('Best f1: {}  Avg f1: {}\n'.format(max_f1, avg_f1))
            outfile.write('Total time cost of fold %d is: %ds\n' % (fi, fold_end - fold_start))
            print('\nTotal time cost of fold %d is: %ds\n' % (fi, fold_end - fold_start))
    max_f1 = max(fold_result['f1'])
    avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Best f1: {}  Avg f1: {}'.format(max_f1, avg_f1))
    print('Total time cost: %ds' % (end - start))


def pipeline_test(args):
    start = time.time()
    if args.agent_mode == 'af':
        folds = read_af_sents(args)
    else:
        folds = read_eas_texts(args)

    #ipdb.set_trace()
    fold_result = {'rec': [], 'pre': [], 'f1': []}
    for fi in xrange(args.start_fold, args.end_fold):
        fold_start = time.time()
        train_data = folds['train'][fi]
        valid_data = folds['valid'][fi]
        train_vecs = np.array(train_data['text_vecs'])
        valid_vecs = np.array(valid_data['text_vecs'])
        train_labels = np.array(train_data['tags'])
        train_x = np.reshape(train_vecs, (-1, 1, args.words_num, args.emb_dim))
        valid_x = np.reshape(valid_vecs, (-1, 1, args.words_num, args.emb_dim))
        pred = []
        print('Train data: %d  Test data: %d' % (len(train_data['tags']), len(valid_data['tags'])))
        with open("%s_fold%d.txt" % (args.result_dir, fi), 'w') as outfile:
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = args.gpu_rate
            set_session(tf.Session(config=config))
            for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            AC = ActionClassifier(args)
            for jj in tqdm(xrange(args.words_num)):
                AC.build_model_mlp(jj)
                if args.is_test == 0:
                    temp_hist = AC.train_one(train_x[:, 0, jj], train_labels[:,jj])
                else:
                    filename = 'weights/%s/%s/fold%d_model%d.h5' % (args.actionDB, args.agent_mode, args.fold_id, jj) 
                    AC.load_weights(filename)
                temp_pr = AC.test_one(valid_x[:, 0, jj])
                #ipdb.set_trace()
                if len(pred) == 0:
                    pred = temp_pr.reshape(-1, 1) 
                else:
                    pred =  np.concatenate([pred, temp_pr.reshape(-1, 1)],axis=1)
                if args.is_test == 0 and args.save_weight:
                    filename = 'weights/%s/%s/fold%d_model%d.h5' % (args.actionDB, args.agent_mode, args.fold_id, jj) 
                    AC.save_weights(filename)
            tf.reset_default_graph()
            rec, pre, f1 = compute_result(args, outfile, pred, valid_data)
            fold_result['rec'].append(rec)
            fold_result['pre'].append(pre)
            fold_result['f1'].append(f1)
            max_f1 = max(fold_result['f1'])
            avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
            fold_end = time.time()
            outfile.write('\n{}\n'.format(fold_result))
            outfile.write('Best f1: {}  Avg f1: {}\n'.format(max_f1, avg_f1))
            outfile.write('Total time cost of fold %d is: %ds\n' % (fi, fold_end - fold_start))
            print('\nTotal time cost of fold %d is: %ds\n' % (fi, fold_end - fold_start))
    max_f1 = max(fold_result['f1'])
    avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Best f1: {}  Avg f1: {}'.format(max_f1, avg_f1))
    print('Total time cost: %ds' % (end - start))



if __name__ == '__main__':
    main(args_init())