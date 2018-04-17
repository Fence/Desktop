#coding:utf-8
import os
import time
import pickle
import numpy as np
from progress.bar import Bar

class ProgressBar(Bar):
    message = 'Loading'
    fill = '#'
    suffix = '%(percent).1f%% | ETA: %(eta)ds'


class CharEmb:
    """
    Generate bow-like character embedding 
    Total 50 chars: "abcdefghijklmnopqrstuvwxyz0123456789,;!?:._'"-@#$%"
    """
    def __init__(self):
        self.chars = """abcdefghijklmnopqrstuvwxyz0123456789,;!?:._'"-@#$%"""
        self.dim = len(self.chars)
        self.char_dict = {}
        for c in self.chars:
            self.char_dict[c] = len(self.char_dict)

    def char_emb(self, word):
        '''
        Generate character embedding for words
        '''
        char_vec = np.zeros(len(self.chars))
        for c in word.lower():
            if c in self.char_dict.keys():
                char_vec[self.char_dict[c]] += 3 #1
        return char_vec


def ten_fold_split_idx(num_data, fname, random=True, k=10):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = int(num_data/k)
    indices = []

    if random:
        tmp_idxs = np.arange(num_data)
        np.random.shuffle(tmp_idxs)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_idxs[i*n: ])
            else:
                indices.append(tmp_idxs[i*n: (i+1)*n])
    else:
        for i in xrange(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices


def index2data_old(indices, data):
    """
    Split data according to given indices
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    num_data = len(data['sents_vec'])
    for i in xrange(len(indices)):
        valid_sents_vec = []
        valid_sents_tag = []
        valid_sent_words = []
        valid_sent_len = []
        train_sents_vec = []
        train_sents_tag = []
        train_sent_words = []
        train_sent_len = []
        for idx in xrange(num_data):
            if idx in indices[i]:
                valid_sents_vec.append(data['sents_vec'][idx])
                valid_sents_tag.append(data['sents_tag'][idx])
                valid_sent_words.append(data['sent_words'][idx])
                valid_sent_len.append(data['sent_len'][idx])
            else:
                train_sents_vec.append(data['sents_vec'][idx])
                train_sents_tag.append(data['sents_tag'][idx])
                train_sent_words.append(data['sent_words'][idx])
                train_sent_len.append(data['sent_len'][idx])

        valid_data = {'sents_vec': valid_sents_vec, 'sents_tag': valid_sents_tag,
                      'sent_words': valid_sent_words, 'sent_len': valid_sent_len}
        train_data = {'sents_vec': train_sents_vec, 'sents_tag': train_sents_tag,
                      'sent_words': train_sent_words, 'sent_len': train_sent_len}
        folds['train'].append(train_data)
        folds['valid'].append(valid_data)

    return folds


def index2data(indices, data):
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    keys = data.keys()
    print('data.keys: {}\n\n'.format(keys))
    num_data = len(data[keys[0]])
    for i in xrange(len(indices)):
        valid_data = {}
        train_data = {}
        for k in keys:
            valid_data[k] = []
            train_data[k] = []
        for idx in xrange(num_data):
            for k in keys:
                if idx in indices[i]:
                    valid_data[k].append(data[k][idx])
                else:
                    train_data[k].append(data[k][idx])
        folds['train'].append(train_data)
        folds['valid'].append(valid_data)

    return folds
