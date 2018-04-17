# coding = utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import re
import sys
import ipdb
import gzip
import json
import jieba
import random
import pickle
import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.python.platform import gfile
from gensim.models import KeyedVectors
#reload(sys)
#sys.setdefaultencoding('utf-8')

_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

_DIGIT_RE = re.compile(r"\d")

def sent2vec(sentence, max_seq_len, dim, wordvec):
    words = basic_tokenizer(sentence)
    sentvec = np.zeros([max_seq_len, dim])
    for i, w in enumerate(words): 
        if i >= max_seq_len:
            break
        if w in wordvec.vocab:
            sentvec[i] = wordvec[w]
    return sentvec


def dense_to_one_hot(labels_dense, num_classes=6):
  num_labels = len(labels_dense)
  labels_one_hot = np.zeros((num_labels, num_classes))
  for i in range(num_labels):
    labels_one_hot[i][int(labels_dense[i])]=1
  return labels_one_hot

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  words.extend(" ".join(jieba.cut(sentence, cut_all=False, HMM=True)).strip().split())
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
      print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
      vocab = {}
      #import ipdb
      #ipdb.set_trace()
      with gfile.GFile(data_path) as f:
        # counter = 0
        for line in f:
          lines=line.strip().split("\t")
          line=lines[0]
          # counter += 1
          # if counter % 1000 == 0:
          #   print("  processing line %d" % counter)
          #line = tf.compat.as_bytes(line)
          tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
          for w in tokens:
            word = _DIGIT_RE.sub("0", w) if normalize_digits else w
            if word in vocab:
              vocab[word] += 1
            else:
              vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
          vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode='w') as vocab_file:
          for w in vocab_list:
            vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip().decode('utf8') for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,max_seq_len=20,tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)   
  else:
    words = basic_tokenizer(sentence)
  # if not normalize_digits:
  #   return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  if len(words)>max_seq_len:
    sentence_to_ids=[vocabulary.get(w, UNK_ID) for w in words[:max_seq_len]]
  else:
    sentence_to_ids=[vocabulary.get(w, UNK_ID) for w in words]+[PAD_ID]*(max_seq_len-len(words))
  return sentence_to_ids


def prepare_data(data_path, max_seq_len, use_emb=False, model_name=None, num_sample=4300,
  vocabulary_path=None, max_vocabulary_size=None, tokenizer=None):
  train_x_input=[]
  train_y_input=[]
  train_seq_lens=[]
  test_x_input=[]
  test_y_input=[]
  test_seq_lens=[]
  tmp_x = [[] for _ in range(6)]
  tmp_y = [[] for _ in range(6)]
  tmp_len = [[] for _ in range(6)]
  #ipdb.set_trace()
  if use_emb:
      wordvec = KeyedVectors.load_word2vec_format(model_name, binary=True)
      dim = int(model_name.split('_')[-1])
      with gfile.GFile(data_path) as f:
          for line in f:
              lines = line.strip().split("\t")
              if len(lines) != 2:
                  continue
              lines[0] = ''.join(re.findall('[\u4e00-\u9eff|，、：；。！？]+', lines[0]))
              ind = int(lines[1])
              sentvec = sent2vec(lines[0], max_seq_len, dim, wordvec)
              tmp_len[ind].append(np.concatenate([np.ones(len(sentvec), dtype=np.int32), 
                  np.zeros(max_seq_len-len(sentvec), dtype=np.int32)]))
              tmp_x[ind].append(sentvec)
              tmp_y[ind].append(lines[1])
  else:
      create_vocabulary(vocabulary_path, data_path , max_vocabulary_size, tokenizer)
      vocab, _ = initialize_vocabulary(vocabulary_path)
      with gfile.GFile(data_path) as f:
          for line in f:
              lines = line.strip().split("\t")
              if len(lines) != 2:
                  continue
              ind = int(lines[1])
              sentvec = sentence_to_token_ids(lines[0], vocab, max_seq_len)
              tmp_len[ind].append(np.concatenate([np.ones(len(sentvec), dtype=np.int32), 
                  np.zeros(max_seq_len-len(sentvec), dtype=np.int32)]))
              tmp_x[ind].append(sentvec)
              tmp_y[ind].append(lines[1])

  if os.path.exists('indexes.pkl'):
      inds = pickle.load(open('indexes.pkl','rb'))
      print('\nLoaded pre-set indexes.\n')
  else:
      inds = []
      for m in range(6):
          tmp_inds = np.arange(len(tmp_x[m]))
          np.random.shuffle(tmp_inds)
          inds.append(tmp_inds)
      with open('indexes.pkl', 'wb') as f:
          pickle.dump(inds, f)
          print('\nSuccessfully saved indexes.\n')

  for j in range(num_sample):
      for i in range(6):
          if j >= len(tmp_x[i]): 
              continue
          ind = inds[i][j]
          if len(test_x_input) < 1200 and j%5 == 0:
              test_x_input.append(tmp_x[i][ind])
              test_y_input.append(tmp_y[i][ind])
              test_seq_lens.append(tmp_len[i][ind])
          else:
              train_x_input.append(tmp_x[i][ind])
              train_y_input.append(tmp_y[i][ind])
              train_seq_lens.append(tmp_len[i][ind])

  train_y_input = dense_to_one_hot(train_y_input, num_classes=6)
  test_y_input = dense_to_one_hot(test_y_input, num_classes=6)
  #ipdb.set_trace()
  return len(wordvec.vocab), train_x_input, train_y_input, train_seq_lens, test_x_input, test_y_input, test_seq_lens
  #return (train_x_input[4000:], train_y_input[4000:],train_seq_lens[4000:],train_x_input[:4000], train_y_input[:4000],train_seq_lens[:4000])
  # return (train_x_input, train_y_input,train_seq_lens)


def prepare_raw_data(data_path,max_seq_len, vocabulary_path,max_vocabulary_size,tokenizer):
    data_x_input=[]
    seq_lens=[]
    vocab,_=initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as f:
      # counter = 0
        for line in f:
            lines=line.strip().split("\t")
            seq_lens.append(len(lines[0]))
            data_x_input.append(sentence_to_token_ids(lines[0],vocab))
    return data_x_input,seq_lens



class NLPCC_data(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a np array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, max_seq_len=20):
        self.data,self.labels,self.seqlen,self.dev_data,self.dev_labels,self.dev_seqlen=prepare_data("./data/original_train_data0306",max_seq_len,"./data/vocabulary",40000,None)
        # self.data,self.labels,self.seqlen=prepare_data("./data/train_data2",max_seq_len,"./data/vocabulary",40000,None)
        # self.dev_data,self.dev_labels,self.dev_seqlen=prepare_data("./data/dev_data",max_seq_len,"./data/vocabulary",40000,None)
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


class Raw_data(object):
    def __init__(self, max_seq_len=20):
        self.data,self.seqlen =prepare_raw_data("/home/lin/Documents/qa_code/dataset/tvsub-master/data/orignal/train/train_bak",max_seq_len,"./data/vocabulary",40000,None)
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data,batch_seqlen
