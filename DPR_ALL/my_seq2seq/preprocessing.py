import re
import os
import ipdb
import random
import pickle
import mysql.connector
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors


def pre_process_database(table, databag_name):
    db = mysql.connector.connect(user='fengwf', password='123', database='test')
    cur = db.cursor()
    if table == 'tag_actions2':
        print "getting long text of WHS ..."
        get_data = 'select * from %s where text_num >= 33'%table
    else:
        get_data = 'select * from %s'%table
    cur.execute(get_data)
    result = cur.fetchall()
    print 'Total sentence: %d\n'%len(result)

    sents = {
        'instruction': [],
        'action':[]
    }
    actions = []
    vocab = {'EOF': 1, 'UNK':1}
    word2ind = {'EOF': 0, 'UNK':1}
    ind2word = {0: 'EOF', 1:'UNK'}
    index = 2
    for idx,data in enumerate(result):
        try:
            words = re.split(r' ', str(data[2])) #data[2].split() #
        except Exception as e:
            print e
            continue
        tags = [int(t) for t in re.split(r' ', data[3])] #data[3].split()] #
        if len(words) != len(tags):
            print 'idx %d not match, len(words)=%d, len(tags)=%d'%(idx, len(words), len(tags))
            print data
            ipdb.set_trace()
            continue
        #assert len(words) == len(tags)
        sents['instruction'].append(words)
        #temp_action = []
        for i,w in enumerate(words):
            #if tags[i] == '1':
            #    temp_action.append(words[i])
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 1
            if w not in word2ind.keys():
                word2ind[w] = index
                ind2word[index] = w
                index += 1
        sents['action'].append(tags)
        #sents['action'].append(temp_action)
        # set low frequency word to -1
        #word_count = 5
        #for w in vocab.keys():
        #    if vocab[w] < 5:
        #        word2ind.pop(w)

    databag = {
        'vocab': vocab,
        'sents': sents,
        'word2ind':word2ind,
        'ind2word':ind2word
    }
    with open('./data/%s_databag.pkl'%databag_name, 'w') as f:
        pickle.dump(databag, f)
    #ipdb.set_trace()
    print 'Successfully save file as %s_databag.pkl\n'%databag_name


def k_flod_split(databag_name, k=10):
    with open('./data/%s_databag.pkl'%databag_name, 'r') as f:
        raw_data = pickle.load(f)
        dict_data = raw_data['sents']
        num_samples = len(dict_data['instruction'])
    num_slice = int(num_samples/float(k))
    element_list = range(num_samples)
    k_slice_data = []
    left_num = num_samples

    while left_num >= 2*num_slice:
        a_slice = random.sample(element_list, num_slice)
        for i in xrange(num_slice):
            #print i,a_slice[i]
            element_list.remove(a_slice[i])
        left_num = len(element_list)
        k_slice_data.append(a_slice)
    k_slice_data.append(element_list)
    assert len(k_slice_data) == k

    with open('./data/%s_k_slice_data.pkl'%databag_name, 'w') as f:
        pickle.dump(k_slice_data, f)
    print 'Successfully save k slices of data.\n'


class DataProcess(object):
    """docstring for DataProcess"""
    def __init__(self, data_fold, databag, path_rawdata=None):
        print 'initializing the processer ...'
        self.databag = databag
        if path_rawdata:
            self.path_rawdata = path_rawdata
        else:
            self.path_rawdata = './data/'

        # vocabulary of navigation instructions
        # raw_data is a dict
        with open(self.path_rawdata+'%s_databag.pkl'%self.databag, 'r') as f:
            raw_data = pickle.load(f)
            self.vocab = raw_data['vocab']
            self.dict_data = raw_data['sents']
            self.word2ind = raw_data['word2ind']
            self.ind2word = raw_data['ind2word']

        with open(self.path_rawdata+'%s_k_slice_data.pkl'%self.databag, 'r') as f:
            self.k_slice_data = pickle.load(f)
            self.devset = self.k_slice_data[data_fold]
            print 'loading %d slices of data, validate the %dth slice\n'%(len(self.k_slice_data), data_fold)
    
        self.dim_action = 68 #50 #len(self.vocab) + 1
        self.dim_lang = len(self.vocab) + 1
        self.dim_model = 100  # wordvec size
        print 'dim_action: %d\tdim_model: %d\n'%(self.dim_action, self.dim_model)

        self.seq_lang_numpy = None  # instruction to word dict, it's a vector
        self.seq_action_numpy = None  # action to index
        self.buckets = [(20, 2), (20, 5), (50, 2), (50, 5)]


    def process_one_data(self, idx_data):
        # tag_split ='train' or 'dev'
        # one_data means an instruction of a map        

        self.seq_lang_numpy = []
        self.seq_action_numpy = [] # the label of actions, 0 for the EOF flag
        self.action_word = []

        for w in self.dict_data['instruction'][idx_data]:
            self.seq_lang_numpy.append(self.word2ind[w])
        for i,t in enumerate(self.dict_data['action'][idx_data]):
            if t == 1:
                #self.seq_action_numpy.append(self.seq_lang_numpy[i])
                self.seq_action_numpy.append(i)
                self.action_word.append(self.dict_data['instruction'][idx_data][i])
        #self.seq_action_numpy.append(-1)

        self.seq_lang_numpy = np.array(self.seq_lang_numpy, dtype=np.int32)
        self.seq_action_numpy = np.array(self.seq_action_numpy, dtype=np.int32) 
        '''
        for w in one_data:
            if len(self.model[w]):
                word_vec = self.model[w]
            else:
                word_vec = np.zeros(self.dim_model)
            self.seq_lang_numpy.append(word_vec)
        '''

    def compute_bucket_type(self):
        acts = self.dict_data['action']
        sents = self.dict_data['instruction']
        word_num = {}
        act_num = {}
        for i in xrange(len(sents)):
            num = len(sents[i])/10 + 1
            if num in word_num:
                word_num[num] += 1
            else:
                word_num[num] = 1
            num1 = sum(acts[i])
            if num1 in act_num:
                act_num[num1] += 1
            else:
                act_num[num1] = 1
        for k,v in word_num.iteritems():
            print k,v
        for k,v in act_num.iteritems():
            print k,v


    def read_data(self):
        if os.path.exists('data/seq_data.pkl'):
            with open('data/seq_data.pkl', 'rb') as f:
                train_data, valid_data = pickle.load(f)
                return train_data, valid_data
        train_data = {}
        valid_data = {}
        acts = self.dict_data['action']
        sents = self.dict_data['instruction']
        for i in xrange(len(sents)):
            for b in self.buckets:
                if len(sents[i]) < b[0] and sum(acts[i]) <= b[1]:
                    sent = np.zeros(b[0], dtype=np.int32)
                    act = np.zeros(b[1], dtype=np.int32)
                    t_w = np.zeros(b[1], dtype=np.int32)
                    idx = 0
                    for j,w in enumerate(sents[i]):
                        sent[j] = self.word2ind[w]
                        if acts[i][j] == 1:
                            act[idx] = self.word2ind[w]
                            t_w[idx] = 1
                            idx += 1
                    if i in self.devset:
                        if b in valid_data:
                            valid_data[b]['sent'].append(sent)
                            valid_data[b]['act'].append(act)
                            valid_data[b]['t_w'].append(t_w)
                        else:
                            valid_data[b] = {}
                            valid_data[b]['sent'] = [sent]
                            valid_data[b]['act'] = [act]
                            valid_data[b]['t_w'] = [t_w]
                    else:
                        if b in train_data:
                            train_data[b]['sent'].append(sent)
                            train_data[b]['act'].append(act)
                            train_data[b]['t_w'].append(t_w)
                        else:
                            train_data[b] = {}
                            train_data[b]['sent'] = [sent]
                            train_data[b]['act'] = [act]
                            train_data[b]['t_w'] = [t_w]
                    break
        ipdb.set_trace()
        with open('data/seq_data.pkl', 'wb') as f:
            pickle.dump((train_data, valid_data), f)
        return train_data, valid_data




if __name__ == '__main__':
    ipdb.set_trace()
    #tables = ['tag_actions', 'tag_actions1', 'tag_actions2', 'tag_actions3', 'tag_actions5']
    #databag_name = ['cooking', 'wikihow', 'windows', 'sail', 'long_wikihow']
    #for i in xrange(len(tables)):
    #    pre_process_database(tables[i], databag_name[i])
    #    k_flod_split(databag_name[i])

    #pre_process_database('tag_actions2', 'long_windows')
    #k_flod_split('long_windows')
    test = DataProcess(-1, 'cooking')
    #test.compute_bucket_type()
    a, b = test.read_data()
    print 'end\n'