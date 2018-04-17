#coding:utf-8
import os
import time
import pickle
import numpy as np


class CharEmb:
    """
    a~z: 97~122    0~9: 48~57
    -: 45  .: 46   _:95   ':39
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
                char_vec[self.char_dict[c]] += 1
        return char_vec


class ReadData(object):
    """
    Read raw data form database, save them into sents and tags
    Read sents and tags, find out action arguments using stanford dependency parser
    """
    def __init__(self):
        from nltk.parse.stanford import StanfordDependencyParser
        path_to_jar = '/home/fengwf/stanford/stanford-corenlp-3.7.0.jar'
        path_to_models_jar = '/home/fengwf/stanford/english-models.jar'
        self.dep_parser = StanfordDependencyParser(
            path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)


    def stanford_find_vp(self, sentences, tags):
        #ipdb.set_trace()
        sents = [' '.join(s) for s in sentences]
        start = time.time()
        print('\ndependency parsing ... ')
        dep = self.dep_parser.raw_parse_sents(sents)
        end = time.time()
        print('dependency parsing time: %.2fs\n' % (end - start))
        data = []
        for k in xrange(len(sents)):
            dep_sent_root = dep.next()
            dep_sent = dep_sent_root.next()
            dep_conll = [i.split() for i in str(dep_sent.to_conll(10)).split('\n') if i]
            verb_obj = {}
            verbs = []
            words = [w[1] for w in dep_conll]
            for line in dep_conll: # use conll format for sentence dependency
                if 'dobj' in line or 'nsubjpass' in line:
                    objs = [line[1]]
                    obj_idxs = [int(line[0]) - 1]
                    verb_idx = int(line[6]) - 1
                    verbs.append(verb_idx)
                    if obj_idxs[-1] >= len(words) or verb_idx >= len(words):
                        continue
                    verb = dep_conll[verb_idx][1]
                    verb_obj[verb_idx] = [obj_idxs[0]]
                    '''
                    for one_line in dep_conll:
                        # find the conjunctive relation objects
                        if int(one_line[6]) - 1 == obj_idxs[0] and one_line[7] == 'conj':
                            #print(one_line)
                            if int(one_line[0]) - 1 >= len(words):
                                continue
                            objs.append(one_line[1])
                            obj_idxs.append(int(one_line[0]) - 1)
                            verb_obj[verb_idx].append(obj_idxs[-1])
                    '''
            if len(sentences[k]) == len(words):
                for i, t in enumerate(tags[k]):
                    if t == '1' and i not in verb_obj: # action
                        verb_obj[i] = []
            else:
                pass
                #print('len(sentences[%d]): %d\tlen(words): %d' % (k, len(sentences[k]), len(words)))
            
            #if verb_obj:
                #print(verb_obj)
            data.append({'sent':words, 'acts':verb_obj})
        return data



def ten_fold_split_idx(num_data, fname, k, random=True):
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
    n = num_data/k
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


def index2data(indices, data):
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': []}
    if type(data) == dict:
        keys = data.keys()
        print('data.keys: {}'.format(keys))
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
    else:
        num_data = len(data)
        for i in xrange(len(indices)):
            valid_data = []
            train_data = []
            for idx in xrange(num_data):
                if idx in indices[i]:
                    valid_data.append(data[idx])
                else:
                    train_data.append(data[idx])
            folds['train'].append(train_data)
            folds['valid'].append(valid_data)

    return folds


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed

def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


def save_pkl(obj, path):
    with open(path, 'w') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path) as f:
        obj = pickle.load(f)
        return obj

@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)

@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj