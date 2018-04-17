#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np
from copy import deepcopy
from utils import ten_fold_split_idx, index2data

#create table tag_actions5(text_num int, sent_num int, af_sent varchar(400), tag_sent varchar(400));
class Environment:
    def __init__(self, args):
        print('Initializing the Environment...')  
        self.emb_dim = args.emb_dim
        self.tag_dim = args.tag_dim
        self.dis_dim = args.dis_dim
        self.act_dim = args.act_dim
        self.obj_dim = args.obj_dim
        self.word_dim = args.word_dim
        self.words_num = args.words_num
        self.action_rate = args.action_rate
        self.use_act_rate = args.use_act_rate
        self.reward_base = args.reward_assign
        self.reward_assign = self.reward_base * np.array([2, 1, -1, -2])

        self.word2vec = args.word2vec
        self.char_emb_flag = args.char_emb_flag
        self.max_char_len = args.max_char_len
        self.fold_id = args.fold_id
        self.use_cross_valid = args.use_cross_valid
        self.k_fold = args.k_fold
        self.k_fold_indices = args.k_fold_indices
        self.actionDB = args.actionDB.split()
        self.test_text_num = args.test_text_num
        self.max_text_num = [int(t) for t in args.max_text_num.split()]

        self.terminal_flag = False
        self.train_epoch_end_flag = False
        self.valid_epoch_end_flag = False
        self.max_data_char_len = 0
        self.max_data_sent_len = 0
        self.data_name = args.data_name
        # self.append_distance = args.append_distance
        self.char_info()
        self.agent_mode = args.agent_mode
        if self.agent_mode == 'af':
            self.read_af_sents()
        elif self.agent_mode == 'eas' or self.agent_mode == 'multi':
            self.read_eas_texts()
        elif self.agent_mode == 'db':
            self.read_database_texts()
        

    def init_predict_eas_text(self, raw_text):
        #ipdb.set_trace()
        #raw_text = re.sub(r'\n|\r|\(|\)|,|;', ' ', raw_text)
        #raw_text = re.split(r'\. |\? |\! ', raw_text)
        text = {'tokens': [], 'sents': [], 'word2sent': {}}
        for s in raw_text:
            words = s.split()
            if len(words) > 0:
                for i in range(len(words)):
                    text['word2sent'][i + len(text['tokens'])] = [len(text['sents']), i]
                text['tokens'].extend(words)
                text['sents'].append(words)

        sent_vec = np.zeros([self.words_num, self.emb_dim])
        for i, w in enumerate(text['tokens']):
            if i >= self.words_num:
                break
            if w in self.word2vec.vocab:
                sent_vec[i][: self.word_dim] = self.word2vec[w]

        self.state = sent_vec
        self.terminal_flag = False
        self.current_text = text


    def act_eas_text(self, action, word_idx):
        self.state[word_idx, self.emb_dim-self.tag_dim:] = action + 1
        if word_idx + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True


    def char_info(self):
        #chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,;!?:._'\"+-*/@#$%"
        chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
        self.char2idx = {}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)
        return self.char2idx


    def read_eas_texts(self):
        with open(self.data_name, 'r') as f:
            indata = pickle.load(f)
            if self.actionDB[0] == 'wikihow': # for wikihow data
                indata = indata[:118]
        eas_texts = []
        #ipdb.set_trace()
        f = open('data/eas_texts.txt', 'w')
        for i in xrange(len(indata)):
            if len(indata[i]['words']) == 0:
                continue
            eas_text = {}
            eas_text['tokens'] = indata[i]['words']
            eas_text['sents'] = indata[i]['sents']
            eas_text['acts'] = indata[i]['acts']
            eas_text['sent_acts'] = indata[i]['sent_acts']
            # for sent_act in indata[i]['sent_acts']:
            #     for act in sent_act:
            #         if len(act['obj_idxs']) != 2:
            #             print(i)
            eas_text['word2sent'] = indata[i]['word2sent']
            eas_text['tags'] = np.ones(len(indata[i]['words']), dtype=np.int32)
            eas_text['act2related'] = {}
            for acts in indata[i]['acts']:
                eas_text['act2related'][acts['act_idx']] = acts['related_acts']
                eas_text['tags'][acts['act_idx']] = acts['act_type'] + 1 # 2, 3, 4
            f.write('text: %d\n' % i)
            f.write('tokens: \n{}\n'.format(eas_text['tokens']))
            f.write('sents: \n{}\n'.format(eas_text['sents']))
            f.write('acts: \n{}\n'.format(eas_text['acts']))
            f.write('tags: \n{}\n'.format(eas_text['tags']))
            f.write('word2sent: \n{}\n\n'.format(eas_text['word2sent']))
            f.write('sent_acts: \n{}\n\n'.format(eas_text['sent_acts']))

            self.create_matrix(eas_text)
            eas_texts.append(eas_text)
        
        if self.use_cross_valid:
            eas_indices = ten_fold_split_idx(len(eas_texts), self.k_fold_indices, self.k_fold)
            eas_folds = index2data(eas_indices, eas_texts)
            self.train_data = eas_folds['train'][self.fold_id]
            self.valid_data = eas_folds['valid'][self.fold_id]
            self.train_steps = len(self.train_data) * self.words_num
            self.valid_steps = len(self.valid_data) * self.words_num
        else:
            self.train_data = eas_texts[self.test_text_num: self.max_text_num[0]]
            self.valid_data = eas_texts[: self.test_text_num]
            self.train_steps = self.words_num * (self.max_text_num[0] - self.test_text_num)
            self.valid_steps = self.words_num * self.test_text_num
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))
        f.close()


    def read_af_sents(self):
        with open(self.data_name, 'r') as f:
            _, __, indata = pickle.load(f)
            if self.actionDB[0] == 'wikihow': # for wikihow data
                indata = indata[:118]
        af_sents = []
        #ipdb.set_trace()
        for i in xrange(len(indata)):
            for j in xrange(len(indata[i])):
                if len(indata[i][j]) == 0:
                    continue
                # -1 obj_idx refer to UNK
                words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK'] 
                sent_len = len(words)
                act_idxs = [a['act_idx'] for a in indata[i][j]['acts'] if a['act_idx'] < self.words_num]
                for k in xrange(len(indata[i][j]['acts'])):
                    act_idx = indata[i][j]['acts'][k]['act_idx']
                    obj_idxs = indata[i][j]['acts'][k]['obj_idxs']
                    af_sent = {}
                    af_tags = np.ones(sent_len, dtype=np.int32)
                    if len(obj_idxs[1]) == 0:
                        af_tags[obj_idxs[0]] = 2 # essential objects
                    else:
                        af_tags[obj_idxs[0]] = 4 # exclusive objects
                        af_tags[obj_idxs[1]] = 4 # exclusive objects
                    position = np.zeros(sent_len, dtype=np.int32)
                    position.fill(act_idx)
                    distance = np.abs(np.arange(sent_len) - position)
                    
                    af_sent['tokens'] = words
                    af_sent['tags'] = af_tags
                    af_sent['distance'] = distance
                    af_sent['act_idxs'] = act_idxs
                    af_sent['obj_idxs'] = obj_idxs
                    self.create_matrix(af_sent)
                    af_sents.append(af_sent)

        if self.use_cross_valid:
            af_indices = ten_fold_split_idx(len(af_sents), self.k_fold_indices, self.k_fold)
            af_folds = index2data(af_indices, af_sents)
            self.train_data = af_folds['train'][self.fold_id]
            self.valid_data = af_folds['valid'][self.fold_id]
            self.train_steps = len(self.train_data) * self.words_num
            self.valid_steps = len(self.valid_data) * self.words_num
        else:
            self.train_data = af_sents[self.test_text_num: self.max_text_num[0]]
            self.valid_data = af_sents[: self.test_text_num]
            self.train_steps = self.words_num * (self.max_text_num[0] - self.test_text_num)
            self.valid_steps = self.words_num * self.test_text_num
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def read_database_texts(self):
        self.db = mysql.connector.connect(user='fengwf',password='123',database='test')
        self.cur = self.db.cursor()
        result = []
        for i in xrange(len(self.max_text_num)):
            for j in xrange(self.max_text_num[i]):
                get_data = "select * from %s where text_num = %d order by sent_num" % (self.actionDB[i], j)
                self.cur.execute(get_data)
                result.append(self.cur.fetchall())

        print('Total texts: %d' % len(result))
        eas_texts = []
        for i in xrange(len(result)):
            eas_text = {}
            eas_text['tokens'] = []
            eas_text['tags'] = []
            for j in xrange(len(result[i])):
                try:
                    tmp_tokens = str(result[i][j][2]).split()
                except Exception as e:
                    continue
                eas_text['tokens'].extend(tmp_tokens)
                eas_text['tags'].extend([int(t)+1 for t in result[i][j][3].split()])
            self.create_matrix(eas_text)
            eas_texts.append(eas_text)
        
        if self.use_cross_valid:
            eas_indices = ten_fold_split_idx(len(eas_texts), self.k_fold_indices, self.k_fold)
            eas_folds = index2data(eas_indices, eas_texts)
            self.train_data = eas_folds['train'][self.fold_id]
            self.valid_data = eas_folds['valid'][self.fold_id]
            self.train_steps = len(self.train_data) * self.words_num
            self.valid_steps = len(self.valid_data) * self.words_num
        else:
            self.train_data = eas_texts[self.test_text_num: self.max_text_num[0]]
            self.valid_data = eas_texts[: self.test_text_num]
            self.train_steps = self.words_num * (self.max_text_num[0] - self.test_text_num)
            self.valid_steps = self.words_num * self.test_text_num
        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('\n\ntraining texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d\tmax_data_char_len: %d' % (self.max_data_sent_len, self.max_data_char_len))
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def create_matrix(self, sentence):
        #ipdb.set_trace()
        sent_vec = []
        char_vec = []
        for w in sentence['tokens']:
            if len(w) > self.max_data_char_len:
                self.max_data_char_len = len(w)
            if w in self.word2vec.vocab:
                sent_vec.append(self.word2vec[w])
            else:
                sent_vec.append(np.zeros(self.word_dim))
            if len(w) < self.max_char_len:
                w = w + ' '*(self.max_char_len - len(w))
            else:
                w = w[: self.max_char_len]
            tmp_char_vec = []
            for c in w:
                if c in self.char2idx:
                    tmp_char_vec.append(self.char2idx[c])
                else:
                    tmp_char_vec.append(0)
            char_vec.append(tmp_char_vec)
        char_vec = np.array(char_vec, dtype=np.int32)
        sent_vec = np.array(sent_vec)
        pad_len = self.words_num - len(sent_vec)
        if len(sent_vec) > self.max_data_sent_len:
            #ipdb.set_trace()
            self.max_data_sent_len = len(sent_vec)
        if pad_len > 0:
            sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
            char_vec = np.concatenate((char_vec, np.zeros([pad_len, self.max_char_len])))
            sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
            if 'distance' in sentence:
                # if self.append_distance:
                #     distance = np.zeros([self.words_num, self.dis_dim])
                #     for d in xrange(len(sentence['distance'])):
                #         distance[d] = sentence['distance'][d]
                #     sent_vec = np.concatenate((sent_vec, distance), axis=1)
                # else:
                for d in xrange(len(sentence['distance'])):
                    sent_vec[d] += sentence['distance'][d]
        else:
            sent_vec = sent_vec[: self.words_num]
            char_vec = char_vec[: self.words_num]
            sentence['tokens'] = sentence['tokens'][: self.words_num]
            sentence['tags'] = np.array(sentence['tags'])[: self.words_num]
            if 'distance' in sentence:
                # if self.append_distance:
                #     distance = np.zeros([self.words_num, self.dis_dim])
                #     for d in xrange(self.words_num):
                #         distance[d] = sentence['distance'][d]
                #     sent_vec = np.concatenate((sent_vec, distance), axis=1)
                # else:
                for d in xrange(self.words_num):
                    sent_vec[d] += sentence['distance'][d]
        sentence['sent_vec'] = sent_vec
        sentence['char_vec'] = char_vec
        #ipdb.set_trace()
        tmp_tags = np.zeros([self.words_num, self.tag_dim], dtype=np.int32)
        for i in xrange(self.words_num):
            tmp_tags[i] = sentence['tags'][i]
        sentence['tags'] = tmp_tags


    def restart(self, train_flag, init=False):
        if train_flag:
            if init:
                self.train_text_idx = -1
                self.train_epoch_end_flag = False
            self.train_text_idx += 1
            if self.train_text_idx >= len(self.train_data):
                self.train_epoch_end_flag = True
                print('\n\n-----train_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.train_data[self.train_text_idx%self.num_train]
            print('\ntrain_text_idx: %d of %d' % (self.train_text_idx, len(self.train_data)))
        else:
            if init:
                self.valid_text_idx = -1
                self.valid_epoch_end_flag = False
            self.valid_text_idx += 1
            if self.valid_text_idx >= len(self.valid_data):
                self.valid_epoch_end_flag = True
                print('\n\n-----valid_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.valid_data[self.valid_text_idx]
            print('\nvalid_text_idx: %d of %d' % (self.valid_text_idx, len(self.valid_data)))
        #self.current_text.keys() = ['tokens', 'acts', 'tags', 'sent_vec', 'char_vec']
        self.text_vec = np.concatenate(
                (self.current_text['sent_vec'], self.current_text['tags']), axis=1)
        #assert self.text_vec.shape == (self.words_num, self.emb_dim)
        self.state = self.text_vec.copy() # NB!
        self.state[:,self.emb_dim-self.tag_dim:] = 0
        self.terminal_flag = False

        
    def act(self, action, word_idx):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        self.state[word_idx, self.emb_dim-self.tag_dim:] = action + 1
        t_a_count = 0  #amount of tagged actions 
        for t in self.state[:,-1]:
            if t == 2: # extract
                t_a_count += 1
        t_a_rate = float(t_a_count)/self.words_num

        label = self.text_vec[word_idx,-1]
        self.real_action_flag = False
        if self.agent_mode == 'af':
            # text_vec is labelled data
            if label >= 2:
                self.real_action_flag = True
            if label == 2:
                if action == 1:
                    reward = 2.0 * self.reward_base
                else:
                    reward = -2.0 * self.reward_base
            elif label == 4:
                right_flag = True
                if word_idx in self.current_text['obj_idxs'][0]:
                    exc_objs = self.current_text['obj_idxs'][1]
                else:
                    exc_objs = self.current_text['obj_idxs'][0]
                for oi in exc_objs: # exclusive objs
                    if self.state[oi, -1] == 2:
                        right_flag = False
                        break
                if action == 1 and right_flag:
                    reward = 3.0 * self.reward_base
                elif action == 2 and not right_flag:
                    reward = 3.0 * self.reward_base
                elif action == 2 and word_idx != self.current_text['obj_idxs'][1][-1]:
                    reward = 3.0 * self.reward_base
                else:
                    reward = -3.0 * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = 1.0 * self.reward_base
                else:
                    reward = -1.0 * self.reward_base

        else: # self.agent_mode == 'eas'
            if label >= 2:
                self.real_action_flag = True 
            if label == 2: #required action
                if action == 1: # extracted as action
                    reward = 2.0 * self.reward_base
                else: # filtered out
                    reward = -2.0 * self.reward_base
            elif label == 3: #optional action
                if action == 1:
                    reward = 1.0 * self.reward_base
                else:
                    reward = 0.0
            elif label == 4: # exclusive action
                #ipdb.set_trace()
                assert word_idx in self.current_text['act2related']
                exclusive_act_idxs = self.current_text['act2related'][word_idx]
                exclusive_flag = False
                not_biggest_flag = False
                for idx in exclusive_act_idxs:
                    if self.state[idx, -1] == 2: # extracted as action
                        exclusive_flag = True
                    if idx > word_idx:
                        not_biggest_flag = True
                if action == 1 and not exclusive_flag:
                # extract current word and no former exclusive action was extracted
                    reward = 3.0 * self.reward_base
                elif action == 0 and exclusive_flag:
                # filtered out current word because one former exclusive action was extracted
                    reward = 3.0 * self.reward_base
                elif action == 0 and not_biggest_flag:
                # filtered out current word and at least one exclusive action left 
                    reward = 3.0 * self.reward_base
                else:
                    reward = -3.0 * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = 1.0 * self.reward_base
                else:
                    reward = -1.0 * self.reward_base
        
        if self.use_act_rate and reward != 0:
            if t_a_rate <= self.action_rate and reward > 0:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
        # all words of current text are tagged, break
        if word_idx + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True
        
        return reward


    def getState(self):
        '''
        Gets current text state
        '''
        return self.state


    def isTerminal(self):
        '''
        Returns if tag_actions is done
        if all the words of a text have been tagged, then terminate
        '''
        return self.terminal_flag



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='/home/fengwf/Documents/mymodel-new-5-50', help="")
    parser.add_argument("--max_char_len", type=int, default=20, help="")
    parser.add_argument("--words_num", type=int, default=500, help="")
    parser.add_argument("--word_dim", type=int, default=50, help="")
    parser.add_argument("--emb_dim", type=int, default=100, help="")
    parser.add_argument("--tag_dim", type=int, default=50, help="")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    parser.add_argument("--fold_id", type=int, default=0, help="")
    parser.add_argument("--actionDB", default='tag_actions6', help="")
    parser.add_argument("--max_text_num", default='96', help="")
    parser.add_argument("--reward_assign", type=float, default=1.0, help="")
    parser.add_argument("--action_rate", type=float, default=0.15, help="")
    parser.add_argument("--test", type=int, default=1, help="")
    parser.add_argument("--test_text_num", type=int, default=10, help="")
    parser.add_argument("--char_emb_flag", type=int, default=0, help="")
    parser.add_argument("--use_cross_valid", type=int, default=1, help="")
    parser.add_argument("--data_name", default='data/cooking_labeled_text_data2.pkl', help='')
    parser.add_argument("--k_fold_indices", type=str, default='data/cooking_eas_10_fold_indices.pkl', help="")
    parser.add_argument("--agent_mode", default='eas', help='')
    parser.add_argument("--batch_act_num", type=int, default=1, help="")

    args = parser.parse_args()
    from gensim.models import KeyedVectors
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)

    env = Environment(args)
    env.read_eas_texts()
    env.agent_mode = 'af'
    env.words_num = 100
    env.data_name = 'data/new_refined_cooking_data.pkl' 
    env.k_fold_indices = 'data/cooking_af_10_fold_indices.pkl' 
    env.read_af_sents()
    env.data_name = 'data/cooking_labeled_text_data2.pkl' 
    env.k_fold_indices = 'data/cooking_context_10_fold_indices.pkl' 
    '''
    env.train_init()
    a = raw_input('Continue?(y/n)').lower()
    while a != 'n':
        env.restart()
        a = raw_input('Continue?(y/n)').lower()

    env.test_init()
    a = ''
    while a != 'n':
        env.restart_test()
        a = raw_input('Continue?(y/n)').lower()
    '''