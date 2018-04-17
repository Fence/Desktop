#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np
from utils import ten_fold_split_idx, index2data

#create table tag_actions5(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
class AFEnvironment:
    def __init__(self, args):
        print('Initializing the AFEnvironment...')  
        self.context_len = args.context_len
        self.words_num = self.context_len
        self.word_dim = args.word_dim
        self.tag_dim = args.tag_dim
        self.emb_dim = args.emb_dim
        if args.use_act_tags:
            self.emb_dim += tag_dim
        self.object_rate = args.object_rate
        self.object_label = args.action_label
        self.non_object_label = args.non_action_label
        self.reward_base = args.reward_assign
        self.reward_assign = self.reward_base * np.array([2, 1, -1, -2])
        self.punish_false_neg, self.punish_false_pos = self.reward_base * np.array([0.5, 0.5])
        
        self.batch_act_num = args.batch_act_num
        self.use_act_tags = args.use_act_tags
        self.max_char_len = args.max_char_len
        self.word2vec = args.word2vec
        self.terminal_flag = False
        self.max_data_char_len = 0
        self.max_data_sent_len = 0
        self.char_info()
    

    def char_info(self):
        #chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,;!?:._'\"+-*/@#$%"
        chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
        self.char2idx = {}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)


    def restart(self, act_idx, env_act):
        self.terminal_flag = False
        self.real_action_flag = False
        self.total_punish_fp = 0.0
        self.total_punish_fn = 0.0
        sents = env_act.current_text['sents']
        sent_acts = env_act.current_text['sent_acts']
        word2sent = env_act.current_text['word2sent']
        if act_idx not in word2sent:
            ipdb.set_trace()
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
        
        af_sent = {}
        tags = np.ones(sent_len, dtype=np.int32)
        if act_idx - start_idx in acts: # this is a tagged right action
            obj_idxs = acts[act_idx - start_idx]
            if 'NULL' in obj_idxs:
                obj_idxs[obj_idxs.index('NULL')] = sent_len - 1
            tags[obj_idxs] = 2
            self.real_action_flag = True
        if env_act.real_action_flag != self.real_action_flag:
            ipdb.set_trace()

        position = np.zeros(sent_len, dtype=np.int32)
        position.fill(act_idx - start_idx)
        distance = list(np.abs(np.arange(sent_len) - position))
        if self.use_act_tags:
            act_tags = np.zeros([sent_len,  self.tag_dim], dtype=np.int32)
            act_tags.fill(3)
            for ai in indata[i][j]['acts']:
                act_tags[ai] = 4 # action label
            af_sent['act_tags'] = act_tags

        af_sent['tokens'] = words
        af_sent['tags'] = tags
        af_sent['distance'] = distance
        af_sent['sent_idx'] = sent_idx
        af_sent['act_idx'] = act_idx
        af_sent['action'] = env_act.current_text['tokens'][act_idx]
        self.create_matrix(af_sent)
        self.current_text = af_sent
        self.text_vec = np.concatenate((af_sent['sent_vec'], af_sent['tags']), axis=1)
        assert self.text_vec.shape == (self.words_num, self.emb_dim)

        self.state = self.text_vec.copy() # NB!
        self.state[:,self.word_dim:] = 0


    def create_matrix(self, sentence):
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
            char_vec.append([self.char2idx[c] for c in w])
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
                for d in xrange(len(sentence['distance'])):
                    sent_vec[d] += sentence['distance'][d]
            if 'act_tags' in sentence:
                tmp_act_tags = np.concatenate((sentence['act_tags'], np.zeros([pad_len, self.tag_dim])))
                sent_vec = np.concatenate((sent_vec, tmp_act_tags), axis=1)
        else:
            sent_vec = sent_vec[: self.words_num]
            char_vec = char_vec[: self.words_num]
            sentence['tags'] = np.array(sentence['tags'])[: self.words_num]
            if 'distance' in sentence:
                for d in xrange(self.words_num):
                    sent_vec[d] += sentence['distance'][d]
            if 'act_tags' in sentence:
                tmp_act_tags = sentence['act_tags'][: self.words_num]
                sent_vec = np.concatenate((sent_vec, tmp_act_tags), axis=1)
        sentence['sent_vec'] = sent_vec
        sentence['char_vec'] = char_vec
        #ipdb.set_trace()
        tmp_tags = np.zeros([self.words_num, self.tag_dim], dtype=np.int32)
        for i in xrange(self.words_num):
            tmp_tags[i] = sentence['tags'][i]
        sentence['tags'] = tmp_tags


    def context_restart(self, act_idx, current_text, init=False):
        #ipdb.set_trace()
        if init:
            self.terminal_flag = False
            context = np.zeros([self.words_num, self.word_dim])
            tags = np.ones([self.words_num, self.tag_dim])
            for i in range(act_idx - self.context_len, act_idx + self.context_len + 1):
                if 0 <= i < self.word_dim:
                    context[i + self.context_len] = current_text['sent_vec'][i]
            if act_idx in current_text['acts']: # real action
                for obj_idx in current_text['acts'][act_idx]:
                    if obj_idx == 'NULL':
                        continue
                    if 0 <= obj_idx - act_idx + self.context_len < self.words_num:
                        tags[obj_idx - act_idx + self.context_len] = 2 #self.object_label
            self.text_vec = np.concatenate((context, tags), axis=1)
        self.state = self.text_vec.copy() # NB!
        self.state[:,self.word_dim:] = 0

        
    def act(self, action, steps):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        act_str = bin(action)[2:]
        if len(act_str) < self.batch_act_num:
            act_str = '0'*(self.batch_act_num-len(act_str)) + act_str
            act_str = act_str[::-1]
        assert len(act_str) == self.batch_act_num
        #print(act_str)
        bacth_act_reward = 0.0
        for i in range(self.batch_act_num):
            word_idx = steps * self.batch_act_num + i
            # all words of current text are tagged, break
            if word_idx >= len(self.current_text['tokens']):
                self.terminal_flag = True
                #ipdb.set_trace()
                break
            if act_str[i] == '1':  
                self.state[word_idx,self.word_dim:] = self.object_label  
            else: # act_str[i] == '1'
                self.state[word_idx,self.word_dim:] = self.non_object_label    
            t_a_count = 0  #amount of tagged actions 
            for t in self.state[:,-1]:
                if t == self.object_label:
                    t_a_count += 1
            t_a_rate = float(t_a_count)/self.words_num

            label = self.text_vec[word_idx,-1]
            # text_vec is labelled data
            if label == self.state[word_idx,-1]:
                if label == self.object_label:
                    reward = self.reward_assign[0]
                else:
                    reward = self.reward_assign[1] 
            else:
                if self.text_vec[word_idx,-1] == self.non_object_label:
                    reward = self.reward_assign[2]
                    # the input action is real action, but tagged wrong objects, punish it 
                    if self.real_action_flag: 
                        reward -= self.punish_false_neg
                        self.total_punish_fn += self.punish_false_neg
                else: # self.text_vec[word_idx,-1] == self.object_label
                    reward = self.reward_assign[3]
                    if not self.real_action_flag:
                        reward -= self.punish_false_pos
                        self.total_punish_fp += self.punish_false_pos
            if t_a_rate <= self.object_rate:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
            bacth_act_reward += reward
        # all words of current text are tagged, break
        #if word_idx >= self.words_num - 1:
        #    self.terminal_flag = True

        return bacth_act_reward


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
    parser.add_argument("--object_rate", type=float, default=0.15, help="")
    parser.add_argument("--action_label", type=int, default=2, help="")
    parser.add_argument("--non_action_label", type=int, default=1, help="")
    parser.add_argument("--test", type=int, default=1, help="")
    parser.add_argument("--test_text_num", type=int, default=10, help="")
    parser.add_argument("--char_emb_flag", type=int, default=0, help="")
    parser.add_argument("--ten_fold_valid", type=int, default=1, help="")
    parser.add_argument("--ten_fold_indices", type=str, default='data/cooking-10-fold-indices.pkl', help="")


    args = parser.parse_args()
    from gensim.models import KeyedVectors
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    env = Environment(args)
    env.read_pkl_texts()
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