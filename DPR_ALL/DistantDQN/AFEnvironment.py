#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np

#create table tag_actions5(text_num int, sent_num int, sent varchar(400), tag_sent varchar(400));
class AFEnvironment:
    def __init__(self, args):
        print('Initializing the AFEnvironment...')  
        self.context_len = args.context_len
        self.words_num = self.context_len
        self.word_dim = args.word_dim
        self.dis_dim = args.dis_dim
        self.act_dim = args.act_dim
        self.tag_dim = args.tag_dim
        self.emb_dim = args.emb_dim
        if args.append_distance:
            self.emb_dim += self.dis_dim
        self.object_rate = args.object_rate
        self.reward_base = args.reward_assign

        self.append_distance = args.append_distance
        self.max_char_len = args.max_char_len
        self.word2vec = args.word2vec
        self.terminal_flag = False
        self.max_data_char_len = 0
        self.max_data_sent_len = 0
        self.char_info()
    

    def init_predict_af_sents(self, act_idx, text):
        self.terminal_flag = False
        sents = text['sents']
        word2sent = text['word2sent']
        sent_idx = word2sent[act_idx][0]
        word_ids = []
        this_sent = sents[sent_idx]
        if sent_idx > 0: # use the former sentence and current one
            last_sent = sents[sent_idx - 1]
            for k, v in word2sent.iteritems():
                if v[0] == sent_idx or v[0] == sent_idx - 1:
                    word_ids.append(k)
        else:
            last_sent = []
            for k, v in word2sent.iteritems():
                if v[0] == sent_idx:
                    word_ids.append(k)
        words = last_sent + this_sent + ['UNKNOWN_TOKEN']
        end_idx = max(word_ids) # the last index of words of these two sents
        start_idx = min(word_ids)
        sent_len = len(words)

        position = np.zeros(sent_len, dtype=np.int32)
        position.fill(act_idx - start_idx)
        distance = np.abs(np.arange(sent_len) - position)
        sent_vec = np.zeros([self.context_len, self.emb_dim])
        for i, w in enumerate(words):
            if i >= self.context_len:
                break
            if w in self.word2vec.vocab:
                sent_vec[i][: self.word_dim] = self.word2vec[w]
            sent_vec[i][self.word_dim: self.word_dim + self.dis_dim] = distance[i]
        self.state = sent_vec
        self.current_text = {'tokens': words, 'word2sent': word2sent, 'distance': distance}
        return last_sent, this_sent


    def act_af_sents(self, action, word_idx):
        self.state[word_idx, self.emb_dim-self.tag_dim:] = action + 1
        if word_idx + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True


    def char_info(self):
        #chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,;!?:._'\"+-*/@#$%"
        chars = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
        self.char2idx = {}
        for c in chars:
            self.char2idx[c] = len(self.char2idx)


    def restart(self, act_idx, env_act):
        self.terminal_flag = False
        self.real_action_flag = False
        self.total_punish_double_wrong = 0
        self.total_punish_right_wrong = 0
        sents = env_act.current_text['sents']
        sent_acts = env_act.current_text['sent_acts']
        word2sent = env_act.current_text['word2sent']
        #ipdb.set_trace()
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
        af_tags = np.ones(sent_len, dtype=np.int32)
        act_idxs = []
        obj_idxs = [[], []]
        for act in acts: # this is a tagged right action
            if act['act_idx'] < self.words_num:
                act_idxs.append(act['act_idx'])
            if act['act_idx'] != act_idx - start_idx:
                continue
            obj_idxs = act['obj_idxs']
            if len(obj_idxs[1]) == 0:
                af_tags[obj_idxs[0]] = 2 # essential objects
            else:
                af_tags[obj_idxs[0]] = 4 # exclusive objects
                af_tags[obj_idxs[1]] = 4 # exclusive objects
            self.real_action_flag = True
            break

        if env_act.real_action_flag != self.real_action_flag:
            ipdb.set_trace()
        # if not self.real_action_flag:
        #     ipdb.set_trace()

        position = np.zeros(sent_len, dtype=np.int32)
        position.fill(act_idx - start_idx)
        distance = list(np.abs(np.arange(sent_len) - position))

        af_sent['tokens'] = words
        af_sent['tags'] = af_tags
        af_sent['distance'] = distance
        af_sent['sent_idx'] = sent_idx
        af_sent['act_idxs'] = act_idxs
        af_sent['obj_idxs'] = obj_idxs
        af_sent['act_idx'] = act_idx
        af_sent['action'] = env_act.current_text['tokens'][act_idx]
        self.create_matrix(af_sent)
        self.current_text = af_sent
        self.text_vec = np.concatenate((af_sent['sent_vec'], af_sent['tags']), axis=1)
        #assert self.text_vec.shape == (self.words_num, self.emb_dim)
        #ipdb.set_trace()

        self.state = self.text_vec.copy() # NB!
        self.state[:,self.emb_dim-self.tag_dim:] = 0


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
                if self.append_distance:
                    distance = np.zeros([self.words_num, self.dis_dim])
                    for d in xrange(len(sentence['distance'])):
                        distance[d] = sentence['distance'][d]
                    sent_vec = np.concatenate((sent_vec, distance), axis=1)
                else:
                    for d in xrange(len(sentence['distance'])):
                        sent_vec[d] += sentence['distance'][d]
        else:
            sent_vec = sent_vec[: self.words_num]
            char_vec = char_vec[: self.words_num]
            sentence['tokens'] = sentence['tokens'][: self.words_num]
            sentence['tags'] = np.array(sentence['tags'])[: self.words_num]
            if 'distance' in sentence:
                if self.append_distance:
                    distance = np.zeros([self.words_num, self.dis_dim])
                    for d in xrange(self.words_num):
                        distance[d] = sentence['distance'][d]
                    sent_vec = np.concatenate((sent_vec, distance), axis=1)
                else:
                    for d in xrange(self.words_num):
                        sent_vec[d] += sentence['distance'][d]
        sentence['sent_vec'] = sent_vec
        sentence['char_vec'] = char_vec
        #ipdb.set_trace()
        tmp_tags = np.zeros([self.words_num, self.tag_dim], dtype=np.int32)
        for i in xrange(self.words_num):
            tmp_tags[i] = sentence['tags'][i]
        sentence['tags'] = tmp_tags

        
    def act(self, action, word_idx):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        self.punish_double_wrong = self.punish_right_wrong = 0
        self.state[word_idx, self.emb_dim-self.tag_dim:] = action + 1
        t_o_count = 0  #amount of tagged actions 
        for t in self.state[:,-1]:
            if t == 2:# extract
                t_o_count += 1
        t_o_rate = float(t_o_count)/self.words_num

        label = self.text_vec[word_idx,-1]
        # text_vec is labelled data
        if label >= 2:
            self.real_action_flag = True
        if label == 2:
            if action == 1:
                reward = 2.0 * self.reward_base
            else:
                reward = -2.0 * self.reward_base
                self.punish_right_wrong = 1
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
                self.punish_right_wrong = 1
        else: #if label == 1: # non_action 
            if action == 0:
                reward = 1.0 * self.reward_base
            else:
                reward = -1.0 * self.reward_base
                if not self.real_action_flag: # wrong action but extracted object
                    self.punish_double_wrong = 1
                
        if t_o_rate <= self.object_rate:
            reward += 5.0 * np.square(t_o_rate) * self.reward_base
        else:
            reward -= 5.0 * np.square(t_o_rate) * self.reward_base
        # all words of current text are tagged, break
        if word_idx + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True
        self.total_punish_right_wrong += self.punish_right_wrong  
        self.total_punish_double_wrong += self.punish_double_wrong
        
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