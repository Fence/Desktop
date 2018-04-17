#coding:utf-8
import re
import ipdb
import time
import pickle
import mysql.connector
import numpy as np
from tqdm import tqdm
from utils import ten_fold_split_idx, index2data

#create table tag_actions5(text_num int, sent_num int, af_sent varchar(400), tag_sent varchar(400));
class Environment:
    def __init__(self, args):
        print('\n\nInitializing the Environment...')  
        self.data_name = args.data_name
        self.max_act_num = args.max_act_num
        self.max_obj_num = args.max_obj_num
        self.ten_fold_indices = args.ten_fold_indices
        self.fold_id = args.fold_id

        self.emb_dim = args.emb_dim
        self.tag_dim = args.tag_dim
        self.word_dim = args.word_dim
        self.word2vec = args.word2vec
        self.words_num = args.words_num
        self.word2vec = args.word2vec
        self.reward_base = args.reward_base

        #self.empty_act_label = np.zeros(self.word_dim)
        #self.empty_obj_label = np.zeros(self.word_dim)
        #self.empty_act_label.fill(args.empty_act_label)
        #self.empty_obj_label.fill(args.empty_obj_label)
        self.act_vocab_size = args.act_vocab_size = args.dynamic_act_vocab_size
        self.obj_vocab_size = args.obj_vocab_size = args.dynamic_obj_vocab_size
        self.max_act_len = 0
        self.max_text_len = 0
        self.read_win2k_texts()
        #ipdb.set_trace()


    def read_win2k_texts(self):
        print('Preparing datasets...')
        with open(self.data_name, 'r') as f:
            indata = pickle.load(f)
        
        # build vocabulary
        self.all_act_words = []
        self.all_obj_words = []
        self.all_words = 0
        self.in_vocab_words = 0
        for i in xrange(len(indata)):
            for act, obj_line in indata[i]['act_seq']:
                #if act not in self.all_act_words:
                self.all_act_words.append(act)
                objs = obj_line.split('_')
                for obj in objs:
                    #if obj not in self.all_obj_words:
                    self.all_obj_words.append(obj)
        self.all_act_words = list(set(self.all_act_words))
        self.all_obj_words = list(set(self.all_obj_words))
        self.word2index = {}
        for w in self.all_act_words + ['EOS'] + self.all_obj_words + ['UNK']:
            if w not in self.word2index:
                self.word2index[w] = len(self.word2index)
        self.index2word = {v:k for k,v in self.word2index.iteritems()}
        
        # create data matrix
        act_seqs = []
        f = open(self.data_name + '.txt', 'w')
        #ipdb.set_trace()
        for i in tqdm(xrange(len(indata))):
            act_seq = {'acts': [], 'objs': []}
            for act, obj_line in indata[i]['act_seq']:
                objs = obj_line.split('_')
                act_seq['acts'].append(act)
                act_seq['objs'].append(objs)
            self.create_matrix(act_seq)
            act_seqs.append(act_seq)
            f.write('\n\nText: %d\n' % i)
            for k, v in act_seq.iteritems():
                f.write('{}:\n{}\n'.format(k, v))
        f.close()
        self.in_vocab_words_rate = float(self.in_vocab_words) / self.all_words
        indices = ten_fold_split_idx(len(act_seqs), self.ten_fold_indices)
        folds = index2data(indices, act_seqs)
        self.train_data = folds['train'][self.fold_id]
        self.valid_data = folds['valid'][self.fold_id]
        self.train_steps = len(self.train_data) * self.words_num * self.max_act_num
        self.valid_steps = len(self.valid_data) * self.words_num * self.max_act_num
        print('\nin-vocab-word rate: %f' % self.in_vocab_words_rate)
        print('action vocab: %d \tobject vocab: %d' % (len(self.all_act_words), len(self.all_obj_words)))
        print('max action num: %d \tmax text len: %d' % (self.max_act_len, self.max_text_len))
        print('training texts: %d \tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('self.train_steps: %d \tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))


    def clustering_actions(self):
        # clustering all actions in training
        pass


    def create_matrix(self, act_seq):
        seq_vec = []
        tmp_objs = []
        for i, act in enumerate(act_seq['acts']):
            #ipdb.set_trace()
            act_vec = np.zeros([self.max_obj_num, self.emb_dim])
            if act in self.word2vec.vocab:
                act_vec[0, : self.word_dim] = self.word2vec[act]
            act_vec[0, self.word_dim: ] = self.word2index[act] # action index
            
            for j, obj in enumerate(act_seq['objs'][i]):
                if j + 1 >= self.max_obj_num - 1:
                    break
                if obj in self.word2vec.vocab:
                    act_vec[j + 1, : self.word_dim] = self.word2vec[obj]
                act_vec[j + 1, self.word_dim: ] = self.word2index[obj] # object index
            seq_vec.append(act_vec)
            tmp_objs.append(act_seq['objs'][i][: j+1])
            if len(seq_vec) * self.max_obj_num > self.words_num - self.max_obj_num:
                break
        #assert len(tmp_objs) == len(act_seq['objs'])
        assert len(seq_vec) > 0
        act_num = len(seq_vec)
        seq_vec = np.concatenate(seq_vec, axis=0)
        text_len = len(seq_vec)
        if act_num > self.max_act_len:
            self.max_act_len = act_num
        if text_len > self.max_text_len:
            self.max_text_len = text_len
        pad_len = self.words_num - len(seq_vec)
        seq_vec = np.concatenate((seq_vec, np.zeros([pad_len, self.word_dim + self.tag_dim])), axis=0)
        act_seq['acts'] = act_seq['acts'][: act_num]
        act_seq['objs'] = tmp_objs
        act_seq['text_len'] = text_len
        act_seq['seq_vec'] = seq_vec

        # compute average action words vector and object words vector
        count_act = 0;    act2idx = {}
        count_obj = 0;    obj2idx = {'EOS': 0}
        avg_act_vec = np.zeros(self.word_dim)
        avg_obj_vec = np.zeros(self.word_dim)
        for act in act_seq['acts'][: -1]: # the last one left for prediction
            if act not in act2idx:
                act2idx[act] = len(act2idx)
            if act in self.word2vec.vocab:
                count_act += 1
                avg_act_vec += self.word2vec[act]
        for i in xrange(len(act_seq['objs']) - 1):  # the last one left for prediction
            for obj in act_seq['objs'][i]:
                if obj not in obj2idx:
                    obj2idx[obj] = len(obj2idx)
                if obj in self.word2vec.vocab:
                    count_obj += 1
                    avg_obj_vec += self.word2vec[obj]

        avg_act_vec /= count_act
        avg_obj_vec /= count_obj
        obj_words = []
        similar_words = []
        topn_act = self.act_vocab_size + len(self.all_obj_words)
        topn_obj = self.obj_vocab_size + len(self.all_act_words)
        
        # find most similar words of object words, and store to dynamic vocab
        selected_acts = self.word2vec.similar_by_vector(avg_act_vec, topn=topn_act)
        selected_objs = self.word2vec.similar_by_vector(avg_obj_vec, topn=topn_obj)
        for word, sorce in selected_acts:
            if len(act2idx) >= self.act_vocab_size:
                break
            # filter out the object words
            if word not in self.all_obj_words and word not in act2idx:
                act2idx[word] = len(act2idx)
                
        for word, sorce in selected_objs:
            if len(obj2idx) >= self.obj_vocab_size:
                break
            # filter out the action words
            if word not in self.all_act_words and word not in obj2idx:
                obj2idx[word] = len(obj2idx)
    
        act_seq['act2idx'] = act2idx
        act_seq['idx2act'] = idx2act = {v:k for k,v in act2idx.iteritems()}
        act_seq['obj2idx'] = obj2idx
        act_seq['idx2obj'] = idx2obj = {v:k for k,v in obj2idx.iteritems()}
        assert len(act2idx) == self.act_vocab_size
        assert len(obj2idx) == self.obj_vocab_size
        contain = 0
        words_count_of_prediction = 0
        words_count_of_prediction += 1
        if act_seq['acts'][-1] in act2idx:
            contain += 1
        for w in tmp_objs[i]:
            words_count_of_prediction += 1
            if w in obj2idx:
                contain += 1
        
        act_seq['IVWR'] = float(contain) / words_count_of_prediction # in-vocabulary-word rate
        self.all_words += words_count_of_prediction
        self.in_vocab_words += contain
        #assert seq_vec.shape == (self.words_num, self.emb_dim)


                
    def create_matrix_old(self, act_seq):
        act_vec = np.zeros([self.max_act_num, self.word_dim])
        obj_vec = np.zeros([self.max_act_num, self.max_obj_num, self.word_dim])
        for i, act in enumerate(act_seq['acts']):
            if act in self.word2vec.vocab:
                act_vec[i] = self.word2vec[act]
        for i, objs in enumerate(act_seq['objs']):
            for j, obj in enumerate(objs):
                if obj in self.word2vec.vocab:
                    obj_vec[i, j] = self.word2vec[obj]
        act_seq['act_vec'] = []
        act_seq['obj_vec'] = []
        for i in xrange(len(act_seq['acts'])):
            tmp_act_vec = act_vec.copy()
            tmp_obj_vec = obj_vec.copy()
            tmp_act_vec[i] = self.empty_act_label
            tmp_obj_vec[i] = self.empty_obj_label
            act_seq['act_vec'].append(tmp_act_vec)
            act_seq['obj_vec'].append(tmp_obj_vec)



    def restart(self, train_flag, init=False):
        if train_flag:
            if init:
                self.train_text_idx = -1
                self.epoch_end_flag = False
            
            self.train_text_idx += 1
            if self.train_text_idx >= len(self.train_data):
                self.epoch_end_flag = True
                #print('\n\n-----epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.train_data[self.train_text_idx]
            if self.train_text_idx % 50 == 0:
                print('\ntrain_text_idx: %d of %d' % (self.train_text_idx, len(self.train_data)))
            self.act_idx = len(self.current_text['acts']) - 1
        else:
            if init:
                self.valid_text_idx = -1
                self.epoch_end_flag = False

            self.valid_text_idx += 1
            if self.valid_text_idx >= len(self.valid_data):
                self.epoch_end_flag = True
                #print('\n\n-----epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.valid_data[self.valid_text_idx]
            if self.valid_text_idx % 50 == 0:
                print('\nvalid_text_idx: %d of %d' % (self.valid_text_idx, len(self.valid_data)))
            self.act_idx = len(self.current_text['acts']) - 1
        self.transition()
        self.terminal_flag = False
    


    def restart_old(self, train_flag, init=False):
        if train_flag:
            if init:
                self.act_idx = -1
                self.train_text_idx = 0
                self.epoch_end_flag = False
                self.current_text = self.train_data[self.train_text_idx]
                print('\ntrain_text_idx: %d of %d' % (self.train_text_idx, len(self.train_data)))
            
            self.act_idx += 1
            if self.act_idx >= len(self.current_text['acts']):
                self.act_idx = 0
                self.train_text_idx += 1
                if self.train_text_idx >= len(self.train_data):
                    self.epoch_end_flag = True
                    #print('\n\n-----epoch_end_flag = True-----\n\n')
                    return
                self.current_text = self.train_data[self.train_text_idx]
                print('\ntrain_text_idx: %d of %d' % (self.train_text_idx, len(self.train_data)))
            #self.act_idx = len(self.current_text['acts']) - 1
            #print('\ntrain_text_idx: %d of %d  act_idx: %d of %d' % 
            #    (self.train_text_idx, len(self.train_data), self.act_idx, len(self.current_text['acts'])))
        else:
            if init:
                self.act_idx = -1
                self.valid_text_idx = 0
                self.epoch_end_flag = False
                self.current_text = self.valid_data[self.valid_text_idx]
                print('\nvalid_text_idx: %d of %d' % (self.valid_text_idx, len(self.valid_data)))
            
            self.act_idx += 1
            if self.act_idx >= len(self.current_text['acts']):
                self.act_idx = 0
                self.valid_text_idx += 1
                if self.valid_text_idx >= len(self.valid_data):
                    self.epoch_end_flag = True
                    #print('\n\n-----epoch_end_flag = True-----\n\n')
                    return
                self.current_text = self.valid_data[self.valid_text_idx]
                print('\nvalid_text_idx: %d of %d' % (self.valid_text_idx, len(self.valid_data)))
            #self.act_idx = len(self.current_text['acts']) - 1
            #print('\nvalid_text_idx: %d of %d  act_idx: %d of %d' % 
            #    (self.valid_text_idx, len(self.valid_data), self.act_idx, len(self.current_text['acts'])))
        self.transition()
        self.terminal_flag = False
        '''
        for i in xrange(len(tmp_objs)):
            for w in tmp_objs[i]:
                if w not in obj_words:
                    obj_words.append(w)
                    if w in self.word2vec.vocab:
                        selected_list = self.word2vec.similar_by_word(w, topn=topn_obj)
                        similar_words.append([w] + [word for word, sorce in selected_list])
        break_flag = False
        for j in xrange(topn_obj + 1):
            for i in xrange(len(similar_words)):
                w = similar_words[i][j]
                if w not in self.all_act_words and w not in obj2idx:
                    obj2idx[w] = len(obj2idx)
                    idx2obj[len(idx2obj)] = w
                    if len(obj2idx) >= self.obj_vocab_size:
                        break_flag = True
                        break
            if break_flag:
                break
        '''



    def transition(self):
        self.text_vec = self.current_text['seq_vec']
        self.start_idx = self.act_idx * self.max_obj_num
        self.end_idx = (self.act_idx + 1) * self.max_obj_num
        self.state = self.text_vec.copy()
        self.state[self.start_idx: self.end_idx] = 0
        # change action word to its index in action vocabulary
        self.real_act = self.current_text['acts'][self.act_idx]
        self.real_act_idx = self.word2index[self.real_act] 
        self.real_objs = self.current_text['objs'][self.act_idx]
        self.real_objs_idx = [self.word2index[obj] for obj in self.real_objs]
        self.tagging_obj_idx = 0
        self.real_tags = [self.real_act_idx] + self.real_objs_idx


    def act(self, action, flag, step, pred_10, train_flag):
        '''
        Performs action and returns reward
        even num refers to tagging action, odd num refer to non-action
        '''
        #ipdb.set_trace()
        # selecting an action 
        if step >= self.max_obj_num:
            self.terminal_flag = True
            return 0

        assert flag in ['act', 'obj']
        if flag == 'act':
            assert action < self.act_vocab_size
            #word_idx = action
            pred_words = [self.current_text['idx2act'][idx] for idx in pred_10]
            pred_10 = []
            for w in pred_words:
                if w in self.word2index:
                    pred_10.append(self.word2index[w])
                else:
                    pred_10.append(self.word2index['UNK'])
            act_word = self.current_text['idx2act'][action] #self.index2word[word_idx]
            if act_word in self.word2index:
                word_idx = self.word2index[act_word] 
            else:
                word_idx = self.word2index['UNK']
            self.state[self.start_idx + step, self.word_dim: ] = word_idx
            #ipdb.set_trace()
            if act_word in self.word2vec.vocab:
                self.state[self.start_idx + step, : self.word_dim] = self.word2vec[act_word]
            else:
                self.state[self.start_idx + step, : self.word_dim] = -1
            # reset the tagging obj idx to 0
            self.tagging_obj_idx = 0
            if word_idx == self.real_act_idx:
                reward = self.reward_base
            else:
                if act_word in self.word2vec.vocab and self.real_act in self.word2vec:
                    similarity = self.word2vec.similarity(act_word, self.real_act)
                    # 0 <= similarity <= 1
                    reward = (similarity - 0.5) * self.reward_base
                else:
                    reward = - self.reward_base
        
        if flag == 'obj':
            assert action >= self.act_vocab_size
            # change word index to object index
            pred_words = [self.current_text['idx2obj'][idx] for idx in pred_10]
            pred_10 = []
            for w in pred_words:
                if w in self.word2index:
                    pred_10.append(self.word2index[w])
                else:
                    pred_10.append(self.word2index['UNK'])
            obj_word = self.current_text['idx2obj'][action - self.act_vocab_size]
            if obj_word in self.word2index:
                word_idx = self.word2index[obj_word]
            else:
                word_idx = self.word2index['UNK']
            self.state[self.start_idx + step, self.word_dim: ] = word_idx
            if obj_word in self.word2vec.vocab:
                self.state[self.start_idx + step, : self.word_dim] = self.word2vec[obj_word]
            else:
                self.state[self.start_idx + step, : self.word_dim] = -1
            # compute reward
            if self.tagging_obj_idx >= len(self.real_objs_idx):
                if word_idx == self.word2index['EOS']:
                    reward = self.reward_base
                else:
                    reward = - self.reward_base
                self.terminal_flag = True
            else:
                if word_idx == self.real_objs_idx[self.tagging_obj_idx]:
                    reward = self.reward_base
                else:
                    if obj_word in self.word2vec.vocab and self.real_objs[self.tagging_obj_idx] in self.word2vec:
                        similarity = self.word2vec.similarity(obj_word, self.real_objs[self.tagging_obj_idx])
                        # 0 <= similarity <= 1
                        reward = (similarity - 0.5) * self.reward_base
                    else:
                        reward = - self.reward_base
            self.tagging_obj_idx += 1
            if train_flag: # if training, stop when meets padding values
                if self.tagging_obj_idx >= len(self.real_objs_idx):
                    self.terminal_flag = True
            else: # if testing, stop when meets EOS
                if self.tagging_obj_idx >= self.max_obj_num - 1:
                    self.terminal_flag = True
                #if word_idx == self.word2index['EOS']:
                #    self.terminal_flag = True

        return reward, pred_10, pred_words



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
    parser.add_argument("--action_label", type=int, default=2, help="")
    parser.add_argument("--non_action_label", type=int, default=1, help="")
    parser.add_argument("--test", type=int, default=1, help="")
    parser.add_argument("--test_text_num", type=int, default=10, help="")
    parser.add_argument("--char_emb_flag", type=int, default=0, help="")
    parser.add_argument("--ten_fold_valid", type=int, default=1, help="")
    parser.add_argument("--data_name", default='data/cooking_labeled_text_data2.pkl', help='')
    parser.add_argument("--ten_fold_indices", type=str, default='data/cooking_eas_10_fold_indices.pkl', help="")
    parser.add_argument("--add_obj_flag", type=int, default=0, help="")
    parser.add_argument("--agent_mode", default='eas', help='')
    parser.add_argument("--af_context", type=int, default=0, help='')

    args = parser.parse_args()
    from gensim.models import KeyedVectors
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)

    env = Environment(args)