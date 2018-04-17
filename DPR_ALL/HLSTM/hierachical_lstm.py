#coding:utf-8
import os
import re
import ipdb
import time
import pickle
import argparse
import mysql.connector
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.contrib import rnn
from gensim.models import KeyedVectors
from utils import ten_fold_split_idx, index2data


class HLSTM(object):
    """docstring for HLSTM"""
    def __init__(self, args, sess):
        self.sess = sess
        self.batch_size = args.batch_size
        self.time_steps = args.time_steps
        self.vec_size = self.hidden_size = self.dis_emb_size = args.vec_size
        self.aux_class_num = self.main_class_num = args.out_dim
        self.keep_prob = args.keep_prob
        self.layer_num = args.layer_num
        self.current_lr = args.init_lr
        self.stop_lr = args.stop_lr
        self.decay = args.decay
        self.max_grad_norm = args.max_grad_norm
        self.epochs = args.epochs
        self.optimizer = args.optimizer
        self.stop_epoch_gap = args.stop_epoch_gap
        self.checkpoint_dir = args.checkpoint_dir
        self.build_model()


    def build_model(self):
        # self.hidden_size = self.vec_size = self.dis_emb_size
        self.aux_inputs = tf.placeholder(tf.float32, [self.batch_size, self.time_steps, self.vec_size])
        self.aux_targets = tf.placeholder(tf.int32, [self.batch_size, self.time_steps])
        self.main_targets = tf.placeholder(tf.int32, [self.batch_size, self.time_steps, self.time_steps])
        self.positions = tf.placeholder(tf.int32, [self.time_steps, self.time_steps])
        distance_embedding = tf.get_variable("dis_emb", [self.time_steps, self.dis_emb_size], dtype=tf.float32)
  
        # ** 1.LSTM 层
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        # ** 2.dropout
        lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)
        # ** 3.多层 LSTM
        cell_fw = rnn.MultiRNNCell([lstm_fw_cell]*self.layer_num, state_is_tuple=True)
        cell_bw = rnn.MultiRNNCell([lstm_bw_cell]*self.layer_num, state_is_tuple=True)
        # ** 4.初始状态
        self.aux_init_state_fw = lstm_fw_cell.zero_state(self.batch_size, tf.float32)
        self.aux_init_state_bw = lstm_bw_cell.zero_state(self.batch_size, tf.float32)
        self.main_init_state_fw = cell_fw.zero_state(self.batch_size*self.time_steps, tf.float32)
        self.main_init_state_bw = cell_bw.zero_state(self.batch_size*self.time_steps, tf.float32) 

        # ** 5. bi-lstm 计算（展开）
        # outputs_fw.shape = [self.batch_size, self.time_steps, self.hidden_size]
        aux_output, aux_output_state_fw, aux_output_state_bw = self.rnn_compution(
            self.aux_inputs, lstm_fw_cell, lstm_bw_cell, self.aux_init_state_fw, self.aux_init_state_bw, 'aux_rnn')
        ipdb.set_trace()
        aux_w = self.weight_variable([self.hidden_size * 2, self.aux_class_num]) 
        aux_b = self.bias_variable([self.aux_class_num]) 
        self.aux_logits = tf.matmul(aux_output, aux_w) + aux_b
        self.a_z = tf.reshape(self.aux_logits, [self.batch_size, self.time_steps, -1])
        self.act_idx = tf.argmax(self.a_z, axis=2) # shape=[batch_size, self.time_steps]

        #self.main_inputs = tf.placeholder(tf.float32, [self.batch_size*self.time_steps, self.hidden_size])
        tmp_inputs = []
        tmp_labels = []
        for t in xrange(self.time_steps):
            dis_emb = tf.nn.embedding_lookup(distance_embedding, self.positions[t])
            for b in xrange(self.batch_size):
                if self.act_idx[b][t] == 1:
                    aux_emb = tf.add(aux_output_state_fw[b], aux_output_state_bw[b])
                    tmp_inputs.append(tf.add(dis_emb, aux_emb))
                    tmp_labels.append(self.main_targets[b][t])
        self.main_inputs = tf.reshape(tmp_inputs, [-1, self.time_steps, self.hidden_size])

        main_output, main_output_state_fw, main_output_state_bw = self.rnn_compution(
            self.main_inputs, cell_fw, cell_bw, self.main_init_state_fw, self.main_init_state_bw, 'main_rnn')
        # output.shape=[self.batch_size,self.time_steps, self.hidden_size*2]
            
            
        main_w = self.weight_variable([self.hidden_size * 2, self.main_class_num]) 
        main_b = self.bias_variable([self.main_class_num]) 
        self.main_logits = tf.matmul(main_output, main_w) + main_b
        self.m_z = tf.reshape(self.main_logits, [self.batch_size, self.time_steps, self.time_steps, -1])
        
        aux_labels = tf.reshape(self.aux_targets, [-1])
        self.aux_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=aux_labels, logits=self.aux_logits))

        #main_labels = tf.reshape(self.main_targets, [-1])
        main_labels = tf.reshape(tmp_labels, [-1])
        self.main_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=main_labels, logits=self.main_logits))

        self.cost = tf.add(self.aux_loss, self.main_loss)
        self.lr = tf.Variable(self.current_lr)
        if self.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adam':    
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        else: 
            self.train_op = tf.train.RMSPropOptimizer(
                0.0025, decay=self.decay, momentum=0.1, epsilon=1e10).minimize(self.cost)
        '''
        # ***** 优化求解 *******
        # 获取模型的所有参数
        tvars = tf.trainable_variables()
        # 获取损失函数对于每个参数的梯度
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # 梯度下降计算
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        '''
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


    def rnn_compution(self, inputs, cell_fw, cell_bw, initial_state_fw, initial_state_bw, name='bi_rnn'):
        with tf.variable_scope(name):
            # *** 下面，两个网络是分别计算 output 和 state 
            # Forward direction
            outputs_fw = list()
            state_fw = initial_state_fw
            with tf.variable_scope('fw'):
                for timestep in xrange(self.time_steps):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_fw, state_fw) = cell_fw(inputs[:, timestep, :], state_fw)
                    outputs_fw.append(output_fw)

            # backward direction
            outputs_bw = list()
            state_bw = initial_state_bw
            with tf.variable_scope('bw') as bw_scope:
                inputs = tf.reverse(inputs, [1])
                for timestep in xrange(self.time_steps):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output_bw, state_bw) = cell_bw(inputs[:, timestep, :], state_bw)
                    outputs_bw.append(output_bw)
            # *** 然后把 output_bw 在 timestep 维度进行翻转
            # outputs_bw.shape = [self.time_steps, batch_size, self.hidden_size]
            outputs_bw = tf.reverse(outputs_bw, [0])
            # output.shape 必须和 y_input.shape=[batch_size,self.time_steps] 对齐
            outputs_fw = tf.transpose(outputs_fw, perm=[1,0,2])
            outputs_bw = tf.transpose(outputs_bw, perm=[1,0,2])
            # 把两个oupputs 拼成 [batch_size, self.time_steps, self.hidden_size*2]
            output = tf.concat([outputs_fw, outputs_bw], 2)  
            #output = tf.transpose(output, perm=[1,0,2])
            # output.shape = [self.batch_size*self.time_steps, self.hidden_size*2]
            output = tf.reshape(output, [-1, self.hidden_size*2])

            return output, outputs_fw, outputs_bw



    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


    def run_epoch(self, data, train_flag, outfile=''):
        sents_vec = np.array(data['sents_vec'])
        acts_tag = np.array(data['act_tag'], dtype=np.int32)
        args_tag = np.array(data['arg_tag'], dtype=np.int32)
        sent_words = data['sent_words']
        sent_len = data['sent_len']
        
        N = len(sents_vec) / self.batch_size
        M = len(sents_vec) % self.batch_size
        if M > 0:
            sents_vec = np.concatenate((sents_vec, 
                np.zeros([self.batch_size-M, self.time_steps, self.vec_size])))
            acts_tag = np.concatenate((acts_tag, 
                np.zeros([self.batch_size-M, self.time_steps])))
            args_tag = np.concatenate((args_tag, 
                np.zeros([self.batch_size-M, self.time_steps, self.time_steps])))
            N += 1

        #ipdb.set_trace()
        a_losses = m_losses = t_losses = 0.0
        t_right = t_total = t_tagged = 0
        a_right = a_total = a_tagged = 0
        m_right = m_total = m_tagged = 0
        positions = np.zeros([args.time_steps, args.time_steps])
        for ii in xrange(args.time_steps):
            tmp_pos = np.zeros(args.time_steps)
            tmp_pos.fill(ii)
            positions[ii] = np.abs(np.arange(args.time_steps) - tmp_pos)

        for idx in xrange(N):
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            inputs = sents_vec[start_idx: end_idx]
            aux_targets = acts_tag[start_idx: end_idx]
            main_targets = args_tag[start_idx: end_idx]
            num_word = sent_len[start_idx: end_idx]
            batch_words = sent_words[start_idx: end_idx]

            feed_dict = {}
            feed_dict[self.aux_inputs] = inputs
            feed_dict[self.aux_targets] = aux_targets
            feed_dict[self.main_targets] = main_targets
            feed_dict[self.positions] = positions
            self.feed_init_state(self.aux_init_state_fw, feed_dict)
            self.feed_init_state(self.aux_init_state_bw, feed_dict)
            self.feed_init_state(self.main_init_state_fw, feed_dict)
            self.feed_init_state(self.main_init_state_bw, feed_dict)
            if train_flag:
                _, loss, aloss, mloss, apred, mpred = self.sess.run(
                                                        [self.train_op, 
                                                        self.cost, 
                                                        self.aux_loss,
                                                        self.main_loss,
                                                        self.a_z,
                                                        self.m_z],
                                                        feed_dict=feed_dict)
            else:
                loss, aloss, mloss, apred, mpred = self.sess.run(
                                                        [self.cost, 
                                                        self.aux_loss,
                                                        self.main_loss,
                                                        self.a_z,
                                                        self.m_z],
                                                        feed_dict=feed_dict)
            t_losses += loss
            a_losses += aloss
            m_losses += mloss
            for b in xrange(len(apred)):
                if b + start_idx >= len(sent_len):
                    break
                #if outfile != '' and b + start_idx < len(sent_words):
                #    outfile.write('\nNO.%d: %s\n' % (b + start_idx, ' '.join(batch_words[b])))
                a_y = np.argmax(apred[b], axis=1) # 0: non-action, 1: action
                for j in xrange(len(a_y)): #(num_word[b]):
                    a_total += sum(aux_targets[b])
                    if aux_targets[b][j] == 1:
                        # find objects for labeled actions
                        m_y = np.argmax(mpred[b][j], axis=1)
                        for k in xrange(len(m_y)):
                            m_total += sum(main_targets[b][j])
                            if m_y[k] == 1:
                                m_tagged += 1
                                if main_targets[b][j][k] == 1:
                                    m_right += 1
                    if a_y[j] == 1:
                        a_tagged += 1
                        if aux_targets[b][j] == 1:
                            a_right += 1
                            #if outfile != '' and b + start_idx < len(sent_words):
                            #    if j < num_word[b]:
                            #        outfile.write('\t'+batch_words[b][j])
        t_total = a_total + m_total
        t_right = a_right + m_right
        t_tagged = a_tagged + m_tagged
        a_rec, a_pre, a_f1 = self.compute_f1_score(a_total, a_right, a_tagged)
        m_rec, m_pre, m_f1 = self.compute_f1_score(m_total, m_right, m_tagged)
        t_rec, t_pre, t_f1 = self.compute_f1_score(t_total, t_right, t_tagged)
        print('a_loss: %f\t a_rec: %f\t a_pre: %f\t a_f1: %f' % (a_losses/N, a_rec, a_pre, a_f1))
        print('m_loss: %f\t m_rec: %f\t m_pre: %f\t m_f1: %f' % (m_losses/N, m_rec, m_pre, m_f1))

        return t_losses/N, t_rec, t_pre, t_f1



    def compute_f1_score(self, total, right, tagged):
        rec = pre = f1 = 0.0
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2*rec*pre / (rec + pre)
        return rec, pre, f1




    def feed_init_state(self, init_name, feed_dict):
        init_value = self.sess.run(init_name)
        if len(init_name) == 2:
            feed_dict[init_name[0]] = init_value[0]
            feed_dict[init_name[1]] = init_value[1]
        else:
            for i, (c, h) in enumerate(init_name):
                feed_dict[c] = init_value[i].c
                feed_dict[h] = init_value[i].h



    def run(self, train_data, valid_data, best_f1s, outfile):
        last_f1 = 0.0
        best_f1 = 0.0
        best_epo = -1
        log_f1s = []
        log_idx = 0
        self.log_loss = []
        for idx in xrange(self.epochs):
            tr_loss, tr_pre, tr_rec, tr_f1 = self.run_epoch(train_data, 1)
            te_loss, te_pre, te_rec, te_f1 = self.run_epoch(valid_data, 0, outfile)

            if best_f1 < te_f1:
                #os.system('rm %s*' % "%sLSTM-f1_%f-model"%(self.checkpoint_dir, best_f1))
                best_f1 = te_f1
                best_epo = idx
                log_idx += 1
                if log_idx %20 == 0:
                    #last_model = "%sLSTM-f1_%f-model" % (self.checkpoint_dir, last_f1)
                    #if os.path.exists(last_model):
                    #    os.system('rm %s*' % last_model)
                    #self.saver.save(self.sess, "%sLSTM-f1_%f-model"%(self.checkpoint_dir, best_f1))
                    last_f1 = best_f1
            self.log_loss.append(te_loss)
            log_f1s.append(te_f1)
            display = {
                'epoch:': idx,
                'train_loss': tr_loss,
                #'train_pre': tr_pre,
                #'train_rec': tr_rec,
                'train_f1': tr_f1,
                'valid_loss': te_loss,
                #'valid_pre': te_pre,
                #'valid_rec': te_rec,
                'valid_f1': te_f1,
                'learning_rate': self.current_lr
            }
            print('{}\n'.format(display))
            outfile.write('\n{}\n'.format(display))
            if self.optimizer == 'sgd':
                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx] > self.log_loss[idx-1] * 0.9999:
                    if self.current_lr > self.stop_lr:
                        self.current_lr *= self.decay #self.current_lr / 1.5
                        self.lr.assign(self.current_lr).eval()
            # stop training if no improvement in self.stop_epoch_gap epoch 
            if idx - best_epo > self.stop_epoch_gap and idx > 200: #train_loss < 1: # 
                self.saver.save(self.sess, "%sHLSTM-f1_%f-model"%(self.checkpoint_dir, best_f1))
                break
        best_f1s.append(best_f1)




def save_data_from_database(args):
    word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    db = mysql.connector.connect(user='fengwf',password='123',database='test')
    cur = db.cursor()
    result =  []
    if type(args.actionDB) == str:
        args.actionDB = args.actionDB.split()
        args.max_text_num = [int(i) for i in args.max_text_num.split()]
    for ind,a in enumerate(args.actionDB):
        for i in xrange(args.max_text_num[ind]):
            get_data = "select * from " + a + ' where text_num = ' + str(i)
            cur.execute(get_data)
            result.append(cur.fetchall())

    data = []
    for j in tqdm(xrange(len(result))):
        text = []
        for k in xrange(len(result[j])):
            words = re.split(r' ',result[j][k][2])
            tags = re.split(r' ',result[j][k][3])
            text.append({'sent': words, 'act_tag': tags})
        data.append(text)
    with open('data/all_texts.pkl', 'wb') as f:
        pickle.dump(data, f)
        print('Successfully save file ... ')


def read_data(args):
    with open(args.out_data_dir, 'rb') as f:
        input_data = pickle.load(f)
    word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    sents_vec = []
    act_tag = []
    arg_tag = []
    sent_len = []
    sent_words = []
    #ipdb.set_trace()
    for sent in input_data:
        num_word = len(sent['sent'])
        words = sent['sent']
        tmp_tags = np.zeros(num_word)
        tmp_sent_vec = []
        for w in words:
            if w in word2vec.vocab:
                word_vec = word2vec[w]
            else:
                word_vec = np.zeros(args.vec_size)
            tmp_sent_vec.append(word_vec)
        tmp_sent_vec = np.array(tmp_sent_vec)
        for act_idx, args_idx in sent['acts'].iteritems():
            tmp_tags[act_idx] = 1
        if num_word < args.time_steps:
            tmp_tags = np.concatenate((tmp_tags, np.zeros(args.time_steps - num_word)))
            tmp_sent_vec = np.concatenate((tmp_sent_vec, 
                                        np.zeros([args.time_steps - num_word, args.vec_size])))
        else:
            tmp_tags = tmp_tags[: args.time_steps]
            words = words[: args.time_steps]
            tmp_sent_vec = tmp_sent_vec[: args.time_steps]
        act_tag.append(tmp_tags)
        sent_len.append(len(words))
        sent_words.append(words)
        sents_vec.append(tmp_sent_vec)
        tmp_arg_tags = []
        for i in xrange(args.time_steps):
            tmp_tags2 = np.zeros(num_word)
            for j in xrange(num_word):
                if i in sent['acts']:
                    tmp_tags2[sent['acts'][i]] = 1
            if num_word < args.time_steps:
                tmp_tags2 = np.concatenate((tmp_tags2, np.zeros(args.time_steps - num_word)))
            else:
                tmp_tags2 = tmp_tags2[: args.time_steps]
            tmp_arg_tags.append(tmp_tags2)
        arg_tag.append(tmp_arg_tags)

    data = {'sents_vec': sents_vec,
            'act_tag': act_tag,
            'arg_tag': arg_tag,
            'sent_words': sent_words,
            'sent_len': sent_len
    }
    print 'Total sents: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds


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
        self.in_data_dir = 'data/all_texts.pkl'
        self.out_data_dir = 'data/new_sents_with_acts.pkl'


    def stanford_find_vp(self):
        ipdb.set_trace()
        if os.path.exists(self.out_data_dir):
            return
        with open(self.in_data_dir, 'rb') as f:
            print('Loading data %s...\n' % self.in_data_dir)
            input_data = pickle.load(f)
        data = []
        for ind, sentences in enumerate(input_data):  
            sents = [' '.join(s['sent']) for s in sentences]
            start = time.time()
            print('\ndependency parsing %d ' % ind)
            try:
                dep = self.dep_parser.raw_parse_sents(sents)
            except Exception as e:
                print(e)
                continue
            end = time.time()
            print('dependency parsing time: %.2fs\n' % (end - start))
            for k in xrange(len(sents)):
                dep_sent_root = dep.next()
                dep_sent = dep_sent_root.next()
                dep_conll = [i.split() for i in str(dep_sent.to_conll(10)).split('\n') if i]
                verb_obj = {}
                verbs = []
                words = [w[1] for w in dep_conll]
                if len(words) != len(sentences[k]['sent']):
                    words = sentences[k]['sent']
                    for i, t in enumerate(sentences[k]['act_tag']):
                        if t == '1':
                            verb_obj[i] = []
                else:
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
                            for one_line in dep_conll:
                                # find the conjunctive relation objects
                                if int(one_line[6]) - 1 == obj_idxs[0] and one_line[7] == 'conj':
                                    #print(one_line)
                                    if int(one_line[0]) - 1 >= len(words):
                                        continue
                                    objs.append(one_line[1])
                                    obj_idxs.append(int(one_line[0]) - 1)
                                    verb_obj[verb_idx].append(obj_idxs[-1])
                    for i, t in enumerate(sentences[k]['act_tag']):
                        if t == '1' and i not in verb_obj:
                            verb_obj[i] = []
                
                data.append({'sent':words, 'acts':verb_obj})
        with open(self.out_data_dir, 'wb') as f:
            print('Saving data %s...\n' % self.out_data_dir)
            pickle.dump(data, f)


if __name__ == '__main__':
    import sys
    reload(sys)
    sys.setdefaultencoding('gb18030')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--text_name", default='test_result.txt', help='')
    parser.add_argument("--idx_fname", default='data/tb0-sents-10-fold-indices.pkl', help='')
    parser.add_argument("--out_data_dir", default='data/new_sents_with_acts.pkl', help='')
    parser.add_argument("--checkpoint_dir", default='checkpoints/', help='')
    parser.add_argument("--actionDB", default='tag_actions', help='')
    parser.add_argument("--max_text_num", default='64', help='')
    parser.add_argument("--batch_size", type=int, default=8, help='')
    parser.add_argument("--time_steps", type=int, default=100, help='')
    parser.add_argument("--vec_size", type=int, default=50, help='')
    parser.add_argument("--out_dim", type=int, default=2, help='')
    parser.add_argument("--layer_num", type=int, default=1, help='')
    parser.add_argument("--keep_prob", type=int, default=1, help='')
    parser.add_argument("--init_lr", type=float, default=1e-4, help='')
    parser.add_argument("--stop_lr", type=float, default=1e-8, help='')
    parser.add_argument("--decay", type=float, default=0.9, help='')
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help='')
    parser.add_argument("--epochs", type=int, default=300, help='')
    parser.add_argument("--stop_epoch_gap", type=int, default=30, help='')
    parser.add_argument("--gpu_rate", type=float, default=0.40, help='')
    parser.add_argument("--optimizer", default='adam', help='')
    args = parser.parse_args()
    
    actionDBs = ['tag_actions', 'tag_actions1', 'tag_actions2', 'tag_actions3',
                'tag_actions5', 'tag_actions6']
    mtns = [64, 52, 33, 54, 35, 43]
    #inputer = ReadData()
    #inputer.stanford_find_vp()
    folds = read_data(args)
    train_data = folds['train'][0]
    valid_data = folds['valid'][0]
    best_f1 = []
    #ipdb.set_trace()
    with open(args.text_name, 'w') as outfile:
        #option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        #config=tf.ConfigProto(gpu_options=option)
        with tf.Session() as sess:
            start = time.time()
            model = HLSTM(args, sess)
            end = time.time()
            print('\n\nBuild model cost time %.2fs\n\n' % (end - start))
            model.run(train_data, valid_data, best_f1, outfile)