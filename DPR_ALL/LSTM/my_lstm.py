#coding:utf-8
import os
import re
import ipdb
import time
import pickle
import pprint
import argparse
import mysql.connector
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from utils import ten_fold_split_idx, index2data

class EALSTM(object):
    def __init__(self, args, sess):
        self.sess = sess
        self.batch_size = args.batch_size   # 8
        self.num_steps = args.words_num     # 100
        self.size = args.emb_dim           # 50
        self.target_output = args.target_output         # 2
        self.num_layers = args.num_layers   # lstm layers num
        self.keep_prob = 1 - args.dropout     # use for dropout
        self.current_lr = args.init_lr
        self.stop_lr = args.stop_lr
        self.decay = args.decay
        self.epochs = args.epochs
        self.optimizer = args.optimizer
        self.stop_epoch_gap = args.stop_epoch_gap
        self.checkpoint_dir = args.checkpoint_dir
        self.build_model()


    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.num_steps, self.size])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)
        if self.keep_prob < 1: # dropout
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True) 
        self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        
        self.outputs = []
        state = self.initial_state # the states of a batch samples
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                self.outputs.append(tf.reshape(cell_output, [-1, 1, self.size]))
                #self.outputs.append(cell_output)  # output: shape[num_steps][batch,hidden_size]

        # concatenate self.outputs to [batch, hidden_size*num_steps] 
        # and then reshape to [batch*numsteps, hidden_size]
        #output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size])
        #ipdb.set_trace()
        output = tf.reshape(tf.concat(self.outputs, 1), [-1, self.size])

        # softmax_w , shape=[hidden_size, self.target_output]
        softmax_w = tf.get_variable(
            "softmax_w", [self.size, self.target_output], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.target_output], dtype=tf.float32)
        self.z = tf.matmul(output, softmax_w) + softmax_b
        self.logits = tf.reshape(self.z, [-1, self.num_steps, self.target_output])

        labels = tf.reshape(self.targets, [-1])
        self.lr = tf.Variable(self.current_lr)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.z))
        self.final_state = state
        if self.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adam':    
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        else: 
            self.train_op = tf.train.RMSPropOptimizer(
                0.0025, decay=self.decay, momentum=0.1, epsilon=1e-10).minimize(self.cost)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


    def run_epoch(self, data, train_flag, outfile=''):
        """Runs the model on the given data."""
        sents_vec = np.array(data['sents_vec'])
        sents_tag = np.array(data['sents_tag'])
        sent_words = data['sent_words']
        sent_len = data['sent_len']

        N = len(sents_vec) / self.batch_size
        M = len(sents_vec) % self.batch_size
        if M > 0:
            sents_vec = np.concatenate((sents_vec, 
                np.zeros([self.batch_size - M, self.num_steps, self.size])))
            sents_tag = np.concatenate((sents_tag, 
                np.zeros([self.batch_size - M, self.num_steps])))
            N += 1

        #
        rec = pre = f1 = losses = 0.0
        right = total = tagged = 0
        for idx in xrange(N):
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            inputs = np.array(sents_vec[start_idx: end_idx])
            targets = np.array(sents_tag[start_idx: end_idx])
            num_word = sent_len[start_idx: end_idx]
            batch_words = sent_words[start_idx: end_idx]
            #targets = np.zeros([self.batch_size, self.num_steps, self.target_output])

            feed_dict = {}
            feed_dict[self.inputs] = inputs
            feed_dict[self.targets] = targets
            init_state = self.sess.run(self.initial_state)
            for i, (c, h) in enumerate(self.initial_state):
                feed_dict[c] = init_state[i].c
                feed_dict[h] = init_state[i].h
            if train_flag:
                _, loss, state_out, pred = self.sess.run(
                                        [self.train_op, self.cost, self.final_state, self.logits], 
                                        feed_dict=feed_dict)
            else:
                loss, state_out, pred = self.sess.run(
                                        [self.cost, self.final_state, self.logits], 
                                        feed_dict=feed_dict)
            losses += loss
            for b, p in enumerate(pred):
                #if b + start_idx >= len(sent_len):
                #    break
                #if outfile != '' and b + start_idx < len(sent_words):
                #    outfile.write('\nNO.%d: %s\n' % (b + start_idx, ' '.join(batch_words[b])))
                y = np.argmax(p, axis=1) # 0: non-action, 1: action
                for j in xrange(len(y)): #(num_word[b]):
                    if targets[b][j] == 1:
                        total += 1
                    if y[j] == 1:
                        tagged += 1
                        if targets[b][j] == 1:
                            right += 1
                            #if outfile != '' and b + start_idx < len(sent_words):
                            #    if j < num_word[b]:
                            #        outfile.write('\t'+batch_words[b][j])
        #ipdb.set_trace()
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2*rec*pre / (rec + pre)

        return losses/N, pre, rec, f1


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
                self.saver.save(self.sess, "%sLSTM-f1_%f-model"%(self.checkpoint_dir, best_f1))
                break
        best_f1s.append(best_f1)


def read_sents(args):
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
    
    sents_vec = []
    sents_tag = []
    sent_len = []
    sent_words = []
    for j in xrange(len(result)):
        for k in xrange(len(result[j])):
            tmp_sent_vec = []
            words = re.split(r' ',result[j][k][2])
            tmp_tag_vec = [int(t) for t in re.split(r' ',result[j][k][3])]
            for l, w in enumerate(words):
                if w in word2vec.vocab:
                    word_vec = word2vec[w]
                else:
                    word_vec = np.zeros(args.word_dim)
                tmp_sent_vec.append(word_vec)
            if len(tmp_sent_vec) < args.words_num:
                for i in xrange(args.words_num - len(tmp_sent_vec)):
                    tmp_sent_vec.append(np.zeros(args.word_dim))
                    tmp_tag_vec.append(0)
                sent_words.append(words)
                sent_len.append(len(words))
            else:
                tmp_sent_vec = tmp_sent_vec[: args.words_num]
                tmp_tag_vec = tmp_tag_vec[: args.words_num]
                sent_words.append(words[: args.words_num])
                sent_len.append(args.words_num)
            sents_vec.append(tmp_sent_vec)
            sents_tag.append(tmp_tag_vec)

    data = {'sents_vec': sents_vec,
            'sents_tag': sents_tag,
            'sent_words': sent_words,
            'sent_len': sent_len
    }
    print 'Total sents: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds


def read_texts(args):
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
    
    sents_vec = []
    sents_tag = []
    sent_len = []
    sent_words = []
    for j in xrange(len(result)):
        tmp_sent_vec = []
        tmp_tag_vec = []
        tmp_words = []
        for k in xrange(len(result[j])):
            words = re.split(r' ',result[j][k][2])
            tags = [int(t) for t in re.split(r' ',result[j][k][3])]
            tmp_tag_vec.extend(tags)
            tmp_words.extend(words)
            for l, w in enumerate(words):
                if w in word2vec.vocab:
                    word_vec = word2vec[w]
                else:
                    word_vec = np.zeros(args.word_dim)
                tmp_sent_vec.append(word_vec)

        if len(tmp_sent_vec) < args.words_num:
            for i in xrange(args.words_num - len(tmp_sent_vec)):
                tmp_sent_vec.append(np.zeros(args.word_dim))
                tmp_tag_vec.append(0)
            sent_words.append(words)
            sent_len.append(len(words))
            sents_vec.append(tmp_sent_vec)
            sents_tag.append(tmp_tag_vec)
        else:
            N = len(tmp_sent_vec) / args.words_num
            M = len(tmp_sent_vec) % args.words_num
            if M > 0:
                for m in xrange(args.words_num - M):
                    tmp_sent_vec.append(np.zeros(args.word_dim))
                    tmp_tag_vec.append(0)
                N += 1
            for n in xrange(N):
                si = n * args.words_num
                ei = (n + 1) * args.words_num
                if n != N - 1:
                    sent_words.append(tmp_words[si: ei])
                    sent_len.append(args.words_num)
                else:
                    sent_words.append(words[si: ])
                    sent_len.append(len(words[si: ]))
                sents_vec.append(tmp_sent_vec[si: ei])
                sents_tag.append(tmp_tag_vec[si: ei])

    data = {'sents_vec': sents_vec,
            'sents_tag': sents_tag,
            'sent_words': sent_words,
            'sent_len': sent_len
    }
    print 'Total sents: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds
            

def read_af_sents_old(args):
    word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    with open(args.data_name, 'rb') as f:
        databag = pickle.load(f)

    sents_vec = []
    sents_tag = []
    sents_len = []
    sents_word = []
    for ssa in databag['sents_act_args']:
        sent_vec = []
        for w in ssa['sent']:
            if w in word2vec.vocab:
                sent_vec.append(word2vec[w])
            else:
                sent_vec.append(np.zeros(args.word_dim))
        words = ssa['sent']
        sent_len = len(ssa['sent'])
        for at, ar in ssa['act_args'].iteritems():
            act_idx = at[1]
            obj_idxs = [a[1] for a in ar]
            tags = np.zeros(sent_len, dtype=np.int32)
            tags[obj_idxs] = 1 # obj label = 1
            position = np.zeros(sent_len, dtype=np.int32)
            position.fill(act_idx)
            distance = np.abs(np.arange(sent_len) - position)
            tmp_sent_vec = np.zeros([sent_len, args.word_dim])
            for l in xrange(sent_len):
                tmp_sent_vec[l] = sent_vec[l] + position[l]
            pad_len = args.words_num - sent_len
            if pad_len > 0:
                tmp_sent_vec = np.concatenate((tmp_sent_vec, np.zeros([pad_len, args.word_dim])))
                tags = np.concatenate((tags, np.zeros(pad_len, dtype=np.int32)))
            else:
                tmp_sent_vec = tmp_sent_vec[: args.words_num]
                tags = tags[: args.words_num]
                words = words[: args.words_num]
                sent_len = args.words_num
            sents_vec.append(tmp_sent_vec)
            sents_tag.append(tags)
            sents_len.append(sent_len)
            sents_word.append(words)

    data = {'sents_vec': sents_vec,
            'sents_tag': sents_tag,
            'sent_words': sents_word,
            'sent_len': sents_len
    }
    print 'Total sents: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds


def read_eas_texts(args):
    word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    with open(args.data_name, 'r') as f:
        indata = pickle.load(f)

    sents_vec = []
    sents_tag = []
    sents_len = []
    sents_word = []
    #ipdb.set_trace()
    for i in xrange(len(indata)):
        words = indata[i]['words']
        sent_vec = []
        for w in words:
            if w in word2vec.vocab:
                sent_vec.append(word2vec[w])
            else:
                sent_vec.append(np.zeros(args.word_dim))
        sent_vec = np.array(sent_vec)
        sent_len = len(words)
        tags = np.zeros(sent_len, dtype=np.int32)
        for act_idx in indata[i]['acts']:
            tags[act_idx] = 1

        pad_len = args.words_num - sent_len
        if pad_len > 0:
            sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, args.word_dim])), axis=0)
            tags = np.concatenate((tags, np.zeros(pad_len, dtype=np.int32)), axis=0)
        else:
            sent_vec = sent_vec[: args.words_num]
            tags = tags[: args.words_num]
            words = words[: args.words_num]
            sent_len = len(words)
        
        sents_vec.append(sent_vec)
        sents_tag.append(tags)
        sents_len.append(sent_len)
        sents_word.append(words)

    data = {'sents_vec': sents_vec,
            'sents_tag': sents_tag,
            'sent_words': sents_word,
            'sent_len': sents_len
    }
    print 'Total texts: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds 



def read_af_sents(args):
    word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    with open(args.data_name, 'r') as f:
        indata = pickle.load(f)

    sents_vec = []
    sents_tag = []
    sents_len = []
    sents_word = []
    for i in xrange(len(indata)):
        for j in xrange(len(indata[i])):
            # -1 obj_idx refer to UNK
            words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK'] 
            sent_vec = []
            for w in words:
                if w in word2vec.vocab:
                    sent_vec.append(word2vec[w])
                else:
                    sent_vec.append(np.zeros(args.word_dim))
            sent_vec = np.array(sent_vec)
            sent_len = len(words)
            act_idxs = []
            for act_idx, obj_idxs in indata[i][j]['acts'].items():
                tmp_sent_vec = sent_vec.copy()
                act_idxs.append(act_idx)
                tags = np.zeros(sent_len, dtype=np.int32)
                tags[obj_idxs] = 1
                position = np.zeros(sent_len, dtype=np.int32)
                position.fill(act_idx)
                distance = np.abs(np.arange(sent_len) - position)

                if args.use_act_tags:
                    act_tags = np.zeros([sent_len,  args.tag_dim], dtype=np.int32)
                    act_tags.fill(3)
                    for ai in indata[i][j]['acts']:
                        act_tags[ai] = 4
                    tmp_sent_vec = np.concatenate((tmp_sent_vec, act_tags), axis=1)

                pad_len = args.words_num - sent_len
                if pad_len > 0:
                    tmp_distance = np.zeros([args.words_num, args.dis_dim])
                    for l in xrange(sent_len):
                        tmp_distance[l] = distance[l]
                    tmp_sent_vec = np.concatenate((tmp_sent_vec, np.zeros([pad_len, tmp_sent_vec.shape[1]])), axis=0)
                    tmp_sent_vec = np.concatenate((tmp_sent_vec, tmp_distance), axis=1)
                    tags = np.concatenate((tags, np.zeros(pad_len, dtype=np.int32)))
                else:
                    tmp_distance = np.zeros([args.words_num, args.dis_dim])
                    for l in xrange(args.words_num):
                        tmp_distance[l] = distance[l]
                    tmp_sent_vec = np.concatenate((tmp_sent_vec[: args.words_num], tmp_distance), axis=1)
                    tags = tags[: args.words_num]
                    words = words[: args.words_num]
                    sent_len = len(words)
                
                sents_vec.append(tmp_sent_vec)
                sents_tag.append(tags)
                sents_len.append(sent_len)
                sents_word.append(words)    
                
            if args.negative_sampling:
                obj_idxs = []
                for k in xrange(len(indata[i][j]['acts'])):
                    while True:
                        act_idx = np.random.randint(sent_len)
                        if act_idx not in act_idxs:
                            act_idxs.append(act_idx)
                            break
                    tmp_sent_vec = sent_vec.copy()
                    tags = np.zeros(sent_len, dtype=np.int32)
                    tags[obj_idxs] = 1
                    position = np.zeros(sent_len, dtype=np.int32)
                    position.fill(act_idx)
                    distance = np.abs(np.arange(sent_len) - position)
                    
                    if args.use_act_tags:
                        act_tags = np.zeros([sent_len,  args.tag_dim], dtype=np.int32)
                        act_tags.fill(3)
                        for ai in indata[i][j]['acts']:
                            act_tags[ai] = 4
                        tmp_sent_vec = np.concatenate((tmp_sent_vec, act_tags), axis=1)

                    pad_len = args.words_num - sent_len
                    if pad_len > 0:
                        tmp_distance = np.zeros([args.words_num, args.dis_dim])
                        for l in xrange(sent_len):
                            tmp_distance[l] = distance[l]
                        tmp_sent_vec = np.concatenate((tmp_sent_vec, np.zeros([pad_len, tmp_sent_vec.shape[1]])), axis=0)
                        tmp_sent_vec = np.concatenate((tmp_sent_vec, tmp_distance), axis=1)
                        tags = np.concatenate((tags, np.zeros(pad_len, dtype=np.int32)))
                    
                    sents_vec.append(tmp_sent_vec)
                    sents_tag.append(tags)
                    sents_len.append(sent_len)
                    sents_word.append(words)

    data = {'sents_vec': sents_vec,
            'sents_tag': sents_tag,
            'sent_words': sents_word,
            'sent_len': sents_len
    }
    print 'Total sents: %d' % len(sents_vec)
    indices = ten_fold_split_idx(len(sents_vec), args.idx_fname)
    folds = index2data(indices, data)
    return folds


def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--text_name", default='test_result.txt', help='')
    parser.add_argument("--idx_fname", default='data/cooking_eas_10_fold_indices.pkl', help='')
    parser.add_argument("--data_name", default='/home/fengwf/Documents/DRL_data/cooking/cooking_labeled_text_data2.pkl', help='')
    parser.add_argument("--checkpoint_dir", default='checkpoints/eas_checkpoints', help='')
    parser.add_argument("--actionDB", default='tag_actions2', help='')
    parser.add_argument("--max_text_num", default='33', help='')
    parser.add_argument("--batch_size", type=int, default=8, help='')
    parser.add_argument("--words_num", type=int, default=500, help='')
    parser.add_argument("--word_dim", type=int, default=50, help='')
    parser.add_argument("--dis_dim", type=int, default=50, help='')
    parser.add_argument("--tag_dim", type=int, default=50, help='')
    parser.add_argument("--target_output", type=int, default=2, help='')
    parser.add_argument("--num_layers", type=int, default=2, help='')
    parser.add_argument("--dropout", type=int, default=0.25, help='')
    parser.add_argument("--init_lr", type=float, default=1e-4, help='')
    parser.add_argument("--stop_lr", type=float, default=1e-8, help='')
    parser.add_argument("--decay", type=float, default=0.9, help='')
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help='')
    parser.add_argument("--epochs", type=int, default=200, help='')
    parser.add_argument("--stop_epoch_gap", type=int, default=10, help='')
    parser.add_argument("--gpu_rate", type=float, default=0.20, help='')
    parser.add_argument("--optimizer", type=str, default='adam', help='')
    parser.add_argument("--use_act_tags", type=int, default=0, help='')
    parser.add_argument("--negative_sampling", type=int, default=0, help='')
    parser.add_argument("--append_distance", type=int, default=1, help='')
    parser.add_argument("--agent_mode", type=str, default='eas', help='')
    args = parser.parse_args()
    return args


def main(args):
    total_time = 0
    if args.agent_mode == 'af':
        args.emb_dim = args.word_dim + args.dis_dim
        if args.use_act_tags:
            args.emb_dim += args.tag_dim
        args.words_num = 100
        args.batch_size = 32
        args.data_name = '/home/fengwf/Documents/DRL_data/cooking/new_refined_cooking_data2.pkl'
        if args.negative_sampling:
            args.idx_fname = 'data/neg_cooking_af_10_fold_indices.pkl'
        else:
            args.idx_fname = 'data/cooking_af_10_fold_indices.pkl'
        args.checkpoint_dir = 'checkpoints/af_checkpoints'
        folds = read_af_sents(args)
    else:
        args.emb_dim = args.word_dim
        folds = read_eas_texts(args)
    best_f1s = []
    for i in xrange(10):
        start = time.time()
        pp = pprint.PrettyPrinter()
        pp.pprint(args.__dict__)
        train_data = folds['train'][i]
        valid_data = folds['valid'][i]
        text_name = 'results/%s/drop0.25_ep200_10_fold%d.txt' % (args.agent_mode, i)
        with open(text_name, 'w') as outfile:
            for k,v in args.__dict__.iteritems():
                    outfile.write('{}: {}\n'.format(k, v))

            option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
            with tf.Session(config=tf.ConfigProto(gpu_options=option)) as sess:
                model = EALSTM(args, sess)
                model.run(train_data, valid_data, best_f1s, outfile)

            tf.reset_default_graph() 
            end = time.time()
            total_time += (end - start)
            print('\n\nbest_f1s: {}'.format(best_f1s))
            print('best_f1: {}'.format(best_f1s[-1]))
            print('avg_f1: {}'.format(sum(best_f1s)/len(best_f1s)))
            print('text_name: %s' % text_name)
            print('Fold %d time cost: %ds\n' % (i, end - start))
            outfile.write('\n\nbest_f1s: {}\n'.format(best_f1s))
            outfile.write('best_f1_value: {}\n'.format(best_f1s[-1]))
            outfile.write('average_f1_value: {}\n'.format(sum(best_f1s)/len(best_f1s)))
            outfile.write('text_name: %s\n' % text_name)
            outfile.write('Fold %d time cost: %ds\n' % (i, end - start))
        print('text_name: %s' % text_name)
        print(best_f1s)
        print('Average f1: %f' % (sum(best_f1s)/len(best_f1s)))
        print('Total time cost: %ds\n' % total_time)


def main_old(args):
    total_time = 0
    #ipdb.set_trace()
    actionDBs = ['tag_actions', 'tag_actions1', 'tag_actions2', 'tag_actions3',
                'tag_actions5', 'tag_actions6']
    mtns = [64, 52, 33, 54, 35, 43]
    for tb in xrange(8, 9):
        args.actionDB = actionDBs #[actionDBs[tb]]
        args.max_text_num = mtns #[mtns[tb]]
        if args.words_num == 100: # take sents as sequence
            args.batch_size = 128
            args.idx_fname = 'data/tb%d-sents-10-fold-indices.pkl' % tb
            args.checkpoint_dir = 'checkpoints/sents_tb%d_' % tb
            folds = read_sents(args)
        else: # take texts as sequence
            args.batch_size = 8
            args.idx_fname = 'data/tb%d-texts-10-fold-indices.pkl' % tb
            args.checkpoint_dir = 'checkpoints/texts_tb%d_' % tb
            folds = read_texts(args)
        best_f1s = []
        for i in xrange(10):
            start = time.time()
            train_data = folds['train'][i]
            valid_data = folds['valid'][i]
            if args.words_num == 100:
                text_name = 'results/tb%d/test_new_sents_fold%d.txt' % (tb, i)
            else:
                text_name = 'results/tb%d/test_new_texts_fold%d.txt' % (tb, i)

            with open(text_name, 'w') as outfile:
                for k,v in args.__dict__.iteritems():
                    outfile.write('{}: {}\n'.format(k, v))

                option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
                with tf.Session(config=tf.ConfigProto(gpu_options=option)) as sess:
                    model = EALSTM(args, sess)
                    model.run(train_data, valid_data, best_f1s, outfile)

                tf.reset_default_graph() 
                end = time.time()
                total_time += (end - start)
                print('\n\nbest_f1s: {}'.format(best_f1s))
                print('best_f1: {}'.format(best_f1s[-1]))
                print('avg_f1: {}'.format(sum(best_f1s)/len(best_f1s)))
                print('text_name: %s' % text_name)
                print('Fold %d time cost: %ds\n' % (i, end - start))
                outfile.write('\n\nbest_f1s: {}\n'.format(best_f1s))
                outfile.write('best_f1_value: {}\n'.format(best_f1s[-1]))
                outfile.write('average_f1_value: {}\n'.format(sum(best_f1s)/len(best_f1s)))
                outfile.write('text_name: %s\n' % text_name)
                outfile.write('Fold %d time cost: %ds\n' % (i, end - start))
        print('text_name: %s' % text_name)
        print(best_f1s)
        print('Average f1: %f' % (sum(best_f1s)/len(best_f1s)))
        print('Total time cost: %ds\n' % total_time)


if __name__ == '__main__':
    #import sys
    #reload(sys)
    #sys.setdefaultencoding('gb18030')
    #main_old(args_init())
    main(args_init())
