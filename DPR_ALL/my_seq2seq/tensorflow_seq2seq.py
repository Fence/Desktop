#coding:utf-8
import os
import re
import ipdb
import time
import pickle
import pprint
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gensim.models import KeyedVectors
from utils import ten_fold_split_idx, index2data

class ActPred(object):
    def __init__(self, args, sess, vocab, idx2word):
        self.sess = sess
        self.vocab = vocab
        self.idx2word = idx2word
        self.batch_size = args.batch_size
        self.vocab_size = args.vocab_size
        self.enc_len = args.enc_len
        self.dec_len = args.dec_len
        self.enc_layers = self.dec_layers = args.num_layers
        self.hidden_size = args.word_dim
        self.num_samples = args.num_samples
        self.optimizer = args.optimizer
        self.current_lr = args.init_lr
        self.stop_lr = args.stop_lr
        self.decay = args.decay
        self.stop_epoch_gap = args.stop_epoch_gap
        self.keep_prob = args.keep_prob
        self.checkpoint_dir = args.checkpoint_dir
        self.start_decay_epoch = args.start_decay_epoch
        self.build_model()


    def build_model(self):
        self.input_ids = tf.placeholder(tf.int32, [self.batch_size, self.enc_len, self.dec_len], 'input_ids')
        self.target_ids = tf.placeholder(tf.int32, [self.batch_size, self.dec_len], 'target_ids')
        self.embedding = tf.get_variable('emb', [self.vocab_size, self.hidden_size], dtype=tf.float32)
        self.enc_input = tf.nn.embedding_lookup(self.embedding, self.input_ids)
        self.dec_target = tf.nn.embedding_lookup(self.embedding, self.target_ids)


        # build action encoder
        enc_act_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        if self.keep_prob < 1: # dropout
            enc_act_cell = tf.contrib.rnn.DropoutWrapper(
                enc_act_cell, output_keep_prob=self.keep_prob)
        enc_act_lstm = tf.contrib.rnn.MultiRNNCell([enc_act_cell] * self.enc_layers)

        self.init_enc_act_state = enc_act_lstm.zero_state(self.batch_size, tf.float32)

        # encoder computation
        #ipdb.set_trace()
        self.enc_act_outputs = []
        state = self.init_enc_act_state # the states of a batch samples
        for act_step in range(self.enc_len):
            tmp_enc_outputs = []
            with tf.variable_scope('enc_act%d' % act_step):
                for time_step in range(self.dec_len):
                    if time_step > 0: 
                        tf.get_variable_scope().reuse_variables()
                    # enc_cell_out: [batch, hidden_size]
                    (enc_cell_output, state) = enc_act_lstm(self.enc_input[:, act_step, time_step, :], state)
                    tmp_enc_outputs.append(tf.reshape(enc_cell_output, [-1, 1, self.hidden_size]))
            # self.enc_act_outputs: [self.enc_len, self.batch_size, 1, self.hidden_size]
            self.enc_act_outputs.append(tmp_enc_outputs[-1])
        # self.enc_act_outputs: [self.batch_size, self.enc_len, self.hidden_size]
        self.enc_act_outputs = tf.transpose(self.enc_act_outputs, perm = [1, 0, 2, 3])
        self.enc_act_outputs = tf.reshape(self.enc_act_outputs, [-1, self.enc_len, self.hidden_size])


        # build plan encoder 
        enc_plan_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        if self.keep_prob < 1: # dropout
            enc_plan_cell = tf.contrib.rnn.DropoutWrapper(
                enc_plan_cell, output_keep_prob=self.keep_prob)
        enc_plan_lstm = tf.contrib.rnn.MultiRNNCell([enc_plan_cell] * self.enc_layers) 
        self.init_enc_plan_state = enc_plan_lstm.zero_state(self.batch_size, tf.float32)

        # encoder computation
        self.enc_plan_outputs = []
        state = self.init_enc_plan_state # the states of a batch samples
        with tf.variable_scope('enc_plan'):
            for time_step in range(self.enc_len):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                # enc_cell_out: [batch, hidden_size]
                (enc_cell_output, state) = enc_plan_lstm(self.enc_act_outputs[:, time_step, :], state)
                self.enc_plan_outputs.append(tf.reshape(enc_cell_output, [-1, 1, self.hidden_size]))
        # enc_output.shape = [batch*enc_len, hidden_size]
        #enc_output = tf.reshape(tf.concat(self.enc_plan_outputs, 1), [-1, self.hidden_size])

        # build decoder
        dec_lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        if self.keep_prob < 1: # dropout
            dec_lstm_cell = tf.contrib.rnn.DropoutWrapper(
                dec_lstm_cell, output_keep_prob=self.keep_prob)
        dec_lstm = tf.contrib.rnn.MultiRNNCell([dec_lstm_cell] * self.dec_layers)
        
        # decoder computation
        self.dec_outputs = []
        #state = tf.reshape(self.enc_plan_outputs[-1], [-1, self.hidden_size])
        with tf.variable_scope('dec'):
            for time_step in range(self.dec_len):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (dec_cell_output, state) = dec_lstm(self.dec_target[:, time_step, :], state)
                self.dec_outputs.append(tf.reshape(dec_cell_output, [-1, 1, self.hidden_size]))
        dec_output = tf.reshape(tf.concat(self.dec_outputs, 1), [-1, self.hidden_size])

        # sampled softmax
        w = tf.get_variable('w', [self.hidden_size, self.vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('b', [self.vocab_size])
        self.z = tf.matmul(dec_output, w) + b
        labels = tf.reshape(self.target_ids, [-1, 1])
        #labels = tf.reshape(self.target_ids, [-1])
        #
        #self.cost = tf.reduce_mean(
        #    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.z))
        self.cost = tf.reduce_sum(tf.nn.sampled_softmax_loss(
                                w_t, b, labels, dec_output, self.num_samples, self.vocab_size))
        #ipdb.set_trace()
        self.lr = tf.Variable(self.current_lr)
        if self.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        elif self.optimizer == 'adam':    
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        else: 
            self.train_op = tf.train.RMSPropOptimizer(
                0.0025, decay=self.decay, momentum=0.1, epsilon=1e-10).minimize(self.cost)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        

    def run_epoch(self, data, train_flag, outfile):
        act_seqs = np.array(data['act_seq'])
        next_acts = np.array(data['next_act'])
        raw_seqs = data['raw_seq']

        N = int(len(act_seqs) / self.batch_size)
        M = len(act_seqs) % self.batch_size
        #ipdb.set_trace()
        if M > 0:
            act_seqs = np.concatenate((act_seqs, 
                        np.zeros([self.batch_size - M, self.enc_len, self.dec_len], dtype=np.int32)))
            next_acts = np.concatenate((next_acts, 
                        np.zeros([self.batch_size - M, self.dec_len], dtype=np.int32)))
            N += 1

        acc = losses = 0.0
        right = total = 0
        
        for idx in range(N):
            start_idx = idx * self.batch_size
            end_idx = (idx + 1) * self.batch_size
            inputs = act_seqs[start_idx: end_idx]
            targets = next_acts[start_idx: end_idx]

            feed_dict = {}
            feed_dict[self.input_ids] = inputs
            feed_dict[self.target_ids] = targets
            init_act_state = self.sess.run(self.init_enc_act_state)
            for i, (c, h) in enumerate(self.init_enc_act_state):
                feed_dict[c] = init_act_state[i].c
                feed_dict[h] = init_act_state[i].h
            init_plan_state = self.sess.run(self.init_enc_plan_state)
            for i, (c, h) in enumerate(self.init_enc_plan_state):
                feed_dict[c] = init_plan_state[i].c
                feed_dict[h] = init_plan_state[i].h

            if train_flag:
                _, loss, pred = self.sess.run([self.train_op, self.cost, self.z],
                                            feed_dict=feed_dict)

            else:
                loss, pred = self.sess.run([self.cost, self.z],
                                            feed_dict=feed_dict)

            losses += loss
            pred = np.reshape(pred, [self.batch_size, self.dec_len, -1])
            pred_y = np.argmax(pred, axis=2)
            total += self.batch_size * self.dec_len
            for b in range(self.batch_size):
                if start_idx + b >= len(raw_seqs):
                    break
                #outfile.write('raw_seq: {}\n'.format(raw_seqs[start_idx + b]))
                for i in range(self.dec_len):
                    if i >= len(raw_seqs[start_idx + b][-1]):
                        break
                    #outfile.write('targets: %s\n' % self.idx2word[targets[b][i]])
                    #outfile.write('predict: %s\n' % self.idx2word[pred_y[b][i]])
                    if pred_y[b][i] == targets[b][i]:
                        right += 1
                #outfile.write('\n\n')

        acc = right / total
        print('right: %d\t  total: %d' % (right, total))
        return acc, losses/N


    def run(self, train_data, valid_data, epochs, outfile):
        last_acc = 0.0
        best_acc = 0.0
        best_epo = -1
        log_idx = 0
        log_acc = []
        self.log_loss = []
        for idx in range(epochs):
            train_acc, train_loss = self.run_epoch(train_data, 1, outfile)
            valid_acc, valid_loss = self.run_epoch(valid_data, 0, outfile)

            if best_acc < valid_acc:
                best_acc = valid_acc
                best_epo = idx
                log_idx += 1
                if log_idx %20 == 0:
                    print('\nTry to save model, log_idx: %d\n' % log_idx)
                    last_model = "%sLSTM-acc_%f-model" % (self.checkpoint_dir, last_acc)
                    if os.path.exists(last_model):
                        os.system('rm %s*' % last_model)
                    self.saver.save(self.sess, "%sLSTM-acc_%f-model"%(self.checkpoint_dir, best_acc))
                    last_acc = best_acc
            self.log_loss.append(valid_loss)
            log_acc.append(valid_acc)
            display = {
                'epoch:': idx,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'learning_rate': self.current_lr
            }
            print('{}\n'.format(display))
            outfile.write('\n{}\n'.format(display))
            if self.optimizer in ['sgd', 'adam'] and idx > self.start_decay_epoch:
                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx] > self.log_loss[idx-1] * 0.9999:
                    if self.current_lr > self.stop_lr:
                        self.current_lr *= self.decay #self.current_lr / 1.5
                        self.lr.assign(self.current_lr).eval()
            # stop training if no improvement in self.stop_epoch_gap epoch 
            if idx - best_epo > self.stop_epoch_gap and idx > 200: #train_loss < 1: # 
                self.saver.save(self.sess, "%sLSTM-acc_%f-model"%(self.checkpoint_dir, best_acc))
                break



def read_data(args):
    with open(args.data_file, 'rb') as f:
        indata = pickle.load(f)

    data_len = {}
    max_enc_len = 0
    max_dec_len = 0
    vocab = {'EOS': 0, 'UNK': 1}
    idx2word = {0: 'EOS', 1:'UNK'}
    data = {'act_seq': [], 'next_act': [], 'raw_seq': []}

    for i in tqdm(xrange(len(indata))):
        text = indata[i]
        if len(text['act_seq']) > max_enc_len:
            max_enc_len = len(text['act_seq'])
        if len(text['act_seq']) < 5 or len(text['act_seq']) > args.enc_len + 1:
            continue
        for act, objs in text['act_seq']:
            if act not in vocab:
                vocab[act] = len(vocab)
                idx2word[len(idx2word)] = act
            for obj in objs.split('_'):
                if obj not in vocab:
                    vocab[obj] = len(vocab)
                    idx2word[len(idx2word)] = obj
    print('vocab size: ', len(vocab))
    args.vocab_size = len(vocab)
    
    for idx in tqdm(xrange(len(indata))):
        text = indata[idx]
        seq_len = len(text['act_seq'])
        if seq_len in data_len:
            data_len[seq_len] += 1
        else:
            data_len[seq_len] = 1
        #continue
        if seq_len < 5 or seq_len > args.enc_len + 1:
            continue

        raw_seq = []
        for act, objs in text['act_seq']:
            tmp_raw_seq = [act]
            tmp_raw_seq.extend(objs.split('_'))
            raw_seq.append(tmp_raw_seq)
        
        tmp_act_seq = []
        tmp_next_act = []
        for i in range(len(raw_seq)):
            tmp_act = []
            tmp_act.append(vocab[raw_seq[i][0]]) # add action word
            for obj in raw_seq[i][1:]:
                tmp_act.append(vocab[obj]) # add object word
            #tmp_act.append(vocab['EOS']) # add EOS flag
            if i < len(raw_seq) - 1:
                tmp_act_seq.append(tmp_act)
            else:
                tmp_next_act = tmp_act
            if len(tmp_act) > max_dec_len:
                max_dec_len = len(tmp_act)
                #print(tmp_act)
 
        for j in range(len(tmp_act_seq)):
            pad_len = args.dec_len - len(tmp_act_seq[j])
            if pad_len > 0:
                for k in range(pad_len):
                    tmp_act_seq[j].append(vocab['EOS'])
            elif pad_len < 0:
                tmp_act_seq[j] = tmp_act_seq[j][: args.dec_len]
                #tmp_act_seq[j].append(vocab['EOS'])
            #tmp_act_seq[j] = np.array(tmp_act_seq[j])
    
        pad_len = args.enc_len - len(tmp_act_seq)
        if pad_len > 0:
            for k in range(pad_len):
                tmp_act_seq.append(np.zeros(args.dec_len, dtype=np.int32))
        elif pad_len < 0:
            tmp_act_seq = tmp_act_seq[: args.enc_len]

        pad_len = args.dec_len - len(tmp_next_act)
        if pad_len > 0:
            for k in range(pad_len):
                tmp_next_act.append(vocab['EOS'])
        elif pad_len < 0:
            tmp_next_act = tmp_next_act[: args.dec_len]

        tmp_act_seq = np.array(tmp_act_seq, dtype=np.int32)
        tmp_next_act = np.array(tmp_next_act, dtype=np.int32)
        data['act_seq'].append(tmp_act_seq)
        data['next_act'].append(tmp_next_act)
        data['raw_seq'].append(raw_seq)
    
    #data_len = sorted(data_len.items(), key=lambda x:x[1], reverse=True)
    #count = 0
    #for k, v in data_len:
    #    count += v
    #    print(k, v, float(v)/len(indata), float(count)/len(indata))
    
    indices = ten_fold_split_idx(len(data['act_seq']), args.idx_fname)
    folds = index2data(indices, data)
    print("len(data['act_seq']): %d  max_enc_len: %d  max_dec_len: %d\n" % 
        (len(data['act_seq']), max_enc_len, max_dec_len))
    ipdb.set_trace()
    return folds, vocab, idx2word





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--text_name", default='results/test_result1.txt', help='')
    parser.add_argument("--idx_fname", default='data/seq_data/wiki_len10_10_fold_indices.pkl', help='')
    parser.add_argument("--data_file", default='/home/fengwf/Documents/DRL_data/wikihow/wikihow_act_seq_refined.pkl', help='')
    parser.add_argument("--checkpoint_dir", default='checkpoints/', help='')
    parser.add_argument("--actionDB", default='tag_actions2', help='')
    parser.add_argument("--vocab_size", type=int, default=1000, help='')
    parser.add_argument("--enc_len", type=int, default=10, help='')
    parser.add_argument("--dec_len", type=int, default=10, help='')
    parser.add_argument("--batch_size", type=int, default=128, help='')
    parser.add_argument("--words_num", type=int, default=500, help='')
    parser.add_argument("--word_dim", type=int, default=50, help='')
    parser.add_argument("--num_samples", type=int, default=512, help='')
    parser.add_argument("--num_layers", type=int, default=1, help='')
    parser.add_argument("--keep_prob", type=int, default=0.5, help='')
    parser.add_argument("--init_lr", type=float, default=0.1, help='')
    parser.add_argument("--stop_lr", type=float, default=1e-4, help='')
    parser.add_argument("--start_decay_epoch", type=int, default=10, help='')
    parser.add_argument("--decay", type=float, default=0.7, help='')
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help='')
    parser.add_argument("--epochs", type=int, default=500, help='')
    parser.add_argument("--stop_epoch_gap", type=int, default=30, help='')
    parser.add_argument("--gpu_rate", type=float, default=0.24, help='')
    parser.add_argument("--optimizer", default='adam', help='')
    args = parser.parse_args()
    
    #ipdb.set_trace()
    folds, vocab, idx2word = read_data(args)
    train_data = folds['train'][-1]
    valid_data = folds['valid'][-1]
    with open(args.text_name, 'w') as outfile:
        option = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        with tf.Session(config=tf.ConfigProto(gpu_options=option)) as sess:
            model = ActPred(args, sess, vocab, idx2word)
            model.run(train_data, valid_data, args.epochs, outfile)
