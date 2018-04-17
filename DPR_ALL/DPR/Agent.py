#coding:utf-8
import ipdb
import time
import random
import numpy as np
from tqdm import tqdm

class Agent:
    def __init__(self, environment, replay_memory, deep_q_network, args):
        print('Initializing the Agent...')
        self.env = environment
        self.mem = replay_memory
        self.net = deep_q_network
        self.words_num = args.words_num
        self.batch_size = args.batch_size

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        
        self.steps = 0  #use to decrease the reward during time steps 
        self.action_label = args.action_label
        self.random_play = args.random_play
        self.epoch_end_flag = False
        self.agent_mode = args.agent_mode
        self.dqn_mode = args.dqn_mode

        self.act_vocab_size = args.act_vocab_size
        self.obj_vocab_size = args.obj_vocab_size
        self.num_actions = args.act_vocab_size + args.obj_vocab_size
        self.top_list_num = args.top_list_num
        self.hit_num = self.top_list_num[1]
        self.outfile = None

    
    def _restart(self, train_flag, init=False):
        #print('\nRestarting in agent, train_flag = {}, init = {}'.format(train_flag, init))
        self.steps = 0
        self.pred_10s = []
        self.pred_words = []
        self.env.restart(train_flag, init)
        self.epoch_end_flag = self.env.epoch_end_flag


    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
            (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end
 

    def step(self, exploration_rate, train_flag=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            if self.steps == 0:
                flag = 'act'
                action = np.random.randint(self.act_vocab_size)
                pred_10 = [action]
            else:
                flag = 'obj'
                action = np.random.randint(self.act_vocab_size, self.num_actions)
                pred_10 = [action - self.act_vocab_size]
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env.getState()
            qvalues = self.net.predict(current_state)
            assert len(qvalues[0]) == self.num_actions
            if self.steps == 0:
                flag = 'act'
                # the first item of action vocab is EOA, so start from 1
                action = np.argmax(qvalues[0][self.act_vocab_size])
                # argsort sort num from small to large
                pred_10 = np.argsort(qvalues[0][self.act_vocab_size])[-1:-(self.hit_num+1):-1]
                #ipdb.set_trace()
            else:
                flag = 'obj'
                action = np.argmax(qvalues[0][self.act_vocab_size: ])
                action += self.act_vocab_size
                pred_10 = np.argsort(qvalues[0][self.act_vocab_size: ])[-1:-(self.hit_num+1):-1]
                #pred_10 += self.act_vocab_size

        # perform the action  
        reward, pred_10, pred_words = self.env.act(action, flag, self.steps, pred_10, train_flag=True)
        state = self.env.getState()
        terminal = self.env.isTerminal()
        
        self.pred_10s.append(pred_10)
        self.pred_words.append(pred_words)
        self.steps += 1
        results = []
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.steps/(1.5*self.num_actions)
            pass  
        else:
            if not train_flag:
                results = self.compute_acc(display=False)
            reward += 2   #give a bonus to the terminal actions
            self._restart(train_flag)

        return action, reward, state, terminal, results

 
    def train(self, train_steps, epoch = 0):
        '''
        Play given number of steps
        '''
        #ipdb.set_trace()
        start = time.time()
        print('\n{}  epoch: {}  {}'.format('-'*10, epoch, '-'*10))
        self._restart(train_flag=True, init=True)
        for i in xrange(train_steps):
            if self.dqn_mode == 'lstm':
                if (i + 1) % 100 == 0:
                    end = time.time()
                    print('\n\nlast_time_cost %.2fs  epoch: %d  Training step: %d' % (end-start, epoch, i))
                    start = time.time()

            if self.random_play:
                action, reward, state, terminal, _ = self.step(1, train_flag=True)
            else:
                action, reward, state, terminal, _ = self.step(self._explorationRate(), train_flag=True)
                self.mem.add(action, reward, state, terminal)

                # Update target network every target_steps steps
                if self.target_steps and i % self.target_steps == 0:
                    self.net.update_target_network()

                # train after every train_frequency steps
                if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
                    # train for train_repeat times
                    for j in xrange(self.train_repeat):
                        # sample minibatch
                        minibatch = self.mem.getMinibatch()
                        # train the network
                        self.net.train(minibatch)
            
            # increase number of training steps for epsilon decay
            self.total_train_steps += 1
            if self.epoch_end_flag:
                break
    

    def test(self, test_steps, outfile):
        '''
        Play given number of steps
        '''
        self.outfile = outfile
        total = hit1 = hit5 = hit10 = 0
        acc1 = acc5 = acc10 = 0.0
        
        self._restart(train_flag=False, init=True)
        for i in xrange(test_steps):
            if self.random_play:
                a, r, s, t, rs = self.step(1, train_flag=False)
            else:
                a, r, s, t, rs = self.step(self.exploration_rate_test, train_flag=False)
            if t:
                tmp_total, h1, a1, h5, a5, h10, a10 = rs
                total += tmp_total
                hit1 += h1
                hit5 += h5
                hit10 += h10

            if self.epoch_end_flag:
                break   

        acc1 = float(hit1) / total
        acc5 = float(hit5) / total
        acc10 = float(hit10) / total

        outfile.write('\n\nSummary:\ttotal: {}\n'.format(total))
        outfile.write('right: {}\tacc: {}\n'.format(hit1, acc1))
        outfile.write('hit5: {}\thit5_acc: {}\n'.format(hit5, acc5))
        outfile.write('hit10: {}\thit10_acc: {}\n'.format(hit10, acc10))  
        print('\n\nSummary:\t total: {}'.format(total))
        print('right: {}\t acc: {}'.format(hit1, acc1))
        print('hit5: {}\t hit5_acc: {}'.format(hit5, acc5))
        print('hit10: {}\t hit10_acc: {}'.format(hit10, acc10))  

        return acc1, acc5, acc10


    def compute_acc(self, display):
        """
        Compute f1 score for current text
        """
        #ipdb.set_trace()
        state_tags = self.env.state[self.env.start_idx: self.env.end_idx, -1]
        total = len(self.env.real_tags)
        hit1 = hit5 = hit10 = 0
        acc1 = acc5 = acc10 = 0.0
        assert len(self.pred_10s) == total
        real_words = [self.env.real_act] + self.env.real_objs
        pred_words = self.pred_words #[]
        
        
        for s in xrange(total):
            #pred_words.append(self.env.idx2word[state_tags[s]])
            if state_tags[s] == self.env.real_tags[s]:
                hit1 += 1
            if self.env.real_tags[s] in self.pred_10s[s]:
                hit10 += 1
                if self.env.real_tags[s] in self.pred_10s[s][: self.top_list_num[0]]:
                    hit5 += 1

        acc1 = float(hit1) / total
        acc5 = float(hit5) / total
        acc10 = float(hit10) / total
        if display:
            print('Total: {}'.format(total))
            print('right: {}\tacc: {}'.format(hit1, acc1))
            print('hit5: {}\thit5_acc: {}'.format(hit5, acc5))
            print('hit10: {}\thit10_acc: {}'.format(hit10, acc10))
        if self.outfile != None:
            self.outfile.write('\nIn-vocabulary-word rate: {}\n'.format(self.env.current_text['IVWR']))
            self.outfile.write('train_idx: {}  valid_idx: {}\n'.format(self.env.train_text_idx, self.env.valid_text_idx))
            self.outfile.write('real words:\n{}\npred words:\n{}\n'.format(real_words, pred_words))  

        return total, hit1, acc1, hit5, acc5, hit10, acc10