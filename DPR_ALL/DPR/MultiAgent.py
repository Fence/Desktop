#coding:utf-8
import ipdb
import time
import random
import numpy as np
from tqdm import tqdm

class MultiAgent:
    def __init__(self, env_act, env_dv, mem_act, mem_dv, net_act, net_dv, args):
        print('Initializing the Agent...')
        self.env_act = env_act
        self.env_dv = env_dv
        self.mem_act = mem_act
        self.mem_dv = mem_dv
        self.net_act = net_act
        self.net_dv = net_dv
        self.num_actions = args.num_actions
        self.words_num = args.words_num

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_act_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        
        self.act_steps = 0  
        self.obj_steps = 0
        self.epoch_end_flag = False
        self.max_act_num = args.max_act_num
        self.max_obj_num = args.max_obj_num
        self.act_vocab_size = args.act_vocab_size
        self.obj_vocab_size = args.obj_vocab_size
        self.total_obj_steps = args.start_epoch * args.train_steps
        
    
    def act_restart(self, train_flag, init=False):
        #print('\nRestarting in agent, train_flag = {}, init = {}'.format(train_flag, init))
        self.act_steps = 0
        self.env_act.restart(train_flag, init)
        self.epoch_end_flag = self.env_act.epoch_end_flag


    def obj_restart(self, act_idx):
        self.obj_steps = 0
        self.env_dv.restart(act_idx, self.env_act)


    def explorationRate(self, steps):
        # calculate decaying exploration rate
        if steps < self.exp_decay_steps:
            return self.exp_rate_start - steps * (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end
 

    def act_step(self, exploration_rate, train_flag=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(self.act_vocab_size)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env_act.getState()
            qvalues = self.net_act.predict(current_state)
            action = np.argmax(qvalues[0])
            assert len(qvalues[0]) == self.act_vocab_size
            
        # perform the action  
        reward = self.env_act.act(action, self.act_steps)
        state = self.env_act.getState()
        terminal = self.env_act.isTerminal()
        
        self.act_steps += 1
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.act_steps/(1.5*self.num_actions)
            results = []  
        else:
            results = self.compute_f1(self.env_act, 'act')
            self.act_steps = 0
            #reward += 2   #give a bonus to the terminal actions
            self.act_restart(train_flag)

        return action, reward, state, terminal, results


    def obj_step(self, exploration_rate, compute=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(self.obj_vocab_size)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env_dv.getState()
            qvalues = self.net_dv.predict(current_state)
            action = np.argmax(qvalues[0])
            assert len(qvalues[0]) == self.obj_vocab_size
            
        # perform the action  
        reward = self.env_dv.act(action, self.obj_steps)
        state = self.env_dv.getState()
        terminal = self.env_dv.isTerminal()
        
        results = []
        self.obj_steps += 1
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.obj_steps/(1.5*self.num_actions)
            pass  
        else:
            if compute:
                results = self.compute_f1(self.env_dv, 'obj')
            self.obj_steps = 0
            reward += 2   #give a bonus to the terminal actions
            #self.obj_restart()

        return action, reward, state, terminal, results

 
    def train(self, train_steps, epoch = 0):
        '''
        Play given number of steps
        '''
        #ipdb.set_trace()
        self.act_restart(train_flag=True, init=True)
        for i in xrange(train_steps):
            #if i % 50 == 0:
            #    print('\n\nepoch: %d  Training step: %d' % (epoch,i))
            act_idx = self.act_steps
            #if len(self.env_act.current_text['tokens']) <= act_idx:
            #    ipdb.set_trace()
            action, reward, state, terminal, _ = self.act_step(
                                            self.explorationRate(self.total_act_steps), train_flag=True)

            if (epoch > 0 or i > 1000) and not terminal:
                #print(action, np.random.random(), self.explorationRate(self.total_act_steps))
                if action == 1 or np.random.random() <= self.explorationRate(self.total_act_steps):
                    if i % 10 == 0:
                        compute = True
                    else:
                        compute = False
                    for k in xrange(self.max_obj_num):
                        #print(k, i)
                        if k == 0:
                            self.obj_restart(act_idx)
                        action2, reward2, state2, terminal2, _ = self.obj_step(
                                                        self.explorationRate(self.total_obj_steps), compute)

                        #print(action2)
                        self.mem_dv.add(action2, reward2, state2, terminal2)
                        if self.target_steps and k % self.target_steps == 0:
                            self.net_dv.update_target_network()

                        if self.mem_dv.count > self.mem_dv.batch_size and i % self.train_frequency == 0:
                            minibatch2 = self.mem_dv.getMinibatch()
                            self.net_dv.train(minibatch2)
                        self.total_obj_steps += 1
                        #print(self.obj_steps)
                        if self.obj_steps == 0:
                            break

                    if self.env_act.real_action_flag != self.env_dv.real_action_flag:
                        ipdb.set_trace()
                    #assert self.env_act.real_action_flag == self.env_dv.real_action_flag
                    if self.env_act.real_action_flag and self.env_dv.total_punish_fn > 0:
                        print('Total punish false negative: %f' % self.env_dv.total_punish_fn)
                        reward += self.env_dv.total_punish_fn
                    elif not self.env_act.real_action_flag and self.env_dv.total_punish_fp > 0:
                        print('Total punish false positive: %f' % self.env_dv.total_punish_fp)
                        reward -= self.env_dv.total_punish_fp

            self.mem_act.add(action, reward, state, terminal)

            # Update target network every target_steps steps
            if self.target_steps and i % self.target_steps == 0:
                self.net_act.update_target_network()

            # train after every train_frequency steps
            if self.mem_act.count > self.mem_act.batch_size and i % self.train_frequency == 0:
                # train for train_repeat times
                for j in xrange(self.train_repeat):
                    # sample minibatch
                    minibatch = self.mem_act.getMinibatch()
                    # train the network
                    self.net_act.train(minibatch)
            # increase number of training steps for epsilon decay
            self.total_act_steps += 1
            if self.epoch_end_flag:
                break
    

    def test(self, test_steps, f1):
        '''
        Play given number of steps
        '''
        t_right_tag = t_right_acts = t_tagged_acts = t_total_acts = t_words = 0
        t_acc = t_rec = t_pre = t_f1_value = 0.0
        
        self.act_restart(train_flag=False, init=True)
        for i in xrange(test_steps):
            act_idx = self.act_steps
            a, r, s, t, rs = self.act_step(self.exploration_rate_test, train_flag=False)
            if a == 1:
                for k in xrange(self.max_obj_num):
                    if k == 0:
                        self.obj_restart(act_idx)
                    a2, r2, s2, t2, _ = self.obj_step(self.exploration_rate_test, True)
                #self.env_act.add_obj_state(state2, act_idx)
            if t:
                temp_words, total_acts, tagged_acts, right_acts, right_tag, acc, rec, pre, f1_value = rs
                f1.write('\nText: %d\ntotal words: %d\n' % (self.env_act.valid_text_idx - 1, temp_words))
                f1.write('total: %d\tright: %d\ttagged: %d\n' % (total_acts, right_acts, tagged_acts))  
                f1.write('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (acc, rec, pre, f1_value))

                t_words += temp_words
                t_right_tag += right_tag                  
                t_right_acts += right_acts
                t_tagged_acts += tagged_acts
                t_total_acts += total_acts    

            if self.epoch_end_flag:
                break   

        t_acc = float(t_right_tag)/t_words
        if t_total_acts > 0:
            t_rec = float(t_right_acts)/t_total_acts
        if t_tagged_acts > 0:
            t_pre = float(t_right_acts)/t_tagged_acts
        if t_rec + t_pre > 0:
            t_f1_value = (2.0 * t_rec * t_pre)/(t_rec + t_pre)

        f1.write('\n\nSummary:\n')
        f1.write('total: %d\tright: %d\ttagged: %d\n' % (t_total_acts, t_right_acts, t_tagged_acts))  
        f1.write('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (t_acc, t_rec, t_pre, t_f1_value))
        print('\n\nSummary:')
        print('total: %d\tright: %d\ttagged: %d' % (t_total_acts, t_right_acts, t_tagged_acts))  
        print('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (t_acc, t_rec, t_pre, t_f1_value))

        return t_rec, t_pre, t_f1_value


    def compute_f1(self, env, agent_mode, label=2):
        """
        Compute f1 score for current text
        """
        text_vec_tags = env.text_vec[:,-1]
        state_tags = env.state[:,-1]
        right_tag = right_acts = tagged_acts = total_acts = 0
        acc = rec = pre = f1_value = 0.0
        
        if agent_mode == 'act':
            total_words = self.words_num
            temp_words = len(env.current_text['tokens'])
        else:
            total_words = self.max_obj_num
            temp_words = total_words
        if temp_words > total_words:
            temp_words = total_words
        for t in text_vec_tags:
            if t == label:
                total_acts += 1

        for s in xrange(temp_words):
            if state_tags[s] == label:
                tagged_acts += 1
                if text_vec_tags[s] == state_tags[s]:
                    right_acts += 1
            if text_vec_tags[s] == state_tags[s]:
                right_tag += 1

        acc = float(right_tag)/temp_words
        if total_acts > 0:
            rec = float(right_acts)/total_acts
        if tagged_acts > 0:
            pre = float(right_acts)/tagged_acts
        if rec + pre > 0:
            f1_value = (2.0 * rec * pre)/(rec + pre)

        print('agent_mode: %s  total words: %d' % (agent_mode, temp_words))
        print('total: %d\tright: %d\ttagged: %d' % (total_acts, right_acts, tagged_acts))  
        print('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (acc, rec, pre, f1_value))

        return temp_words, total_acts, tagged_acts, right_acts, right_tag, acc, rec, pre, f1_value