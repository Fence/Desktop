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
        self.dqn_mode = args.dqn_mode
        self.agent_mode = args.agent_mode
        self.words_num = args.words_num
        self.batch_size = args.batch_size
        self.target_output = args.target_output

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        self.random_play = args.random_play
        self.display_training_result = args.display_training_result
        
        self.steps = 0  #use to decrease the reward during time steps 
        #self.epoch_end_flag = False
        self.use_thompson_sampling = args.use_thompson_sampling
        self.ts_param_table = np.ones(self.target_output)
        self.sampling_prob = random.random()
        self.filter_act_idx = args.filter_act_idx 

    
    def _restart(self, train_flag, init=False):
        #print('\nRestarting in agent, train_flag = {}, init = {}'.format(train_flag, init))
        self.steps = 0
        self.env.restart(train_flag, init)
        #self.epoch_end_flag = self.env.epoch_end_flag


    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
            (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end


    def Thompson_Sampling(self):
        # adopt thompson sampling instead of random selecting
        if self.sampling_prob <= self.ts_param_table[0] / sum(self.ts_param_table):
            return 0
        else:
            return 1


    def update_TS_parameter(self, action, reward):
        # record the action with positive reward
        if reward > 0:
            self.ts_param_table[action] += 1
        else:
            self.ts_param_table[action] += 1
        # get a random num ahead, and avoid getting continuous two random num
        # which may be not random according to system time
        self.sampling_prob = random.random() 
 

    def step(self, exploration_rate, train_flag=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            if self.use_thompson_sampling:
                action = self.Thompson_Sampling()
            else:
                action = np.random.randint(self.target_output)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env.getState()
            qvalues = self.net.predict(current_state)
            if self.dqn_mode == 'lstm':
                #assert qvalues.shape == (1, self.words_num, self.target_output)
                action = np.argmax(qvalues[0, self.steps])
            else:
                #assert qvalues.shape == (1, self.target_output)
                action = np.argmax(qvalues[0])
            
        # perform the action  
        reward = self.env.act(action, self.steps)
        state = self.env.getState()
        terminal = self.env.isTerminal()
        if self.dqn_mode == 'lstm':
            action = self.steps * 2 + action
        
        results = []
        self.steps += 1
        #self.update_TS_parameter(action, reward)
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.steps/(3.0*self.words_num)
            pass  
        else:
            if self.display_training_result:
                results = self.compute_f1(self.display_training_result)
            elif not train_flag:
                results = self.compute_f1(self.display_training_result)
                #print('Thompson Sampling Table: {}\t{}'.format(
                #    self.ts_param_table, self.ts_param_table/sum(self.ts_param_table)))
            self.steps = 0
            reward += 2   #give a bonus to the terminal actions
            self._restart(train_flag)

        return action, reward, state, terminal, results

 
    def train(self, train_steps, epoch, online_text_num=-1, restart_init=True):
        '''
        Play given number of steps
        '''
        trained_texts = 0
        if restart_init:
            self._restart(train_flag=True, init=True)
        for i in xrange(train_steps):
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
            if terminal:
                trained_texts += 1
            if online_text_num >= 0 and trained_texts >= online_text_num:
                print('\n-----End of an online training epoch!-----\n')
                break
            elif self.env.train_epoch_end_flag:
                break
    

    def test(self, test_steps, outfile):
        '''
        Play given number of steps
        '''
        t_total_rqs = t_tagged_rqs = t_right_rqs = 0
        t_total_ops = t_tagged_ops = t_right_ops = 0
        t_total_ecs = t_tagged_ecs = t_right_ecs = 0
        t_right_tag = t_right_acts = t_tagged_acts = t_total_acts = t_words = 0
        t_acc = t_rec = t_pre = t_f1 = 0.0
        
        cumulative_reward = 0
        self._restart(train_flag=False, init=True)
        for i in xrange(test_steps):
            if self.random_play:
                a, r, s, t, rs = self.step(1, train_flag=False)
            else:
                a, r, s, t, rs = self.step(self.exploration_rate_test, train_flag=False)
            cumulative_reward += r
            if t:
                outfile.write('\nText: %d\ntotal words: %d\n' % (self.env.valid_text_idx - 1, rs[0]))
                outfile.write('total: %d\tright: %d\ttagged: %d\n' % (rs[10], rs[11], rs[12]))  
                outfile.write('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (rs[14], rs[15], rs[16], rs[17]))

                t_words += rs[0]
                t_total_rqs += rs[1]
                t_right_rqs += rs[2]
                t_tagged_rqs += rs[3]
                t_total_ops += rs[4]
                t_right_ops += rs[5]
                t_tagged_ops += rs[6]
                t_total_ecs += rs[7]
                t_right_ecs += rs[8]
                t_tagged_ecs += rs[9]
                t_total_acts += rs[10] 
                t_right_acts += rs[11]
                t_tagged_acts += rs[12]
                t_right_tag += rs[13]    
                
            if self.env.valid_epoch_end_flag:
                break   

        t_acc = float(t_right_tag)/t_words
        results = {'rec': [], 'pre': [], 'f1': []}
        self.basic_f1(t_total_rqs, t_right_rqs, t_tagged_rqs, results)
        self.basic_f1(t_total_ops, t_right_ops, t_tagged_ops, results)
        self.basic_f1(t_total_ecs, t_right_ecs, t_tagged_ecs, results)
        self.basic_f1(t_total_acts, t_right_acts, t_tagged_acts, results)
        t_rec = results['rec'][-1]
        t_pre = results['pre'][-1]
        t_f1 = results['f1'][-1]

        outfile.write('\n\nSummary:\n')
        outfile.write('total_rqs: %d\t right_rqs: %d\t tagged_rqs: %d\n' % (t_total_rqs, t_right_rqs, t_tagged_rqs))
        outfile.write('total_ops: %d\t right_ops: %d\t tagged_ops: %d\n' % (t_total_ops, t_right_ops, t_tagged_ops))
        outfile.write('total_ecs: %d\t right_ecs: %d\t tagged_ecs: %d\n' % (t_total_ecs, t_right_ecs, t_tagged_ecs))
        outfile.write('total_act: %d\t right_act: %d\t tagged_act: %d\n' % (t_total_acts, t_right_acts, t_tagged_acts))  
        outfile.write('acc: %f\t rec: %f\t pre: %f\t f1: %f\n' % (t_acc, t_rec, t_pre, t_f1))
        for k, v in results.iteritems():
            outfile.write('{}: {}\n'.format(k, v))
            print(k, v)
        outfile.write('\ncumulative reward: %f\n'%cumulative_reward)
        print('\n\nSummary:')
        print('total_rqs: %d\t right_rqs: %d\t tagged_rqs: %d' % (t_total_rqs, t_right_rqs, t_tagged_rqs))
        print('total_ops: %d\t right_ops: %d\t tagged_ops: %d' % (t_total_ops, t_right_ops, t_tagged_ops))
        print('total_ecs: %d\t right_ecs: %d\t tagged_ecs: %d' % (t_total_ecs, t_right_ecs, t_tagged_ecs))
        print('total_act: %d\t right_act: %d\t tagged_act: %d' % (t_total_acts, t_right_acts, t_tagged_acts))  
        print('acc: %f\t rec: %f\t pre: %f\t f1: %f' % (t_acc, t_rec, t_pre, t_f1))
        print('\ncumulative reward: %f\n'%cumulative_reward)
        return t_rec, t_pre, t_f1, cumulative_reward


    def compute_f1(self, display=False):
        """
        Compute f1 score for current text
        """
        text_vec_tags = self.env.text_vec[:,-1]
        state_tags = self.env.state[:,-1]
        if self.agent_mode == 'af' and self.filter_act_idx: # act_idxs are not obj_idxs
            state_tags[self.env.current_text['act_idxs']] = 1
        
        total_words = self.words_num
        temp_words = len(self.env.current_text['tokens'])
        if temp_words > total_words:
            temp_words = total_words

        record_ecs_act_idxs = []
        right_tag = right_acts = tagged_acts = total_acts = 0
        total_rqs = right_rqs = tagged_rqs = 0
        total_ecs = right_ecs = tagged_ecs = 0
        total_ops = right_ops = tagged_ops = 0
        for s in xrange(temp_words):
            if state_tags[s] == 2:
                tagged_acts += 1
            if text_vec_tags[s] == 2: # required actions
                total_acts += 1
                total_rqs += 1
                if state_tags[s] == 2: # extract
                    tagged_rqs += 1
                    right_rqs += 1
                    right_acts += 1
                    right_tag += 1
            elif text_vec_tags[s] == 3: # optional actions
                #total_acts += 1
                #total_ops += 1
                if state_tags[s] == 2: # extract
                    total_acts += 1
                    total_ops += 1
                    tagged_ops += 1
                    right_ops += 1
                    right_acts += 1
                    right_tag += 1
            elif text_vec_tags[s] == 4: # exclusive actions
                if state_tags[s] == 2:
                    tagged_ecs += 1
                if s not in record_ecs_act_idxs:
                    total_acts += 1
                    total_ecs += 1
                    record_ecs_act_idxs.append(s)
                if self.agent_mode == 'af':
                    right_flag = True
                    if s in self.env.current_text['obj_idxs'][0]:
                        exc_objs = self.env.current_text['obj_idxs'][1]
                    else:
                        exc_objs = self.env.current_text['obj_idxs'][0]
                    record_ecs_act_idxs.extend(exc_objs)
                    for oi in exc_objs:
                        if state_tags[oi] == 2:
                            right_flag = False
                            break
                    if state_tags[s] == 2 and right_flag:
                        right_ecs += 1
                        right_acts += 1
                        right_tag += 1
                    elif state_tags[s] == 1 and not right_flag:
                        right_tag += 1
                else:
                    assert s in self.env.current_text['act2related']
                    exclusive_act_idxs = self.env.current_text['act2related'][s]
                    record_ecs_act_idxs.extend(exclusive_act_idxs)
                    exclusive_flag = False
                    for idx in exclusive_act_idxs:
                        if state_tags[idx] == 2: # extracted as action
                            exclusive_flag = True
                            break
                    if not exclusive_flag and state_tags[s] == 2: # extract
                        right_ecs += 1
                        right_acts += 1
                        right_tag += 1
                    elif exclusive_flag and state_tags[s] == 1: # filtered out
                        right_tag += 1
            elif text_vec_tags[s] == 1: # non_actions
                if state_tags[s] == 1:
                    right_tag += 1
                else:
                    tagged_rqs += 1

        acc = float(right_tag)/temp_words
        results = {'rec': [], 'pre': [], 'f1': []}
        self.basic_f1(total_rqs, right_rqs, tagged_rqs, results)
        self.basic_f1(total_ops, right_ops, tagged_ops, results)
        self.basic_f1(total_ecs, right_ecs, tagged_ecs, results)
        self.basic_f1(total_acts, right_acts, tagged_acts, results)
        if display:
            for k, v in results.iteritems():
                print(k, v)
        rec = results['rec'][-1]
        pre = results['pre'][-1]
        f1 = results['f1'][-1]
        return temp_words, total_rqs, right_rqs, tagged_rqs, total_ops, right_ops, tagged_ops, total_ecs, right_ecs, tagged_ecs, total_acts, right_acts, tagged_acts, right_tag, acc, rec, pre, f1


    def basic_f1(self, total, right, tagged, results):
        rec = pre = f1 = 0.0
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2 * pre * rec / (pre + rec)
        results['rec'].append(rec)
        results['pre'].append(pre)
        results['f1'].append(f1)