#coding:utf-8
import ipdb
import time
import random
import numpy as np
from tqdm import tqdm

class MultiAgent:
    def __init__(self, env_act, env_obj, mem_act, mem_obj, net_act, net_obj, args):
        print('Initializing the Agent...')
        self.env_act = env_act
        self.env_obj = env_obj
        self.mem_act = mem_act
        self.mem_obj = mem_obj
        self.net_act = net_act
        self.net_obj = net_obj
        self.words_num = args.words_num
        self.context_len = args.context_len
        self.num_actions = args.num_actions
        self.target_output = args.target_output

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_act_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        self.display_training_result = args.display_training_result
        
        self.act_steps = 0  
        self.obj_steps = 0
        self.obj_train_steps = args.context_len
        self.total_obj_steps = args.start_epoch * args.train_steps * self.obj_train_steps / 10

    
    def act_restart(self, train_flag, init=False):
        #print('\nRestarting in agent, train_flag = {}, init = {}'.format(train_flag, init))
        self.act_steps = 0
        self.env_act.restart(train_flag, init)


    def obj_restart(self, act_idx):
        self.obj_steps = 0
        self.env_obj.restart(act_idx, self.env_act)


    def explorationRate(self, steps):
        # calculate decaying exploration rate
        if steps < self.exp_decay_steps:
            return self.exp_rate_start - steps * (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end
 

    def act_step(self, exploration_rate, train_flag=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(self.target_output)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env_act.getState()
            qvalues = self.net_act.predict(current_state)
            action = np.argmax(qvalues[0])
            #assert len(qvalues[0]) == self.target_output
            
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
            results = self.compute_f1(self.env_act, 'act', display=self.display_training_result)
            #self.act_steps = 0
            reward += 2   #give a bonus to the terminal actions
            #self.act_restart(train_flag)

        return action, reward, state, terminal, results


    def obj_step(self, exploration_rate, compute=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(self.target_output)
        else:
            # otherwise choose action with highest Q-value
            current_state = self.env_obj.getState()
            qvalues = self.net_obj.predict(current_state)
            action = np.argmax(qvalues[0])
            #assert len(qvalues[0]) == self.target_output
            
        # perform the action  
        reward = self.env_obj.act(action, self.obj_steps)
        state = self.env_obj.getState()
        terminal = self.env_obj.isTerminal()
        
        results = []
        self.obj_steps += 1
        if not terminal:
            #decrease the reward with time steps
            #reward -= abs(reward)*self.obj_steps/(1.5*self.num_actions)
            pass
        else:
            if compute:
                results = self.compute_f1(self.env_obj, 'obj')
            reward += 2   #give a bonus to the terminal actions

        return action, reward, state, terminal, results

 
    def train(self, train_steps, epoch = 0):
        '''
        Play given number of steps
        '''
        #ipdb.set_trace()
        self.act_restart(train_flag=True, init=True)
        start = time.time()
        tmp_right_obj = tmp_tagged_obj = tmp_total_obj = 0
        for i in xrange(train_steps):
            if i % 100 == 0:
                print('\n\nepoch: %d  Training step: %d' % (epoch,i))
            action, reward, state, terminal, _ = self.act_step(
                                    self.explorationRate(self.total_act_steps), train_flag=True)
            act_idx = self.act_steps - 1
            weight_act = weight_obj = 0
            #if (epoch > 0 or i > 1000) and not terminal:
            #print(action, np.random.random(), self.explorationRate(self.total_act_steps))
            if action == 1 or np.random.random() <= self.explorationRate(self.total_act_steps):
                for k in xrange(self.obj_train_steps):
                    #print(k, i)
                    if k == 0:
                        self.obj_restart(act_idx)
                    action2, reward2, state2, terminal2, results2 = self.obj_step(
                                        self.explorationRate(self.total_obj_steps), compute=True)

                    #print(action2)
                    weight_obj = self.env_obj.punish_right_wrong + self.env_obj.punish_double_wrong
                    self.mem_obj.add(action2, reward2, state2, terminal2, weight_obj + 1)
                    if self.target_steps and k % self.target_steps == 0:
                        self.net_obj.update_target_network()

                    if self.mem_obj.count > self.mem_obj.batch_size and k % self.train_frequency == 0:
                        # train for train_repeat times
                        for j in xrange(self.train_repeat):
                            # sample minibatch
                            minibatch2 = self.mem_obj.getMinibatch()
                            self.net_obj.train(minibatch2)
                    self.total_obj_steps += 1
                    if terminal2:
                        tmp_total_obj += results2[10]
                        tmp_right_obj += results2[11]
                        tmp_tagged_obj += results2[12]
                        break

                if self.env_act.real_action_flag != self.env_obj.real_action_flag:
                    ipdb.set_trace()
                #assert self.env_act.real_action_flag == self.env_obj.real_action_flag
                if self.env_act.real_action_flag and self.env_obj.total_punish_right_wrong > 0:
                    #print('Total total_punish_right_wrong: %d' % self.env_obj.total_punish_right_wrong)
                    reward += self.env_obj.total_punish_right_wrong
                elif (not self.env_act.real_action_flag) and (self.env_obj.total_punish_double_wrong > 0):
                    #print('Total total_punish_double_wrong: %d' % self.env_obj.total_punish_double_wrong)
                    reward -= self.env_obj.total_punish_double_wrong
                weight_act = np.sign(self.env_obj.total_punish_right_wrong + self.env_obj.total_punish_double_wrong)
            if terminal:
                end = time.time()
                print('time cost: %.2fs' % (end - start))
                start = end
                self.compute_tmp_f1('obj', tmp_right_obj, tmp_tagged_obj, tmp_total_obj)
                tmp_right_obj = tmp_tagged_obj = tmp_total_obj = 0
                self.act_restart(train_flag=True)
            
            self.mem_act.add(action, reward, state, terminal, weight_act + 1)
            #if weight_act > 0:
            #    print('weight_act: %d  weight_obj: %d\n' % (weight_act, weight_obj))

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
            if self.env_act.train_epoch_end_flag:
                break
    

    def test(self, test_steps, outfile):
        '''
        Play given number of steps
        '''
        act_to_obj = {}
        predActionLabels = []
        test_texts = []
        text_lens = []

        #ipdb.set_trace()
        cumulative_reward = 0
        self.act_restart(train_flag=False, init=True)
        start = time.time()
        tmp_right_act = tmp_tagged_act = tmp_total_act = 0
        tmp_right_obj = tmp_tagged_obj = tmp_total_obj = 0
        for i in xrange(test_steps):
            a, r, s, t, rs = self.act_step(self.exploration_rate_test, train_flag=False)
            cumulative_reward += r
            act_idx = self.act_steps - 1
            text_idx = self.env_act.valid_text_idx
            current_text = self.env_act.current_text
            predActTags = self.env_act.state[:, -1]
            if a == 1:# and not t:
                for k in xrange(self.obj_train_steps):
                    if k == 0:
                        self.obj_restart(act_idx)
                    a2, r2, s2, t2, rs2 = self.obj_step(self.exploration_rate_test, compute=True)
                    cumulative_reward += r2
                    if t2:
                        tmp_total_obj += rs2[10]
                        tmp_right_obj += rs2[11]
                        tmp_tagged_obj += rs2[12]
                        #print(rs2[1: 4])
                        break
                if text_idx not in act_to_obj:
                    act_to_obj[text_idx] = {}
                act_to_obj[text_idx][act_idx] = {'predtags': self.env_obj.state[:, -1],
                                                'realtags': self.env_obj.text_vec[:, -1],
                                                'text_len': rs2[0],
                                                'obj_idxs': self.env_obj.current_text['obj_idxs']}
            if t:
                #end = time.time()
                #print('time cost: %.2fs' % (end - start))
                #start = end
                #self.compute_tmp_f1('obj', tmp_right_obj, tmp_tagged_obj, tmp_total_obj, outfile)
                #tmp_right_obj = tmp_tagged_obj = tmp_total_obj = 0
                tmp_total_act += rs[10]
                tmp_right_act += rs[11]
                tmp_tagged_act += rs[12]
                predActionLabels.append(predActTags)
                test_texts.append(current_text)
                text_lens.append(rs[0])  
                self.act_restart(train_flag=False)  

            if self.env_act.valid_epoch_end_flag:
                break
        self.compute_tmp_f1('act', tmp_right_act, tmp_tagged_act, tmp_total_act, outfile)
        self.compute_tmp_f1('obj', tmp_right_obj, tmp_tagged_obj, tmp_total_obj, outfile)   
        #ipdb.set_trace()
        rec = pre = f1 = 0.0
        right_acts = right_obj = 0
        total_acts = total_obj = 0
        tagged_acts = tagged_obj = 0
        for i in xrange(len(predActionLabels)):
            record_ecs_act_idxs = []
            for act_idx, t in enumerate(predActionLabels[i]):
                if act_idx >= text_lens[i]:
                    break
                act_label = test_texts[i]['tags'][:, -1][act_idx]
                if t == 2:
                    tagged_acts += 1
                    right_act_flag = False
                    if act_label == 2 or act_label == 3:
                        total_acts += 1
                        right_acts += 1
                        right_act_flag = True
                    elif act_label == 4:
                        if act_idx not in record_ecs_act_idxs:
                            total_acts += 1
                            record_ecs_act_idxs.append(act_idx)
                        assert act_idx in test_texts[i]['act2related']
                        exclusive_act_idxs = test_texts[i]['act2related'][act_idx]
                        record_ecs_act_idxs.extend(exclusive_act_idxs)
                        exclusive_flag = False
                        for idx in exclusive_act_idxs:
                            if predActionLabels[i][idx] == 2: # extracted as action
                                exclusive_flag = True
                                break
                        if not exclusive_flag:
                            right_acts += 1
                            right_act_flag = True
                    pred_obj_tags = act_to_obj[i][act_idx]['predtags']
                    real_obj_tags = act_to_obj[i][act_idx]['realtags']
                    obj_idxs = act_to_obj[i][act_idx]['obj_idxs']
                    record_ecs_obj_idxs = []
                    for obj_idx, ot in enumerate(pred_obj_tags):
                        obj_label = real_obj_tags[obj_idx]
                        if ot == 2: # tagged object
                            tagged_obj += 1
                            if right_act_flag: # real action
                                if obj_label == 2: # real essential objects
                                    right_obj += 1
                                    total_obj += 1
                                elif obj_label == 4: # exclusive objects
                                    if obj_idx not in record_ecs_obj_idxs:
                                        total_obj += 1
                                        record_ecs_obj_idxs.append(obj_idx)
                                    right_flag = True
                                    if obj_idx in obj_idxs[0]:
                                        exc_objs = obj_idxs[1]
                                    else:
                                        exc_objs = obj_idxs[0]
                                    record_ecs_obj_idxs.extend(exc_objs)
                                    for oi in exc_objs:
                                        if pred_obj_tags[oi] == 2:
                                            right_flag = False
                                            break
                                    if right_flag: # real exclusive objects
                                        right_obj += 1
                        else: # ot == 1
                            if obj_label == 2:
                                total_obj += 1
                            elif obj_label == 4 and obj_idx not in record_ecs_obj_idxs:
                                total_obj += 1
                                record_ecs_obj_idxs.append(obj_idx)
                                if obj_idx in obj_idxs[0]:
                                    exc_objs = obj_idxs[1]
                                else:
                                    exc_objs = obj_idxs[0]
                                record_ecs_obj_idxs.extend(exc_objs)
                else: # t == 1
                    if act_label == 2:
                        total_acts += 1
                    elif act_label == 4 and act_idx not in record_ecs_act_idxs:
                        total_acts += 1
                        record_ecs_act_idxs.append(act_idx)
                        assert act_idx in test_texts[i]['act2related']
                        exclusive_act_idxs = test_texts[i]['act2related'][act_idx]
                        record_ecs_act_idxs.extend(exclusive_act_idxs)



        if total_acts + total_obj > 0:
            rec = (right_acts + right_obj) / float(total_acts + total_obj)
        if tagged_acts + tagged_obj > 0:
            pre = (right_acts + right_obj) / float(tagged_acts + tagged_obj)
        if rec + pre > 0:
            f1 = 2*rec*pre / (rec+pre)

        print('\nMulti-Agent Results:')
        print('total_acts: %d  total_obj: %d\nright_act: %d  right_obj: %d\ntagged_act: %d  tagged_obj: %d' 
            % (total_acts, total_obj, right_acts, right_obj, tagged_acts, tagged_obj))
        print('rec = %f\tpre = %f\tf1 = %f' % (rec, pre, f1))
        outfile.write('\nMulti-Agent Results:\n')
        outfile.write('total_acts: %d  total_obj: %d\nright_act: %d  right_obj: %d\ntagged_act: %d  tagged_obj: %d\n' 
            % (total_acts, total_obj, right_acts, right_obj, tagged_acts, tagged_obj))
        outfile.write('rec = %f\tpre = %f\tf1 = %f\n' % (rec, pre, f1))
        return rec, pre, f1, cumulative_reward



    def compute_f1(self, env, agent_mode, display=False):
        text_vec_tags = env.text_vec[:,-1]
        state_tags = env.state[:,-1]
        if agent_mode == 'obj': # act_idxs are not obj_idxs
            #ipdb.set_trace()
            state_tags[env.current_text['act_idxs']] = 1
        
        if agent_mode == 'act':
            total_words = self.words_num
            temp_words = len(env.current_text['tokens'])
        else:
            total_words = self.obj_train_steps
            temp_words = total_words
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
                if agent_mode == 'obj':
                    right_flag = True
                    if s in env.current_text['obj_idxs'][0]:
                        exc_objs = env.current_text['obj_idxs'][1]
                    else:
                        exc_objs = env.current_text['obj_idxs'][0]
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
                    assert s in env.current_text['act2related']
                    exclusive_act_idxs = env.current_text['act2related'][s]
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
        # for k, v in results.iteritems():
        #     print(k, v)
        rec = results['rec'][-1]
        pre = results['pre'][-1]
        f1 = results['f1'][-1]
        if display:
            print('agent_mode: %s  total words: %d' % (agent_mode, temp_words))
            print('total: %d\tright: %d\ttagged: %d' % (total_acts, right_acts, tagged_acts))  
            print('acc: %f\trec: %f\tpre: %f\tf1: %f\n' % (acc, rec, pre, f1))
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


    def compute_tmp_f1(self, agent_mode, right, tagged, total, outfile=''):
        rec = pre = f1 = 0.0
        if total > 0:
            rec = float(right) / total
        if tagged > 0:
            pre = float(right) / tagged
        if rec + pre > 0:
            f1 = (2.0 * rec * pre)/(rec + pre)
        print('agent_mode: %s' % agent_mode)
        print('total: %d\tright: %d\ttagged: %d' % (total, right, tagged))  
        print('rec: %f\tpre: %f\tf1: %f\n' % (rec, pre, f1))
        if outfile:
            outfile.write('agent_mode: %s\n' % agent_mode)
            outfile.write('total: %d\tright: %d\ttagged: %d\n' % (total, right, tagged))  
            outfile.write('rec: %f\tpre: %f\tf1: %f\n\n' % (rec, pre, f1))


    def predict(self, text):
        # e.g. text = ['Cook the rice the day before.', 'Use leftover rice.']
        self.env_act.init_predict_eas_text(text)
        #act_seq = []
        sents = []
        for i in range(len(self.env_act.current_text['sents'])):
            if i > 0:
                last_sent = self.env_act.current_text['sents'][i - 1]
            else:
                last_sent = []
            this_sent = self.env_act.current_text['sents'][i]
            sents.append({'last_sent': last_sent, 'this_sent': this_sent, 'acts': []})
        #ipdb.set_trace()
        for i in range(self.words_num):
            state_act = self.env_act.getState()
            qvalues_act = self.net_act.predict(state_act)
            action_act = np.argmax(qvalues_act[0])
            self.env_act.act_eas_text(action_act, i)
            if action_act == 1:
                last_sent, this_sent = self.env_obj.init_predict_af_sents(i, self.env_act.current_text)
                for j in range(self.context_len):
                    state_obj = self.env_obj.getState()
                    qvalues_obj = self.net_obj.predict(state_obj)
                    action_obj = np.argmax(qvalues_obj[0])
                    self.env_obj.act_af_sents(action_obj, j)
                    if self.env_obj.terminal_flag:
                        break
                #act_name = self.env_act.current_text['tokens'][i]
                #act_obj = [act_name]
                act_idx = i
                obj_idxs = []
                sent_words = self.env_obj.current_text['tokens']
                tmp_num = self.context_len if len(sent_words) >= self.context_len else len(sent_words)
                for j in range(tmp_num):
                    if self.env_obj.state[j, -1] == 2:
                        #act_obj.append(sent_words[j])
                        if j == len(sent_words) - 1:
                            j = -1
                        obj_idxs.append(j)
                if len(obj_idxs) == 0:
                    #act_obj.append(sent_words[-1])
                    obj_idxs.append(-1)
                #ipdb.set_trace()
                si, ai = self.env_act.current_text['word2sent'][i]
                ai += len(sents[si]['last_sent'])
                sents[si]['acts'].append({'act_idx': ai, 'obj_idxs': [obj_idxs, []],
                                            'act_type': 1, 'related_acts': []})
                #act_seq.append(act_obj)
            if self.env_act.terminal_flag:
                break
        #for k, v in act_seq.iteritems():
        #    print(k, v)
        #ipdb.set_trace()
        return sents