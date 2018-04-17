#coding:utf-8
import ipdb
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, args, agent_mode):
        print('Initializing ReplayMemory...')
        self.size = args.replay_size
        if agent_mode == 'act':
            self.words_num = args.words_num
            self.emb_dim = args.emb_dim
        elif agent_mode == 'obj':
            self.words_num = args.context_len
            self.emb_dim = args.emb_dim
        #     if args.append_distance:
        #         self.emb_dim += args.dis_dim

        # if args.sampling_with_weights: # initial weight = 1
        #     self.sample_weights = np.ones(self.size, dtype = np.float32)
        # self.sample_flags = np.zeros(self.size, dtype = np.int32)
        # self.sampling_with_weights = args.sampling_with_weights
        self.actions = np.zeros(self.size, dtype = np.uint8)
        self.rewards = np.zeros(self.size, dtype = np.float32)
        self.states = np.zeros((self.size, self.words_num, self.emb_dim))
        self.terminals = np.zeros(self.size, dtype = np.bool)
        self.dims = (self.words_num, self.emb_dim)
        self.priority = args.priority
        self.positive_rate = args.positive_rate
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        if args.load_replay:
            self.load(args.save_replay_name)

        
    def add(self, action, reward, state, terminal, weight=1.0):
        #assert state.shape == self.dims
        # NB! state is post-state, after action and reward
        # if reward >= 0:
        #     self.sample_flags[self.current] = 1 # positive samples
        # else:
        #     self.sample_flags[self.current] = -1 # negative samples
        # if self.sampling_with_weights:
        #     self.sample_weights[self.current] = weight
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size


    def getMinibatch_with_weights(self):
        """
        Memory must include poststate, prestate and history
        Sample random indexes or with priority
        """
        #ipdb.set_trace()
        if self.priority:
            indexes = []
            pos_amount =  int(self.positive_rate*self.batch_size) 
            pos_idxs = []
            neg_idxs = []
            for i in xrange(self.count):
                if self.sample_flags[i] == 1:
                    pos_idxs.append(i)
                elif self.sample_flags[i] == -1:
                    neg_idxs.append(i)
            if self.sampling_with_weights:
                indexes = []
                pos_weights = np.zeros(len(pos_idxs))
                tmp_weights = self.sample_weights[pos_idxs]
                total_pos_weights = int(sum(tmp_weights))
                pos_weights[0] = tmp_weights[0]
                for j in xrange(1, len(tmp_weights)):
                    pos_weights[j] = pos_weights[j - 1] + tmp_weights[j]

                neg_weights = np.zeros(len(neg_idxs))
                tmp_weights = self.sample_weights[neg_idxs]
                total_neg_weights = int(sum(tmp_weights))
                neg_weights[0] = tmp_weights[0]
                for j in xrange(1, len(tmp_weights)):
                    neg_weights[j] = neg_weights[j - 1] + tmp_weights[j]

                for i in xrange(pos_amount):
                    idx = np.random.randint(total_pos_weights)
                    for j in xrange(len(pos_weights)):
                        if j == 0 and pos_weights[j] >= idx:
                            indexes.append(pos_idxs[j])
                        elif j > 0 and pos_weights[j - 1] < idx <= pos_weights[j]:
                            indexes.append(pos_idxs[j]) 
                for i in xrange(self.batch_size - pos_amount):
                    idx = np.random.randint(total_neg_weights)
                    for j in xrange(len(neg_weights)):
                        if j == 0 and neg_weights[j] >= idx:
                            indexes.append(neg_idxs[j])
                        elif j > 0 and neg_weights[j - 1] < idx <= neg_weights[j]:
                            indexes.append(neg_idxs[j])

            else:
                np.random.shuffle(pos_idxs)
                np.random.shuffle(neg_idxs)
                if len(pos_idxs) > 0 and len(neg_idxs) > 0:
                    indexes = []
                    idx = 0
                    while len(indexes) < pos_amount:
                        indexes.append(pos_idxs[idx % len(pos_idxs)])
                        idx += 1
                    while len(indexes) < self.batch_size:
                        indexes.append(neg_idxs[idx % len(neg_idxs)])
                        idx += 1
                else:
                    indexes = pos_idxs[: self.batch_size] + neg_idxs[: self.batch_size]
        
        else: # randomly replay
            indexes = range(len(self.states))
            np.random.shuffle(indexes)
            indexes = indexes[: self.batch_size]

        prestates = np.array([self.states[i] for i in indexes])
        poststates = self.states[indexes]
        actions = self.actions[indexes]  
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return prestates, actions, rewards, poststates, terminals


    def getMinibatch(self):
        """
        Memory must include poststate, prestate and history
        Sample random indexes or with priority
        """
        prestates = np.zeros([self.batch_size, self.words_num, self.emb_dim])
        poststates = np.zeros([self.batch_size, self.words_num, self.emb_dim])
        if self.priority:
            pos_amount =  int(self.positive_rate*self.batch_size) 

        indexes = []
        count_pos = 0
        count_neg = 0
        count = 0 
        max_circles = 10*self.batch_size # max times for choosing positive samples or nagative samples
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over) 
                index = np.random.randint(1, self.count - 1)
                # NB! poststate (last state) can be terminal state!
                if self.terminals[index - 1]:
                    continue
                # use prioritized replay trick
                if self.priority:
                    if count < max_circles:
                        # if num_pos is already enough but current idx is also pos sample, continue
                        if (count_pos >= pos_amount) and (self.rewards[index] > 0):
                            count += 1
                            continue
                        # elif num_nag is already enough but current idx is also nag sample, continue
                        elif (count_neg >= self.batch_size - pos_amount) and (self.rewards[index] < 0): 
                            count += 1
                            continue
                    if self.rewards[index] > 0:
                        count_pos += 1
                    else:
                        count_neg += 1
                break
            
            prestates[len(indexes)] = self.states[index - 1]
            indexes.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indexes]  
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        poststates = self.states[indexes]
        return prestates, actions, rewards, poststates, terminals


    def save(self, fname, size):
        if size > self.size:
            size = self.size
        databag = {}
        databag['actions'] = self.actions[: size]
        databag['rewards'] = self.rewards[: size]
        databag['states'] = self.states[: size]
        databag['terminals'] = self.terminals[: size]
        with open(fname, 'wb') as f:
            print('Try to save replay memory ...')
            pickle.dump(databag, f)
            print('Replay memory is successfully saved as %s' % fname)


    def load(self, fname):
        if not os.path.exists(fname):
            print("%s doesn't exist!" % fname)
            return
        with open(fname, 'rb') as f:
            print('Loading replay memory from %s ...' % fname)
            databag = pickle.load(f)
            size = len(databag['states'])
            self.states[: size] = databag['states']
            self.actions[: size] = databag['actions']
            self.rewards[: size] = databag['rewards']
            self.terminals[: size] = databag['terminals']
            self.count = size
            self.current = size