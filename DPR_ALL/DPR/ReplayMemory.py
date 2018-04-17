#coding:utf-8
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, args, agent_mode):
        print('Initializing ReplayMemory...')
        self.size = args.replay_size
        if agent_mode == 'act':
            self.words_num = args.words_num
            self.emb_dim = args.emb_dim
        else:
            self.words_num = args.max_obj_num
            self.emb_dim = args.word_dim
        self.actions = np.zeros(self.size, dtype = np.uint8)
        self.rewards = np.zeros(self.size, dtype = np.float32)
        self.states = np.zeros((self.size, self.words_num, self.emb_dim))
        self.terminals = np.zeros(self.size, dtype = np.bool)
        self.priority = args.priority
        self.positive_rate = args.positive_rate
        self.dims = (self.words_num, self.emb_dim)
        self.batch_size = args.batch_size
        self.time_step_batch = args.time_step_batch
        self.count = 0
        self.current = 0

        if args.load_replay:
            self.load(args.save_replay_name)

        
    def add(self, action, reward, state, terminal):
        assert state.shape == self.dims
        # NB! state is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states[self.current] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size

    
    def getState(self, index):
        assert self.count > 0, "replay memory is zeros, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.channel - 1:
          # use faster slicing
          return self.states[(index - (self.channel - 1)):(index + 1), ...]
        else:
          # otherwise normalize indexes and use slower list based access
          indexes = [(index - i) % self.count for i in reversed(range(self.channel))]
          return self.states[indexes, ...]


    def getMinibatch(self):
        """
        Memory must include poststate, prestate and history
        Sample random indexes or with priority
        """
        if self.time_step_batch:
            indexes = range(self.count - self.batch_size, self.count)
            pre_indexes = range(self.count - self.batch_size - 1, self.count - 1)
            #if self.count % 73 == 0:
            #    print(indexes, pre_indexes)
            prestates = self.states[pre_indexes]
            poststates = self.states[indexes]
        else:
            prestates = np.zeros([args.batch_size, self.words_num, self.emb_dim])
            poststates = np.zeros([args.batch_size, self.words_num, self.emb_dim])
            if self.priority:
                pos_amount =  int(self.positive_rate*self.batch_size) 

            indexes = []
            count_pos = 0
            count_neg = 0
            count = 0 
            max_circles = 1000 # max times for choosing positive samples or nagative samples
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