#coding:utf-8
import os
import sys
import time
import ipdb
import pickle
import argparse
import tensorflow as tf

from utils import get_time
from Agent import Agent
from MultiAgent import MultiAgent
from KerasEADQN import DeepQLearner
from Environment import Environment
from AFEnvironment import AFEnvironment
from ReplayMemory import ReplayMemory
from gensim.models import KeyedVectors
from keras.backend.tensorflow_backend import set_session


def args_init():
    parser = argparse.ArgumentParser()

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--model_dir", type=str, default='/home/fengwf/Documents/mymodel-new-5-50', help="")
    envarg.add_argument("--actionDB", type=str, default='wikihow', help="")
    envarg.add_argument("--max_text_num", type=str, default='10', help="")
    envarg.add_argument("--words_num", type=int, default=500, help="")
    envarg.add_argument("--word_dim", type=int, default=50, help="")
    envarg.add_argument("--tag_dim", type=int, default=50, help="")
    envarg.add_argument("--obj_dim", type=int, default=50, help="")
    envarg.add_argument("--dis_dim", type=int, default=50, help="")
    envarg.add_argument("--act_dim", type=int, default=50, help="")
    envarg.add_argument("--nchars", type=int, default=93, help="")
    envarg.add_argument("--char_dim", type=int, default=30, help="")
    envarg.add_argument("--context_len", type=int, default=100, help="")
    envarg.add_argument("--max_char_len", type=int, default=20, help="")
    envarg.add_argument("--reward_assign", type=float, default=50.0, help="")
    envarg.add_argument("--object_rate", type=float, default=0.07, help='')
    envarg.add_argument("--action_rate", type=float, default=0.10, help="")
    envarg.add_argument("--use_act_rate", type=int, default=1, help='')
    envarg.add_argument("--char_emb_flag", type=int, default=0, help="")
    # envarg.add_argument("--append_distance", type=int, default=1, help='')
    
    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--sampling_with_weights", type=int, default=0, help="")
    memarg.add_argument("--positive_rate", type=float, default=0.9, help="")
    memarg.add_argument("--priority", type=int, default=1, help="")
    memarg.add_argument("--save_replay", type=int, default=0, help="")
    memarg.add_argument("--load_replay", type=int, default=0, help="")
    memarg.add_argument("--save_replay_size", type=int, default=1000, help="")
    memarg.add_argument("--save_replay_name", type=str, default='data/saved_replay_memory.pkl', help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--dqn_mode", type=str, default='cnn', help="")
    netarg.add_argument("--gram_num", type=int, default=5, help="")
    netarg.add_argument("--batch_size", type=int, default=32, help="")
    netarg.add_argument("--filter_num", type=int, default=32, help="")
    netarg.add_argument("--dense_dim", type=int, default=256, help="")
    netarg.add_argument("--target_output", type=int, default=2, help="")
    netarg.add_argument("--optimizer", type=str, default='adam', help="")
    netarg.add_argument("--learning_rate", type=float, default=0.0025, help="")
    netarg.add_argument("--momentum", type=float, default=0.8, help="")
    netarg.add_argument("--epsilon", type=float, default=1e-6, help="")
    netarg.add_argument("--dropout", type=float, default=0.5, help="")
    netarg.add_argument("--decay_rate", type=float, default=0.88, help="")
    netarg.add_argument("--discount_rate", type=float, default=0.9, help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start", type=float, default=1, help="")
    antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="")
    antarg.add_argument("--exploration_rate_test", type=float, default=0.0, help="")
    antarg.add_argument("--exploration_decay_steps", type=int, default=1000, help="")
    antarg.add_argument("--train_frequency", type=int, default=1, help="")
    antarg.add_argument("--train_repeat", type=int, default=1, help="")
    antarg.add_argument("--target_steps", type=int, default=5, help="")
    antarg.add_argument("--random_play", type=int, default=0, help="")
    antarg.add_argument("--use_thompson_sampling", type=int, default=0, help="")
    antarg.add_argument("--display_training_result", type=int, default=1, help='')
    antarg.add_argument("--filter_act_idx", type=int, default=1, help='')

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--train_steps", type=int, default=0, help="")
    mainarg.add_argument("--log_train_process", type=int, default=0, help="")
    mainarg.add_argument("--online_text_num", type=int, default=-1, help="")
    mainarg.add_argument("--test_text_num", type=int, default=10, help="")
    mainarg.add_argument("--epochs", type=int, default=2, help="")
    mainarg.add_argument("--start_epoch", type=int, default=0, help="")
    mainarg.add_argument("--stop_epoch_gap", type=int, default=5, help="")
    mainarg.add_argument("--load_weights", type=str, default="", help="")
    mainarg.add_argument("--save_weights_prefix", type=str, default="", help="")
    mainarg.add_argument("--computer_id", type=int, default=1, help="")
    mainarg.add_argument("--max_replay_size", type=list, default=[100000,35000,100000], help="")
    mainarg.add_argument("--gpu_rate", type=float, default=0.24, help="")
    mainarg.add_argument("--fold_id", type=int, default=0, help="")
    mainarg.add_argument("--start_fold", type=int, default=0, help='')
    mainarg.add_argument("--end_fold", type=int, default=0, help='')
    mainarg.add_argument("--use_cross_valid", type=int, default=1, help="")
    mainarg.add_argument("--k_fold", type=int, default=10, help="")
    mainarg.add_argument("--result_dir", type=str, default="test_cnn", help="")
    mainarg.add_argument("--agent_mode", type=str, default='eas', help='')
    
    args = parser.parse_args()
    args.num_actions = 2*args.words_num
    args.emb_dim = args.word_dim + args.tag_dim
    args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    if args.load_weights:
        args.exploration_rate_start = args.exploration_rate_end
    if args.char_emb_flag:
        args.emb_dim += args.char_dim
    if args.agent_mode == 'af':
        args.words_num = 100
        args.display_training_result = 0
        args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        args.k_fold_indices = 'data/indices/%s_af_%d_fold_indices.pkl' % (args.actionDB, args.k_fold)
        # if args.append_distance:
        #     args.emb_dim += args.dis_dim
    else:
        args.data_name = 'data/%s_labeled_text_data.pkl' % args.actionDB 
        args.k_fold_indices = 'data/indices/%s_eas_%d_fold_indices.pkl' % (args.actionDB, args.k_fold)
    args.result_dir = 'results/%s/%s/%s' % (args.actionDB, args.agent_mode, args.result_dir)
    if args.end_fold == 0:
        args.end_fold = args.k_fold
    return args


def main(args):
    start = time.time()
    print('Current time is: %s'%get_time())
    print('Starting at main...')
    fold_result = {'rec': [], 'pre': [], 'f1': [], 'rw': []}

    for fi in xrange(args.start_fold, args.end_fold):
        fold_start = time.time()
        args.fold_id = fi
        #Initial environment, replay memory, deep_q_net and agent
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        
        env_act = Environment(args)
        net_act = DeepQLearner(args, 'act')

        temp_size = env_act.train_steps * args.epochs + env_act.valid_steps
        if temp_size > args.max_replay_size[args.computer_id]:
            temp_size = args.max_replay_size[args.computer_id]
        args.replay_size = temp_size
        args.train_steps = env_act.train_steps
        args.valid_steps = env_act.valid_steps
        args.test_text_num = env_act.valid_steps / args.words_num
        #assert args.replay_size > 0

        mem_act = ReplayMemory(args, 'act')
        agent = Agent(env_act, mem_act, net_act, args)

        if args.load_weights:
            print('Loading weights ...')
            net_act.load_weights(args.load_weights)  #load last trained weights

        # loop over epochs
        best_f1 = {'rec': 0.0, 'pre': 0.0, 'f1': 0.0, 'rw': 0.0}
        log_epoch = 0
        with open("%s_fold%d.txt" % (args.result_dir, args.fold_id), 'w') as outfile:
            for epoch in xrange(args.start_epoch, args.start_epoch + args.epochs):
                if epoch == args.start_epoch:
                    for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                        print('{}: {}'.format(k, v))
                        outfile.write('{}: {}\n'.format(k, v))
                
                agent.train(args.train_steps, epoch, args.online_text_num)
                r, p, f, rw = agent.test(args.valid_steps, outfile)

                if f > best_f1['f1']:
                    if args.save_weights_prefix:
                        filename = 'weights/%s/%s/k%d_fold%d.h5' % \
                                (args.actionDB, args.agent_mode, args.k_fold, args.fold_id) 
                        net_act.save_weights(filename)

                    best_f1['f1'] = f
                    best_f1['rec'] = r
                    best_f1['pre'] = p
                    best_f1['rw'] = rw
                    log_epoch = epoch
                    outfile.write('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                    print('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                # if no improvement after args.stop_epoch_gap, break
                if epoch - log_epoch >= args.stop_epoch_gap:
                    outfile.write('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                    print('\nepoch: %d  result_dir: %s' % (epoch, args.result_dir))
                    print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                    break
            if args.save_replay:
                mem_act.save(args.save_replay_name, args.save_replay_size)
            for k in best_f1:
                fold_result[k].append(best_f1[k])
            max_f1 = max(fold_result['f1'])
            avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
            max_rw = max(fold_result['rw'])
            avg_rw = sum(fold_result['rw']) / len(fold_result['rw'])
            fold_end = time.time()
            outfile.write('\n{}\n'.format(fold_result))
            outfile.write('\n\nBest f1: {}  Avg f1: {}  Best reward: {}  Avg reward: {}\n'.format(
                max_f1, avg_f1, max_rw, avg_rw))
            print('\n\nBest f1: {}  Avg f1: {}  Best reward: {}  Avg reward: {}\n'.format(
                max_f1, avg_f1, max_rw, avg_rw))
            print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
            outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))
        tf.reset_default_graph()
    max_f1 = max(fold_result['f1'])
    avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
    max_rw = max(fold_result['rw'])
    avg_rw = sum(fold_result['rw']) / len(fold_result['rw'])
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Best f1: {}  Avg f1: {}  Best reward: {}  Avg reward: {}'.format(max_f1, avg_f1, max_rw, avg_rw))
    print('Total time cost: %ds' % (end - start))
    print('Current time is: %s\n' % get_time())



def train_process_main(args):
    start = time.time()
    print('Current time is: %s'%get_time())
    print('Starting at online test ...')
    fold_result = {'rec': [], 'pre': [], 'f1': [], 'rw': [], 'steps': []}
    assert args.online_text_num > 0

    for fi in xrange(args.start_fold, args.end_fold):
        fold_start = time.time()
        args.fold_id = fi
        #Initial environment, replay memory, deep_q_net and agent
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        
        env_act = Environment(args)
        net_act = DeepQLearner(args, 'act')

        temp_size = env_act.train_steps * args.epochs + env_act.valid_steps
        if temp_size > args.max_replay_size[args.computer_id]:
            temp_size = args.max_replay_size[args.computer_id]
        args.replay_size = temp_size
        args.train_steps = env_act.train_steps
        args.valid_steps = env_act.valid_steps
        args.train_text_num = env_act.train_steps / args.words_num
        args.test_text_num = env_act.valid_steps / args.words_num

        mem_act = ReplayMemory(args, 'act')
        agent = Agent(env_act, mem_act, net_act, args)

        log_results = {'rec': [], 'pre': [], 'f1': [], 'rw': [], 'steps': []}
        with open("%s_fold%d.txt" % (args.result_dir, args.fold_id), 'w') as outfile:
            for epoch in xrange(args.start_epoch, args.start_epoch + args.epochs):
                if epoch == args.start_epoch:
                    for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                        print('{}: {}'.format(k, v))
                        outfile.write('{}: {}\n'.format(k, v))
                
                total_trained_texts = 0
                restart_init = True
                while total_trained_texts < args.train_text_num:
                    if total_trained_texts > 0:
                        restart_init = False
                    agent.train(args.train_steps, epoch, args.online_text_num, restart_init)
                    rec, pre, f1, rw = agent.test(args.valid_steps, outfile)
                    total_trained_texts += args.online_text_num
                    log_results['rec'].append(rec)
                    log_results['pre'].append(pre)
                    log_results['f1'].append(f1)
                    log_results['rw'].append(rw)
                    log_results['steps'].append(agent.total_train_steps)
                    for k, v in log_results.iteritems():
                        print('{}: {}\n'.format(k, v))
            for k in log_results:
                fold_result[k].append(log_results[k])
            fold_end = time.time()
            outfile.write('\n{}\n'.format(fold_result))
            outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))
            print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
        tf.reset_default_graph()
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Total time cost: %ds' % (end - start))
    print('Current time is: %s\n' % get_time())



def multi_main(args):
    start = time.time()
    print('Current time is: %s'%get_time())
    print('Starting at multi_main...')
    fold_result = {'rec': [], 'pre': [], 'f1': [], 'rw': []}
    args.exploration_rate_start = 0
    args.exploration_decay_steps = 1
    #args.epochs = 1
    #args.load_weights = 'weights'
    #args.save_weights_prefix = 'weights'
    #args.result_dir = 'results/multi/test'
    #args.display_training_result = 0

    for eas_fold_id in xrange(args.k_fold):
        fold_start = time.time()
        args.fold_id = eas_fold_id
        #Initial environment, replay memory, deep_q_net and agent
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
        set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
        
        #args.optimizer = 'rmsprop'
        #args.positive_rate = 0.6
        #args.learning_rate = 0.001
        #args.sampling_with_weights = False
        env_act = Environment(args)
        net_act = DeepQLearner(args, 'act')
        
        temp_size = env_act.train_steps * args.epochs + env_act.valid_steps
        if temp_size > args.max_replay_size[args.computer_id]:
            temp_size = args.max_replay_size[args.computer_id]
        args.replay_size = temp_size
        args.train_steps = env_act.train_steps
        args.valid_steps = env_act.valid_steps
        args.test_text_num = env_act.valid_steps / args.words_num
        #assert args.replay_size > 0

        mem_act = ReplayMemory(args, 'act') #'' #
        #args.optimizer = 'adadelta'
        #args.positive_rate = 0.6
        #args.sampling_with_weights = False
        #args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        #args.k_fold_indices = 'data/indices/%s_af_%d_fold_indices.pkl' % (args.actionDB, args.k_fold)
        env_obj = AFEnvironment(args)
        net_obj = DeepQLearner(args, 'obj')
        mem_obj = ReplayMemory(args, 'obj') #'' #

        print('Loading weights ...')
        filename = 'weights/online_test/%s/eas/fold%d.h5' % (args.actionDB, args.fold_id) 
        net_act.load_weights(filename)
        filename = 'weights/online_test/%s/af/fold%d.h5' % (args.actionDB, args.fold_id) 
        net_obj.load_weights(filename)

        #agent = Agent(env_act, mem_act, net_act, args)
        agent = MultiAgent(env_act, env_obj, mem_act, mem_obj, net_act, net_obj, args)

        # loop over epochs
        best_f1 = {'rec': 0.0, 'pre': 0.0, 'f1': 0.0, 'rw': 0.0}
        log_epoch = 0
        result_dir = "%s_fold%d.txt" % (args.result_dir, eas_fold_id)
        with open(result_dir, 'w') as outfile:
            outfile.write('Pre_training result:\n')
            rec, pre, f1, reward = agent.test(args.valid_steps, outfile)
            # fold_result['rec'].append(rec)
            # fold_result['pre'].append(pre)
            # fold_result['f1'].append(f1)
            # fold_result['rw'].append(reward)

            for epoch in xrange(args.start_epoch, args.start_epoch + args.epochs):
                if epoch == args.start_epoch:
                    for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                        #print('{}: {}'.format(k, v))
                        outfile.write('{}: {}\n'.format(k, v))
                
                agent.train(args.train_steps, epoch)
                r, p, f, rw = agent.test(args.valid_steps, outfile)

                if f > best_f1['f1']:
                    if args.save_weights_prefix:
                        filename = 'weights/multi/eas_fold%d.h5' % eas_fold_id
                        net_act.save_weights(filename)
                        filename = 'weights/multi/af_fold%d.h5' % af_fold_id
                        net_obj.save_weights(filename)

                    best_f1['f1'] = f
                    best_f1['rec'] = r
                    best_f1['pre'] = p
                    log_epoch = epoch
                    outfile.write('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                    print('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                # if no improvement after args.stop_epoch_gap, break
                if epoch - log_epoch > args.stop_epoch_gap:
                    outfile.write('\n\nBest f1 value: {}  best epoch: {}\n'.format(best_f1, log_epoch))
                    print('\nepoch: %d  result_dir: %s' % (epoch, args.result_dir))
                    print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                    break
            if args.save_replay:
                mem_act.save(args.save_replay_name, args.save_replay_size)
            for k in best_f1:
                fold_result[k].append(best_f1[k])
            max_f1 = max(fold_result['f1'])
            avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
            fold_end = time.time()
            outfile.write('\n{}\n'.format(fold_result))
            outfile.write('\n\nBest f1: {}  Avg f1: {}\n'.format(max_f1, avg_f1))
            print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
            outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))

        tf.reset_default_graph()
    max_f1 = max(fold_result['f1'])
    avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Best f1: {}  Avg f1: {}'.format(max_f1, avg_f1))
    print('Total time cost: %ds' % (end - start))
    print('Current time is: %s\n' % get_time())



if __name__ == '__main__':
    args = args_init()
    if args.agent_mode == 'multi':
        multi_main(args)
    elif args.log_train_process:
        train_process_main(args)
    else:
        main(args)