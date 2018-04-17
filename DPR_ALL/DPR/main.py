#coding:utf-8
import os
import sys
import time
import ipdb
import copy
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
    envarg.add_argument("--words_num", type=int, default=200, help="")
    envarg.add_argument("--word_dim", type=int, default=50, help="")
    envarg.add_argument("--tag_dim", type=int, default=1, help="")
    envarg.add_argument("--obj_dim", type=int, default=50, help="")
    envarg.add_argument("--nchars", type=int, default=93, help="")
    envarg.add_argument("--char_dim", type=int, default=30, help="")
    envarg.add_argument("--max_char_len", type=int, default=20, help="")
    envarg.add_argument("--char_emb_flag", type=int, default=0, help="")
    envarg.add_argument("--actionDB", type=str, default='tag_actions1', help="")
    envarg.add_argument("--max_text_num", type=str, default='10', help="")
    envarg.add_argument("--reward_base", type=float, default=10.0, help="")
    envarg.add_argument("--action_rate", type=float, default=0.15, help="")
    envarg.add_argument("--action_label", type=int, default=2, help="")
    envarg.add_argument("--non_action_label", type=int, default=1, help="")
    envarg.add_argument("--add_obj_flag", type=int, default=0, help="")
    envarg.add_argument("--context_len", type=int, default=100, help="")
    envarg.add_argument("--object_rate", type=float, default=0.15, help='')
    envarg.add_argument("--af_context", type=int, default=0, help='')
    envarg.add_argument("--use_act_tags", type=int, default=0, help='')
    envarg.add_argument("--dynamic_act_vocab_size", type=int, default=50, help='')
    envarg.add_argument("--dynamic_obj_vocab_size", type=int, default=400, help='')
    
    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--positive_rate", type=float, default=0.75, help="")
    memarg.add_argument("--priority", type=int, default=1, help="")
    memarg.add_argument("--save_replay", type=int, default=0, help="")
    memarg.add_argument("--load_replay", type=int, default=0, help="")
    memarg.add_argument("--save_replay_size", type=int, default=1000, help="")
    memarg.add_argument("--save_replay_name", type=str, default='saved_replay_memory.pkl', help="")
    memarg.add_argument("--time_step_batch", type=int, default=0, help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--dqn_mode", type=str, default='cnn', help="")
    netarg.add_argument("--optimizer", type=str, default='rmsprop', help="")
    netarg.add_argument("--learning_rate", type=float, default=0.0025, help="")
    netarg.add_argument("--momentum", type=float, default=0.8, help="")
    netarg.add_argument("--epsilon", type=float, default=1e-6, help="")
    netarg.add_argument("--decay_rate", type=float, default=0.88, help="")
    netarg.add_argument("--discount_rate", type=float, default=0.9, help="")
    netarg.add_argument("--batch_size", type=int, default=32, help="")
    netarg.add_argument("--act_vocab_size", type=int, default=5000, help="")
    netarg.add_argument("--obj_vocab_size", type=int, default=10000, help="")
    netarg.add_argument("--max_act_num", type=int, default=20, help="")
    netarg.add_argument("--max_obj_num", type=int, default=10, help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start", type=float, default=1, help="")
    antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="")
    antarg.add_argument("--exploration_rate_test", type=float, default=0.0, help="")
    antarg.add_argument("--exploration_decay_steps", type=int, default=1000, help="")
    antarg.add_argument("--train_frequency", type=int, default=1, help="")
    antarg.add_argument("--train_repeat", type=int, default=3, help="")
    antarg.add_argument("--target_steps", type=int, default=5, help="")
    antarg.add_argument("--random_play", type=int, default=0, help="")
    antarg.add_argument("--top_list_num", type=list, default=[3, 5], help="")

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--train_steps", type=int, default=0, help="")
    mainarg.add_argument("--test_one_flag", type=int, default=0, help="")
    mainarg.add_argument("--test_text_num", type=int, default=2, help="")
    mainarg.add_argument("--epochs", type=int, default=20, help="")
    mainarg.add_argument("--stop_epoch_gap", type=int, default=5, help="")
    mainarg.add_argument("--start_epoch", type=int, default=0, help="")
    mainarg.add_argument("--load_weights", type=str, default="", help="")
    mainarg.add_argument("--save_weights_prefix", type=str, default="", help="")
    mainarg.add_argument("--computer_id", type=int, default=1, help="")
    mainarg.add_argument("--max_replay_size", type=list, default=[100000,35000,100000], help="")
    mainarg.add_argument("--gpu_rate", type=float, default=0.20, help="")
    mainarg.add_argument("--fold_id", type=int, default=0, help="")
    mainarg.add_argument("--ten_fold_valid", type=int, default=1, help="")
    mainarg.add_argument("--ten_fold_indices", type=str, default='data/wiki_15_ten_fold_indices.pkl', help="")
    mainarg.add_argument("--result_dir", type=str, default="results/wiki_test_dynamic_vocab", help="")
    mainarg.add_argument("--data_name", type=str, default='/home/fengwf/Documents/DRL_data/wikihow/wikihow_act_seq_15.pkl', help='')
    mainarg.add_argument("--agent_mode", type=str, default='multi', help='')
    
    args = parser.parse_args()
    if args.load_weights:
        args.exploration_decay_steps = 1                                                        
    args.num_actions = 2*args.words_num
    if args.char_emb_flag:
        args.emb_dim = args.word_dim + args.tag_dim + args.char_dim
    else:
        args.emb_dim = args.word_dim + args.tag_dim
    if args.add_obj_flag:
        args.emb_dim += args.obj_dim
    # NB !!!!
    #args.word2vec = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    #act_model_dir = '/home/fengwf/Documents/DRL_data/win2k/win2k_model_50_1'
    act_model_dir = '/home/fengwf/Documents/DRL_data/wikihow/wikihow_model_50_5'
    args.word2vec = KeyedVectors.load_word2vec_format(act_model_dir, binary=True)
    return args


def main(args):
    start = time.time()
    print 'Current time is: %s'%get_time()
    print 'Starting at main.py...'
    fold_result = {'acc1': [], 'acc5': [], 'acc10': []}

    for fi in xrange(10):
        fold_start = time.time()
        args.fold_id = fi
        #args.save_weights_prefix = 'weights/%s_fold%d' % (args.agent_mode, fi)
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
        assert args.replay_size > 0

        mem_act = ReplayMemory(args, 'act')
        if args.agent_mode == 'multi':
            env_dv = AFEnvironment(args)
            net_dv = DeepQLearner(args, 'dv', env_act.embedding)
            mem_dv = ReplayMemory(args, 'dv')
            agent = MultiAgent(env_act, env_dv, mem_act, mem_dv, net_act, net_dv, args)
        else:
            agent = Agent(env_act, mem_act, net_act, args)

        if args.load_weights:
            print('Loading weights from %s...' % args.load_weights)
            if os.path.exists(args.load_weights):
                net_act.load_weights(args.load_weights)  #load last trained weights
                #if args.agent_mode == 'multi':
                #    net_dv.load_weights(args.load_weights)
            else:
                print("\n!!! load_weights '%s' doesn't exist !!!\n" % args.load_weights)


        if args.test_one_flag and args.load_weights:
            pass
        else:
            # loop over epochs
            best_acc = {'acc1': 0.0, 'acc5': 0.0, 'acc10': 0.0}
            log_epoch = 0
            with open("%s_fold%d.txt" % (args.result_dir, args.fold_id), 'w') as outfile:
                for epoch in xrange(args.start_epoch, args.start_epoch + args.epochs):
                    if epoch == args.start_epoch:
                        for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                            print('{}: {}'.format(k, v))
                            outfile.write('{}: {}\n'.format(k, v))
                        outfile.write('in-vocab-word rate: {}\n'.format(env_act.in_vocab_words_rate))

                    agent.train(args.train_steps, epoch)
                    acc1, acc5, acc10 = agent.test(args.valid_steps, outfile)

                    if acc1 > best_acc['acc1']:
                        if args.save_weights_prefix:
                            filename = args.save_weights_prefix + "_%d.prm" % (epoch + 1)
                            net_act.save_weights(filename)
                            #if args.agent_mode == 'multi':
                            #    net_dv.save_weights(filename)
                        best_acc['acc1'] = acc1
                        best_acc['acc5'] = acc5
                        best_acc['acc10'] = acc10
                        log_epoch = epoch
                        outfile.write('\n{}\n'.format(best_acc))
                        outfile.write('Best acc1: %f  best epoch: %d\n' % (best_acc['acc1'], epoch))
                    # if no improvement after args.stop_epoch_gap, break
                    if epoch - log_epoch > args.stop_epoch_gap:
                        outfile.write('Best acc1: %f  best epoch: %d\n' % (best_acc['acc1'], epoch))
                        print('\nepoch: %d  result_dir: %s' % (epoch, args.result_dir))
                        print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                        break
                if args.save_replay:
                    mem_act.save(args.save_replay_name, args.save_replay_size)
                for k in best_acc:
                    fold_result[k].append(best_acc[k])
                max_acc1 = max(fold_result['acc1'])
                avg_acc1 = sum(fold_result['acc1']) / len(fold_result['acc1'])
                fold_end = time.time()
                outfile.write('\n{}\n'.format(fold_result))
                outfile.write('\n\nBest acc1: {}  Avg acc1: {}\n'.format(max_acc1, avg_acc1))
                print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
                outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))
        tf.reset_default_graph()
    max_acc1 = max(fold_result['acc1'])
    avg_acc1 = sum(fold_result['acc1']) / len(fold_result['acc1'])
    fold_end = time.time()
    end = time.time()
    print('\n{}\n'.format(fold_result))
    print('Best acc1: {}  Avg acc1: {}'.format(max_acc1, avg_acc1))
    print('Total time cost: %ds' % (end - start))
    print('Current time is: %s\n' % get_time())


if __name__ == '__main__':
    main(args_init())