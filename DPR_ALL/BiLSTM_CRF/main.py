import os
import sys
import ipdb
import math
import time
import pickle
import logging
import argparse
import numpy as np
import tensorflow as tf
from copy import deepcopy
from keras.backend.tensorflow_backend import set_session
from BiLSTM import BiLSTM
from preprocessing import read_af_sents, read_eas_texts
from preprocessing import perpareDataset, createDataMatrices, easOutput_to_afInput

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import sys
reload(sys)
sys.setdefaultencoding('gb18030')

def args_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default='/home/fengwf/Documents/mymodel-new-5-50', help='')
    parser.add_argument("--word_dim", type=int, default=50, help='')
    parser.add_argument("--label_key", default='tags', help='')
    parser.add_argument("--fold_id", type=int, default=0, help='')
    parser.add_argument("--epochs", type=int, default=50, help='')
    parser.add_argument("--use_act_tags", type=int, default=0, help='')
    parser.add_argument("--eas_fold_num", type=int, default=0, help='')
    parser.add_argument("--af_fold_num", type=int, default=0, help='')
    parser.add_argument("--select_data", type=int, default=1, help='')
    parser.add_argument("--max_expend_num", type=int, default=2, help='')
    parser.add_argument("--op_count", type=int, default=2, help='')
    parser.add_argument("--ex_count", type=int, default=3, help='')
    parser.add_argument("--k_fold", type=int, default=5, help="")
    parser.add_argument("--start_fold", type=int, default=0, help='')
    parser.add_argument("--end_fold", type=int, default=0, help='')
    parser.add_argument("--batch_size", type=int, default=128, help='')
    parser.add_argument("--gpu_rate", type=float, default=0.20, help='')
    parser.add_argument("--is_test", type=int, default=0, help='')
    parser.add_argument("--online_text_num", type=int, default=0, help='')
    parser.add_argument("--online_text_step", type=int, default=50, help='')
    parser.add_argument("--load_weights", type=str, default='', help='')
    parser.add_argument("--result_dir", type=str, default='test', help='')
    parser.add_argument("--actionDB", default='cooking', help='')
    parser.add_argument("--train_mode", type=str, default='eas', help='')
    args = parser.parse_args()

    #Parameters of the network
    args.params = {'dropout': [0.25, 0.25], 
                'classifier': 'CRF', #'tanh-crf',#'Softmax',#
                'LSTM-Size': [500], 
                'optimizer': 'nadam', 
                'charEmbeddings': 'CNN', 
                'miniBatchSize': args.batch_size,
                'earlyStopping': 5,
                'train_mode': args.train_mode,
                'ex_count': args.ex_count}
    if args.end_fold == 0:
        args.end_fold = args.k_fold
    if args.select_data and args.train_mode == 'eas':
        args.ten_fold_indices = 'data/indices/max_num%d_%s_%s_%d_fold_indices.pkl' \
            % (args.max_expend_num, args.actionDB, args.train_mode, args.k_fold)
    else:
        args.ten_fold_indices = 'data/indices/all_data_%s_%s_%d_fold_indices.pkl' \
            % (args.actionDB, args.train_mode, args.k_fold)
    return args


def main(args):
    total_time = 0
    #ipdb.set_trace()
    best_f1s = []
    t1 = time.time()
    if args.train_mode == 'af':
        args.params['LSTM-Size'] = [100]
        args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        folds = read_af_sents(args)
        args.params['op_count'] = math.ceil(args.ex_count / 2.0)
    else:
        args.data_name = 'data/%s_labeled_text_data.pkl' % args.actionDB
        folds = read_eas_texts(args)
    t2 = time.time()

    for j in xrange(args.start_fold, args.end_fold):
        start = time.time()
        args.fold_id = j 
        if args.select_data: 
            result_dir = 'results/%s/%s/%s_fold%d.txt' \
                    % (args.actionDB, args.train_mode, args.result_dir, j)
        else:
            result_dir = 'results/%s/%s/%s_fold%d.txt' \
                    % (args.actionDB, args.train_mode, args.result_dir, j)
        args.save_weights = "weights/%s/%s/new_fold%d_weights.h5" \
                    % (args.actionDB, args.train_mode, args.fold_id)

        with open(result_dir, 'w') as outfile: 
            #Prepare embeddings and datasets
            t3 = time.time()
            datasets = perpareDataset(folds, args)
            t4 = time.time()
            embeddings = datasets['embeddings']
            word2Idx = datasets['word2Idx']
            data = datasets['data']

            print('\nread_data: %.2fs, prepare_data: %.2fs\n'%(t2 - t1, t4 - t3))
            outfile.write(('\nread_data: %.2fs, prepare_data: %.2fs\n'%(t2 - t1, t4 - t3)))
            for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                print('{}: {}'.format(k, v))
                outfile.write('{}: {}\n'.format(k, v))
            print("data['mappings'].keys:  \n\n" % data['mappings'].keys())
            outfile.write("{}\n".format(data['mappings'].keys()))

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
            set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

            model = BiLSTM(outfile, args.params)
            if args.train_mode == 'af':
                if args.use_act_tags:
                    model.additionalFeatures = ['distance', 'act_tags']
                else:
                    model.additionalFeatures = ['distance']
            model.setMappings(embeddings, data['mappings'])
            model.setTrainDataset(data, args.label_key)
            if args.load_weights:
                filename = 'weights/%s/%s/new_fold0_weights.h5' % (args.load_weights, args.train_mode)
                model.loadModel(filename)
            model.computeScores(data['devMatrix'], data['testMatrix'])
            #ipdb.set_trace()
            model.verboseBuild = True
            model.modelSavePath = args.save_weights
            model.evaluate(args.epochs, best_f1s)
            tf.reset_default_graph()
            end = time.time()
            total_time += (end - start)

            print('\n\nbest_f1s: {}\n'.format(best_f1s))
            print('best_f1_value: {}\n'.format(best_f1s[-1]))
            print('average_f1_value: {}\n'.format(sum(best_f1s)/len(best_f1s)))
            print('args.result_dir: %s\n' % args.result_dir)
            print('Fold %d time cost: %ds\n' % (j, end - start))
            outfile.write('\n\nbest_f1s: {}\n'.format(best_f1s))
            outfile.write('best_f1_value: {}\n'.format(best_f1s[-1]))
            outfile.write('average_f1_value: {}\n'.format(sum(best_f1s)/len(best_f1s)))
            outfile.write('args.result_dir: %s\n' % args.result_dir)
            outfile.write('Fold %d time cost: %ds\n' % (j, end - start))

        print('args.result_dir: %s' % args.result_dir)
        print(best_f1s)
        print('Average f1: %f' % (sum(best_f1s)/len(best_f1s)))
        print('Total time cost: %ds\n' % total_time)



def online_test(args):
    total_time = 0
    #ipdb.set_trace()
    t1 = time.time()
    if args.train_mode == 'af':
        args.params['LSTM-Size'] = [100]
        args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        folds = read_af_sents(args)
        args.params['op_count'] = math.ceil(args.ex_count / 2.0)
        if args.load_weights == 'cooking':
            args.params['maxAddFeatureValue'] = 84
        elif args.load_weights == 'win2k':
            args.params['maxAddFeatureValue'] = 44
        else:
            args.params['maxAddFeatureValue'] = 143
    else:
        args.data_name = 'data/%s_labeled_text_data.pkl' % args.actionDB
        folds = read_eas_texts(args)
    t2 = time.time()
    args.params['miniBatchSize'] = 50

    fold_results = [[] for i in xrange(args.end_fold)]
    assert args.online_text_num > 0
    final_outfile = 'results/%s/%s/%s_all_folds.txt' % (args.actionDB, args.train_mode, args.result_dir)
    for j in xrange(args.start_fold, args.end_fold):
        args.fold_id = j 
        result_dir = 'results/%s/%s/%s_fold%d.txt' % (args.actionDB, args.train_mode, args.result_dir, j)
        #Prepare embeddings and datasets
        t3 = time.time()
        datasets = perpareDataset(folds, args)
        t4 = time.time()
        embeddings = datasets['embeddings']
        word2Idx = datasets['word2Idx']
        print('\nread_data: %.2fs, prepare_data: %.2fs\n'%(t2 - t1, t4 - t3))

        best_f1s = []
        for i in xrange(args.online_text_num):
            tn = i * args.online_text_step if i > 0 else 1
            with open(result_dir, 'w') as outfile: 
                start = time.time()
                data = deepcopy(datasets['data'])
                data['trainMatrix'] = data['trainMatrix'][: tn]
                #ipdb.set_trace()
                print('\ntn = %d\n' % tn)
                for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                    print('{}: {}'.format(k, v))
                    outfile.write('{}: {}\n'.format(k, v))
                print("data['mappings'].keys:  \n\n" % data['mappings'].keys())
                outfile.write("{}\n".format(data['mappings'].keys()))

                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_rate)
                set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

                model = BiLSTM(outfile, args.params)
                if args.train_mode == 'af':
                    if args.use_act_tags:
                        model.additionalFeatures = ['distance', 'act_tags']
                    else:
                        model.additionalFeatures = ['distance']
                model.setMappings(embeddings, data['mappings'])
                model.setTrainDataset(data, args.label_key)
                if args.load_weights:
                    filename = 'weights/%s/%s/new_fold0_weights.h5' % (args.load_weights, args.train_mode)
                    model.loadModel(filename)
                model.computeScores(data['devMatrix'], data['testMatrix'])
                #ipdb.set_trace()
                model.verboseBuild = True
                model.evaluate(args.epochs, best_f1s)
                tf.reset_default_graph()
                end = time.time()
                total_time += (end - start)

                print('\n\nbest_f1s: {}\n'.format(best_f1s))
                print('best_f1_value: {}\n'.format(max(best_f1s)))
                print('args.result_dir: %s\n' % args.result_dir)
                print('Text num %d time cost: %ds\n' % (tn, end - start))
                outfile.write('\n\nbest_f1s: {}\n'.format(best_f1s))
                outfile.write('best_f1_value: {}\n'.format(max(best_f1s)))
                outfile.write('args.result_dir: %s\n' % args.result_dir)
                outfile.write('Text num %d time cost: %ds\n' % (tn, end - start))
        fold_results.append(best_f1s)
    fold_results = np.array(fold_results)
    avg_f1 = np.mean(fold_results, axis=0)
    print(fold_results)
    print('Average f1: {}'.format(avg_f1) )
    print('Total time cost: %ds\n' % total_time)
    f = open(final_outfile, 'w')
    for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
        f.write('{}: {}\n'.format(k, v))
    f.write('\n{}\n'.format(fold_results))
    f.write('Average f1: {}'.format(avg_f1) )
    f.write('Total time cost: %ds\n' % total_time)
    f.close()


def test(args, results, fi=0):
    #ipdb.set_trace()
    args.fold_id = fi
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_rate
    set_session(tf.Session(config=config))

    args.result_dir = 'results/pipeline/%s/fold%d.txt' % (args.actionDB, args.fold_id)
    with open(args.result_dir, 'w') as outfile:
        args.data_name = 'data/%s_labeled_text_data.pkl' % args.actionDB
        args.ten_fold_indices = 'data/indices/max_num%d_%s_%s_%d_fold_indices.pkl' \
        % (args.max_expend_num, args.actionDB, args.train_mode, args.k_fold)
        args.params['LSTM-Size'] = [500]
        args.params['train_mode'] = 'eas'
        eas_folds = read_eas_texts(args)
        test_texts = eas_folds['test'][args.fold_id]

        eas_datasets = perpareDataset(eas_folds, args)
        eas_embeddings = eas_datasets['embeddings']
        eas_word2Idx = eas_datasets['word2Idx']
        eas_data = eas_datasets['data']

        eas_model_dir = "models/%s/eas/fold%d_weights.h5" % (args.actionDB, args.fold_id)
        eas_model = BiLSTM(outfile, args.params)
        eas_model.setMappings(eas_embeddings, eas_data['mappings'])
        eas_model.setTrainDataset(eas_data, args.label_key)
        eas_model.verboseBuild = False
        eas_model.loadModel(eas_model_dir)
        
        args.fold_id = args.af_fold_num
        args.params['LSTM-Size'] = [100]
        args.params['train_mode'] = 'af'
        args.data_name = 'data/refined_%s_data.pkl' % args.actionDB
        args.ten_fold_indices = 'data/indices/all_data_%s_%s_%d_fold_indices.pkl' \
        % (args.actionDB, args.train_mode, args.k_fold)
        af_folds = read_af_sents(args)

        af_datasets = perpareDataset(af_folds, args)
        af_embeddings = af_datasets['embeddings']
        af_word2Idx = af_datasets['word2Idx']
        af_data = af_datasets['data']

        af_model_dir = "models/%s/af/fold%d_weights.h5" % (args.actionDB, args.fold_id)
        af_model = BiLSTM(outfile, args.params)
        if args.use_act_tags:
            af_model.additionalFeatures = ['distance', 'act_tags']
        else:
            af_model.additionalFeatures = ['distance']
        af_model.setMappings(af_embeddings, af_data['mappings'])
        af_model.setTrainDataset(af_data, args.label_key)
        af_model.verboseBuild = False
        af_model.loadModel(af_model_dir)

        ipdb.set_trace()
        easMatrix = createDataMatrices(test_texts, args.label_key, eas_model.mappings)
        predActionLabels = eas_model.tagSentences(easMatrix)
        afMatrix, raw_af_sents = easOutput_to_afInput(predActionLabels, test_texts, af_model.mappings, args)
        predObjectLabels = af_model.tagSentences(afMatrix)

        print('\nValidation f1 Scores:')
        outfile.write(('Validation f1 Scores:\neas results:\n'))
        eas_model.computeScores(easMatrix, easMatrix)
        outfile.write(('\naf results:\n'))
        af_model.computeScores(af_data['devMatrix'], af_data['testMatrix'])
        outfile.write(('\naf pipeline results:\n'))
        af_model.computeScores(afMatrix, afMatrix)
        #ipdb.set_trace()
        rec, pre, f1 = compute_test_score(predActionLabels, predObjectLabels, test_texts, raw_af_sents, outfile)
        print(args.result_dir)
        results['rec'].append(rec)
        results['pre'].append(pre)
        results['f1'].append(f1)
        tf.reset_default_graph()
        avg_f1 = sum(results['f1']) / len(results['f1'])
        best_f1 = max(results['f1'])
        worse_f1 = min(results['f1'])
        outfile.write('\n\nrec:\n{}\npre:\n{}\nf1:{}\n\n'.format(results['rec'], results['pre'], results['f1']))
        outfile.write('avg f1: {}  best f1: {}  worse f1: {}\n'.format(avg_f1, best_f1, worse_f1))


def compute_test_score(predActionLabels, predObjectLabels, test_texts, raw_af_sents, outfile):
    rec = pre = f1 = 0.0
    total_act = total_obj = 0
    tagged_act = tagged_obj = 0
    right_act = right_obj = 0
    act_to_obj = {}
    #ipdb.set_trace()
    for i, labels in enumerate(predObjectLabels):
        text_idx = raw_af_sents[i]['text_idx']
        act_idx = raw_af_sents[i]['act_idx']
        words = raw_af_sents[i]['tokens']
        if text_idx not in act_to_obj:
            act_to_obj[text_idx] = {}
        act_to_obj[text_idx][act_idx] = {'predtags': labels, 'realtags': raw_af_sents[i]['tags']}
        predobjs = []
        realobjs = []
        for j in xrange(len(labels)):
            if labels[j] in ['1', '3']:
                predobjs.append(words[j])
            if raw_af_sents[i]['tags'][j] in ['1', '3']:
                realobjs.append(words[j])
        outfile.write('No.{} {}\n  action: {}\n  predobjs: {}\n  realobjs: {}\n\n'.format(
            i, words, raw_af_sents[i]['action'], predobjs, realobjs))

    for i in xrange(len(predActionLabels)):
        for act_idx, t in enumerate(predActionLabels[i]):
            if t in ['1', '2', '3']: # tagged action
                tagged_act += 1
                if test_texts[i]['tags'][act_idx] == t:
                    right_act += 1
                pred_obj_tags = act_to_obj[i][act_idx]['predtags']
                real_obj_tags = act_to_obj[i][act_idx]['realtags']
                for obj_idx, ot in enumerate(pred_obj_tags):
                    if ot in ['1', '3']: # tagged object
                        tagged_obj += 1
                        if ot == real_obj_tags[obj_idx] and test_texts[i]['tags'][act_idx] in ['1', '2', '3']:
                            right_obj += 1


    if total_act + total_obj > 0:
        rec = (right_act + right_obj) / float(total_act + total_obj)
    if tagged_act + tagged_obj > 0:
        pre = (right_act + right_obj) / float(tagged_act + tagged_obj)
    if rec + pre > 0:
        f1 = 2*rec*pre / (rec+pre)

    print('\nPipeline Results:')
    print('total_act: %d  total_obj: %d\nright_act: %d  right_obj: %d\ntagged_act: %d  tagged_obj: %d' 
        % (total_act, total_obj, right_act, right_obj, tagged_act, tagged_obj))
    print('rec = %f\tpre = %f\tf1 = %f' % (rec, pre, f1))
    outfile.write('\nPipeline Results:\n')
    outfile.write('total_act: %d  total_obj: %d\nright_act: %d  right_obj: %d\ntagged_act: %d  tagged_obj: %d\n' 
        % (total_act, total_obj, right_act, right_obj, tagged_act, tagged_obj))
    outfile.write('rec = %f\tpre = %f\tf1 = %f\n' % (rec, pre, f1))
    return rec, pre, f1





if __name__ == '__main__':
    args = args_init()
    if args.online_text_num > 0:
        online_test(args)
    elif args.is_test:
        results = {'rec': [], 'pre': [], 'f1': []}
        start = time.time()
        for i in xrange(5):
            test(args_init(), results, i)
            print('\n\n{}\n\n'.format(e))
        end = time.time()
        avg_f1 = sum(results['f1']) / len(results['f1'])
        best_f1 = max(results['f1'])
        worse_f1 = min(results['f1'])
        print('\n\nrec:\n{}\npre:\n{}\nf1:{}\n\n'.format(results['rec'], results['pre'], results['f1']))
        print('avg f1: {}  best f1: {}  worse f1: {}\n'.format(avg_f1, best_f1, worse_f1))
        print('Total time cost: %.2fs\n' % (end - start))
    else:
        main(args)
