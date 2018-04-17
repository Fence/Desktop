import os
import re
import time
import ipdb
import pickle
from tqdm import tqdm


class STF(object):
    """docstring for STF"""
    def __init__(self):
        from nltk.stem import WordNetLemmatizer
        from nltk.parse.stanford import StanfordDependencyParser
        core = '/home/fengwf/stanford/stanford-corenlp-3.7.0.jar'
        model = '/home/fengwf/stanford/english-models.jar'
        self.dep_parser = StanfordDependencyParser(path_to_jar=core, path_to_models_jar=model,
                        encoding='utf8', java_options='-mx2000m')
        self.lemma = WordNetLemmatizer()
        self.data_name = '/home/fengwf/Documents/DRL_data/name/name_labeled_text_data.pkl'


    def stanford_eas(self, actionDB, outfile):
        #ipdb.set_trace()
        data_name = self.data_name.replace('name', actionDB)
        indata = pickle.load(open(data_name, 'rb'))

        #outdata = []
        total_act = right_act = tagged_act = 0
        total_obj = right_obj = tagged_obj = 0
        total_exc_act = right_exc_act = tagged_exc_act = 0
        total_exc_obj = right_exc_obj = tagged_exc_obj = 0
        for i in tqdm(range(len(indata))):
            try:
                sents = [' '.join(s) for s in indata[i]['sents']]
                dep = self.dep_parser.raw_parse_sents(sents)
            except AssertionError:
                pass
            except Exception as e:
                #print(e)
                continue

            #sent_acts = []
            for j in range(len(sents)):
                try:
                    dep_root = next(dep)
                    dep_sent = next(dep_root)
                    conll = [_.split() for _ in str(dep_sent.to_conll(10)).split('\n') if _]
                except StopIteration:
                    #print('j = %d len(sents) = %d Raise StopIteration.\n' % (j, len(sents)))
                    break 
                except Exception as e:
                    #print(e)
                    continue
                words = []
                idx2word = {}
                for w in conll:
                    idx2word[w[0]] = w[1]
                    words.append(w[1]) #word_lemma
                if len(words) != len(indata[i]['sents'][j]):
                    continue
                    #print('len(words) != len(sents[j])', len(words), len(indata[i]['sents'][j]))
                    #ipdb.set_trace()

                acts = {}
                real_acts = {}
                act2related = {}
                exc_acts = []
                bias = 0
                if j > 0:
                    bias = len(indata[i]['sents'][j - 1])
                for act in indata[i]['sent_acts'][j]:
                    act_idx = act['act_idx'] - bias
                    real_acts[act_idx] = [[], []]
                    for idx in xrange(len(real_acts[act_idx])):
                        for oi in act['obj_idxs'][idx]:
                            if oi - bias >= 0:
                                real_acts[act_idx][idx].append(oi - bias)
                            elif oi == -1:
                                real_acts[act_idx][idx].append(oi)
                    act2related[act_idx] = [ai - bias for ai in act['related_acts'] if ai - bias >= 0]
                    if len(act2related[act_idx]) > 0 and act_idx not in exc_acts:
                        total_exc_act += 1
                        exc_acts.append(act_idx)
                        exc_acts.extend(act2related[act_idx])
                    if len(real_acts[act_idx][1]) > 0:
                        total_exc_obj += 1

                total_act += len(real_acts)
                total_obj += sum([len(v) for k,v in real_acts.items()])
                for line in conll:
                    if 'dobj' in line or 'nsubjpass' in line:
                        obj_idxs = [int(line[0]) - 1]
                        act_idx = int(line[6]) - 1
                        for one_line in conll:
                            if one_line[6] == obj_idxs[0] and one_line[7] == 'conj':
                                obj_idxs.append(int(one_line[0]) - 1)
                        acts[act_idx] = obj_idxs
                for act_idx in acts:
                    tagged_act += 1
                    obj_idxs = acts[act_idx]
                    tagged_obj += len(obj_idxs)
                    if act_idx in real_acts:
                        right_act_flag = True
                        if len(act2related[act_idx]) > 0:
                            tagged_exc_act += 1
                        for ai in act2related[act_idx]:
                            if ai in acts:
                                right_act_flag = False
                                break
                        if right_act_flag:
                            right_act += 1
                            if len(act2related[act_idx]) > 0:
                                right_exc_act += 1
                            for oi in obj_idxs:
                                right_obj_flag = True
                                if len(real_acts[act_idx][1]) > 0:
                                    tagged_exc_obj += 1
                                if oi in real_acts[act_idx][0]:
                                    for toi in real_acts[act_idx][1]:
                                        if toi in obj_idxs:
                                            right_obj_flag = False
                                            break
                                else:
                                    for toi in real_acts[act_idx][0]:
                                        if toi in obj_idxs:
                                            right_obj_flag = False
                                            break
                                if right_obj_flag:
                                    right_obj += 1
                                    if len(real_acts[act_idx][1]) > 0:
                                        right_exc_obj += 1
                #sent_acts.append(acts)
            #outdata.append(sent_acts)
        total = total_act + total_obj
        right = right_act + right_obj
        tagged = tagged_act + tagged_obj
        act_result = self.basic_f1(total_act, right_act, tagged_act)
        obj_result = self.basic_f1(total_obj, right_obj, tagged_obj)
        act_exc_result = self.basic_f1(total_exc_act, right_exc_act, tagged_exc_act)
        obj_exc_result = self.basic_f1(total_exc_obj, right_exc_obj, tagged_exc_obj)
        multi_result = self.basic_f1(total, right, tagged)
        print('\ndataset: {}'.format(name))
        print('act_result: {}, {}, {}, {}'.format(total_act, right_act, tagged_act, act_result))
        print('obj_reault: {}, {}, {}, {}'.format(total_obj, right_obj, tagged_obj, obj_result))
        print('exc_act_result: {}, {}, {}, {}'.format(total_exc_act, right_exc_act, tagged_exc_act, act_exc_result))
        print('exc_obj_reault: {}, {}, {}, {}'.format(total_exc_obj, right_exc_obj, tagged_exc_obj, obj_exc_result))
        print('multi_result: {}, {}, {}, {}\n'.format(total, right, tagged, multi_result))
        outfile.write('dataset: {}\n'.format(name))
        outfile.write('act_result: {}, {}, {}, {}\n'.format(total_act, right_act, tagged_act, act_result))
        outfile.write('obj_reault: {}, {}, {}, {}\n'.format(total_obj, right_obj, tagged_obj, obj_result))
        outfile.write('exc_act_result: {}, {}, {}, {}\n'.format(total_exc_act, right_exc_act, tagged_exc_act, act_exc_result))
        outfile.write('exc_obj_reault: {}, {}, {}, {}\n'.format(total_exc_obj, right_exc_obj, tagged_exc_obj, obj_exc_result))
        outfile.write('multi_result: {}, {}, {}, {}\n'.format(total, right, tagged, multi_result))



    def basic_f1(self, total, right, tagged):
        rec = pre = f1 = 0.0
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2 * pre * rec / (pre + rec)
        return rec, pre, f1



if __name__ == '__main__':
    start = time.time()
    model = STF()
    for name in ['wikihow', 'cooking', 'win2k']:
        with open('results/test_exc_%s_result.txt'%name, 'w+') as outfile:
            model.stanford_eas(name, outfile)
    end = time.time()
    print('\nTotal time cost: %.2fs\n'%(end - start))