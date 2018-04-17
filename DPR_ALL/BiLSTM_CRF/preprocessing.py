import os
import re
import ipdb
import pickle
import logging
import numpy as np
import mysql.connector
from tqdm import tqdm
from nltk import FreqDist
from gensim.models import KeyedVectors


def perpareDataset(folds, args, mini_count=1):
    """
    Reads in the pre-trained embeddings (in text format) from embeddingsPath and prepares those 
    to be used with the LSTM network.
    Unknown words in the trainDataPath-file are added, if they appear at least mini_count times
    """
    # :: Read in word embeddings ::   
    word2Idx = {}
    embeddings = []

    w_model = KeyedVectors.load_word2vec_format(args.model_dir, binary=True)
    embeddingsDimension = args.word_dim
    #logging.info('Vocabulary size: %d\tvector dim: %d\n' % (len(w_model.vocab), args.word_dim))
    for word in w_model.vocab:
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension) 
            embeddings.append(vector)
            
            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension) 
            embeddings.append(vector)
        
        vector = w_model[word]

        if word not in word2Idx:                     
            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)
    
    
    # Extend embeddings file with new tokens 
    def createFD(sents, fd, word2Idx):
        #ipdb.set_trace()
        for sent in sents:
            splits = sent['tokens']
            for word in splits:
                wordLower = word.lower() 
                wordNormalized = wordNormalize(wordLower)
                
                if word not in word2Idx and wordLower not in word2Idx and wordNormalized not in word2Idx: 
                    fd[wordNormalized] += 1                    
            
    #if mini_count != None and mini_count >= 0:
    #    fd = FreqDist()
        
    #    trainSentences = folds['train'][args.fold_id]
    #    createFD(trainSentences, fd, word2Idx)
    #    addedWords = 0
    #    for word, freq in fd.most_common(10000):
    #        if freq < mini_count:
    #            break
            
    #        addedWords += 1        
    #        word2Idx[word] = len(word2Idx)
            #Alternativ -sqrt(3/dim) ... sqrt(3/dim)
    #        vector = np.random.uniform(-0.25, 0.25, args.word_dim)
    #        embeddings.append(vector)
            
    #        assert(len(word2Idx) == len(embeddings))
        #logging.info("Added words: %d" % addedWords)
    
    embeddings = np.array(embeddings)
    pklObjects = {'embeddings': embeddings, 'word2Idx': word2Idx, 'data': {}}
    
    casing2Idx = getCasingVocab()
    pklObjects['data'] = createPklFiles(folds, args, word2Idx, casing2Idx)
    
    return pklObjects 



def wordNormalize(word):
    word = word.lower()
    word = word.replace("--", "-")
    word = re.sub("\"+", '"', word)
    word = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", 'DATE_TOKEN', word)
    word = re.sub("[0-9]{2}:[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    word = re.sub("[0-9]{2}:[0-9]{2}", 'TIME_TOKEN', word)
    word = re.sub("[0-9.,]+", 'NUMBER_TOKEN', word)
    return word



def addCharInformation(sentences):
    """Breaks every token into the characters"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['characters'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            chars = [c for c in token]
            sentences[sentenceIdx]['characters'].append(chars)



def addCasingInformation(sentences):
    """Adds information of the casing of words"""
    for sentenceIdx in range(len(sentences)):
        sentences[sentenceIdx]['casing'] = []
        for tokenIdx in range(len(sentences[sentenceIdx]['tokens'])):
            token = sentences[sentenceIdx]['tokens'][tokenIdx]
            sentences[sentenceIdx]['casing'].append(getCasing(token))
       
       

def getCasing(word):   
    """Returns the casing for a word"""
    casing = 'other'
    
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    
    return casing



def getCasingVocab():
    entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 
                'allLower', 'allUpper', 'initialUpper', 'contains_digit']
    return {entries[idx]:idx for idx in range(len(entries))}



def createMatrices(sentences, mappings, padOneTokenSentence=True):
    data = []
    numTokens = 0
    numUnknownTokens = 0    
    missingTokens = FreqDist()
    paddedSentences = 0

    #ipdb.set_trace()
    for i in tqdm(xrange(len(sentences))):
        sentence = sentences[i]
        row = {name: [] for name in list(mappings.keys())+['raw_tokens']}
        
        for mapping, str2Idx in mappings.items():    
            if mapping not in sentence:
                continue
                    
            for entry in sentence[mapping]:                
                if mapping.lower() == 'tokens':
                    numTokens += 1
                    idx = str2Idx['UNKNOWN_TOKEN']
                    
                    if entry in str2Idx:
                        idx = str2Idx[entry]
                    elif entry.lower() in str2Idx:
                        idx = str2Idx[entry.lower()]
                    elif wordNormalize(entry) in str2Idx:
                        idx = str2Idx[wordNormalize(entry)]
                    else:
                        numUnknownTokens += 1    
                        missingTokens[wordNormalize(entry)] += 1
                        
                    row['raw_tokens'].append(entry)
                elif mapping.lower() == 'characters':  
                    idx = []
                    for c in entry:
                        if c in str2Idx:
                            idx.append(str2Idx[c])
                        else:
                            idx.append(str2Idx['UNKNOWN'])                           
                                      
                else:
                    idx = str2Idx[entry]
                                    
                row[mapping].append(idx)

        # add additional features
        if 'distance' in sentence:
            row['distance'] = sentence['distance']
        if 'act_tags' in sentence:
            row['act_tags'] = sentence['act_tags']
        if 'act2related' in sentence:
            row['act2related'] = sentence['act2related']
        if 'exc_objs' in sentence:
            row['exc_objs'] = sentence['exc_objs']
                
        if len(row['tokens']) == 1 and padOneTokenSentence:
            paddedSentences += 1
            for mapping, str2Idx in mappings.items():
                if mapping.lower() == 'tokens':
                    row['tokens'].append(mappings['tokens']['PADDING_TOKEN'])
                    row['raw_tokens'].append('PADDING_TOKEN')
                elif mapping.lower() == 'characters':
                    row['characters'].append([0])
                else:
                    row[mapping].append(0)
            if 'distance' in sentence:
                if len(row['distance']) == 1:
                    row['distance'].append(len(row['tokens']))
                else:
                    row['distance'] = row['distance'][:2]
            if 'act_tags' in sentence:
                if len(row['act_tags']) == 1:
                    row['act_tags'].append(1) # non-action label
                else:
                    row['act_tags'] = row['act_tags'][:2]
            
        data.append(row)
    
    #if numTokens > 0:           
    #    logging.info("Unknown-Tokens: %.2f%%" % (numUnknownTokens/float(numTokens)*100))
        
    return data
    
  
  
def createPklFiles(folds, args, word2Idx, casing2Idx):       
              
    trainSentences = folds['train'][args.fold_id]
    devSentences = folds['valid'][args.fold_id]
    testSentences = folds['test'][args.fold_id] 
    #import ipdb
    #ipdb.set_trace()  
   
    mappings = createMappings(trainSentences+devSentences+testSentences, args.label_key)
    mappings['tokens'] = word2Idx
    mappings['casing'] = casing2Idx
                
    
    charset = {"PADDING":0, "UNKNOWN":1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        charset[c] = len(charset)
    mappings['characters'] = charset
    
    addCharInformation(trainSentences)
    addCasingInformation(trainSentences)
    
    addCharInformation(devSentences)
    addCasingInformation(devSentences)
    
    #addCharInformation(testSentences)   
    #addCasingInformation(testSentences)   
    
  
    trainMatrix = createMatrices(trainSentences, mappings)
    devMatrix = createMatrices(devSentences, mappings)
    #testMatrix = createMatrices(testSentences, mappings)       

    
    data = { 'mappings': mappings,
                'trainMatrix': trainMatrix,
                'devMatrix': devMatrix,
                'testMatrix': devMatrix #testMatrix
            }        
       
    return data



def createMappings(sentences, keys):
    #sentenceKeys = list(sentences[0].keys())
    #sentenceKeys.remove('tokens')
    sentenceKeys = [keys]
    
    vocabs = {name:{'O':0} for name in sentenceKeys} #Use 'O' also for padding
    #vocabs = {name:{} for name in sentenceKeys}
    for sentence in sentences:
        for name in sentenceKeys:
            for item in sentence[name]:              
                if item not in vocabs[name]:
                    vocabs[name][item] = len(vocabs[name]) 
                    
    return vocabs  


    
def createDataMatrices(sentences, label_key, mappings):       
    addCharInformation(sentences)
    addCasingInformation(sentences)     
    testMatrix = createMatrices(sentences, mappings)             
       
    return testMatrix



def easOutput_to_afInput(predtags, test_texts, mappings, args):
    assert len(predtags) == len(test_texts)
    af_sents = []
    #ipdb.set_trace()
    for i in xrange(len(predtags)):
        assert len(predtags[i]) == len(test_texts[i]['tokens'])
        sents = test_texts[i]['sents']
        sent_acts = test_texts[i]['sent_acts']
        word2sent = test_texts[i]['word2sent']
        for act_idx, t in enumerate(predtags[i]):
            if t in ['1', '2', '3']: # action label
                sent_idx = word2sent[act_idx]
                word_ids = []
                if sent_idx > 0: # use the former sentence and current one
                    words = sents[sent_idx - 1] + sents[sent_idx] + ['UNKNOWN_TOKEN']
                    for k, v in word2sent.iteritems():
                        if v == sent_idx or v == sent_idx - 1:
                            word_ids.append(k)
                else:
                    words = sents[sent_idx] + ['UNKNOWN_TOKEN']
                    for k, v in word2sent.iteritems():
                        if v == sent_idx:
                            word_ids.append(k)
                end_idx = max(word_ids) # the last index of words of these two sents
                start_idx = min(word_ids)
                sent_len = len(words)
                acts = sent_acts[sent_idx]
                
                af_sent = {}
                tags = np.zeros(sent_len, dtype=np.int32)
                act_idxs = []
                obj_idxs = [[], []]
                for act in acts: # this is a tagged right action
                    if act['act_idx'] < sent_len:
                        act_idxs.append(act['act_idx'])
                    if act['act_idx'] != act_idx - start_idx:
                        continue
                    obj_idxs = act['obj_idxs']
                    if len(obj_idxs[1]) == 0:
                        tags[obj_idxs[0]] = 1 # essential objects
                    else:
                        tags[obj_idxs[0]] = 3 # exclusive objects
                        tags[obj_idxs[1]] = 3 # exclusive objects
                str_tags = [str(t) for t in tags]

                position = np.zeros(sent_len, dtype=np.int32)
                position.fill(act_idx - start_idx)
                distance = list(np.abs(np.arange(sent_len) - position))
                if args.use_act_tags:
                    act_tags = [int(pt)+1 for pt in predtags[i][start_idx: end_idx + 1]]
                    act_tags.append(1)
                    assert len(act_tags) == sent_len
                    af_sent['act_tags'] = act_tags

                af_sent['tokens'] = words
                af_sent['tags'] = str_tags
                af_sent['distance'] = distance
                af_sent['text_idx'] = i
                af_sent['sent_idx'] = sent_idx
                af_sent['act_idxs'] = act_idxs
                af_sent['obj_idxs'] = obj_idxs
                af_sent['act_idx'] = act_idx
                af_sent['action'] = test_texts[i]['tokens'][act_idx]
                af_sents.append(af_sent)
    print('Total af_sents: %d' % len(af_sents))
    #ipdb.set_trace()
    pred_afMatrix = createDataMatrices(af_sents, 'tags', mappings)
    return pred_afMatrix, af_sents



def read_af_sents(args):
    with open(args.data_name, 'rb') as f:
        indata = pickle.load(f)[-1]
        if args.actionDB == 'wikihow':
            indata = indata[:118]
    #ipdb.set_trace()
    raw_sents = []
    af_sents = []
    exc_obj_pairs = 0
    for i in xrange(len(indata)):
        #print('Text %d of %d'%(i, len(indata)))
        for j in xrange(len(indata[i])):
            if len(indata[i][j]) == 0:
                continue
            raw_sents.append(indata[i][j])
            words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNKNOWN_TOKEN']
            acts = indata[i][j]['acts']
            sent_len = len(words)
            act_idxs = [a['act_idx'] for a in indata[i][j]['acts']]
            act_tags = np.ones(sent_len, dtype=np.int32)
            act_tags[act_idxs] = 2
            for k in xrange(len(indata[i][j]['acts'])):
                tmp_af_sents = []
                act_idx = indata[i][j]['acts'][k]['act_idx']
                obj_idxs = indata[i][j]['acts'][k]['obj_idxs']
                # all indices in obj_idxs are int
                position = np.zeros(sent_len, dtype=np.int32)
                position.fill(act_idx)
                distance = list(np.abs(np.arange(sent_len) - position))
                if len(obj_idxs[1]) == 0:
                    af_sent = {}
                    af_tags = np.zeros(sent_len, dtype=np.int32)
                    af_tags[obj_idxs[0]] = 1 # essential objects
                    str_af_tags = [str(t) for t in af_tags]
                    if args.use_act_tags:
                        af_sent['act_tags'] = act_tags
                    af_sent['exc_objs'] = []
                    af_sent['tokens'] = words
                    af_sent['tags'] = str_af_tags
                    af_sent['distance'] = distance
                    tmp_af_sents.append(af_sent)
                else:
                    af_sent = {}
                    af_tags = np.zeros(sent_len, dtype=np.int32)
                    af_tags[obj_idxs[0]] = 3 # exclusive objects
                    str_af_tags = [str(t) for t in af_tags]
                    if args.use_act_tags:
                        af_sent['act_tags'] = act_tags
                    af_sent['exc_objs'] = obj_idxs[1]
                    af_sent['tokens'] = words
                    af_sent['tags'] = str_af_tags
                    af_sent['distance'] = distance
                    tmp_af_sents.append(af_sent)

                    af_sent = {}
                    af_tags = np.zeros(sent_len, dtype=np.int32)
                    af_tags[obj_idxs[1]] = 3 # exclusive objects
                    str_af_tags = [str(t) for t in af_tags]
                    if args.use_act_tags:
                        af_sent['act_tags'] = act_tags
                    af_sent['exc_objs'] = obj_idxs[0]
                    af_sent['tokens'] = words
                    af_sent['tags'] = str_af_tags
                    af_sent['distance'] = distance
                    tmp_af_sents.append(af_sent)
                    exc_obj_pairs += 1
                af_sents.append(tmp_af_sents)

    print('Total af_sents: %d\texc_obj_pairs: %d\n' % (len(af_sents), exc_obj_pairs))
    indices = ten_fold_split_idx(len(af_sents), args.ten_fold_indices, args.k_fold)
    folds = index2data(indices, af_sents)
    return folds


def read_eas_texts(args):
    if os.path.isfile(args.data_name):
        with open(args.data_name, 'rb') as f:
            indata = pickle.load(f)
            if args.actionDB == 'wikihow':
                indata = indata[:118]
    else:
        raise("[!] Data %s not found" % args.data_name)

    eas_texts = []
    record = {'op': [], 'ex': [], 'both': [], 'non': []}
    options = {}
    for i in xrange(len(indata)):
        if len(indata[i]['words']) == 0:
            ipdb.set_trace()
        print('Text %d of %d'%(i, len(indata)))
        tmp_eas_texts = []
        eas_text = {}
        eas_text['tokens'] = indata[i]['words']
        eas_text['sents'] = indata[i]['sents']
        eas_text['acts'] = indata[i]['acts']
        eas_text['sent_acts'] = indata[i]['sent_acts']
        eas_text['word2sent'] = indata[i]['word2sent']
        eas_text['act2related'] = {}
        required_actions = []
        optional_actions = []
        exclusive_actions = []
        exclusive_pairs = []
        for acts in indata[i]['acts']:
            eas_text['act2related'][acts['act_idx']] = acts['related_acts']
            if acts['act_type'] == 1:
                required_actions.append(acts['act_idx'])
            elif acts['act_type'] == 2:
                optional_actions.append(acts['act_idx'])
            elif acts['act_type'] == 3:
                if acts['act_idx'] not in exclusive_actions:
                    exclusive_actions.append(acts['act_idx'])
                    exclusive_actions.extend(acts['related_acts'])
                    pairs = [acts['act_idx']]
                    pairs.extend(acts['related_acts'])
                    exclusive_pairs.append(pairs)
        if len(optional_actions) > 0 and len(exclusive_actions) > 0:
            record['both'].append(i)
        elif len(optional_actions) > 0 and len(exclusive_actions) == 0:
            record['op'].append(i)
        elif len(optional_actions) == 0 and len(exclusive_actions) > 0:
            record['ex'].append(i)
        else:
            record['non'].append(i)

        tmp_ex_seq = []
        tmp_op_seq = []
        if len(exclusive_pairs) > 0:
            ex_nums = 1
            for p in exclusive_pairs:
                ex_nums *= len(p)
            _, tmp_ex_seq = all_ecs_acts(exclusive_pairs, [], [], 0)
            if len(tmp_ex_seq) != ex_nums:
                print('len(tmp_ex_seq) != ex_nums')
        
        if len(optional_actions) > 0:
            op_list = np.zeros([len(optional_actions), 2], dtype=np.uint8)
            op_list[:, 1] = 1
            _, tmp_op_seq = all_ecs_acts(op_list, [], [], 0)
            if len(tmp_op_seq) != 2 ** len(optional_actions):
                print('len(tmp_op_seq) != ex_nums')
        
        max_num = 2**args.max_expend_num
        option_num = max([len(tmp_ex_seq), len(tmp_op_seq), len(tmp_ex_seq)*len(tmp_op_seq)])
        if len(tmp_op_seq) > max_num and args.select_data:
            option_num = max_num
        if option_num not in options:
            options[option_num] = 1
        else:
            options[option_num] += 1
        if len(tmp_ex_seq) > 0 or len(tmp_op_seq) > 0:
            if len(tmp_ex_seq) == 0:
                for ii in xrange(option_num):
                    j = np.random.randint(len(tmp_op_seq))
                    op_act_idxs = []
                    for k in xrange(len(tmp_op_seq[j])):
                        if tmp_op_seq[j][k] == 1:
                            op_act_idxs.append(optional_actions[k])
                    eas_text['tags'] = np.zeros(len(indata[i]['words']), dtype=np.int32)
                    eas_text['tags'][op_act_idxs] = 2
                    eas_text['tags'][required_actions] = 1
                    eas_text['tags'] = [str(t) for t in eas_text['tags']]
                    tmp_eas_texts.append(eas_text)
            else:
                if len(tmp_op_seq) == 0:
                    for act_idxs in tmp_ex_seq:
                        eas_text['tags'] = np.zeros(len(indata[i]['words']), dtype=np.int32)
                        eas_text['tags'][act_idxs] = 3
                        eas_text['tags'][required_actions] = 1
                        eas_text['tags'] = [str(t) for t in eas_text['tags']]
                        tmp_eas_texts.append(eas_text)
                else:
                    for jj, act_idxs in enumerate(tmp_ex_seq):
                        #print('EXA %d of %d'%(jj, len(tmp_ex_seq)))
                        for ii in xrange(option_num):
                            j = np.random.randint(len(tmp_op_seq))
                            op_act_idxs = []
                            for k in xrange(len(tmp_op_seq[j])):
                                if tmp_op_seq[j][k] == 1:
                                    op_act_idxs.append(optional_actions[k])
                            eas_text['tags'] = np.zeros(len(indata[i]['words']), dtype=np.int32)
                            eas_text['tags'][act_idxs] = 3
                            eas_text['tags'][op_act_idxs] = 2
                            eas_text['tags'][required_actions] = 1
                            eas_text['tags'] = [str(t) for t in eas_text['tags']]
                            tmp_eas_texts.append(eas_text)
        else:
            eas_text['tags'] = np.zeros(len(indata[i]['words']), dtype=np.int32)
            eas_text['tags'][required_actions] = 1
            eas_text['tags'] = [str(t) for t in eas_text['tags']]
            tmp_eas_texts.append(eas_text)
        eas_texts.append(tmp_eas_texts)
    for k, v in record.iteritems():
        print('{}:\n{}\n'.format(k, v))
    for k, v in sorted(options.items(), key=lambda x:x[1], reverse=True):
        print(k, v)
    #ipdb.set_trace()
    print('Total eas texts: %d' % sum([len(t) for t in eas_texts]))
    #indices = ten_fold_split_idx(len(eas_texts), args.ten_fold_indices)
    #folds = index2data(indices, eas_texts)
    folds = split_data_with_prior(args.ten_fold_indices, record, eas_texts, args.k_fold)

    return folds


def split_data_with_prior(fname, record, data, k=10):
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            train_idx, valid_idx, test_idx = pickle.load(f)
    else:
        train_idx = []
        valid_idx = []
        test_idx = []
        for i in xrange(k):
            tmp_train_idx = []
            tmp_valid_idx = []
            for idx_type, idx_list in record.iteritems():
                np.random.shuffle(idx_list)
                split_point = int(len(idx_list) / k) + 1
                tmp_train_idx.extend(idx_list[split_point: ])
                tmp_valid_idx.extend(idx_list[: split_point])

            train_idx.append(tmp_train_idx)
            valid_idx.append(tmp_valid_idx)
            test_idx.append(tmp_valid_idx)
        with open(fname, 'wb') as f:
            pickle.dump([train_idx, valid_idx, test_idx], f)

    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': [], 'test': []}
    for i in xrange(k):
        train_sents = []
        valid_sents = []
        for j in train_idx[i]:
            train_sents.extend(data[j])
        for j in valid_idx[i]:
            valid_sents.extend(data[j])
        # ensure that training data are more than validation data
        if len(train_sents) > len(valid_sents):
            folds['train'].append(train_sents)
            folds['valid'].append(valid_sents)
            folds['test'].append(valid_sents)
        else:
            folds['train'].append(valid_sents)
            folds['valid'].append(train_sents)
            folds['test'].append(train_sents)
    
    for i in xrange(k):
        print('fold: %d' % i)
        for key in folds:
            print(key, len(folds[key][i]))
    #ipdb.set_trace()
    return folds 
        


def all_ecs_acts(act_list, cur_list, out_list, layer):
    if len(act_list) == 0:
        #print(cur_list)
        out_list.append(cur_list)
        return cur_list, out_list
    else:
        for idx in act_list[0]:
            cur_list.append(idx)
            #print('{}\n{}\n{}\n'.format(idx, cur_list, out_list))
            cur_list, out_list = all_ecs_acts(act_list[1:], cur_list, out_list, layer+1)
            cur_list = cur_list[: layer]
        cur_list = cur_list[: layer-1]
        return cur_list, out_list


           
def ten_fold_split_idx(num_data, fname, k=10, random=True):
    """
    Split data for 10-fold-cross-validation
    Split randomly or sequentially
    Retutn the indecies of splited data
    """
    print('Getting tenfold indices ...')
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading tenfold indices from %s\n' % fname)
            indices = pickle.load(f)
            return indices
    n = num_data/k
    indices = []

    if random:
        tmp_idxs = np.arange(num_data)
        np.random.shuffle(tmp_idxs)
        for i in range(k):
            if i == k - 1:
                indices.append(tmp_idxs[i*n: ])
            else:
                indices.append(tmp_idxs[i*n: (i+1)*n])
    else:
        for i in xrange(k):
            indices.append(range(i*n, (i+1)*n))

    with open(fname, 'wb') as f:
        pickle.dump(indices, f)
    return indices


def index2data(indices, data, extd=True):
    """
    Split data according to given indices
    """
    print('Spliting data according to indices ...')
    folds = {'train': [], 'valid': [], 'test': []}
    num_data = len(data)
    k = len(indices)
    for i in xrange(k):
        train_sents = []
        valid_sents = []
        test_sents = []
        if extd:
            for idx in xrange(num_data):
                if idx in indices[i]:
                    valid_sents.extend(data[idx])
                    test_sents.extend(data[idx])
                #elif idx in indices[(i+1)%k]:
                #    test_sents.extend(data[idx])
                else:
                    train_sents.extend(data[idx])
        else:
            for idx in xrange(num_data):
                if idx in indices[i]:
                    valid_sents.append(data[idx])
                    test_sents.append(data[idx])
                #elif idx in indices[(i+1)%k]:
                #    test_sents.append(data[idx])
                else:
                    train_sents.append(data[idx])

        folds['train'].append(train_sents)
        folds['valid'].append(valid_sents)
        folds['test'].append(test_sents)
    #ipdb.set_trace()
    return folds 


if __name__ == '__main__':
    #ipdb.set_trace()
    act_list = np.arange(9).reshape([-1, 3])
    cur_list, out_list = all_ecs_acts(act_list, [], [], 0)
    for l in out_list:
        print(l)
    print(len(out_list))