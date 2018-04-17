import time
import pickle 
import numpy as np

pos_dict = pickle.load(open('3db_pos_dict.pkl', 'rb'))
pos = ['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB']#, 'X']
# 'X' means out of vocabulary words
#import ipdb
#ipdb.set_trace()
for dim in range(10):
    pos_vec = {}
    for word, poses in pos_dict.iteritems():
        tmp_vec = [0] * 11 * dim #np.zeros(12)
        if 'X' in poses:
            if len(poses) >= 2:
                print word, poses
                poses.remove('X')
            elif len(poses) == 1:
            #    pos_dict.pop(word)
                continue
        for p in poses:
            idx = pos.index(p)
            tmp_vec[idx*dim: (idx+1)*dim] = [1]*dim
        pos_vec[word] = tmp_vec
    print len(pos_dict),len(pos_vec)

    count = 0
    for w in pos_vec:
        print w,pos_dict[w],pos_vec[w]
        count += 1
        if count >= 10:
            break
    with open('dim%d_pos_vec.pkl'%dim, 'wb') as f:
        pickle.dump(pos_vec, f)            
