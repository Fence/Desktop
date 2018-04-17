import re
import ipdb
import jieba
import pickle
import json
from tqdm import tqdm
from gensim.models import Word2Vec

keys = ['1537790411', '2155226773', '3952070245', '1195230310', '1730726637', '1445962081', 
'1195242865', '1268518877', '2396658275', '1740577714', '2970036311', '2547916923', 
'1615743184', '512789686', '1965681503', '1086233511', '1784537661', '2150511032']


def preprocessing():
    #ipdb.set_trace()
    raw_sent = []
    sentences = []
    for i in range(len(keys)):
        print('\n\nNO.%d file of %d\n' % (i, len(keys)))
        obj = pickle.load(open('%s.pkl'%keys[i], 'rb')) 
        try:
            for j in tqdm(range(len(obj))):
                if not ('content' in obj[j] and 'comment' in obj[j]):
                    continue
                cont = [obj[j]['content']]
                cont.extend(obj[j]['comment'])
                for par in cont:
                    #sents = re.split(r'！。？', par)
                    raw_sent.append(par)
                    sents = re.findall('[\u4e00-\u9fa5|，、]+', par)
                    if len(sents) <= 1:
                        continue
                    else:
                        sents = sents[1:]
                    for s in sents:
                        sentences.append([w for w in jieba.cut(s)])
        except Exception as e:
            print(e)

    with open('sentences.json', 'w') as f:
        json.dump(sentences, f)
    with open('raw_sent.txt', 'w') as f:
        f.write('\n'.join(raw_sent))


class sentence_iterator():
    def __init__(self):
        self.keys = keys


    def __iter__(self):
        pass
        

if __name__ == '__main__':
    preprocessing()
