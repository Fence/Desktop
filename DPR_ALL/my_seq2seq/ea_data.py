import os
import re
import pickle

home_dir = '/home/fengwf/Documents/DRL_data/'
folders = ['paragraph_instructions/', 'wikihow/', 'cooking_tutorials/old_out/',
            'windows_help_support/windows2k_articles/']
files = ['navigation_vocab.pkl', 'wikihow_vocab.pkl', 'cooking_vocab.pkl',
            'windows_vocab.pkl']
texts = [54, 52, 64, 33]

def build_dict(folder, text_num, file_name):
    vocab = {}
    for i in range(text_num):
        text = home_dir + folder + str(i+1) + '.txt'
        try:
            if os.path.exists(text):
                print 'text: %s'%text
                content = open(text).read()
                raw_words = re.split(r' |\n', content)
                for w in raw_words:
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
        except Exception as e:
            print '\nError: ', e

    with open(file_name, 'w') as f:
        pickle.dump(vocab, f)

def test_shell_loop(start_idx):
    print 'start index is %d'%start_idx


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-si', '--start_idx', type=int, default=-1, help='start index')
    args = parser.parse_args()
    test_shell_loop(args.start_idx)
    #import pdb
    #pdb.set_trace()
    '''
    for i in range(len(folders)):
        print '\nCurrent folder: %s'%folders[i]
        build_dict(folders[i], texts[i], files[i])
    '''