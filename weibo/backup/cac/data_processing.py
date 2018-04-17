import os
import pickle
import ipdb

ipdb.set_trace()
source = '/home/fengwf/Desktop/weibo/backup/cac/'
folders = os.listdir(source)
folders.sort()
file_name = 'weibo_content_and_comment.pkl'

wcac = {}
for fd in folders:
    subpath = source + fd + '/'
    if os.path.isdir(subpath):
        print('\nProcessing path: %s'%subpath)
        try:
            a = pickle.load(open(subpath+file_name, 'rb'))
            wcac.update(a)
        except Exception as e:
            print(e)
            print('Error file: %s\n'%subpath)
        print('\nwcac.keys:',wcac.keys())

with open(file_name, 'wb') as f:
    print('\nTry to save files...\n')
    pickle.dump(wcac, f)
    print('\nSuccessfully save {}\n'.format(file_name))

        
