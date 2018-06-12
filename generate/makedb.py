import os
import pickle
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np


def onehot(text, CLASSES):
    _text = []
    for c in text:
        _text.append(np.asarray(CLASSES == c, dtype=np.int8))

    return np.asarray(_text, dtype=np.int8)


def is_valid(img, text, specs):
    assert img.shape == specs['shape']
    assert len(text) == specs['text_length']

    return True


def images_to_db(data_path, db_path, specs):
    SHAPE = specs['shape']
    VOCAB = specs['vocab']
    CLASSES = np.asarray(list(VOCAB))

    finds = os.listdir(data_path)[:20000]
    finds = [f for f in finds if f.split('.')[-1] == specs['format']]

    total = len(finds)
    ignored = []

    if specs['db_type'] == 'h5db':
        h5db = h5py.File(db_path, 'w')
        xset = h5db.create_dataset('X', (total,) + SHAPE, dtype=np.float32)
        yset = h5db.create_dataset('Y', 
                                    (total, specs['text_length'], len(VOCAB)), 
                                    dtype=np.int8)
    else:
        xset = np.zeros((total,) + SHAPE, dtype=np.float32)
        yset = np.zeros((total, specs['text_length'], len(VOCAB)), dtype=np.int8)

    print('looking for images....')
    cnt_read = 0
    for cnt, fname in enumerate(finds):
        path = os.path.join(data_path, fname)
        if cnt % int(total/20) == 0:
            print('read {}/{}.....'.format(cnt, total))

        try:
            text = re.search('_(.*)\.', fname).group(1)
            text = ''.join(re.findall('[{}]'.format(VOCAB), text))
            img = plt.imread(path, format=specs['format'])
            assert is_valid(img, text, specs)

        except Exception as e:
            ignored.append(path)
            print('File on file {}. Ignoring ....'.format(path))
            continue

        xset[cnt_read] = img
        yset[cnt_read] = onehot(text, CLASSES)
        cnt_read += 1

    print('read {}/{}.....'.format(cnt_read, total))
    print('ignored {}/{}.....'.format(len(ignored), total))

    idxs = np.random.permutation(cnt_read)
    cut = int(5*cnt_read/6)
    idxs_train = np.asarray(idxs[:cut])
    idxs_test = np.asarray(idxs[cut:])

    if specs['db_type'] == 'h5db':
        train = h5db.create_dataset('train', idxs_train.shape, dtype=idxs_train.dtype)
        test = h5db.create_dataset('test', idxs_test.shape, dtype=idxs_test.dtype)
        train[...] = idxs_train
        test[...] = idxs_test
        h5db.close()

    else:
        with open(db_path, 'wb') as f:
            pickle.dump({
                            'Xtrain':xset[idxs_train],
                            'Ytrain':yset[idxs_train],
                            'Xtest':xset[idxs_test],
                            'Ytest':yset[idxs_test],
                        }, f)

    print('done.')


specs = {
    'vocab': '0123456789abcdefghijklmnopqrstuvwxyz',
    'shape': (50, 200, 3),
    'text_length': 5,
    'format': 'png',
    'db_type': 'pkl',
}

data_path = './images'
#db_path = './data.hdf5'
db_path = './data20k.pkl'
images_to_db(data_path, db_path, specs)