import numpy as np
import tensorflow as tf
import time
import math
import pickle
from util.load import load_to_batches
import pandas as pd

TEST = False

if TEST:
    nepochs = 3
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }
else:
    nepochs = 40
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 30,
        'report_bsize': 30,
        'val_bsize': 30,
    }

pkeep = 0.7
train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
max_learning_rate = 0.01
min_learning_rate = 0.0001


TRAIN = True
img_high = 50
img_witdh = 200
img_depth = 3


img_size = img_high * img_witdh * img_depth
n_classes = 36



Xtrue = tf.placeholder(tf.float32, [None, img_high, img_witdh, img_depth])
Ytrue = tf.placeholder(tf.float32, [None, 5, n_classes])
pkeep = tf.placeholder(tf.float32, [])

N = 30000
O = n_classes

#W = tf.Variable(tf.zeros([img_size, N]))
#B = tf.Variable(tf.zeros([N]))

W51 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B51 = tf.Variable(tf.ones([O])/10)

W52 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B52 = tf.Variable(tf.ones([O])/10)

W53 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B53 = tf.Variable(tf.ones([O])/10)

W54 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B54 = tf.Variable(tf.ones([O])/10)

W55 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B55 = tf.Variable(tf.ones([O])/10)

W56 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B56 = tf.Variable(tf.ones([O])/10)

Y1true = tf.reshape(Ytrue[:, 0, :], [-1, n_classes])
Y2true = tf.reshape(Ytrue[:, 1, :], [-1, n_classes])
Y3true = tf.reshape(Ytrue[:, 2, :], [-1, n_classes])
Y4true = tf.reshape(Ytrue[:, 3, :], [-1, n_classes])
Y5true = tf.reshape(Ytrue[:, 4, :], [-1, n_classes])

YY = tf.reshape(Xtrue, shape=[-1, img_size])
#YYd = tf.nn.dropout(YY, pkeep)
#YYd = YY

#Y4 = tf.nn.relu(tf.matmul(YYd, W) + B)
#Y4d = tf.nn.dropout(Y4, pkeep)
#Y4d = Y4

Y4d = YY

Y51logits = tf.matmul(Y4d, W51) + B51
Y52logits = tf.matmul(Y4d, W52) + B52
Y53logits = tf.matmul(Y4d, W53) + B53
Y54logits = tf.matmul(Y4d, W54) + B54
Y55logits = tf.matmul(Y4d, W55) + B55

Y51preds = tf.nn.softmax(Y51logits)
Y52preds = tf.nn.softmax(Y52logits)
Y53preds = tf.nn.softmax(Y53logits)
Y54preds = tf.nn.softmax(Y54logits)
Y55preds = tf.nn.softmax(Y55logits)

cpreds1 = tf.equal(tf.argmax(Y51preds, 1), tf.argmax(Y1true, 1))
cpreds2 = tf.equal(tf.argmax(Y52preds, 1), tf.argmax(Y2true, 1))
cpreds3 = tf.equal(tf.argmax(Y53preds, 1), tf.argmax(Y3true, 1))
cpreds4 = tf.equal(tf.argmax(Y54preds, 1), tf.argmax(Y4true, 1))
cpreds5 = tf.equal(tf.argmax(Y55preds, 1), tf.argmax(Y5true, 1))

accuracy1 = tf.reduce_mean(tf.cast(cpreds1, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(cpreds2, tf.float32))
accuracy3 = tf.reduce_mean(tf.cast(cpreds3, tf.float32))
accuracy4 = tf.reduce_mean(tf.cast(cpreds4, tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(cpreds5, tf.float32))

xentropy1 = tf.nn.softmax_cross_entropy_with_logits(logits=Y51logits, labels=Y1true)
xentropy2 = tf.nn.softmax_cross_entropy_with_logits(logits=Y52logits, labels=Y2true)
xentropy3 = tf.nn.softmax_cross_entropy_with_logits(logits=Y53logits, labels=Y3true)
xentropy4 = tf.nn.softmax_cross_entropy_with_logits(logits=Y54logits, labels=Y4true)
xentropy5 = tf.nn.softmax_cross_entropy_with_logits(logits=Y55logits, labels=Y5true)

loss1 = tf.reduce_mean(xentropy1) * 1000
loss2 = tf.reduce_mean(xentropy2) * 1000
loss3 = tf.reduce_mean(xentropy3) * 1000
loss4 = tf.reduce_mean(xentropy4) * 1000
loss5 = tf.reduce_mean(xentropy5) * 1000

train1 = tf.train.AdamOptimizer(0.002).minimize(xentropy1)
train2 = tf.train.AdamOptimizer(0.002).minimize(xentropy2)
train3 = tf.train.AdamOptimizer(0.002).minimize(xentropy3)
train4 = tf.train.AdamOptimizer(0.002).minimize(xentropy4)
train5 = tf.train.AdamOptimizer(0.002).minimize(xentropy5)

train = [train1, train2, train3, train4, train5]
loss = [loss1, loss2, loss3, loss4, loss5]
acc = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5] 

def food(batch, istrain=False):
    return {
        Xtrue: batch[0],
        Ytrue: batch[1],
        pkeep: 0.7 if istrain else 1.0}


TRAIN = False


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    def report(batch_gen):
        ls = np.zeros([5]) # loss/char
        nc = np.zeros([5]) #acerto/char
        for batch in batch_gen.gen_batches():
            _ls, _acc = sess.run([loss, acc], feed_dict=food(batch))
            ls += _ls
            nc += _acc

        SIZE = batch_gen.max_seek * batch_gen.batch_size
        ls /= SIZE
        nc /= SIZE

        keys = ['loss_{}'.format(i) for i in range(5)] + ['loss_avg']  + ['acc_{}'.format(i) for i in range(5)] + ['wprob']
        vals = list(ls)                                + [sum(ls)/5.0] + list(nc)                              + [np.prod(nc)]
        return {k:v for k,v in zip(keys, vals)}

    train_list, test_list = [], []
    for epoch in range(nepochs):
        tinit = time.time()
        for i, batch in enumerate(train_gen.gen_batches()):
            sess.run(train, feed_dict=food(batch, istrain=True))

        tdelta_train = time.time() - tinit

        train_dict = report(train_gen)
        test_dict = report(val_gen)
        
        tdelta_total = time.time() - tinit

        #print('TRAIN:')
        #print(train_dict)
        #print('TEST:')
        #print(test_dict)
        print('J:{:.2f}/{:.2f} P:{:.2f}/{:.2f}'.format(train_dict['loss_avg'],
                                                       test_dict['loss_avg'],
                                                       train_dict['wprob'],
                                                       test_dict['wprob']))



        train_dict.update({'train_time': tdelta_train, 'total_time': tdelta_total, 'epoch': epoch + 1})
        test_dict.update({'train_time': tdelta_train, 'total_time': tdelta_total, 'epoch': epoch + 1})

        train_list.append(train_dict)
        test_list.append(test_dict)

    df_train = pd.DataFrame(train_list)
    df_test = pd.DataFrame(test_list)

    print('Train:')
    print(df_train[['loss_avg', 'wprob']])

    print('Test:')
    print(df_test[['loss_avg', 'wprob']])

