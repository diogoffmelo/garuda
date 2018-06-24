import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import time

from util.batch import NumpyBatchGenerator, SampleMode
from util.report import report, report_meal
from util.load import load_to_batches

from models.base import StackedLayers
from models.linear import LinearMultiCharOutputLayer, LinearLayer
from models.linear import LinearSingleCharOutputLayer
from models.base import LinearReshapeLayer, InputLayer
from models.conv import ConvLayer

TEST = False

if TEST:
    nepochs = 20
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }
else:
    nepochs = 30
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 30,
        'report_bsize': 30,
        'val_bsize': 30,
    }


train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
graph = tf.Graph()




model = StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        #ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearLayer(200, graph, 'linear'),
        #LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        #ConvLayer([5, 5, 6, 12], 2, graph, 'conv1'),
        #LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #LinearSingleCharOutputLayer(0, graph, 'classificador'),
    ).model



with graph.as_default(), tf.name_scope('backprop'):

    # learning rate decay
    max_learning_rate = 0.01
    min_learning_rate = 0.01
    decay_speed = nepochs

    pkeep = tf.placeholder(tf.float32)    
    learn_step = tf.placeholder(tf.float32)

    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * tf.exp(-learn_step/decay_speed) 
    loss = sum(model.xent)

    #train = tf.train.AdamOptimizer(0.001).minimize(model.xent)
    
    train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]

    #train = tf.train.AdamOptimizer(0.001).minimize(loss)
    #train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]
    #train = tf.train.AdamOptimizer(0.001).minimize(model.xent)

# writer = tf.summary.FileWriter(model_path_boards)
# writer.add_graph(model.g)
# sum_merged = tf.summary.merge(model.summ)

print('start training ...')

with tf.Session(graph=model.g) as sess:
    
    sess.run(tf.global_variables_initializer())

    def report(batch_gen, epoch, tdelta):
        nw = 0 # Acerto palavra completa
        ls = np.zeros([5]) # loss/char
        nc = np.zeros([5]) #acerto/char
        for batch in batch_gen.gen_batches():
            _nw, _ls, _nc = sess.run([model.nacc, model.loss, model.cnacc], feed_dict=model.food(batch))
            nw += _nw
            ls += _ls
            nc += _nc

        SIZE = batch_gen.max_seek * batch_gen.batch_size
        nw /= SIZE
        ls /= SIZE
        nc /= SIZE

        keys = ['delta'] + ['wacc'] + ['loss_{}'.format(i) for i in range(5)] + ['loss_avg'] + ['acc_{}'.format(i) for i in range(5)] + ['wprob']
        vals = [tdelta, nw] + list(ls) + [sum(ls)/5.0] + list(nc) + [np.prod(nc)]
        return pd.DataFrame({k:v for k,v in zip(keys, vals)}, index=[epoch])

    df_train = report(train_gen, 0, 0)
    df_val = report(val_gen, 0, 0)
    print(df_train)
    print(df_val)


    #import matplotlib.pyplot as plt
    #plt.ion()
    #plt.figure(0)
    #plt.figure(1)

    for epoch in range(nepochs):
        print(sess.run(learning_rate, feed_dict={learn_step: epoch}))


        tinit = time.time()
        for batch in train_gen.gen_batches():
            food = model.food(batch)
            food.update({learn_step: epoch})
            sess.run(train, feed_dict=food)

        tfinal = time.time()
        tdelta = tfinal - tinit

        print('TRAIN/TEST:')
        df_train = df_train.append(report(train_gen, epoch+1, tdelta))
        df_val = df_val.append(report(val_gen, epoch+1, tdelta))
        #print(df_train.iloc[-1][['delta','loss_avg']])
        #print(df_val.iloc[-1][['delta','loss_avg']])
        

        print('TRAIN:')
        print(df_train[['loss_avg', 'wacc', 'wprob']])
        
        print('TEST:')
        print(df_val[['loss_avg', 'wacc', 'wprob']])

        #import ipdb; ipdb.set_trace()

        #plt.figure(0)
        #plt.clf()
        #plt.plot(list(df_train['loss_avg']), 'r*')
        #plt.plot(list(df_val['loss_avg']), 'b*')
        

        #plt.figure(1)
        #plt.clf()
        #plt.plot(list(df_train['wacc']), 'r*')
        #plt.plot(list(df_val['wacc']), 'b*')

        #plt.pause(0.0001)

        #plt.plot(range(epoch+1), df_train['loss_avg'])
        #plt.plot(range(epoch+1), df_val['loss_avg'])

        # plt.figure(0)
        # plt.clf()
        # plt.plot(range(epoch+1), df_train[['loss_avg']])


        #import ipdb; ipdb.set_trace()


        # batch_gen = train_gen
        # nw = 0 # Acerto palavra completa
        # ls = np.zeros([5]) # loss/char
        # nc = np.zeros([5]) #acerto/char
        # for batch in batch_gen.gen_batches():
        #     _nw, _ls, _nc = sess.run([model.nacc, model.loss, model.cnacc], feed_dict=model.food(batch))
        #     nw += _nw
        #     ls += _ls
        #     nc += _nc

        # SIZE = batch_gen.max_seek * batch_gen.batch_size
        # nw /= SIZE
        # ls /= SIZE
        # nc /= SIZE


        # print('TRAIN/TEST:')
        # print('wacc={} loss={} cacc={}'.format(nw, ls, nc))

        # batch_gen = val_gen
        # nw = 0 # Acerto palavra completa
        # ls = np.zeros([5]) # loss/char
        # nc = np.zeros([5]) #acerto/char
        # for batch in batch_gen.gen_batches():
        #     _nw, _ls, _nc = sess.run([model.nacc, model.loss, model.cnacc], feed_dict=model.food(batch))
        #     nw += _nw
        #     ls += _ls
        #     nc += _nc

        # SIZE = batch_gen.max_seek * batch_gen.batch_size
        # nw /= SIZE
        # ls /= SIZE
        # nc /= SIZE

        # print('wacc={} loss={} cacc={}'.format(nw, ls, nc))


    # df_columns = [, 'accw']
    # df_train = pd.DataFrame(columns=df_columns)
    # df_validation = pd.DataFrame(columns=df_columns)



    # cnt = 0
    # for batch in train_gen.gen_batch_epochs(nepochs, epoch_hook):
    #     cnt += 1
    #     sess.run(train, feed_dict=model.food(batch))
    #     if cnt % 10 == 0:
    #         nw, ls, nc = sess.run([model.nacc, model.loss, model.cnacc], feed_dict=model.food(batch))
    #         print('#wacc={} loss={} #cacc={}'.format(nw, ls, nc))

    #         #if cnt % 30 == 0:
    #         #print(report_meal(model, [model.food(batch)], sess))
    #         #cnt = 0


    # print('Vadation:')
    # print(report(model, val_gen, sess))

#plt.show(block=True)