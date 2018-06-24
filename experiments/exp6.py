import pickle
import uuid

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
from models.base import LinearReshapeLayer, InputLayer, DropOutLayer
from models.conv import ConvLayer

TEST = False
CONVERGENCE_TOL_RATE = 0.3

if TEST:
    MIN_EPOCHS = 2
    MAX_EPOCHS = 2

    nepochs = MAX_EPOCHS
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }
else:
    MIN_EPOCHS = 20
    MAX_EPOCHS = 50


    nepochs = MAX_EPOCHS
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }

pkeep = 0.7
train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
#max_learning_rate = 0.01
#min_learning_rate = 0.0001



# max_learning_rate = 2
# min_learning_rate = 0.001



def stop_criterion(epoch, train_hist, test_hist):
    if epoch < MIN_EPOCHS:
        return False

    if epoch > MAX_EPOCHS:
        return True




def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    summary = []

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        summary.append(tf.summary.scalar('mean', mean))
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    
    summary.append(tf.summary.scalar('stddev', stddev))
    summary.append(tf.summary.scalar('max', tf.reduce_max(var)))
    summary.append(tf.summary.scalar('min', tf.reduce_min(var)))
    summary.append(tf.summary.histogram('histogram', var))

    return summary


def run(model_spec):
    model_id = uuid.uuid4()
    print('start training {}...'.format(model_id))
    print('lrate {} -> {}'.format(max_learning_rate, min_learning_rate))

    with tf.Session(graph=model_spec.g) as sess:

        #summary = []
        # for l,layer in enunmerate(model_spec.layers):
        #     if isinstance(layer, ConvLayer):

        #         with tf.name_scope('conv{}.W'.format(l)):
        #             summary += variable_summaries(layer.conv['W'])

        #         with tf.name_scope('conv{}.b'.format(l)):
        #             summary += variable_summaries(layer.conv['b'])

        model = model_spec.model

        with tf.name_scope('backprop'):
            learn_step = tf.placeholder(tf.float32)
            learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (learn_step/(nepochs -1))

            #train = [tf.train.AdamOptimizer(learning_rate).minimize(x) for x in model.loss]
            train = [tf.train.AdamOptimizer(learning_rate).minimize(x) for x in model.xent]
            #train = [tf.train.GradientDescentOptimizer(learning_rate).minimize(x) for x in model.xent]


            #summary.append(tf.summary.scalar('lrate', learning_rate))

        #writer = tf.summary.FileWriter('./experimento/{}'.format(model_id), graph)
        #sum_merged = tf.summary.merge(summary + model_spec.model.summ_ext)



        sess.run(tf.global_variables_initializer())
        def report(batch_gen):
            nw = 0 # Acerto palavra completa
            ls = np.zeros([5]) # loss/char
            nc = np.zeros([5]) #acerto/char
            for batch in batch_gen.gen_batches():
                _nw, _ls, _nc = sess.run([model.nacc, model.loss, model.cnacc], feed_dict=model_spec.food(batch))
                nw += _nw
                ls += _ls
                nc += _nc

            SIZE = batch_gen.max_seek * batch_gen.batch_size
            nw /= SIZE
            ls /= SIZE
            nc /= SIZE

            keys = ['wacc'] + ['loss_{}'.format(i) for i in range(5)] + ['loss_avg']  + ['acc_{}'.format(i) for i in range(5)] + ['wprob']
            vals = [nw]     + list(ls)                                + [sum(ls)/5.0] + list(nc)                               + [np.prod(nc)]
            return {k:v for k,v in zip(keys, vals)}


        train_dict = report(train_gen)
        train_dict.update({'train_time': 0, 'total_time': 0, 'epoch': 0})

        test_dict = report(val_gen)
        test_dict.update({'train_time': 0, 'total_time': 0, 'epoch': 0})

        train_list = [train_dict]
        test_list = [test_dict]

        for epoch in range(nepochs):
            #print(sess.run(learning_rate, feed_dict={learn_step: epoch}))

            tinit = time.time()
            for i, batch in enumerate(train_gen.gen_batches()):
                food = model_spec.food(batch, pkeep=pkeep)
                food.update({learn_step: epoch})
                sess.run(train, feed_dict=food)

                #if i > 250:
                #    break

            #_summary = sess.run(sum_merged, feed_dict=food)
            #writer.add_summary(_summary, epoch)


            tdelta_train = time.time() - tinit

            train_dict = report(train_gen)
            test_dict = report(val_gen)
            
            tdelta_total = time.time() - tinit

            #print('TRAIN:')
            #print(train_dict)
            #print('TEST:')
            #print(test_dict)
            print('J:{:.4f}/{:.4f} P:{:.4f}/{:.4f} P[0]:{:.4f}/{:.4f}'.format(
                train_dict['loss_avg'],
                test_dict['loss_avg'],
                train_dict['wacc'],
                test_dict['wacc'],
                train_dict['acc_0'],
                test_dict['acc_0']
            ))



            train_dict.update({'train_time': tdelta_train, 'total_time': tdelta_total, 'epoch': epoch + 1})
            test_dict.update({'train_time': tdelta_train, 'total_time': tdelta_total, 'epoch': epoch + 1})

            train_list.append(train_dict)
            test_list.append(test_dict)

    df_train = pd.DataFrame(train_list)
    df_test = pd.DataFrame(test_list)

    train_report = {
        'lrange': (min_learning_rate, max_learning_rate),
        'model': model_spec.desc(),
        'df_train': df_train,
        'df_test': df_test,
    }

    print('model:')
    print(train_report['model'])

    print('Train:')
    print(df_train[['loss_avg', 'wacc', 'wprob']])

    print('Test:')
    print(df_test[['loss_avg', 'wacc', 'wprob']])


    file_name = '{}.pkl'.format(model_id)
    with open(file_name, 'bw') as f:
        pickle.dump(train_report, f)


def C5s6C5s12RMch(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )


def C5s6C5s12RMchD(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        DropOutLayer(graph, 'dropout1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        DropOutLayer(graph, 'dropout2'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )



def C5s2o6C5s2o12C5s2o6RMch(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 2, graph, 'conv1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        ConvLayer([5, 5, 12, 12], 2, graph, 'conv2'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )



def C5s2o6C5s2o12C5s2o6RMchD(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 2, graph, 'conv1'),
        DropOutLayer(graph, 'dropout1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        DropOutLayer(graph, 'dropout2'),
        ConvLayer([5, 5, 12, 12], 2, graph, 'conv2'),
        DropOutLayer(graph, 'dropout3'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )



def C5s2o6C5s2o12C5s2o6Rfl100MchD(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 2, graph, 'conv1'),
        DropOutLayer(graph, 'dropout1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        DropOutLayer(graph, 'dropout2'),
        ConvLayer([5, 5, 12, 12], 2, graph, 'conv2'),
        DropOutLayer(graph, 'dropout3'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearLayer(100, graph, 'dense1'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )


def C5s6RMch(graph):
    return StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
    )



# model_func = C5s6C5s12RMch
# max_learning_rate = 0.001
# min_learning_rate = 0.00001
# graph = tf.Graph()
# model_spec = model_func(graph)
# run(model_spec)


#model_func = C5s6C5s12RMchD

# min_learning_rate = 0.0001
# max_learning_rate = 0.0001
# graph = tf.Graph()
# model_spec = model_func(graph)
# run(model_spec)


# max_learning_rate = 0.001
# min_learning_rate = 0.00001
# graph = tf.Graph()
# model_spec = model_func(graph)
# run(model_spec)


model_func = C5s2o6C5s2o12C5s2o6Rfl100MchD

# min_learning_rate = 0.0001
# max_learning_rate = 0.0001
# graph = tf.Graph()
# model_spec = model_func(graph)
# run(model_spec)


max_learning_rate = 0.001
min_learning_rate = 0.00001
graph = tf.Graph()
model_spec = model_func(graph)
run(model_spec)



model_func = C5s2o6C5s2o12C5s2o6RMchD

# min_learning_rate = 0.0001
# graph = tf.Graph()
# model_spec = model_func(graph)
# run(model_spec)


max_learning_rate = 0.01
min_learning_rate = 0.0001
graph = tf.Graph()
model_spec = model_func(graph)
run(model_spec)
