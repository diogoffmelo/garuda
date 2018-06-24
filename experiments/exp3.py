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

if TEST:
    nepochs = 3
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 30,
        'report_bsize': 30,
        'val_bsize': 30,
    }
else:
    nepochs = 50
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

def run(model_spec):
    print('start training ...')


    #writer = tf.summary.FileWriter('./experimento', graph)
    #sum_merged = tf.summary.merge(model_spec.model.summ_ext)

    with tf.Session(graph=model_spec.g) as sess:
        model = model_spec.model

        with tf.name_scope('backprop'):
            learn_step = tf.placeholder(tf.float32)
            learning_rate = max_learning_rate + (min_learning_rate - max_learning_rate) * (learn_step/(nepochs -1))
            train = [tf.train.AdamOptimizer(learning_rate).minimize(x) for x in model.xent]


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


                #summary, _ = sess.run([sum_merged, train], feed_dict=food)
                #writer.add_summary(summary, i+epoch*train_gen.set_size)


            tdelta_train = time.time() - tinit

            train_dict = report(train_gen)
            test_dict = report(val_gen)
            
            tdelta_total = time.time() - tinit

            #print('TRAIN:')
            #print(train_dict)
            #print('TEST:')
            #print(test_dict)
            # print('J:{:.2f}/{:.2f} P:{:.2f}/{:.2f}'.format(train_dict['loss_avg'],
            #                                                test_dict['loss_avg'],
            #                                                train_dict['wacc'],
            #                                                test_dict['wacc']))



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


    file_name = '{}.pkl'.format(uuid.uuid4())
    with open(file_name, 'bw') as f:
        pickle.dump(train_report, f)



for lrate in [0.5, 0.005, 0.0001]:
    max_learning_rate = lrate
    min_learning_rate = lrate

    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                LinearReshapeLayer(graph, 'reshape'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass

    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                LinearReshapeLayer(graph, 'reshape'),
                DropOutLayer(graph, 'reg1'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass

    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
                LinearReshapeLayer(graph, 'reshape'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass


    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
                DropOutLayer(graph, 'reg1'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
                DropOutLayer(graph, 'reg2'),
                LinearReshapeLayer(graph, 'reshape'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass



    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv3'),
                LinearReshapeLayer(graph, 'reshape'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass



    try:
        graph = tf.Graph()
        model_spec = StackedLayers(
                InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
                ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
                DropOutLayer(graph, 'reg1'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
                DropOutLayer(graph, 'reg2'),
                ConvLayer([5, 5, 6, 12], 2, graph, 'conv3'),
                DropOutLayer(graph, 'reg3'),
                LinearReshapeLayer(graph, 'reshape'),
                LinearMultiCharOutputLayer(5, graph, 'classificador'),
            )
        run(model_spec)

    except Exception as e:
        print(e)
        pass


max_learning_rate = 0.5
min_learning_rate = 0.0001

try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            LinearReshapeLayer(graph, 'reshape'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass

try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            LinearReshapeLayer(graph, 'reshape'),
            DropOutLayer(graph, 'reg1'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass

try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
            LinearReshapeLayer(graph, 'reshape'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass


try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
            DropOutLayer(graph, 'reg1'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
            DropOutLayer(graph, 'reg2'),
            LinearReshapeLayer(graph, 'reshape'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass



try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv3'),
            LinearReshapeLayer(graph, 'reshape'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass


try:
    graph = tf.Graph()
    model_spec = StackedLayers(
            InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
            ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
            DropOutLayer(graph, 'reg1'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
            DropOutLayer(graph, 'reg2'),
            ConvLayer([5, 5, 6, 12], 2, graph, 'conv3'),
            DropOutLayer(graph, 'reg3'),
            LinearReshapeLayer(graph, 'reshape'),
            LinearMultiCharOutputLayer(5, graph, 'classificador'),
        )
    run(model_spec)

except Exception as e:
    print(e)
    pass







# max_learning_rate = 0.001
# min_learning_rate = 0.001
# graph = tf.Graph()
# model_spec = StackedLayers(
#         InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#         LinearReshapeLayer(graph, 'reshape'),
#         LinearMultiCharOutputLayer(5, graph, 'classificador'),
#     )
# run(model_spec)


#     max_learning_rate = 0.001
#     min_learning_rate = 0.001
#     graph = tf.Graph()
#     model_spec = StackedLayers(
#             InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#             LinearReshapeLayer(graph, 'reshape'),
#             LinearLayer([400], graph, 'linear'),
#             LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         )
#     run(model_spec)

# try:
#     max_learning_rate = 0.001
#     min_learning_rate = 0.001
#     graph = tf.Graph()
#     model_spec = StackedLayers(
#             InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#             LinearReshapeLayer(graph, 'reshape'),
#             LinearLayer(400, graph, 'linear'),
#             LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         )
#     run(model_spec)
# except Exception as e:
#     print(e)
#     pass

# try:
#     max_learning_rate = 0.001
#     min_learning_rate = 0.001
#     graph = tf.Graph()
#     model_spec = StackedLayers(
#             InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#             LinearReshapeLayer(graph, 'reshape'),
#             LinearLayer(800, graph, 'linear'),
#             LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         )
#     run(model_spec)
# except Exception as e:
#     print(e)
#     pass

# try:
#     max_learning_rate = 0.001
#     min_learning_rate = 0.001
#     graph = tf.Graph()
#     model_spec = StackedLayers(
#             InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#             LinearReshapeLayer(graph, 'reshape'),
#             LinearLayer(2000, graph, 'linear'),
#             LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         )
#     run(model_spec)
# except Exception as e:
#     print(e)
#     pass


# max_learning_rate = 0.0001
# min_learning_rate = 0.0001
# graph = tf.Graph()
# model_spec = StackedLayers(
#         InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#         LinearReshapeLayer(graph, 'reshape'),
#         LinearLayer(200, graph, 'linear'),
#         LinearMultiCharOutputLayer(5, graph, 'classificador'),
#     )
# run(model_spec)




# graph = tf.Graph()
# model_spec = StackedLayers(
#         InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#         ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
#         ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
#         LinearReshapeLayer(graph, 'reshape'),
#         LinearLayer(200, graph, 'linear'),
#         LinearMultiCharOutputLayer(5, graph, 'classificador'),
#     )
# run(model_spec)





# graph = tf.Graph()
# model_spec = StackedLayers(
#         InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
#         #ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
#         #ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
#         LinearReshapeLayer(graph, 'reshape'),
#         LinearLayer(200, graph, 'linear'),
#         #LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         #ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
#         #ConvLayer([5, 5, 6, 12], 2, graph, 'conv1'),
#         #LinearReshapeLayer(graph, 'reshape'),
#         LinearMultiCharOutputLayer(5, graph, 'classificador'),
#         #LinearSingleCharOutputLayer(0, graph, 'classificador'),
#     )

# run(model_spec)
