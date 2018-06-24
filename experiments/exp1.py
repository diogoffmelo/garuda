import numpy as np
import tensorflow as tf

from util.batch import NumpyBatchGenerator, SampleMode
from util.report import report, report_meal
from util.load import load_to_batches

from models.base import StackedLayers
from models.linear import LinearMultiCharOutputLayer, LinearLayer
from models.linear import LinearSingleCharOutputLayer
from models.base import LinearReshapeLayer, InputLayer
from models.conv import ConvLayer


TEST = False
model_path = './graphs/linearw/'

model_path_ckpts = model_path + 'ckpts/epoch' 
model_path_boards = model_path


if TEST:
    nepochs = 4
    dbargs = {
        'path': '../generate/datatest.hdf5',
    }
    batchargs = {
        'train_bsize': 1,
        'report_bsize': 1,
        'val_bsize': 1,
    }
else:
    nepochs = 30
    dbargs = {
        'path': '../generate/data.hdf5',
    }
    batchargs = {
        'train_bsize': 10,
        'report_bsize': 10,
        'val_bsize': 10,
    }




train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
graph = tf.Graph()

model_spec = StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        ConvLayer([5, 5, 6, 12], 2, graph, 'conv2'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearLayer(200, graph, 'linear'),
        #LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        #ConvLayer([5, 5, 6, 12], 2, graph, 'conv1'),
        #LinearReshapeLayer(graph, 'reshape'),
        #LinearLayer([(50 * 200 * 3), 2000], graph, 'linear'),
        #LinearLayer([2000, 200], graph, 'LINEAR2'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #LinearSingleCharOutputLayer(0, graph, 'classificador'),
    )

model = model_spec.model



with graph.as_default(), tf.name_scope('backprop'):
    train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]
    #train = tf.train.AdamOptimizer(0.001).minimize(model.xent)

# writer = tf.summary.FileWriter(model_path_boards)
# writer.add_graph(model.g)
# sum_merged = tf.summary.merge(model.summ)

print('start training ...')

with tf.Session(graph=model.g) as sess:
    #saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=nepochs+1)
    
    def epoch_hook(epoch):
        print(report(model, val_gen, sess))
        #saver.save(sess, model_path_ckpts, global_step=epoch+1, write_meta_graph=False)
        # return
        # for vbatch in val_gen.gen_batches():
        #     summ = sess.run(sum_merged, feed_dict=model.food(vbatch))
        #     writer.add_summary(summ, epoch)

        # if epoch%3 == 0:
        #     print('Train @t={}'.format(epoch))
            #print(report(model, report_gen, sess))
            #for vbatch in val_gen.gen_batches():
            #    summ = sess.run(sum_merged, feed_dict=model.food(vbatch))
            #    writer.add_summary(summ, epoch)


    sess.run(tf.global_variables_initializer())
    #saver.save(sess, model_path_ckpts, global_step=0)

    cnt = 0
    for batch in train_gen.gen_batch_epochs(nepochs, epoch_hook):
        cnt += 1
        sess.run(train, feed_dict=model.food(batch))
        if cnt == 9: #cnt % 10 == 0:
            print(report_meal(model, [model_spec.food(batch)], sess))
            cnt = 0


    print('Vadation:')
    print(report(model, val_gen, sess))
