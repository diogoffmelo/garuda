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


model_path = './graphs/conv2a/'

model_path_ckpts = model_path + 'ckpts/epoch' 
model_path_boards = model_path



nepochs = 20
dbargs = {
    'path': '../generate/datatest.hdf5',
}
batchargs = {
    'train_bsize': 10,
    'report_bsize': 10,
    'val_bsize': 0,
}

# nepochs = 30
# dbargs = {
#     'path': '../generate/data.hdf5',
# }
# batchargs = {
#     'train_bsize': 100,
#     'report_bsize': 100,
#     'val_bsize': 0,
# }


train_gen, report_gen, val_gen = load_to_batches(dbargs, batchargs)
graph = tf.Graph()


model = StackedLayers(
        InputLayer([None, 50, 200, 3], [None, 5, 36], graph, 'input'),
        # ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        
        # ConvLayer([5, 5, 6, 12], 2, graph, 'conv1'),
        LinearReshapeLayer(graph, 'reshape'),
        LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #ConvLayer([5, 5, 3, 6], 1, graph, 'conv1'),
        #ConvLayer([5, 5, 6, 12], 2, graph, 'conv1'),
        #LinearReshapeLayer(graph, 'reshape'),
        #LinearLayer([(50 * 200 * 3), 2000], graph, 'linear'),
        #LinearLayer([2000, 200], graph, 'LINEAR2'),
        #LinearMultiCharOutputLayer(5, graph, 'classificador'),
        #LinearSingleCharOutputLayer(0, graph, 'classificador'),
    ).model



with graph.as_default(), tf.name_scope('backprop'):
    train = [tf.train.AdamOptimizer(0.001).minimize(x) for x in model.xent]
    #train = tf.train.AdamOptimizer(0.001).minimize(model.xent)



writer = tf.summary.FileWriter(model_path_boards)
writer.add_graph(model.g)
sum_merged = tf.summary.merge(model.summ)

with tf.Session(graph=model.g) as sess:
    saver = tf.train.Saver(max_to_keep=nepochs+1)
    
    def epoch_hook(epoch):
        saver.save(sess, model_path_ckpts, global_step=epoch+1, write_meta_graph=False)
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

    #init = tf.global_variables_initializer()
    #sess.run(init)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, model_path_ckpts, global_step=0)

    cnt = 0
    for batch in train_gen.gen_batch_epochs(nepochs, epoch_hook):
        cnt += 1
        sess.run(train, feed_dict=model.food(batch))
        # if cnt == 9: #cnt % 10 == 0:
        #     print(report_meal(model, [model.food(batch)], sess))
        #     cnt = 0


    print('Vadation:')
    print(report(model, val_gen, sess))


tf.reset_default_graph()
with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph('./graphs/conv2a/ckpts/epoch-0.meta')
    saver.restore(sess, './graphs/conv2a/ckpts/epoch-0')
    g = sess.graph
    X = g.get_tensor_by_name('input.placeholders/X:0')
    Y = g.get_tensor_by_name('input.placeholders/Y:0')
    wacc = g.get_tensor_by_name('classificador.wacc/wacc:0')
    pacc = g.get_tensor_by_name('classificador.wacc/pacc:0')
    __X, __Y = list(val_gen.gen_batches())[0]

    for i in range(nepochs+1):
        saver.restore(sess, './graphs/conv2a/ckpts/epoch-{}'.format(i))
        print('Vadation:')

        print(sess.run([wacc, pacc], feed_dict={X: __X, Y: __Y}))

    #print(report(model, val_gen, sess))

#     saver = tf.train.import_meta_graph('my_test_model-1000.meta') 
#     saver.restore(sess,tf.train.latest_checkpoint('./'))
#     op_to_restore = graph.get_tensor_by_name("op_to_restore:0")





