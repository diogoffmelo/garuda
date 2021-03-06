import tensorflow as tf
from tensorflow import summary


def _cname(namescope, varname):
    return '{}.{}'.format(namescope, varname) if namescope else varname


def conv_layer(ain, shape, namespace, **kwargs):
    layer = {}
    stridep = kwargs.get('stridep', 1)
    strides = kwargs.get('strides', [1, stridep, stridep, 1])
    padding = kwargs.get('padding', 'VALID')
    activation = kwargs.get('activation', None)
    with tf.name_scope(namespace):
        stdev = tf.sqrt(2.0/(shape[0] * shape[1] * shape[-1]))
        layer['W'] = tf.Variable(tf.truncated_normal(shape, stddev=stdev), name='W')
        layer['b'] = tf.Variable(tf.zeros([shape[-1]]), name='b')
        conv = tf.nn.conv2d(ain,
                            layer['W'],
                            strides=strides,
                            padding=padding,
                            name='conv')
        layer['a'] = tf.add(conv, layer['b'], name='convsig')
        layer['aout'] = activation(layer['a']) if activation else layer['a']
        layer['summary'] = [
            tf.summary.histogram('W', layer['W'], family=namespace),
            tf.summary.histogram('b', layer['b'], family=namespace),
            tf.summary.histogram('a', layer['a'], family=namespace),
        ]

    return layer


def conv_layer_max(ain, shape, namespace, **kwargs):
    layer = {}
    stridep = kwargs.get('stridep', 1)
    strides = kwargs.get('strides', [1, stridep, stridep, 1])
    padding = kwargs.get('padding', 'VALID')
    activation = kwargs.get('activation', None)
    with tf.name_scope(namespace):
        stdev = tf.sqrt(2.0/(shape[0] * shape[1] * shape[-1]))
        layer['W'] = tf.Variable(tf.truncated_normal(shape, stddev=stdev), name='W')
        layer['b'] = tf.Variable(tf.zeros([shape[-1]]), name='b')
        conv = tf.nn.conv2d(ain,
                            layer['W'],
                            strides=[1, 1, 1, 1],
                            padding=padding,
                            name='conv')
        layer['a'] = tf.add(conv, layer['b'], name='convsig')
        layer['_aout'] = activation(layer['a']) if activation else layer['a']
        layer['aout'] = tf.layers.max_pooling2d(layer['_aout'], strides[1], strides[1], 'VALID')

        layer['summary'] = [
            tf.summary.histogram('W', layer['W'], family=namespace),
            tf.summary.histogram('b', layer['b'], family=namespace),
            tf.summary.histogram('a', layer['a'], family=namespace),
        ]

    return layer


def full_layer(ain, shape, namescope, activation=None):
    layer = {}
    with tf.name_scope(namescope):
        stdev = tf.sqrt(2.0/shape[-1])
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
        b = tf.Variable(tf.zeros([shape[-1]]), name='b')
        a = tf.matmul(ain, W) + b
        aout = activation(a) if activation else a
        
        layer['ain'] = ain
        layer['W'] = W
        layer['b'] = b
        layer['aout'] = aout
        layer['summary'] = [
            summary.histogram(_cname(namescope,'W'), W),
            summary.histogram(_cname(namescope,'b'), b),
            summary.histogram(_cname(namescope,'aout'), aout),
        ]

    return layer


def char_accuracy(Ytrue, Ypred, namescope):
    layer = {}
    with tf.name_scope(namescope):
        cpred = tf.equal(tf.argmax(Ypred, 1), tf.argmax(Ytrue, 1))
        nacc = tf.reduce_sum(tf.cast(cpred, tf.int8), name='ncacc')
        cacc = tf.reduce_mean(tf.cast(cpred, tf.float32), name='cacc')

        layer['cpred'] = cpred
        layer['cacc'] = cacc
        layer['nacc'] = nacc
        layer['summary'] = [
             summary.scalar(_cname(namescope,'cacc'), cacc),
        ]

    return layer


def word_accuracy(preds, accs, namescope):
    layer = {}    
    with tf.name_scope(namescope):
        pacc = tf.reduce_prod(accs, name='pacc')
        nacc = tf.reduce_sum(
                    tf.cast(tf.reduce_all(preds, axis=0),
                            tf.float32),
                    name='nacc')
        wacc = tf.reduce_mean(
                    tf.cast(tf.reduce_all(preds, axis=0),
                            tf.float32),
                    name='wacc')

        layer['pacc'] = pacc
        layer['wacc'] = wacc
        layer['nacc'] = nacc
        layer['summary'] = [
            summary.scalar(_cname(namescope,'pacc'), pacc),
            summary.scalar(_cname(namescope,'wacc'), wacc),
        ]

    return layer


def xentropy_loss(Ylogit, Ytrue, namescope):
    layer = {}
    with tf.name_scope(namescope):
        xent = tf.nn.softmax_cross_entropy_with_logits(
            logits=Ylogit, labels=Ytrue)
        
        loss = tf.reduce_mean(xent)

        layer['loss'] = loss
        layer['xent'] = xent
        layer['summary'] = [
            summary.histogram(_cname(namescope,'xent'), xent),
            summary.scalar(_cname(namescope,'loss'), loss),
        ]

    return layer


