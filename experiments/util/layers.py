import tensorflow as tf
from tensorflow import summary

def _cname(namescope, varname):
    return '{}.{}'.format(namescope, varname) if namescope else varname

def full_layer(ain, shape, namescope, activation=None):
    layer = {}
    with tf.name_scope(namescope):
        W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
        b = tf.Variable(tf.ones([shape[-1]])/10.0, name='b')
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
        cacc = tf.reduce_mean(tf.cast(cpred, tf.float32), name='cacc')

        layer['cpred'] = cpred
        layer['cacc'] = cacc
        layer['summary'] = [
             summary.scalar(_cname(namescope,'cacc'), cacc),
        ]

    return layer


def word_accuracy(preds, accs, namescope):
    layer = {}    
    with tf.name_scope(namescope):
        pacc = tf.reduce_prod(accs, name='pacc')
        wacc = tf.reduce_mean(
                    tf.cast(tf.reduce_all(preds, axis=0),
                            tf.float32),
                    name='wacc')

        layer['pacc'] = pacc
        layer['wacc'] = wacc
        layer['summary'] = [
            summary.scalar(_cname(namescope,'pacc'), pacc),
            summary.scalar(_cname(namescope,'wacc'), wacc),
        ]

    return layer


def xentropy_loss(Ylogit, Ytrue, namescope, norm=1000):
    layer = {}
    with tf.name_scope(namescope):
        xent = tf.nn.softmax_cross_entropy_with_logits(
            logits=Ylogit, labels=Ytrue)
        
        loss = tf.reduce_mean(xent) * norm

        layer['loss'] = loss
        layer['xent'] = xent
        layer['summary'] = [
            summary.histogram(_cname(namescope,'xent'), xent),
            summary.scalar(_cname(namescope,'loss'), loss),
        ]

    return layer
