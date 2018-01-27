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
        correct_pred = tf.equal(tf.argmax(Ypred, 1), tf.argmax(Ytrue, 1))
        accuracy = tf.reduce_mean(
            tf.cast(correct_pred, tf.float32), name='accuracy')

        layer['correct_pred'] = correct_pred
        layer['accuracy'] = accuracy
        layer['summary'] = [
             summary.scalar(_cname(namescope,'accuracy'), accuracy),
        ]

    return layer


def word_accuracy(correct_preds, namescope):
    correct_preds = [tf.reshape(c, [-1, 1]) for c in correct_preds]
    layer = {}    
    with tf.name_scope(namescope):
        def match(acc, elems):
            return tf.foldl(
                    fn=tf.logical_and, 
                    elems=elems,
                    #initializer=tf.constant(True, shape=[1], dtype=tf.bool),
                    back_prop=False,
                )

        preds_red = tf.scan(
                fn=match,
                elems=correct_preds,
                initializer=tf.constant(True, shape=[1], dtype=tf.bool),
                back_prop=False,
            )    

        accuracy = tf.reduce_mean(
            tf.cast(preds_red, tf.float32), name='waccuracy')

        layer['accuracy'] = accuracy

    return layer


def xentropy_loss(Ylogit, Ytrue, namescope, norm=1000):
    layer = {}
    with tf.name_scope(namescope):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=Ylogit, labels=Ytrue)
        
        loss = tf.reduce_mean(xentropy) * norm

        layer['loss'] = loss
        layer['xentropy'] = xentropy
        layer['summary'] = [
            summary.histogram(_cname(namescope,'xentropy'), xentropy),
            summary.scalar(_cname(namescope,'loss'), loss),
        ]

    return layer
