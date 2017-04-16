import tensorflow as tf

class AutoEncoder(object):
    def __init__(self, n, h):
        with tf.variable_scope('enc'):
            X = tf.placeholder(tf.float32, [None, n])
            Y = X # output = input

            W_e = tf.Variable(tf.random_normal([n,h]))
            b_e = tf.Variable(tf.random_normal([h]))

            W_d = tf.Variable(tf.random_normal([h,n]))
            b_d = tf.Variable(tf.random_normal([n]))

            ec = tf.nn.sigmoid(tf.add(tf.matmul(X,W_e),b_e)) # encoded
            dc = tf.nn.sigmoid(tf.add(tf.matmul(ec,W_d),b_d)) # decoded

            cost = tf.reduce_mean(tf.pow(Y - dc, 2))
            opt = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

            self.train = lambda sess,x : sess.run([cost,opt], feed_dict = {X : x})[0]
            self.test = lambda sess,x : sess.run(dc, feed_dict = {X : x})

    def init(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='enc')
        return tf.variables_initializer(variables)

