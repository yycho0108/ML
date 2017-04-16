import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

class NNet(object):
    idx = 0
    def __init__(self, topology, scope=None):
        self.idx = NNet.idx
        NNet.idx += 1

        if scope is None:
            scope = 'nnet_%d' % self.idx
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.W = []
            self.b = []
            for i, (I, O) in enumerate(zip(topology[:-1], topology[1:])):
                w = tf.get_variable(('W_%d' % i), (I,O), initializer=tf.random_normal_initializer(stddev=0.01))
                b = tf.get_variable(('b_%d' % i), (O,), initializer=tf.zeros_initializer())
                self.W.append(w)
                self.b.append(b)

    def output(self, X = None):
        O = X
        n = len(self.W)
        for i in range(n):
            w = self.W[i]
            b = self.b[i]

            O = tf.matmul(O,w) + b
            if (i+1) == n: # last layer
                O = tf.sigmoid(O)
            else:
                O = tf.nn.relu(O)
        return O
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
    def init(self):
        return tf.variables_initializer(self.var_list())

### PARAMS ###
n_epoch = 20
batch_size = 100
learning_rate = 0.0002
n_batch = int(mnist.train.num_examples/batch_size)

n_input = 28*28
n_hidden = 256
n_noise = 128


class GAN(object):
    idx = 0
    def __init__(self, scope=None):

        ### SCOPE ###
        self.idx = GAN.idx
        GAN.idx += 1
        if scope is None:
            scope = 'gan_%d' % self.idx
        self.scope = scope
        #############

        self.generator = NNet([n_noise, n_hidden, n_input])
        self.discriminator = NNet([n_input, n_hidden, 1])

        self.X = tf.placeholder(tf.float32, [None, n_input])
        self.Z = tf.placeholder(tf.float32, [None, n_noise])

        self.G = self.generator.output(self.Z)
        D_gene = self.discriminator.output(self.G)
        D_real = self.discriminator.output(self.X)

        self.loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
        self.loss_G = tf.reduce_mean(tf.log(D_gene))

        with tf.variable_scope(self.scope):
            self.train_D = tf.train.AdamOptimizer(learning_rate).minimize(-self.loss_D, var_list = self.discriminator.var_list())
            self.train_G = tf.train.AdamOptimizer(learning_rate).minimize(-self.loss_G, var_list = self.generator.var_list())

    def get_noise(self, batch_size):
        z = np.random.normal(size=(batch_size, n_noise))
        return z

    def train(self, sess, x):
        z = self.get_noise(batch_size)
        _, l_D = sess.run([self.train_D, self.loss_D], feed_dict = {self.X : x, self.Z : z})
        _, l_G = sess.run([self.train_G, self.loss_G], feed_dict = {self.Z : z})
        return l_D, l_G

    def init(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return [self.generator.init(), self.discriminator.init(), tf.variables_initializer(variables)]

def main():
    with tf.Session() as sess:
        gan = GAN()
        sess.run(gan.init())
        #sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            for i in range(n_batch):
                x, y = mnist.train.next_batch(batch_size)
                l_D, l_G = gan.train(sess, x)

            print 'Epoch:', '%04d' % (epoch + 1), \
                  'D loss: {:.4}'.format(l_D), \
                  'G loss: {:.4}'.format(l_G)

            ### SAVE FIGURES ###
            sample_size = 10
            noise = gan.get_noise(sample_size)
            samples = sess.run(gan.G, feed_dict={gan.Z: noise})

            fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28, 28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)



if __name__ == "__main__":
    main()




















