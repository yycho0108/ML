#!/usr/bin/python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import sys
import png

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

    def output(self, X = None, xfn=tf.nn.relu, ofn=tf.sigmoid):
        O = X
        n = len(self.W)
        for i in range(n):
            w = self.W[i]
            b = self.b[i]

            O = tf.matmul(O,w) + b
            if (i+1) == n: # last layer
                if ofn is not None:
                    O = ofn(O)
            else:
                if xfn is not None:
                    O = xfn(O)
        return O
    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
    def init(self):
        return tf.variables_initializer(self.var_list())

### PARAMS ###
n_epoch = 100
batch_size = 128
learning_rate = 0.0002
n_batch = int(mnist.train.num_examples/batch_size)

n_input = 28*28
n_hidden = 256
n_noise = 128
n_class = 10


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

        self.generator = NNet([n_noise + n_class, n_hidden, n_input])
        self.discriminator = NNet([n_input + n_class, n_hidden, n_hidden, 1])

        self.X = tf.placeholder(tf.float32, [None, n_input])
        self.Y = tf.placeholder(tf.float32, [None, n_class]) # labels
        self.Z = tf.placeholder(tf.float32, [None, n_noise])

        self.G = self.generator.output( tf.concat([self.Z, self.Y], 1),ofn=tf.nn.relu)
        D_gene = self.discriminator.output( tf.concat([self.G, self.Y], 1),ofn=None)
        D_real = self.discriminator.output( tf.concat([self.X, self.Y], 1),ofn=None)

        def loss(lg,lb):
            return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits = lg, labels = (tf.ones_like(lg) if lb else tf.zeros_like(lg))
                        )
                    )

        self.loss_D = loss(D_real,1) + loss(D_gene, 0) # discriminator wants truth to be true only
        self.loss_G = loss(D_gene,1) # generator wants to be true!

        with tf.variable_scope(self.scope):
            self.train_D = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_D, var_list = self.discriminator.var_list())
            self.train_G = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_G, var_list = self.generator.var_list())

    def get_noise(self, batch_size):
        z = np.random.normal(size=(batch_size, n_noise))
        return z

    def train(self, sess, x, y):
        z = self.get_noise(batch_size)
        _, l_D = sess.run([self.train_D, self.loss_D], feed_dict = {self.X : x, self.Y : y, self.Z : z})
        _, l_G = sess.run([self.train_G, self.loss_G], feed_dict = {self.Z : z, self.Y : y})
        return l_D, l_G

    def init(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        return [self.generator.init(), self.discriminator.init(), tf.variables_initializer(variables)]

def train(gan, sess):
    for epoch in range(n_epoch):
        for i in range(n_batch):
            x, y = mnist.train.next_batch(batch_size)
            l_D, l_G = gan.train(sess, x, y)

        print 'Epoch:', '%04d' % (epoch + 1), \
              'D loss: {:.4}'.format(l_D), \
              'G loss: {:.4}'.format(l_G)

        ### SAVE FIGURES ###
        if epoch % 10 == 0:
            sample_size = 30
            noise = np.random.uniform(-1., 1., size=[sample_size, n_noise])
            samples = sess.run(gan.G, feed_dict={gan.Y: mnist.validation.labels[:sample_size], gan.Z: noise})

            fig, ax = plt.subplots(6, n_class, figsize=(n_class, 6))

            for i in range(n_class):
                for j in range(6):
                    ax[j][i].set_axis_off()

                for j in range(3):
                    ax[0+(j*2)][i].imshow(np.reshape(mnist.validation.images[i+(j*n_class)], (28, 28)))
                    ax[1+(j*2)][i].imshow(np.reshape(samples[i+(j*n_class)], (28, 28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

def save_png(samples, name):
    samples = np.reshape(samples*255, (-1,28,28))
    samples = np.hstack(samples).astype(np.uint8)
    png.from_array(samples,'L').save(name)

def main():
    with tf.Session() as sess:
        gan = GAN()
        sess.run(gan.init())

        saver = tf.train.Saver()
        if len(sys.argv) > 1 and sys.argv[1].lower() == 'load':
            # load ...
            saver.restore(sess, './ckpt/gan2.ckpt')
            print 'loaded'
        else:
            train(gan, sess)
            save_path = saver.save(sess, './ckpt/gan2.ckpt')
            print 'Model Saved in File : %s' % save_path

        ### CREATE LABELS ###
        numbers = [3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8]
        labels = map(lambda x : [float(i==x) for i in range(10)], numbers)
        labels = np.array(labels, dtype=np.float32)
        #####################
        #labels = np.eye(10, dtype=np.float32) # one-hot labels 0-9

        noise = np.random.uniform(-1., 1., size=[len(labels), n_noise])
        samples = sess.run(gan.G, feed_dict={gan.Y : labels, gan.Z : noise})

        save_png(samples, 'test.png')
        save_png(samples / np.max(samples), 'test_n.png') # normalized

            
if __name__ == "__main__":
    main()
