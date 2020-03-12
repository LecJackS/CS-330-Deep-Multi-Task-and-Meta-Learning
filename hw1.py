from __future__ import print_function
import sys
import numpy as np
import random
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 20, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 4,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    #bce  = tf.keras.losses.BinaryCrossentropy()
    #labels = tf.reshape(labels, tf.TensorShape(new_shape))
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds)
    loss = tf.keras.losses.categorical_crossentropy(y_true=labels[:,-1:,:,:], y_pred=preds[:,-1:,:,:], from_logits=True)
    loss = tf.reduce_mean(loss)
    #############################
    return loss

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels:       [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        # Data processing as in SNAIL
        _,K,N,I = input_images.shape
        # First K data+labels (for each N classes)
        data1 = tf.concat([input_images[:,0:-1,:,:],
                           input_labels[:,0:-1,:,:]], axis=3)
        # Last N examples of data+zeros 
        data2 = tf.concat([input_images[:,-1:,:,:],
                           tf.zeros_like(input_labels)[:,-1:,:,:]], axis=3)
        data = tf.concat([data1, data2], axis=1)
        data = tf.reshape(data, [-1, K*N, I+N])
        # Pass data through network
        x = self.layer1(data)#[i,:,:,:])
        x = self.layer2(x)
        # Return original shape [B,K+1,N,N]
        out = tf.reshape(x, [-1, K, N, N])
        #############################
        return out

#Placeholders for images and labels
ims = tf.placeholder(tf.float32,
            shape=(None,
                   FLAGS.num_samples + 1,
                   FLAGS.num_classes,
                   784))
labels = tf.placeholder(tf.float32,
            shape=(None,
                   FLAGS.num_samples + 1,
                   FLAGS.num_classes,
                   FLAGS.num_classes))

data_generator = DataGenerator(
                    FLAGS.num_classes,
                    FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
print("...tf.trainable_variables():",tf.trainable_variables())

optim = tf.train.AdamOptimizer(0.001)
#optim = tf.compat.v1.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)
print("...Starts training...")
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    
    for step in range(50000):
        i, l = data_generator.sample_batch(batch_type='train', batch_size=FLAGS.meta_batch_size)
        #print("i.shape:",i.shape)
        feed = {ims:    i.astype(np.float32),
                labels: l.astype(np.float32)}
        #print("feed[ims].shape:", feed[ims].shape)
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch(batch_type='test', batch_size=100)
            feed = {ims:    i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                    -1, FLAGS.num_samples + 1,
                    FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            print("Test Accuracy", (1.0 * (pred == l)).mean())
