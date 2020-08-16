import argparse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import glob
import matplotlib.pyplot as plt
from load_data import DataGenerator
from models.ProtoNet import ProtoNet, ProtoLoss


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_path',
						type=str,
						default='./omniglot_resized',
						help='Path to the omniglot dataset.')
	parser.add_argument('--n-way',
						'-w',
						type=int,
						default=20,
						help="N-way classification")
	parser.add_argument('--k-shot',
						'-s',
						type=int,
						default=1,
						help="Perform K-shot learning")
	parser.add_argument('--n-query',
						'-q',
						type=int,
						default=5,
						help="Number of queries for Prototypical Networks")
	parser.add_argument('--n-meta-test-way',
						type=int,
						default=20,
						help="N-way classification at meta-test time")
	parser.add_argument('--k-meta-test-shot',
						type=int,
						default=5,
						help="Perform K-shot learning at meta-test time")
	parser.add_argument('--n-meta-test-query',
						type=int,
						default=5,
						help="Number of queries for Prototypical Networks at meta-test time")

	args = parser.parse_args()

	return args


if __name__ == '__main__':
	args = parse_args()

	n_epochs = 20
	n_episodes = 100
	n_way = args.n_way
	k_shot = args.k_shot
	n_query = args.n_query
	im_width, im_height, channels = 28, 28, 1
	num_filters = 16
	latent_dim = 16
	num_conv_layers = 3
	n_meta_test_episodes = 1000
	n_meta_test_way = args.n_meta_test_way
	k_meta_test_shot = args.k_meta_test_shot
	n_meta_test_query = args.n_meta_test_query

	x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
	q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
	x_shape = tf.shape(x)
	q_shape = tf.shape(q)
	num_classes, num_support = x_shape[0], x_shape[1]
	num_queries = q_shape[1]
	labels_ph = tf.placeholder(tf.float32, [None, None, None])
	model = ProtoNet([num_filters]*num_conv_layers, latent_dim)
	x_latent = model(tf.reshape(x, [-1, im_height, im_width, channels]))
	q_latent = model(tf.reshape(q, [-1, im_height, im_width, channels]))
	ce_loss, acc = ProtoLoss(x_latent, q_latent, labels_ph, num_classes, num_support, num_queries)
	train_op = tf.train.AdamOptimizer().minimize(ce_loss)
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth=True
	sess = tf.InteractiveSession(config=tf_config)
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

    # call DataGenerator with k_shot+n_query samples per class
	data_generator = DataGenerator(n_way, k_shot+n_query, n_meta_test_way, k_meta_test_shot+n_meta_test_query, config={'data_folder': args.data_path})
	for ep in range(n_epochs):
		for epi in range(n_episodes):
			#############################
			#### YOUR CODE GOES HERE ####

			# sample a batch of training data and partition into
		    # support and query sets
			
			support, query, labels = None, None, None
			#############################
			_, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, labels_ph:labels})
			if (epi+1) % 50 == 0:
				#############################
				#### YOUR CODE GOES HERE ####

                # sample a batch of validation data and partition into
		        # support and query sets

				support, query, labels = None, None, None
				#############################
				val_ls, val_ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, labels_ph:labels})
				print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}, meta-val loss: {:.5f}, meta-val acc: {:.5f}'.format(ep+1,
																																		n_epochs,
																																		epi+1,
																																		n_episodes,
																																		ls,
																																		ac,
																																		val_ls,
																																		val_ac))

	print('Testing...')
	meta_test_accuracies = []
	for epi in range(n_meta_test_episodes):
		#############################
		#### YOUR CODE GOES HERE ####

        # sample a batch of test data and partition into
        # support and query sets

		support, query, labels = None, None, None
		#############################
		ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, labels_ph:labels})
		meta_test_accuracies.append(ac)
		if (epi+1) % 50 == 0:
			print('[meta-test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_meta_test_episodes, ls, ac))
	avg_acc = np.mean(meta_test_accuracies)
	stds = np.std(meta_test_accuracies)
	print('Average Meta-Test Accuracy: {:.5f}, Meta-Test Accuracy Std: {:.5f}'.format(avg_acc, stds))

