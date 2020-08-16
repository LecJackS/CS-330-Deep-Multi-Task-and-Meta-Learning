"""
Usage Instructions:
	5-way, 1-shot omniglot:
		python main.py --meta_train_iterations=15000 --meta_batch_size=25 --k_shot=1 --inner_update_lr=0.4 --num_inner_updates=1 --logdir=logs/omniglot5way/
	20-way, 1-shot omniglot:
		python main.py --meta_train_iterations=15000 --meta_batch_size=16 --k_shot=1 --n_way=20 --inner_update_lr=0.1 --num_inner_updates=5 --logdir=logs/omniglot20way/
	To run evaluation, use the '--meta_train=False' flag and the '--meta_test_set=True' flag to use the meta-test set.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from load_data import DataGenerator
from models.maml import MAML
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('n_way', 5, 'number of classes used in classification (e.g. 5-way classification).')

## Training options
flags.DEFINE_integer('meta_train_iterations', 15000, 'number of meta-training iterations.')
# batch size during each step of meta-update (testing, validation, training)
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('k_shot', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('inner_update_lr', 0.4, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('num_inner_updates', 1, 'number of inner gradient updates during meta-training.')
flags.DEFINE_integer('num_filters', 16, 'number of filters for conv nets.')
flags.DEFINE_bool('learn_inner_update_lr', False, 'learn the per-layer update learning rate.')

## Logging, saving, and testing options
flags.DEFINE_string('data_path', './omniglot_resized', 'path to the dataset.')
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('meta_train', True, 'True to meta-train, False to meta-test.')
flags.DEFINE_integer('meta_test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('meta_test_set', False, 'Set to true to test on the the meta-test set, False for the meta-training set.')
flags.DEFINE_integer('meta_train_k_shot', -1, 'number of examples used for gradient update during meta-training (use if you want to meta-test with a different number).')
flags.DEFINE_float('meta_train_inner_update_lr', -1, 'value of inner gradient step step during meta-training. (use if you want to meta-test with a different value)')
flags.DEFINE_integer('meta_test_num_inner_updates', 1, 'number of inner gradient updates during meta-test.')

def meta_train(model, saver, sess, exp_string, data_generator, resume_itr=0):
	SUMMARY_INTERVAL = 10    # interval for writing a summary (reduced from 100)
	SAVE_INTERVAL = 100
	PRINT_INTERVAL = 10      # interval for how often to print (reduced from 100)
	TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

	if FLAGS.log:
		train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
	print('Done initializing, starting training.')
	pre_accuracies, post_accuracies = [], []

	num_classes = data_generator.num_classes

	for itr in range(resume_itr, FLAGS.meta_train_iterations):
		#############################
		#### YOUR CODE GOES HERE ####

		# sample a batch of training data and partition into
		# group a (inputa, labela) and group b (inputb, labelb)

		inputa, inputb, labela, labelb = None, None, None, None
		#############################
		feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

		input_tensors = [model.metatrain_op]

		if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
			input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_inner_updates-1],
									model.total_accuracy1, model.total_accuracies2[FLAGS.num_inner_updates-1]])

		result = sess.run(input_tensors, feed_dict)

		if itr % SUMMARY_INTERVAL == 0:
			pre_accuracies.append(result[-2])
			if FLAGS.log:
				train_writer.add_summary(result[1], itr)
			post_accuracies.append(result[-1])

		if (itr!=0) and itr % PRINT_INTERVAL == 0:
			print_str = 'Iteration %d: pre-inner-loop accuracy: %.5f, post-inner-loop accuracy: %.5f' % (itr, np.mean(pre_accuracies), np.mean(post_accuracies))
			print(print_str)
			pre_accuracies, post_accuracies = [], []

		if (itr!=0) and itr % SAVE_INTERVAL == 0:
			saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

		if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
			#############################
			#### YOUR CODE GOES HERE ####

		    # sample a batch of validation data and partition into
		    # group a (inputa, labela) and group b (inputb, labelb)

			inputa, inputb, labela, labelb = None, None, None, None
			#############################
			feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
			input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_inner_updates-1]]

			result = sess.run(input_tensors, feed_dict)
			print('Meta-validation pre-inner-loop accuracy: %.5f, meta-validation post-inner-loop accuracy: %.5f' % (result[-2], result[-1]))

	saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_META_TEST_POINTS = 600

def meta_test(model, saver, sess, exp_string, data_generator, meta_test_num_inner_updates=None):
	num_classes = data_generator.num_classes

	np.random.seed(1)
	random.seed(1)

	meta_test_accuracies = []

	for _ in range(NUM_META_TEST_POINTS):
		#############################
		#### YOUR CODE GOES HERE ####

		# sample a batch of test data and partition into
		# group a (inputa, labela) and group b (inputb, labelb)

		inputa, inputb, labela, labelb = None, None, None, None
		#############################
		feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

		result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
		meta_test_accuracies.append(result)

	meta_test_accuracies = np.array(meta_test_accuracies)
	means = np.mean(meta_test_accuracies, 0)
	stds = np.std(meta_test_accuracies, 0)
	ci95 = 1.96*stds/np.sqrt(NUM_META_TEST_POINTS)

	print('Mean meta-test accuracy/loss, stddev, and confidence intervals')
	print((means, stds, ci95))

	out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'meta_test_ubs' + str(FLAGS.k_shot) + '_inner_update_lr' + str(FLAGS.inner_update_lr) + '.csv'
	out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'meta_test_ubs' + str(FLAGS.k_shot) + '_inner_update_lr' + str(FLAGS.inner_update_lr) + '.pkl'
	with open(out_pkl, 'wb') as f:
		pickle.dump({'mses': meta_test_accuracies}, f)
	with open(out_filename, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['update'+str(i) for i in range(len(means))])
		writer.writerow(means)
		writer.writerow(stds)
		writer.writerow(ci95)

def main():
	if FLAGS.meta_train == False:
		orig_meta_batch_size = FLAGS.meta_batch_size
		# always use meta batch size of 1 when testing.
		FLAGS.meta_batch_size = 1

    # call data_generator and get data with FLAGS.k_shot*2 samples per class
	data_generator = DataGenerator(FLAGS.n_way, FLAGS.k_shot*2, FLAGS.n_way, FLAGS.k_shot*2, config={'data_folder': FLAGS.data_path})

    # set up MAML model
	dim_output = data_generator.dim_output
	dim_input = data_generator.dim_input
	meta_test_num_inner_updates = FLAGS.meta_test_num_inner_updates
	model = MAML(dim_input, dim_output, meta_test_num_inner_updates=meta_test_num_inner_updates)
	model.construct_model(prefix='maml')
	model.summ_op = tf.summary.merge_all()

	saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth=True
	sess = tf.InteractiveSession(config=tf_config)

	if FLAGS.meta_train == False:
		# change to original meta batch size when loading model.
		FLAGS.meta_batch_size = orig_meta_batch_size

	if FLAGS.meta_train_k_shot == -1:
		FLAGS.meta_train_k_shot = FLAGS.k_shot
	if FLAGS.meta_train_inner_update_lr == -1:
		FLAGS.meta_train_inner_update_lr = FLAGS.inner_update_lr

	exp_string = 'cls_'+str(FLAGS.n_way)+'.mbs_'+str(FLAGS.meta_batch_size) + '.k_shot_' + str(FLAGS.meta_train_k_shot) + '.inner_numstep' + str(FLAGS.num_inner_updates) + '.inner_updatelr' + str(FLAGS.meta_train_inner_update_lr)

	resume_itr = 0
	model_file = None

	tf.global_variables_initializer().run()

	if FLAGS.resume or not FLAGS.meta_train:
		model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
		if FLAGS.meta_test_iter > 0:
			model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.meta_test_iter)
		if model_file:
			ind1 = model_file.index('model')
			resume_itr = int(model_file[ind1+5:])
			print("Restoring model weights from " + model_file)
			saver.restore(sess, model_file)

	if FLAGS.meta_train:
		meta_train(model, saver, sess, exp_string, data_generator, resume_itr)
	else:
		FLAGS.meta_batch_size = 1
		meta_test(model, saver, sess, exp_string, data_generator, meta_test_num_inner_updates)

if __name__ == "__main__":
	main()