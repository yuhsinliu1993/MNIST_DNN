import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os
import yaml


# Load Hyperparameter settings
with open('hyperparameters.yaml', 'r') as f:
    configs = yaml.load(f)


def add_layer(inputs, in_size, out_size, n_layer, keep_prob, activation=None, reg_type=None, reg_constant=0.0):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer

    with tf.variable_scope(layer_name):
        if reg_type is None:
            weights = tf.get_variable(name='weights%d' % n_layer, shape=[
                                      in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(configs['img_size']))))
        elif reg_type == 'l1':
            weights = tf.get_variable(name='weights%d' % n_layer, shape=[in_size, out_size], initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(configs['img_size']))), regularizer=layers.l1_regularizer(reg_constant))
        elif reg_type == 'l2':
            weights = tf.get_variable(name='weights%d' % n_layer, shape=[in_size, out_size], initializer=tf.truncated_normal_initializer(
                stddev=1.0 / np.sqrt(float(configs['img_size']))), regularizer=layers.l2_regularizer(reg_constant))
        else:
            weights = tf.get_variable(name='weights%d' % n_layer, shape=[
                                      in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(configs['img_size']))))

        biases = tf.Variable(tf.zeros([1, out_size]) +
                             0.1, name='biases%d' % n_layer)
        Wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

        if activation is None:
            outputs = Wx_plus_b
        else:
            outputs = activation(Wx_plus_b)
            outputs = tf.nn.dropout(x=outputs, keep_prob=keep_prob)

        weight_summary = tf.summary.histogram('weights', weights)

    return outputs, weight_summary


def loss_function(logits, labels, using_reg=False):
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy')

        if using_reg:
            l = tf.reduce_mean(
                cross_entropy) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            l = tf.reduce_mean(cross_entropy)

        loss_summary = tf.summary.scalar('loss', l)

    return l, loss_summary


def training(loss, learning_rate, opt_type=None):
    if opt_type == 'sgd' or opt_type is None:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate).minimize(loss)
    elif opt_type == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)

    return optimizer


def calculate_accuracy(logits, y_true):
    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    return accuracy, accuracy_summary


def build_dnn_model(x, fc_neuron_layers, reg, keep_prob, reg_constant):
    model = {}
    weights_summary = {}
    for i in range(len(fc_neuron_layers)):
        if i == 0:      # First layer
            model[i], weights_summary[i] = add_layer(inputs=x,
                                                     in_size=configs[
                                                         'img_size'],
                                                     out_size=fc_neuron_layers[
                                                         0],
                                                     n_layer=i + 1,
                                                     activation=tf.nn.relu,
                                                     keep_prob=keep_prob,
                                                     reg_type=reg,
                                                     reg_constant=reg_constant)
        elif i == len(fc_neuron_layers) - 1:  # last layer
            model[i], weights_summary[i] = add_layer(inputs=model[i - 1],
                                                     in_size=fc_neuron_layers[
                                                         i - 1],
                                                     out_size=fc_neuron_layers[
                                                         i],
                                                     n_layer=i + 1,
                                                     activation=None,
                                                     keep_prob=keep_prob,
                                                     reg_type=reg,
                                                     reg_constant=reg_constant)
        else:
            model[i], weights_summary[i] = add_layer(inputs=model[i - 1],
                                                     in_size=fc_neuron_layers[
                                                         i - 1],
                                                     out_size=fc_neuron_layers[
                                                         i],
                                                     n_layer=i + 1,
                                                     activation=None,
                                                     keep_prob=keep_prob,
                                                     reg_type=reg,
                                                     reg_constant=reg_constant)

    return model, weights_summary


def create_log_folders(name):
    log_path = 'results/%s/losses' % name
    train_log_path = 'results/%s/train_logs' % name
    val_log_path = 'results/%s/val_logs' % name

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(train_log_path):
        os.makedirs(train_log_path)
    if not os.path.exists(val_log_path):
        os.makedirs(val_log_path)

    return log_path, train_log_path, val_log_path


def _get_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--regularization", type=str,
                        help="Specify the type of regularization. ( l1 or l2 )")
    parser.add_argument("-o", "--optimization", type=str,
                        help="Specify the optimization")
    parser.add_argument("-l", "--layers", nargs="+", type=int,
                        help="Specify the number of neurons of each layer. (e.g. -l 256 266 10)", required=True)
    parser.add_argument("-a", "--alpha", type=float,
                        help="Specify the regularization constant")
    parser.add_argument("-t", "--lr", type=float,
                        help="Specify the learning rate")
    parser.add_argument("-p", "--keepprob", type=float,
                        help="Using dropout and specify the dropout prob_keep ")
    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Print out the loss/accuracy in each epoch")

    return vars(parser.parse_args())


def run(**kwargs):

    if not kwargs:
        kwargs = _get_kwargs()

    if kwargs['regularization'] in ['l1', 'l2']:
        using_reg = True
    else:
        using_reg = False

    if kwargs['keepprob'] is not None:
        using_dropout = True
        keepprob = kwargs['keepprob']
    else:
        using_dropout = False
        keepprob = 1.

    if kwargs['alpha'] is None:
        kwargs['alpha'] = 0

    # Load MNIST dataset
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, configs['img_size']], name='x')
    y_true = tf.placeholder(
        tf.float32, shape=[None, configs['n_classes']], name='y_true')
    keep_prob = tf.placeholder(tf.float32)

    # Build Model
    fc_neuron_layers = kwargs['layers']
    model, weights_summary = build_dnn_model(x, fc_neuron_layers=fc_neuron_layers, reg=kwargs[
                                             'regularization'], keep_prob=keep_prob, reg_constant=kwargs['alpha'])

    pred = tf.nn.softmax(logits=model[len(fc_neuron_layers) - 1])

    accuracy, accuracy_summary = calculate_accuracy(
        model[len(fc_neuron_layers) - 1], y_true)
    loss, loss_summary = loss_function(
        logits=model[len(fc_neuron_layers) - 1], labels=y_true, using_reg=using_reg)

    optimizer = training(
        loss, kwargs['lr'], kwargs['optimization'])

    merged_LW = tf.summary.merge(
        [weights_summary[i]for i in range(len(fc_neuron_layers))] + [loss_summary])
    # merged_L = tf.summary.merge([loss_summary])
    merged_A = tf.summary.merge([accuracy_summary])

    fc_name = ''
    for neuron in fc_neuron_layers:
        fc_name += '_' + str(neuron)

    if using_dropout:
        folder_name = '%s_%.e_%.e_%s%s_%d_%.e' % (kwargs['regularization'], kwargs['alpha'], kwargs['keepprob'], kwargs[
            'optimization'], fc_name, (len(fc_neuron_layers)), kwargs['lr'])
    else:
        folder_name = '%s_%.e_%s%s_%d_%.e' % (kwargs['regularization'], kwargs['alpha'], kwargs[
            'optimization'], fc_name, (len(fc_neuron_layers)), kwargs['lr'])

    log_path, train_log_path, val_log_path = create_log_folders(folder_name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        train_summary_writer = tf.summary.FileWriter(
            train_log_path, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_path, sess.graph)

        for epoch in range(configs['training_epochs']):
            avg_loss = 0
            total_batch = int(data.train.num_examples / configs['batch_size'])

            for i in range(total_batch):
                x_train, y_train = data.train.next_batch(configs['batch_size'])
                # x_val, y_val = data.validation.next_batch(
                #     configs['batch_size'])
                # x_test, y_test = data.test.next_batch(configs['batch_size'])

                _, l, lw_summary = sess.run([optimizer, loss, merged_LW], feed_dict={
                    x: x_train, y_true: y_train, keep_prob: keepprob})
                summary_writer.add_summary(lw_summary, epoch * total_batch + i)

                avg_loss += l / total_batch

            if kwargs['verbose']:
                if (epoch + 1) % configs['display_step'] == 0:
                    print "Epoch: %03d     loss=%.9f" % ((epoch + 1), avg_loss)

            training_summary, training_acc = sess.run([merged_A, accuracy], feed_dict={
                                                      x: data.train.images, y_true: data.train.labels, keep_prob: 1.})
            # val_summary, val_acc = sess.run([merged_A, accuracy], feed_dict={
            # x: data.validation.images, y_true: data.validation.labels,
            # keep_prob: 1.})
            test_summary, test_acc = sess.run([merged_A, accuracy], feed_dict={
                                              x: data.test.images, y_true: data.test.labels, keep_prob: 1.})

            if kwargs['verbose']:
                print "*" * 20
                print "Training accuracy at the end of epoch %i: %f" % (epoch, training_acc)
                # print "Validation accuracy at the end of epoch %i %f" % (epoch,
                # val_acc)
                print "Test accuracy at the end of epoch %i %f" % (epoch, test_acc)
                print "*" * 20
                print "\n"

            train_summary_writer.add_summary(training_summary, epoch)
            val_summary_writer.add_summary(test_summary, epoch)

        print("Optimization Finished!")

        test_accuracy = sess.run(accuracy, feed_dict={
            x: data.test.images,
            y_true: data.test.labels,
            keep_prob: 1.
        })
        print('Model: %s' % folder_name)
        print('Test accuracy %.4f' % test_accuracy)
        print('------------------------------------------------------')

run()
