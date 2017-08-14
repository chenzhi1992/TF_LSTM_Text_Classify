#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from input_helpers2 import InputHelper
# from Mybi_lstm import MyLSTM
from Mylstm import MyLSTM
from tensorflow.contrib import learn
import gzip
from random import random
import vector_helper as wv

# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 100, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("class_num", 6, "Number of class ")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("sentence_words_num", 30, "The number of words in each sentence (default: 64)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.training_files == None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

# max_document_length=30
training_files = '/data/question_matching_framework/RNN_Text_Classify-master/aaa'
inpH = InputHelper()
train_set, dev_set, sum_no_of_batches = inpH.getDataSets(training_files, 20, FLAGS.batch_size)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        Model = MyLSTM(
            sequence_length=FLAGS.sentence_words_num,
            embedding_size=FLAGS.embedding_dim,
            hidden_units=FLAGS.hidden_units,
            class_num=FLAGS.class_num,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized Model object")

    grads_and_vars = optimizer.compute_gradients(Model.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", Model.loss)
    acc_summary = tf.summary.scalar("accuracy", Model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # Write vocabulary
    # vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, y_batch):
        """
        A single training step
        """
        x_batch_1 = list(x1_batch)
        x1_batch = wv.embedding_lookup(len(x_batch_1), FLAGS.sentence_words_num, FLAGS.embedding_dim,
                                       x_batch_1, 1)


        feed_dict = {
            Model.input_x: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: FLAGS.dropout_keep_prob,
            Model.b_size: len(y_batch)
        }

        _, step, cost, accuracy = sess.run(
            [tr_op_set, global_step, Model.cost, Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print("the %i step, train cost is: %f and the train accuracy is %f" % (step, cost, accuracy))
        # print (y_batch, dist, d)


    def dev_step(x1_batch, y_batch):
        """
        A single training step
        """
        x_batch_1 = list(x1_batch)

        x1_batch = wv.embedding_lookup(len(x_batch_1), FLAGS.sentence_words_num, FLAGS.embedding_dim,
                                       x_batch_1, 1)


        feed_dict = {
            Model.input_x: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: 1.0,  # 测试中关闭dropout
            Model.b_size: len(y_batch)
        }

        step, cost, accuracy = sess.run(
            [global_step, Model.cost, Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print("the %i step, dev cost is: %f and the train accuracy is %f" % (step, cost, accuracy))
        # print (y_batch, dist, d)
        return accuracy


    # Generate batches
    batches = inpH.batch_iter(
        list(zip(train_set[0], train_set[1])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr = 0
    max_validation_acc = 0.0
    for nn in range(sum_no_of_batches * FLAGS.num_epochs):
        batch = batches.__next__()
        if len(batch) < 1:
            continue
        x1_batch,  y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                acc = dev_step(x1_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
            print('--------')
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                # writer = tf.summary.FileWriter('./graphs', sess.graph)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                     as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc,
                                                                                      checkpoint_prefix))
                # writer.close()
