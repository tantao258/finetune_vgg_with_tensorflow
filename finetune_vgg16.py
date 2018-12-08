import os
import time
import tensorflow as tf
from vgg16 import VGG16
from utils import ImageDataGenerator
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

"""
Configuration Part.
"""
# Parameters
tf.app.flags.DEFINE_string("train_file", '/path/to/train.txt', "the path of train data")
tf.app.flags.DEFINE_string("val_file", '/path/to/val.txt', "the path of val data")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learn_rate(default:0.01)")
tf.app.flags.DEFINE_integer("num_epochs", 10, "num_epoches(default:10)")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch_size(default:128)")
tf.app.flags.DEFINE_integer("num_classes", 2, "num_classes(default:2)")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout_rate(default:0.5)")
tf.app.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 3, "num_checkpoints(default:3)")
FLAGS = tf.app.flags.FLAGS
train_layers = ['fc6', 'fc7', 'fc8']

"""
Main Part of the finetuning Script.
"""

# Load data on the cpu
print("Loading data...")
with tf.device('/cpu:0'):
    train_iterator = ImageDataGenerator(txt_file=FLAGS.train_file,
                                        mode='training',
                                        batch_size=FLAGS.batch_size,
                                        num_classes=FLAGS.num_classes,
                                        shuffle=True)

    val_iterator = ImageDataGenerator(txt_file=FLAGS.val_file,
                                      mode='inference',
                                      batch_size=FLAGS.batch_size,
                                      num_classes=FLAGS.num_classes,
                                      shuffle=False)

    train_next_batch = train_iterator.iterator.get_next()
    val_next_batch = val_iterator.iterator.get_next()

# Initialize model
vgg16 = VGG16(keep_prob=FLAGS.keep_prob,
              num_classes=FLAGS.num_classes,
              train_layers=train_layers,
              learning_rate=FLAGS.learning_rate,
              model="train"
              )

with tf.Session() as sess:
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # define summary
    grad_summaries = []
    for g, v in vgg16.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    loss_summary = tf.summary.scalar("loss", vgg16.loss)
    acc_summary = tf.summary.scalar("accuracy", vgg16.accuracy)

    # merge all the train summary
    train_summary_merged = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), graph=sess.graph)

    # merge all the dev summary
    dev_summary_merged = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "dev"), graph=sess.graph)

    # checkPoint saver
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())
    # Load the pretrained weights into the non-trainable layer
    vgg16.load_initial_weights(sess)
    print("run the tensorboard in terminal: \ntensorboard --logdir={} --port=6006".format(out_dir))

    while True:
        # train loop
        x_batch_train, y_batch_train = sess.run(train_next_batch)
        _, step, train_summaries, loss, accuracy = sess.run([vgg16.train_op, vgg16.global_step, train_summary_merged, vgg16.loss, vgg16.accuracy],
                                                            feed_dict={
                                                                       vgg16.x_input: x_batch_train,
                                                                       vgg16.y_input: y_batch_train,
                                                                       vgg16.keep_prob: FLAGS.keep_prob
                                                            })
        train_summary_writer.add_summary(train_summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

        # validation
        current_step = tf.train.global_step(sess, vgg16.global_step)

        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            x_batch_val, y_batch_val = sess.run(val_next_batch)
            step, dev_summaries, loss, accuracy = sess.run([vgg16.global_step, dev_summary_merged, vgg16.loss, vgg16.accuracy],
                                                           feed_dict={
                                                                      vgg16.x_input: x_batch_val,
                                                                      vgg16.y_input: y_batch_val,
                                                                      vgg16.keep_prob: 1
                                                           })
            dev_summary_writer.add_summary(dev_summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, acc: {:g}".format(time_str, step, loss, accuracy))

        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

        # break conditon
        if accuracy > 0.95:
            exit()