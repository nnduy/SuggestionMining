#! /usr/bin/env python
import sklearn as sk
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import sys

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./suggestion.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./non_suggestion.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# print("FLAGS:", FLAGS)
# FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# print("FLAGS.eval_train:", FLAGS.eval_train)

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # print("y_test values:",y_test)
    y_test = np.argmax(y_test, axis=1)
    # print("y_test values:",y_test)
else:
    x_raw = ["its getting annoying please fix this", "settings  current live tile size   and more"]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
# checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
# print("FLAGS.checkpoint_dir=======", FLAGS.checkpoint_dir,"============")
FLAGS.checkpoint_dir = FLAGS.checkpoint_dir[:-1]
print("FLAGS.checkpoint_dir=======", FLAGS.checkpoint_dir,"============")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    # print("all_predictions: ",all_predictions)
    # print("y_test: ",y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, all_predictions).ravel()
    print("True Negative :", tn)
    print("False Positive:", fp)
    print("False Negative:", fn)
    print("True Positive :", tp)
    print("Precision     : %8.2f" % (sk.metrics.precision_score(y_test, all_predictions)))
    print("Recall        : %8.2f" % (sk.metrics.recall_score(y_test, all_predictions)))
    print("f1_score      : %8.2f" % (sk.metrics.f1_score(y_test, all_predictions)))
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy      : {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
