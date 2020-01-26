# import files
import os
import numpy as np
import tensorflow as tf
import input_data
import model

# vairables
N_CLASSES = 2  # 2 types of pics
IMG_W = 64  # pics resizing, original sizes were too big to train in a short time
IMG_H = 64
BATCH_SIZE = 20
CAPACITY = 2000
MAX_STEP = 10000  # should be >10K
learning_rate = 0.0001  # should be <0.0001

# get batch
train_dir = 'L:/ICL/Vase_project/pics/'  # path of training set
logs_train_dir = 'L:/ICL/Vase_project/pics_input_data/logs'  # logs saving path

# train, train_label = input_data.get_files(train_dir) 
train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)
# training files and labels
train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# testing files and labels
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# define training
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.trainning(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

# define testing
test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model.losses(test_logits, val_label_batch)
test_acc = model.evaluation(test_logits, val_label_batch)

# record of log
summary_op = tf.summary.merge_all() 

sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# batch training
try:
    # MAX_STEP training, each step one batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        # print loss and acc every 50 steps，put down log into writer
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        # save model every 100 steps
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()