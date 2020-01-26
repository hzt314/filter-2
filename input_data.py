import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================================
# -----------------get lists of pics and labels-------------------------------

train_dir = 'L:/ICL/Vase_project/pics/'

good = []
label_good = []
bad = []
label_bad = []


# step1： get all pics' path and add labels on them, save the path and labels 
# in separate list
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/good'):
        good.append(file_dir + '/good' + '/' + file)
        label_good.append(0)
    for file in os.listdir(file_dir + '/bad'):
        bad.append(file_dir + '/bad' + '/' + file)
        label_bad.append(1)

    # step2：shuffle image and label lists
    image_list = np.hstack((good,bad))
    label_list = np.hstack((label_good, label_bad))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # regain shuffle lists（img and lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # convert img and lab into list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # divide List into 2 parts, one for training，one for test val
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * 0.2))  # test samples
    n_train = n_sample - n_val  # train samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------get Batch----------------------------------------------

# step1：input former list into get_batch(), change the type，get a  input queue. 
# Because img and lab are in different lists，we use tf.train.slice_input_producer()，
# then tf.read_file() to read images from the queue
#   image_W, image_H: set up width and length of image 
#   set up batch_size：number of images in every batch
#   capacity： max in one queue
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # change the type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：decoding images, here you should use only one type of image (e.g .jpg)
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：pre-arrange，standardize images so that make the model better
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    # step4：get batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # rearrange label，number of rows = [batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch