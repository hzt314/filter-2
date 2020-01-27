from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files
import glob as gb
import cv2
import pandas as pd
import os

file_path = 'L:/ICL/Vase_project/pics/T2_good/'
result_list = []
# get an image
def get_one_image(train):
    # input：train,path of training image
    # return：image，randomly get a image from the batch
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # randomly choose image

    img = Image.open(img_dir)
    plt.imshow(img)
    plt.show()
    image = np.array(img)
    return image


# testing
def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 64, 64, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'L:/ICL/Vase_project/pics_input_data/logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                result_list.append ('good: %.6f' % prediction[:, 0])
            elif max_index == 1:
                result_list.append ('bad: %.6f' % prediction[:, 1])
                


# ------------------------------------------------------------------------
file_name =[]
if __name__ == '__main__':
    for filename in os.listdir(file_path): 
        file_name.append (filename)
        print (filename)
        img = Image.open(file_path + filename)
        imag = img.resize([64, 64])
        image = np.array(imag)
        evaluate_one_image(image)
        
    dataframe  = pd.DataFrame({'file_name': file_name,'result': result_list})
    dataframe.to_csv("test2.csv",index=False,sep=',')