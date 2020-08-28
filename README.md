
## filter-2
This is the second model I developed to filter images. It is a CNN model based on tensorflow. Note that this is a binary filter, change the structure if you want to apply it to more complicated use (depend on how many types of your images).
<br>The structure of this network is shown in CNN.jpg.
<br>The performance on test set was described by roc in roc.jpg
## How to use:

### 1. Standradize all your images

<br> 1.1 Prepare all your images and put them in a folder. 
<br> 1.2 Run change_size.py to make images all in a same size so that our model can read them more easily.
<br> 1.3 Run 8to24.py to make all your images into 24 bit because tensorflow can handle 1 type of images at a time.

### 2. Preparing data
Run input_data.py to create lists and define virables you need.

### 3. training
To do the training you need a training set subset from your data, it need to be manually sorted into good/bad types. The larger the trainning set, the better the performance.
<br> Put your good/bad images under separate path, run train.py

### 4. testing
Run test_0.1.py, the result will be in the csv file.
<br>Use roc.py to see the performance.
### repeat 3 & 4
You may need to adjust the variables (e.g batch size, learning rate) to improve the performance on your data set.
