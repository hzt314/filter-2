
## filter-2
CNN model
This ia a model for good/bad beazleys filter based on tensorflow, gitclone the project if you like.
The structure of this network is shown in CNN.jpg.
The performance on test set was described by roc in roc.jpg
## How to use:

# 1. Standradize all your images

<br> 1.1 download all your images and put them in a folder. 
<br> 1.2 Run change_size.py to make images all in a same size so that our model can read them more easily.
<br> 1.3 Run 8to24.py to make all your images into 24 bit because tensorflow can handle 1 type of images at a time.

# 2. Preparing data
Run input_data.py to create lists and define virables you need.

# 3. training
To do the training you need a training set (about 8,000 images, divided in 2 types: good/bad).
Put your good/bad images under separate path, run train.py

# 4. testing
Run test_0.1.py, the result will be in the csv file.
