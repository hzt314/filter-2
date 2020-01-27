import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import sys
import shutil

#this code is to convert 8 bit images into 24 bit because test function can only recognize 24 bit images
path='L:/ICL/Vase_project/pics/T1/'
newpath='L:/ICL/Vase_project/pics/T2/'
def turnto24(path):
    files = os.listdir(path)
    files = np.sort(files)
    i=0
    for f in files:
        imgpath = path + f
        img=Image.open(imgpath).convert('RGB')
        dirpath = newpath 
        file_name, file_extend = os.path.splitext(f)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
        img.save(dst)

turnto24(path)
