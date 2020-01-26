from PIL import Image
import os
image_width = 64
image_height = 64
def fixed_size(filePath,savePath):

    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)


def changeSize():
    filePath = r'L:\ICL\Vase_project\pics\bad_origin'
    destPath = r'L:\ICL\Vase_project\pics\bad'
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1]=='g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))
    print('Done')

if __name__ == '__main__':
    changeSize()
