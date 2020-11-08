#This is for producing low-resolution test images from test raw dataset which stors high-resolution images
import os
import cv2
import numpy as np
import h5py


path_test_raw_dataset = ''
path_test_inputs=''
path_test_reference=''
scale_factor = 5
test_raw_image_names = os.listdir(path_test_raw_dataset)
max_acceptable_image_size = 5

def modcrop(image, scale):
    size = image.shape
    return image[0:size[0]-size[0]%scale,0:size[1]-size[1]%scale]

for name in test_raw_image_names:
    
    if os.path.getsize(path_test_raw_dataset+'/'+name)/1024/1024>max_acceptable_image_size:
        continue
    #The order of color is BGR (blue, green, red)
    try:
        test_raw_image = cv2.imread(path_test_raw_dataset+'/'+name)
        test_raw_image = modcrop(test_raw_image,scale_factor)
    except:
        continue
    (height,width,_) = test_raw_image.shape
    test_lr_image = cv2.resize(test_raw_image,(width//scale_factor,height//scale_factor),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path_test_inputs+'/'+name, test_lr_image)
    cv2.imwrite(path_test_reference+'/'+name, test_raw_image)