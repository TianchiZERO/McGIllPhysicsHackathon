#This is for producing low-resolution train inputs and reference outputs from raw dataset which stors high-resolution images

import os
import cv2
import numpy as np
import h5py

path_raw_dataset = ''
path_train_dataset=''

size_input = 33
#size_output = size_input − f1 − f2 − f3 + 3
size_output = size_input-9-1-5+3
padding = abs(size_input-size_output)//2;
scale_factor = 5
stride = 14
raw_image_names = os.listdir(path_raw_dataset)
max_acceptable_image_size = 3 # unit MB

def modcrop(image, scale):
    size = image.shape
    return image[0:size[0]-size[0]%scale,0:size[1]-size[1]%scale]

     
subimage_input_list = []
subimage_output_list = []

for name in raw_image_names:
    
    if os.path.getsize(path_raw_dataset+'/'+name)/1024/1024>max_acceptable_image_size:
        continue
    #The order of color is BGR (blue, green, red)
    
    try:
        raw_image = cv2.imread(path_raw_dataset+'/'+name)
        raw_image.shape
    except:
        continue
    #raw_image = raw_image.astype('float32')
    raw_image = modcrop(raw_image,scale_factor)
    (height,width,_) = raw_image.shape
 
    lr_image = cv2.resize(raw_image,(width//scale_factor,height//scale_factor),interpolation=cv2.INTER_CUBIC)
    lr_image = cv2.resize(lr_image,(width,height),interpolation=cv2.INTER_CUBIC)
    for h in range(0,height-size_input+1,stride):
        for w in range(0,width-size_input+1,stride):
            subimage_input = raw_image[h:h+size_input,w:w+size_input,:]
            subimage_output = lr_image[h+padding:h+padding+size_output,w+padding:w+padding+size_output,:]
            subimage_input_list.append(subimage_input.transpose(2,0,1))
            subimage_output_list.append(subimage_output.transpose(2,0,1))
         
subimage_input_list = np.array(subimage_input_list)
subimage_output_list = np.array(subimage_output_list)
trainset = h5py.File(path_train_dataset, 'w')
trainset.create_dataset(name='input',data=subimage_input_list)
trainset.create_dataset(name='output',data=subimage_output_list)
trainset.close()
x = h5py.File(path_train_dataset,'r')
print(x['output'].shape)

    
    