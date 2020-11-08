import cv2
import numpy as np
import h5py
import SRCNNModel
import torch
import os
import ImageProcess

path_test_inputs=''
path_trained_model = ''
path_test_reference = ''
path_test_interpolation = ''
scale_factor = 5

test_input_image_names = os.listdir(path_test_inputs)

model = SRCNNModel.SRCNN()
checkpoint = torch.load(path_trained_model)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
for name in test_input_image_names:
    try:
        test_input = cv2.imread(path_test_inputs+'/'+name)
        test_input = test_input  
        (height,width,_) = test_input.shape
    except:
        continue
    reference = cv2.imread(path_test_reference+'/'+name)
    test_input = cv2.resize(test_input,(width*scale_factor,height*scale_factor),interpolation=cv2.INTER_CUBIC)    
    cv2.imwrite(path_test_interpolation+'/'+name, test_input)
    print(name,ImageProcess.psnr(test_input,reference))
    
    test_input = (test_input.transpose(2,0,1))/255.0
    test_input = np.expand_dims(test_input,0)
    with torch.no_grad():
        test_output = model.forward(torch.from_numpy(test_input))[0]
    test_output = ((test_output.detach().numpy()).transpose(1,2,0))*255
    test_output = np.clip(test_output,0.0,255.0)
    test_output = test_output.astype(np.uint8) 
    cv2.imwrite(path_test_prediction+'/'+name, test_output)
    
    test_output = cv2.resize(test_output,(reference.shape[1],reference.shape[0]),interpolation=cv2.INTER_CUBIC)
    print(name,ImageProcess.psnr(test_output,reference))
  
    
    
    
    
    
    