# McGillPhysicsHackathon
Code for project resolution restoration for astronomic images

File description:
ImageProcess.py defines image process functions;
SRCNNModel.py defines the Super Resolution Convolutional Neural Network (SRCNN) model;
TrainDataset.py is for generating low-resolution batches and high-resolution labels for training from raw images;
Train.py is for training the model;
TestDataset.py if for generating test input images and reference images;
Test.py is for testing the model with given input images.

Warning: one should overwrite the path in TrainDataset.py, Train.py, TestDataset.py and Test.py.
