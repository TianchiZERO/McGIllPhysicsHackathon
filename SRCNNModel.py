import torch.nn as nn
import torch
import numpy as np

class SRCNN(nn.Module):
    def __init__(self, num_channel=3,n1=64,n2=32,f1=9,f2=1,f3=5):
        # We usually choose n1>n2, f1<f2, f3=1 means no averaging
        # A typical setting can be f1=9,f2=1,f3=5,n1=64,n2=32
        super(SRCNN, self).__init__()
        # We can set padding=(f1//2), but it can cause border effects
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=n1, kernel_size=f1)
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=f2)
        self.conv3 = nn.Conv2d(in_channels=n2, out_channels=num_channel, kernel_size=f3)
        self.relu = nn.ReLU(inplace=True)
      
        # Initialize weight and bias
        # The filter weights of each layer are initialized by drawing randomly from a Gaussian 
        # distribution with zero mean and standard deviation 0.001 (and 0 for biases)
        self.conv1.weight = nn.Parameter(torch.from_numpy(np.random.normal(0,0.001,(n1,num_channel,f1,f1))))
        self.conv1.bias = nn.Parameter(torch.from_numpy(np.zeros(n1)))
        self.conv2.weight = nn.Parameter(torch.from_numpy(np.random.normal(0,0.001,(n2,n1,f2,f2))))
        self.conv2.bias = nn.Parameter(torch.from_numpy(np.zeros(n2)))
        self.conv3.weight = nn.Parameter(torch.from_numpy(np.random.normal(0,0.001,(num_channel,n2,f3,f3))))
        self.conv3.bias = nn.Parameter(torch.from_numpy(np.zeros(num_channel)))
  

    def forward(self, x):
        output_layer1 = self.relu(self.conv1(x))
        output_layer2 = self.relu(self.conv2(output_layer1))
        output_layer3 = self.conv3(output_layer2)
        return output_layer3
