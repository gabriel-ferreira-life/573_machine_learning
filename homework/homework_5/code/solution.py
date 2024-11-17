import argparse
import torch.nn as nn
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='drop rate')
    args, _ = parser.parse_known_args()
    return args


class LeNet(nn.Module):
    def __init__(self, args):
        super(LeNet, self).__init__()
        ### YOUR CODE HERE

        # Convolution Layers
        self.C1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.MaxP = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.BN_C1 = nn.BatchNorm2d(6)
        self.BN_C2 = nn.BatchNorm2d(16)

        # Fully Connected Layers
        self.FC1 = nn.Linear(16*5*5, 120)
        self.FC2 = nn.Linear(120, 84)
        self.FC3 = nn.Linear(84, 10)
        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x = F.relu(self.C1(x)) # convolution + ReLU
        x = self.MaxP(x) # max pooling
        x = self.BN_C1(x) # batch normalization
        
        x = F.relu(self.C2(x)) # convolution + ReLU
        x = self.MaxP(x) # max pooling
        x = self.BN_C2(x) # batch normalization


        x = x.view(-1, 16*5*5) # flatten the tensor

        x = F.relu(self.FC1(x)) # fully connected layer + ReLU
        x = F.relu(self.FC2(x)) # fully connected layer + ReLU
        x = self.FC3(x) # output layer

        return x
        ### END YOUR CODE