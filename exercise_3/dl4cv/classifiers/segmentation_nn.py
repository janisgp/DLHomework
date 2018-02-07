"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationNN(nn.Module):

    def __init__(self,pre_trained_model,num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        
        self.Conv1 = pre_trained_model[0]
        #self.Conv1 = nn.Conv2d(3,8,(3,3),stride=(1,1),padding=(1,1))
        #self.Relu1 = pre_trained_model[1]
        #self.MaxPool1 = pre_trained_model[2]
        
        self.Conv2 = pre_trained_model[3]
        #self.Conv2 = nn.Conv2d(8,16,(3,3),stride=(1,1),padding=(1,1))
        #self.Relu2 = pre_trained_model[4]
        #self.MaxPool2 = nn.[5]
        
        self.Conv3 = pre_trained_model[6]
        #self.Conv3 = nn.Conv2d(16,16,(3,3),stride=(1,1),padding=(1,1))
        #self.pre_trained = pre_trained_model
        
        #self.upsample = nn.Upsample(size=(240,240))
        self.upsample = nn.Upsample(size=(240,240))
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        
        #x = self.MaxPool1(x)
        #x = self.MaxPool1(x)
        x = F.max_pool2d(x,(2,2),stride=2)
        
        x = self.Conv1(x)
        
        x = F.relu(x)
        
        #x = self.MaxPool1(x)
        x = F.max_pool2d(x,(2,2),stride=2)
        x = self.Conv2(x)
        x = F.relu(x)
        #x = self.MaxPool2(x)
        x = F.max_pool2d(x,(2,2),stride=2)
        x = self.Conv3(x)
        
        x = self.upsample(x)
        #x = nn.Upsample(x,size=(240,240))
        #x = F.upsample(x,size=(240,240))
        #x = nn.UpsamplingBilinear2d(size=(240,240))
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
