import torch
import torch.nn as nn
from pyro.nn import PyroModule

# implementation from - https://github.com/arayabrain/PermutationalNetworks

class PermutationLayer(PyroModule):
    '''
    PointNet inspired nn architecture for permutation invariance
    GitHub repo: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py#L11
    Paper: https://arxiv.org/pdf/1612.00593.pdf
    '''
    def __init__(self, in_ch=784, hidden=2048, output_size=1, last=False, equamclust=False):
        super(PermutationLayer, self).__init__()
        
        self.in_ch = in_ch
        self.hidden = hidden
        self.output_size = output_size
        self.last = last
        self.equamclust = equamclust
        
        if equamclust:
            self.inner_network = nn.Sequential(
                nn.Conv2d(2*self.in_ch, self.hidden, 5, bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden),
                nn.Dropout(0.3),

                nn.Conv2d(self.hidden, self.hidden, 5, bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden),
                nn.Dropout(0.3),
                
                nn.Conv2d(self.hidden, self.hidden, 5, bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden),
                nn.Dropout(0.3),
                
                nn.Conv2d(self.hidden, self.hidden, 5, bias=False),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(self.hidden),
                nn.Dropout(0.3)
            )
            
        else:
            self.inner_network = nn.Sequential(
                nn.Conv2d(2*self.in_ch, self.hidden, 1, bias=False),
                nn.SELU(),
                nn.BatchNorm2d(self.hidden),

                nn.Conv2d(self.hidden, self.hidden, 1, bias=False),
                nn.SELU(),
                nn.BatchNorm2d(self.hidden),

                nn.Conv2d(self.hidden, self.hidden, 1, bias=False),
                nn.SELU(),
                nn.BatchNorm2d(self.hidden),            

                nn.Conv2d(self.hidden, self.hidden, 1, bias=False),
                nn.SELU(),
                nn.BatchNorm2d(self.hidden)
            )
        
        self.last_layer = nn.Conv2d(self.hidden, self.output_size, 1)
            

    def forward(self, x):
        '''
        Input: (B, N, M) tensor
        '''
        
        # do some reshaping to construct pairs
        num_objs = x.shape[2]
        x = x.unsqueeze(3)
        z1 = torch.tile(x, (1,1,1,num_objs))
        z2 = z1.permute((0,1,3,2))

        # shape: [1, 1568, 500, 500] => [BATCH_SIZE, 2*NUM_PIXELS, NUM_IMGS, NUM_IMGS]
        # the Z tensor contains the interactions between every possible pair of images in the dataset
        Z = torch.cat([z1,z2], axis=1)
        
        # execute inner network
        Y = self.inner_network(Z)
        
        # run one more layer if output layer
        if self.last:
            Y = self.last_layer(Y)
            
            # take max twice to get desired shape 
            Y = torch.max(Y, axis=-1)[0]
            Y = torch.max(Y, axis=-1)[0]
            return Y
            
        Y = torch.max(Y, axis=3)[0]  # torch.max returns a tuple 
        return Y

    
class PermNet(PyroModule):
    def __init__(self, in_ch, hidden, output_size):
        super(PermNet, self).__init__() 
        
        self.perm_layer1 = PermutationLayer(in_ch, hidden)
        self.perm_layer2 = PermutationLayer(hidden, hidden)
        self.perm_layer3 = PermutationLayer(hidden, hidden, output_size=output_size, last=True)
        
    def forward(self, x):
        '''
        Input: (*, N, M) tensor
        
        Returns: (B, self.output_size)
        '''
        
        if len(x.shape) == 2:
            # add dimension in case incoming tensor is only 2D
            x = x[None,:,:]
        
        x = x.permute(0,2,1)
        
        x = self.perm_layer1(x)
        x = self.perm_layer2(x)
        x = self.perm_layer3(x)
        x = torch.abs(x)

        return x