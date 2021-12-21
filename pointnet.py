import torch
import torch.nn as nn
from pyro.nn import PyroModule

class PointNet(PyroModule):
  '''
    PointNet inspired nn architecture for permutation invariance
    GitHub repo: https://github.com/fxia22/pointnet.pytorch/blob/f0c2430b0b1529e3f76fb5d6cd6ca14be763d975/pointnet/model.py#L11
    Paper: https://arxiv.org/pdf/1612.00593.pdf
  '''
  def __init__(self, in_ch, output_size, softmax=True):
    super().__init__()
    
    self.output_size = output_size
    self.in_ch = in_ch
    self.softmax = softmax
    
    self.t_net = nn.Sequential(
        nn.Conv1d(self.in_ch, 64, 1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Conv1d(64, 128, 1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        
        nn.Conv1d(128, 1024, 1),
        nn.BatchNorm1d(1024),
        nn.ReLU()
    )
    
    self.fully_connected = nn.Sequential(
        nn.Linear(1024, 512),
        # nn.BatchNorm1d(512),  # don't need batchnorm for only 1 batch
        # nn.ReLU(),
        nn.SELU(),
        
        nn.Linear(512, 256),
        # nn.BatchNorm1d(256),
        # nn.ReLU(),
        nn.SELU(),
        
        nn.Linear(256, self.output_size),
        # nn.ReLU()
        nn.SELU()
    )
    
    self.softmax = nn.Softmax(1)
    

  def forward(self, x):
    '''
    Input: (*, N, M) tensor
    
    Returns: (B, self.output_size)
    '''
    
    if len(x.shape) == 2:
        # add dimension in case incoming tensor is only 2D
        x = x[None,:,:]
    
    x = x.permute(0, 2, 1)
        
    x = self.t_net(x)
        
    x = torch.max(x, 2, keepdim=True)[0]  # maxpooling layer
    x = x.view(-1, 1024)
    
    x = self.fully_connected(x)
    
    if self.softmax:
        x = self.softmax(x) 
        
    x = torch.abs(x)
    
    return x 