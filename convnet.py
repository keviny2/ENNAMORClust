import torch
import torch.nn as nn
from pyro.nn import PyroModule

class ConvNet(PyroModule):
  '''
    Simple CNN
  '''
  def __init__(self, in_ch, hidden, output_size):
    super().__init__()
    
    self.in_ch = in_ch
    self.hidden = hidden
    self.output_size = output_size
    
    self.conv_layers = nn.Sequential(
        nn.Conv1d(self.in_ch, hidden, kernel_size=3, bias=False),
#         nn.BatchNorm1d(hidden),
        nn.ReLU(),
        
        nn.Conv1d(hidden, 2*hidden, kernel_size=3, bias=False),
#         nn.BatchNorm1d(2*hidden),
        nn.ReLU(),
        
        nn.Conv1d(2*hidden, 4*hidden, kernel_size=3, bias=False),
#         nn.BatchNorm1d(4*hidden),
        nn.ReLU(),
        
        nn.Conv1d(4*hidden, 2*hidden, kernel_size=3, bias=False),
#         nn.BatchNorm1d(2*hidden),
        nn.ReLU(),
        
        nn.Conv1d(2*hidden, hidden, kernel_size=5, bias=False),
#         nn.BatchNorm1d(hidden),
        nn.ReLU(),
        
        nn.Conv1d(hidden, 1, kernel_size=5, bias=False),
#         nn.BatchNorm1d()
        nn.ReLU()
    )
    
    
    

  def forward(self, x):
    '''
    Input: (*, N, M) tensor
    
    Returns: (B, self.output_size)
    '''
    
    if len(x.shape) == 2:
        # add dimension in case incoming tensor is only 2D
        x = x[None,:,:]
    
    x = x.permute(0, 2, 1)
    
    x = self.conv_layers(x)
    x = x.reshape()
    x = nn.Linear(x.shape[-1], self.output_size)(x)
    x = torch.square(x)
    
    return x 