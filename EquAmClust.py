import torch
import torch.nn as nn
from pyro.nn import PyroModule

from permnet import PermutationLayer

class VarPermNet(PyroModule):
    '''
    A variant of the Permutation Layer in PermNet
    '''
    def __init__(self, in_ch, hidden, output_size, num_layers):
        super(VarPermNet, self).__init__() 
        
        self.in_ch = in_ch
        self.hidden = hidden
        self.output_size = output_size
        self.num_layers = num_layers
        
#         # add initial convolutional layers with tanh activation and batchnorm
#         self.initial_convs = nn.Sequential(
#             nn.Conv1d(self.in_ch, hidden, kernel_size=1, bias=False),
#             nn.Tanh(), 
#             nn.BatchNorm1d(hidden),
#             nn.Dropout(0.3),
            
#             nn.Conv1d(hidden, hidden*2, kernel_size=1, bias=False),
#             nn.Tanh(),
#             nn.BatchNorm1d(hidden*2),
#             nn.Dropout(0.3)
#         )
        
        # PermNet
        # create specified number of desired permutation layers
        self.perm_layers = nn.ModuleList([PermutationLayer(self.in_ch, hidden, equamclust=True)])
        for i in range(num_layers - 2):
            self.perm_layers.append(PermutationLayer(hidden, hidden, equamclust=True))
#         self.perm_layers.append(PermutationLayer(hidden*2, hidden, output_size=output_size, last=True, equamclust=True))
        self.perm_layers.append(PermutationLayer(hidden, hidden, equamclust=True))
        
        # Convolutional Layer
        # add initial convolutional layers with tanh activation and batchnorm
        self.initial_convs = nn.Sequential(
            nn.Conv1d(hidden, hidden*2, kernel_size=1, bias=False),
            nn.Tanh(), 
            nn.BatchNorm1d(hidden*2),
            nn.Dropout(0.3),
            
            nn.Conv1d(hidden*2, hidden*4, kernel_size=1, bias=False),
            nn.Tanh(), 
            nn.BatchNorm1d(hidden*4),
            nn.Dropout(0.3),

            nn.Conv1d(hidden*4, hidden*4, kernel_size=1, bias=False),
            nn.Tanh(), 
            nn.BatchNorm1d(hidden*4),
            nn.Dropout(0.3),
            
            nn.Conv1d(hidden*4, hidden*2, kernel_size=1, bias=False),
            nn.Tanh(), 
            nn.BatchNorm1d(hidden*2),
            nn.Dropout(0.3),            
            
            nn.Conv1d(hidden*2, self.output_size, kernel_size=1, bias=False)
        )
        
        
    def forward(self, x):
        '''
        Input: (*, N, M) tensor
        
        Returns: (B, self.output_size)
        '''
        
        if len(x.shape) == 2:
            # add dimension in case incoming tensor is only 2D
            x = x[None,:,:]
        
        x = x.permute(0,2,1)
        
        for i in range(self.num_layers):
            x = self.perm_layers[i](x)
        x = self.initial_convs(x)
        
        x = torch.max(x, axis=-1)[0]
        x = torch.max(x, axis=-1)[0]
        
        x = torch.abs(x)

        return x
    

class EquAmNet(PyroModule):
    '''
    Equivariant Neural Network which uses ensemble voting for inferring cluster assignments
    '''
    def __init__(self, in_ch, hidden, output_size, num_networks=3, num_layers=3):
        super(EquAmNet, self).__init__()
        
        self.in_ch = in_ch
        self.hidden = hidden
        self.output_size = output_size
        self.num_networks = num_networks
        
        # create specified number of desired PermNets to train in parallel
        self.permnets = nn.ModuleList([VarPermNet(self.in_ch, self.hidden, self.output_size, num_layers) for i in range(num_networks)])
            

    def forward(self, x):
        '''
        Input: (*, N, M) tensor
        
        Returns: (B, self.output_size)
        '''
        
        votes = torch.empty(size=(x.shape[0], self.num_networks, self.output_size))
        votes[0, 0] = self.permnets[0](x) 
        for i in range(1, self.num_networks):
            votes[0, i] = self.permnets[i](x)
        res = torch.mean(votes, dim=-2)
        
        return res