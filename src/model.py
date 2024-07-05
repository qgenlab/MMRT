import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.layer1= nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.layer2= nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, dtype=torch.float64),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.layer3= nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, dtype=torch.float64),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        
        self.output= nn.Sequential(
            nn.Linear(hidden_dim, output_dim, dtype=torch.float64),
            nn.Sigmoid()
        )
    
    def _forward(self,layer, m):
        """
        Recursive forward function
        """
        if m.shape[1]==1:
            return layer(m.squeeze(1))
        n=layer(m[:,0,:].squeeze(1))
        p=self._forward(layer, m[:,1:,:])
        x = self.layer3(torch.concat([n,p],axis=1))
        return x
    
    def forward(self, wts, mts):
        w=self._forward(self.layer1, wts)
        m=self._forward(self.layer2, mts)
        x=self.layer3(torch.concat([w,m],axis=1))
        return self.output(x)