import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_layers import GraphConvLayer, MessageFunc, UpdateFunc


class ValueDecomposition(nn.Module):
    "VDN implementation as a critic"
    def __init__(self, sa_dim, n_agents, hidden_size, share_params=False):
        super(ValueDecomposition, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.share_params = share_params
        
        if share_params:
            self.linear1 = nn.Linear(sa_dim, hidden_size)
            self.linear2 = nn.Linear(hidden_size, 1)
        else:
            layers = []
            for i in range(n_agents):
                layers.append(nn.Sequential(
                    nn.Linear(sa_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 1)
                ))
                
            self.layers = nn.ModuleList(layers)

        
    def forward(self, x):
        """Forward pass through the VDN network
        
        Args:
            x: [batch_size, self.n_agents, self.sa_dim] tensor

        Returns:
            [batch_size, 1] tensor
        """
        
        if self.share_params:
            h1 = F.relu(self.linear1(x))
            h2 = F.relu(self.linear2(h1))
            return torch.sum(torch.squeeze(h2, 2), 1)
        else:
            # x: [self.n_agents, batch_size, self.sa_dim] tensor
            x = torch.transpose(x, 0, 1)
            Q = self.layers[0](x[0, :, :])
            for i in range(1, self.n_agents):
                Q += self.layers[i](x[i, :, :])
            return Q

class DeepSet(nn.Module):
    """A deep set."""
    def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        super(DeepSet, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        if use_agent_id:
            agent_id_attr_dim = 2
            self.h1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.h1 = nn.Linear(sa_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)    

    def forward(self, x):
        """Forward pass through the deep set network
        
        Args:
            x: [batch_size, self.n_agent, self.sa_dim] tensor

        Returns:
            [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)
        #print(x.size()) #[32, 6, 31] 32はbatch_size

        h1 = self.h1(x)
        #print(h1.size()) [32, 6, 185]
        h1 = F.relu(h1)
        h1_summed = torch.sum(h1, 1)
        #print(h1_summed.size()) #[32, 185]
        h2 = self.h2(h1_summed)
        h2 = F.relu(h2)

        # Compute V
        V = self.V(h2)
        #print(V.size()) [32, 1]

        return V    

class DeepSet2(nn.Module):
    """A deep set."""
    def __init__(self, sa_dim, n_agents, hidden_size, agent_id=0,
                 pool_type='avg', use_agent_id=False):
        super(DeepSet2, self).__init__()
        self.sa_dim = sa_dim
        self.n_agents = n_agents
        self.pool_type = pool_type
        if use_agent_id:
            agent_id_attr_dim = 2
            self.h1 = nn.Linear(sa_dim + agent_id_attr_dim, hidden_size)
        else:
            self.h1 = nn.Linear(sa_dim, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

        self.use_agent_id = use_agent_id

        self.agent_id = agent_id

        if use_agent_id:
            self.curr_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)
            self.other_agent_attr = nn.Parameter(
                torch.randn(agent_id_attr_dim), requires_grad=True)

            agent_att = []
            for k in range(self.n_agents):
                if k == self.agent_id:
                    agent_att.append(self.curr_agent_attr.unsqueeze(-1))
                else:
                    agent_att.append(self.other_agent_attr.unsqueeze(-1))
            agent_att = torch.cat(agent_att, -1)
            self.agent_att = agent_att.unsqueeze(0)    

    def forward(self, x):
        """Forward pass through the deep set network
        
        Args:
            x: [batch_size, self.n_agent, self.sa_dim] tensor

        Returns:
            [batch_size, self.output_dim] tensor
        """
        if self.use_agent_id:
            agent_att = torch.cat([self.agent_att] * x.shape[0], 0)
            x = torch.cat([x, agent_att], 1)
        #print(x.size())

        h1 = F.relu(self.h1(x))
        h1_summed = torch.sum(h1, 1)
        h1_summed = self.h2(h1_summed)
        h1_summed /= self.n_agents**(1/2)
        h2 = F.relu(h1_summed)
        #print(h2)
        # Compute V
        V = self.V(h2)
        #print(V)
        V /= self.n_agents
        #print(V)

        return V  