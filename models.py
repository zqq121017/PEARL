import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SSGConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class SSGGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, K, alpha):
        super(SSGGraphConvolution, self).__init__()
        self.conv = SSGConv(in_features, out_features, K=K, alpha=alpha)

    def forward(self, x, adj):
        if adj.is_sparse:
            edge_index, edge_weight = adj._indices(), adj._values()
        else:
            edge_index = adj.nonzero().t().contiguous()
            edge_weight = adj[edge_index[0], edge_index[1]]
        return self.conv(x, edge_index, edge_weight)

class GCN_E(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, K, alpha):
        super(GCN_E, self).__init__()
        self.gc1 = SSGGraphConvolution(in_dim, hidden_dim, K=K, alpha=alpha)
        self.gc2 = SSGGraphConvolution(hidden_dim, out_dim, K=K, alpha=alpha)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return x

class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class CombinedPoolingMLP(nn.Module):
    def __init__(self, num_view, num_cls, hidden_dims=[64], activation='relu', 
                 normalization='batchnorm', dropout_rates=[0.7], residual=True, 
                 input_normalization=False, output_activation=None):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view
        self.input_normalization = input_normalization
        self.output_activation = output_activation
        
        if input_normalization:
            self.input_norm = nn.BatchNorm1d(num_cls * 4)
        
        input_dim = num_cls * 4  # Combined pooling always uses 4 features
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.get_activation(activation))
            
            if normalization == 'batchnorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layernorm':
                layers.append(nn.LayerNorm(hidden_dim))
            
            dropout_rate = dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]
            layers.append(nn.Dropout(dropout_rate))
            
            if residual and in_dim == hidden_dim:
                layers.append(ResidualConnection())
            
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_cls))
        
        if self.output_activation:
            layers.append(self.get_activation(self.output_activation))
        
        self.mlp = nn.Sequential(*layers)
    
    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, in_list):
        stacked = torch.stack(in_list)
        max_pooled = torch.max(stacked, dim=0)[0]
        min_pooled = torch.min(stacked, dim=0)[0]
        avg_pooled = torch.mean(stacked, dim=0)
        x = torch.cat([max_pooled, min_pooled, avg_pooled, max_pooled - min_pooled], dim=1)
        
        if self.input_normalization:
            x = self.input_norm(x)
        
        return self.mlp(x)

class EnhancedConcatMLPIntegration(nn.Module):
    def __init__(self, num_view, num_cls, hidden_dims=[64], activation='relu', 
                 normalization='layernorm', dropout_rates=[0.7], residual=True, 
                 input_normalization=False, output_activation=None):
        super().__init__()
        self.num_cls = num_cls
        self.num_view = num_view
        self.input_dim = num_view * num_cls
        self.input_normalization = input_normalization
        self.output_activation = output_activation
        
        if input_normalization:
            self.input_norm = nn.BatchNorm1d(self.input_dim)
        
        layers = []
        in_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self.get_activation(activation))
            
            if normalization == 'batchnorm':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif normalization == 'layernorm':
                layers.append(nn.LayerNorm(hidden_dim))
            
            dropout_rate = dropout_rates[i] if i < len(dropout_rates) else dropout_rates[-1]
            layers.append(nn.Dropout(dropout_rate))
            
            if residual and in_dim == hidden_dim:
                layers.append(ResidualConnection())
            
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_cls))
        
        if self.output_activation:
            layers.append(self.get_activation(self.output_activation))
        
        self.mlp = nn.Sequential(*layers)
    
    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, in_list):
        x = torch.cat(in_list, dim=1)
        if self.input_normalization:
            x = self.input_norm(x)
        return self.mlp(x)

class ResidualConnection(nn.Module):
    def forward(self, x):
        return x + self.branch(x)
    
    def branch(self, x):
        return x

def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, dataset, K, alpha, 
                    hidden_dims=[64], activation='relu', normalization='layernorm', 
                    dropout_rates=[0.7], residual=True, input_normalization=False, 
                    output_activation=None):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list[0], dim_he_list[-1], K=K, alpha=alpha).to(device)
        model_dict["C{:}".format(i+1)] = Classifier_1(dim_he_list[-1], num_class).to(device)
    if num_view >= 2:
        if dataset == 'BRCA':
            model_dict["C"] = CombinedPoolingMLP(
                num_view, num_class, hidden_dims=hidden_dims, activation=activation,
                normalization=normalization, dropout_rates=dropout_rates, residual=residual,
                input_normalization=input_normalization, output_activation=output_activation
            ).to(device)
        elif dataset == 'ROSMAP':
            model_dict["C"] = EnhancedConcatMLPIntegration(
                num_view, num_class, hidden_dims=hidden_dims, activation=activation,
                normalization=normalization, dropout_rates=dropout_rates, residual=residual,
                input_normalization=input_normalization, output_activation=output_activation
            ).to(device)
        elif dataset.startswith('synthetic_data_'):
            # Use the same model as BRCA for synthetic data
            model_dict["C"] = CombinedPoolingMLP(
                num_view, num_class, hidden_dims=hidden_dims, activation=activation,
                normalization=normalization, dropout_rates=dropout_rates, residual=residual,
                input_normalization=input_normalization, output_activation=output_activation
            ).to(device)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    return model_dict

def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(
                list(model_dict["E{:}".format(i+1)].parameters()) + list(model_dict["C{:}".format(i+1)].parameters()), 
                lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict