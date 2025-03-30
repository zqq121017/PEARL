import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#cuda = True if torch.cuda.is_available() else False

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)

    eps = 1e-8
    inverse_weights = len(labels) / (num_class * (count + eps))
    
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = inverse_weights[i]
    
    sample_weight = sample_weight / np.sum(sample_weight)
    
    return sample_weight

def one_hot_tensor(y, num_dim):
    device = y.device
    y_onehot = torch.zeros(y.shape[0], num_dim, device=device)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    return y_onehot

def weighted_pearson_correlation(x1, x2=None):
    # Feature Mean Weighting    
    feature_means = torch.mean(x1, dim=0)
    weight = feature_means
    # Ensure no zero weights
    epsilon = 1e-8
    weight = weight + epsilon
    weight = weight * (feature_means.shape[0] / torch.sum(weight))
    
    if x2 is None:
        x2 = x1
    
    x1_weighted = x1 * weight
    x2_weighted = x2 * weight
    
    x1_centered = x1_weighted - x1_weighted.mean(dim=1, keepdim=True)
    x2_centered = x2_weighted - x2_weighted.mean(dim=1, keepdim=True)
    
    corr = torch.mm(x1_centered, x2_centered.t()) / (
        torch.norm(x1_centered, dim=1).unsqueeze(1) * 
        torch.norm(x2_centered, dim=1).unsqueeze(0)
    )
    
    return corr.float()

def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.shape)

def cal_adj_mat_parameter(edge_per_node, data):
    sim = weighted_pearson_correlation(data)
    parameter = torch.sort(sim.reshape(-1,), descending=True).values[edge_per_node*data.shape[0]]
    return parameter.item()

def graph_from_dist_tensor(sim, parameter, self_dist=True):
    if self_dist:
        assert sim.shape[0] == sim.shape[1], "Input is not pairwise similarity matrix"
    g = (sim >= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
    return g

def gen_adj_mat_tensor(data, parameter):
    sim = weighted_pearson_correlation(data)
    sim = sim.to(device)
    g = (sim >= parameter).float()
    diag_idx = np.diag_indices(g.shape[0])
    g[diag_idx[0], diag_idx[1]] = 0
    
    adj = sim * g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0], device=device)
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    return adj

def gen_test_adj_mat_tensor(data, trte_idx, parameter):
    num_tr = len(trte_idx["tr"])
    num_te = len(trte_idx["te"])
    
    sim = weighted_pearson_correlation(data)
    sim = sim.to(device)
    
    adj = torch.zeros((num_tr + num_te, num_tr + num_te), device=device)
    
    # Train to test
    sim_tr2te = sim[:num_tr, num_tr:]
    g_tr2te = (sim_tr2te >= parameter).float()
    adj[:num_tr, num_tr:] = sim_tr2te * g_tr2te
    
    # Test to train
    sim_te2tr = sim[num_tr:, :num_tr]
    g_te2tr = (sim_te2tr >= parameter).float()
    adj[num_tr:, :num_tr] = sim_te2tr * g_te2tr
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0], device=device)
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj

def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
            
def load_model_dict(folder, model_dict):
    """Load model state dictionaries from files."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
            model_dict[module].load_state_dict(
                torch.load(
                    os.path.join(folder, module+".pth"),
                    map_location=device
                )
            )
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        
        model_dict[module].to(device)
    
    return model_dict