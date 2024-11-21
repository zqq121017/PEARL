import os
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from utils import load_model_dict
from models import init_model_dict
from train_test import prepare_trte_data, gen_trte_adj_mat, test_epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cal_feat_imp(data_folder, model_folder, view_list, num_class):
    num_view = len(view_list)
    
    # Dataset specific parameters
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
        dataset = 'ROSMAP'
        K = 1
        alpha = 0.5
        hidden_dims = [64]
        activation = 'relu'
        normalization = 'layernorm'
        dropout_rates = [0.7]
        residual = True
        input_normalization = False
        output_activation = None
    elif data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]
        dataset = 'BRCA'
        K = 7
        alpha = 0.1
        hidden_dims = [64]
        activation = 'relu'
        normalization = 'batchnorm'
        dropout_rates = [0.7]
        residual = True
        input_normalization = False
        output_activation = None
    else:
        raise ValueError(f"Unsupported dataset: {data_folder}")
    
    # Load data and prepare adjacency matrices
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    
    # Load feature names
    featname_list = []
    for v in view_list:
        df = pd.read_csv(os.path.join(data_folder, str(v)+"_featname.csv"), header=None)
        featname_list.append(df.values.flatten())
    
    # Initialize model
    dim_list = [x.shape[1] for x in data_tr_list]
    dim_hvcdn = pow(num_class, num_view)
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn, dataset,
                                K=K, alpha=alpha, hidden_dims=hidden_dims, activation=activation,
                                normalization=normalization, dropout_rates=dropout_rates,
                                residual=residual, input_normalization=input_normalization,
                                output_activation=output_activation)
    
    # Load trained model
    model_dict = load_model_dict(model_folder, model_dict)
    
    # Calculate base performance
    te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
    if num_class == 2:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
    else:
        f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
    
    # Calculate feature importance
    feat_imp_list = []
    for i in range(len(featname_list)):
        feat_imp = {"feat_name": featname_list[i]}
        feat_imp['imp'] = np.zeros(dim_list[i])
        
        for j in range(dim_list[i]):
            # Store original feature values
            feat_tr = data_tr_list[i][:,j].clone()
            feat_trte = data_trte_list[i][:,j].clone()
            
            # Set feature to zero
            data_tr_list[i][:,j] = 0
            data_trte_list[i][:,j] = 0
            
            # Recalculate adjacency matrices and test
            adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            
            # Calculate performance drop
            if num_class == 2:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            else:
                f1_tmp = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
            
            # Store importance and restore original values
            feat_imp['imp'][j] = (f1-f1_tmp)*dim_list[i]
            data_tr_list[i][:,j] = feat_tr.clone()
            data_trte_list[i][:,j] = feat_trte.clone()
            
        feat_imp_list.append(pd.DataFrame(data=feat_imp))
    
    return feat_imp_list

def summarize_imp_feat(featimp_list_list, topn=30):
    num_rep = len(featimp_list_list)
    num_view = len(featimp_list_list[0])
    
    # Process first repetition
    df_tmp_list = []
    for v in range(num_view):
        df_tmp = copy.deepcopy(featimp_list_list[0][v])
        df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
        df_tmp_list.append(df_tmp.copy(deep=True))
    df_featimp = pd.concat(df_tmp_list).copy(deep=True)
    
    # Process remaining repetitions
    for r in range(1, num_rep):
        for v in range(num_view):
            df_tmp = copy.deepcopy(featimp_list_list[r][v])
            df_tmp['omics'] = np.ones(df_tmp.shape[0], dtype=int)*v
            df_featimp = pd.concat([df_featimp, df_tmp.copy(deep=True)])
    
    # Aggregate and sort results
    df_featimp_top = df_featimp.groupby(['feat_name', 'omics'])['imp'].sum().reset_index()
    df_featimp_top = df_featimp_top.sort_values(by='imp', ascending=False)
    df_featimp_top = df_featimp_top.iloc[:topn]
    
    print('{:}\t{:}\t{:}\t{:}'.format('Rank', 'Feature name', 'Omics', 'Importance'))
    print('-' * 60)
    for i in range(len(df_featimp_top)):
        print('{:}\t{:}\t{:}\t{:.4f}'.format(
            i+1,
            df_featimp_top.iloc[i]['feat_name'],
            df_featimp_top.iloc[i]['omics'],
            df_featimp_top.iloc[i]['imp']
        ))
    
    return df_featimp_top