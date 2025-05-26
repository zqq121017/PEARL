# PEARL
**Overview**
![figure1_PEARL](https://github.com/user-attachments/assets/caa66e48-6155-4619-93fe-7816aae24d33)
PEARL (Pearson-Enhanced spectrAl gRaph convoLutional networks) is a multi-omics integration method based on deep graph learning for biomedical classification and functional important omics feature identification. PEARL leverages a simple yet effective learning architecture to achieve superior and robust performance especially in high-dimensional and low-sample-size multi-omics settings.

**Key Features**  
•	Robust Multi-omics Integration: Effectively combines multiple types of omics data (e.g., mRNA, methylation, miRNA) for improved classification performance  
•	Biomarker Discovery: Identifies functionally important omics features that contribute to prediction  
•	Superior Performance: Outperforms state-of-the-art methods in both synthetic and real datasets  
•	Sample Efficiency: Maintains robust performance even with limited sample sizes  
•	Biological Interpretability: Identifies disease-associated genes and pathways with significant enrichment in relevant biological processes

**Model Architecture**
PEARL consists of three key components:
1.	Similarity Network Construction: Uses weighted Pearson correlation to construct sample similarity networks for different omics data types
2.	Feature Refinement: Employs Simple Spectral Graph Convolutional Networks (SSGConv) to enhance feature representation
3.	Feature Integration: Unifies features using either combined pooling or concatenation methods

**Tested Environment**
PEARL has been developed and tested with:  
•	Python: 3.12.7  
•	PyTorch: 2.6.0+cu126 (with CUDA 12.6 support)  
•	PyTorch Geometric: 2.6.1  
•	NumPy: 1.26.4  
•	Pandas: 2.2.2  
•	Scikit-learn: 1.5.1  
•	SciPy: 1.13.1

**Usage**

**Cross-Validation**
PEARL implements a comprehensive cross-validation framework using StratifiedShuffleSplit:
```
from sklearn.model_selection import StratifiedShuffleSplit
from train_test import prepare_trte_data, train_epoch, test_epoch, init_model_dict, init_optim, gen_trte_adj_mat
from utils import cal_sample_weight, one_hot_tensor
import numpy as np
import torch

# Example for ROSMAP dataset
data_folder = 'ROSMAP'
view_list = [1, 2, 3]
num_class = 2

# Dataset-specific parameters
adj_parameter = 2  # Use 10 for BRCA
dim_he_list = [200, 200, 100]  # Use [400, 400, 200] for BRCA
K = 1
alpha = 0.7

# Prepare data
data_tr_list, data_trte_list, idx_dict, labels_trte = prepare_trte_data(data_folder, view_list)
original_train_data = [data[idx_dict["tr"]] for data in data_trte_list]
original_train_labels = labels_trte[idx_dict["tr"]]

# Cross-validation with 30 folds
sss = StratifiedShuffleSplit(n_splits=30, test_size=0.2, random_state=42)

# Loop through each fold
for fold, (train_index, val_index) in enumerate(sss.split(original_train_data[0], original_train_labels)):
    # Split data, setup model, train and evaluate
    # See cross-validation scripts for complete implementation
```
**Feature Importance Analysis**
```
from feat_importance import cal_feat_imp, summarize_imp_feat

# Calculate feature importance for a specific dataset
featimp_list = cal_feat_imp(
    data_folder='ROSMAP',
    model_folder='ROSMAP/models/1',
    view_list=[1, 2, 3],
    num_class=2
)

# Summarize feature importance across multiple repetitions
df_featimp_summary = summarize_imp_feat(featimp_list_list, topn=50)
```
Running the Complete Pipeline
```
# For biomarker identification
python main_biomarker.py
```
**Acknowledgments**  
•	The ROSMAP dataset was obtained from AMP-AD Knowledge Portal ([AD Knowledge Portal]([url](https://adknowledgeportal.synapse.org/)))  
•	BRCA omics data was obtained from The Cancer Genome Atlas Program (TCGA) through Broad GDAC Firehose ([Broad GDAC Firehose]([url](https://gdac.broadinstitute.org/)))
