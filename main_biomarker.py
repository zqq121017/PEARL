import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    # Configuration
    data_folder = 'ROSMAP'  # or 'ROSMAP'
    model_folder = os.path.join(data_folder, 'models')
    view_list = [1, 2, 3]
    num_class = 5 if data_folder == 'BRCA' else 2
    
    print(f"Running feature importance analysis for {data_folder} dataset")
    print(f"Number of views: {len(view_list)}")
    print(f"Number of classes: {num_class}")
    print("=" * 60)
    
    # Calculate feature importance across multiple repetitions
    featimp_list_list = []
    for rep in range(5):
        print(f"\nProcessing repetition {rep+1}/5...")
        model_rep_folder = os.path.join(model_folder, str(rep+1))
        featimp_list = cal_feat_imp(data_folder, model_rep_folder, view_list, num_class)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    
    print("\nSummarizing feature importance across all repetitions:")
    print("=" * 60)
    df_top_features = summarize_imp_feat(featimp_list_list)
    
    # Save results
    output_folder = os.path.join(data_folder, 'results')
    os.makedirs(output_folder, exist_ok=True)
    df_top_features.to_csv(os.path.join(output_folder, 'top_features.csv'), index=False)
    print(f"\nResults saved to {os.path.join(output_folder, 'top_features.csv')}")