from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_imputation_metrics(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame, 
    missing_mask: pd.DataFrame
) -> Dict[str, float]:
    metrics = {}
    
    # Get numeric columns only
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        # Filter numeric data
        orig_numeric = original_df[numeric_cols][missing_mask[numeric_cols]]
        imp_numeric = imputed_df[numeric_cols][missing_mask[numeric_cols]]
        
        # Remove any remaining non-numeric values
        valid_mask = ~(pd.isna(orig_numeric) | pd.isna(imp_numeric))
        if valid_mask.any().any():
            orig_valid = pd.to_numeric(orig_numeric[valid_mask], errors='coerce')
            imp_valid = pd.to_numeric(imp_numeric[valid_mask], errors='coerce')
            
            # Drop any remaining NaN values
            valid_idx = ~(pd.isna(orig_valid) | pd.isna(imp_valid))
            if valid_idx.any():
                orig_final = orig_valid[valid_idx]
                imp_final = imp_valid[valid_idx]
                
                metrics['rmse'] = np.sqrt(mean_squared_error(orig_final, imp_final))
                metrics['mae'] = mean_absolute_error(orig_final, imp_final)
                metrics['within_std_pct'] = (
                    np.abs(orig_final - imp_final) <= orig_final.std()
                ).mean() * 100
    
    # Distribution metrics for numeric columns only
    for column in numeric_cols:
        if missing_mask[column].any():
            orig_col = pd.to_numeric(original_df[column], errors='coerce')
            imp_col = pd.to_numeric(imputed_df[column], errors='coerce')
            
            valid_idx = ~(pd.isna(orig_col) | pd.isna(imp_col))
            if valid_idx.any():
                from scipy import stats
                ks_stat, _ = stats.ks_2samp(
                    orig_col[valid_idx],
                    imp_col[valid_idx]
                )
                metrics[f'{column}_ks_stat'] = ks_stat
                metrics[f'{column}_mean_diff'] = abs(
                    orig_col[valid_idx].mean() - imp_col[valid_idx].mean()
                )
                metrics[f'{column}_std_diff'] = abs(
                    orig_col[valid_idx].std() - imp_col[valid_idx].std()
                )
    
    return metrics

def save_metrics_to_csv(mice_metrics, mf_metrics, knn_metrics, output_path="data/imputation_metrics.csv"):
    """
    Saves imputation metrics from different methods to a CSV file.
    
    Args:
        mice_metrics: Metrics from sklearn MICE imputation
        mf_metrics: Metrics from miceforest imputation
        knn_metrics: Metrics from KNN imputation
        output_path: Path to save the CSV file
    """
    # Create a dictionary to store metrics for each method
    all_metrics = {
        'Method': [],
        'Metric': [],
        'Value': [],
        'Feature': []
    }
    
    # Process metrics for each method
    for method_name, metrics in [
        ('MICE (sklearn)', mice_metrics),
        ('MICE (forest)', mf_metrics),
        ('KNN', knn_metrics)
    ]:
        # Add overall metrics
        for metric_name in ['rmse', 'mae', 'within_std_pct']:
            if metric_name in metrics:
                all_metrics['Method'].append(method_name)
                all_metrics['Metric'].append(metric_name)
                all_metrics['Value'].append(metrics[metric_name])
                all_metrics['Feature'].append('overall')
        
        # Add feature-specific metrics
        for key, value in metrics.items():
            if '_ks_stat' in key:
                feature = key.replace('_ks_stat', '')
                all_metrics['Method'].append(method_name)
                all_metrics['Metric'].append('ks_stat')
                all_metrics['Value'].append(value)
                all_metrics['Feature'].append(feature)
            elif '_mean_diff' in key:
                feature = key.replace('_mean_diff', '')
                all_metrics['Method'].append(method_name)
                all_metrics['Metric'].append('mean_diff')
                all_metrics['Value'].append(value)
                all_metrics['Feature'].append(feature)
            elif '_std_diff' in key:
                feature = key.replace('_std_diff', '')
                all_metrics['Method'].append(method_name)
                all_metrics['Metric'].append('std_diff')
                all_metrics['Value'].append(value)
                all_metrics['Feature'].append(feature)
    
    # Convert to DataFrame and save
    pd.DataFrame(all_metrics).to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")