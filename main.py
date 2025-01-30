
from src.data_generator import generate_customer_data
from src.imputation import impute_knn, impute_miceforest, impute_mice
from src.evaluation import calculate_imputation_metrics, save_metrics_to_csv

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    original_data = generate_customer_data(n_samples=1000)
    
    # Store missing mask for evaluation
    missing_mask = original_data.isna()
    
    # Perform MICE imputation (sklearn)
    imputed_df, indicators = impute_mice(
        original_data, 
        max_iter=100,  # Increase iterations
        tol=1e-2      # Relax tolerance
    )

    
    # Perform MICE imputation (miceforest)
    mf_imputed, mf_indicators, all_mf_imputations = impute_miceforest(
        original_data,
        num_imputations=10  # Increased for better stability
    )
    print("Shape of imputed data:", mf_imputed.shape)
    
    # Perform KNN imputation
    knn_imputed, knn_indicators = impute_knn(original_data)
    
    # Calculate metrics for each method
    mice_metrics = calculate_imputation_metrics(
        original_data,
        imputed_df,
        missing_mask
    )
    print("\nMICE (sklearn) Imputation Metrics:", mice_metrics)
    
    mf_metrics = calculate_imputation_metrics(
        original_data,
        mf_imputed,
        missing_mask
    )
    print("\nMICE (miceforest) Imputation Metrics:", mf_metrics)
    
    knn_metrics = calculate_imputation_metrics(
        original_data,
        knn_imputed,
        missing_mask
    )
    print("\nKNN Imputation Metrics:", knn_metrics)

        # Save metrics to CSV
    save_metrics_to_csv(mice_metrics, mf_metrics, knn_metrics)