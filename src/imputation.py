from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
import miceforest as mf

def impute_mice(
    df: pd.DataFrame,
    max_iter: int = 50,
    random_state: Optional[int] = 42,
    tol: float = 1e-3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform MICE imputation using sklearn's IterativeImputer with proper handling
    of both numerical and categorical variables.
    
    Args:
        df: Input DataFrame with missing values
        max_iter: Maximum number of imputation iterations
        random_state: Random state for reproducibility
        tol: Tolerance for the stopping criterion
    
    Returns:
        Tuple of (imputed DataFrame, missing indicators DataFrame)
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Store missing indicators
    missing_indicators = pd.DataFrame(
        df_copy.isna(),
        columns=[f"{col}_missing" for col in df_copy.columns]
    )
    
    # Initialize dictionaries to store encoders and scalers
    label_encoders = {}
    original_dtypes = df_copy.dtypes.to_dict()
    
    # Process each column
    for column in df_copy.columns:
        if df_copy[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df_copy[column]):
            # Handle categorical columns
            le = LabelEncoder()
            # Get non-null values for fitting
            non_null_values = df_copy[column].dropna()
            if len(non_null_values) > 0:  # Only encode if we have non-null values
                # Fit on non-null values
                le.fit(non_null_values)
                # Transform non-null values
                df_copy.loc[df_copy[column].notna(), column] = le.transform(non_null_values)
                # Store encoder for later use
                label_encoders[column] = le
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        else:
            # Handle numeric columns: replace infinite values with NaN
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            df_copy[column].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Scale numeric data
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_copy),
        columns=df_copy.columns,
        index=df_copy.index
    )
    
    # Perform imputation
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        imputation_order='random',
        verbose=0
    )
    
    # Fit and transform
    imputed_array = imputer.fit_transform(df_scaled)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(
        scaler.inverse_transform(imputed_array),
        columns=df_copy.columns,
        index=df_copy.index
    )
    
    # Reverse label encoding for categorical variables
    for column, le in label_encoders.items():
        # Round values since we're dealing with categorical indices
        rounded_values = imputed_df[column].round()
        # Clip values to be within valid range for label encoder
        rounded_values = rounded_values.clip(0, len(le.classes_) - 1)
        # Convert back to original categories
        imputed_df[column] = le.inverse_transform(rounded_values.astype(int))
        
    # Restore original dtypes where possible
    for column, dtype in original_dtypes.items():
        try:
            if column not in label_encoders:  # Don't convert back encoded columns
                imputed_df[column] = imputed_df[column].astype(dtype)
        except Exception as e:
            print(f"Warning: Could not convert column {column} back to {dtype}: {str(e)}")
    
    return imputed_df, missing_indicators

def impute_miceforest(
    df: pd.DataFrame,
    num_imputations: int = 5,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    """
    Perform MICE imputation using miceforest package with improved categorical handling.
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Store missing indicators
    missing_indicators = pd.DataFrame(
        df_copy.isna(),
        columns=[f"{col}_missing" for col in df_copy.columns]
    )
    
    try:
        # Create variable schema
        variable_schema = {}
        for col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]) and not pd.api.types.is_categorical_dtype(df_copy[col]):
                unique_count = df_copy[col].nunique()
                if unique_count < 10:
                    variable_schema[col] = 'categorical'
                else:
                    variable_schema[col] = 'continuous'
            else:
                variable_schema[col] = 'categorical'
        
        # Pre-process data
        for col in df_copy.select_dtypes(include=[np.number]):
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
        
        # Initialize kernel
        kernel = mf.ImputationKernel(
            df_copy,
            random_state=random_state,
            save_all_iterations_data=True,
            variable_schema=variable_schema
        )
        
        # Run imputation
        print("Running MICE imputations...")
        kernel.mice(
            num_imputations,
            verbose=True
        )
        
        # Get all imputations
        imputed_dfs = []
        for i in range(num_imputations):
            completed_data = kernel.complete_data(iteration=i, dataset=0)
            imputed_dfs.append(completed_data)
        
        # Calculate mean imputation
        mean_imputed = pd.DataFrame(index=df.index, columns=df.columns)
        for col in df.columns:
            if variable_schema[col] == 'categorical':
                # Use mode for categorical variables
                mode_values = pd.concat([df[col] for df in imputed_dfs]).mode()
                mean_imputed[col] = mode_values.iloc[0] if not mode_values.empty else imputed_dfs[0][col]
            else:
                # Use mean for continuous variables
                mean_imputed[col] = pd.concat([df[col] for df in imputed_dfs]).groupby(level=0).mean()
        
        # Restore original dtypes
        for col in df.columns:
            mean_imputed[col] = mean_imputed[col].astype(df[col].dtype)
        
        print(f"Successfully completed {num_imputations} imputations")
        return mean_imputed, missing_indicators, imputed_dfs
        
    except Exception as e:
        print(f"Error during miceforest imputation: {str(e)}")
        print("Falling back to basic MICE imputation...")
        imputed_df, indicators = impute_mice(df_copy, max_iter=25, random_state=random_state)
        return imputed_df, indicators, [imputed_df]

def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    weights: str = "uniform"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform KNN imputation with proper categorical handling.
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Store missing indicators
    missing_indicators = pd.DataFrame(
        df_copy.isna(),
        columns=[f"{col}_missing" for col in df_copy.columns]
    )
    
    # Store original dtypes
    original_dtypes = df_copy.dtypes.to_dict()
    
    # Initialize label encoders dictionary
    label_encoders = {}
    
    # Process each column
    for column in df_copy.columns:
        if df_copy[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df_copy[column]):
            # Handle categorical columns
            le = LabelEncoder()
            # Get non-null values for fitting
            non_null_values = df_copy[column].dropna()
            if len(non_null_values) > 0:
                # Fit on non-null values
                le.fit(non_null_values)
                # Transform non-null values
                df_copy.loc[df_copy[column].notna(), column] = le.transform(non_null_values)
                # Store encoder for later use
                label_encoders[column] = le
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        else:
            # Handle numeric columns
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            df_copy[column].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Scale the data
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_copy),
        columns=df_copy.columns,
        index=df_copy.index
    )
    
    # Initialize and fit imputer
    imputer = KNNImputer(
        n_neighbors=min(n_neighbors, len(df_copy) - 1),
        weights=weights,
        metric='nan_euclidean'
    )
    
    # Perform imputation
    imputed_array = imputer.fit_transform(df_scaled)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(
        scaler.inverse_transform(imputed_array),
        columns=df_copy.columns,
        index=df_copy.index
    )
    
    # Reverse label encoding for categorical variables
    for column, le in label_encoders.items():
        # Round values since we're dealing with categorical indices
        rounded_values = imputed_df[column].round()
        # Clip values to be within valid range for label encoder
        rounded_values = rounded_values.clip(0, len(le.classes_) - 1)
        # Convert back to original categories
        imputed_df[column] = le.inverse_transform(rounded_values.astype(int))
    
    # Restore original dtypes where possible
    for column, dtype in original_dtypes.items():
        try:
            if column not in label_encoders:  # Don't convert back encoded columns
                imputed_df[column] = imputed_df[column].astype(dtype)
        except Exception as e:
            print(f"Warning: Could not convert column {column} back to {dtype}: {str(e)}")
    
    return imputed_df, missing_indicators