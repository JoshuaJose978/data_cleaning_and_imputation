from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from datetime import datetime, timedelta

def generate_customer_data(
    n_samples: int = 1000,
    random_state: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generate synthetic customer dataset with realistic features, correlations, and missing patterns.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random state for reproducibility
    
    Returns:
        DataFrame with synthetic customer data and missing values
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate correlated numerical features
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,  # Create some correlated features
        n_clusters_per_class=2,  # Reduced from 4 to meet constraints
        n_classes=2,
        random_state=random_state
    )
    
    # Convert to DataFrame with initial numerical features
    df = pd.DataFrame(X)
    
    # Generate age with realistic distribution (slightly right-skewed)
    age = np.random.normal(42, 15, n_samples)
    age = np.clip(age, 18, 85).round()
    df['age'] = age
    
    # Generate income based on age (positive correlation + variation)
    base_income = 30000 + (age - 18) * 1500  # Base income increases with age
    income_variation = np.random.normal(0, 15000, n_samples)
    income = base_income + income_variation
    income = np.clip(income, 20000, 250000).round()
    df['income'] = income
    
    # Credit score correlated with age and income
    base_credit = 600 + (age - 18) * 3 + (income - 50000) * 0.001
    credit_variation = np.random.normal(0, 30, n_samples)
    credit_score = base_credit + credit_variation
    df['credit_score'] = np.clip(credit_score, 300, 850).round()
    
    # Education level (categorical)
    education_probs = {
        'High School': 0.25,
        'Some College': 0.25,
        'Bachelor': 0.35,
        'Master': 0.12,
        'PhD': 0.03
    }
    df['education'] = np.random.choice(
        list(education_probs.keys()),
        size=n_samples,
        p=list(education_probs.values())
    )
    
    # Employment status based on age
    def get_employment_status(age):
        if age < 22:
            probs = [0.6, 0.3, 0.1, 0]  # Higher chance of part-time/student
        elif age < 65:
            probs = [0.85, 0.05, 0.05, 0.05]  # Higher chance of full-time
        else:
            probs = [0.2, 0.1, 0.1, 0.6]  # Higher chance of retired
        return np.random.choice(
            ['Full-time', 'Part-time', 'Unemployed', 'Retired'],
            p=probs
        )
    
    df['employment_status'] = df['age'].apply(get_employment_status)
    
    # Years employed (correlated with age and employment status)
    df['years_employed'] = np.where(
        df['employment_status'].isin(['Full-time', 'Part-time']),
        np.clip((df['age'] - 22) * 0.8 + np.random.normal(0, 2, n_samples), 0, 40),
        0
    ).round(1)
    
    # Account balance (correlated with income and age)
    balance_base = income * (0.1 + np.random.beta(2, 5, n_samples))
    df['account_balance'] = np.clip(balance_base, 0, income * 2).round()
    
    # Number of credit cards (correlated with income and credit score)
    card_base = (income / 50000) * (credit_score / 600) * np.random.normal(2, 0.5, n_samples)
    df['num_credit_cards'] = np.clip(card_base, 0, 8).round()
    
    # Debt ratio (correlated with income and credit score negatively)
    debt_base = 0.4 - (income / 500000) - (credit_score - 600) * 0.0005
    debt_variation = np.random.beta(2, 5, n_samples) * 0.2
    df['debt_ratio'] = np.clip(debt_base + debt_variation, 0.05, 0.8).round(3)
    
    # Marital status with age-dependent probabilities
    def get_marital_status(age):
        if age < 25:
            return np.random.choice(['Single', 'Married', 'Divorced'], p=[0.8, 0.19, 0.01])
        elif age < 35:
            return np.random.choice(['Single', 'Married', 'Divorced'], p=[0.45, 0.45, 0.1])
        else:
            return np.random.choice(['Single', 'Married', 'Divorced'], p=[0.25, 0.6, 0.15])
    
    df['marital_status'] = df['age'].apply(get_marital_status)
    
    # Location (state) with realistic population distribution
    state_populations = {
        'CA': 0.12, 'TX': 0.09, 'FL': 0.07, 'NY': 0.06, 'IL': 0.04,
        'PA': 0.04, 'OH': 0.035, 'GA': 0.033, 'NC': 0.032, 'MI': 0.03,
        'Other': 0.46
    }
    
    # Convert to numpy array and normalize to ensure sum is exactly 1
    states = list(state_populations.keys())
    probs = np.array(list(state_populations.values()))
    probs = probs / np.sum(probs)  # Normalize to ensure exactly 1
    
    df['state'] = np.random.choice(
        states,
        size=n_samples,
        p=probs
    )
    
    # Generate realistic dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*3)  # 3 years of history
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
    df['customer_since'] = np.random.choice(dates, size=n_samples)
    
    # Introduce missing patterns with realistic dependencies
    df = introduce_missing_patterns(df)
    df.to_csv("synthetic_data.csv",index=False)

    return df

def introduce_missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Introduce missing values with realistic patterns and dependencies.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with introduced missing values
    """
    df_missing = df.copy()
    
    # MCAR (Missing Completely at Random) - Very small percentage
    mask_mcar = np.random.random(df.shape[0]) < 0.02
    columns_mcar = ['num_credit_cards', 'years_employed']
    for col in columns_mcar:
        df_missing.loc[mask_mcar, col] = np.nan
    
    # MAR (Missing at Random) patterns:
    
    # 1. Income more likely missing for younger people and students
    young_student_mask = (
        (df['age'] < 25) & 
        (df['employment_status'].isin(['Part-time', 'Unemployed']))
    )
    income_missing_prob = np.where(young_student_mask, 0.3, 0.05)
    df_missing.loc[np.random.random(len(df)) < income_missing_prob, 'income'] = np.nan
    
    # 2. Credit score more likely missing for new customers
    recent_customers = (datetime.now() - pd.to_datetime(df['customer_since'])).dt.days < 30
    credit_missing_prob = np.where(recent_customers, 0.4, 0.05)
    df_missing.loc[np.random.random(len(df)) < credit_missing_prob, 'credit_score'] = np.nan
    
    # MNAR (Missing Not at Random) patterns:
    
    # 1. High income people less likely to report debt ratio
    high_income_mask = df['income'] > np.percentile(df['income'], 80)
    debt_missing_prob = np.where(high_income_mask, 0.4, 0.1)
    df_missing.loc[np.random.random(len(df)) < debt_missing_prob, 'debt_ratio'] = np.nan
    
    # 2. People with low credit scores more likely to have missing account balance
    low_credit_mask = df['credit_score'] < np.percentile(df['credit_score'], 20)
    balance_missing_prob = np.where(low_credit_mask, 0.35, 0.05)
    df_missing.loc[np.random.random(len(df)) < balance_missing_prob, 'account_balance'] = np.nan
    
    return df_missing

# Example usage
if __name__ == "__main__":
    df = generate_customer_data(n_samples=1000)
    print("\nData Shape:", df.shape)
    print("\nFeature Summary:")
    print(df.describe())
    print("\nMissing Values Summary:")
    print(df.isnull().sum())