# Missing Data Imputation Pipeline

This repository contains a comprehensive pipeline for generating synthetic customer data, performing missing data imputation using multiple methods, and evaluating imputation performance through a interactive dashboard.

![Alt text](docs\assets\demos\ImputationMethodsComparison.gif "Imputation Metrics dashboard")


## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Imputation Methods](#imputation-methods)
  - [Evaluation Metrics](#evaluation-metrics)
- [Dashboard](#dashboard)
- [Project Structure](#project-structure)

## Overview

This project implements a complete pipeline for:
1. Generating realistic synthetic customer data with missing values.
2. Implementing multiple imputation methods.
3. Evaluating imputation performance.
4. Visualizing results through a dashboard.

## Features

- Synthetic data generation with realistic correlations and missing patterns
- Multiple imputation methods:
  - MICE (Multiple Imputation by Chained Equations) using scikit-learn
  - MICE using MiceForest (Random Forest-based)
  - K-Nearest Neighbors (KNN) imputation
- Comprehensive evaluation metrics
- Interactive visualization dashboard
- Automated setup and execution scripts

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JoshuaJose978/data_cleaning_and_imputation.git
cd missing-data-imputation
```

2. Run the setup script(I use Git BASH, btw):
```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

The setup script will:
- Create a Python virtual environment
- Install required dependencies

![Alt text](docs\assets\screenshots\setup_1.PNG "Virt Env Setup and installation")

- Generate synthetic data
- Run imputation pipeline

![Alt text](docs\assets\screenshots\setup_2.PNG "Virt Env Setup and installation")

- Start the visualization dashboard

![Alt text](docs\assets\screenshots\setup_3.PNG "Virt Env Setup and installation")


If the Shell script can't be run, you can run `python main.py` and `python -m http.server 8000` and go to `http://localhost:8000` in your browser to see the imputation dashboard.

## Technical Details

### Synthetic Data Generation

The `generate_customer_data()` function creates realistic customer data with the following features:

- Demographic features:
  - Age (18-85 years)
  - Income ($20,000-$250,000)
  - Credit Score (300-850)
  - Education Level
  - Employment Status
  - Years Employed
  - Marital Status
  - Location (US States)

Missing data patterns are introduced in three categories:
1. MCAR (Missing Completely at Random)
   - Small percentage (2%) of random missingness in credit cards and years employed
   
2. MAR (Missing at Random)
   - Income more likely missing for younger people and students
   - Credit score more likely missing for new customers

3. MNAR (Missing Not at Random)
   - High income individuals less likely to report debt ratio
   - Low credit score individuals more likely to have missing account balance

### Imputation Methods

The pipeline implements three different imputation methods:

1. **MICE (scikit-learn)**
   - Uses `IterativeImputer` with default linear regression
   - Handles both numerical and categorical variables
   - Features:
     - Robust scaling of numerical variables
     - Label encoding for categorical variables
     - Random imputation order
     - Configurable iteration count and tolerance

2. **MICE (MiceForest)**
   - Random Forest-based MICE implementation
   - Features:
     - Multiple imputations with aggregation
     - Automatic variable type detection
     - Parallel processing support
     - Better handling of non-linear relationships

3. **KNN Imputation**
   - Uses `KNNImputer` with nan-Euclidean distance
   - Features:
     - Configurable number of neighbors
     - Weighted/uniform options
     - Automatic scaling
     - Categorical variable handling

### Evaluation Metrics

The pipeline calculates several metrics to evaluate imputation quality:

1. **Overall Metrics**
   - RMSE (Root Mean Square Error)
   - MAE (Mean Absolute Error)
   - Percentage within one standard deviation

2. **Distribution Metrics** (per feature)
   - Kolmogorov-Smirnov test statistic
   - Mean difference
   - Standard deviation difference

### Dashboard

The dashboard provides interactive visualizations of:
- Original vs imputed data distributions
- Imputation performance metrics
- Missing data patterns
- Comparison of different imputation methods

To access the dashboard:
1. Run `./setup.sh`
2. Open `http://localhost:8000` in your browser

## Project Structure

```
data_cleaning_and_imputation/
├── src/
│   ├── data_generator.py     # Synthetic data generation
│   ├── imputation.py         # Imputation methods         
│   └── evaluation.py         # Metrics calculation
├── data/                     # Generated data and metrics
├── run_analysis.sh           # Setup script
├── index.html                # Imputation Dashboard html file
├── styles.css                # Imputation Dashboard css file
├── script.js                 # Imputation Dashboard js file
├── requirements.txt
└── README.md                 # This documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.