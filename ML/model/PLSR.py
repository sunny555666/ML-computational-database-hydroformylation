import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import math
import matplotlib.pyplot as plt
import os

# Get number of cores from environment variable, default to 1
n_cores = int(os.getenv('PYTHON_NUM_CORES', '1'))

# Set matplotlib font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

# File paths - modify these according to your data location
X_file_path = 'data/features.xlsx'  # Path to feature data
y_file_path = 'data/targets.xlsx'   # Path to target data

# Read feature data from Excel file
df_X = pd.read_excel(X_file_path, header=0, sheet_name=0)

# Store original column names
original_columns = df_X.columns.tolist()

# Select feature columns (modify column indices as needed)
# Example: select columns 2 to 169 (adjust based on your data structure)
feature_start_col = 2
feature_end_col = 169
X = df_X.iloc[:, feature_start_col:feature_end_col]
selected_columns = original_columns[feature_start_col:feature_end_col]

# Reset index to ensure continuity
df_X.reset_index(drop=True, inplace=True)

print("NaN check after data loading:")
print(pd.DataFrame(X).isna().sum())

# Read target variable data
df_y = pd.read_excel(y_file_path, header=0)

# Reset index for target data
df_y.reset_index(drop=True, inplace=True)

# Select target column (modify column index as needed)
target_col = 1  # Adjust based on your target column position
y = df_y.iloc[:, target_col]

print("Missing values in X:", df_X.isnull().sum().sum())
print("Missing values in y:", df_y.isnull().sum().sum())

# Ensure X and y have the same number of samples
if X.shape[0] != y.shape[0]:
    print("X and y have different numbers of samples. Please check the data.")
else:
    print("X and y have the same number of samples.")
    
    # Shuffle data while maintaining X-y correspondence
    shuffled_indices = df_X.sample(frac=1, random_state=42).index
    X_shuffled = X.iloc[shuffled_indices].reset_index(drop=True)
    y_shuffled = y.iloc[shuffled_indices].reset_index(drop=True)

# Get current timestamp for file naming
current_time = datetime.now().strftime('%Y%m%d%H%M')

# Create output directory if it doesn't exist
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

def perform_cross_validation(X, y, n_components, num_iterations=20):
    """
    Perform cross-validation and return average performance metrics.
    
    Parameters:
    X: Feature matrix
    y: Target vector
    n_components: Number of PLS components
    num_iterations: Number of cross-validation iterations
    
    Returns:
    Dictionary with average performance metrics
    """
    r2_train_scores = []
    r2_test_scores = []
    mse_train_scores = []
    mse_test_scores = []
    mae_train_scores = []
    mae_test_scores = []
    
    total_y_true_train = []
    total_y_pred_train = []
    total_y_true_test = []
    total_y_pred_test = []
    
    for seed in range(1, num_iterations + 1):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        fold_r2_train = []
        fold_r2_test = []
        fold_mse_train = []
        fold_mse_test = []
        fold_mae_train = []
        fold_mae_test = []
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Standardize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train PLS model
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = pls.predict(X_train_scaled)
            y_test_pred = pls.predict(X_test_scaled)
            
            # Calculate metrics
            fold_r2_train.append(r2_score(y_train, y_train_pred))
            fold_r2_test.append(r2_score(y_test, y_test_pred))
            fold_mse_train.append(mean_squared_error(y_train, y_train_pred))
            fold_mse_test.append(mean_squared_error(y_test, y_test_pred))
            fold_mae_train.append(mean_absolute_error(y_train, y_train_pred))
            fold_mae_test.append(mean_absolute_error(y_test, y_test_pred))
            
            # Accumulate predictions for overall metrics
            total_y_true_train.extend(y_train.tolist())
            total_y_pred_train.extend(y_train_pred.flatten().tolist())
            total_y_true_test.extend(y_test.tolist())
            total_y_pred_test.extend(y_test_pred.flatten().tolist())
        
        # Store fold averages
        r2_train_scores.append(np.mean(fold_r2_train))
        r2_test_scores.append(np.mean(fold_r2_test))
        mse_train_scores.append(np.mean(fold_mse_train))
        mse_test_scores.append(np.mean(fold_mse_test))
        mae_train_scores.append(np.mean(fold_mae_train))
        mae_test_scores.append(np.mean(fold_mae_test))
    
    # Calculate overall metrics
    overall_r2_train = r2_score(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_r2_test = r2_score(np.array(total_y_true_test), np.array(total_y_pred_test))
    overall_mse_train = mean_squared_error(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_mse_test = mean_squared_error(np.array(total_y_true_test), np.array(total_y_pred_test))
    overall_mae_train = mean_absolute_error(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_mae_test = mean_absolute_error(np.array(total_y_true_test), np.array(total_y_pred_test))
    
    return {
        'r2_train': overall_r2_train,
        'r2_test': overall_r2_test,
        'mse_train': overall_mse_train,
        'mse_test': overall_mse_test,
        'rmse_train': math.sqrt(overall_mse_train),
        'rmse_test': math.sqrt(overall_mse_test),
        'mae_train': overall_mae_train,
        'mae_test': overall_mae_test
    }


def create_performance_plots(feature_counts, r2_train, r2_test, mse_train, mse_test,
                           rmse_train, rmse_test, mae_train, mae_test,
                           n_components, timestamp, output_dir):
    """
    Create and save performance plots for different metrics.
    
    Parameters:
    feature_counts: List of feature counts at each iteration
    *_train, *_test: Training and testing performance scores
    n_components: Number of PLS components
    timestamp: Timestamp for file naming
    output_dir: Output directory for plots
    """
    
    # Plot R2 scores
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, r2_train, label='Train R²', marker='o', linewidth=2, markersize=6)
    plt.plot(feature_counts, r2_test, label='Test R²', marker='o', linewidth=2, markersize=6)
    plt.title('R² Scores by Feature Count', fontfamily='Arial', fontweight='bold', fontsize=14)
    plt.xlabel('Feature Count', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.ylabel('R² Score', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.legend(prop={'family': 'Arial', 'weight': 'bold', 'size': 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'R2_scores_{timestamp}_comp{n_components}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MSE scores
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, mse_train, label='Train MSE', marker='o', linewidth=2, markersize=6)
    plt.plot(feature_counts, mse_test, label='Test MSE', marker='o', linewidth=2, markersize=6)
    plt.title('MSE Scores by Feature Count', fontfamily='Arial', fontweight='bold', fontsize=14)
    plt.xlabel('Feature Count', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.ylabel('MSE Score', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.legend(prop={'family': 'Arial', 'weight': 'bold', 'size': 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'MSE_scores_{timestamp}_comp{n_components}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot RMSE scores
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, rmse_train, label='Train RMSE', marker='o', linewidth=2, markersize=6)
    plt.plot(feature_counts, rmse_test, label='Test RMSE', marker='o', linewidth=2, markersize=6)
    plt.title('RMSE Scores by Feature Count', fontfamily='Arial', fontweight='bold', fontsize=14)
    plt.xlabel('Feature Count', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.ylabel('RMSE Score', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.legend(prop={'family': 'Arial', 'weight': 'bold', 'size': 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'RMSE_scores_{timestamp}_comp{n_components}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MAE scores
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, mae_train, label='Train MAE', marker='o', linewidth=2, markersize=6)
    plt.plot(feature_counts, mae_test, label='Test MAE', marker='o', linewidth=2, markersize=6)
    plt.title('MAE Scores by Feature Count', fontfamily='Arial', fontweight='bold', fontsize=14)
    plt.xlabel('Feature Count', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.ylabel('MAE Score', fontfamily='Arial', fontweight='bold', fontsize=12)
    plt.legend(prop={'family': 'Arial', 'weight': 'bold', 'size': 12})
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'MAE_scores_{timestamp}_comp{n_components}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Outer loop to control PLS components
max_components = 20  # Maximum number of PLS components to test
for n_components in range(1, max_components):
    print(f"\n=== Processing PLS with {n_components} components ===")
    
    # Reload feature data for each component iteration
    X = df_X.iloc[:, feature_start_col:feature_end_col]
    
    # Initialize dictionary to store feature coefficients using original column names
    feature_coefs = {col: [] for col in original_columns[feature_start_col:feature_end_col]}
    
    # Initialize Excel data storage dictionary
    excel_data = {
        'Iteration': [],
        'Feature_Count': [],
        'R2_Train': [],
        'R2_Test': [],
        'MSE_Train': [],
        'MSE_Test': [],
        'RMSE_Train': [],
        'RMSE_Test': [],
        'MAE_Train': [],
        'MAE_Test': []
    }
    
    # Initialize lists to store results
    r2_train_scores = []
    r2_test_scores = []
    mse_train_scores = []
    mse_test_scores = []
    rmse_train_scores = []
    rmse_test_scores = []
    mae_train_scores = []
    mae_test_scores = []
    
    # Initialize cumulative variables for overall R2 and MSE calculation
    total_y_true_train = []
    total_y_pred_train = []
    total_y_true_test = []
    total_y_pred_test = []
    
    # Specify output file path
    output_file_path = os.path.join(output_dir, f'pls_results_{current_time}_5fold100_comp{n_components}.txt')
    
    # Initialize dictionary to store feature coefficient absolute values
    feature_coefs = {feature: [] for feature in X.columns}
    
    # 100 iterations of 5-fold cross-validation
    num_iterations = 100
    num_folds = 5
    
    for seed in range(1, num_iterations + 1):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        
        # Reset lists for storing results of each iteration
        fold_r2_train_scores = []
        fold_r2_test_scores = []
        fold_mse_train_scores = []
        fold_mse_test_scores = []
        fold_rmse_train_scores = []
        fold_rmse_test_scores = []
        fold_mae_train_scores = []
        fold_mae_test_scores = []
        
        # 5-fold cross-validation
        for fold_index, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Standardize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)  # Fit only on training data
            X_test_scaled = scaler.transform(X_test)  # Transform test data using training parameters
            
            # Train PLS model
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = pls.predict(X_train_scaled)
            y_test_pred = pls.predict(X_test_scaled)
            
            # Calculate R², MSE, RMSE, and MAE
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_train = math.sqrt(mse_train)
            rmse_test = math.sqrt(mse_test)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            
            # Store results
            fold_r2_train_scores.append(r2_train)
            fold_r2_test_scores.append(r2_test)
            fold_mse_train_scores.append(mse_train)
            fold_mse_test_scores.append(mse_test)
            fold_rmse_train_scores.append(rmse_train)
            fold_rmse_test_scores.append(rmse_test)
            fold_mae_train_scores.append(mae_train)
            fold_mae_test_scores.append(mae_test)
            
            # Store absolute values of feature coefficients
            coef = pls.coef_[0]  # Get coefficients for the first PLS component
            for i, feature in enumerate(X.columns):
                feature_coefs[feature].append(abs(coef[i]))
            
            # Accumulate all predicted and actual values
            total_y_true_train.extend(y_train.tolist())
            total_y_pred_train.extend(y_train_pred.flatten().tolist())
            total_y_true_test.extend(y_test.tolist())
            total_y_pred_test.extend(y_test_pred.flatten().tolist())
        
        # Store results of each 5-fold cross-validation
        r2_train_scores.append(np.mean(fold_r2_train_scores))
        r2_test_scores.append(np.mean(fold_r2_test_scores))
        mse_train_scores.append(np.mean(fold_mse_train_scores))
        mse_test_scores.append(np.mean(fold_mse_test_scores))
        rmse_train_scores.append(np.mean(fold_rmse_train_scores))
        rmse_test_scores.append(np.mean(fold_rmse_test_scores))
        mae_train_scores.append(np.mean(fold_mae_train_scores))
        mae_test_scores.append(np.mean(fold_mae_test_scores))
        
        # Print results of each 5-fold cross-validation
        if seed % 10 == 0 or seed <= 5:  # Print first 5 and every 10th iteration
            print(f"Seed {seed}:")
            print(f"Average R2 Train score: {np.mean(fold_r2_train_scores):.4f}")
            print(f"Average R2 Test score: {np.mean(fold_r2_test_scores):.4f}")
            print(f"Average MSE Train: {np.mean(fold_mse_train_scores):.4f}")
            print(f"Average MSE Test: {np.mean(fold_mse_test_scores):.4f}")
            print(f"Average RMSE Train: {np.mean(fold_rmse_train_scores):.4f}")
            print(f"Average RMSE Test: {np.mean(fold_rmse_test_scores):.4f}")
    
    # Calculate overall average R2 and MSE
    overall_r2_train = r2_score(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_mse_train = mean_squared_error(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_r2_test = r2_score(np.array(total_y_true_test), np.array(total_y_pred_test))
    overall_mse_test = mean_squared_error(np.array(total_y_true_test), np.array(total_y_pred_test))
    
    # Calculate overall RMSE and MAE
    overall_rmse_train = math.sqrt(overall_mse_train)
    overall_rmse_test = math.sqrt(overall_mse_test)
    overall_mae_train = mean_absolute_error(np.array(total_y_true_train), np.array(total_y_pred_train))
    overall_mae_test = mean_absolute_error(np.array(total_y_true_test), np.array(total_y_pred_test))
    
    # Print overall average results
    print(f"\nOverall Average R2 Train score: {overall_r2_train:.4f}")
    print(f"Overall Average MSE Train: {overall_mse_train:.4f}")
    print(f"Overall Average RMSE Train: {overall_rmse_train:.4f}")
    print(f"Overall Average MAE Train: {overall_mae_train:.4f}")
    print(f"Overall Average R2 Test score: {overall_r2_test:.4f}")
    print(f"Overall Average MSE Test: {overall_mse_test:.4f}")
    print(f"Overall Average RMSE Test: {overall_rmse_test:.4f}")
    print(f"Overall Average MAE Test: {overall_mae_test:.4f}")
    
    # Save all results to text file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(f"PLS Regression Results with {n_components} components\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Overall Average R2 Train score: {overall_r2_train:.4f}\n")
        file.write(f"Overall Average MSE Train: {overall_mse_train:.4f}\n")
        file.write(f"Overall Average RMSE Train: {overall_rmse_train:.4f}\n")
        file.write(f"Overall Average MAE Train: {overall_mae_train:.4f}\n")
        file.write(f"Overall Average R2 Test score: {overall_r2_test:.4f}\n")
        file.write(f"Overall Average MSE Test: {overall_mse_test:.4f}\n")
        file.write(f"Overall Average RMSE Test: {overall_rmse_test:.4f}\n")
        file.write(f"Overall Average MAE Test: {overall_mae_test:.4f}\n\n")
        
        # Write average R2, MSE, RMSE, and MAE with standard deviations
        file.write(f"Average R2 Train score: {np.mean(r2_train_scores):.4f} (+/- {np.std(r2_train_scores):.4f})\n")
        file.write(f"Average R2 Test score: {np.mean(r2_test_scores):.4f} (+/- {np.std(r2_test_scores):.4f})\n")
        file.write(f"Average MSE Train: {np.mean(mse_train_scores):.4f} (+/- {np.std(mse_train_scores):.4f})\n")
        file.write(f"Average MSE Test: {np.mean(mse_test_scores):.4f} (+/- {np.std(mse_test_scores):.4f})\n")
        file.write(f"Average RMSE Train: {np.mean(rmse_train_scores):.4f} (+/- {np.std(rmse_train_scores):.4f})\n")
        file.write(f"Average RMSE Test: {np.mean(rmse_test_scores):.4f} (+/- {np.std(rmse_test_scores):.4f})\n")
        file.write(f"Average MAE Train: {np.mean(mae_train_scores):.4f} (+/- {np.std(mae_train_scores):.4f})\n")
        file.write(f"Average MAE Test: {np.mean(mae_test_scores):.4f} (+/- {np.std(mae_test_scores):.4f})\n\n")
    
    # Calculate average coefficient absolute values for each feature
    average_coefs = {feature: np.mean(coefs) for feature, coefs in feature_coefs.items()}
    
    # Sort features by average coefficient absolute values (descending)
    sorted_average_coefs = sorted(average_coefs.items(), key=lambda item: item[1], reverse=True)
    
    # Print sorted feature weights
    print("Features sorted by average absolute coefficients from largest to smallest:")
    for i, (feature, coef) in enumerate(sorted_average_coefs[:10]):  # Show top 10
        print(f"{i+1}. {feature}: {coef:.4f}")
    
    # Write sorted feature weights to file
    with open(output_file_path, 'a', encoding='utf-8') as file:
        file.write("Features sorted by average absolute coefficients from largest to smallest:\n")
        for feature, coef in sorted_average_coefs:
            file.write(f"{feature}: {coef:.4f}\n")
    
    # Feature Selection Phase: Iterative removal of least important features
    print(f"\n=== Starting Feature Selection for {n_components} components ===")
    
    # Initialize storage for overall R2, MSE, RMSE, and MAE across iterations
    overall_r2_train_scores = []
    overall_mse_train_scores = []
    overall_rmse_train_scores = []
    overall_mae_train_scores = []
    overall_r2_test_scores = []
    overall_mse_test_scores = []
    overall_rmse_test_scores = []
    overall_mae_test_scores = []
    
    # Initialize list to record number of remaining features after each iteration
    feature_counts = []
    
    # Calculate total number of iterations for feature selection
    # Remove features more aggressively at first, then more conservatively
    min_features = max(n_components + 5, 10)  # Minimum number of features to retain
    total_iterations = min(50, X.shape[1] - min_features)  # Limit iterations
    
    for iteration in range(total_iterations):
        print(f"\nIteration {iteration + 1}/{total_iterations}")
        
        # Calculate average coefficient absolute values for each feature
        average_coefs = {feature: np.mean(coefs) for feature, coefs in feature_coefs.items()}
        
        # Sort features by average coefficient absolute values (descending)
        sorted_average_coefs = sorted(average_coefs.items(), key=lambda item: item[1], reverse=True)
        
        # Reorder features by importance
        sorted_features = [feature for feature, _ in sorted_average_coefs]
        X_sorted = X[sorted_features]
        
        # Determine number of features to remove based on iteration
        if iteration > total_iterations // 2:
            # Remove fewer features in later iterations for fine-tuning
            features_to_remove = 1
        else:
            # Remove more features in early iterations for faster convergence
            features_to_remove = min(2, max(1, X.shape[1] // 20))
        
        # Find least important features to remove
        least_important_features = [feature for feature, _ in sorted_average_coefs[-features_to_remove:]]
        
        # Check if these features exist in DataFrame and remove them
        least_important_features_to_drop = [feature for feature in least_important_features if feature in X.columns]
        
        if len(least_important_features_to_drop) > 0 and X.shape[1] > min_features:
            X = X.drop(least_important_features_to_drop, axis=1)
            
            # Update feature coefficients dictionary
            for feature in least_important_features_to_drop:
                feature_coefs.pop(feature, None)
            
            print(f"Removed {len(least_important_features_to_drop)} features. Remaining: {X.shape[1]}")
        else:
            print(f"Reached minimum feature count or no features to remove. Stopping at {X.shape[1]} features.")
            break
        
        # Store current feature count
        feature_counts.append(X.shape[1])
        
        # Perform cross-validation with remaining features
        iteration_results = perform_cross_validation(X, y, n_components, num_iterations=20)  # Fewer iterations for speed
        
        # Store results
        overall_r2_train_scores.append(iteration_results['r2_train'])
        overall_r2_test_scores.append(iteration_results['r2_test'])
        overall_mse_train_scores.append(iteration_results['mse_train'])
        overall_mse_test_scores.append(iteration_results['mse_test'])
        overall_rmse_train_scores.append(iteration_results['rmse_train'])
        overall_rmse_test_scores.append(iteration_results['rmse_test'])
        overall_mae_train_scores.append(iteration_results['mae_train'])
        overall_mae_test_scores.append(iteration_results['mae_test'])
        
        # Update Excel data
        excel_data['Iteration'].append(iteration + 1)
        excel_data['Feature_Count'].append(X.shape[1])
        excel_data['R2_Train'].append(iteration_results['r2_train'])
        excel_data['R2_Test'].append(iteration_results['r2_test'])
        excel_data['MSE_Train'].append(iteration_results['mse_train'])
        excel_data['MSE_Test'].append(iteration_results['mse_test'])
        excel_data['RMSE_Train'].append(iteration_results['rmse_train'])
        excel_data['RMSE_Test'].append(iteration_results['rmse_test'])
        excel_data['MAE_Train'].append(iteration_results['mae_train'])
        excel_data['MAE_Test'].append(iteration_results['mae_test'])
        
        # Print iteration results
        print(f"R2 Test: {iteration_results['r2_test']:.4f}, MSE Test: {iteration_results['mse_test']:.4f}")
    
    # Find optimal values and their corresponding indices
    if overall_r2_test_scores:
        max_r2_test_idx = np.argmax(overall_r2_test_scores)
        max_r2_test = overall_r2_test_scores[max_r2_test_idx]
        max_r2_train = overall_r2_train_scores[max_r2_test_idx]
        
        min_mse_test_idx = np.argmin(overall_mse_test_scores)
        min_mse_test = overall_mse_test_scores[min_mse_test_idx]
        min_mse_train = overall_mse_train_scores[min_mse_test_idx]
        
        min_rmse_test_idx = np.argmin(overall_rmse_test_scores)
        min_rmse_test = overall_rmse_test_scores[min_rmse_test_idx]
        min_rmse_train = overall_rmse_train_scores[min_rmse_test_idx]
        
        min_mae_test_idx = np.argmin(overall_mae_test_scores)
        min_mae_test = overall_mae_test_scores[min_mae_test_idx]
        min_mae_train = overall_mae_train_scores[min_mae_test_idx]
        
        print(f"\nBest Results for {n_components} components:")
        print(f"Max R2 Test: {max_r2_test:.4f} at {feature_counts[max_r2_test_idx]} features")
        print(f"Min MSE Test: {min_mse_test:.4f} at {feature_counts[min_mse_test_idx]} features")
        print(f"Min RMSE Test: {min_rmse_test:.4f} at {feature_counts[min_rmse_test_idx]} features")
        print(f"Min MAE Test: {min_mae_test:.4f} at {feature_counts[min_mae_test_idx]} features")
    
    # Save Excel data
    excel_file_path = os.path.join(output_dir, f'pls_data_{current_time}_comp{n_components}.xlsx')
    df_excel = pd.DataFrame(excel_data)
    df_excel.to_excel(excel_file_path, index=False)
    print(f"Data saved to Excel file: {excel_file_path}")
    
    # Generate plots if we have feature selection results
    if feature_counts and overall_r2_test_scores:
        create_performance_plots(feature_counts, overall_r2_train_scores, overall_r2_test_scores,
                               overall_mse_train_scores, overall_mse_test_scores,
                               overall_rmse_train_scores, overall_rmse_test_scores,
                               overall_mae_train_scores, overall_mae_test_scores,
                               n_components, current_time, output_dir)


def main():
    """
    Main function to run the PLS regression with feature selection analysis.
    """
    print("Starting PLS Regression Analysis with Feature Selection")
    print("=" * 60)
    
    # The main analysis code runs when the script is executed
    # All the processing is already included in the global scope above
    
    print("\nAnalysis completed successfully!")
    print("Check the 'results' directory for output files and plots.")


if __name__ == "__main__":
    main()