import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.pipeline import Pipeline
import shap
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration
N_CORES = int(os.getenv('PYTHON_NUM_CORES', '4'))  # Default to 4 cores
DATA_PATH = 'data.xlsx'  # Path to your Excel file containing features and target
OUTPUT_BASE_DIR = './feature_selection_results'  # Base directory for outputs

# Define your feature list here - replace with your actual feature names
SELECTED_FEATURES = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
    # Add your actual features here...
]

def load_data(data_path=DATA_PATH, target_column_index=1):
    """
    Load data from Excel file
    
    Parameters:
    -----------
    data_path : str
        Path to the Excel file containing both features and target
    target_column_index : int
        Index of the target column (0-based)
        
    Returns:
    --------
    X : pd.DataFrame
        Features dataframe
    y : pd.Series
        Target variable
    original_columns : list
        List of original column names
    """
    # Read feature data
    df_X = pd.read_excel(data_path, header=0, sheet_name=0)
    original_columns = df_X.columns.tolist()
    df_X.reset_index(drop=True, inplace=True)
    
    # Read target data (assuming same file, different column)
    df_y = pd.read_excel(data_path, header=0, sheet_name=0)
    df_y.reset_index(drop=True, inplace=True)

    # Data validation
    missing_cols = [f for f in SELECTED_FEATURES if f not in df_X.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    X = df_X[SELECTED_FEATURES]
    y = df_y.iloc[:, target_column_index]  # Use specified target column
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of samples")
    
    # Shuffle data
    shuffled_idx = df_X.sample(frac=1, random_state=42).index
    return X.iloc[shuffled_idx].reset_index(drop=True), y.iloc[shuffled_idx].reset_index(drop=True), original_columns

def feature_selection_shap(X, y, model, n_features_to_select):
    """
    Perform feature selection using SHAP values
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    model : sklearn estimator
        Trained model
    n_features_to_select : int
        Number of features to select
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    feature_importance : pd.Series
        Feature importance scores sorted in descending order
    """
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    
    # Ensure X is DataFrame to preserve feature names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=range(X.shape[1]))
        
    shap_values = explainer(X)
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Get importance ranking
    feature_names = X.columns
    feature_importance = pd.Series(feature_importance, index=feature_names)
    sorted_features = feature_importance.sort_values(ascending=False)
    
    # Return top n features and full ranking
    return sorted_features.index[:n_features_to_select].tolist(), sorted_features

def feature_selection_importance(model, feature_names, n_features_to_select):
    """
    Feature selection based on XGBoost feature importance
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    feature_names : list
        List of feature names
    n_features_to_select : int
        Number of features to select
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=feature_names)
    sorted_features = feature_importance.sort_values(ascending=False)
    return sorted_features.index[:n_features_to_select].tolist()

def main():
    """
    Main function to run XGBoost feature selection with SHAP
    """
    start_time = time.time()
    
    # Set environment variables for parallel processing
    os.environ['OMP_NUM_THREADS'] = str(N_CORES)
    os.environ['MKL_NUM_THREADS'] = str(N_CORES)
    os.environ['OPENBLAS_NUM_THREADS'] = str(N_CORES)
    
    # Load data
    X, y, original_columns = load_data()
    feature_list = SELECTED_FEATURES.copy()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'feature_selection_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        'n_features': [],
        'train_r2': [], 'train_r2_std': [],
        'val_r2': [], 'val_r2_std': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'params': [],
        'dropped_features': [],
        'selected_features': [],
        'feature_importance': []
    }
    
    # Initial feature selection
    current_features = feature_list.copy()
    
    # Main feature elimination loop
    while len(current_features) > 0:
        X_curr = X[current_features]
        
        print(f"\n{'='*80}")
        print(f"Current number of features: {len(current_features)}")
        print(f"{'='*80}")
        
        # Bayesian optimization for hyperparameter tuning
        def xgb_cv(n_estimators, max_depth, min_child_weight, gamma, subsample,
                colsample_bytree, reg_lambda, alpha, learning_rate):
            """Objective function for Bayesian optimization"""
            model = XGBRegressor(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=alpha,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=N_CORES
            )
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            return np.mean(cross_val_score(pipeline, X_curr, y, cv=kf, scoring='r2'))
        
        # Adjust optimization iterations based on feature count
        n_init = 20
        n_iter = 50
        
        # Define parameter bounds for Bayesian optimization
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds = {
                'n_estimators': (300, 1000),
                'max_depth': (3, 7),
                'min_child_weight': (3, 10),
                'gamma': (0.1, 1),
                'subsample': (0.6, 0.9),
                'colsample_bytree': (0.6, 0.9),
                'reg_lambda': (0, 50),
                'alpha': (0.1, 5),
                'learning_rate': (0.005, 0.05)
            },
            random_state=42
        )
        
        # Run optimization
        optimizer.maximize(init_points=n_init, n_iter=n_iter)
        best_params = optimizer.max['params']
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        
        # Train final model with best parameters
        model = XGBRegressor(**best_params, n_jobs=N_CORES)
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_curr)
        X_scaled_df = pd.DataFrame(X_scaled, columns=current_features)
        
        model.fit(X_scaled_df, y)
        
        # Feature selection using SHAP values
        if len(current_features) > 1:
            selected_features_new, feature_importance = feature_selection_shap(
                X_scaled_df, y, model, len(current_features)-1)
            
            # Record dropped feature
            dropped_feature = [f for f in current_features if f not in selected_features_new][0]
            print(f"\nDropped feature: {dropped_feature}")
            
            # Save current feature importance
            results['feature_importance'].append(feature_importance.to_dict())
            
            # Update feature list
            current_features = selected_features_new
        else:
            # If only one feature remains, save its importance and prepare to end loop
            _, feature_importance = feature_selection_shap(X_scaled_df, y, model, 1)
            results['feature_importance'].append(feature_importance.to_dict())
            dropped_feature = "Final feature"
        
        # Evaluate model performance
        train_r2, val_r2 = [], []
        train_rmse, val_rmse = [], []
        train_mae, val_mae = [], []
    
        # Reduce repetitions to speed up computation
        n_repeats = 20
        
        for seed in range(n_repeats):
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            
            for train_idx, val_idx in kf.split(X_curr):
                X_train, X_val = X_curr.iloc[train_idx], X_curr.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Standardization
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = XGBRegressor(
                    **best_params,
                    n_jobs=N_CORES,
                    early_stopping_rounds=20,
                    eval_metric='rmse'
                )
                
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False
                )
                
                # Training metrics
                y_train_pred = model.predict(X_train_scaled)
                train_r2.append(r2_score(y_train, y_train_pred))
                train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
                train_mae.append(mean_absolute_error(y_train, y_train_pred))
                
                # Validation metrics
                y_val_pred = model.predict(X_val_scaled)
                val_r2.append(r2_score(y_val, y_val_pred))
                val_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
                val_mae.append(mean_absolute_error(y_val, y_val_pred))

        # Save SHAP plots when feature count is small enough
        if len(current_features) <= 20:
            # Create SHAP explainer
            explainer = shap.Explainer(model)
            # Recalculate SHAP values for visualization
            shap_values = explainer(X_scaled_df)
            
            # Save SHAP summary plot
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_scaled_df, show=False, max_display=len(current_features))
            plt.title(f'SHAP Values Summary (Features: {len(current_features)})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_summary_{len(current_features)}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save SHAP bar plot
            plt.figure(figsize=(12, 10))
            shap.plots.bar(shap_values, show=False, max_display=len(current_features))
            plt.title(f'SHAP Feature Importance (Features: {len(current_features)})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_bar_{len(current_features)}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create dependence plots for small feature sets
            if len(current_features) <= 10:
                for feature in current_features:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(feature, shap_values.values, X_scaled_df, show=False)
                    plt.title(f'SHAP Dependence Plot for {feature} (Features: {len(current_features)})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}_{len(current_features)}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # Save results
        results['n_features'].append(len(current_features))
        results['train_r2'].append(np.mean(train_r2))
        results['train_r2_std'].append(np.std(train_r2))
        results['val_r2'].append(np.mean(val_r2))
        results['val_r2_std'].append(np.std(val_r2))
        results['train_rmse'].append(np.mean(train_rmse))
        results['val_rmse'].append(np.mean(val_rmse))
        results['train_mae'].append(np.mean(train_mae))
        results['val_mae'].append(np.mean(val_mae))
        results['params'].append(str(best_params))
        results['dropped_features'].append(dropped_feature)
        results['selected_features'].append(current_features.copy())
        
        # Print current performance
        print(f"\nCurrent model performance (Features: {len(current_features)}):")
        print(f"Train R²: {np.mean(train_r2):.4f} (±{np.std(train_r2):.4f})")
        print(f"Validation R²: {np.mean(val_r2):.4f} (±{np.std(val_r2):.4f})")
        print(f"Train/Validation R² Ratio: {np.mean(train_r2)/np.mean(val_r2):.2f}")

        # Save intermediate results every 5 iterations or for small feature counts
        if len(results['n_features']) % 5 == 0 or len(current_features) <= 10:
            save_intermediate_results(results, output_dir)
        
        # Stop if only one feature remains
        if len(current_features) <= 1:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save runtime information
    with open(os.path.join(output_dir, 'runtime_info.txt'), 'w') as f:
        f.write(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Start time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total features processed: {len(SELECTED_FEATURES)}\n")
        f.write(f"Average time per feature: {total_time/len(SELECTED_FEATURES):.2f} seconds\n")
    
    save_final_results(results, output_dir)


def save_intermediate_results(results, output_dir):
    """Save intermediate results during feature selection process"""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    df = pd.DataFrame({k: v for k, v in results.items() if k != 'feature_importance'})
    df.to_csv(os.path.join(output_dir, f'intermediate_results_{current_time}.csv'), index=False)
    
    # Save feature importance to separate files
    for i, imp in enumerate(results['feature_importance']):
        n_feat = results['n_features'][i]
        pd.Series(imp).to_csv(os.path.join(output_dir, f'feature_importance_{n_feat}_{current_time}.csv'))

def save_final_results(results, output_dir):
    """Save final results and generate comprehensive reports"""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results (excluding feature importance) to CSV
    df = pd.DataFrame({k: v for k, v in results.items() if k != 'feature_importance'})
    df.to_csv(os.path.join(output_dir, f'xgb_results_{current_time}.csv'), index=False)
    
    # Reverse list order to display feature count from small to large
    reversed_indices = list(range(len(results['n_features'])-1, -1, -1))
    
    # Plot R² performance
    plt.figure(figsize=(14, 8))
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['train_r2'][i] for i in reversed_indices], 'bo-', label='Train R²')
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['val_r2'][i] for i in reversed_indices], 'ro-', label='Validation R²')
    
    # Find best validation R² point
    best_idx = np.argmax([results['val_r2'][i] for i in reversed_indices])
    actual_best_idx = reversed_indices[best_idx]
    
    plt.scatter(results['n_features'][actual_best_idx], results['val_r2'][actual_best_idx],
                color='red', s=150, 
                label=f'Best Val R²: {results["val_r2"][actual_best_idx]:.4f}\nFeatures: {results["n_features"][actual_best_idx]}')
    
    plt.xlabel('Number of Features')
    plt.ylabel('R² Score')
    plt.title('XGBoost Performance with SHAP Feature Selection')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'xgb_performance_r2_{current_time}.png'), dpi=300)
    plt.close()
    
    # Plot RMSE performance
    plt.figure(figsize=(14, 8))
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['train_rmse'][i] for i in reversed_indices], 'b--', label='Train RMSE')
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['val_rmse'][i] for i in reversed_indices], 'r--', label='Validation RMSE')
    
    # Find best validation RMSE point
    best_rmse_idx = np.argmin([results['val_rmse'][i] for i in reversed_indices])
    actual_best_rmse_idx = reversed_indices[best_rmse_idx]
    
    plt.scatter(results['n_features'][actual_best_rmse_idx], results['val_rmse'][actual_best_rmse_idx],
                color='red', s=150, 
                label=f'Best Val RMSE: {results["val_rmse"][actual_best_rmse_idx]:.4f}\nFeatures: {results["n_features"][actual_best_rmse_idx]}')
    
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('XGBoost RMSE with SHAP Feature Selection')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'xgb_performance_rmse_{current_time}.png'), dpi=300)
    plt.close()
    
    # Plot standard deviation (model stability)
    plt.figure(figsize=(14, 8))
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['train_r2_std'][i] for i in reversed_indices], 'bo-', label='Train R² Std')
    plt.plot([results['n_features'][i] for i in reversed_indices], 
             [results['val_r2_std'][i] for i in reversed_indices], 'ro-', label='Validation R² Std')
    
    plt.xlabel('Number of Features')
    plt.ylabel('R² Standard Deviation')
    plt.title('XGBoost Model Stability with SHAP Feature Selection')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'xgb_stability_{current_time}.png'), dpi=300)
    plt.close()
    
    # Create detailed report
    with open(os.path.join(output_dir, f'xgb_detailed_report_{current_time}.txt'), 'w') as f:
        f.write(f"XGBoost Feature Selection Report\n")
        f.write(f"=============================\n\n")
        
        f.write(f"Optimal Feature Count: {results['n_features'][actual_best_idx]}\n")
        f.write(f"Validation R²: {results['val_r2'][actual_best_idx]:.4f} (±{results['val_r2_std'][actual_best_idx]:.4f})\n")
        f.write(f"Training R²: {results['train_r2'][actual_best_idx]:.4f} (±{results['train_r2_std'][actual_best_idx]:.4f})\n")
        f.write(f"Train/Val R² Ratio: {results['train_r2'][actual_best_idx]/results['val_r2'][actual_best_idx]:.2f}\n")
        f.write(f"Validation RMSE: {results['val_rmse'][actual_best_idx]:.4f}\n")
        f.write(f"Validation MAE: {results['val_mae'][actual_best_idx]:.4f}\n")
        
        f.write(f"\nBest Parameters:\n{results['params'][actual_best_idx]}\n")
        
        f.write(f"\nSelected Features at Best Performance ({results['n_features'][actual_best_idx]}):\n")
        f.write("\n".join(results['selected_features'][actual_best_idx]))
        
        f.write(f"\n\nFeature Elimination Order:\n")
        for i, feature in enumerate(results['dropped_features']):
            if i < len(results['dropped_features']) - 1:  # Skip the last "Final feature" marker
                f.write(f"{i+1}. {feature}\n")
        
        # Feature count vs performance summary table
        f.write("\n\nFeature Count vs Performance Summary:\n")
        f.write(f"{'Features':<10} {'Train R²':<15} {'Val R²':<15} {'Val RMSE':<15}\n")
        f.write(f"{'-'*55}\n")
        
        for i in reversed_indices:
            f.write(f"{results['n_features'][i]:<10} "
                   f"{results['train_r2'][i]:.4f}±{results['train_r2_std'][i]:.4f} "
                   f"{results['val_r2'][i]:.4f}±{results['val_r2_std'][i]:.4f} "
                   f"{results['val_rmse'][i]:.4f}\n")
        
        # Calculate model performance elbow point and recommend feature count
        val_r2_values = [results['val_r2'][i] for i in reversed_indices]
        feature_counts = [results['n_features'][i] for i in reversed_indices]
        
        # Calculate first and second differences
        if len(val_r2_values) > 2:
            first_diff = np.diff(val_r2_values)
            second_diff = np.diff(first_diff)
            
            # Find maximum second difference position (elbow point)
            elbow_idx = np.argmax(np.abs(second_diff)) + 1
            if elbow_idx < len(feature_counts) - 1:
                elbow_features = feature_counts[elbow_idx]
                
                f.write(f"\n\nPotential Elbow Point: {elbow_features} features with Val R²: {val_r2_values[elbow_idx]:.4f}\n")
    
    # Create feature importance summary plot
    top_n = min(30, len(SELECTED_FEATURES))
    feature_importance_count = {}
    
    # Count feature importance rankings across different models
    for i, imp_dict in enumerate(results['feature_importance']):
        n_features = results['n_features'][i]
        if n_features <= top_n:
            # Calculate rankings
            series = pd.Series(imp_dict)
            sorted_features = series.sort_values(ascending=False).index.tolist()
            
            for rank, feature in enumerate(sorted_features):
                if feature not in feature_importance_count:
                    feature_importance_count[feature] = []
                feature_importance_count[feature].append(rank + 1)
    
    # Calculate average ranking for each feature
    avg_ranks = {}
    for feature, ranks in feature_importance_count.items():
        avg_ranks[feature] = np.mean(ranks)
    
    # Sort and visualize
    if avg_ranks:
        sorted_features_by_rank = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
        top_features = sorted_features_by_rank[:min(20, len(sorted_features_by_rank))]
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        avg_rank_values = [avg_ranks[f] for f in top_features]
        
        plt.barh(y_pos, avg_rank_values, align='center')
        plt.yticks(y_pos, top_features)
        plt.xlabel('Average Rank (lower is better)')
        plt.title('Top Features by Average Importance Rank')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_features_avg_rank_{current_time}.png'), dpi=300)
        plt.close()
        
        # Save ranking information
        rank_df = pd.DataFrame({
            'Feature': list(avg_ranks.keys()),
            'AvgRank': list(avg_ranks.values())
        }).sort_values('AvgRank')
        rank_df.to_csv(os.path.join(output_dir, f'feature_avg_ranks_{current_time}.csv'), index=False)
    
    # Create recommended feature subsets list
    with open(os.path.join(output_dir, f'recommended_feature_subsets_{current_time}.txt'), 'w') as f:
        f.write("Recommended Feature Subsets\n")
        f.write("==========================\n\n")
        
        # Best validation performance feature subset
        f.write(f"Best Validation Performance Subset ({results['n_features'][actual_best_idx]} features, R² = {results['val_r2'][actual_best_idx]:.4f}):\n")
        for feature in results['selected_features'][actual_best_idx]:
            f.write(f"- {feature}\n")
        
        # Most compact but effective feature subset (R² at least 95% of best)
        threshold_r2 = 0.95 * results['val_r2'][actual_best_idx]
        for i in reversed_indices:
            if results['val_r2'][i] >= threshold_r2 and results['n_features'][i] < results['n_features'][actual_best_idx]:
                most_compact_idx = i
                break
        else:
            most_compact_idx = actual_best_idx
        
        if most_compact_idx != actual_best_idx:
            f.write(f"\nMost Compact Effective Subset ({results['n_features'][most_compact_idx]} features, R² = {results['val_r2'][most_compact_idx]:.4f}):\n")
            for feature in results['selected_features'][most_compact_idx]:
                f.write(f"- {feature}\n")
        
        # Top 10/5/3/1 feature subsets
        for n in [10, 5, 3, 1]:
            for i in results['n_features']:
                if i == n:
                    idx = results['n_features'].index(i)
                    f.write(f"\nTop {n} Features (R² = {results['val_r2'][idx]:.4f}):\n")
                    for feature in results['selected_features'][idx]:
                        f.write(f"- {feature}\n")
                    break


if __name__ == "__main__":
    main()