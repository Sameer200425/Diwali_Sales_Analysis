"""
Diwali Sales Prediction - Machine Learning Training Pipeline
Trains multiple regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
to predict customer purchase amounts based on demographic and transaction features.
Performs strict train/test splitting, cross-validation, metric benchmarking (MAE, RMSE, R2),
feature importance extraction, and artifact serialization.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

DATA_PATH = 'Diwali_Sales_Cleaned.csv'
MODEL_PATH = 'diwali_sales_model.joblib'
METRICS_PATH = 'model_metrics.json'
FEATURES_PATH = 'model_features.json'
ASSETS_DIR = 'assets'


def load_dataset(file_path=DATA_PATH):
    """Load cleaned dataset or generate from raw if needed."""
    if not os.path.exists(file_path):
        print(f"[Warn] '{file_path}' not found. Loading and cleaning 'Diwali Sales Data.csv'...")
        from run_analysis import load_and_clean_data
        return load_and_clean_data()
    df = pd.read_csv(file_path)
    print(f"[1/6] Dataset loaded successfully. Shape: {df.shape}")
    return df


def prepare_features(df):
    """
    Selects predictive features and isolates target.
    Features: Age, Gender, Marital_Status, Zone, Occupation, Product_Category, Orders.
    Target: Amount
    """
    feature_cols = ['Age', 'Gender', 'Marital_Status', 'Zone', 'Occupation', 'Product_Category', 'Orders']
    target_col = 'Amount'

    # Filter available columns
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].copy()
    y = df[target_col].copy()

    # Define categorical vs numerical features
    categorical_cols = [c for c in ['Gender', 'Marital_Status', 'Zone', 'Occupation', 'Product_Category'] if c in X.columns]
    numerical_cols = [c for c in ['Age', 'Orders'] if c in X.columns]

    print(f"[2/6] Feature selection complete.")
    print(f"      Numerical features   : {numerical_cols}")
    print(f"      Categorical features : {categorical_cols}")
    print(f"      Total samples        : {len(X)}")

    return X, y, numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    """
    Builds a robust scikit-learn ColumnTransformer.
    StandardScaler for numerical features, OneHotEncoder for categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor


def train_and_benchmark_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Trains multiple regression candidates using Pipeline to prevent any data leakage.
    Evaluates MAE, MSE, RMSE, and R2 on holdout test set and 5-fold cross-validation.
    """
    print("[3/6] Benchmarking regression algorithms...")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"      -> Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # 5-Fold cross-validated R2 on training set
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
        cv_r2_mean = float(np.mean(cv_scores))
        cv_r2_std = float(np.std(cv_scores))

        results[name] = {
            'MAE': round(float(mae), 2),
            'RMSE': round(float(rmse), 2),
            'R2_Score': round(float(r2), 4),
            'CV_R2_Mean': round(cv_r2_mean, 4),
            'CV_R2_Std': round(cv_r2_std, 4)
        }
        fitted_pipelines[name] = pipeline

        print(f"         MAE: INR {mae:,.2f} | RMSE: INR {rmse:,.2f} | Test R2: {r2:.4f} | 5-Fold CV R2: {cv_r2_mean:.4f}")

    return results, fitted_pipelines


def save_visualizations(best_name, best_pipeline, X_test, y_test, categorical_cols, numerical_cols):
    """
    Plots feature importances and residual distribution for best model.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print(f"\n[4/6] Generating evaluation visual artifacts for best model: '{best_name}'...")

    y_pred = best_pipeline.predict(X_test)

    # 1. Residuals and Actual vs Predicted Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuals = y_test - y_pred

    # Scatter: Actual vs Predicted
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], alpha=0.3, color='#2b5c8f')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    axes[0].set_title(f'Actual vs Predicted Spend ({best_name})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Amount (INR)', fontsize=10)
    axes[0].set_ylabel('Predicted Amount (INR)', fontsize=10)
    axes[0].legend()

    # Residuals distribution
    sns.histplot(residuals, ax=axes[1], kde=True, color='#e26d5c', bins=35)
    axes[1].axvline(0, color='black', linestyle='--', lw=1.5)
    axes[1].set_title('Residuals Distribution (Errors)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual (Actual - Predicted)', fontsize=10)
    axes[1].set_ylabel('Frequency', fontsize=10)

    plt.tight_layout()
    resid_path = os.path.join(ASSETS_DIR, '09_model_residuals.png')
    plt.savefig(resid_path)
    plt.close()
    print(f"         Saved: {resid_path}")

    # 2. Feature Importance Plot (if tree based)
    # 2. Feature Importance / Coefficients Plot
    regressor = best_pipeline.named_steps['regressor']
    preprocessor = best_pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + encoded_cat_names

    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df, x='Importance', y='Feature', palette='mako')
        plt.title(f'Top 15 Feature Importances ({best_name})', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Relative Feature Importance Score', fontsize=10)
        plt.ylabel('Feature', fontsize=10)
        plt.tight_layout()
        feat_path = os.path.join(ASSETS_DIR, '10_model_feature_impact.png')
        plt.savefig(feat_path)
        plt.close()
        print(f"         Saved: {feat_path}")
    elif hasattr(regressor, 'coef_'):
        coefs = regressor.coef_
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Coefficient': coefs
        })
        # Top 8 positive and top 8 negative impact features
        feat_df['Abs_Coef'] = feat_df['Coefficient'].abs()
        top_feats = feat_df.sort_values('Abs_Coef', ascending=False).head(16).sort_values('Coefficient')

        plt.figure(figsize=(10, 6))
        colors = ['#e26d5c' if c < 0 else '#2b5c8f' for c in top_feats['Coefficient']]
        plt.barh(top_feats['Feature'], top_feats['Coefficient'], color=colors)
        plt.axvline(0, color='gray', linestyle='--', lw=1)
        plt.title(f'Top Influential Feature Coefficients ({best_name})', fontsize=12, fontweight='bold', pad=12)
        plt.xlabel('Regression Coefficient (INR Impact on Spend)', fontsize=10)
        plt.ylabel('Feature', fontsize=10)
        plt.tight_layout()
        feat_path = os.path.join(ASSETS_DIR, '10_model_feature_impact.png')
        plt.savefig(feat_path)
        plt.close()
        print(f"         Saved: {feat_path}")


def save_artifacts(best_name, best_pipeline, results, df, numerical_cols, categorical_cols):
    """
    Saves serialized pipeline, benchmark results JSON, and feature schema JSON.
    """
    print(f"\n[5/6] Serializing model and metadata artifacts...")

    # Save model pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"         Model serialized -> '{MODEL_PATH}'")

    # Save benchmark metrics
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'best_model': best_name,
            'models': results
        }, f, indent=4)
    print(f"         Benchmark metrics serialized -> '{METRICS_PATH}'")

    # Extract schema details for inference validation
    schema = {
        'numerical_features': numerical_cols,
        'categorical_features': categorical_cols,
        'categorical_options': {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols},
        'numerical_ranges': {
            'Age': {'min': int(df['Age'].min()), 'max': int(df['Age'].max())},
            'Orders': {'min': int(df['Orders'].min()), 'max': int(df['Orders'].max())}
        }
    }

    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=4)
    print(f"         Feature schema serialized -> '{FEATURES_PATH}'")


def run_training_pipeline():
    """Main orchestration routine."""
    print("=" * 65)
    print("      DIWALI SALES PREDICTION - ML TRAINING PIPELINE")
    print("=" * 65)

    df = load_dataset()
    X, y, numerical_cols, categorical_cols = prepare_features(df)

    # Featurization Ordering: Split BEFORE preprocessor fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"      Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # Train and benchmark all models
    results, fitted_pipelines = train_and_benchmark_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # Determine best model based on R2 Score
    best_name = max(results.keys(), key=lambda k: results[k]['R2_Score'])
    best_pipeline = fitted_pipelines[best_name]

    print("\n" + "-" * 65)
    print(f"  * BEST PERFORMING MODEL: {best_name}")
    print(f"    R2 Score: {results[best_name]['R2_Score']} | MAE: INR {results[best_name]['MAE']:,.2f}")
    print("-" * 65)

    save_visualizations(best_name, best_pipeline, X_test, y_test, categorical_cols, numerical_cols)
    save_artifacts(best_name, best_pipeline, results, df, numerical_cols, categorical_cols)

    print("\n[6/6] ML Training pipeline executed successfully!")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    run_training_pipeline()
