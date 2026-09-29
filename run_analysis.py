"""
Diwali Sales Analysis - Comprehensive Data Analysis & Visualization Pipeline
Loads raw data, performs cleaning, conducts in-depth EDA, generates summary statistics,
and exports cleaned data along with high-resolution visualization figures.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

DATA_RAW_PATH = 'Diwali Sales Data.csv'
DATA_CLEANED_PATH = 'Diwali_Sales_Cleaned.csv'
ASSETS_DIR = 'assets'


def setup_environment():
    """Ensure output directories exist."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print(f"[Init] Output directory '{ASSETS_DIR}' verified.")


def load_and_clean_data(file_path=DATA_RAW_PATH):
    """
    Loads raw CSV dataset, inspects shape, drops redundant null columns,
    removes duplicates, drops null amounts, and casts datatypes.
    """
    print(f"\n[Step 1] Loading raw dataset from '{file_path}'...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path, encoding='latin1')
    initial_shape = df.shape
    print(f"         Raw data loaded. Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # Drop blank columns
    drop_cols = ['Status', 'unnamed1']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Remove duplicates
    duplicates_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # Drop missing values in target Amount & Orders
    null_amount_count = df['Amount'].isnull().sum()
    df = df.dropna(subset=['Amount', 'Orders'])

    # Cast datatypes
    df['Amount'] = df['Amount'].astype(int)
    df['Orders'] = df['Orders'].astype(int)
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(int)

    cleaned_shape = df.shape
    print(f"         Dropped {duplicates_count} duplicates and {null_amount_count} rows with null Amount.")
    print(f"         Cleaned data shape: {cleaned_shape[0]} rows, {cleaned_shape[1]} columns")

    # Export cleaned CSV
    df.to_csv(DATA_CLEANED_PATH, index=False)
    print(f"         Exported cleaned dataset to '{DATA_CLEANED_PATH}'.")

    return df


def generate_eda_visualizations(df):
    """
    Generates and saves all core exploratory data analysis charts.
    """
    print("\n[Step 2] Generating exploratory data analysis (EDA) charts...")

    # 1. Distribution of Amount and Orders
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['Amount'], bins=30, ax=axes[0], kde=True, color='#2b5c8f')
    axes[0].set_title('Distribution of Purchase Amount (INR)', fontsize=13, fontweight='bold', pad=10)
    axes[0].set_xlabel('Amount (₹)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)

    sns.histplot(df['Orders'], bins=10, ax=axes[1], kde=False, color='#e26d5c', discrete=True)
    axes[1].set_title('Distribution of Order Quantities', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Number of Orders', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    plt.tight_layout()
    dist_path = os.path.join(ASSETS_DIR, '01_distribution_amount_orders.png')
    plt.savefig(dist_path)
    plt.close()
    print(f"         Saved: {dist_path}")

    # 2. Gender Analysis: Orders & Total Spend
    gender_agg = df.groupby('Gender').agg({'Amount': 'sum', 'Orders': 'sum', 'User_ID': 'count'}).rename(columns={'User_ID': 'Buyers'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#ff9999', '#66b3ff']

    # Total Amount by Gender
    sns.barplot(x=gender_agg.index, y=gender_agg['Amount'], ax=axes[0], palette=colors, hue=gender_agg.index, legend=False)
    axes[0].set_title('Total Sales Amount by Gender', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Amount (₹)', fontsize=10)
    for bar in axes[0].patches:
        val = bar.get_height()
        axes[0].annotate(f'₹{val:,.0f}', (bar.get_x() + bar.get_width() / 2, val * 0.9),
                         ha='center', va='center', color='white', fontweight='bold')

    # Total Orders by Gender
    sns.barplot(x=gender_agg.index, y=gender_agg['Orders'], ax=axes[1], palette=colors, hue=gender_agg.index, legend=False)
    axes[1].set_title('Total Order Count by Gender', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Total Orders', fontsize=10)
    for bar in axes[1].patches:
        val = bar.get_height()
        axes[1].annotate(f'{int(val):,}', (bar.get_x() + bar.get_width() / 2, val * 0.9),
                         ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    gender_path = os.path.join(ASSETS_DIR, '02_gender_analysis.png')
    plt.savefig(gender_path)
    plt.close()
    print(f"         Saved: {gender_path}")

    # 3. Age Group & Gender Breakdown
    plt.figure(figsize=(10, 6))
    age_gender = df.groupby(['Age Group', 'Gender'])['Amount'].sum().reset_index()
    order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
    sns.barplot(data=age_gender, x='Age Group', y='Amount', hue='Gender', order=order, palette=['#ff9999', '#66b3ff'])
    plt.title('Total Sales Amount by Age Group & Gender', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Age Group', fontsize=11)
    plt.ylabel('Total Sales Amount (₹)', fontsize=11)
    plt.legend(title='Gender', frameon=True)
    plt.tight_layout()
    age_path = os.path.join(ASSETS_DIR, '03_age_group_analysis.png')
    plt.savefig(age_path)
    plt.close()
    print(f"         Saved: {age_path}")

    # 4. Top 10 States by Sales Amount
    top_states = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()
    plt.figure(figsize=(11, 5))
    sns.barplot(data=top_states, x='Amount', y='State', palette='mako', hue='State', legend=False)
    plt.title('Top 10 States by Diwali Sales Revenue', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Total Revenue (₹)', fontsize=11)
    plt.ylabel('State', fontsize=11)
    plt.tight_layout()
    state_path = os.path.join(ASSETS_DIR, '04_top_states_sales.png')
    plt.savefig(state_path)
    plt.close()
    print(f"         Saved: {state_path}")

    # 5. Product Category Sales & Average Order Value
    cat_sales = df.groupby('Product_Category').agg({'Amount': 'sum', 'Orders': 'sum'}).sort_values('Amount', ascending=False).reset_index()
    cat_sales['Avg_Order_Value'] = cat_sales['Amount'] / cat_sales['Orders']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=cat_sales, x='Amount', y='Product_Category', palette='viridis', hue='Product_Category', legend=False)
    plt.title('Sales Amount by Product Category', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Total Revenue (₹)', fontsize=11)
    plt.ylabel('Product Category', fontsize=11)
    plt.tight_layout()
    cat_path = os.path.join(ASSETS_DIR, '05_product_category_sales.png')
    plt.savefig(cat_path)
    plt.close()
    print(f"         Saved: {cat_path}")

    # 6. Sales by Occupation
    occ_sales = df.groupby('Occupation')['Amount'].sum().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(11, 5))
    sns.barplot(data=occ_sales, x='Amount', y='Occupation', palette='flare', hue='Occupation', legend=False)
    plt.title('Diwali Spend by Customer Occupation', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Total Amount (₹)', fontsize=11)
    plt.ylabel('Occupation', fontsize=11)
    plt.tight_layout()
    occ_path = os.path.join(ASSETS_DIR, '06_occupation_sales.png')
    plt.savefig(occ_path)
    plt.close()
    print(f"         Saved: {occ_path}")

    # 7. Zone vs Product Category Heatmap
    pivot = df.pivot_table(index='Zone', columns='Product_Category', values='Amount', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(13, 6))
    sns.heatmap(pivot / 1e5, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Revenue (Lakh ₹)'})
    plt.title('Regional Heatmap: Zone vs Product Category Revenue (Lakh ₹)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Product Category', fontsize=11)
    plt.ylabel('Zone', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    heatmap_path = os.path.join(ASSETS_DIR, '07_zone_category_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"         Saved: {heatmap_path}")

    # 8. Marital Status & Gender Comparison
    marital_df = df.groupby(['Marital_Status', 'Gender'])['Amount'].sum().reset_index()
    marital_df['Status_Label'] = marital_df['Marital_Status'].map({0: 'Single / Unmarried', 1: 'Married'})
    plt.figure(figsize=(7, 5))
    sns.barplot(data=marital_df, x='Status_Label', y='Amount', hue='Gender', palette=['#ff9999', '#66b3ff'])
    plt.title('Sales by Marital Status and Gender', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Marital Status', fontsize=11)
    plt.ylabel('Total Spend (₹)', fontsize=11)
    plt.tight_layout()
    marital_path = os.path.join(ASSETS_DIR, '08_marital_status_sales.png')
    plt.savefig(marital_path)
    plt.close()
    print(f"         Saved: {marital_path}")


def print_executive_summary(df):
    """Outputs text summary of key statistics."""
    total_revenue = df['Amount'].sum()
    total_orders = df['Orders'].sum()
    total_customers = df['User_ID'].nunique()
    avg_order_val = total_revenue / total_orders
    top_state = df.groupby('State')['Amount'].sum().idxmax()
    top_cat = df.groupby('Product_Category')['Amount'].sum().idxmax()
    top_occ = df.groupby('Occupation')['Amount'].sum().idxmax()
    female_spend_pct = (df[df['Gender'] == 'F']['Amount'].sum() / total_revenue) * 100

    print("\n" + "=" * 60)
    print("           DIWALI SALES EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"* Total Revenue Generated : INR {total_revenue:,.2f}")
    print(f"* Total Orders Placed     : {total_orders:,}")
    print(f"* Unique Customers        : {total_customers:,}")
    print(f"* Average Order Value     : INR {avg_order_val:,.2f}")
    print(f"* Dominant Gender Spender : Female ({female_spend_pct:.1f}% of total spend)")
    print(f"* Highest Spending State  : {top_state}")
    print(f"* Top Product Category    : {top_cat}")
    print(f"* Top Buyer Occupation    : {top_occ}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    setup_environment()
    cleaned_df = load_and_clean_data()
    generate_eda_visualizations(cleaned_df)
    print_executive_summary(cleaned_df)
    print("[Success] Data Analysis and EDA figures generated successfully!")
