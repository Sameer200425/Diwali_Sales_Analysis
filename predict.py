"""
Diwali Sales Prediction - CLI & Inference Pipeline
Predicts expected customer purchase spend (Amount in INR) using the trained machine learning model.
Supports interactive mode, demo benchmarks, and CLI argument parsing.
"""

import os
import sys
import json
import argparse
import warnings
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

MODEL_PATH = 'diwali_sales_model.joblib'
FEATURES_PATH = 'model_features.json'


def load_artifacts():
    """Load serialized model pipeline and schema metadata."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model file '{MODEL_PATH}' not found. Please run 'train.py' first.")
        sys.exit(1)
    if not os.path.exists(FEATURES_PATH):
        print(f"[Error] Feature metadata '{FEATURES_PATH}' not found. Please run 'train.py' first.")
        sys.exit(1)

    model_pipeline = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    return model_pipeline, schema


def predict_spend(customer_data, model_pipeline):
    """
    Given a customer record dict or DataFrame:
    Keys: Age, Gender, Marital_Status, Zone, Occupation, Product_Category, Orders
    Returns predicted purchase amount in INR.
    """
    if isinstance(customer_data, dict):
        df_input = pd.DataFrame([customer_data])
    else:
        df_input = customer_data.copy()

    # Ensure correct data types
    df_input['Age'] = df_input['Age'].astype(int)
    df_input['Orders'] = df_input['Orders'].astype(int)
    df_input['Marital_Status'] = df_input['Marital_Status'].astype(int)

    prediction = model_pipeline.predict(df_input)
    return prediction


def run_demo(model_pipeline, schema):
    """Executes prediction on 5 representative consumer personas."""
    print("\n" + "=" * 72)
    print("      DIWALI SALES PREDICTION - CONSUMER PERSONA BENCHMARKS")
    print("=" * 72)

    demo_personas = [
        {
            "Persona": "Tech Professional (High-Spender Segment)",
            "Data": {
                "Age": 29,
                "Gender": "F",
                "Marital_Status": 1,
                "Zone": "Central",
                "Occupation": "IT Sector",
                "Product_Category": "Food",
                "Orders": 3
            }
        },
        {
            "Persona": "Young Healthcare Worker",
            "Data": {
                "Age": 26,
                "Gender": "F",
                "Marital_Status": 0,
                "Zone": "Southern",
                "Occupation": "Healthcare",
                "Product_Category": "Clothing & Apparel",
                "Orders": 2
            }
        },
        {
            "Persona": "Senior Banking Executive",
            "Data": {
                "Age": 48,
                "Gender": "M",
                "Marital_Status": 1,
                "Zone": "Western",
                "Occupation": "Banking",
                "Product_Category": "Electronics & Gadgets",
                "Orders": 4
            }
        },
        {
            "Persona": "Aviation Professional",
            "Data": {
                "Age": 33,
                "Gender": "F",
                "Marital_Status": 1,
                "Zone": "Northern",
                "Occupation": "Aviation",
                "Product_Category": "Footwear & Shoes",
                "Orders": 2
            }
        },
        {
            "Persona": "Government Official",
            "Data": {
                "Age": 42,
                "Gender": "M",
                "Marital_Status": 1,
                "Zone": "Eastern",
                "Occupation": "Govt",
                "Product_Category": "Furniture",
                "Orders": 1
            }
        }
    ]

    for item in demo_personas:
        p_name = item["Persona"]
        c_data = item["Data"]
        pred_val = predict_spend(c_data, model_pipeline)[0]

        print(f"\n* [Persona] {p_name}")
        print(f"  - Profile    : {c_data['Age']}yo {c_data['Gender']} | {'Married' if c_data['Marital_Status'] == 1 else 'Single'} | {c_data['Occupation']} in {c_data['Zone']} Zone")
        print(f"  - Cart       : Category = {c_data['Product_Category']} | Orders = {c_data['Orders']}")
        print(f"  - Prediction : INR {pred_val:,.2f}")

    print("\n" + "=" * 72 + "\n")


def parse_arguments():
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Predict Diwali customer purchase amount based on demographics and shopping cart features."
    )
    parser.add_argument('--demo', action='store_true', help="Run benchmark predictions on sample customer personas")
    parser.add_argument('--age', type=int, default=30, help="Customer age (e.g. 28)")
    parser.add_argument('--gender', type=str, default='F', choices=['F', 'M'], help="Gender ('F' or 'M')")
    parser.add_argument('--marital-status', type=int, default=1, choices=[0, 1], help="Marital status (0: Single, 1: Married)")
    parser.add_argument('--zone', type=str, default='Central', help="Geographic zone (Central, Southern, Western, Northern, Eastern)")
    parser.add_argument('--occupation', type=str, default='IT Sector', help="Occupation (e.g. IT Sector, Healthcare, Aviation, Banking, Govt)")
    parser.add_argument('--category', type=str, default='Food', help="Product category (e.g. Food, Clothing & Apparel, Electronics & Gadgets, Footwear & Shoes)")
    parser.add_argument('--orders', type=int, default=2, help="Number of items/orders placed")

    return parser.parse_args()


def main():
    args = parse_arguments()
    model_pipeline, schema = load_artifacts()

    if args.demo:
        run_demo(model_pipeline, schema)
        return

    # Single prediction from CLI arguments
    customer_dict = {
        'Age': args.age,
        'Gender': args.gender.upper(),
        'Marital_Status': args.marital_status,
        'Zone': args.zone,
        'Occupation': args.occupation,
        'Product_Category': args.category,
        'Orders': args.orders
    }

    print("\n" + "=" * 60)
    print("      DIWALI SALES PREDICTION - CUSTOM INFERENCE")
    print("=" * 60)
    print(f"Customer Input:")
    for k, v in customer_dict.items():
        print(f"  * {k:<18}: {v}")

    predicted_amount = predict_spend(customer_dict, model_pipeline)[0]

    print("-" * 60)
    print(f"Predicted Purchase Amount: INR {predicted_amount:,.2f}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
