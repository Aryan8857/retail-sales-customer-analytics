"""
IBM Data Analytics Internship Project
Project: Retail Sales & Customer Analytics Using Python

Dataset:
    retail_sales_data.csv
Note:
    The dataset is synthetic and was created for portfolio/internship demonstration.

Run:
    pip install -r requirements.txt
    python retail_sales_analysis.py

Outputs:
    outputs/monthly_revenue.png
    outputs/category_revenue.png
    outputs/top_products.png
    outputs/quantity_profit.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "retail_sales_data.csv"
OUTPUT_DIR = "outputs"

def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def clean_data(df):
    df = df.drop_duplicates().copy()
    numeric_cols = ["Quantity", "Unit_Price", "Discount", "Revenue", "Cost", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Category", "Product", "Revenue", "Profit"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    return df

def analyze(df):
    monthly = df.groupby("Month")[["Revenue", "Profit"]].sum()
    category = df.groupby("Category")[["Revenue", "Profit", "Quantity"]].sum().sort_values("Revenue", ascending=False)
    products = df.groupby("Product")[["Revenue", "Profit", "Quantity"]].sum().sort_values("Revenue", ascending=False)
    customers = df.groupby("Customer_ID")[["Revenue", "Profit"]].sum().sort_values("Revenue", ascending=False)

    print("\n--- BUSINESS SUMMARY ---")
    print(f"Total Revenue : {df['Revenue'].sum():,.2f}")
    print(f"Total Profit  : {df['Profit'].sum():,.2f}")
    print(f"Profit Margin : {(df['Profit'].sum()/df['Revenue'].sum())*100:.2f}%")
    print(f"Orders/Rows   : {len(df):,}")
    print(f"Customers     : {df['Customer_ID'].nunique():,}")

    print("\n--- TOP CATEGORIES ---")
    print(category.head())

    print("\n--- TOP 10 PRODUCTS ---")
    print(products.head(10))

    print("\n--- TOP 10 CUSTOMERS ---")
    print(customers.head(10))

    return monthly, category, products, customers

def create_charts(df, monthly, category, products):
    # Monthly revenue
    plt.figure(figsize=(10,5))
    plt.plot(monthly.index, monthly["Revenue"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_revenue.png", dpi=160)
    plt.close()

    # Category revenue
    plt.figure(figsize=(8,5))
    plt.bar(category.index, category["Revenue"])
    plt.xticks(rotation=25, ha="right")
    plt.title("Revenue by Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_revenue.png", dpi=160)
    plt.close()

    # Top products
    top10 = products.head(10).sort_values("Revenue")
    plt.figure(figsize=(9,5))
    plt.barh(top10.index, top10["Revenue"])
    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_products.png", dpi=160)
    plt.close()

    # Quantity vs profit
    plt.figure(figsize=(8,5))
    plt.scatter(df["Quantity"], df["Profit"], alpha=0.35)
    plt.title("Quantity vs Profit")
    plt.xlabel("Quantity")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/quantity_profit.png", dpi=160)
    plt.close()

def main():
    df = load_data()
    df = clean_data(df)
    monthly, category, products, customers = analyze(df)
    create_charts(df, monthly, category, products)
    print("\nAnalysis completed successfully. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
