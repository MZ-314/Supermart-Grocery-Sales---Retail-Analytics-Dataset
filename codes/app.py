import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Page configuration
st.set_page_config(page_title="Supermarket Sales Analytics", layout="wide", page_icon="🛒")

# Loading the saved models with corrected paths
@st.cache_resource
def load_models():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the models folder (one level up from codes/)
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    
    model_path = os.path.join(models_dir, 'rf_sales_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure the model files are in the 'models/' folder.")
        st.stop()

model, scaler = load_models()

# Title and description
st.title("🛒 Supermarket Sales Analytics Dashboard")
st.markdown("### Predict sales and analyze grocery store performance")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Sales Prediction", "📈 Data Analysis"])

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Sales Overview Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your sales data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
        df['month_no'] = df['Order Date'].dt.month
        df['Month'] = df['Order Date'].dt.strftime('%B')
        df['year'] = df['Order Date'].dt.year
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        with col2:
            st.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
        with col3:
            st.metric("Average Order", f"${df['Sales'].mean():.2f}")
        with col4:
            st.metric("Total Orders", f"{len(df):,}")
        
        st.markdown("---")
        
        # Visualizations in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales by Category")
            sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            sales_by_category.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Category')
            ax.set_ylabel('Total Sales')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sales by Region")
            sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 5))
            sales_by_region.plot(kind='bar', ax=ax, color='coral')
            ax.set_xlabel('Region')
            ax.set_ylabel('Total Sales')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Sales Trend")
            sales_by_month = df.groupby('month_no')['Sales'].sum()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sales_by_month.index, sales_by_month.values, marker='o', linewidth=2, color='green')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Sales')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Yearly Sales Distribution")
            sales_by_year = df.groupby('year')['Sales'].sum()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(sales_by_year, labels=sales_by_year.index, autopct='%1.1f%%', startangle=90)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Top cities
        st.subheader("Top 5 Cities by Sales")
        top_cities = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_cities.index, top_cities.values, color='purple')
        ax.set_xlabel('City')
        ax.set_ylabel('Total Sales')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("👆 Please upload a CSV file to view the dashboard")

# Sales Prediction Page
elif page == "🔮 Sales Prediction":
    st.header("Sales Prediction Tool")
    st.markdown("Enter the details below to predict sales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", 
            ['Oil & Masala', 'Beverages', 'Food Grains', 'Fruits & Veggies', 'Bakery', 'Snacks', 'Eggs, Meat & Fish'])
        
        sub_category = st.text_input("Sub Category", "Masalas")
        
        city = st.text_input("City", "Chennai")
        
        region = st.selectbox("Region", ['North', 'South', 'West', 'Central', 'East'])
    
    with col2:
        discount = st.slider("Discount", 0.10, 0.35, 0.20, 0.01)
        
        profit = st.number_input("Expected Profit", min_value=0.0, value=300.0, step=10.0)
        
        month = st.selectbox("Month", range(1, 13), format_func=lambda x: 
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
        
        year = st.selectbox("Year", [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
    
    if st.button("Predict Sales", type="primary"):
        # Encoding the inputs
        le = LabelEncoder()
        
        category_encoded = ['Bakery', 'Beverages', 'Eggs, Meat & Fish', 'Food Grains', 
                           'Fruits & Veggies', 'Oil & Masala', 'Snacks'].index(category)
        
        # Creating a simple hash for sub_category and city since we don't have the original encoder
        sub_category_encoded = hash(sub_category) % 100
        city_encoded = hash(city) % 100
        region_encoded = ['Central', 'East', 'North', 'South', 'West'].index(region)
        
        # Preparing input for prediction
        input_data = np.array([[category_encoded, sub_category_encoded, city_encoded, 
                               region_encoded, discount, profit, month, year]])
        
        # Scaling the input
        input_scaled = scaler.transform(input_data)
        
        # Making prediction
        prediction = model.predict(input_scaled)[0]
        
        # Displaying result
        st.success(f"### Predicted Sales: ${prediction:,.2f}")
        
        # Additional insights
        st.info(f"""
        **Prediction Details:**
        - Expected Revenue: ${prediction:,.2f}
        - Expected Profit: ${profit:,.2f}
        - Profit Margin: {(profit/prediction)*100:.2f}%
        - Discount Applied: {discount*100:.0f}%
        """)

# Data Analysis Page
elif page == "📈 Data Analysis":
    st.header("Detailed Data Analysis")
    
    uploaded_file = st.file_uploader("Upload your sales data for analysis (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
        df['month_no'] = df['Order Date'].dt.month
        df['year'] = df['Order Date'].dt.year
        
        st.subheader("Dataset Overview")
        st.write(f"Total Records: {len(df)}")
        st.write(f"Date Range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
        
        # Show data
        if st.checkbox("Show Raw Data"):
            st.dataframe(df.head(100))
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df[['Sales', 'Discount', 'Profit']].describe())
        
        st.markdown("---")
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        numerical_cols = df[['Sales', 'Discount', 'Profit', 'month_no', 'year']]
        corr = numerical_cols.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Sales and Profit distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Sales'], bins=30, color='skyblue', edgecolor='black')
            ax.set_xlabel('Sales')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Profit Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(df['Profit'], bins=30, color='lightgreen', edgecolor='black')
            ax.set_xlabel('Profit')
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Sales vs Profit
        st.subheader("Sales vs Profit Relationship")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['Sales'], df['Profit'], alpha=0.5, color='orange')
        ax.set_xlabel('Sales')
        ax.set_ylabel('Profit')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("👆 Please upload a CSV file for detailed analysis")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About:** This dashboard provides sales analytics and prediction capabilities for supermarket data.
Built with Streamlit and Scikit-learn.

**Author:** Mustafiz Ahmed  
**Project:** Supermarket Grocery Sales Analysis  
**Organization:** Unified Mentor Private Limited
""")