# Supermarket Grocery Sales - Retail Analytics & ML Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive end-to-end machine learning project analyzing supermarket grocery sales data from Tamil Nadu, India (2015-2018). The project includes exploratory data analysis, predictive modeling using Random Forest and Linear Regression, and an interactive Streamlit web application for real-time sales forecasting.

## 📊 Project Overview

This project demonstrates the complete data science workflow from raw data to deployed machine learning models, analyzing **9,994 transactions** with **₹14.96 million in total sales** and delivering actionable business insights through interactive dashboards.

### Key Highlights

🎯 **Business Value:**
- Total Sales: ₹14,956,982
- Total Profit: ₹3,747,121 (25% margin)
- 67.3% sales growth from 2015 to 2018 (18.7% CAGR)
- 7 product categories across 24 cities in 5 regions

🤖 **Machine Learning:**
- **Random Forest Regressor**: R² = 0.356, MAE = ₹377.81
- **Linear Regression**: R² = 0.354, MAE = ₹379.27
- Feature importance: Profit (78.6%), Discount (4.7%)

📈 **Key Insights:**
- **Top Category**: Eggs, Meat & Fish (15.2% of sales, ₹2.27M)
- **Top Region**: West (32.1% of sales, ₹4.80M)
- **Peak Months**: September (₹1.71M) and November (₹1.79M)
- **Top Cities**: Kanyakumari, Vellore, Bodi

🚀 **Deployment:**
- Interactive Streamlit dashboard with 3 pages
- Real-time sales prediction interface
- Comprehensive data visualization suite
- User-friendly file upload functionality

## 📁 Project Structure

```
08_Supermarket_Grocery_Sales/
├── Charts/                              # 13 visualization outputs
│   ├── Actual vs Predicted Sales.png
│   ├── Correlation Heatmap.png
│   ├── Sales Distribution by Year.png
│   └── ... (10 more charts)
├── codes/
│   ├── Supermarket_Grocery_Sales.ipynb  # Main analysis notebook
│   └── app.py                           # Streamlit web application
├── data/
│   └── supermarket.csv                  # Dataset (9,994 records)
├── docs/
│   ├── Supermarket_Grocery_Sales_report.pdf  # 48-page detailed report
│   └── Supermart Grocery Sales - Retail Analytics Dataset.pdf
├── models/
│   ├── rf_sales_model.pkl              # Trained Random Forest model
│   └── scaler.pkl                      # Feature scaler
├── README.md
├── requirements.txt
└── .gitignore
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/MZ-314/Supermart-Grocery-Sales---Retail-Analytics-Dataset.git
cd supermarket-sales-analysis
```

2. **Create a virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Streamlit Web Application

**From the project root directory:**
```bash
streamlit run codes/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Running the Jupyter Notebook

**Option 1: Google Colab (Recommended)**
1. Upload `Supermarket_Grocery_Sales.ipynb` to Google Colab
2. Mount Google Drive and upload the dataset
3. Update file paths in the notebook
4. Run all cells

**Option 2: Local Jupyter**
```bash
jupyter notebook codes/Supermarket_Grocery_Sales.ipynb
```

## 📱 Web Application Features

### 📊 Dashboard Page
- **Key Metrics**: Total sales, profit, average order value, order count
- **Visualizations**:
  - Sales by Category (bar chart)
  - Sales by Region (bar chart)
  - Monthly Sales Trend (line chart)
  - Yearly Sales Distribution (pie chart)
  - Top 5 Cities by Sales (bar chart)
- **File Upload**: Upload custom CSV files for analysis

### 🔮 Sales Prediction Page
- **Input Fields**:
  - Category (dropdown)
  - Sub Category (text input)
  - City (text input)
  - Region (dropdown)
  - Discount (slider: 10%-35%)
  - Expected Profit (number input)
  - Month (dropdown)
  - Year (dropdown)
- **Output**:
  - Predicted sales amount
  - Expected profit
  - Profit margin calculation
  - Discount percentage

### 📈 Data Analysis Page
- **Statistical Summary**: Descriptive statistics for sales, discount, profit
- **Correlation Heatmap**: Visual representation of feature relationships
- **Distributions**:
  - Sales distribution histogram
  - Profit distribution histogram
  - Sales vs Profit scatter plot
- **Raw Data Viewer**: Inspect first 100 records

## 📊 Dataset Information

**Source:** Fictional grocery delivery application data (Tamil Nadu, India)

**Specifications:**
- **Records:** 9,994 transactions
- **Time Period:** 2015-2018 (4 years)
- **Geographic Coverage:** 24 cities across 5 regions
- **Product Categories:** 7 main categories

**Features:**
| Column | Description | Type |
|--------|-------------|------|
| Order ID | Unique transaction identifier | String |
| Customer Name | Customer identifier | String |
| Category | Product category (7 categories) | Categorical |
| Sub Category | Detailed product classification | Categorical |
| City | Transaction city (24 cities) | Categorical |
| Order Date | Transaction date (2015-2018) | DateTime |
| Region | Geographic region (5 regions) | Categorical |
| Sales | Order value in INR (₹500-₹2,500) | Integer |
| Discount | Discount rate (10%-35%) | Float |
| Profit | Profit in INR (₹25-₹1,121) | Float |
| State | Tamil Nadu (constant) | String |

**Engineered Features:**
- `month_no`: Month number (1-12)
- `Month`: Month name (January-December)
- `year`: Year (2015-2018)

## 🤖 Machine Learning Models

### Model Performance Comparison

| Model | R² Score | RMSE | MAE | MSE |
|-------|----------|------|-----|-----|
| **Random Forest** | **0.356** | ₹460.86 | **₹377.81** | 212,393.05 |
| Linear Regression | 0.354 | ₹461.58 | ₹379.27 | 213,058.77 |

### Feature Importance (Random Forest)

1. **Profit**: 78.6% - Dominant predictor
2. **Discount**: 4.7% - Secondary factor
3. **City**: 4.2% - Geographic influence
4. **Category**: 3.8% - Product type impact
5. **Year**: 2.9% - Temporal trend
6. **Region**: 2.7% - Regional variation
7. **Sub Category**: 2.1% - Granular product detail
8. **Month**: 1.0% - Seasonal pattern

### Model Development Process

1. **Data Preprocessing**:
   - Date conversion (mixed format handling)
   - Missing value check (0 missing)
   - Duplicate detection (0 duplicates)

2. **Feature Engineering**:
   - Temporal feature extraction (month, year)
   - Label encoding for categorical variables

3. **Train-Test Split**: 80/20 (7,995 training / 1,999 testing)

4. **Feature Scaling**: StandardScaler normalization

5. **Model Training**:
   - Linear Regression (baseline)
   - Random Forest (n_estimators=100, max_depth=10)

6. **Model Persistence**: Pickle serialization

## 📈 Business Insights & Recommendations

### Top Findings

1. **Category Performance**:
   - Eggs, Meat & Fish: ₹2,267,401 (15.2%)
   - Snacks: ₹2,237,546 (15.0%)
   - Food Grains: ₹2,115,272 (14.1%)
   - Remarkably balanced portfolio (13.6%-15.2%)

2. **Regional Dynamics**:
   - West region dominance: 32.1% of sales
   - East region strong: 28.4% of sales
   - North region concern: Only ₹1,254 total sales

3. **Temporal Patterns**:
   - Peak months: September (₹1.71M), November (₹1.79M)
   - Trough month: February (₹830K)
   - 116% swing between peak and trough

4. **Growth Trajectory**:
   - 2015: ₹2,975,599
   - 2018: ₹4,977,512
   - 67.3% total growth, 18.7% CAGR

### Strategic Recommendations

✅ **Immediate Actions:**
1. Prioritize high-profit item promotion
2. Increase Eggs, Meat & Fish category investment
3. Investigate North region performance issues
4. Plan seasonal inventory for Sept/Nov peaks

✅ **Medium-Term Initiatives:**
5. Strengthen East region presence
6. Expand into Central and South regions
7. Optimize working capital for seasonal fluctuations
8. Reassess discount strategy effectiveness

✅ **Long-Term Strategy:**
9. Implement dynamic pricing algorithms
10. Deploy predictive analytics dashboard enterprise-wide
11. Develop customer analytics platform
12. Expand feature set with external data (weather, holidays)

## 🛠️ Technologies Used

**Core Languages & Frameworks:**
- Python 3.8+
- Streamlit (Web Application)

**Data Science & ML:**
- Pandas (Data manipulation)
- NumPy (Numerical computing)
- Scikit-learn (Machine learning)
  - LinearRegression
  - RandomForestRegressor
  - StandardScaler
  - LabelEncoder
  - train_test_split

**Visualization:**
- Matplotlib (Static plots)
- Seaborn (Statistical graphics)

**Development Environment:**
- Google Colab (Cloud notebook)
- Jupyter Notebook (Local development)

**Model Persistence:**
- Pickle (Model serialization)

**Version Control:**
- Git
- GitHub

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **Supermarket_Grocery_Sales_report.pdf** (48 pages):
  - Abstract & Introduction
  - Literature Review & Background
  - Dataset Description
  - Methodology (preprocessing, EDA, modeling)
  - Exploratory Data Analysis (13 visualizations)
  - Model Development & Evaluation
  - Business Insights & Recommendations
  - Deployment Guide
  - Limitations & Future Scope
  - Conclusions

## 🔍 Model Evaluation Details

### Error Distribution
- **Errors < ₹200**: ~25% of predictions
- **Errors ₹200-₹400**: ~35% of predictions
- **Errors ₹400-₹600**: ~25% of predictions
- **Errors > ₹600**: ~15% of predictions

**60% of predictions within ±₹400** (±27% relative error)

### Prediction Accuracy by Sales Range

| Sales Range | Avg Error | R² Score | Notes |
|-------------|-----------|----------|-------|
| Low (₹500-₹1,000) | ±₹350 | ~0.28 | Limited feature differentiation |
| Medium (₹1,000-₹2,000) | ±₹365 | ~0.40 | Best performance (most data) |
| High (₹2,000-₹2,500) | ±₹520 | ~0.32 | Higher variance (fewer samples) |

## 🚧 Limitations

- **Temporal Scope**: Limited to 4 years (2015-2018)
- **Geographic Scope**: Single state (Tamil Nadu)
- **Feature Set**: Lacks external data (weather, holidays, competition)
- **Fictional Data**: Synthetic dataset may not capture real-world complexity
- **Model Performance**: R² of 0.356 indicates room for improvement
- **Profit as Predictor**: In real forecasting, profit is unknown ex-ante

## 🔮 Future Enhancements

### Phase 1: Advanced Modeling (Months 1-6)
- ⭐ XGBoost and neural network implementations
- ⭐ Comprehensive feature engineering
- ⭐ Hyperparameter optimization (GridSearch/RandomSearch)
- ⭐ External data integration (weather, holidays, economic indicators)
- ⭐ MLOps foundation (model registry, monitoring)

### Phase 2: Scale & Expand (Months 6-12)
- 🚀 Real-time forecasting system
- 🚀 Cloud infrastructure migration (AWS/GCP/Azure)
- 🚀 Customer-level analytics
- 🚀 SKU-level forecasting
- 🚀 Mobile application development

### Phase 3: Advanced Capabilities (Months 12-24)
- 🎯 Prescriptive analytics and optimization
- 🎯 Causal inference framework
- 🎯 B2B channel analytics
- 🎯 Geographic expansion decision support
- 🎯 Automated retraining pipeline

### Phase 4: Innovation (Months 24-36)
- 🔬 Automated machine learning (AutoML)
- 🔬 Explainable AI (SHAP, LIME)
- 🔬 Multi-modal learning
- 🔬 Reinforcement learning for dynamic pricing

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Complete data science workflow execution
- ✅ Real-world retail analytics application
- ✅ Machine learning model development & deployment
- ✅ Interactive web application creation
- ✅ Business insight generation from data
- ✅ Professional documentation practices
- ✅ Model evaluation and comparison
- ✅ Feature engineering techniques
- ✅ Data visualization best practices

## 👨‍💻 Author

**Mustafiz Ahmed**  
UMID: 05072548678  
Project Duration: July 2025 - January 2026  
Organization: Unified Mentor Private Limited

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: Fictional dataset created for educational purposes
- **Organization**: Unified Mentor Private Limited
- **Tools**: Google Colab, Python Data Science Stack, Streamlit

## 📞 Contact & Contributions

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](../../issues) if you want to contribute.

For questions or collaboration opportunities, please open an issue or submit a pull request.

---

## 🚀 Quick Start Commands

```bash
# Clone repository
git clone https://github.com/MZ-314/Supermart-Grocery-Sales---Retail-Analytics-Dataset.git
cd supermarket-sales-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run codes/app.py

# Open notebook
jupyter notebook codes/Supermarket_Grocery_Sales.ipynb
```

---

**⭐ If you find this project useful, please consider giving it a star!**

**📊 Data-Driven Retail Analytics for Business Growth**

---

*Last Updated: February 2026*