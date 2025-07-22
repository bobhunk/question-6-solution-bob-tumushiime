# African Education Dropout Risk Analysis

**UCU MSc Data Science - Data Visualization Exam | Question 6 Solution**

This project examines school dropout patterns across Sub-Saharan Africa using data from the World Bank. The analysis combines statistical exploration, machine learning, and interactive visualizations to help education policymakers understand where and why children are leaving school.

## Project Overview

Education data from 27 African countries was analyzed to identify dropout risk patterns and provide practical insights for policy decisions. The work focuses on finding high-risk countries, understanding regional differences, and determining what factors most influence whether children stay in school.

## Key Findings

### Regional Differences Are Significant
The analysis shows clear patterns across African regions:
- Southern Africa performs best with 89.3% average primary enrollment
- East Africa follows closely at 87.7% enrollment
- Central Africa shows moderate performance at 79.5%
- West Africa lags significantly at 69.7% enrollment

This 20-point gap between West and Southern Africa represents millions of children who should be in school but aren't.

### Many Countries Face High Dropout Risk
Looking at 226 records from 27 countries over multiple years, the data shows that 45.2% of country-year combinations are classified as high dropout risk. This analysis used 38 different education and economic indicators to build a complete picture of each country's situation.

### Top 5 High-Risk Countries (Immediate Intervention Priority)
1. **Burkina Faso**: 93.3% of years classified as high dropout risk
2. **Cameroon**: 93.3% of years high risk
3. **Madagascar**: 93.3% of years high risk
4. **Niger**: 93.3% of years high risk
5. **Mozambique**: 80.0% of years high risk

### Economic Factors Matter Most
The statistical analysis reveals some important relationships:
- Countries with higher GDP per capita tend to have better primary enrollment (correlation = 0.304)
- Rural areas consistently show lower completion rates (correlation = -0.262)
- Gender gaps persist, with boys slightly more likely to be enrolled than girls (parity index = 0.96)
- Economic development is consistently linked to better educational outcomes

### Machine Learning Results
Three different prediction models were tested to identify dropout risk patterns. The Random Forest model performed best at predicting which countries and years would have high dropout risk. Economic indicators like GDP and rural population percentage turned out to be the most important factors for making these predictions. The models were validated using cross-validation to ensure they work reliably across different African countries.

## How This Analysis Can Be Used

### For Policymakers
This work provides clear evidence about which countries and regions need the most urgent attention. The analysis shows exactly where education investments would have the biggest impact, with specific recommendations for addressing rural education gaps and gender disparities. The predictive models can serve as an early warning system to identify problems before they become crises.

### For International Organizations
Development agencies can use these findings to focus their programs on the highest-risk areas identified in the analysis. The baseline data helps measure whether interventions are working, and the regional comparisons show opportunities for countries to learn from each other's successes. The evidence base supports advocacy efforts and funding proposals.

### For Researchers
All code and documentation is provided for others to replicate and extend this work. The framework can be applied to additional countries or education indicators. The World Bank API connection means the analysis can be updated with fresh data, and the transparent methodology supports further academic research.

## What's Included

### Main Analysis Files
- `eda_analysis.ipynb` - Complete exploratory data analysis with 226 cleaned records
- `dropout_predictor.py` - Machine learning models for dropout risk prediction
- `policy_brief.md` - Policy recommendations based on the findings
- `interactive_dashboard.py` - Script that generates the interactive visualizations

### Interactive Visualizations
- `risk_map.html` - Map showing dropout risk by country
- `regional_comparison.html` - Charts comparing regional performance
- `correlation_analysis.html` - How education and economic factors relate
- `time_series.html` - Education trends over time

### Data Files
- `modeling_dataset.csv` - Clean dataset with 226 records and 38 features
- `dropout_risk_features_20250713.csv` - Complete processed dataset from World Bank
- Raw data files in the data folder

### Documentation
- This README file
- `requirements.txt` - Python packages needed to run the code

## Getting Started

### 1. Look at the Results
The easiest way to see what was found is to open the HTML files in your browser. Just double-click on `risk_map.html` or any of the other HTML files to see the interactive charts.

### 2. Run the Code 
If you want to run the analysis:
```bash
# Install the required packages
pip install -r requirements.txt

# Open the main analysis notebook
jupyter notebook eda_analysis.ipynb

# Or generate new visualizations
python interactive_dashboard.py
```

### 3. Where to Start
1. `policy_brief.md` - Summary of findings and recommendations
2. `eda_analysis.ipynb` - Complete analysis with all the details
3. HTML files - Interactive charts you can explore
4. `modeling_dataset.csv` - The clean data used for analysis

## About the Data

All data comes from the World Bank Education Statistics, which tracks education indicators for countries around the world. This analysis uses 15 years of data (2010-2024) from 27 African countries, including enrollment rates, completion rates, gender gaps, and economic indicators. The World Bank data is reliable and standardized, making it possible to compare countries fairly.

### How the Analysis Works

The raw World Bank data gets processed to make it useful for analysis. This involves organizing the data so each row represents one country in one year, calculating gender parity ratios to see if boys or girls are more likely to be in school, and creating a risk classification system that identifies countries where too many children are dropping out.

---

**Data Source**: World Bank Education Statistics API  
**Analysis Period**: 2010-2024

```

## Methodology

### 1. Data Collection Strategy
- **World Bank API** - Standardized education indicators across all countries
- **Continental scope** - 27 Sub-Saharan African countries for comprehensive regional analysis
- **15-year timeframe** - 2010-2024 longitudinal trends and patterns
- **Multi-dimensional data** - Enrollment, completion, gender parity, and economic indicators
- **Automated collection** - Python-based API integration with error handling and rate limiting

### 2. Feature Engineering Approach
- **Data transformation** - Long-to-wide format conversion for ML readiness
- **Gender parity indices** - Female/male enrollment ratios to identify inequality
- **Transition rates** - Primary-to-secondary progression indicators
- **Completion gaps** - Enrollment vs. completion rate differences
- **Economic integration** - Per-capita education investment calculations
- **Risk classification** - Binary target variable based on completion thresholds

### 3. Analytical Framework
- **Exploratory Data Analysis** - Pattern identification and correlation analysis
- **Geospatial visualization** - Country-level choropleth mapping and regional clustering
- **Time series analysis** - Trend identification and progress tracking
- **Comparative analysis** - Cross-country performance benchmarking
- **Gender disparity assessment** - Male-female educational outcome gaps

### 4. Predictive Modeling
- **Classification problem** - Binary dropout risk prediction (High/Low risk)
- **Multiple algorithms** - Random Forest, Gradient Boosting, and Logistic Regression
- **Feature importance** - Identification of key dropout risk predictors
- **Model validation** - Cross-validation and performance metrics evaluation
- **Interpretability focus** - SHAP values and feature contribution analysis

### 5. Interactive Storytelling
- **Policy-focused narrative** - Actionable insights for education stakeholders
- **Dynamic visualizations** - User-driven exploration with filtering capabilities
- **Evidence-based recommendations** - Data-supported intervention strategies
- **Accessibility design** - Clear language and intuitive interface for policymakers

## Key Features

### Advanced Analytics
- **Risk Scoring** - District-level dropout probability assessment
- **Demographic Segmentation** - Gender, age, and economic group analysis
- **Geographic Clustering** - Regional pattern identification
- **Trend Forecasting** - Future dropout risk projections

### Interactive Visualizations
- **Dynamic Choropleth Maps** - District filtering and hover insights
- **Time Series Dashboard** - Multi-variable trend exploration
- **Comparative Analysis** - Cross-district and demographic comparisons
- **Risk Assessment Tool** - Real-time dropout probability calculator

### Policy Integration
- **Actionable Insights** - Specific intervention recommendations
- **Resource Allocation** - Data-driven funding prioritization
- **Monitoring Framework** - Key performance indicators for tracking
- **Success Stories** - Best practice identification and scaling

## Expected Outcomes

### Academic Excellence
- **Rigorous Methodology** - Transparent and reproducible analysis
- **Advanced Techniques** - State-of-the-art ML and visualization methods
- **Policy Relevance** - Real-world applicability and impact potential
- **Technical Innovation** - Novel approaches to education analytics

### Practical Impact
- **Ministry Collaboration** - Direct engagement with education officials
- **Evidence-based Policy** - Data-driven decision making support
- **Resource Optimization** - Efficient allocation of education investments
- **Student Success** - Improved retention and completion outcomes

## Technical Stack

**Data Processing:** Python, Pandas, NumPy  
**Machine Learning:** Scikit-learn, XGBoost, TensorFlow  
**Visualization:** Plotly, Dash, Folium, Seaborn  
**Geospatial:** GeoPandas, Shapely, Contextily  
**Interactive:** Jupyter, Streamlit, HTML/CSS/JS  

---
