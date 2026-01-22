# Global Analytics: Financial Markets & Price Dynamics

## Overview
This project presents an end-to-end exploratory data analysis and visualization of financial market data, with a focus on price trends, trading volumes, and seasonal patterns. It applies statistical and visual analytics techniques to support data-driven insights for investors, analysts, and policymakers.

The work consolidates academic analysis (Phase 1) and applied data exploration (Phase 2) into a single, portfolio-ready project.

---

## Objectives
- Analyze market price dynamics using opening, closing, and adjusted prices  
- Explore distributions, volatility, and outliers in financial data  
- Identify seasonal and cumulative trends across time  
- Demonstrate effective use of data visualization for market interpretation  

---

## Dataset
The dataset includes market-level financial indicators such as:
- Opening price  
- Closing price  
- Adjusted closing price  
- Trading volume  
- Temporal attributes (date, month, year)  

All raw and cleaned datasets are stored in the `data/` directory.

---

## Methodology
1. **Data Cleaning & Preparation**  
   - Handling missing values and formatting time-series variables  
   - Creating derived variables for trend and seasonality analysis  

2. **Exploratory Data Analysis (EDA)**  
   - Line and scatter plots for price trend analysis  
   - Violin and KDE plots to understand data distributions  
   - ECDF plots to analyze cumulative behavior  
   - Bar, count, and pie charts for seasonal and categorical insights  

3. **Visualization & Interpretation**  
   - Translating statistical patterns into clear visual narratives  
   - Interpreting market behavior through volume-price relationships  

Detailed methodology is documented in the project notebooks and reports.

---

## Tools & Technologies
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Git & GitHub  

---

## Project Structure
```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── Phase2_Group51.ipynb
├── reports/
│   ├── Phase1_Group51.pdf
│   └── Phase2_Group51.pdf
├── docs/
│   └── literature_review.pdf
└── README.md


---

## Key Insights
* Strong correlation between opening and closing prices, indicating market continuity
* Distinct distribution patterns revealing volatility and skewness
* Seasonal variations in trading activity across months
* Trading volume as a key indicator of market participation and liquidity

---

## Limitations
* Analysis is based on historical data and does not account for real-time market shocks
* Limited to available features; macroeconomic variables are not included
* Findings are exploratory and not predictive in nature

---

## Future Works
* Integrate predictive models for price forecasting
* Extend analysis with macroeconomic and sentiment data
* Deploy an interactive dashboard or Streamlit web app for live exploration

---

## Author

Hishikesh Phukan
Master of Data Science (RMIT University)
