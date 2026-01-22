import streamlit as st

st.title("🤖 Model Performance (Phase 2)")

st.markdown("""
### Models Evaluated
- Linear Regression  
- Random Forest  
- XGBoost  
- LSTM  

### Best Performing Model
**XGBoost Regressor**

### Why XGBoost?
- Lowest RMSE
- Strong handling of non-linear relationships
- Robust to outliers after log transformation
- Scales well for large datasets

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

Refer to the Phase 2 report for detailed metric tables and comparisons.
""")
