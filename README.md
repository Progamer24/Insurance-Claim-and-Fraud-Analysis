# 🛡️ Insurance Claim and Fraud Analysis System

**A comprehensive solution for detecting fraudulent insurance claims using machine learning**

## 🔍 Overview

This end-to-end system helps insurance companies:
- Automatically flag potentially fraudulent claims
- Provide explainable risk scores (0-100%)
- Track fraudulent patterns over time
- Manage user access and investigations

## ✨ Key Features

### Machine Learning
- 🧠 XGBoost classifier trained on claim history
- 📊 Probability scores with confidence intervals
- 🔄 Continuous retraining with new labeled data

### Dashboard
- 📈 Interactive analytics with Plotly visualizations
- 👮 Admin review system for disputed claims
- 📤 PDF/CSV export capabilities

### Security
- 🔐 Role-based access (Admin/User)
- ⚠️ Automatic user banning after fraud threshold
- 📝 Full audit logging

## 🛠️ Tech Stack

**Backend**  
`Python` `XGBoost` `Scikit-learn` `SQLite` `Joblib`

**Frontend**  
`Streamlit` `Plotly` `FPDF`
