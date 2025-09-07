# 🚗 **Car Price Prediction App**
This project is a **Streamlit web application** that predicts used car prices based on car details like model, age, ownership, kilometers driven, fuel type, and transmission type.  
The app is powered by a **Machine Learning model (LightGBM)** trained on the used car price dataset from June 2025.

---

## 📂 **Project Structure**
📂 car-price-app/
├── app.py # Streamlit app code
├── model_carprice.pkl # Saved ML model
├── Car_Variants.csv # List of available car variants
├── requirements.txt # Dependencies
└── README.md # Project documentation

🌐 **Deployment**

This project is deployed on Streamlit Community Cloud.
Click below to try the app: https://used-car-price-prediction-2025-using-light-gbm.streamlit.app/

🛠 **Tech Stack**
Python
Pandas, NumPy, Scikit-learn, LightGBM
Streamlit for frontend

📊 **Model Info**
Trained on historical used car dataset
Preprocessing steps include:
One-hot encoding
Feature hashing for model/variant
Log transformation & scaling

*Evaluation Metrics:*
**MAE: ~1.50
RMSE: ~3.19
R²: ~0.68

✨ **Features**
User-friendly dropdowns for car model & variant
Predicts car price instantly
Deployed with Streamlit Cloud

🙌 **Acknowledgments**
Dataset: Kaggle used car dataset https://www.kaggle.com/datasets/sukhmansaran/used-cars-prices-cars-24
Framework: Streamlit
