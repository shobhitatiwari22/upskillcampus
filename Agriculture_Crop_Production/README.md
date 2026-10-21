# 🌾 Indian Crop Yield Prediction Pipeline

## 📖 Project Overview
This project provides an end-to-end data engineering, statistical analysis, and machine learning pipeline for predicting regional agricultural crop yields across India. The main objective is to translate messy, complex cultivation economics data into a live, interactive production-ready forecasting tool. By inputting specific crop types, target states, and cultivation budgets, farmers and agricultural stakeholders can immediately simulate and estimate expected crop yields per hectare.

## 🌐 Live Interactive Application
🎯 **[Try the Deployed Web App on Streamlit Cloud Here!](https://indian-crop-predictor.streamlit.app/)**

*Interact with the deployed Machine Learning model in real-time. Simply select a crop type, target state, input the cultivation budgets, and calculate predicted outputs instantly.*

---

## 🚀 Key Features
* **Interactive Live Web App:** A fully responsive web interface deployed on the cloud for real-time yield simulations.
* **Intelligent Shape Shifting:** Implemented a robust architectural migration from a highly sparse wide format into an analytics-ready long format, preventing data loss during cleaning.
* **Dynamic One-Hot Inference:** Custom logic designed to accept clean user inputs and seamlessly translate them into structural binary dummy arrays required by the machine learning backend.
* **Localized Context:** Features tailored data tracking across major agricultural states (such as Uttar Pradesh, Tamil Nadu, and Punjab) and distinct regional crop varieties.

## 🛠️ Technologies Used
* **Programming Language:** Python
* **Data Engineering & Analysis:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib
* **Model Serialization:** Pickle
* **Cloud Application Framework:** Streamlit Community Cloud

## 🧠 Model Information
* **Core Algorithm:** Random Forest Regressor
* **Prediction Accuracy:** **96.6%** on unseen testing data splits.
* **Feature Weights & Architecture:** The model maps highly correlated inputs—specifically the Cost of Cultivation (`C2` and `A2_FL` parameters)—against historical performance metrics to construct randomized decision trees for low-latency regressions.
* **Deployment Assets:** Weight states are serialized into `crop_model.pkl`, while structural input columns are strictly matched against indexes stored inside `model_columns.pkl`.

## 📊 Dataset Details
The machine learning pipeline is trained on historical data spanning Indian agricultural metrics from 1993 through 2014. 
* **Data Reshaping Breakthrough:** Initial raw data columns contained highly sparse features that resulted in zero viable records when cleaned using standard rows dropping. By executing a global `pd.melt()` operation, the architecture was successfully restructured into a dense long format, securing **4,272 high-quality entries** for model training.
* **Key Statistical Insight:** The dataset reveals a massive positive statistical correlation (~0.88) between localized cultivation investments (`C2` costs) and the final crop output, proving that structured agricultural financial planning corresponds directly to peak yield optimization.

## 📂 Project Files Directory
* `Prediction of Agriculture Crop Production in India.ipynb` — Documented Jupyter Notebook detailing full exploratory data analysis (EDA), cleaning routines, and model training.
* `app.py` — Production script managing the front-end interface layout, input tracking, and model prediction logic.
* `requirements.txt` — Configuration text file defining explicit server library installations.
* `crop_model.pkl` — Serialized pre-trained Random Forest Regressor binary weight file.
* `model_columns.pkl` — Serialized list of structural categorical training feature indices.
* `*.csv` — Historical raw dataset catalogs detailing Indian agricultural cost and frequency matrices.

---
## 📸 Application Previews

**Application Screenshot:**

<a href="https://indian-crop-predictor.streamlit.app" target="_blank">
  <img width="1600" height="779" alt="Application Interface" src="https://github.com/user-attachments/assets/b68264e7-589b-4350-b782-4e0f5679b49b" />
</a>

**Result:**

<a href="https://streamlit.app" target="_blank">
  <img width="1600" height="779" alt="Prediction Result" src="https://github.com/user-attachments/assets/b3f00794-5092-4e24-a1c3-774c5852c160" />
</a>



---

## 👤 Author
**Shobhita Tiwari**
