# 🏙️ Smart City AI Traffic Hub & Pattern Detection Engine

An advanced time-series machine learning deployment pipeline and interactive dashboard built to monitor, analyze, and forecast dynamic urban vehicle traffic distributions across major city grid junctions.

---

## 📝 Project Overview
Rapid urbanization requires intelligent transportation systems (ITS) capable of anticipating traffic gridloads before they manifest. This project addresses the urban congestion challenge by engineering an end-to-end predictive machine learning model. Moving away from primitive baselines that rely on manual entries or suffer from temporal data leakage, this architecture leverages advanced cyclical spatial-temporal wave geometry to map, detect, and forecast real-time municipal traffic behaviors with elite precision.

---

## 🌐 Live Interactive Application
The production dashboard is fully compiled, hosted, and live on the internet via Streamlit Cloud infrastructure. 

* **Live Application URL:** https://smartcity-traffic-forecast.streamlit.app/
* **Operational Status:** Active / Publicly Accessible

---

## ⚡ Key Features
* **Zero-Input Operations:** Fully automated dynamic infrastructure. Operators are not required to type in manual historical logs; they simply choose a junction and a future calendar date, and the engine computes the entire 24-hour sequence on the fly.
* **Glassmorphic Analytics Cards:** Modern high-contrast KPI metric panels that instantly expose Volumetric Daily Aggregates, Peak Congestion Spikes, and Average Flow Rates.
* **Fluid Interactive Charts:** Deployed high-end Plotly time-series vectors equipped with hover tooltips for real-time traffic grid auditing.
* **Leakage-Proof Validation:** Engineered with strict chronological sequence splitting to guarantee real-world forecasting reliability.

---

## 🛠️ Technologies Used
* **Core Language:** Python 3.x
* **Dashboard Interface:** Streamlit Engine
* **Machine Learning Framework:** XGBoost Regressor (Extreme Gradient Boosting)
* **Data Engineering & Math:** Pandas, NumPy, Scikit-Learn
* **Interactive Visualizations:** Plotly Graph Objects, Seaborn, Matplotlib
* **Model Serialization:** Joblib

---

## 🧠 Model Information
* **Predictive Architecture:** Optimized Gradient Boosted Decision Tree (GBDT) array natively tuned to map complex, non-linear seasonal waves.
* **Temporal Feature Engineering:** Raw time dimensions are converted into 2D circular wave topologies using Trigonometric Sine and Cosine encodings
* * **Temporal Feature Engineering:** Raw time dimensions are converted into 2D circular wave topologies using Trigonometric Sine and Cosine encodings:
  * `Hour_sin` = $\sin(2\pi \times \text{Hour} / 24)$
  * `Hour_cos` = $\cos(2\pi \times \text{Hour} / 24)$ to capture adjacent temporal patterns seamlessly.
* **Evaluation Metrics (Unseen Validation Data):**
  * **Authentic $R^2$ Accuracy Score:** **0.97** (Explaining 97% of total vehicular sequence variance without historical data leakage)
  * **Mean Absolute Error (MAE):** **2.91 vehicles**
  * **Root Mean Squared Error (RMSE):** **4.78 vehicles**

---

## 📊 Dataset Details
The model was trained on historical smart city grid logging profiles monitoring multi-junction urban corridors:
* **Target Footprint:** Junction 1 operates as the city's primary logistics hub, commanding ~60% of total vehicular density.
* **Commuting Profiles:** Clear bimodal rush-hour surges peaking at **9:00 AM - 10:00 AM** and **6:00 PM - 7:00 PM**.
* **Weekly Cycles:** Dense operational profiles across weekdays with flat, decentralized curves over weekends.

---

## 📂 Project Files Blueprint
* `Smart_city_Traffic_Pattern_Detection.ipynb` — Comprehensive research notebook mapping data cleaning, visual EDA, and model validation stages.
* `app.py` — Production script containing the glassmorphic user interface architecture.
* `traffic_production_model.pkl` — Serialized XGBoost model weight payload.
* `smart_city_traffic_predictions.csv` — Final sanitized predictions dataset computed over the unseen testing timeline.
* `requirements.txt` — Cloud dependency manifest mapping required for server environment deployment.

---

## 📸 Smart City Traffic AI Hub Preview
Click on the linkable preview placeholder below to inspect the interface layout, metric components, and interactive graphing engine.

<a href="https://smartcity-traffic-forecast.streamlit.app/" target="_blank">
"<img width="1908" height="928" alt="image" src="https://github.com/user-attachments/assets/b6364a35-0e5f-46ca-9906-bdfebd0b4d42" />
</a>


