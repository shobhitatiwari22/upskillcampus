# 🏙️ Smart City AI Traffic Hub & Pattern Detection Engine

An end-to-end time-series machine learning deployment pipeline and interactive glassmorphic dashboard built to monitor, analyze, and forecast dynamic urban vehicle traffic distributions across major city grid junctions.

**Live Application Link:** https://smartcity-traffic-forecast.streamlit.app/

---

## 📊 Core Analytical Insights (Exploratory Data Analysis)
Deep exploratory research on the city's historical traffic framework revealed three dominant patterns:
* **Major Traffic Footprint:** **Junction 1 serves as the city's primary traffic hub, accounting for roughly 60% of all recorded vehicle volume.** Junctions 2, 3, and 4 maintain lower, stable baselines.
* **Bimodal Commuting Patterns:** Traffic exhibits classic rush hour spikes, rising sharply during the morning commute (**9:00 AM - 10:00 AM**) and hitting peak daily congestion during the evening return window (**6:00 PM - 7:00 PM**).
* **Weekly Variation Load:** High-density loads remain concentrated across weekdays (Monday through Friday), while weekends display highly dispersed traffic with absent morning rush hours.

---

## 🧠 Model Architecture & Machine Learning Pipeline
Unlike primitive baseline models that require manual data inputs or introduce future-data leakage through historical lag columns, this production framework uses a **100% independent standalone spatial-temporal wave configuration**.

### 1. Cyclical Trigonometric Feature Engineering
Time is naturally circular. To teach the model that hour `23` (11 PM) sits directly next to hour `0` (Midnight), raw timestamps are mapped into two-dimensional circular wave spaces using Sine and Cosine transformations:
* $\text{Hour\_sin} = \sin(2\pi \times \text{Hour} / 24)$
* $\text{Hour\_cos} = \cos(2\pi \times \text{Hour} / 24)$
* Similar geometric curves are automatically calculated for `DayOfWeek` (7-day wave) and `Month` (12-month wave).

### 2. Validation Split & Algorithm Tuning
* **Chronological Slicing:** Avoided standard random shuffling which ruins time-series data. Implemented a strict chronological validation split (80% historical past for training, 20% future timeline for evaluation).
* **Predictive Architecture:** Deployed an advanced **XGBoost Regressor** optimized to track complex non-linear commuter curves entirely from cyclical spatial-temporal inputs.

---

## 📈 Final Performance Verification Results
The optimized cyclical boosting framework yielded outstanding, authentic performance metrics on unseen chronological validation data:

* **Authentic $R^2$ Accuracy Score:** **0.97** (Explaining 97% of all future traffic sequence variance)
* **Mean Absolute Error (MAE):** **2.91 vehicles** (On average, predictions are off by fewer than 3 cars per hour)
* **Root Mean Squared Error (RMSE):** **4.78 vehicles**

---

## 💻 UI/UX Production Features
The user-facing control dashboard is styled using professional enterprise design systems:
* **Glassmorphic Metric Panels:** Instantly exposes Volumetric Aggregates, Peak Congestion Spikes, and Average Flow Rates for any targeted timeline horizon.
* **Interactive Timeline Graphing:** Built using responsive Plotly curves featuring precise tracking nodes and active tooltip inspection windows on user hover.
* **Zero-Input Operations:** Fully automated infrastructure utility. System operators simply pick a junction and calendar date, and the engine maps out the entire 24-hour dispatch projection dynamically.

---

## 📂 Repository File Blueprint
* `Smart_city_Traffic_Pattern_Detection.ipynb` — Research notebook containing data cleaning, visual EDA charts, and model execution steps.
* `app.py` — Streamlit dashboard script utilizing custom dark-theme glassmorphic styling sheets.
* `traffic_production_model.pkl` — Serialized XGBoost model weights artifact exported via joblib.
* `smart_city_traffic_predictions.csv` — Sanitized tabular output of predictions computed over the unseen test set.
* `requirements.txt` — Python library dependencies list file required for cloud deployment initialization.

