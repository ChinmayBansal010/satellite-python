
# 🌿 Greenery Analysis Using NDVI and Machine Learning

## 🌍 Overview

This project leverages **Artificial Intelligence (AI)** and **satellite imagery** to analyze and predict vegetation health using the **Normalized Difference Vegetation Index (NDVI)**. By integrating **Google Earth Engine** with multiple **machine learning models**, it forecasts future greenery trends across defined geospatial regions.

This solution demonstrates the power of combining **remote sensing** with **AI/ML** for real-world environmental insights.

---

## 🔑 Features

- ✅ Fetch NDVI values using Google Earth Engine from Sentinel-2 imagery.
- 🧠 Train multiple machine learning models:
  - Random Forest
  - Support Vector Regressor (SVR)
  - Gradient Boosting
  - Decision Tree
  - XGBoost
- 🔮 Predict NDVI values for future timeframes.
- 📈 Visualize:
  - Actual NDVI (Blue)
  - Fitted NDVI on training data (Orange)
  - Predicted NDVI (Green or Red for trends)
- 🔍 Automatic trend classification: **increasing** or **decreasing** greenery.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.9 or above
- Google Earth Engine account
- Internet connectivity

### 🧩 Required Python Libraries

Install all dependencies with:

```bash
pip install -r requirements.txt
````

Or manually:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost earthengine-api
```

### 🔐 Authenticate Google Earth Engine

Run this command to authenticate your Earth Engine access:

```bash
earthengine authenticate
```

---

## 📂 Project Structure

```
.
├── main.py              # Main script with NDVI analysis and ML predictions
├── requirements.txt     # Python library dependencies
└── README.md            # Documentation (this file)
```

---

## 📌 How to Use

1. **Set your locations** in `main.py` like so:

```python
locations = {
    'Amazon Rainforest': [-3.5, -60.0, -3.0, -59.5],
    'Sahara Desert': [24.0, 3.0, 24.5, 3.5],
    'Central Park, NY': [40.7644, -73.9818, 40.8005, -73.9580]
}
```

2. **Run the script:**

```bash
python main.py
```

3. **View outputs:**

   * NDVI trends plotted for each location
   * Model-wise prediction graphs
   * AI-inferred trend (increasing/decreasing)

---

## 📊 Output Explanation

| Component         | Description                              |
| ----------------- | ---------------------------------------- |
| 🔵 Actual NDVI    | Raw satellite data over training years   |
| 🟠 Fitted NDVI    | Regressor's curve over the training data |
| 🟢 Predicted NDVI | Estimated NDVI values for future years   |
| 🔺 Trend Color    | Green if increasing, Red if decreasing   |

> The **Support Vector Regressor (SVR)** gave the most consistent and accurate predictions across regions.

---

## 🧠 Why This Project Fits AI / GenAI Domain

* Uses **AI regression models** for temporal prediction.
* Applies AI to **geospatial satellite data**.
* Could be extended to:

  * Generate natural-language reports (GenAI)
  * Build dashboards (with Streamlit/Gradio)
  * Recommend environmental actions (Decision AI)

---

## 🛠️ Troubleshooting

* If NDVI values are missing:

  * Check cloud cover threshold (10% default).
  * Verify location coordinates are within valid bounds.
* If Earth Engine throws an error:

  * Ensure you've run `earthengine authenticate` and approved access.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE.txt) file for details.

---

## 🙏 Acknowledgments

* **Google Earth Engine** – Satellite image access and NDVI calculation
* **Scikit-learn & XGBoost** – Powerful regression algorithms
* **Matplotlib** – Clean and intuitive visualizations

---

## 👨‍💻 Author

**Chinmay Bansal**

B.Tech CSE (AI & ML)

Manav Rachna International Institute of Research and Studies

---

*This project brings together the worlds of AI, environmental sustainability, and geospatial data to predict and preserve green life on Earth.* 🌱
