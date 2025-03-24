# Greenery Analysis Using NDVI and Machine Learning

## Overview
This project analyzes the greenery trend using NDVI (Normalized Difference Vegetation Index) values derived from satellite images. It applies multiple machine learning models to predict future greenery trends based on historical data.

## Features
- Fetch NDVI values using Google Earth Engine.
- Train machine learning models (RandomForest, SVR, GradientBoosting, DecisionTree, and XGBoost).
- Predict NDVI values for future years.
- Visualize actual, fitted, and predicted NDVI values.
- Display greenery trends based on predictions.
- Display greenery trend based on historical NDVI values.

## Prerequisites
- Python 3.9 or higher
- Google Earth Engine API
- Required Python Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `xgboost`
  - `earthengine-api`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-url.git
    ```
2. Install required libraries:
    ```bash
    pip install numpy pandas matplotlib scikit-learn xgboost earthengine-api
    ```
3. Authenticate with Google Earth Engine:
    ```bash
    earthengine authenticate
    ```

## Project Structure
```
.
├── main.py                    # Main script for analysis
├── README.md                  # Project documentation
└── requirements.txt           # List of dependencies
```

## Usage
1. Edit the `locations` dictionary in `main.py` to specify the coordinates of the areas of interest in the format:
    ```python
    locations = {
        'Location 1': [latitude1, longitude1, latitude2, longitude2]
    }
    ```
2. Run the script:
    ```bash
    python main.py
    ```

3. View the visualized results and predicted greenery trends.

## Explanation of Outputs
- **Actual NDVI (Blue)**: Historical NDVI values from satellite images.
- **Fitted NDVI (Orange)**: Model's prediction on the training data.
- **Predicted NDVI (Green)**: Future NDVI values predicted by the model.
- **Greenery Trend**: Indicates if the greenery is increasing or decreasing.

## Troubleshooting
- Ensure Google Earth Engine API is authenticated.
- Confirm the coordinates are correct.
- Check internet connectivity.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Google Earth Engine for satellite data.
- Scikit-Learn and XGBoost for regression models.

For further questions, feel free to reach out!

