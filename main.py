import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import GradientBoostingRegressor

class GreenAreaAnalyzer:
    def __init__(self, locations, train_years=3, predict_years=1):
        ee.Initialize(project='satellite-454512')
        self.locations = locations
        self.train_years = train_years
        self.predict_years = predict_years
        self.local_folder = 'NDVI_Images'
        os.makedirs(self.local_folder, exist_ok=True)

    def fetch_and_calculate_ndvi(self, name, coords, start_date, end_date):
        area_of_interest = ee.Geometry.Rectangle([coords[2], coords[0], coords[3], coords[1]])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(area_of_interest)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                      .select(['B8', 'B4']))
        images = collection.toList(collection.size()).getInfo()

        if len(images) == 0:
            print(f"No images found for {name} in the given date range.")
            return name, pd.DataFrame()

        ndvi_values, dates = [], []
        for image_info in images:
            image = ee.Image(image_info['id'])
            timestamp = image_info['properties']['system:time_start'] / 1000
            date = datetime.fromtimestamp(timestamp)
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            result = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=area_of_interest,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            if result and 'NDVI' in result:
                ndvi_values.append(float(result['NDVI']))
                dates.append(date)

        return name, pd.DataFrame({'Date': dates, 'NDVI': ndvi_values})

    def analyze_parallel(self, start_date, end_date):
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda loc: self.fetch_and_calculate_ndvi(loc[0], loc[1], start_date, end_date), self.locations.items())
        return dict(results)

    def plot_and_predict(self):
        base_year = 2019
        train_start_date = datetime(base_year, 1, 1).strftime('%Y-%m-%d')
        train_end_date = datetime(base_year + self.train_years, 1, 1).strftime('%Y-%m-%d')
        predict_end_date = datetime(base_year + self.train_years + self.predict_years, 1, 1).strftime('%Y-%m-%d')

        results = self.analyze_parallel(train_start_date, train_end_date)
        actual_future_results = self.analyze_parallel(train_end_date, predict_end_date)

        num_locations = len(results)
        cols = min(3, num_locations)
        rows = (num_locations + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

        greenery_change = []

        for i, (name, df) in enumerate(results.items()):
            ax = axs[i // cols, i % cols]
            if df.empty:
                print(f"No data available for {name}.")
                ax.axis('off')
                continue

            df['Timestamp'] = pd.to_datetime(df['Date']).astype('int64') / 10**9
            X = df['Timestamp'].values.reshape(-1, 1)
            y = df['NDVI'].values.reshape(-1, 1)

            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X, y.ravel())

            future_dates = pd.date_range(start=train_end_date, end=predict_end_date, freq='ME')
            future_timestamps = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            future_timestamps = np.array(future_timestamps).reshape(-1, 1)

            if len(future_timestamps) > 0:
                predictions = model.predict(future_timestamps)
                ax.plot(future_dates, predictions, color='green', label='Predicted NDVI', linewidth=1.5)

            ax.scatter(df['Date'], y, color='blue', label='Actual NDVI', s=30)
            ax.plot(df['Date'], model.predict(X), color='orange', label='Fitted NDVI', linewidth=1.5)

            if name in actual_future_results and not actual_future_results[name].empty:
                future_actual_df = actual_future_results[name]
                ax.scatter(future_actual_df['Date'], future_actual_df['NDVI'], color='red', label='Actual Future NDVI', s=30)

            ax.set_xlabel('Date')
            ax.set_ylabel('NDVI')
            ax.set_title(f'NDVI Trend and Prediction for {name}')
            ax.legend()
            ax.grid(True)

            change = "increased" if y[-1] > y[0] else "decreased"
            greenery_change.append((name, change))
            print(f"Greenery {change} in {name}")

        for j in range(i + 1, rows * cols):
            axs[j // cols, j % cols].axis('off')

        plt.tight_layout()
        plt.show()

        increased_locations = [name for name, change in greenery_change if change == "increased"]
        print("Locations with increased greenery:", increased_locations)

if __name__ == "__main__":
    locations = {
        'Location 1': [28.5, 28.7, 77.2, 77.4]
    }
    analyzer = GreenAreaAnalyzer(locations)
    analyzer.plot_and_predict()
