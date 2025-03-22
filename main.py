import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import GradientBoostingRegressor

class GreenAreaAnalyzer:
    def __init__(self, locations, start_date='2024-01-01', end_date='2025-01-01'):
        ee.Initialize(project='satellite-454512')
        self.locations = locations
        self.start_date = start_date
        self.end_date = end_date
        self.local_folder = 'NDVI_Images'
        # os.makedirs(self.local_folder, exist_ok=True)

    def fetch_and_calculate_ndvi(self, name, coords):
        area_of_interest = ee.Geometry.Rectangle([coords[2], coords[0], coords[3], coords[1]])
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(area_of_interest)
                      .filterDate(self.start_date, self.end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                      .select(['B8', 'B4']))
        images = collection.toList(collection.size()).getInfo()

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

    def analyze_parallel(self):
        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda loc: self.fetch_and_calculate_ndvi(loc[0], loc[1]), self.locations.items())
        return dict(results)

    def plot_and_predict(self):
        results = self.analyze_parallel()
        num_locations = len(results)
        cols = min(5, num_locations)
        rows = (num_locations + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

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

            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X, y.ravel())
            predictions = model.predict(X)

            ax.scatter(df['Date'], y, color='blue', label='Actual NDVI', s=30)
            ax.plot(df['Date'], predictions, color='red', label='Predicted NDVI', linewidth=1.5)
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

        # Display locations where greenery increased
        increased_locations = [name for name, change in greenery_change if change == "increased"]
        print("Locations with increased greenery:", increased_locations)

if __name__ == "__main__":
    locations = {
        'Location 1': [28.5, 28.7, 77.2, 77.4],
        'Location 2': [28.4, 28.6, 77.1, 77.3],
        'Location 3': [28.3, 28.5, 77.0, 77.2],
        'Location 4': [28.2, 28.4, 76.9, 77.1],
        'Location 5': [28.1, 28.3, 76.8, 77.0],
        'Location 6': [28.0, 28.2, 76.7, 76.9],
        'Location 7': [27.9, 28.1, 76.6, 76.8],
        'Location 8': [27.8, 28.0, 76.5, 76.7],
        'Location 9': [27.7, 27.9, 76.4, 76.6],
        'Location 10': [27.6, 27.8, 76.3, 76.5]
    }
    analyzer = GreenAreaAnalyzer(locations)
    analyzer.plot_and_predict()
