import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class GreenAreaAnalyzer:
    def __init__(self, locations, train_years=3, predict_years=1):
        ee.Initialize(project='satellite-454512')
        self.locations = locations
        self.train_years = train_years
        self.predict_years = predict_years

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

    def analyze_sequential(self, start_date, end_date):
        results = {}
        for loc in self.locations.items():
            name, data = self.fetch_and_calculate_ndvi(loc[0], loc[1], start_date, end_date)
            results[name] = data
        return results

    def plot_and_predict(self):
        base_year = 2020
        train_start_date = datetime(base_year, 1, 1).strftime('%Y-%m-%d')
        train_end_date = datetime(base_year + self.train_years, 1, 1).strftime('%Y-%m-%d')
        predict_end_date = datetime(base_year + self.train_years + self.predict_years, 1, 1).strftime('%Y-%m-%d')

        results = self.analyze_sequential(train_start_date, train_end_date)
        actual_future_results = self.analyze_sequential(train_end_date, predict_end_date)

        for name, df in results.items():
            if df.empty:
                print(f"No data available for {name}.")
                continue

            df['Timestamp'] = pd.to_datetime(df['Date']).astype('int64') / 10**9
            X = df['Timestamp'].values.reshape(-1, 1)
            y = df['NDVI'].values.reshape(-1, 1)

            # Scaling the data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)

            future_dates = pd.date_range(start=train_end_date, end=predict_end_date, freq='ME')
            future_timestamps = (future_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            future_timestamps = np.array(future_timestamps).reshape(-1, 1)
            future_timestamps_scaled = scaler_X.transform(future_timestamps)

            # SVR Model Optimization
            svr = SVR()
            svr.fit(X_scaled, y_scaled.ravel())
            predictions_scaled = svr.predict(future_timestamps_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

            plt.figure(figsize=(10, 6))
            plt.scatter(df['Date'], y, color='blue', label='Actual NDVI', s=30)
            plt.plot(df['Date'], scaler_y.inverse_transform(svr.predict(X_scaled).reshape(-1, 1)), color='orange', label='Fitted NDVI', linewidth=1.5)
            plt.plot(future_dates, predictions, color='green', label='Predicted NDVI', linewidth=1.5)
            if name in actual_future_results and not actual_future_results[name].empty:
                future_actual_df = actual_future_results[name]
                plt.scatter(future_actual_df['Date'], future_actual_df['NDVI'], color='red', label='Actual Future NDVI', s=30)
            plt.title(f'SVR (Optimized) - {name}')
            plt.legend()
            plt.grid(True)

            # Calculate Trend
            if len(predictions) > 1 and predictions[-1] > predictions[0]:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            plt.figtext(0.15, 0.85, f'Greenery Trend: {trend}', fontsize=12, color='green' if trend == 'increasing' else 'red')
            plt.show()

            # Plot Greenery Trend
            plt.figure(figsize=(8, 5))
            plt.plot(future_dates, predictions, color='green' if trend == 'increasing' else 'red', marker='o')
            plt.title(f'Greenery Trend Over Time - {name}')
            plt.xlabel('Date')
            plt.ylabel('NDVI')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    locations = {
        'Amazon Rainforest': [-3.5, -3.0, -60.0, -59.5]
    }
    analyzer = GreenAreaAnalyzer(locations)
    analyzer.plot_and_predict()

