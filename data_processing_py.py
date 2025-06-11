import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
import geopandas as gpd
from geopy.distance import geodesic
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):

        try:
            self.cars = pd.read_csv(self.config['paths']['cars_file'])
            self.users = pd.read_csv(self.config['paths']['users_file'])
            self.travels = pd.read_csv(self.config['paths']['travels_file'])
            self.offers = pd.read_csv(self.config['paths']['offers_file'])
            
            print(f"Завантажено: {len(self.cars)} автомобілів, {len(self.users)} користувачів")
            print(f"           {len(self.travels)} поїздок, {len(self.offers)} пропозицій")
            return True
        except Exception as e:
            print(f"Помилка завантаження даних: {e}")
            return False
    
    def clean_data(self):

        # Очищення даних про автомобілі
        self.cars = self.cars.dropna(subset=['brand', 'model', 'seats'])
        self.cars['seats'] = pd.to_numeric(self.cars['seats'], errors='coerce')
        self.cars['year'] = pd.to_numeric(self.cars['year'], errors='coerce')
        self.cars = self.cars[(self.cars['seats'] >= 1) & (self.cars['seats'] <= 10)]
        self.cars = self.cars[(self.cars['year'] >= 1980) & (self.cars['year'] <= 2025)]
        

        self.users = self.users.dropna(subset=['name', 'rating'])
        self.users['rating'] = pd.to_numeric(self.users['rating'], errors='coerce')
        self.users['tripsTaken'] = pd.to_numeric(self.users['tripsTaken'], errors='coerce')
        self.users = self.users[(self.users['rating'] >= 0) & (self.users['rating'] <= 10)]
        

        self.offers = self.offers.dropna(subset=['pricePerKm', 'pricePerMin'])
        self.offers['pricePerKm'] = pd.to_numeric(self.offers['pricePerKm'], errors='coerce')
        self.offers['pricePerMin'] = pd.to_numeric(self.offers['pricePerMin'], errors='coerce')
        
        print("Дані очищено успішно")
    
    def parse_json_fields(self):

        def safe_json_parse(x):
            try:
                if pd.isna(x) or x == '':
                    return {}
                return json.loads(x)
            except:
                return {}
        

        self.travels['passengers_parsed'] = self.travels['passengers'].apply(safe_json_parse)
        self.travels['num_passengers'] = self.travels['passengers_parsed'].apply(
            lambda x: len(x.get('users', [])) if isinstance(x, dict) else 0
        )

        self.travels['destination_parsed'] = self.travels['destination'].apply(safe_json_parse)
        self.travels['dest_longitude'] = self.travels['destination_parsed'].apply(
            lambda x: x.get('longitude') if isinstance(x, dict) else None
        )
        self.travels['dest_latitude'] = self.travels['destination_parsed'].apply(
            lambda x: x.get('latitude') if isinstance(x, dict) else None
        )
        
s
        self.travels['offers_parsed'] = self.travels['offers'].apply(safe_json_parse)
        
        print("JSON поля розпарсено")
    
    def create_temporal_features(self):


        self.offers['createdAt'] = pd.to_datetime(self.offers['createdAt'], errors='coerce')
        self.offers['validUntil'] = pd.to_datetime(self.offers['validUntil'], errors='coerce')

        self.offers['hour'] = self.offers['createdAt'].dt.hour
        self.offers['day_of_week'] = self.offers['createdAt'].dt.dayofweek
        self.offers['month'] = self.offers['createdAt'].dt.month
        self.offers['is_weekend'] = self.offers['day_of_week'].isin([5, 6]).astype(int)
 
        self.offers['hour_sin'] = np.sin(2 * np.pi * self.offers['hour'] / 24)
        self.offers['hour_cos'] = np.cos(2 * np.pi * self.offers['hour'] / 24)
        self.offers['dow_sin'] = np.sin(2 * np.pi * self.offers['day_of_week'] / 7)
        self.offers['dow_cos'] = np.cos(2 * np.pi * self.offers['day_of_week'] / 7)
        
        print("Часові ознаки створено")
    
    def create_spatial_features(self):

        def parse_route_coordinates(route_str):
            try:
                if pd.isna(route_str):
                    return None, None
                route = json.loads(route_str)
                if isinstance(route, list) and len(route) > 0:
                    if isinstance(route[0], dict):
                        return route[0].get('longitude'), route[0].get('latitude')
                return None, None
            except:
                return None, None
        

        coords = self.offers['route'].apply(parse_route_coordinates)
        self.offers['start_longitude'] = [c[0] for c in coords]
        self.offers['start_latitude'] = [c[1] for c in coords]

        self.offers = self.offers.dropna(subset=['start_longitude', 'start_latitude'])
        

        city_center_lat = self.offers['start_latitude'].median()
        city_center_lon = self.offers['start_longitude'].median()
        

        self.offers['distance_to_center'] = self.offers.apply(
            lambda row: geodesic(
                (row['start_latitude'], row['start_longitude']),
                (city_center_lat, city_center_lon)
            ).kilometers if pd.notna(row['start_latitude']) else np.nan,
            axis=1
        )
        
        print("Просторові ознаки створено")
    
    def encode_categorical_features(self):


        categorical_cols = ['brand', 'model', 'colour']
        
        for col in categorical_cols:
            if col in self.cars.columns:
                le = LabelEncoder()
                self.cars[f'{col}_encoded'] = le.fit_transform(self.cars[col].astype(str))
                self.label_encoders[col] = le
        
        print("Категоріальні ознаки закодовано")
    
    def create_demand_features(self):


        self.offers['hour_location'] = (
            self.offers['hour'].astype(str) + '_' + 
            self.offers['start_latitude'].round(2).astype(str) + '_' + 
            self.offers['start_longitude'].round(2).astype(str)
        )
        

        demand_data = self.offers.groupby(['hour', 'day_of_week']).agg({
            'id': 'count',
            'pricePerKm': 'mean',
            'pricePerMin': 'mean',
            'reward': 'mean',
            'distance_to_center': 'mean'
        }).reset_index()
        
        demand_data.columns = [
            'hour', 'day_of_week', 'demand_count', 
            'avg_price_km', 'avg_price_min', 'avg_reward', 'avg_distance'
        ]
        
        self.demand_features = demand_data
        print("Ознаки попиту створено")
    
    def create_ranking_features(self):

        def extract_car_info(car_str):
            try:
                if pd.isna(car_str):
                    return None
                car_data = json.loads(car_str)
                return car_data.get('id') if isinstance(car_data, dict) else None
            except:
                return None
        
        self.offers['car_id'] = self.offers['car'].apply(extract_car_info)
        

        self.ranking_data = self.offers.merge(
            self.cars[['id', 'brand_encoded', 'model_encoded', 'seats', 'year']], 
            left_on='car_id', 
            right_on='id', 
            how='left'
        )
        
        print("Ознаки ранжування створено")
    
    def prepare_datasets(self):

        demand_features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'avg_price_km', 'avg_price_min', 'avg_distance'
        ]
        
        self.X_demand = self.demand_features[demand_features].fillna(0)
        self.y_demand = self.demand_features['demand_count']
        

        ranking_features = [
            'seats', 'year', 'pricePerKm', 'pricePerMin', 'reward',
            'distance_to_center', 'hour', 'day_of_week'
        ]
        
        self.X_ranking = self.ranking_data[ranking_features].fillna(0)
        

        self.y_ranking = (self.ranking_data['reward'] > self.ranking_data['reward'].median()).astype(int)
        
        print("Датасети підготовлено")
        print(f"Demand dataset: {self.X_demand.shape}")
        print(f"Ranking dataset: {self.X_ranking.shape}")
    
    def process_all(self):

        if not self.load_data():
            return False
        
        self.clean_data()
        self.parse_json_fields()
        self.create_temporal_features()
        self.create_spatial_features()
        self.encode_categorical_features()
        self.create_demand_features()
        self.create_ranking_features()
        self.prepare_datasets()
        
        return True
    
    def save_processed_data(self):

        self.X_demand.to_csv('data/processed_demand_features.csv', index=False)
        self.X_ranking.to_csv('data/processed_ranking_features.csv', index=False)
        

        pd.Series(self.y_demand).to_csv('data/demand_labels.csv', index=False)
        pd.Series(self.y_ranking).to_csv('data/ranking_labels.csv', index=False)
        
        print("Оброблені дані збережено")