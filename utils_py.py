
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import json
import os
import yaml
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:

    
    @staticmethod
    def calculate_demand_metrics(y_true, y_pred):

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-7))) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }
    
    @staticmethod
    def calculate_ranking_metrics(y_true, y_scores, k=5):

        # Precision@k
        top_k_indices = np.argsort(y_scores)[-k:]
        precision_at_k = np.sum(y_true[top_k_indices]) / k
        
        # Recall@k
        total_relevant = np.sum(y_true)
        recall_at_k = np.sum(y_true[top_k_indices]) / max(total_relevant, 1)
        
        # NDCG@k
        dcg = 0
        idcg = 0
        
        for i, idx in enumerate(top_k_indices):
            rank = i + 1
            dcg += (2 ** y_true[idx] - 1) / np.log2(rank + 1)
        
        sorted_true = np.sort(y_true)[::-1][:k]
        for i, relevance in enumerate(sorted_true):
            rank = i + 1
            idcg += (2 ** relevance - 1) / np.log2(rank + 1)
        
        ndcg = dcg / max(idcg, 1e-7)
        
        return {
            'Precision@k': precision_at_k,
            'Recall@k': recall_at_k,
            'NDCG@k': ndcg
        }
    
    @staticmethod
    def calculate_business_metrics(recommendations_data):

        if not recommendations_data:
            return {}
        

        total_recommendations = len(recommendations_data)
        accepted_recommendations = sum(1 for r in recommendations_data if r.get('accepted', False))
        conversion_rate = accepted_recommendations / max(total_recommendations, 1)
        

        response_times = [r.get('response_time', 0) for r in recommendations_data]
        avg_response_time = np.mean(response_times)
        

        satisfaction_scores = [r.get('user_rating', 0) for r in recommendations_data if r.get('user_rating')]
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0
        
        return {
            'conversion_rate': conversion_rate,
            'avg_response_time': avg_response_time,
            'avg_satisfaction': avg_satisfaction,
            'total_recommendations': total_recommendations
        }


class Visualizer:

    
    @staticmethod
    def plot_demand_forecast(historical_data, predictions, title="Прогноз попиту"):

        plt.figure(figsize=(12, 6))
        

        plt.plot(historical_data.index, historical_data.values, 
                label='Історичні дані', color='blue', alpha=0.7)
        

        plt.plot(predictions.index, predictions.values, 
                label='Прогноз', color='red', linestyle='--', alpha=0.8)
        
        plt.title(title)
        plt.xlabel('Час')
        plt.ylabel('Попит')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance_df, title="Важливість ознак"):

        plt.figure(figsize=(10, 8))
        

        top_features = importance_df.head(15)
        
        sns.barplot(data=top_features, y='feature', x='importance', 
                   palette='viridis')
        plt.title(title)
        plt.xlabel('Важливість')
        plt.ylabel('Ознаки')

        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_spatial_distribution(cars_data, user_location=None, title="Розподіл автомобілів"):

        fig = go.Figure()
        

        fig.add_trace(go.Scattermapbox(
            lat=cars_data['latitude'],
            lon=cars_data['longitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=cars_data.get('final_score', [0.5] * len(cars_data)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Рейтинг")
            ),
            text=[f"ID: {car['car_id']}<br>Рейтинг: {car.get('final_score', 0):.2f}<br>"
                  f"Відстань: {car.get('distance', 0):.1f} км" 
                  for _, car in cars_data.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name='Автомобілі'
        ))
        

        if user_location:
            fig.add_trace(go.Scattermapbox(
                lat=[user_location[0]],
                lon=[user_location[1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                text=['Ваше місцезнаходження'],
                name='Користувач'
            ))
        

        center_lat = cars_data['latitude'].mean()
        center_lon = cars_data['longitude'].mean()
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            title=title,
            height=600
        )
        
        fig.show()
    
    @staticmethod
    def plot_optimization_progress(fitness_history, title="Прогрес оптимізації"):

        plt.figure(figsize=(10, 6))
        
        plt.plot(fitness_history, linewidth=2, color='blue')
        plt.title(title)
        plt.xlabel('Покоління')
        plt.ylabel('Найкраща пристосованість')
        plt.grid(True, alpha=0.3)
        

        best_fitness = max(fitness_history)
        best_generation = fitness_history.index(best_fitness)
        
        plt.axhline(y=best_fitness, color='red', linestyle='--', alpha=0.7, 
                   label=f'Найкращий результат: {best_fitness:.2f}')
        plt.axvline(x=best_generation, color='green', linestyle='--', alpha=0.7,
                   label=f'Покоління: {best_generation}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict, title="Порівняння метрик"):

        metrics_df = pd.DataFrame(metrics_dict).T
        
        fig, axes = plt.subplots(1, len(metrics_df.columns), figsize=(15, 5))
        if len(metrics_df.columns) == 1:
            axes = [axes]
        
        for i, column in enumerate(metrics_df.columns):
            metrics_df[column].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'{column}')
            axes[i].set_ylabel('Значення')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_time_series_analysis(data, date_column, value_column, title="Аналіз часових рядів"):

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        

        axes[0, 0].plot(data[date_column], data[value_column])
        axes[0, 0].set_title('Часовий ряд')
        axes[0, 0].set_xlabel('Дата')
        axes[0, 0].set_ylabel('Значення')
        

        hourly_avg = data.groupby(data[date_column].dt.hour)[value_column].mean()
        axes[0, 1].bar(hourly_avg.index, hourly_avg.values)
        axes[0, 1].set_title('Середні значення по годинах')
        axes[0, 1].set_xlabel('Година дня')
        axes[0, 1].set_ylabel('Середнє значення')
        

        daily_avg = data.groupby(data[date_column].dt.dayofweek)[value_column].mean()
        days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд']
        axes[1, 0].bar(range(len(daily_avg)), daily_avg.values)
        axes[1, 0].set_xticks(range(len(days)))
        axes[1, 0].set_xticklabels(days)
        axes[1, 0].set_title('Середні значення по днях тижня')
        axes[1, 0].set_ylabel('Середнє значення')
        

        axes[1, 1].hist(data[value_column], bins=30, alpha=0.7)
        axes[1, 1].set_title('Розподіл значень')
        axes[1, 1].set_xlabel('Значення')
        axes[1, 1].set_ylabel('Частота')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


class Logger:

    
    def __init__(self, log_file='carsharing_system.log'):
        self.logger = logging.getLogger('CarSharingSystem')
        self.logger.setLevel(logging.INFO)
        

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_recommendation(self, user_id, recommendations, execution_time):

        self.logger.info(
            f"Recommendation generated for user {user_id}: "
            f"{len(recommendations)} cars found in {execution_time:.3f}s"
        )
    
    def log_model_training(self, model_name, metrics):

        self.logger.info(
            f"Model {model_name} trained with metrics: {metrics}"
        )
    
    def log_optimization(self, generation, fitness, execution_time):

        self.logger.info(
            f"Optimization generation {generation}: "
            f"fitness={fitness:.3f}, time={execution_time:.3f}s"
        )
    
    def log_error(self, error_message, exception=None):

        if exception:
            self.logger.error(f"{error_message}: {str(exception)}")
        else:
            self.logger.error(error_message)
    
    def log_info(self, message):

        self.logger.info(message)
    
    def log_warning(self, message):

        self.logger.warning(message)


class DataValidator:

    
    @staticmethod
    def validate_user_location(location):

        if not isinstance(location, (list, tuple)) or len(location) != 2:
            return False, "Локація повинна бути списком з двох координат"
        
        lat, lon = location
        if not (-90 <= lat <= 90):
            return False, "Широта повинна бути в діапазоні [-90, 90]"
        
        if not (-180 <= lon <= 180):
            return False, "Довгота повинна бути в діапазоні [-180, 180]"
        
        return True, "OK"
    
    @staticmethod
    def validate_user_preferences(preferences):

        if not isinstance(preferences, dict):
            return False, "Вподобання повинні бути словником"
        
        # Перевірка основних полів
        if 'preferred_seats' in preferences:
            if not isinstance(preferences['preferred_seats'], int) or preferences['preferred_seats'] < 1:
                return False, "Кількість місць повинна бути позитивним цілим числом"
        
        if 'max_price_per_km' in preferences:
            if not isinstance(preferences['max_price_per_km'], (int, float)) or preferences['max_price_per_km'] <= 0:
                return False, "Максимальна ціна повинна бути позитивним числом"
        
        if 'max_distance' in preferences:
            if not isinstance(preferences['max_distance'], (int, float)) or preferences['max_distance'] <= 0:
                return False, "Максимальна відстань повинна бути позитивним числом"
        
        return True, "OK"
    
    @staticmethod
    def validate_csv_data(df, required_columns):

        errors = []
        

        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Відсутні колонки: {missing_columns}")
        

        if df.empty:
            errors.append("Датафрейм порожній")
        

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Знайдено {duplicates} дублікатів")

        for col in required_columns:
            if col in df.columns:
                missing_values = df[col].isna().sum()
                if missing_values > 0:
                    errors.append(f"Колонка {col} містить {missing_values} відсутніх значень")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_model_input(X, feature_names=None):

        errors = []
        

        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            errors.append("Дані повинні бути DataFrame або numpy array")
            return False, errors
        

        if hasattr(X, 'isna'):
            if X.isna().any().any():
                errors.append("Дані містять відсутні значення")
        elif np.isnan(X).any():
            errors.append("Дані містять NaN значення")
        

        if hasattr(X, 'isin'):
            if X.isin([np.inf, -np.inf]).any().any():
                errors.append("Дані містять нескінченні значення")
        elif np.isinf(X).any():
            errors.append("Дані містять нескінченні значення")
        

        if feature_names and hasattr(X, 'columns'):
            missing_features = set(feature_names) - set(X.columns)
            if missing_features:
                errors.append(f"Відсутні ознаки: {missing_features}")
        
        return len(errors) == 0, errors


class ConfigManager:

    
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):

        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Помилка завантаження конфігурації: {e}")
            return self.get_default_config()
    
    def get_default_config(self):

        return {
            'demand_model': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'ranking_model': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'optimization': {
                'population_size': 50,
                'max_generations': 100,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'elite_size': 10
            },
            'data_processing': {
                'test_size': 0.2,
                'validation_size': 0.2,
                'random_state': 42,
                'spatial_radius_km': 5.0,
                'temporal_window_hours': 24
            },
            'web_app': {
                'host': '127.0.0.1',
                'port': 5000,
                'debug': True
            },
            'paths': {
                'data_dir': 'data/',
                'models_dir': 'data/models/',
                'cars_file': 'data/cars.csv',
                'users_file': 'data/users.csv',
                'travels_file': 'data/travels.csv',
                'offers_file': 'data/offers.csv'
            }
        }
    
    def get(self, key, default=None):

        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, key, value):

        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self):

        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, allow_unicode=True)
            print("Конфігурацію збережено")
        except Exception as e:
            print(f"Помилка збереження конфігурації: {e}")


class PerformanceProfiler:

    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.memory_usage = {}
    
    def start_timer(self, operation_name):
        """Початок вимірювання часу"""
        self.start_times[operation_name] = datetime.now()
    
    def end_timer(self, operation_name):
        """Завершення вимірювання часу"""
        if operation_name in self.start_times:
            duration = (datetime.now() - self.start_times[operation_name]).total_seconds()
            
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            
            self.timings[operation_name].append(duration)
            del self.start_times[operation_name]
            
            return duration
        return None
    
    def measure_memory(self, operation_name):

        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if operation_name not in self.memory_usage:
                self.memory_usage[operation_name] = []
            
            self.memory_usage[operation_name].append(memory_mb)
            return memory_mb
        except ImportError:
            return None
    
    def get_statistics(self):

        stats = {}
        
        for operation, times in self.timings.items():
            stats[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times)
            }
            
            # Додавання статистики пам'яті якщо є
            if operation in self.memory_usage:
                memory_data = self.memory_usage[operation]
                stats[operation].update({
                    'avg_memory_mb': np.mean(memory_data),
                    'max_memory_mb': max(memory_data),
                    'min_memory_mb': min(memory_data)
                })
        
        return stats
    
    def print_report(self):

        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 ЗВІТ ПРОДУКТИВНОСТІ")
        print("=" * 60)
        
        for operation, data in stats.items():
            print(f"\n🔧 Операція: {operation}")
            print(f"  📊 Кількість виконань: {data['count']}")
            print(f"  ⏱️  Загальний час: {format_time_duration(data['total_time'])}")
            print(f"  📈 Середній час: {format_time_duration(data['avg_time'])}")
            print(f"  ⚡ Мін/Макс час: {format_time_duration(data['min_time'])} / {format_time_duration(data['max_time'])}")
            
            if 'avg_memory_mb' in data:
                print(f"  💾 Середня пам'ять: {data['avg_memory_mb']:.1f} MB")
                print(f"  🔺 Макс пам'ять: {data['max_memory_mb']:.1f} MB")
    
    def reset(self):

        self.timings.clear()
        self.start_times.clear()
        self.memory_usage.clear()


class DataExporter:
  
    
    @staticmethod
    def export_recommendations_to_json(recommendations, filename):

        try:
            # Створення директорії якщо не існує
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, ensure_ascii=False, indent=2, default=str)
            print(f" Рекомендації експортовано в {filename}")
        except Exception as e:
            print(f" Помилка експорту: {e}")
    
    @staticmethod
    def export_metrics_to_csv(metrics_dict, filename):

        try:
            # Створення директорії якщо не існує
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            if isinstance(metrics_dict, dict):

                if all(isinstance(v, dict) for v in metrics_dict.values()):
                    metrics_df = pd.DataFrame(metrics_dict).T
                else:
                    metrics_df = pd.DataFrame([metrics_dict])
            else:
                metrics_df = pd.DataFrame(metrics_dict)
            
            metrics_df.to_csv(filename, index=True)
            print(f" Метрики експортовано в {filename}")
        except Exception as e:
            print(f" Помилка експорту метрик: {e}")
    
    @staticmethod
    def export_model_summary(model, feature_importance, filename):

        try:

            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            summary = {
                'model_type': type(model).__name__,
                'timestamp': datetime.now().isoformat(),
                'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else [],
                'model_params': model.get_params() if hasattr(model, 'get_params') else {}
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            print(f" Опис моделі експортовано в {filename}")
        except Exception as e:
            print(f" Помилка експорту опису моделі: {e}")
    
    @staticmethod
    def export_dataframe_to_excel(df, filename, sheet_name='Sheet1'):

        try:

            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            df.to_excel(filename, sheet_name=sheet_name, index=False)
            print(f" Дані експортовано в {filename}")
        except Exception as e:
            print(f" Помилка експорту в Excel: {e}")


def format_time_duration(seconds):

    if seconds < 1:
        return f"{seconds*1000:.0f} мс"
    elif seconds < 60:
        return f"{seconds:.1f} с"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f} хв {secs:.0f} с"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f} год {minutes:.0f} хв"


def create_sample_data():

    print("🔧 Створення тестових даних...")
    

    os.makedirs('data', exist_ok=True)
    
    # Тестові автомобілі
    cars_data = []
    brands = ['Toyota', 'Honda', 'BMW', 'Audi', 'Ford', 'Volkswagen', 'Nissan', 'Hyundai']
    colors = ['white', 'black', 'silver', 'red', 'blue', 'gray', 'green', 'yellow']
    
    for i in range(100):
        cars_data.append({
            'id': f'car_{i}',
            'car licence plate': f'AA{i:04d}BB',
            'brand': np.random.choice(brands),
            'model': f'Model_{i}',
            'colour': np.random.choice(colors),
            'seats': np.random.choice([2, 4, 5, 7], p=[0.1, 0.6, 0.25, 0.05]),
            'year': np.random.randint(2015, 2024),
            'owner': f'Owner_{i}',
            'deleted': np.random.choice(['true', 'false'], p=[0.1, 0.9])
        })
    

    users_data = []
    names = ['Олександр', 'Марія', 'Дмитро', 'Анна', 'Сергій', 'Олена', 'Андрій', 'Ірина']
    
    for i in range(50):
        users_data.append({
            'id': i,
            'uuid': f'user-uuid-{i:04d}',
            'name': f'{np.random.choice(names)}_{i}',
            'tripsTaken': np.random.poisson(20),  # Пуассонівський розподіл
            'rating': round(np.random.beta(8, 2) * 10, 1)  # Бета-розподіл для реалістичних рейтингів
        })
    

    travels_data = []
    base_lat, base_lon = 49.4200, 27.0058  # Хмельницький
    
    for i in range(1000):

        lat_offset = np.random.normal(0, 0.05)  # ~5 км радіус
        lon_offset = np.random.normal(0, 0.05)
        
        travels_data.append({
            'id': f'travel_{i}',
            'passengers': json.dumps({
                "users": [f"user_{np.random.randint(0, 50)}" for _ in range(np.random.randint(1, 4))]
            }),
            'offers': json.dumps({"id": f"offer_{i}"}),
            'pickups': json.dumps([{
                "user": f"user_{np.random.randint(0, 50)}",
                "time": (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                "coordinates": {
                    "longitude": base_lon + lon_offset,
                    "latitude": base_lat + lat_offset
                }
            }]),
            'dropoffs': json.dumps([{
                "user": f"user_{np.random.randint(0, 50)}",
                "time": (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                "coordinates": {
                    "longitude": base_lon + lon_offset + np.random.normal(0, 0.02),
                    "latitude": base_lat + lat_offset + np.random.normal(0, 0.02)
                }
            }]),
            'destination': json.dumps({
                "longitude": base_lon + lon_offset + np.random.normal(0, 0.03),
                "latitude": base_lat + lat_offset + np.random.normal(0, 0.03)
            }),
            'ratings': json.dumps([{
                "user": f"user_{np.random.randint(0, 50)}",
                "rating": round(np.random.beta(8, 2) * 5, 1)
            }]),
            'uuid': f'travel-uuid-{i:04d}',
            'finishes': json.dumps([{
                "user": f"user_{np.random.randint(0, 50)}",
                "time": (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                "state": "completed",
                "coordinates": {
                    "longitude": base_lon + lon_offset,
                    "latitude": base_lat + lat_offset
                }
            }])
        })
    
  
    offers_data = []
    
    for i in range(2000):

        base_price_km = np.random.lognormal(2, 0.3)  # Логнормальний розподіл
        base_price_min = np.random.lognormal(0.5, 0.2)
        

        created_at = datetime.now() - timedelta(
            days=np.random.randint(0, 365),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        valid_until = created_at + timedelta(
            hours=np.random.randint(1, 24),
            minutes=np.random.randint(0, 60)
        )
        

        route_points = []
        current_lat = base_lat + np.random.normal(0, 0.05)
        current_lon = base_lon + np.random.normal(0, 0.05)
        
        num_points = np.random.randint(2, 6)
        for j in range(num_points):
            route_points.append({
                "longitude": current_lon + np.random.normal(0, 0.01),
                "latitude": current_lat + np.random.normal(0, 0.01)
            })
            current_lat += np.random.normal(0, 0.02)
            current_lon += np.random.normal(0, 0.02)
        
        offers_data.append({
            'id': f'offer_{i}',
            'route': json.dumps(route_points),
            'car': json.dumps({"id": f"car_{np.random.randint(0, 100)}"}),
            'pricePerKm': round(base_price_km, 2),
            'pricePerMin': round(base_price_min, 2),
            'createdAt': created_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'validUntil': valid_until.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'reward': round(np.random.exponential(5), 1)  # Експоненційний розподіл для винагород
        })
    

    try:
        pd.DataFrame(cars_data).to_csv('data/cars.csv', index=False)
        pd.DataFrame(users_data).to_csv('data/users.csv', index=False)
        pd.DataFrame(travels_data).to_csv('data/travels.csv', index=False)
        pd.DataFrame(offers_data).to_csv('data/offers.csv', index=False)
        
        print(" Тестові дані створено успішно:")
        print(f"   Автомобілі: {len(cars_data)} записів")
        print(f"   Користувачі: {len(users_data)} записів")
        print(f"   Поїздки: {len(travels_data)} записів")
        print(f"   Пропозиції: {len(offers_data)} записів")
        
    except Exception as e:
        print(f" Помилка створення тестових даних: {e}")


def analyze_data_quality(data_dir='data/'):

    print(" Аналіз якості даних...")
    
    files = ['cars.csv', 'users.csv', 'travels.csv', 'offers.csv']
    quality_report = {}
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Основна статистика
                report = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_values': df.isnull().sum().sum(),
                    'duplicate_rows': df.duplicated().sum(),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                
                # Аналіз колонок
                column_analysis = {}
                for col in df.columns:
                    col_data = df[col]
                    column_analysis[col] = {
                        'dtype': str(col_data.dtype),
                        'missing': col_data.isnull().sum(),
                        'unique_values': col_data.nunique(),
                        'missing_percentage': round(col_data.isnull().sum() / len(df) * 100, 2)
                    }
                
                report['columns_analysis'] = column_analysis
                quality_report[file] = report
                
                print(f" {file}: {report['rows']} рядків, {report['missing_values']} відсутніх значень")
                
            except Exception as e:
                print(f" Помилка аналізу {file}: {e}")
                quality_report[file] = {'error': str(e)}
        else:
            print(f"  Файл {file} не знайдено")
            quality_report[file] = {'error': 'File not found'}
    
    return quality_report


def generate_synthetic_features(df, feature_type='temporal'):

    if feature_type == 'temporal' and 'createdAt' in df.columns:

        df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')
        

        df['hour'] = df['createdAt'].dt.hour
        df['day_of_week'] = df['createdAt'].dt.dayofweek
        df['month'] = df['createdAt'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
    elif feature_type == 'spatial' and all(col in df.columns for col in ['latitude', 'longitude']):

        center_lat = df['latitude'].median()
        center_lon = df['longitude'].median()
        

        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - center_lat) ** 2 + 
            (df['longitude'] - center_lon) ** 2
        ) * 111  # Приблизне перетворення в км
        

        df['quadrant'] = 0
        df.loc[(df['latitude'] >= center_lat) & (df['longitude'] >= center_lon), 'quadrant'] = 1
        df.loc[(df['latitude'] >= center_lat) & (df['longitude'] < center_lon), 'quadrant'] = 2
        df.loc[(df['latitude'] < center_lat) & (df['longitude'] < center_lon), 'quadrant'] = 3
        df.loc[(df['latitude'] < center_lat) & (df['longitude'] >= center_lon), 'quadrant'] = 4
    
    return df


def calculate_system_metrics(predictions, actuals, metric_type='regression'):

    if metric_type == 'regression':
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {
            'MAE': mean_absolute_error(actuals, predictions),
            'MSE': mean_squared_error(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'R2': r2_score(actuals, predictions),
            'MAPE': np.mean(np.abs((actuals - predictions) / np.maximum(np.abs(actuals), 1e-7))) * 100
        }
        
    elif metric_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'Accuracy': accuracy_score(actuals, predictions),
            'Precision': precision_score(actuals, predictions, average='weighted'),
            'Recall': recall_score(actuals, predictions, average='weighted'),
            'F1': f1_score(actuals, predictions, average='weighted')
        }
    
    return metrics


def timing_decorator(func):

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"  Функція {func.__name__} виконалась за {format_time_duration(duration)}")
        return result
    
    return wrapper



def log_calls(logger):
    """Декоратор для логування викликів функцій"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log_info(f"Виклик функції {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.log_info(f"Функція {func.__name__} виконана успішно")
                return result
            except Exception as e:
                logger.log_error(f"Помилка в функції {func.__name__}", e)
                raise
        return wrapper
    return decorator


class PerformanceContext:
    
    def __init__(self, profiler, operation_name):
        self.profiler = profiler
        self.operation_name = operation_name
    
    def __enter__(self):
        self.profiler.start_timer(self.operation_name)
        self.profiler.measure_memory(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = self.profiler.end_timer(self.operation_name)
        if duration:
            print(f"  {self.operation_name}: {format_time_duration(duration)}")


class DataPipeline:

    
    def __init__(self, steps=None):
        self.steps = steps or []
        self.results = {}
        
    def add_step(self, name, func, **kwargs):

        self.steps.append({
            'name': name,
            'function': func,
            'kwargs': kwargs
        })
    
    def execute(self, data):

        current_data = data
        
        for step in self.steps:
            print(f" Виконання кроку: {step['name']}")
            try:
                start_time = datetime.now()
                current_data = step['function'](current_data, **step['kwargs'])
                duration = (datetime.now() - start_time).total_seconds()
                
                self.results[step['name']] = {
                    'status': 'success',
                    'duration': duration,
                    'data_shape': getattr(current_data, 'shape', len(current_data) if hasattr(current_data, '__len__') else 'unknown')
                }
                
                print(f" Крок {step['name']} виконано за {format_time_duration(duration)}")
                
            except Exception as e:
                print(f" Помилка в кроці {step['name']}: {e}")
                self.results[step['name']] = {
                    'status': 'error',
                    'error': str(e)
                }
                break
        
        return current_data
    
    def get_report(self):

        total_time = sum(
            step.get('duration', 0) 
            for step in self.results.values() 
            if step.get('status') == 'success'
        )
        
        report = {
            'total_steps': len(self.steps),
            'successful_steps': len([s for s in self.results.values() if s.get('status') == 'success']),
            'failed_steps': len([s for s in self.results.values() if s.get('status') == 'error']),
            'total_execution_time': total_time,
            'step_details': self.results
        }
        
        return report


def setup_logging(log_level='INFO', log_file=None):


    level = getattr(logging, log_level.upper(), logging.INFO)
    

    log_config = {
        'level': level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        log_config['filename'] = log_file
        log_config['filemode'] = 'a'
        log_config['encoding'] = 'utf-8'
    
    logging.basicConfig(**log_config)
    

    logger = logging.getLogger('CarSharingSystem')
    
    return logger


def create_project_structure():

    directories = [
        'data',
        'data/models',
        'data/processed',
        'logs',
        'results',
        'plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Створено директорію: {directory}")
    
    print(" Структура проєкту створена")


def main():

    print("  ДЕМОНСТРАЦІЯ УТИЛІТ СИСТЕМИ КАРШЕРИНГУ")
    print("=" * 60)
    

    create_project_structure()
    

    create_sample_data()
    

    quality_report = analyze_data_quality()
    

    DataExporter.export_recommendations_to_json(
        quality_report, 
        'results/data_quality_report.json'
    )
    
    print("\n Демонстрація завершена!")
    print(" Результати збережено в папці results/")


if __name__ == "__main__":
    main()