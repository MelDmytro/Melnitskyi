#!/usr/bin/env python3


import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


from src.data_processing import DataProcessor
from src.xgboost_predictor import DemandPredictor, CarRanker, ModelEvaluator, HyperparameterTuner
from src.utils import Logger, ConfigManager, PerformanceProfiler, MetricsCalculator, timing_decorator


@timing_decorator
def load_configuration():

    config_manager = ConfigManager('config.yaml')
    return config_manager.config


@timing_decorator
def prepare_data(config):

    print("=== ЕТАП 1: ПІДГОТОВКА ДАНИХ ===")
    

    data_processor = DataProcessor(config)
    

    if not data_processor.process_all():
        raise Exception("Помилка при обробці даних")
    
    print(f"Підготовлено датасетів:")
    print(f"  - Прогнозування попиту: {data_processor.X_demand.shape}")
    print(f"  - Ранжування автомобілів: {data_processor.X_ranking.shape}")
    
    return data_processor


@timing_decorator
def train_demand_model(data_processor, config, logger):

    print("\n=== ЕТАП 2: ТРЕНУВАННЯ МОДЕЛІ ПОПИТУ ===")
    

    demand_model = DemandPredictor(config)
    

    logger.log_model_training("DemandPredictor", "Training started")
    mae, rmse, mape = demand_model.train(data_processor.X_demand, data_processor.y_demand)
    

    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    logger.log_model_training("DemandPredictor", metrics)
    

    os.makedirs('data/models', exist_ok=True)
    demand_model.save_model('data/models/demand_model.pkl')
    

    demand_model.plot_feature_importance()
    
    return demand_model, metrics


@timing_decorator
def train_ranking_model(data_processor, config, logger):

    print("\n=== ЕТАП 3: ТРЕНУВАННЯ МОДЕЛІ РАНЖУВАННЯ ===")

    ranking_model = CarRanker(config)
    

    logger.log_model_training("CarRanker", "Training started")
    precision, recall, f1 = ranking_model.train(data_processor.X_ranking, data_processor.y_ranking)
    

    metrics = {'Precision': precision, 'Recall': recall, 'F1': f1}
    logger.log_model_training("CarRanker", metrics)
    

    ranking_model.save_model('data/models/ranking_model.pkl')
    

    ranking_model.plot_feature_importance()
    
    return ranking_model, metrics


@timing_decorator
def evaluate_models(demand_model, ranking_model, data_processor):

    print("\n=== ЕТАП 4: ОЦІНКА МОДЕЛЕЙ ===")

    print("Оцінка моделі прогнозування попиту:")
    demand_eval = ModelEvaluator.evaluate_demand_model(
        demand_model, data_processor.X_demand, data_processor.y_demand
    )
    

    print("\nОцінка моделі ранжування:")
    ranking_eval = ModelEvaluator.evaluate_ranking_model(
        ranking_model, data_processor.X_ranking, data_processor.y_ranking
    )
    
    return demand_eval, ranking_eval


def hyperparameter_tuning(data_processor):
"
    print("\n=== ЕТАП 5: ПІДБІР ГІПЕРПАРАМЕТРІВ ===")
    print("Увага: Цей процес може зайняти багато часу!")
    
    choice = input("Виконати підбір гіперпараметрів? (y/n): ").lower()
    
    if choice == 'y':
        print("Підбір параметрів для моделі попиту...")
        demand_best_params = HyperparameterTuner.tune_demand_model(
            data_processor.X_demand, data_processor.y_demand
        )
        
        print("\nПідбір параметрів для моделі ранжування...")
        ranking_best_params = HyperparameterTuner.tune_ranking_model(
            data_processor.X_ranking, data_processor.y_ranking
        )
        
        return demand_best_params, ranking_best_params
    
    return None, None


def save_training_summary(demand_metrics, ranking_metrics, demand_eval, ranking_eval):
"""
    from src.utils import DataExporter
    
    print("\n=== ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ ===")
    

    training_summary = {
        'timestamp': datetime.now().isoformat(),
        'demand_model': {
            'training_metrics': demand_metrics,
            'evaluation_metrics': demand_eval
        },
        'ranking_model': {
            'training_metrics': ranking_metrics,
            'evaluation_metrics': ranking_eval
        }
    }
    

    DataExporter.export_recommendations_to_json(
        training_summary, 'data/models/training_summary.json'
    )
    
    # Збереження метрик в CSV
    all_metrics = {
        'Demand_MAE': [demand_metrics['MAE']],
        'Demand_RMSE': [demand_metrics['RMSE']],
        'Demand_MAPE': [demand_metrics['MAPE']],
        'Ranking_Precision': [ranking_metrics['Precision']],
        'Ranking_Recall': [ranking_metrics['Recall']],
        'Ranking_F1': [ranking_metrics['F1']]
    }
    
    DataExporter.export_metrics_to_csv(all_metrics, 'data/models/metrics.csv')
    
    print("Результати тренування збережено")


def create_test_data_if_needed():

    required_files = ['data/cars.csv', 'data/users.csv', 'data/travels.csv', 'data/offers.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Відсутні файли: {missing_files}")
        print("Створення тестових даних...")
        
        from src.utils import create_sample_data
        create_sample_data()
        

        if not os.path.exists('data/travels.csv'):
            travels_data = []
            for i in range(1000):
                travels_data.append({
                    'id': f'travel_{i}',
                    'passengers': f'{{"users": ["user_{i%50}"]}',
                    'offers': f'{{"id": "offer_{i}"}}',
                    'destination': f'{{"longitude": {37.4 + np.random.uniform(-0.1, 0.1)}, "latitude": {-6.0 + np.random.uniform(-0.1, 0.1)}}}'
                })
            pd.DataFrame(travels_data).to_csv('data/travels.csv', index=False)
        
        if not os.path.exists('data/offers.csv'):
            offers_data = []
            for i in range(2000):
                offers_data.append({
                    'id': f'offer_{i}',
                    'route': f'[{{"longitude": {37.4 + np.random.uniform(-0.1, 0.1)}, "latitude": {-6.0 + np.random.uniform(-0.1, 0.1)}}}]',
                    'car': f'{{"id": "car_{i%100}"}}',
                    'pricePerKm': round(np.random.uniform(5, 15), 2),
                    'pricePerMin': round(np.random.uniform(1, 5), 2),
                    'createdAt': f'2024-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}T{np.random.randint(0,24):02d}:00:00Z',
                    'validUntil': f'2024-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}T{np.random.randint(0,24):02d}:00:00Z',
                    'reward': round(np.random.uniform(0, 20), 1)
                })
            pd.DataFrame(offers_data).to_csv('data/offers.csv', index=False)
        
        print("Тестові дані створено успішно")


def main():

    print(" СИСТЕМА АДАПТИВНОГО ПІДБОРУ АВТОМОБІЛІВ КАРШЕРИНГУ")
    print(" Початок процесу тренування моделей XGBoost")
    print("=" * 60)
    

    logger = Logger('training.log')
    profiler = PerformanceProfiler()
    
    try:

        os.makedirs('data', exist_ok=True)
        create_test_data_if_needed()
        
        profiler.start_timer('total_training')
        

        profiler.start_timer('load_config')
        config = load_configuration()
        profiler.end_timer('load_config')
        

        profiler.start_timer('data_preparation')
        data_processor = prepare_data(config)
        profiler.end_timer('data_preparation')
        

        profiler.start_timer('train_demand')
        demand_model, demand_metrics = train_demand_model(data_processor, config, logger)
        profiler.end_timer('train_demand')
        

        profiler.start_timer('train_ranking')
        ranking_model, ranking_metrics = train_ranking_model(data_processor, config, logger)
        profiler.end_timer('train_ranking')
        

        profiler.start_timer('evaluation')
        demand_eval, ranking_eval = evaluate_models(demand_model, ranking_model, data_processor)
        profiler.end_timer('evaluation')
        

        hyperparameter_tuning(data_processor)
        
     
        save_training_summary(demand_metrics, ranking_metrics, demand_eval, ranking_eval)
        
        profiler.end_timer('total_training')

        profiler.print_report()
        
        print("\n ТРЕНУВАННЯ ЗАВЕРШЕНО УСПІШНО!")
        print(f" Моделі збережено в папці: data/models/")
        print(f" Метрики збережено в: data/models/metrics.csv")
        print(f" Детальний звіт: data/models/training_summary.json")
        
    except Exception as e:
        logger.log_error("Помилка під час тренування", e)
        print(f" ПОМИЛКА: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)