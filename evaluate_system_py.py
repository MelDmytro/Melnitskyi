#!/usr/bin/env python3


import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score

from src.data_processing import DataProcessor
from src.xgboost_predictor import DemandPredictor, CarRanker, ModelEvaluator
from src.car_optimizer import CarMatcher, ScheduleOptimizer, RecommendationEngine
from src.utils import (Logger, ConfigManager, MetricsCalculator, Visualizer, 
                      PerformanceProfiler, DataExporter, timing_decorator)


class SystemEvaluator:

    
    def __init__(self, config):
        self.config = config
        self.logger = Logger('evaluation.log')
        self.profiler = PerformanceProfiler()
        self.results = {}
        
    def load_models(self):

        print("=== ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ===")
        
        try:

            self.demand_model = DemandPredictor(self.config)
            self.demand_model.load_model('data/models/demand_model.pkl')
            

            self.ranking_model = CarRanker(self.config)
            self.ranking_model.load_model('data/models/ranking_model.pkl')
            

            self.car_matcher = CarMatcher(self.config, self.demand_model, self.ranking_model)
            self.recommendation_engine = RecommendationEngine(self.car_matcher)
            
            print(" Всі моделі завантажено успішно")
            return True
            
        except Exception as e:
            print(f" Помилка завантаження моделей: {e}")
            return False
    
    @timing_decorator
    def evaluate_demand_prediction(self, data_processor):

        print("\n=== ОЦІНКА ПРОГНОЗУВАННЯ ПОПИТУ ===")
        
        self.profiler.start_timer('demand_evaluation')
        

        X_test = data_processor.X_demand
        y_test = data_processor.y_demand
        

        y_pred = self.demand_model.predict(X_test)
        

        metrics = MetricsCalculator.calculate_demand_metrics(y_test, y_pred)
        

        self.visualize_demand_results(y_test, y_pred)
        
        self.profiler.end_timer('demand_evaluation')
        
        print(f"Метрики прогнозування попиту:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        self.results['demand_prediction'] = metrics
        return metrics
    
    @timing_decorator
    def evaluate_car_ranking(self, data_processor):

        print("\n=== ОЦІНКА РАНЖУВАННЯ АВТОМОБІЛІВ ===")
        
        self.profiler.start_timer('ranking_evaluation')
        

        X_test = data_processor.X_ranking
        y_test = data_processor.y_ranking
        

        y_scores = self.ranking_model.rank_cars(X_test)
        y_pred = self.ranking_model.predict(X_test)
        

        ranking_metrics = MetricsCalculator.calculate_ranking_metrics(y_test, y_scores, k=5)
        

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        ranking_metrics.update({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        

        self.visualize_ranking_results(y_test, y_scores, y_pred)
        
        self.profiler.end_timer('ranking_evaluation')
        
        print(f"Метрики ранжування автомобілів:")
        for metric, value in ranking_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        self.results['car_ranking'] = ranking_metrics
        return ranking_metrics
    
    @timing_decorator
    def evaluate_recommendation_system(self):

        print("\n=== ОЦІНКА СИСТЕМИ РЕКОМЕНДАЦІЙ ===")
        
        self.profiler.start_timer('recommendation_evaluation')
        
        # Тестові сценарії
        test_scenarios = [
            {
                'location': (49.4200, 27.0058),  # Хмельницький
                'preferences': {'preferred_seats': 4, 'max_price_per_km': 10},
                'description': 'Стандартний запит'
            },
            {
                'location': (50.4501, 30.5234),  # Київ
                'preferences': {'preferred_seats': 7, 'max_price_per_km': 15},
                'description': 'Великий автомобіль'
            },
            {
                'location': (46.4825, 30.7233),  # Одеса
                'preferences': {'preferred_seats': 2, 'max_price_per_km': 8},
                'description': 'Економ клас'
            }
        ]
        
        recommendation_results = []
        
        for i, scenario in enumerate(test_scenarios):
            print(f"Тестування сценарію {i+1}: {scenario['description']}")
            
            start_time = time.time()
            

            recommendations = self.recommendation_engine.generate_recommendations(
                f'test_user_{i}',
                scenario['location'],
                scenario['preferences']
            )
            
            response_time = time.time() - start_time
            
            result = {
                'scenario': scenario['description'],
                'location': scenario['location'],
                'recommendations_count': len(recommendations['recommendations']),
                'response_time': response_time,
                'avg_score': np.mean([car.get('final_score', 0) for car in recommendations['recommendations']]) if recommendations['recommendations'] else 0,
                'demand_info': recommendations['demand_info']
            }
            
            recommendation_results.append(result)
            print(f"  Знайдено {result['recommendations_count']} автомобілів за {result['response_time']:.3f}s")
        

        avg_response_time = np.mean([r['response_time'] for r in recommendation_results])
        avg_recommendations = np.mean([r['recommendations_count'] for r in recommendation_results])
        
        recommendation_metrics = {
            'avg_response_time': avg_response_time,
            'avg_recommendations_count': avg_recommendations,
            'scenarios_tested': len(test_scenarios)
        }
        
        self.profiler.end_timer('recommendation_evaluation')
        
        print(f"\nМетрики системи рекомендацій:")
        print(f"  Середній час відгуку: {avg_response_time:.3f}s")
        print(f"  Середня кількість рекомендацій: {avg_recommendations:.1f}")
        
        self.results['recommendation_system'] = {
            'metrics': recommendation_metrics,
            'scenarios': recommendation_results
        }
        
        return recommendation_metrics
    
    @timing_decorator
    def evaluate_optimization_algorithm(self, data_processor):

        print("\n=== ОЦІНКА АЛГОРИТМУ ОПТИМІЗАЦІЇ ===")
        
        self.profiler.start_timer('optimization_evaluation')
        

        optimizer = ScheduleOptimizer(self.config)
        

        cars_data = pd.DataFrame({
            'id': [f'car_{i}' for i in range(20)],
            'brand': np.random.choice(['Toyota', 'Honda', 'BMW'], 20),
            'available': [True] * 20
        })
        
        users_data = pd.DataFrame({
            'id': range(1, 31),
            'name': [f'User_{i}' for i in range(1, 31)]
        })
        

        print("Запуск еволюційної оптимізації...")
        best_schedule, fitness_history = optimizer.optimize_schedule(cars_data, users_data)
        

        optimization_metrics = {
            'best_fitness': best_schedule['fitness'],
            'generations': len(fitness_history),
            'improvement': fitness_history[-1] - fitness_history[0] if len(fitness_history) > 1 else 0,
            'convergence_generation': self.find_convergence_point(fitness_history)
        }
        

        Visualizer.plot_optimization_progress(fitness_history, "Прогрес еволюційної оптимізації")
        
        self.profiler.end_timer('optimization_evaluation')
        
        print(f"Метрики оптимізації:")
        print(f"  Найкраща пристосованість: {optimization_metrics['best_fitness']:.2f}")
        print(f"  Поколінь: {optimization_metrics['generations']}")
        print(f"  Покращення: {optimization_metrics['improvement']:.2f}")
        
        self.results['optimization'] = optimization_metrics
        return optimization_metrics
    
    def compare_with_baseline_methods(self, data_processor):

        print("\n=== ПОРІВНЯННЯ З БАЗОВИМИ МЕТОДАМИ ===")
        

        baseline_random = self.evaluate_random_selection()
        

        baseline_distance = self.evaluate_distance_based_selection()
        

        baseline_price = self.evaluate_price_based_selection()
        

        our_method = {
            'demand_mae': self.results['demand_prediction']['MAE'],
            'ranking_f1': self.results['car_ranking']['F1-Score'],
            'response_time': self.results['recommendation_system']['metrics']['avg_response_time']
        }
        

        comparison = {
            'Random Selection': baseline_random,
            'Distance-Based': baseline_distance,
            'Price-Based': baseline_price,
            'Our XGBoost Method': our_method
        }
        