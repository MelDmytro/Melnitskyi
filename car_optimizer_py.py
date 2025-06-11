
import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class CarMatcher:

    
    def __init__(self, config, demand_model, ranking_model):
        self.config = config
        self.demand_model = demand_model
        self.ranking_model = ranking_model
        
    def find_available_cars(self, user_location, radius_km=5, current_time=None):

        if current_time is None:
            current_time = datetime.now()
        

        available_cars = []
        
        for i in range(20):  # Симуляція 20 доступних автомобілів
            # Випадкові координати в радіусі
            lat_offset = random.uniform(-radius_km/111, radius_km/111)  # 1 градус ≈ 111 км
            lon_offset = random.uniform(-radius_km/111, radius_km/111)
            
            car_location = (
                user_location[0] + lat_offset,
                user_location[1] + lon_offset
            )
            
            distance = geodesic(user_location, car_location).kilometers
            
            if distance <= radius_km:
                available_cars.append({
                    'car_id': f'car_{i}',
                    'latitude': car_location[0],
                    'longitude': car_location[1],
                    'distance': distance,
                    'brand': random.choice(['Toyota', 'Honda', 'BMW', 'Audi', 'Ford']),
                    'model': f'Model_{i}',
                    'seats': random.choice([2, 4, 5, 7]),
                    'year': random.randint(2015, 2023),
                    'rating': round(random.uniform(3.5, 5.0), 1),
                    'price_per_km': round(random.uniform(5, 15), 2),
                    'price_per_min': round(random.uniform(1, 5), 2)
                })
        
        return pd.DataFrame(available_cars)
    
    def predict_demand(self, location, time_features):

        demand_features = pd.DataFrame([time_features])
        predicted_demand = self.demand_model.predict(demand_features)[0]
        return max(0, predicted_demand)  # Попит не може бути негативним
    
    def rank_cars_for_user(self, cars_df, user_preferences):

        if cars_df.empty:
            return cars_df
        

        ranking_features = []
        
        for _, car in cars_df.iterrows():
            features = {
                'seats': car['seats'],
                'year': car['year'],
                'pricePerKm': car['price_per_km'],
                'pricePerMin': car['price_per_min'],
                'reward': car['rating'] * 2,  # Використовуємо рейтинг як винагороду
                'distance_to_center': car['distance'],
                'hour': datetime.now().hour,
                'day_of_week': datetime.now().weekday()
            }
            ranking_features.append(features)
        
        ranking_df = pd.DataFrame(ranking_features)
        

        car_scores = self.ranking_model.rank_cars(ranking_df)
        

        cars_df = cars_df.copy()
        cars_df['ml_score'] = car_scores
        

        cars_df['distance_score'] = 1 / (1 + cars_df['distance'])  # Чим ближче, тим краще
        cars_df['price_score'] = 1 / (1 + cars_df['price_per_km'])  # Чим дешевше, тим краще

        cars_df['final_score'] = (
            0.4 * cars_df['ml_score'] +
            0.3 * cars_df['distance_score'] +
            0.2 * cars_df['price_score'] +
            0.1 * (cars_df['rating'] / 5.0)
        )
        
        return cars_df.sort_values('final_score', ascending=False)
    
    def match_car_to_user(self, user_location, user_preferences, max_results=5):


        available_cars = self.find_available_cars(user_location)
        
        if available_cars.empty:
            return [], "Немає доступних автомобілів в радіусі"

        current_time = datetime.now()
        time_features = {
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'month': current_time.month,
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * current_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * current_time.hour / 24),
            'dow_sin': np.sin(2 * np.pi * current_time.weekday() / 7),
            'dow_cos': np.cos(2 * np.pi * current_time.weekday() / 7),
            'avg_price_km': available_cars['price_per_km'].mean(),
            'avg_price_min': available_cars['price_per_min'].mean(),
            'avg_distance': available_cars['distance'].mean()
        }
        
        predicted_demand = self.predict_demand(user_location, time_features)
        

        ranked_cars = self.rank_cars_for_user(available_cars, user_preferences)
        

        top_cars = ranked_cars.head(max_results)
        
        return top_cars.to_dict('records'), f"Прогнозований попит: {predicted_demand:.1f}"


class ScheduleOptimizer:

    
    def __init__(self, config):
        self.config = config['optimization']
        self.population_size = self.config['population_size']
        self.max_generations = self.config['max_generations']
        self.mutation_rate = self.config['mutation_rate']
        self.crossover_rate = self.config['crossover_rate']
        self.elite_size = self.config['elite_size']
        
    def create_initial_schedule(self, cars_data, users_data):

        schedule = {
            'car_assignments': {},
            'time_slots': {},
            'fitness': 0
        }
        

        for car_id in cars_data['id'][:self.population_size]:
            user_id = random.choice(users_data['id'].tolist())
            start_time = random.randint(6, 22)  # Години роботи
            duration = random.randint(1, 8)  # Тривалість в годинах
            
            schedule['car_assignments'][car_id] = {
                'user_id': user_id,
                'start_time': start_time,
                'duration': duration
            }
        
        return schedule
    
    def fitness_function(self, schedule):

        fitness = 0
        

        utilization = len(schedule['car_assignments']) / max(1, self.population_size)
        fitness += utilization * 40
        

        time_conflicts = 0
        time_slots = {}
        
        for car_id, assignment in schedule['car_assignments'].items():
            start = assignment['start_time']
            end = start + assignment['duration']
            
            for hour in range(start, end):
                if hour in time_slots:
                    time_conflicts += 1
                else:
                    time_slots[hour] = car_id
        
        fitness -= time_conflicts * 10
        

        user_counts = {}
        for assignment in schedule['car_assignments'].values():
            user_id = assignment['user_id']
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        if user_counts:
            balance_score = 1 / (1 + np.std(list(user_counts.values())))
            fitness += balance_score * 20
        
        schedule['fitness'] = max(0, fitness)
        return schedule['fitness']
    
    def mutate_schedule(self, schedule, mutation_type='parametric'):

        mutated = schedule.copy()
        mutated['car_assignments'] = schedule['car_assignments'].copy()
        
        if not mutated['car_assignments']:
            return mutated
        
        car_id = random.choice(list(mutated['car_assignments'].keys()))
        
        if mutation_type == 'parametric':

            mutated['car_assignments'][car_id]['start_time'] = random.randint(6, 22)
            mutated['car_assignments'][car_id]['duration'] = random.randint(1, 8)
        else:

            new_user = random.randint(1, 100)  # Випадковий користувач
            mutated['car_assignments'][car_id]['user_id'] = new_user
        
        return mutated
    
    def crossover_schedules(self, parent1, parent2):

        child = {
            'car_assignments': {},
            'time_slots': {},
            'fitness': 0
        }

        all_cars = list(set(list(parent1['car_assignments'].keys()) + 
                           list(parent2['car_assignments'].keys())))
        
        for car_id in all_cars:
            if random.random() < 0.5 and car_id in parent1['car_assignments']:
                child['car_assignments'][car_id] = parent1['car_assignments'][car_id].copy()
            elif car_id in parent2['car_assignments']:
                child['car_assignments'][car_id] = parent2['car_assignments'][car_id].copy()
        
        return child
    
    def optimize_schedule(self, cars_data, users_data):


        population = []
        for _ in range(self.population_size):
            schedule = self.create_initial_schedule(cars_data, users_data)
            self.fitness_function(schedule)
            population.append(schedule)
        
        best_fitness_history = []
        
        for generation in range(self.max_generations):

            population.sort(key=lambda x: x['fitness'], reverse=True)
            
у
            best_fitness = population[0]['fitness']
            best_fitness_history.append(best_fitness)
            

            new_population = population[:self.elite_size].copy()
            

            while len(new_population) < self.population_size:
                # Селекція батьків
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
  
                if random.random() < self.crossover_rate:
                    child = self.crossover_schedules(parent1, parent2)
                else:
                    child = parent1.copy()

                if random.random() < self.mutation_rate:
                    mutation_type = random.choice(['parametric', 'point'])
                    child = self.mutate_schedule(child, mutation_type)

                self.fitness_function(child)
                new_population.append(child)
            
            population = new_population
            

            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
        

        population.sort(key=lambda x: x['fitness'], reverse=True)
        best_schedule = population[0]
        
        return best_schedule, best_fitness_history
    
    def tournament_selection(self, population, tournament_size=3):

        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])


class RecommendationEngine:

    
    def __init__(self, car_matcher):
        self.car_matcher = car_matcher
        
    def generate_recommendations(self, user_id, user_location, user_preferences=None):

        if user_preferences is None:
            user_preferences = {
                'preferred_seats': 4,
                'max_price_per_km': 10,
                'max_distance': 3,
                'preferred_brands': ['Toyota', 'Honda']
            }
        

        recommended_cars, demand_info = self.car_matcher.match_car_to_user(
            user_location, user_preferences, max_results=10
        )
        

        for car in recommended_cars:

            car['estimated_cost'] = car['price_per_km'] * 5 + car['price_per_min'] * 30  # Приблизна поїздка
            

            car['recommendation_score'] = car['final_score'] * 100

            reasons = []
            if car['distance'] < 1:
                reasons.append("Близько до вас")
            if car['price_per_km'] < 8:
                reasons.append("Низька ціна")
            if car['rating'] > 4.5:
                reasons.append("Високий рейтинг")
            if car['seats'] >= user_preferences.get('preferred_seats', 4):
                reasons.append("Достатньо місць")
            
            car['recommendation_reasons'] = reasons
        
        return {
            'user_id': user_id,
            'location': user_location,
            'recommendations': recommended_cars,
            'demand_info': demand_info,
            'timestamp': datetime.now().isoformat()
        }
    
    def explain_recommendation(self, car_data):

        explanation = {
            'car_id': car_data['car_id'],
            'factors': {
                'ml_prediction': f"ML модель оцінила відповідність: {car_data['ml_score']:.2f}",
                'distance': f"Відстань: {car_data['distance']:.1f} км",
                'price': f"Ціна: {car_data['price_per_km']:.2f} грн/км",
                'rating': f"Рейтинг: {car_data['rating']}/5.0",
                'final_score': f"Загальний рейтинг: {car_data['final_score']:.2f}"
            },
            'reasons': car_data.get('recommendation_reasons', [])
        }
        return explanation