#!/usr/bin/env python3


import os
import json
import traceback
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np


from src.data_processing import DataProcessor
from src.xgboost_predictor import DemandPredictor, CarRanker
from src.car_optimizer import CarMatcher, RecommendationEngine
from src.utils import Logger, ConfigManager, DataValidator, PerformanceProfiler



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Глобальні змінні для моделей та даних
demand_model = None
ranking_model = None
car_matcher = None
recommendation_engine = None
config = None
logger = None
profiler = PerformanceProfiler()


def initialize_system():
    """Ініціалізація системи при запуску"""
    global demand_model, ranking_model, car_matcher, recommendation_engine, config, logger
    
    try:

        config_manager = ConfigManager('config.yaml')
        config = config_manager.config
        

        logger = Logger('app.log')
        logger.logger.info("Запуск веб-додатку каршерингу")
        

        if os.path.exists('data/models/demand_model.pkl') and os.path.exists('data/models/ranking_model.pkl'):
            demand_model = DemandPredictor(config)
            demand_model.load_model('data/models/demand_model.pkl')
            
            ranking_model = CarRanker(config)
            ranking_model.load_model('data/models/ranking_model.pkl')
            

            car_matcher = CarMatcher(config, demand_model, ranking_model)
            recommendation_engine = RecommendationEngine(car_matcher)
            
            logger.logger.info("Моделі завантажено успішно")
            return True
        else:
            logger.logger.warning("Моделі не знайдено. Потрібно спочатку виконати тренування.")
            return False
            
    except Exception as e:
        if logger:
            logger.log_error("Помилка ініціалізації системи", e)
        else:
            print(f"Помилка ініціалізації: {e}")
        return False



MAIN_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> Система Адаптивного Підбору Автомобілів Каршерингу</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .search-section {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #2196F3;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .btn {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .results {
            margin-top: 30px;
        }
        
        .car-card {
            background: white;
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .car-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .car-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .car-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #333;
        }
        
        .car-score {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
        }
        
        .car-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .car-detail {
            text-align: center;
            padding: 10px;
            background: #f8f9ff;
            border-radius: 8px;
        }
        
        .car-detail-label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }
        
        .car-detail-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .status-section {
            background: #e8f5e8;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .api-section {
            background: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
        }
        
        .api-endpoint {
            background: #333;
            color: #00ff00;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1> Система Адаптивного Підбору Автомобілів</h1>
            <p>Каршеринг з інтеграцією технологій штучного інтелекту</p>
        </div>
        
        <div class="content">
            <div class="status-section">
                <h3> Статус системи</h3>
                <p><strong>Моделі:</strong> {{ " Завантажено" if models_loaded else " Не завантажено" }}</p>
                <p><strong>Час запуску:</strong> {{ startup_time }}</p>
                {% if not models_loaded %}
                <p style="color: #d32f2f;"> Для роботи системи потрібно спочатку виконати тренування моделей командою: <code>python train_models.py</code></p>
                {% endif %}
            </div>
            
            {% if models_loaded %}
            <div class="search-section">
                <h3> Пошук автомобіля</h3>
                <form id="searchForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="latitude"> Широта</label>
                            <input type="number" id="latitude" step="0.000001" value="49.4200" placeholder="49.4200">
                        </div>
                        <div class="form-group">
                            <label for="longitude"> Довгота</label>
                            <input type="number" id="longitude" step="0.000001" value="27.0058" placeholder="27.0058">
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-group">
                            <label for="seats"> Кількість місць</label>
                            <select id="seats">
                                <option value="2">2 місця</option>
                                <option value="4" selected>4 місця</option>
                                <option value="5">5 місць</option>
                                <option value="7">7 місць</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="maxPrice"> Макс. ціна за км (грн)</label>
                            <input type="number" id="maxPrice" value="10" min="1" max="50">
                        </div>
                    </div>
                    
                    <button type="submit" class="btn"> Знайти автомобіль</button>
                </form>
            </div>
            
            <div id="results" class="results"></div>
            {% endif %}
            
            <div class="api-section">
                <h3> API Endpoints</h3>
                <p><strong>POST /api/recommend</strong> - Отримання рекомендацій</p>
                <div class="api-endpoint">
curl -X POST http://localhost:5000/api/recommend \
-H "Content-Type: application/json" \
-d '{"latitude": 49.4200, "longitude": 27.0058, "preferences": {"preferred_seats": 4}}'
                </div>
                
                <p><strong>GET /api/status</strong> - Статус системи</p>
                <div class="api-endpoint">curl http://localhost:5000/api/status</div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('searchForm')?.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading"> Пошук автомобілів...</div>';
            
            const formData = {
                latitude: parseFloat(document.getElementById('latitude').value),
                longitude: parseFloat(document.getElementById('longitude').value),
                preferences: {
                    preferred_seats: parseInt(document.getElementById('seats').value),
                    max_price_per_km: parseFloat(document.getElementById('maxPrice').value)
                }
            };
            
            try {
                const response = await fetch('/api/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    displayResults(data.data);
                } else {
                    resultsDiv.innerHTML = `<div class="error"> ${data.message}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="error"> Помилка: ${error.message}</div>`;
            }
        });
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (!data.recommendations || data.recommendations.length === 0) {
                resultsDiv.innerHTML = '<div class="error"> Автомобілі не знайдено</div>';
                return;
            }
            
            let html = `
                <h3>🚗 Знайдено ${data.recommendations.length} автомобілів</h3>
                <p style="margin-bottom: 20px;"><strong>Інформація:</strong> ${data.demand_info}</p>
            `;
            
            data.recommendations.forEach((car, index) => {
                html += `
                    <div class="car-card">
                        <div class="car-header">
                            <div class="car-title">${car.brand} ${car.model}</div>
                            <div class="car-score">${(car.recommendation_score || car.final_score * 100).toFixed(0)}%</div>
                        </div>
                        <div class="car-details">
                            <div class="car-detail">
                                <div class="car-detail-label"> Відстань</div>
                                <div class="car-detail-value">${car.distance.toFixed(1)} км</div>
                            </div>
                            <div class="car-detail">
                                <div class="car-detail-label"> Місць</div>
                                <div class="car-detail-value">${car.seats}</div>
                            </div>
                            <div class="car-detail">
                                <div class="car-detail-label"> Рік</div>
                                <div class="car-detail-value">${car.year}</div>
                            </div>
                            <div class="car-detail">
                                <div class="car-detail-label"> Ціна/км</div>
                                <div class="car-detail-value">${car.price_per_km} грн</div>
                            </div>
                            <div class="car-detail">
                                <div class="car-detail-label"> Рейтинг</div>
                                <div class="car-detail-value">${car.rating}/5</div>
                            </div>
                            <div class="car-detail">
                                <div class="car-detail-label"> Орієнтовна вартість</div>
                                <div class="car-detail-value">${car.estimated_cost ? car.estimated_cost.toFixed(0) + ' грн' : 'N/A'}</div>
                            </div>
                        </div>
                        ${car.recommendation_reasons && car.recommendation_reasons.length > 0 ? 
                            `<div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 8px;">
                                <strong> Чому рекомендуємо:</strong> ${car.recommendation_reasons.join(', ')}
                            </div>` : ''
                        }
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():

    models_loaded = demand_model is not None and ranking_model is not None
    return render_template_string(MAIN_PAGE_TEMPLATE, 
                                models_loaded=models_loaded,
                                startup_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@app.route('/api/status')
def api_status():

    try:
        status = {
            'status': 'online',
            'models_loaded': demand_model is not None and ranking_model is not None,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        if profiler.timings:
            status['performance'] = profiler.get_statistics()
        
        return jsonify(status)
    
    except Exception as e:
        logger.log_error("Помилка API статусу", e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def api_recommend():

    try:
        profiler.start_timer('recommendation_request')
        
        # Перевірка наявності моделей
        if not demand_model or not ranking_model or not recommendation_engine:
            return jsonify({
                'status': 'error',
                'message': 'Моделі не завантажено. Виконайте спочатку тренування.'
            }), 503
        

        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Не надано JSON даних'
            }), 400
        

        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is None or longitude is None:
            return jsonify({
                'status': 'error',
                'message': 'Потрібно вказати latitude та longitude'
            }), 400
        
        user_location = (float(latitude), float(longitude))
        

        is_valid, error_msg = DataValidator.validate_user_location(user_location)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': f'Некоректні координати: {error_msg}'
            }), 400
        

        preferences = data.get('preferences', {})
        

        is_valid, error_msg = DataValidator.validate_user_preferences(preferences)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': f'Некоректні вподобання: {error_msg}'
            }), 400
        

        user_id = data.get('user_id', 'anonymous')
        recommendations = recommendation_engine.generate_recommendations(
            user_id, user_location, preferences
        )
        
        execution_time = profiler.end_timer('recommendation_request')
        

        logger.log_recommendation(
            user_id, 
            recommendations['recommendations'], 
            execution_time
        )
        
        return jsonify({
            'status': 'success',
            'data': recommendations,
            'execution_time': execution_time
        })
    
    except Exception as e:
        logger.log_error("Помилка API рекомендацій", e)
        return jsonify({
            'status': 'error',
            'message': 'Внутрішня помилка сервера',
            'details': str(e) if app.debug else None
        }), 500


@app.route('/api/explain/<car_id>')
def api_explain(car_id):

    try:

        explanation = {
            'car_id': car_id,
            'explanation': 'Автомобіль рекомендовано на основі ML моделі та ваших вподобань',
            'factors': {
                'distance': 'Близько до вас',
                'price': 'Оптимальна ціна',
                'ml_score': 'Висока оцінка ML моделі',
                'availability': 'Доступний зараз'
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': explanation
        })
    
    except Exception as e:
        logger.log_error("Помилка API пояснення", e)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/performance')
def api_performance():

    try:
        return jsonify({
            'status': 'success',
            'data': profiler.get_statistics()
        })
    
    except Exception as e:
        logger.log_error("Помилка API продуктивності", e)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):

    return jsonify({
        'status': 'error',
        'message': 'Ендпоінт не знайдено'
    }), 404


@app.errorhandler(500)
def internal_error(error):

    if logger:
        logger.log_error("Внутрішня помилка сервера", error)
    
    return jsonify({
        'status': 'error',
        'message': 'Внутрішня помилка сервера'
    }), 500


def main():

    print(" Запуск системи адаптивного підбору автомобілів каршерингу")
    print("=" * 60)
    
    # Ініціалізація системи
    if initialize_system():
        print(" Система ініціалізована успішно")
        print(" Веб-інтерфейс: http://localhost:5000")
        print(" API документація: http://localhost:5000/api/status")
        print(" Налаштування в config.yaml")
        print("=" * 60)
        
        # Запуск Flask додатку
        app.run(
            host=config.get('web_app.host', '127.0.0.1'),
            port=config.get('web_app.port', 5000),
            debug=config.get('web_app.debug', True)
        )
    else:
        print(" Помилка ініціалізації системи")
        print(" Рекомендації:")
        print("   1. Перевірте наявність файлів даних в папці data/")
        print("   2. Виконайте тренування моделей: python train_models.py")
        print("   3. Перевірте config.yaml")


if __name__ == '__main__':
    main()
            