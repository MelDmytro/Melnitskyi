import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, precision_score, 
                           recall_score, f1_score, roc_auc_score, classification_report,
                           roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class BaseXGBoostModel:

    
    def __init__(self, config, model_type='regressor'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.scaler = StandardScaler() if config.get('use_scaling', False) else None
        self.is_trained = False
        
    def _initialize_model(self):
        """Ініціалізація XGBoost моделі"""
        if self.model_type == 'regressor':
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 5),
                min_child_weight=self.config.get('min_child_weight', 3),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
                reg_alpha=self.config.get('reg_alpha', 0.1),
                reg_lambda=self.config.get('reg_lambda', 1.0)
            )
        elif self.model_type == 'classifier':
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 5),
                min_child_weight=self.config.get('min_child_weight', 3),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
                reg_alpha=self.config.get('reg_alpha', 0.1),
                reg_lambda=self.config.get('reg_lambda', 1.0)
            )
        elif self.model_type == 'ranker':
            self.model = xgb.XGBRanker(
                n_estimators=self.config.get('n_estimators', 100),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 5),
                min_child_weight=self.config.get('min_child_weight', 3),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
                reg_alpha=self.config.get('reg_alpha', 0.1),
                reg_lambda=self.config.get('reg_lambda', 1.0)
            )
    
    def _prepare_data(self, X, y=None):

        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)

        if self.scaler is not None:
            if y is not None:  # Тренування
                X_scaled = self.scaler.fit_transform(X)
            else:  # Прогнозування
                X_scaled = self.scaler.transform(X)
            

            if hasattr(X, 'columns'):
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            return X_scaled
        
        return X
    
    def _calculate_feature_importance(self):
"
        if self.model is None or not self.is_trained:
            return None
        
        importance_values = self.model.feature_importances_
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance_values))]
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def get_model_params(self):

        if self.model is None:
            return {}
        return self.model.get_params()
    
    def save_model(self, path):

        if self.model is None:
            raise ValueError("Модель не ініціалізована")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': self.config,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f" Модель збережено: {path}")
    
    def load_model(self, path):

        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names')
        self.feature_importance = model_data.get('feature_importance')
        self.config = model_data.get('config', self.config)
        self.model_type = model_data.get('model_type', self.model_type)
        self.is_trained = model_data.get('is_trained', True)
        
        print(f" Модель завантажено: {path}")


class DemandPredictor(BaseXGBoostModel):

    
    def __init__(self, config):
        super().__init__(config['demand_model'], model_type='regressor')
        self.validation_scores = {}
        
    def train(self, X, y, validation_split=0.2, early_stopping_rounds=20):

        print(" Початок тренування моделі прогнозування попиту...")
        

        self._initialize_model()

        X_processed = self._prepare_data(X, y)
        

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=validation_split*2, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f" Розподіл даних:")
        print(f"  Тренування: {len(X_train)} зразків")
        print(f"  Валідація: {len(X_val)} зразків")
        print(f"  Тестування: {len(X_test)} зразків")
        

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        self.is_trained = True
        

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        y_test_pred = self.model.predict(X_test)
        

        train_metrics = self._calculate_regression_metrics(y_train, y_train_pred, "Training")
        val_metrics = self._calculate_regression_metrics(y_val, y_val_pred, "Validation")
        test_metrics = self._calculate_regression_metrics(y_test, y_test_pred, "Test")
        

        self.validation_scores = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        

        self._calculate_feature_importance()
        
        print(f"\n Результати тренування:")
        print(f"  Test MAE: {test_metrics['MAE']:.4f}")
        print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  Test MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"  Test R²: {test_metrics['R2']:.4f}")
        
        return test_metrics
    
    def _calculate_regression_metrics(self, y_true, y_pred, dataset_name=""):

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE з захистом від ділення на нуль
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-7))) * 100
        
        # R² (коефіцієнт детермінації)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-7))
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        if dataset_name:
            print(f"  {dataset_name} метрики:")
            for metric, value in metrics.items():
                if metric == 'MAPE':
                    print(f"    {metric}: {value:.2f}%")
                else:
                    print(f"    {metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, X):

        if not self.is_trained:
            raise ValueError("Модель не натренована")
        
        X_processed = self._prepare_data(X)
        predictions = self.model.predict(X_processed)
        

        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_uncertainty(self, X, n_estimators_range=(10, None)):

        if not self.is_trained:
            raise ValueError("Модель не натренована")
        
        X_processed = self._prepare_data(X)
        

        start_estimators = n_estimators_range[0]
        end_estimators = n_estimators_range[1] or self.model.n_estimators
        
        predictions_list = []
        
        for n_est in range(start_estimators, min(end_estimators + 1, self.model.n_estimators + 1), 10):

            pred = self.model.predict(X_processed, ntree_limit=n_est)
            predictions_list.append(pred)
        
        if not predictions_list:

            pred = self.model.predict(X_processed)
            return {
                'predictions': pred,
                'uncertainty': np.zeros_like(pred),
                'confidence_lower': pred,
                'confidence_upper': pred
            }
        
        predictions_array = np.array(predictions_list)
        

        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'confidence_lower': mean_pred - 1.96 * std_pred,
            'confidence_upper': mean_pred + 1.96 * std_pred
        }
    
    def plot_feature_importance(self, top_n=15):

        if self.feature_importance is None:
            print(" Спочатку потрібно натренувати модель")
            return
        
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        top_features = self.feature_importance.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Важливість ознаки')
        plt.title(f'Топ-{top_n} найважливіших ознак для прогнозування попиту')
        plt.gca().invert_yaxis()
        

        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_curves(self):

        if not hasattr(self.model, 'evals_result_') or not self.model.evals_result_:
            print(" Дані кривих навчання недоступні")
            return
        
        results = self.model.evals_result_
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        
        plt.figure(figsize=(12, 4))
        

        plt.subplot(1, 2, 1)
        plt.plot(x_axis, results['validation_0']['rmse'], label='Validation RMSE')
        plt.xlabel('Епохи')
        plt.ylabel('RMSE')
        plt.title('Крива навчання (RMSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        

        if 'train' in results:
            plt.subplot(1, 2, 2)
            plt.plot(x_axis, results['train']['rmse'], label='Train RMSE')
            plt.plot(x_axis, results['validation_0']['rmse'], label='Validation RMSE')
            plt.xlabel('Епохи')
            plt.ylabel('RMSE')
            plt.title('Переднавчання')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class CarRanker(BaseXGBoostModel):

    
    def __init__(self, config):
        super().__init__(config['ranking_model'], model_type='classifier')
        self.validation_scores = {}
        
    def train(self, X, y, validation_split=0.2):

        print(" Початок тренування моделі ранжування автомобілів...")
        

        self._initialize_model()
        

        X_processed = self._prepare_data(X, y)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y, test_size=validation_split*2, 
            random_state=42, stratify=y
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, 
            random_state=42, stratify=y_temp
        )
        
        print(f" Розподіл даних:")
        print(f"  Тренування: {len(X_train)} зразків")
        print(f"  Валідація: {len(X_val)} зразків")
        print(f"  Тестування: {len(X_test)} зразків")
        print(f"  Розподіл класів: {np.bincount(y_train)}")

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        self.is_trained = True
        
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        

        train_metrics = self._calculate_classification_metrics(
            y_train, y_train_pred, y_train_proba, "Training"
        )
        val_metrics = self._calculate_classification_metrics(
            y_val, y_val_pred, y_val_proba, "Validation"
        )
        test_metrics = self._calculate_classification_metrics(
            y_test, y_test_pred, y_test_proba, "Test"
        )
        

        self.validation_scores = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        

        self._calculate_feature_importance()
        
        print(f"\n Результати тренування:")
        print(f"  Test Precision: {test_metrics['Precision']:.4f}")
        print(f"  Test Recall: {test_metrics['Recall']:.4f}")
        print(f"  Test F1-Score: {test_metrics['F1']:.4f}")
        print(f"  Test AUC-ROC: {test_metrics['AUC']:.4f}")
        
        return test_metrics
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_proba, dataset_name=""):

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        

        accuracy = np.mean(y_true == y_pred)
        
        metrics = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'Accuracy': accuracy
        }
        
        if dataset_name:
            print(f"  {dataset_name} метрики:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        return metrics
    
    def rank_cars(self, X):

        if not self.is_trained:
            raise ValueError("Модель не натренована")
        
        X_processed = self._prepare_data(X)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return probabilities
    
    def predict(self, X):

        if not self.is_trained:
            raise ValueError("Модель не натренована")
        
        X_processed = self._prepare_data(X)
        predictions = self.model.predict(X_processed)
        
        return predictions
    
    def predict_top_k(self, X, k=5):

        scores = self.rank_cars(X)
        top_k_indices = np.argsort(scores)[-k:][::-1]  # Сортування по спаданню
        
        return {
            'indices': top_k_indices,
            'scores': scores[top_k_indices],
            'rankings': list(range(1, k + 1))
        }
    
    def plot_feature_importance(self, top_n=15):

        if self.feature_importance is None:
            print(" Спочатку потрібно натренувати модель")
            return
        
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        top_features = self.feature_importance.head(top_n)
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(top_features)))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Важливість ознаки')
        plt.title(f'Топ-{top_n} найважливіших ознак для ранжування автомобілів')
        plt.gca().invert_yaxis()
        

        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(importance + 0.001, i, f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test):

        if not self.is_trained:
            print(" Модель не натренована")
            return
        
        X_processed = self._prepare_data(X_test)
        y_proba = self.model.predict_proba(X_processed)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC крива (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Випадковий класифікатор')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC крива для ранжування автомобілів')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()


class HyperparameterTuner:

    
    @staticmethod
    def tune_demand_model(X, y, search_type='grid', n_iter=20):

        print(f"Підбір гіперпараметрів для моделі попиту ({search_type} search)...")
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        if search_type == 'grid':

            reduced_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
            
            search = GridSearchCV(
                xgb_model, reduced_grid,
                cv=3, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                xgb_model, param_grid,
                n_iter=n_iter, cv=3, scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(X, y)
        
        print("Найкращі параметри для моделі ранжування:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"  Найкращий F1: {search.best_score_:.4f}")
        
        return search.best_params_


class ModelEvaluator:

    
    @staticmethod
    def evaluate_demand_model(model, X, y, cv_folds=5):
"
        print(" Комплексна оцінка моделі прогнозування попиту...")
        
        if not model.is_trained:
            print("Модель не натренована")
            return {}
        
        cv_scores_mae = cross_val_score(
            model.model, X, y, cv=cv_folds, 
            scoring='neg_mean_absolute_error', n_jobs=-1
        )
        cv_scores_rmse = cross_val_score(
            model.model, X, y, cv=cv_folds,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        cv_scores_r2 = cross_val_score(
            model.model, X, y, cv=cv_folds,
            scoring='r2', n_jobs=-1
        )
        
        print(f" Cross-validation результати ({cv_folds} фолдів):")
        print(f"  MAE: {-cv_scores_mae.mean():.4f} (+/- {cv_scores_mae.std() * 2:.4f})")
        print(f"  RMSE: {-cv_scores_rmse.mean():.4f} (+/- {cv_scores_rmse.std() * 2:.4f})")
        print(f"  R²: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")
        

        y_pred = model.predict(X)
        

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        

        mape = np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-7))) * 100
        

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-7))
        

        max_error = np.max(np.abs(y - y_pred))
        median_ae = np.median(np.abs(y - y_pred))
        

        residuals = y - y_pred
        residuals_std = np.std(residuals)
        residuals_skew = ModelEvaluator._calculate_skewness(residuals)
        residuals_kurtosis = ModelEvaluator._calculate_kurtosis(residuals)
        
        evaluation_results = {

            'cv_mae_mean': -cv_scores_mae.mean(),
            'cv_mae_std': cv_scores_mae.std(),
            'cv_rmse_mean': -cv_scores_rmse.mean(),
            'cv_rmse_std': cv_scores_rmse.std(),
            'cv_r2_mean': cv_scores_r2.mean(),
            'cv_r2_std': cv_scores_r2.std(),
            

            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            

            'max_error': max_error,
            'median_absolute_error': median_ae,
            

            'residuals_std': residuals_std,
            'residuals_skewness': residuals_skew,
            'residuals_kurtosis': residuals_kurtosis
        }
        
        # Візуалізація результатів
        ModelEvaluator._plot_demand_evaluation(y, y_pred, residuals)
        
        return evaluation_results
    
    @staticmethod
    def evaluate_ranking_model(model, X, y, cv_folds=5):

        print(" Комплексна оцінка моделі ранжування...")
        
        if not model.is_trained:
            print(" Модель не натренована")
            return {}
        

        cv_scores_f1 = cross_val_score(
            model.model, X, y, cv=cv_folds, scoring='f1', n_jobs=-1
        )
        cv_scores_precision = cross_val_score(
            model.model, X, y, cv=cv_folds, scoring='precision', n_jobs=-1
        )
        cv_scores_recall = cross_val_score(
            model.model, X, y, cv=cv_folds, scoring='recall', n_jobs=-1
        )
        cv_scores_roc_auc = cross_val_score(
            model.model, X, y, cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )
        
        print(f" Cross-validation результати ({cv_folds} фолдів):")
        print(f"  F1: {cv_scores_f1.mean():.4f} (+/- {cv_scores_f1.std() * 2:.4f})")
        print(f"  Precision: {cv_scores_precision.mean():.4f} (+/- {cv_scores_precision.std() * 2:.4f})")
        print(f"  Recall: {cv_scores_recall.mean():.4f} (+/- {cv_scores_recall.std() * 2:.4f})")
        print(f"  ROC-AUC: {cv_scores_roc_auc.mean():.4f} (+/- {cv_scores_roc_auc.std() * 2:.4f})")
        

        y_pred = model.predict(X)
        y_pred_proba = model.rank_cars(X)
        

        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        accuracy = np.mean(y == y_pred)
        

        ranking_metrics = ModelEvaluator._calculate_ranking_metrics(y, y_pred_proba)
        

        class_report = classification_report(y, y_pred, output_dict=True)
        
        evaluation_results = {

            'cv_f1_mean': cv_scores_f1.mean(),
            'cv_f1_std': cv_scores_f1.std(),
            'cv_precision_mean': cv_scores_precision.mean(),
            'cv_precision_std': cv_scores_precision.std(),
            'cv_recall_mean': cv_scores_recall.mean(),
            'cv_recall_std': cv_scores_recall.std(),
            'cv_auc_mean': cv_scores_roc_auc.mean(),
            'cv_auc_std': cv_scores_roc_auc.std(),
            

            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'accuracy': accuracy,
            
            'classification_report': class_report
        }
        

        ModelEvaluator._plot_ranking_evaluation(y, y_pred, y_pred_proba)
        
        return evaluation_results
    
    @staticmethod
    def _calculate_ranking_metrics(y_true, y_scores, k_values=[5, 10, 20]):

        metrics = {}
        
        for k in k_values:
            if k > len(y_true):
                continue
            
            # Precision@k
            top_k_indices = np.argsort(y_scores)[-k:]
            precision_at_k = np.sum(y_true[top_k_indices]) / k
            
            # Recall@k
            total_relevant = np.sum(y_true)
            recall_at_k = np.sum(y_true[top_k_indices]) / max(total_relevant, 1)
            
            # NDCG@k
            dcg = 0
            idcg = 0
            

            sorted_indices = np.argsort(y_scores)[::-1][:k]
            for i, idx in enumerate(sorted_indices):
                rank = i + 1
                dcg += (2 ** y_true[idx] - 1) / np.log2(rank + 1)
            

            sorted_true = np.sort(y_true)[::-1][:k]
            for i, relevance in enumerate(sorted_true):
                rank = i + 1
                idcg += (2 ** relevance - 1) / np.log2(rank + 1)
            
            ndcg = dcg / max(idcg, 1e-7)
            
            metrics[f'precision_at_{k}'] = precision_at_k
            metrics[f'recall_at_{k}'] = recall_at_k
            metrics[f'ndcg_at_{k}'] = ndcg
        
        return metrics
    
    @staticmethod
    def _calculate_skewness(data):

        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness
    
    @staticmethod
    def _calculate_kurtosis(data):

        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurtosis = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurtosis
    
    @staticmethod
    def _plot_demand_evaluation(y_true, y_pred, residuals):

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        

        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Фактичні значення')
        axes[0, 0].set_ylabel('Прогнозовані значення')
        axes[0, 0].set_title('Фактичні vs Прогнозовані значення')
        axes[0, 0].grid(True, alpha=0.3)
        

        axes[0, 1].hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Залишки (y_true - y_pred)')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].set_title('Розподіл залишків')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot залишків')
            axes[1, 0].grid(True, alpha=0.3)
        except ImportError:
            axes[1, 0].text(0.5, 0.5, 'SciPy не доступна\nдля Q-Q plot', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Q-Q Plot залишків (недоступний)')
        

        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Прогнозовані значення')
        axes[1, 1].set_ylabel('Залишки')
        axes[1, 1].set_title('Залишки vs Прогнозовані значення')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _plot_ranking_evaluation(y_true, y_pred, y_pred_proba):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC крива (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC крива')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, color='blue', lw=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall крива')
        axes[0, 1].grid(True, alpha=0.3)
        

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Прогнозований клас')
        axes[1, 0].set_ylabel('Фактичний клас')
        axes[1, 0].set_title('Матриця помилок')
        

        axes[1, 1].hist([y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]], 
                       bins=30, alpha=0.7, label=['Клас 0', 'Клас 1'], 
                       color=['lightcoral', 'lightblue'])
        axes[1, 1].set_xlabel('Прогнозована ймовірність')
        axes[1, 1].set_ylabel('Частота')
        axes[1, 1].set_title('Розподіл ймовірностей за класами')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_models(models_results, metric='f1'):

        print(f" Порівняння моделей за метрикою: {metric}")
        
        model_names = list(models_results.keys())
        metric_values = [results.get(metric, 0) for results in models_results.values()]
        
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Metric_Value': metric_values
        }).sort_values('Metric_Value', ascending=False)
        
        print("\n Рейтинг моделей:")
        for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"  {i}. {row['Model']}: {row['Metric_Value']:.4f}")
        

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(comparison_df['Model'], comparison_df['Metric_Value'], color=colors)
        
        plt.xlabel('Моделі')
        plt.ylabel(f'{metric.upper()}')
        plt.title(f'Порівняння моделей за метрикою {metric.upper()}')
        plt.xticks(rotation=45)
        

        for bar, value in zip(bars, comparison_df['Metric_Value']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df


class EnsemblePredictor:

    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_trained = False
    
    def train(self, X, y):

        print(" Тренування ансамблю моделей...")
        
        for i, model in enumerate(self.models):
            print(f"  Тренування моделі {i+1}/{len(self.models)}...")
            model.train(X, y)
        
        self.is_trained = True
        print(" Ансамбль натренований")
    
    def predict(self, X):

        if not self.is_trained:
            raise ValueError("Ансамбль не натренований")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def rank_cars(self, X):

        if not all(hasattr(model, 'rank_cars') for model in self.models):
            raise ValueError("Не всі моделі підтримують ранжування")
        
        rankings = []
        for model in self.models:
            rank = model.rank_cars(X)
            rankings.append(rank)
        

        ensemble_rank = np.average(rankings, axis=0, weights=self.weights)
        return ensemble_rank


def create_pipeline(config):

    print(" Створення ML пайплайну...")
    

    demand_predictor = DemandPredictor(config)
    car_ranker = CarRanker(config)
    
    pipeline = {
        'demand_predictor': demand_predictor,
        'car_ranker': car_ranker,
        'tuner': HyperparameterTuner(),
        'evaluator': ModelEvaluator()
    }
    
    print(" Пайплайн створено успішно")
    return pipeline


def main():

    print(" ДЕМОНСТРАЦІЯ XGBOOST МОДЕЛЕЙ")
    print("=" * 50)
    
    # Конфігурація за замовчуванням
    config = {
        'demand_model': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 3,
            'random_state': 42,
            'use_scaling': False
        },
        'ranking_model': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 3,
            'random_state': 42,
            'use_scaling': False
        }
    }
    

    print(" Створення тестових даних...")
    np.random.seed(42)
    

    X_demand = pd.DataFrame(np.random.randn(1000, 10), 
                           columns=[f'demand_feature_{i}' for i in range(10)])
    y_demand = np.random.poisson(5, 1000)  # Пуассонівський розподіл для попиту
    

    X_ranking = pd.DataFrame(np.random.randn(1000, 8), 
                            columns=[f'ranking_feature_{i}' for i in range(8)])
    y_ranking = np.random.binomial(1, 0.3, 1000)  # Бінарні мітки для ранжування
    
    print(f"  Дані попиту: {X_demand.shape}, середній попит: {y_demand.mean():.2f}")
    print(f"  Дані ранжування: {X_ranking.shape}, позитивних: {y_ranking.sum()}")
    
    try:

        print("\n Тренування моделі прогнозування попиту...")
        demand_model = DemandPredictor(config)
        demand_metrics = demand_model.train(X_demand, y_demand)
        

        print("\n Тренування моделі ранжування...")
        ranking_model = CarRanker(config)
        ranking_metrics = ranking_model.train(X_ranking, y_ranking)
        

        print("\n Демонстрація прогнозування:")
        

        sample_demand = demand_model.predict(X_demand[:5])
        print(f"  Прогноз попиту для 5 локацій: {sample_demand.round(2)}")
        

        sample_rankings = ranking_model.rank_cars(X_ranking[:5])
        print(f"  Рейтинги 5 автомобілів: {sample_rankings.round(3)}")
        

        top_k_result = ranking_model.predict_top_k(X_ranking[:10], k=3)
        print(f"  Топ-3 автомобілі (індекси): {top_k_result['indices']}")
        print(f"  Їх рейтинги: {top_k_result['scores'].round(3)}")
        

        print("\n Прогнозування з оцінкою невизначеності:")
        uncertainty_result = demand_model.predict_with_uncertainty(X_demand[:3])
        print(f"  Прогнози: {uncertainty_result['predictions'].round(2)}")
        print(f"  Невизначеність: {uncertainty_result['uncertainty'].round(2)}")

        print("\n Комплексна оцінка моделей:")
        demand_evaluation = ModelEvaluator.evaluate_demand_model(demand_model, X_demand, y_demand)
        ranking_evaluation = ModelEvaluator.evaluate_ranking_model(ranking_model, X_ranking, y_ranking)
        

        print("\n Збереження моделей...")
        import os
        os.makedirs('temp_models', exist_ok=True)
        
        demand_model.save_model('temp_models/demo_demand_model.pkl')
        ranking_model.save_model('temp_models/demo_ranking_model.pkl')
        

        print("\n Тестування завантаження моделей...")
        new_demand_model = DemandPredictor(config)
        new_demand_model.load_model('temp_models/demo_demand_model.pkl')
        
        new_ranking_model = CarRanker(config)
        new_ranking_model.load_model('temp_models/demo_ranking_model.pkl')
        

        restored_predictions = new_demand_model.predict(X_demand[:3])
        original_predictions = demand_model.predict(X_demand[:3])
        
        if np.allclose(restored_predictions, original_predictions):
            print(" Моделі збережено та відновлено коректно")
        else:
            print(" Помилка при збереженні/відновленні моделей")
        

        print("\n Демонстрація ансамблю моделей...")
        

        models_for_ensemble = []
        for i in range(3):
            temp_config = config.copy()
            temp_config['demand_model'] = config['demand_model'].copy()
            temp_config['demand_model']['random_state'] = 42 + i
            temp_config['demand_model']['n_estimators'] = 50 + i * 25
            
            ensemble_model = DemandPredictor(temp_config)
            ensemble_model.train(X_demand, y_demand)
            models_for_ensemble.append(ensemble_model)
        

        ensemble = EnsemblePredictor(models_for_ensemble, weights=[0.4, 0.35, 0.25])
        ensemble_predictions = ensemble.predict(X_demand[:5])
        
        print(f"  Ансамблеві прогнози: {ensemble_predictions.round(2)}")
        print(f"  Оригінальні прогнози: {sample_demand.round(2)}")
        

        print("\n Демонстрація підбору гіперпараметрів...")
        

        small_X = X_demand[:200]
        small_y = y_demand[:200]
        
        best_params = HyperparameterTuner.tune_demand_model(
            small_X, small_y, search_type='random', n_iter=5
        )
        
        print(" Найкращі параметри знайдено")
        

        try:
            demand_model.plot_feature_importance(top_n=8)
            ranking_model.plot_feature_importance(top_n=6)
            ranking_model.plot_roc_curve(X_ranking, y_ranking)
        except:
            print("  Візуалізація недоступна в неінтерактивному режимі")
        

        print("\n Очищення тимчасових файлів...")
        import shutil
        if os.path.exists('temp_models'):
            shutil.rmtree('temp_models')
        
        print("\n Демонстрація завершена успішно!")
        print("\n Підсумок:")
        print(f"   Модель попиту - Test MAE: {demand_metrics.get('MAE', 0):.4f}")
        print(f"   Модель ранжування - Test F1: {ranking_metrics.get('F1', 0):.4f}")
        print(f"   Ансамбль працює успішно")
        
    except Exception as e:
        print(f" Помилка під час демонстрації: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            import shutil
            import os
            if os.path.exists('temp_models'):
                shutil.rmtree('temp_models')
        except:
            pass


if __name__ == "__main__":
    main()
        
        print(" Найкращі параметри для моделі попиту:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"  Найкращий MAE: {-search.best_score_:.4f}")
        
        return search.best_params_
    
    @staticmethod
    def tune_ranking_model(X, y, search_type='grid', n_iter=20):
        print(f" Підбір гіперпараметрів для моделі ранжування ({search_type} search)...")
        
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
        if search_type == 'grid':
            reduced_grid = {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5]
            }
            
            search = GridSearchCV(
                xgb_model, reduced_grid,
                cv=3, scoring='f1',
                n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                xgb_model, param_grid,
                n_iter=n_iter, cv=3, scoring='f1',
                n_jobs=-1, verbose=1, random_state=42
            )
        
        search.fit(X, y)
        