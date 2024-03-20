from sklearn.linear_model import ElasticNet, BayesianRidge, SGDRegressor, HuberRegressor, LassoLars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor,LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

class MoistureBaseModel:
    def __init__(self):
        self.best_model = None
        self.best_r2 = -float('inf')
        self.best_rmse = float('inf')
        self.data = None
        
    def load_data(self):
        try:
            # Mock-up: replace this with data loading logic
            self.data = pd.read_excel('./sensor_data.xlsx') # Assuming db.select returns a DataFrame
            self.data.dropna()  # This will drop NA values
            return self.data
        except Exception as e:
            print(f"Error in load_data: {e}")

    def preprocess_data(self):
        try:
            categorical_columns = ['sensorId']
            numerical_columns = ['sensorValueMoisture', 'sensorValueTemperature','sensorValueDensity']

            X = self.data[categorical_columns + numerical_columns]
            y = self.data['moisture'].values

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", RobustScaler(), numerical_columns),
                    ("cat", OneHotEncoder(drop='first', sparse=False), categorical_columns)
                ]
            )
            return X, y, preprocessor
        except Exception as e:
            print(f"Error in preprocess_data: {e}")
            
    def train_test_split(self, X, y):
        try:
            return train_test_split(X, y, test_size=0.3, random_state=0)
        except Exception as e:
            print(f"Error in train_test_split: {e}")
        
    def try_various_models(self, X_train, X_test, y_train, y_test, preprocessor):
        try:
            models_to_try = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "SVR": SVR(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "ElasticNet": ElasticNet(),
                "BayesianRidge": BayesianRidge(),
                "SGDRegressor": SGDRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "BaggingRegressor": BaggingRegressor(),
                "ExtraTreesRegressor": ExtraTreesRegressor(),
                "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
                "HuberRegressor": HuberRegressor(),
                "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
                "PassiveAggressiveRegressor": PassiveAggressiveRegressor()
            }

            for name, model in models_to_try.items():
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                print(f"Model: {name}, R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

                if r2 > self.best_r2 or (r2 == self.best_r2 and rmse < self.best_rmse):
                    self.best_r2 = r2
                    self.best_rmse = rmse
                    self.best_model = pipeline
        except Exception as e:
            print(f"Error in try_various_models: {e}")
                
    def tune_best_model(self, X_train, y_train):
        try:
            if self.best_model is None:
                print("No best model selected yet.")
                return
            
            model_name = self.best_model.named_steps['model'].__class__.__name__
            
            if model_name == 'RandomForestRegressor':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_features': ['auto', 'sqrt', 'log2'],
                    'model__max_depth': [10, 20, 30, None],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'HistGradientBoostingRegressor':
                param_grid = {
                    'model__learning_rate': [0.01, 0.1],
                    'model__max_iter': [100, 200],
                    'model__max_depth': [10, 20, None]
                }
            elif model_name == 'GradientBoostingRegressor':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.5],
                    'model__subsample': [0.8, 0.9, 1.0],
                    'model__max_depth': [3, 4, 5]
                }
            elif model_name == 'SVR':
                param_grid = {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['auto', 'scale']
                }
            elif model_name == 'KNeighborsRegressor':
                param_grid = {
                    'model__n_neighbors': [3, 5, 11],
                    'model__weights': ['uniform', 'distance'],
                    'model__metric': ['euclidean', 'manhattan']
                }
            elif model_name == 'ElasticNet':
                param_grid = {
                    'model__alpha': [0.1, 0.5, 1, 2, 5],
                    'model__l1_ratio': [0, 0.25, 0.5, 0.75, 1]
                }
            elif model_name == 'Lasso':
                param_grid = {
                    'model__alpha': [0.1, 0.5, 1, 2, 5]
                }

            elif model_name == 'Ridge':
                param_grid = {
                    'model__alpha': [0.1, 0.5, 1, 2, 5]
                }

            elif model_name == 'BayesianRidge':
                param_grid = {
                    'model__alpha_1': [1e-6, 1e-5, 1e-4],
                    'model__alpha_2': [1e-6, 1e-5, 1e-4],
                    'model__lambda_1': [1e-6, 1e-5, 1e-4],
                    'model__lambda_2': [1e-6, 1e-5, 1e-4]
                }
            else:
                param_grid = {}  # Default, no hyperparameter tuning

            grid_search = GridSearchCV(estimator=self.best_model, param_grid=param_grid, cv=3, verbose=2)
            grid_search.fit(X_train, y_train)
            
            print("Best parameters found: ", grid_search.best_params_)
            self.best_model = grid_search.best_estimator_

        except Exception as e:
            print(f"Error in tune_best_model: {e}")
            
    # ส่วนของ CassavaModel Class
    def save_best_model(self, path):
        try:
            if self.best_model is not None:
                joblib.dump(self.best_model, path)
                print(f"Saved best model to {path}")
                
        except Exception as e:
            print(f"Error in save_best_model: {e}")  

if __name__ == "__main__":
    try:
        moisture_model = MoistureBaseModel()
        moisture_model.load_data()
        X, y, preprocessor = moisture_model.preprocess_data()
        X_train, X_test, y_train, y_test = moisture_model.train_test_split(X, y)
        
        # Add this line to train and evaluate different models
        moisture_model.try_various_models(X_train, X_test, y_train, y_test, preprocessor)

        print("Best Model based on R2 and RMSE:")
        print(moisture_model.best_model)
        moisture_model.tune_best_model(X_train, y_train)
        if moisture_model.best_model is not None:
            print(f"Best Model: {moisture_model.best_model.named_steps['model'].__class__.__name__}, R2 Score: {moisture_model.best_r2:.4f}, RMSE: {moisture_model.best_rmse:.4f}")
            moisture_model.save_best_model('./linear_moisture_model.pkl')
        else:
            print("No best model found.")
    except Exception as e:
        print(f"Error in Application model_create start: {e}")
