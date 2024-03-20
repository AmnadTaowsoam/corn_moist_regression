from model.LinerRegressorModel import MoistureBaseModel
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        cassava_model = MoistureBaseModel()
        cassava_model.load_data()
        X, y, preprocessor = cassava_model.preprocess_data()
        X_train, X_test, y_train, y_test = cassava_model.train_test_split(X, y)
        
        # Add this line to train and evaluate different models
        cassava_model.try_various_models(X_train, X_test, y_train, y_test, preprocessor)

        print("Best Model based on R2 and RMSE:")
        print(cassava_model.best_model)
        cassava_model.tune_best_model(X_train, y_train)
        if cassava_model.best_model is not None:
            print(f"Best Model: {cassava_model.best_model.named_steps['model'].__class__.__name__}, R2 Score: {cassava_model.best_r2:.4f}, RMSE: {cassava_model.best_rmse:.4f}")
            cassava_model.save_best_model('./Linear_Cassava_Model.pkl')
        else:
            print("No best model found.")
    except Exception as e:
        print(f"Error in Application model_create start: {e}")