import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, StackingRegressor, AdaBoostRegressor, 
                              BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor  # Ensure you have xgboost installed
from lightgbm import LGBMRegressor  # Ensure you have lightgbm installed
from catboost import CatBoostRegressor  # Ensure you have catboost installed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from typing import Dict

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Regressor Strategy
class ModelBuildingRegressorStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RegressorMixin:
        pass

# Linear Regression Strategy
class LinearRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> LinearRegression:
        try:
            logging.info("Initializing Linear Regression model.")
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise

# Ridge Regression Strategy
class RidgeRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> Ridge:
        try:
            logging.info("Initializing Ridge Regression model.")
            model = Ridge(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Ridge Regression model: {e}")
            raise

# Lasso Regression Strategy
class LassoRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> Lasso:
        try:
            logging.info("Initializing Lasso Regression model.")
            model = Lasso(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Lasso Regression model: {e}")
            raise

# ElasticNet Regression Strategy
class ElasticNetRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> ElasticNet:
        try:
            logging.info("Initializing ElasticNet Regression model.")
            model = ElasticNet(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training ElasticNet Regression model: {e}")
            raise

# MLP Regressor Strategy (Neural Networks)
class MLPRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> MLPRegressor:
        try:
            logging.info("Initializing MLP Regressor model.")
            model = MLPRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training MLP Regressor model: {e}")
            raise

# Decision Tree Regressor Strategy
class DecisionTreeRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> DecisionTreeRegressor:
        try:
            logging.info("Initializing Decision Tree Regressor model.")
            model = DecisionTreeRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Decision Tree Regressor model: {e}")
            raise

# KNN Regressor Strategy (K-Nearest Neighbors)
class KNeighborsRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> KNeighborsRegressor:
        try:
            logging.info("Initializing KNN Regressor model.")
            model = KNeighborsRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training KNN Regressor model: {e}")
            raise

# Support Vector Regressor Strategy (SVM)
class SVRStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> SVR:
        try:
            logging.info("Initializing Support Vector Regressor (SVM) model.")
            model = SVR(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Support Vector Regressor model: {e}")
            raise

# RandomForest Regressor Strategy (Bagging)
class RandomForestRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RandomForestRegressor:
        try:
            logging.info("Initializing Random Forest Regressor model.")
            model = RandomForestRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Random Forest Regressor model: {e}")
            raise

# Gradient Boosting Regressor Strategy (Boosting)
class GradientBoostingRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> GradientBoostingRegressor:
        try:
            logging.info("Initializing Gradient Boosting Regressor model.")
            model = GradientBoostingRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Gradient Boosting Regressor model: {e}")
            raise
class ExtraTreesRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> ExtraTreesRegressor:
        try:
            logging.info("Initializing Extra Trees Regressor model.")
            model = ExtraTreesRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Extra Trees Regressor model: {e}")
            raise e  # Re-raise the original exception
# AdaBoost Regressor Strategy (Boosting)
class AdaBoostRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> AdaBoostRegressor:
        try:
            logging.info("Initializing AdaBoost Regressor model.")
            model = AdaBoostRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training AdaBoost Regressor model: {e}")
            raise

# XGBoost Regressor Strategy (Boosting)
class XGBRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> XGBRegressor:
        try:
            logging.info("Initializing XGBoost Regressor model.")
            model = XGBRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training XGBoost Regressor model: {e}")
            raise

# LightGBM Regressor Strategy (Boosting)
class LGBMRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> LGBMRegressor:
        try:
            logging.info("Initializing LightGBM Regressor model.")
            model = LGBMRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training LightGBM Regressor model: {e}")
            raise

# CatBoost Regressor Strategy (Boosting)
class CatBoostRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> CatBoostRegressor:
        try:
            logging.info("Initializing CatBoost Regressor model.")
            model = CatBoostRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training CatBoost Regressor model: {e}")
            raise

# Stacking Regressor Strategy
class StackingRegressorStrategy(ModelBuildingRegressorStrategy):
    def __init__(self, regressors: Dict[str, ModelBuildingRegressorStrategy], final_estimator: ModelBuildingRegressorStrategy):
        self.regressors = regressors
        self.final_estimator = final_estimator

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> StackingRegressor:
        try:
            logging.info("Initializing Stacking Regressor model.")
            base_models = {name: strategy.build_and_train_model(X_train, y_train, **kwargs) for name, strategy in self.regressors.items()}
            final_model = self.final_estimator.build_and_train_model(X_train, y_train, **kwargs)
            model = StackingRegressor(estimators=list(base_models.items()), final_estimator=final_model)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Stacking Regressor model: {e}")
            raise
class BaggingRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> BaggingRegressor:
        try:
            logging.info("Initializing Bagging Regressor model.")
            model = BaggingRegressor(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Bagging Regressor model: {e}")
            raise

# Voting Regressor Strategy
class VotingRegressorStrategy(ModelBuildingRegressorStrategy):
    def __init__(self, regressors: Dict[str, ModelBuildingRegressorStrategy]):
        self.regressors = regressors

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> VotingRegressor:
        try:
            logging.info("Initializing Voting Regressor model.")
            base_models = {name: strategy.build_and_train_model(X_train, y_train, **kwargs) for name, strategy in self.regressors.items()}
            model = VotingRegressor(estimators=list(base_models.items()))
            model.fit(X_train, y_train)
            logging.info("Model training completed.")
            return model
        except Exception as e:
            logging.error(f"Error in training Voting Regressor model: {e}")
            raise
            

class ModelBuilder:
    def __init__(self, strategy: ModelBuildingRegressorStrategy):
        self._strategy = strategy

    def build_and_train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any):
        try:
            return self._strategy.build_and_train_model(X_train, y_train, **kwargs)
        except Exception as e:
            logging.error(f"Error in building and training model: {e}")
            raise
# Example Usage
'''if __name__ == "__main__":
# Example Usage:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score, mean_squared_error
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # Load the California Housing dataset
        data = fetch_california_housing()

        # Separate features and target
        X = data.data
        y = data.target

        # Train/test split for example
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing (e.g., imputing missing values)
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Example parameters
        params = {
            'ct_shape': True,
            'strict_shape': False,
            'n_samples': 50,
            'n_iterations': 50,
            'n_classes': 6,
            'n_trees_in_forest': 100,
            'learning_rate': 0.08,
            'max_depth': 6,
            'min_samples_split': 7,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': 1,
            
        }

        # Initialize a model strategy (example: XGBRegressorStrategy)
        model_builder = ModelBuilder(XGBRegressorStrategy())

        # Build and train the model
        model = model_builder.build_and_train(X_train, y_train, **params)  # Pass params with **kwargs

        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f"Model evaluation - R2: {r2:.4f}, MSE: {mse:.4f}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")'''
# Example Usage of the Bagging Regressor Strategy
'''if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import logging

    logging.basicConfig(level=logging.INFO)

    try:
        # Load the California Housing dataset
        data = fetch_california_housing()

        # Separate features and target
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instantiate the Bagging Regressor Strategy
        bagging_strategy = BaggingRegressorStrategy()

        # Build and train the model
        model_builder = ModelBuilder(bagging_strategy)
        model = model_builder.build_and_train(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"R-squared: {r2}")
    
    except Exception as e:
        logging.error(f"An error occurred during model training or evaluation: {e}")'''
'''f __name__ == "__main__":

    try:
        # Load the California Housing dataset
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create an instance of the VotingRegressorStrategy with two base models
        voting_strategy = VotingRegressorStrategy(regressors={
            'RandomForest': RandomForestRegressorStrategy(),
            'XGBoost':DecisionTreeRegressorStrategy()
        })

        # Train the model using the voting strategy
        model = voting_strategy.build_and_train_model(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log evaluation results
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"R-squared: {r2}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")    '''
if __name__ == "__main__":
    """try:
        # Load the California Housing dataset
        from sklearn.datasets import fetch_california_housing
       
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create an instance of the StackingRegressorStrategy
        stacking_strategy = StackingRegressorStrategy(
            regressors={
                'RandomForest': RandomForestRegressorStrategy(),
                'XGBoost': XGBRegressorStrategy()
            },
            final_estimator=LinearRegressionStrategy()
        )

        # Train the stacking model
        model = stacking_strategy.build_and_train_model(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log evaluation results
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"R-squared: {r2}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")            
"""
pass
