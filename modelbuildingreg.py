import logging
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, VotingRegressor, StackingRegressor)
from xgboost import XGBRegressor  # Ensure you have xgboost installed
from lightgbm import LGBMRegressor  # Ensure you have lightgbm installed
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor

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
        logging.info("Initializing Linear Regression model.")
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Ridge Regression Strategy
class RidgeRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> Ridge:
        logging.info("Initializing Ridge Regression model.")
        model = Ridge(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Lasso Regression Strategy
class LassoRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> Lasso:
        logging.info("Initializing Lasso Regression model.")
        model = Lasso(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Elastic Net Regression Strategy
class ElasticNetRegressionStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> ElasticNet:
        logging.info("Initializing Elastic Net model.")
        model = ElasticNet(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Support Vector Regression Strategy
class SVRStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> SVR:
        logging.info("Initializing Support Vector Regression model.")
        model = SVR(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Decision Tree Regression Strategy
class DecisionTreeRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> DecisionTreeRegressor:
        logging.info("Initializing Decision Tree Regressor model.")
        model = DecisionTreeRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Random Forest Regression Strategy
class RandomForestRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RandomForestRegressor:
        logging.info("Initializing Random Forest Regressor model.")
        model = RandomForestRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Gradient Boosting Regression Strategy
class GradientBoostingRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> GradientBoostingRegressor:
        logging.info("Initializing Gradient Boosting Regressor model.")
        model = GradientBoostingRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# HistGradient Boosting Regression Strategy
class HistGradientBoostingRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> HistGradientBoostingRegressor:
        logging.info("Initializing HistGradient Boosting Regressor model.")
        model = HistGradientBoostingRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# XGBoost Regression Strategy
class XGBRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> XGBRegressor:
        logging.info("Initializing XGBoost Regressor model.")
        model = XGBRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# LightGBM Regression Strategy
class LGBMRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> LGBMRegressor:
        logging.info("Initializing LightGBM Regressor model.")
        model = LGBMRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# AdaBoost Regression Strategy
class AdaBoostRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> AdaBoostRegressor:
        logging.info("Initializing AdaBoost Regressor model.")
        model = AdaBoostRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# CatBoost Regression Strategy
class CatBoostRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> CatBoostRegressor:
        logging.info("Initializing CatBoost Regressor model.")
        model = CatBoostRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model
# K-Nearest Neighbors Regression Strategy
class KNeighborsRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> KNeighborsRegressor:
        logging.info("Initializing K-Nearest Neighbors Regressor model.")
        model = KNeighborsRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Multi-layer Perceptron Regression Strategy
class MLPRegressorStrategy(ModelBuildingRegressorStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> MLPRegressor:
        logging.info("Initializing Multi-layer Perceptron Regressor model.")
        model = MLPRegressor(**kwargs)
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Voting Regressor Strategy
class VotingRegressorStrategy(ModelBuildingRegressorStrategy):
    def __init__(self, regressors: Dict[str, ModelBuildingRegressorStrategy]):
        self.regressors = regressors

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> VotingRegressor:
        from sklearn.ensemble import VotingRegressor
        logging.info("Initializing Voting Regressor model.")
        models = {name: strategy.build_and_train_model(X_train, y_train, **kwargs) for name, strategy in self.regressors.items()}
        model = VotingRegressor(estimators=list(models.items()))
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Stacking Regressor Strategy
class StackingRegressorStrategy(ModelBuildingRegressorStrategy):
    def __init__(self, regressors: Dict[str, ModelBuildingRegressorStrategy], final_estimator: ModelBuildingRegressorStrategy):
        self.regressors = regressors
        self.final_estimator = final_estimator

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> StackingRegressor:
        from sklearn.ensemble import StackingRegressor
        logging.info("Initializing Stacking Regressor model.")
        models = {name: strategy.build_and_train_model(X_train, y_train, **kwargs) for name, strategy in self.regressors.items()}
        model = StackingRegressor(estimators=list(models.items()), final_estimator=self.final_estimator.build_and_train_model(X_train, y_train, **kwargs))
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model

# Context Class for Model Building
class ModelBuilder:
    def __init__(self, strategy: ModelBuildingRegressorStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ModelBuildingRegressorStrategy):
        logging.info("Switching model building strategy.")
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RegressorMixin:
        logging.info("Building and training the model using the selected strategy.")
        return self._strategy.build_and_train_model(X_train, y_train, **kwargs)

if __name__ == "__main__":
   ''' # Load the California Housing dataset
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import r2_score

    housing_data = fetch_california_housing(as_frame=True)
    df = housing_data.frame

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    imputed_values = imputer.fit_transform(df)
    df = pd.DataFrame(imputed_values, columns=df.columns)  # Recreate DataFrame with imputed values

    X = df.drop(columns=['MedHouseVal'])  # Features
    y = df['MedHouseVal']  # Target variable

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define regressors for Voting and Stacking using strategy classes
    regressors = {
        'ridge': RidgeRegressionStrategy(),
        'dt': DecisionTreeRegressorStrategy()
    }

    # Initialize model strategies
    voting_strategy = VotingRegressorStrategy(regressors)
    stacking_strategy = StackingRegressorStrategy(regressors, final_estimator=RidgeRegressionStrategy())

    # Create ModelBuilder instance for Voting Regressor
    voting_builder = ModelBuilder(voting_strategy)

    # Build and train Voting Regressor
    voting_model = voting_builder.build_model(X_train, y_train)

    # Predictions and evaluation for Voting Regressor
    voting_predictions = voting_model.predict(X_test)
    voting_mse = mean_squared_error(y_test, voting_predictions)
    voting_r2 = r2_score(y_test, voting_predictions)

    logging.info(f"Voting Regressor - Mean Squared Error: {voting_mse}")
    logging.info(f"Voting Regressor - R^2 Score: {voting_r2}")

    # Create ModelBuilder instance for Stacking Regressor
    stacking_builder = ModelBuilder(stacking_strategy)

    # Build and train Stacking Regressor
    stacking_model = stacking_builder.build_model(X_train, y_train)

    # Predictions and evaluation for Stacking Regressor
    stacking_predictions = stacking_model.predict(X_test)
    stacking_mse = mean_squared_error(y_test, stacking_predictions)
    stacking_r2 = r2_score(y_test, stacking_predictions)

    logging.info(f"Stacking Regressor - Mean Squared Error: {stacking_mse}")
    logging.info(f"Stacking Regressor - R^2 Score: {stacking_r2}")
'''
pass