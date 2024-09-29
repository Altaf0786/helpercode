import logging
from abc import ABC, abstractmethod
import category_encoders as ce
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import sklearn
from sklearn.preprocessing import PowerTransformer
from scipy.special import boxcox1p
from sklearn.preprocessing import FunctionTransformer


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Define the Strategy Interface
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Apply the encoder to transform the data.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the features to transform.
        y (optional): The target variable for supervised encoders.
        
        Returns:
        pd.DataFrame: The dataframe with encoded features.
        """
        pass

# 2. Implement Concrete Strategies
class CommonEncoder(FeatureEngineeringStrategy):
    def __init__(self, features, encoder_class, **params):
        """
        Initialize the CommonEncoder with a specific encoder class and parameters.
        
        Parameters:
        features (list): List of feature columns to encode.
        encoder_class: The encoder class from category_encoders to use.
        params (optional dict): Parameters for the encoder.
        """
        self.features = features
        self.encoder_class = encoder_class
        self.params = params
        self.encoder = None
        logging.info(f"Initialized {self.__class__.__name__} with features {features} and parameters {params}")

    def apply_transformation(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit the encoder and transform the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the features to transform.
        y (optional): The target variable for supervised encoders.
        
        Returns:
        pd.DataFrame: The dataframe with encoded features.
        """
        try:
            logging.info(f"Applying transformation with parameters: {self.params}")
            self.encoder = self.encoder_class(cols=self.features, **self.params)
            transformed = self.encoder.fit_transform(df[self.features], y)

            transformed_df = df.drop(columns=self.features).join(transformed)
            logging.info(f"Transformed columns: {transformed_df.columns.tolist()}")

            return transformed_df
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            raise

# 3. Concrete encoding strategies
class BackwardDifferenceEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.BackwardDifferenceEncoder, **params)

class BaseNEncoding(CommonEncoder):
    def __init__(self, features, base=3, **params):
        params.setdefault('base', base)
        super().__init__(features, ce.BaseNEncoder, **params)

class BinaryEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.BinaryEncoder, **params)

class CatBoostEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.CatBoostEncoder, **params)

class CountEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.CountEncoder, **params)

class GLMMEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.GLMMEncoder, **params)

class GrayEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.GrayEncoder, **params)

class HashingEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.HashingEncoder, **params)

class HelmertEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.HelmertEncoder, **params)

class JamesSteinEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.JamesSteinEncoder, **params)

class LeaveOneOutEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.LeaveOneOutEncoder, **params)

class OneHotEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.OneHotEncoder, **params)

class OrdinalEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.OrdinalEncoder, **params)

class PolynomialEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.PolynomialEncoder, **params)

class TargetEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.TargetEncoder, **params)

class SumEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.SumEncoder, **params)

class WoEEncoding(CommonEncoder):
    def __init__(self, features, **params):
        super().__init__(features, ce.WOEEncoder, **params)

# 2. Implement Concrete Transformation Strategies
class TransformationStrategy(FeatureEngineeringStrategy):
    def __init__(self, features, transformer_func, **params):
        """
        Initialize the TransformationStrategy with a specific transformer function and parameters.
        
        Parameters:
        features (list): List of feature columns to transform.
        transformer_func: The function to apply to the data.
        **params: Dynamic parameters for the transformation.
        """
        self.features = features
        self.transformer_func = transformer_func
        self.params = params
        logging.info(f"Initialized {self.__class__.__name__} with features {features} and params {params}")

    def apply_transformation(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Apply the transformation function to the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the features to transform.
        **params: Dynamic parameters for the transformation.
        
        Returns:
        pd.DataFrame: The dataframe with transformed features.
        """
        logging.info("Applying transformation...")
        transformed_df = df.copy()
        
        combined_params = {**self.params, **params}
        
        for feature in self.features:
            if feature in transformed_df.columns:
                transformed_df[feature] = self.transformer_func(transformed_df[feature], **combined_params)
            else:
                logging.warning(f"Feature {feature} not found in dataframe.")
        
        logging.info(f"Transformed columns: {transformed_df.columns.tolist()}")
        return transformed_df

# 3. Concrete Transformation Implementations
def log_transform(series: pd.Series, **kwargs) -> pd.Series:
    return np.log1p(series)

def reciprocal_transform(series: pd.Series, **kwargs) -> pd.Series:
    return np.reciprocal(series.replace(0, np.nan))

def square_transform(series: pd.Series, **kwargs) -> pd.Series:
    return series ** 2

def square_root_transform(series: pd.Series, **kwargs) -> pd.Series:
    return np.sqrt(series)

def cube_transform(series: pd.Series, **kwargs) -> pd.Series:
    return series ** 3

def boxcox_transform(series: pd.Series, **kwargs) -> pd.Series:
    lmbda = kwargs.get('lmbda', 0.5)  # Default lambda value
    return boxcox1p(series + 1e-9, lmbda)  # Corrected call

def yeojohnson_transform(series: pd.Series, **kwargs) -> pd.Series:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    return pd.Series(pt.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)

# Implement concrete transformation strategies
class LogTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, log_transform, **params)

class ReciprocalTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, reciprocal_transform, **params)

class SquareTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, square_transform, **params)

class SquareRootTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, square_root_transform, **params)

class CubeTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, cube_transform, **params)

class BoxCoxTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, boxcox_transform, **params)

class YeoJohnsonTransformation(TransformationStrategy):
    def __init__(self, features, **params):
        super().__init__(features, yeojohnson_transform, **params)
class ScalingStrategy(FeatureEngineeringStrategy):
    def __init__(self, scaler, features: list, **kwargs):
        """
        Initialize the ScalingStrategy with a specific scaler.
        
        Parameters:
        scaler: The scaler to apply to the data.
        features (list): List of feature columns to scale.
        kwargs: Additional parameters for customization.
        """
        self.scaler = scaler
        self.features = features
        self.kwargs = kwargs
        logging.info(f"Initialized {self.__class__.__name__} with features {features} and params {kwargs}")

    def apply_transformation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Apply the scaler to the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the features to scale.
        kwargs: Additional parameters for customization (overrides instance kwargs).
        
        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info("Applying scaling...")
        transformed_df = df.copy()
        scaler = self.scaler(**{**self.kwargs, **kwargs})
        transformed_df[self.features] = scaler.fit_transform(df[self.features])
        
        logging.info(f"Scaled columns: {transformed_df.columns.tolist()}")
        return transformed_df

# Concrete Scaling Implementations
class StandardScaling(ScalingStrategy):
    def __init__(self, features: list, **kwargs):
        super().__init__(StandardScaler, features, **kwargs)

class MinMaxScaling(ScalingStrategy):
    def __init__(self, features: list, **kwargs):
        super().__init__(MinMaxScaler, features, **kwargs)

class RobustScaling(ScalingStrategy):
    def __init__(self, features: list, **kwargs):
        super().__init__(RobustScaler, features, **kwargs)

class NormalizationScaling(ScalingStrategy):
    def __init__(self, features: list, **kwargs):
        super().__init__(Normalizer, features, **kwargs)        
# 4. Implement the Context Class
class Feature_Engineering:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy
        logging.info(f"EncodingContext initialized with strategy {strategy.__class__.__name__}")

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy
        logging.info(f"Strategy set to {strategy.__class__.__name__}")

    def apply_transformation(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Transform the dataframe using the strategy's apply_transformation method.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the features to transform.
        y (optional): The target variable for supervised encoders.
        
        Returns:
        pd.DataFrame: The dataframe with encoded features.
        """
        logging.info("Applying transformation in context...")
        return self.strategy.apply_transformation(df, y)

# Example usage
if __name__ == "__main__":
    '''# Load Titanic dataset
    df = sns.load_dataset('titanic')
    
    # Select features for encoding and target variable
    categorical_features = ['sex', 'embarked']
    target = df['survived']
    
    # Choose an encoding strategy
    encoding_strategy = TargetEncoding(
        features=categorical_features,
        verbose=1,  # Adjust parameters as needed
        drop_invariant=False,
        return_df=True,
        handle_unknown='value',
        min_samples_leaf=50,
        smoothing=20,
        hierarchy=None
    )
    
    # Create context with the chosen strategy
   # context =Feature_Engineering(strategy=encoding_strategy)
    
    # Apply transformation
    #df_encoded1 = context.apply_transformation(df, target)
    
    # Display the transfonrmed dataframe
    features_to_transform = ['fare','age']
    
    # Create a SquareTransformation strategy
    #square_transformation = SquareTransformation(features=features_to_transform)
    
    # Apply transformation
    #transformed_df = square_transformation.apply_transformation(df)
   
    # create a reciprocal streategy
    #reciprocal= ReciprocalTransformation(features=features_to_transform)
    #apply reciprocal transformation
    #transformed_df = reciprocal.apply_transformation(df)
     
    #create a log transformation strategy
    #log_transformation = LogTransformation(features=features_to_transform)
    #apply log transformation
    #transformed_df = log_transformation.apply_transformation(df)

    #create a boxcox transformation strategy
    #boxcox_transformation = BoxCoxTransformation(features=features_to_transform)
    #apply boxcox transformation
    #transformed_df = boxcox_transformation.apply_transformation(df)


    #create a yeojohnson transformation strategy
    #yeojohnson_transformation = YeoJohnsonTransformation(features=features_to_transform)
    #apply yeojohnson transformation
    #transformed_df = yeojohnson_transformation.apply_transformation(df)


    #create a standard scaling strategy
    #standard_scaling = StandardScaling(features=features_to_transform)
    #apply standard scaling
    #transformed_df = standard_scaling.apply_transformation(df)

    #create a minmax scaling strategy
    #minmax_scaling = MinMaxScaling(features=features_to_transform)
    #apply minmax scaling
    #transformed_df = minmax_scaling.apply_transformation(df)

    #create a robust scaling strategy
    #robust_scaling = RobustScaling(features=features_to_transform)
    #apply robust scaling
    #transformed_df = robust_scaling.apply_transformation(df)


    #create a normalization scaling strategy
    #normalization_scaling = NormalizationScaling(features=features_to_transform)
    #apply normalization scaling
    #transformed_df = normalization_scaling.apply_transformation(df)

    '''
    pass