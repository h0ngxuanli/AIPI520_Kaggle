from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from optuna import trial


def get_model(params, model_name):
    """
    Create and return a regression model based on the specified model name and optuna parameters.

    Parameters:
        params (dict): Dictionary containing hyperparameters for the regression model.
        model_name (str): Name of the regression model ("lightgbm", "catboost", or "xgboost").

    Returns:
        model: Regression model object.
    """
    
    model_dict = {
                  "lightgbm":LGBMRegressor, 
                  "catboost":CatBoostRegressor,
                  "xgboost":XGBRegressor
                  }
    
    return  model_dict[model_name](**params)