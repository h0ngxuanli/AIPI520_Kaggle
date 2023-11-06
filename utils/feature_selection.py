from sklearn.metrics import roc_auc_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from models import *
import seaborn as sns
import pandas as pd
    
def feature_selection(study, X_train, y_train, output_path):
    
    """
    Perform feature selection using the optimal model parameters obtained from Optuna.

    Parameters:
        study (optuna.study.Study): Optuna study object containing optimization results.
        X_train (pd.DataFrame): Training feature dataset.
        y_train (pd.Series): Training target variable.
        output_path (pathlib.Path): Path to save the feature importance results.

    Returns:
        final_importance (pd.DataFrame): DataFrame containing feature names and their importance scores.
    """
    
    # get optimal hyperparameters
    params = study.best_params   
    params["objective"] = "regression"
    params["seed"] = 42
    params["metric"] = "auc"
    params["boosting_type"] = "gbdt"
    params['is_unbalance'] = True
    
    # retrain model on training datset 
    model = get_model(params, "lightgbm") 
    model.fit(X_train,y_train)
    
    # save feature_importance dataframe
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = X_train.columns
    feature_importance_df["importance"] = model.booster_.feature_importance()
    final_importance = feature_importance_df.sort_values(by="importance", ascending=False)    
    final_importance.reset_index(inplace=True)
    final_importance.to_csv(output_path / "feature_importance.csv", index = False)
    
    # show feature importance through bar plot
    plt.figure(figsize=(14,25))
    sns.barplot(x="importance",y="feature",data=final_importance)
    plt.tight_layout()
    plt.savefig(output_path / "feature_importance.png")
    
    return final_importance

