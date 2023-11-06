
from models.model import get_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import optuna
import sys
sys.path.append("..")

def objective(trial, X_train, y_train):
    """
    Objective function for hyperparameter optimization using Optuna with model.

    Parameters:
        trial (optuna.Trial): Optuna trial object.
        X_train (pd.DataFrame): Training feature dataset.
        y_train (pd.Series): Training target variable.

    Returns:
        float: Mean AUC score for 5-fold cross-validation.
    """

    # paramter to be optimized
    model_params = {
        "objective": "regression",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",                
        "seed": 42,
        'is_unbalance': True, 

        'learning_rate': trial.suggest_categorical('learning_rate', [0.00001,0.00005, 0.001,0.0005, 0.01,0.05,0.1,0.5,1]),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),

        'max_bin': trial.suggest_int('max_bin', 2, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 100),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
    }

    nfolds = 5
    fold_pred = np.zeros(len(X_train))
    AUC = []
    
    
    # utilize StratifiedKFold to avoid class-imbalance problem
    folds = StratifiedKFold(n_splits=nfolds)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values,y_train.values)):
        print("fold n°{}".format(fold_))
        
        #fit training data
        model = get_model(model_params, "lightgbm") 
        model.fit(X_train.iloc[trn_idx],y_train.iloc[trn_idx])
        
        #get prediction on validation data
        fold_pred[val_idx] = model.predict(X_train.iloc[val_idx])
        AUC.append(roc_auc_score(y_train.iloc[val_idx].values, fold_pred[val_idx]))
    
    #return mean AUC for 5-fold cross-validation
    print("AUC score: {:<8.5f}".format(np.mean(AUC)))
    return np.mean(AUC)
    
def train_model_with_optuna(X_train, y_train, n_trials):
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    
    return study
