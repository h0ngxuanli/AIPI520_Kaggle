from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def generate_prediction(study, test_df, submission_df, X_train, y_train, X_test):

    """
    a. get optimal hyperparameters from optuna trials
    b. retrain model on full data
    c. generate final predictions
    """ 
    
    
    # get optimal hyperparameters
    params=study.best_params   
    params["objective"] = "regression"
    params["seed"] = 42
    params["metric"] = "auc"
    params["boosting_type"] = "gbdt"
    params['is_unbalance'] = True
    
    
    # retrain model on training datset
    model = LGBMRegressor(**params)  
    model.fit(X_train,y_train)

    # get probabilistic label
    pred = model.predict(X_test)
    pred_df = pd.concat([test_df, pd.DataFrame({"Predicted": pred})], axis = 1)
    
    # merge to submission file
    submission_df = pd.read_csv("/home/featurize/work/AIPI520_kaggle/sample_submission.csv")
    submission_df = pd.merge(submission_df.iloc[:,0], pred_df, how = "inner", on = "ID")
    submission_df.to_csv("submission.csv", index = False)
    
    return submission_df