import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from copy import copy
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.trial import Trial
import json
from datasets import *
from utils import *
import warnings
import datetime
# Ignore all warnings
warnings.filterwarnings("ignore")

N_TRIAL = 20


    
def main():
    """
    Main function to process data, train models, conduct feature selection, generate predictions, and save results.

    This function performs the following steps:
    1. Loads data from CSV files.
    2. Engineers features from loaded data.
    3. Trains a model using LightGBM with hyperparameter optimization.
    4. Conducts feature selection based on model results.
    5. Retrains the model using selected features.
    6. Generates predictions for the test data.
    7. Saves the submission file, optimization results, selected features, and best hyperparameters.
    """
    

    
    # get data path
    root_path = Path("/home/featurize/work/AIPI520_Kaggle")#os.getcwd())
    data_path = root_path / "data"
    trial_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_path = root_path / "results" / trial_time
    
    # initialize the output path for different trails
    output_path.mkdir(parents=True, exist_ok=True)
    
    # create logger
    log_file = output_path / ('train_%s.log' % trial_time)
    logger = common_utils.create_logger(log_file)
    
    

    logger.info(f'********************** Loading Data **********************')
    # load data
    accounts_df = pd.read_csv(data_path / "account.csv", encoding="ISO-8859-1") # location info for each patron and donation history
    zipcodes_df =  pd.read_csv(data_path / "zipcodes.csv") # location and demographic information for zipcodes
    tickets_df =  pd.read_csv(data_path / "tickets_all.csv")
    subscriptions_df =  pd.read_csv(data_path / "subscriptions.csv")
    concerts_df =  pd.read_csv(data_path / "concerts.csv")
    concerts1415_df = pd.read_csv(data_path / "concerts_2014-15.csv")
    train_df = pd.read_csv(data_path / "train.csv") # whether the patrons have purchased a 2014-15 subscription or not
    test_df = pd.read_csv(data_path / "test.csv")
    submission_df = pd.read_csv(data_path / "sample_submission.csv")

    logger.info(f'********************** Data Engineering **********************')
    # load engineered data
    accounts_data = engineer_account(accounts_df)
    zipcodes_data = engineer_zipcode(zipcodes_df)
    subscriptions_data = engineer_subscription(subscriptions_df, tickets_df, zipcodes_df, concerts_df, concerts1415_df)

    # get training and testing data
    X_train, y_train = get_engineered_data(train_df, "train", subscriptions_data, zipcodes_data, accounts_data)
    X_test, _ =  get_engineered_data(test_df, "test", subscriptions_data, zipcodes_data, accounts_data)
    
    
    logger.info(f'********************** Feature Selection with LightGBM **********************')
    # train model
    study = train_model_with_optuna(X_train, y_train, n_trials = N_TRIAL)
    train_optuna_result = study.trials_dataframe()
    
    
    # conduct feature selection
    feature_importance = feature_selection(study, X_train, y_train, output_path)

    # determine threshold through feature importance dataframe
    threshold = 7
    selected_features = feature_importance.iloc[:threshold, 1].values
    X_train_ft_selected = X_train[selected_features]
    X_test_ft_selected = X_test[selected_features]

    
    logger.info(f'********************** Train the Final Model **********************')
    # retrain model on data after feature selection
    study_ft_selected = train_model_with_optuna(X_train_ft_selected, y_train, n_trials = N_TRIAL)
    train_optuna_ft_selected_result = study_ft_selected.trials_dataframe()
    print_study(study_ft_selected, logger)
    
    logger.info(f'********************** Create Submission & Save Experiment Results **********************')
    # generate submission result
    submission = generate_prediction(study_ft_selected, test_df, submission_df, X_train_ft_selected, y_train, X_test_ft_selected)
    
    # save result
    submission.to_csv(output_path / "submission.csv", index = False)
    train_optuna_result.to_csv(output_path / "train_optuna_result.csv", index = False)
    train_optuna_ft_selected_result.to_csv("train_optuna_ft_selected_result.csv", index = False)
    
    
    
    # save best params to json
    best_params = study_ft_selected.best_trial.params
    best_params["objective"] = "regression"
    best_params["seed"] = 42
    best_params["metric"] = "auc"
    best_params["boosting_type"] = "gbdt"
    best_params['is_unbalance'] = True
    best_params["n_trials"] = N_TRIAL
    save_json(best_params, output_path / "lightgbm_best_param.json")
    
    logger.info(f'********************** Done **********************')
    
if __name__ == "__main__":
    main()