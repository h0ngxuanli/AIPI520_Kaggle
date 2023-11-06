# Fall 2023 AIPI520 Classical Music Meets Classical ML Kaggle

This repository contains the code and data for the Kaggle competition in the AIPI 520 class.

## Steps for running the code

### Setup

#### Environment

The code is based on Python 3. You can create a virtual environment and install the dependencies using the following commands:

```bash
$ python -m venv venv
$ source venv/bin/activate  # On Windows, use venv\Scripts\activate
(venv)$ pip install -r requirements.txt
```

### Data

The data for this project is located in the `data/` directory. It includes the following files:

- `test.csv`
- `tickets_all.csv`
- `concerts_2014-15.csv`
- `account.csv`
- `train.csv`
- `subscriptions.csv`
- `zipcodes.csv`
- `concerts.csv`
- `sample_submission.csv`

### Running the code

To run the main script, use the following command:

```bash
(venv)$ python main.py
```

The script trains models, conducts hyperparameter optimization using Optuna, performs feature selection, and generates predictions for the test data. The results and logs are saved in the `Results/` directory with a timestamped subdirectory.

## Directory Structure

```
AIPI520_Kaggle/
│
├── Results/
│   └── 20231106-131827/
│       ├── lightgbm_best_param.json
│       ├── train_optuna_result.csv
│       ├── submission.csv
│       ├── feature_importance.csv
│       ├── feature_importance.png
│       └── train_20231106-131827.log
│
├── data/
│   ├── test.csv
│   ├── tickets_all.csv
│   ├── concerts_2014-15.csv
│   ├── account.csv
│   ├── train.csv
│   ├── subscriptions.csv
│   ├── zipcodes.csv
│   ├── concerts.csv
│   └── sample_submission.csv
│
├── main.py
│
├── datasets/
│   ├── __init__.py
│   ├── dataset.py
│   └── engineer_data.py
│
├── models/
│   ├── __init__.py
│   └── model.py
│
├── utils/
│   ├── __init__.py
│   ├── common_utils.py
│   ├── logger.py
│   └── helpers.py
│
└── requirements.txt
```

In this structure:

- **`Results/`**: Contains subdirectories for each run of the code. Each run's results, logs, and output files are saved in a timestamped directory.

- **`data/`**: Contains the dataset files required for the project.

- **`main.py`**: The main script to run the project.

- **`models/`**: Contains the module for creating regression models (`model.py`).

- **`utils/`**: Contains utility modules for training, testing, logging, common function.

- **`requirements.txt`**: Lists all the required packages for the project.

Feel free to adjust the structure and add more subdirectories or files as necessary for your project. Let me know if you need any further assistance!