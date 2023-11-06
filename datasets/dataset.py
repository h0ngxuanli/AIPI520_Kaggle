from .engineer_data import *
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder

def get_engineered_data(data_df, stage, subscriptions_data, zipcodes_data, accounts_data):
    """
    Prepare training and testing data by incorporating subscriptions, zipcodes, and accounts information.
    Fill missing data caused by merging the above dataframes.
    Encode categorical features into digits.

    Parameters:
        data_df (pd.DataFrame): Original data.
        stage (str): Indicates whether the function is preparing training or testing data.
        subscriptions_data (pd.DataFrame): DataFrame containing music taste information.
        zipcodes_data (pd.DataFrame): DataFrame containing zipcode information.
        accounts_data (pd.DataFrame): DataFrame containing accounts information.

    Returns:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Target variable.
    """
    
    df = deepcopy(data_df)
    
    # change column name to facilitate merging process
    if stage == "test":
        df.rename(columns = {"ID":"account.id"}, inplace = True)

    # incorporate accounts information
    df = pd.merge(df, accounts_data, how = "left", on= "account.id")
    
    # incorporate zipcode information
    # convert billingzipcode column to float column
    df["billing.zip.code"] = df["billing.zip.code"].apply(lambda x: 0 if type(x) == str else x)
    df = pd.merge( df, 
                    zipcodes_data[["Zipcode", "City", "Lat", "Long", "Decommisioned", "TaxReturnsFiled", "EstimatedPopulation", "TotalWages"]], 
                    how = "left", left_on = "billing.zip.code", right_on = "Zipcode"
                    )
    # incorporate music taste information
    df = pd.merge(df, subscriptions_data, how = "left", on = "account.id")



    # drop and rename duplicate columns
    drop_columns = ['billing.zip.code', "billing.city", "Zipcode", "City"]
    df.drop(columns = drop_columns, inplace = True)
    df.rename(columns = {
                        "Lat_x": "Lat_account", "Long_x": "Long_account",
                        "Lat_y": "Lat_sub", "Long_y": "Long_sub"
                        }, inplace = True)


    # fill missing data
    df["amount.donated.2013"].fillna(0, inplace = True)
    df["amount.donated.lifetime"].fillna(0, inplace = True)
    df["no.donations.lifetime"].fillna(0, inplace = True)
    df["first.donated"].fillna(0, inplace = True)
    df["Lat_account"].fillna(0, inplace = True)
    df["Long_account"].fillna(0, inplace = True)
    df["Decommisioned"].fillna(-1, inplace = True)
    df["TaxReturnsFiled"].fillna(0, inplace = True)
    df["EstimatedPopulation"].fillna(0, inplace = True)
    df["TotalWages"].fillna(0, inplace = True)
    df["season"].fillna(0, inplace = True)
    df["package"].fillna("None", inplace = True)
    df["no.seats"].fillna(-1, inplace = True)
    df["section"].fillna("None", inplace = True)
    df["multiple.subs"].fillna("None", inplace = True)
    df["section"].fillna("None", inplace = True)
    df["price.level"].fillna(-1, inplace = True)
    df["subscription_tier"].fillna(-1, inplace = True)
    df["Lat_sub"].fillna(0, inplace = True)
    df["Long_sub"].fillna(0, inplace = True)
    df["music_taste"].fillna(0, inplace = True)

    
    # find object type columns
    category_columns = []
    for column in df.columns[1:]:
        if df[column].dtype == "object":
            category_columns.append(column)

    # create individusal label encoder for each column
    label_encoders = [LabelEncoder() for i in range(len(category_columns))]
    le_dict = dict(zip(category_columns, label_encoders))
    
    # encode category columns
    for column in category_columns:
        le_dict[column].fit(df[column])
        df[column] = le_dict[column].transform(df[column])
    
    # return final data 
    if stage == "test":

        X = df.iloc[:, 1:]
        y = None
    else:
        X = df.iloc[:, 2:]
        y = df.iloc[:, 1]

    return X, y
