from copy import copy
import pandas as pd
import numpy as np
from copy import deepcopy


def get_name(string):
    """
    Extract names from a string.

    Parameters:
        string (str): Input string containing names.

    Returns:
        list: List of extracted names.
    """  
    
    import re
    TITLE = r"(?:[A-Z][a-z]*\.\s*)?"
    NAME1 = r"[A-Z][a-z]+,?\s+"
    MIDDLE_I = r"(?:[A-Z][a-z]*\.?\s*)?"
    NAME2 = r"[A-Z][a-z]+"
    return re.findall(TITLE + NAME1 + MIDDLE_I + NAME2, string)

def get_musical_taste(musician):
    """
    Calculate musical taste of a person based on the number of musicians they like from 2014-2015 concerts.

    Parameters:
        musician (list): List of musicians liked by the person.

    Returns:
        int: Number of musicians liked by the person from 2014-2015 concerts.
    """

    return len( list(set(musician) & set(concert_201415_musical_name)) )

def get_music_taste_from_tickets_subs(subscriptions_data, tickets_data, concerts_data, concerts1415_data):
    """
    Music taste is engineered by calculating how many musicians in concert 2014-2015 
    are supported by accounts in the form of buying tickets or subscriptions in the previous concerts.
    a. Extract musician names with concerts_data and concerts1415_data.
    b. Compute music taste for tickets_data.
    c. Compute music taste for subscriptions_data.
    d. Combine music taste information by concatenating subscriptions_data and tickets_data.

    Parameters:
        subscriptions_data (pd.DataFrame): Subscriptions data.
        tickets_data (pd.DataFrame): Tickets data.
        concerts_data (pd.DataFrame): Concerts data.
        concerts1415_data (pd.DataFrame): Concerts 2014-2015 data.

    Returns:
        combined_music_taste (pd.DataFrame): Combined music taste information from subscriptions and tickets data.
    """

    # get musician names in concerts 201415
    global concert_201415_musical_name 
    concert_201415_musical_name  = np.unique(concerts1415_data["who"].apply(get_name).sum())

    # extract music taste from tickets information
    tickets = pd.merge(tickets_data, concerts_data[["season", "location", "who", "set"]], how = "inner", on = ["season", "set", "location"])
    tickets["music_taste"] = tickets["who"].str.split("\r").apply(lambda x: ", ".join(x)).apply(get_name)
    tickets["music_taste"] = tickets["music_taste"].apply(get_musical_taste)

    # prepare tickets_data to be combined with subsriptions_df
    drop_columns = ["marketing.source", "set", "who"]
    tickets.drop(columns = drop_columns, inplace = True)
    tickets["package"] = "None"
    tickets["section"] = "None"
    tickets["subscription_tier"] = 0
    tickets.rename(columns = {"multiple.tickets":"multiple.subs"}, inplace = True)
  
    # extract music taste from subsciptions information
    # one season and one location may include multiple set of concerts
    subscriptions = pd.merge(subscriptions_data, concerts_data[["season", "location", "who"]], how = "left", on = ["season", "location"])
    subscriptions["who"].fillna("None", inplace = True)
    subscriptions["music_taste"] = subscriptions["who"].str.split("\r").apply(lambda x: ", ".join(x)).apply(get_name)
    subscriptions["music_taste"] = subscriptions["music_taste"].apply(get_musical_taste)
    
    # add up music taste for one season and one location subscription
    sub_music_taste = subscriptions.groupby(["account.id", "season", "location"])["music_taste"].sum().reset_index()
    subscriptions = pd.merge(subscriptions_data, sub_music_taste, how = "left", on = ["account.id","season", "location"])

    # combine music taste information from both subscription and tickets df
    return pd.concat([subscriptions, tickets], axis = 0)

def engineer_account(accounts_df):
    
    """
    Clean account dataframe.
    a. Drop columns with few values.
    b. Convert 'billing.zip.code' to int to be merged with zipcode dataframe.
    c. Extract the year as an integer from the 'first.donated' column.

    Parameters:
        accounts_df (pd.DataFrame): Accounts dataframe.

    Returns:
        accounts_data (pd.DataFrame): Cleaned accounts dataframe.
    """
    #load data
    accounts_data = deepcopy(accounts_df)
    
    # drop columns without sufficient information
    drop_columns = ["shipping.zip.code", "shipping.city", "relationship"]    
    accounts_data.drop(columns = drop_columns, inplace = True)
    
    # fillna for billing zip code
    accounts_data['billing.zip.code'].fillna("0", inplace = True)
    
    # split zipcode with "-"
    accounts_data['billing.zip.code'] = accounts_data['billing.zip.code'].str.split("-").str[0]
    
    # turn digit zipcode to int
    accounts_data.update(accounts_data["billing.zip.code"].loc[accounts_data["billing.zip.code"].str.isdigit()].astype(np.int64))
    
    # extract year for donate year
    accounts_data.update(accounts_data["first.donated"][accounts_data["first.donated"].notnull()].str.split("/").str[0].astype(np.int64))

    return accounts_data

def engineer_zipcode(zipcodes_df):
    
    """
    Clean zipcode dataframe.
    a. Capitalize the City to unify the format across different dataframes.
    b. Convert the 'Decommisioned' column to int type.

    Parameters:
        zipcodes_df (pd.DataFrame): Zipcodes dataframe.

    Returns:
        zipcodes_data (pd.DataFrame): Cleaned zipcodes dataframe.
    """  
    # load data
    
    zipcodes_data = deepcopy(zipcodes_df)
    # uninfy city format
    zipcodes_data.City = zipcodes_data.City.str.title()
    
    # turn bool column to int
    zipcodes_data.Decommisioned.fillna(-1, inplace = True)
    zipcodes_data.update(zipcodes_data.Decommisioned.astype(np.int64))
    
    return zipcodes_data

def engineer_tickets(tickets_df):
    """
    Clean tickets dataframe price.level column.
    a. Filter out strings not in ["0", "1", "2", "3", "4"] and replace with NaN.
    b. Convert the price.level column to float and fill NaN values with mean value.

    Parameters:
        tickets_df (pd.DataFrame): Tickets dataframe.

    Returns:
        tickets_data (pd.DataFrame): Cleaned tickets dataframe.
    """
    
    tickets_data = deepcopy(tickets_df)
    
    # turn price.level not in ["0", "1", "2", "3", "4"] into Nan 
    tickets_data["price.level"][~tickets_data["price.level"].str.contains("|".join(list("01234")), na=False)] = np.nan
    
    # convert price.level into float and fill Nan with mean value
    tickets_data["price.level"] = tickets_data["price.level"].astype(np.float64)
    tickets_data["price.level"].fillna(tickets_data["price.level"].mean(), inplace = True)   
    
    return tickets_data

def engineer_subscription(subscriptions_df, tickets_df, zipcodes_df, concerts_df, concerts1415_df):
    """
    Clean subscription data 
    a. Combine music taste information from tickets
    b. Clean location, package, section feature
    c. Get latitude and longitude information for subscribed concerts' places from zipcodes df. 
    d. Fill missing data and get representative subscription information for each account.

    Parameters:
        subscriptions_df (pd.DataFrame): Subscriptions data.
        tickets_df (pd.DataFrame): Tickets data.
        zipcodes_df (pd.DataFrame): Zipcodes data.
        concerts_df (pd.DataFrame): Concerts data.
        concerts1415_df (pd.DataFrame): Concerts 2014-2015 data.

    Returns:
        sub_info (pd.DataFrame): Engineered subscription data with representative information for each account.
    """
    # deepcoy data
    concerts_data = deepcopy(concerts_df)
    concerts1415_data = deepcopy(concerts1415_df)
    subscriptions_data = deepcopy(subscriptions_df)
    tickets_data = engineer_tickets(tickets_df)
    zipcodes_data = engineer_zipcode(zipcodes_df)
    
    # combine music taste information into subsrciption data
    subscriptions_data = get_music_taste_from_tickets_subs(subscriptions_data, tickets_data, concerts_data, concerts1415_data)
    
    
    # clean location information in the subscriptions_data
    subscriptions_data.location.replace({
                                    "Berkeley Saturday":"Berkeley", 
                                    "Berkeley Sunday":"Berkeley", 
                                    'Orange County': "Orange",
                                    'Contra Costa':'Costa'
                                    }, inplace=True)
    
    # clean package information in the subscriptions_data
    subscriptions_df.package.replace({
                                "Quartet CC": "Quartet", 
                                "Quartet B": "Quartet", 
                                "Quartet A": "Quartet",
                                "Trio A": "Trio",
                                "Trio B": "Trio",
                                }, inplace = True)
    
    
    # clean section information in the subscriptions_data
    subscriptions_df.section.replace({
                                "Premium Orchestra": "Orchestra", 
                                "Orchestra Rear": "Orchestra", 
                                "Orchestra Front": "Orchestra",
                                "Balcony Front": "Balcony",
                                "Balcony Rear": "Balcony",
                                "Boxes House Right":"Box",
                                "Boxes House left":"Box",
                                }, inplace = True)
    
    # add lat and long information of the subscribed concerts'place 
    subscriptions_data = pd.merge(
                        subscriptions_data, 
                        zipcodes_data.groupby("City")[["Lat", "Long"]].mean().reset_index(), 
                        how = "left", left_on = "location", right_on="City"
                        )
    # print(zipcodes_df.groupby("City")[["Lat", "Long"]].mean().reset_index())
    
    # fill missing data with "None", mean value or the value with the max value counts
    subscriptions_data["season"].fillna("None", inplace = True)
    subscriptions_data["package"].fillna("None", inplace = True)
    subscriptions_data["no.seats"].fillna(subscriptions_data["no.seats"].mean(), inplace = True) ### 
    subscriptions_data["section"].fillna("None", inplace = True)
    subscriptions_data["multiple.subs"].fillna("None", inplace = True)
    subscriptions_data["section"].fillna("None", inplace = True)
    subscriptions_data["price.level"].fillna(subscriptions_data["price.level"].mean(), inplace = True)
    subscriptions_data["subscription_tier"].fillna(subscriptions_data["subscription_tier"].mean(), inplace = True)
    
    
    subscriptions_data["Lat"].fillna( subscriptions_data["Lat"].value_counts().index[0], inplace = True)
    subscriptions_data["Long"].fillna( subscriptions_data["Long"].value_counts().index[0], inplace = True)    


    
    
    # get representative subsription information for each account by computing mean value, max value or max value counts
    sub_info_per_account = [
            subscriptions_data.groupby("account.id")['season'].count().reset_index(),
            subscriptions_data.groupby("account.id")['package'].agg(lambda x: x.value_counts().index[0]).reset_index(),
            subscriptions_data.groupby("account.id")['no.seats'].mean().reset_index(),
            subscriptions_data.groupby("account.id")['section'].agg(lambda x: x.value_counts().index[0]).reset_index(),
            subscriptions_data.groupby("account.id")['multiple.subs'].agg(lambda x: x.value_counts().index[0]).reset_index(),
            subscriptions_data.groupby("account.id")['price.level'].mean().reset_index(),
            subscriptions_data.groupby("account.id")['subscription_tier'].mean().reset_index(),
            subscriptions_data.groupby("account.id")['Lat'].agg(lambda x:x.value_counts().index[0]).reset_index(),
            subscriptions_data.groupby("account.id")['Long'].agg(lambda x:x.value_counts().index[0]).reset_index(),
            subscriptions_data.groupby("account.id")['music_taste'].sum().reset_index()
    ]
    for i in range(len(sub_info_per_account)):
        if i==0:
            sub_info = sub_info_per_account[i]
        else:
            sub_info = sub_info.merge(sub_info_per_account[i], how = "inner", on = "account.id")
    return sub_info