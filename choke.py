#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dateutil.parser import parse
import warnings, os, time, requests
from pathlib import Path
from glob import glob

warnings.filterwarnings('ignore')


# Global configuration variables
vars_ = {key: None for key in ['APP_TOKEN', 'LIMIT', 'DATASET', 'BASE_URL', 'URI_IDENTIFIERS', 'SAVE_TO_DIRECTORY',
                               'START_DATE', 'END_DATE', 'RES_PER_RESPONSE']}


def setDefaults(dataset, start_date, end_date, save_to_directory, results_per_response=10000, limit=100000):
    global vars_
    vars_['APP_TOKEN'] = "jlyZazYtAq9hSQNf0Ow4pNySj"
    vars_['LIMIT'] = limit
    vars_['DATASET'] = dataset
    vars_['BASE_URL'] = "https://data.cityofnewyork.us/resource/"
    vars_['URI_IDENTIFIERS'] = ['5uac-w243', 'qgea-i56i']
    vars_['START_DATE'] = start_date
    vars_['END_DATE'] = end_date
    vars_['RES_PER_RESPONSE'] = results_per_response
    vars_['SAVE_FILE_ENDING'] = 'Historic' if vars_['DATASET'] == 'qgea-i56i' else 'YTD'
    vars_['SAVE_TO_DIRECTORY'] = save_to_directory + '_' + vars_['SAVE_FILE_ENDING']

    if all(map(lambda x: x is not None, vars_.values())):
        print("Log: Defaults successfully loaded.")
    else:
        print("Log: Some defaults weren't loaded correctly.")


def isValid(dataset, start_date, end_date, save_to_directory, results_per_response=10000, limit=100000):
    global vars_

    def validDate(date):
        try:
            parse(date, fuzzy=True)
            return True
        except ValueError:
            print("Error: Incorrect date.")
            return False

    if dataset is None:
        print("Error: Invalid URI.")
        return -1
    elif start_date is None:
        print("Error: Start date is empty.")
        return -1
    elif end_date is None:
        print("Error: End date is empty.")
        return -1
    elif save_to_directory is None:
        print("Error: Directory to write data is not present.")
        return -1
    elif results_per_response <= 0 or limit <= 0:
        print("Error: results_per_response and limit must be > 0.")
        return -1
    elif validDate(start_date) and validDate(end_date):
        return 1
    else:
        return -1


def isValidAndSetDefaults(dataset_uri, start_date, end_date, save_to_directory, results_per_response=10000, limit=100000):
    ret_value = isValid(dataset_uri, start_date, end_date, save_to_directory)
    if ret_value in [0, 1]:
        setDefaults(dataset_uri, start_date, end_date, save_to_directory, results_per_response, limit)
        return True
    return False


def getData():
    def buildURL():
        url = f"{vars_['BASE_URL']}{vars_['DATASET']}.json?$where=cmplnt_fr_dt >='{vars_['START_DATE']}' and cmplnt_fr_dt < '{vars_['END_DATE']}'&$limit={vars_['RES_PER_RESPONSE']}&$order=cmplnt_fr_dt"
        return url

    def callAPI(url):
        token = vars_['APP_TOKEN']
        header = {"X-App-Token": token}
        res = requests.get(url, headers=header, verify=False)
        if res.status_code == 200:
            return res.json()
        return None

    def getDataByChunks():
        offset = 0
        prevFound = vars_['RES_PER_RESPONSE'] - 1
        while prevFound > 0:
            url = buildURL() + f"&$offset={offset}"
            results = callAPI(url)
            if results is None:
                prevFound = 0
            else:
                prevFound = len(results)
                if prevFound > 0:
                    df = pd.json_normalize(results)
                    fileName = f"NewYork_Crime_{vars_['START_DATE']}_{vars_['END_DATE']}_{str(offset).zfill(20)}.csv"
                    p = Path(os.getcwd(), vars_['SAVE_TO_DIRECTORY'])
                    df.to_csv(p / fileName, index=False)
                    offset += vars_['RES_PER_RESPONSE']
                    time.sleep(5)

    getDataByChunks()


def mergeCSV():
    path = Path(os.getcwd(), vars_['SAVE_TO_DIRECTORY'])
    all_files = glob(os.path.join(path, "*.csv"))
    concatenated = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        concatenated.append(df)
    if concatenated:
        frame = pd.concat(concatenated, axis=0, sort=True)
        frame_header = f"NYPD_Complaint_Data_{vars_['SAVE_FILE_ENDING']}.csv"
        frame.to_csv(frame_header, index=False)


def preprocessData():
    def preprocessHistoric(dataURI):
        isValidAndSetDefaults(dataURI, "2018-01-01", "2018-06-30", "data")
        getData()
        mergeCSV()

    def preprocessYearToDate(dataURI):
        isValidAndSetDefaults(dataURI, "2019-01-01", "2019-06-30", "data")
        getData()
        mergeCSV()

    preprocessHistoric("qgea-i56i")
    preprocessYearToDate("5uac-w243")

    df_historic = pd.read_csv("NYPD_Complaint_Data_Historic.csv", index_col=0)
    df_ytd = pd.read_csv("NYPD_Complaint_Data_YTD.csv", index_col=0)
    return df_historic, df_ytd


def trimNANColumns(df):
    length_df = df.shape[0]
    trim_cols = [col for col in df.columns if df[col].isnull().sum() / length_df > 0.5]
    trimmed_df = df.drop(columns=trim_cols)
    return trimmed_df


def fillMissingData(df):
    def fillMissingComplaintDate(df):
        return df[['cmplnt_fr_dt', 'cmplnt_to_dt', 'rpt_dt']]

    def fillNANsInComplaintDate(row):
        if pd.isnull(row["cmplnt_fr_dt"]):
            if pd.notnull(row["cmplnt_to_dt"]):
                return row["cmplnt_to_dt"]
            elif pd.notnull(row["rpt_dt"]):
                return row['rpt_dt']
            else:
                return row["cmplnt_fr_dt"]
        else:
            return row["cmplnt_fr_dt"]

    df_fm = fillMissingComplaintDate(df)
    df['cmplnt_fr_dt'] = df.apply(fillNANsInComplaintDate, axis=1)
    return df, df_fm


def reTrimAndRenameColumns(df):
    df_retrim = df.drop(columns=['cmplnt_to_dt', 'cmplnt_to_tm', 'rpt_dt', 'loc_of_occur_desc'])
    df_retrim.rename(columns={
        'cmplnt_num': 'ComplaintID',
        'cmplnt_fr_dt': 'Date',
        'cmplnt_fr_tm': 'Time',
        'ky_cd': 'Offence Code',
        'ofns_desc': 'Description',
        'pd_cd': 'Internal Code',
        'pd_desc': 'Internal Description',
        'crm_atpt_cptd_cd': 'Status',
        'law_cat_cd': 'Offence Level',
        'boro_nm': 'Borough',
        'addr_pct_cd': 'Neighborhood',
        'prem_typ_desc': 'Premise Description'
    }, inplace=True)
    df_retrim.reset_index(inplace=True)
    return df_retrim.drop(columns='index')


def reduceMemory(df):
    for col in ['Offence Code', 'Internal Code', 'Status', 'Offence Level', 'Neighborhood', 'Borough']:
        df[col] = df[col].astype('category')
    df['Internal Code'] = df["Internal Code"].astype(int).astype(str).astype('category')
    df['Offence Code'] = df["Offence Code"].astype(str).astype('category')
    return df


def extractYearAndMonth(df):
    df['Year'] = df.Date.str[:4]
    df['Month'] = df.Date.str[5:7]
    return df


def processDates(df):
    df['Date'] = df['Date'].str.split('T').str[0]
    df['Datetime'] = df['Date'] + ' ' + df['Time']
    df["Date"] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    return df


def load_cleaned_dataset():
    return pd.read_csv('NYPD_Complaint_Data_Cleaned.csv', index_col='Date', parse_dates=True, low_memory=False)


def formGroups(df):
    df_count_y_cid = df.groupby(['Year'])[['ComplaintID']].count().reset_index()
    df_count_yb_cid = df.groupby(['Year', 'Borough'])[['ComplaintID']].count().reset_index()
    return df_count_y_cid, df_count_yb_cid


def crimeCountByYear(df_count_by_year_cid):
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(df_count_by_year_cid.Year, df_count_by_year_cid.ComplaintID)
    sns.set(font_scale=1.5)
    regplot = sns.lmplot(x='Year', y='ComplaintID', data=df_count_by_year_cid, height=7, aspect=1.5)
    regplot.set_axis_labels("Year", "Number of Offences")
    plt.title(f"Crime Rate in NYC, correlation = {rvalue:.2f}, R2 = {rvalue ** 2:.2f}, p-value = {pvalue:.4f}, slope = {slope:.2f}")
    plt.show()


def crimeCountByBorough(df_count_by_year_borough_cid):
    flatui = ["#003f5c", "#22af5d", "#bc5090", "#ff6361", "#ffa600", "#3ecc71"]
    sns.set(font_scale=2)
    tplot = sns.lmplot(x='Year', y='ComplaintID', data=df_count_by_year_borough_cid, hue='Borough', palette=flatui, height=10, aspect=1.5)
    tplot.set_axis_labels("Year", "Number of Crimes")
    plt.title("Crime Rate in New York(per Borough)")
    plt.show()


if __name__ == "__main__":
    dfH, dfY = preprocessData()
    dfHplusY = pd.concat([dfH, dfY], sort=True)
    dfHplusY_trimmed = trimNANColumns(dfHplusY)
    dfHplusY_trim_filled, _ = fillMissingData(dfHplusY_trimmed)
    dfHplusY_cleaned = reTrimAndRenameColumns(dfHplusY_trim_filled)
    dfHplusY_cleaned = reduceMemory(dfHplusY_cleaned)
    dfHplusY_cleaned = extractYearAndMonth(dfHplusY_cleaned)
    dfHplusY_cleaned = processDates(dfHplusY_cleaned)
    dfHplusY_cleaned.to_csv('NYPD_Complaint_Data_Cleaned.csv')
