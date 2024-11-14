import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

injury_history_raw = pd.read_csv("data/injury_history(injury_history).csv", sep = ",", encoding = 'ISO-8859-1')
muscle_imbalance_raw = pd.read_csv("data/injury_history(muscle_imbalance_data).csv", sep = ",", encoding = 'ISO-8859-1')
player_sessions_raw = pd.read_csv("data/injury_history(player_sessions).csv", sep = ",", encoding = 'ISO-8859-1')

injury_history_raw['Injury Date'] = pd.to_datetime(injury_history_raw['Injury Date'], errors = 'coerce')
injury_history_raw['Month'] = injury_history_raw['Injury Date'].dt.month
injury_history_raw['Year'] = injury_history_raw['Injury Date'].dt.year

player_sessions_raw['Month'] = pd.to_datetime(player_sessions_raw['Session_Date']).dt.month
player_sessions_raw['Year'] = pd.to_datetime(player_sessions_raw['Session_Date']).dt.year

muscle_imbalance_raw['Month'] = pd.to_datetime(muscle_imbalance_raw['Date Recorded']).dt.month
muscle_imbalance_raw['Year'] = pd.to_datetime(muscle_imbalance_raw['Date Recorded']).dt.year

injury_history_raw.sort_values(['Player.ID', 'Injury Date', 'Month', 'Year'], inplace = True)
injury_history_raw['injury_number'] = injury_history_raw.groupby('Player.ID').cumcount() + 1
injury_history_raw.sort_index(inplace = True)

twothird_data = pd.merge(muscle_imbalance_raw, player_sessions_raw, on = ['Player.ID', 'Month', 'Year'], how = 'right')
obt = pd.merge(twothird_data, injury_history_raw, on = ['Player.ID', 'Month', 'Year'], how = 'left')

distinct_obt = obt.drop(columns = ['Group.name', 'League.ID', 'Name_y', 'Group.Id_y', 'Name_x'])
distinct_obt['Injury Type'].fillna("Not Injured", inplace=True)
distinct_obt['Body Part'].fillna("None", inplace = True)
distinct_obt['Injury Date'].fillna("12/31/2026", inplace = True)
distinct_obt['Recovery Time (days)'].fillna(0, inplace = True) 
distinct_obt['Additional Notes'].fillna("Not Injured", inplace = True)
distinct_obt['injury_number'].fillna(0, inplace = True)
distinct_obt['Injury Date'] = pd.to_datetime(distinct_obt['Injury Date'])
distinct_obt['Session_Date'] = pd.to_datetime(distinct_obt['Session_Date'])
distinct_obt['injury_timeline'] = distinct_obt.apply(
     lambda row: 'before injury' if row['Session_Date'] < row['Injury Date'] else 'after injury',
    axis=1
)
distinct_obt['days_before_injury'] = distinct_obt.apply(
     lambda row: row['Injury Date'] - row['Session_Date'] if row['Injury Date'] != "2026-12-31" else 999,
    axis=1
)
distinct_obt.loc[distinct_obt['Injury Type'] == "Not Injured", 'Severity'] = distinct_obt.loc[distinct_obt['Injury Type'] == "Not Injured", 'Severity'].fillna("Grade 0")
distinct_obt.loc[distinct_obt['Injury Type'] == "Not Injured", 'Side'] = distinct_obt.loc[distinct_obt['Injury Type'] == "Not Injured", 'Side'].fillna("No Injury")

clean_obt = distinct_obt.dropna()
clean_sorted_obt = clean_obt.sort_values(['Player.ID', 'Session ID', 'Date Recorded', 'Session.ID', 'Session_Date', 'Injury Date'])
clean_sorted_obt['Date Recorded'] = pd.to_datetime(clean_sorted_obt['Date Recorded'])
clean_sorted_obt['Session_Date'] = pd.to_datetime(clean_sorted_obt['Session_Date'])
clean_sorted_obt['Injury Date'] = pd.to_datetime(clean_sorted_obt['Injury Date'])

columns_to_drop = [ 
    'Group.Id_x', 
    'Session ID', 
    'Session.ID', 
    'Changes.of.Orientation', 
    'Player Name', 
    'Additional Notes']
for column in columns_to_drop:
    if column in clean_sorted_obt.columns:
        clean_sorted_obt.drop(columns = column, inplace = True)