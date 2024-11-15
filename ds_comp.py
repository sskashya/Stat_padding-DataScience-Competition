import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import root_mean_squared_error

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
injury_history_raw['injury_number'] = injury_history_raw.groupby(['Player.ID', 'Body Part']).cumcount() + 1
injury_history_raw.sort_index(inplace = True)

twothird_data = pd.merge(muscle_imbalance_raw, player_sessions_raw, on = ['Player.ID', 'Month', 'Year'], how = 'right')
obt = pd.merge(twothird_data, injury_history_raw, on = ['Player.ID', 'Month', 'Year'], how = 'left')

distinct_obt = obt.drop(columns = ['Group.name', 'League.ID', 'Name_y', 'Group.Id_y', 'Name_x', 'Severity', 'Side'])
distinct_obt['Injury Type'].fillna("Not Injured", inplace=True)
distinct_obt['Body Part'].fillna("None", inplace = True)
distinct_obt['Injury Date'].fillna("12/31/2026", inplace = True)
distinct_obt['Recovery Time (days)'].fillna(0, inplace = True) 
distinct_obt['Additional Notes'].fillna("Not Injured", inplace = True)
distinct_obt['injury_number'].fillna(0, inplace = True)
distinct_obt['Injury Date'] = pd.to_datetime(distinct_obt['Injury Date'])
distinct_obt['Session_Date'] = pd.to_datetime(distinct_obt['Session_Date'])

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

columns_to_choose = [
    'Injury Type', 
    'Body Part', 
    'Recovery Time (days)']

model3_df = clean_sorted_obt[columns_to_choose]

model3_predictors = model3_df.drop(columns = 'Recovery Time (days)')
model3_target = model3_df['Recovery Time (days)']

ohe = OneHotEncoder(sparse_output = False, drop = None)
ohe_model3_predictors = ohe.fit_transform(model3_predictors)
ohe_model3_predictor_df = pd.DataFrame(ohe_model3_predictors, columns=list(ohe.get_feature_names_out()))
ohe_model3_predictors = ohe_model3_predictor_df.drop(columns = ['Injury Type_Not Injured', 'Body Part_None'])

rfr = RandomForestRegressor(n_estimators = 600)
#ohe_model3_predictors['injury_number'] = clean_sorted_obt['injury_number']
x_train, x_test, y_train, y_test = train_test_split(ohe_model3_predictors, model3_target, train_size = 0.7, random_state=999999)
cv_score = np.mean(cross_val_score(rfr, x_train, y_train, cv = 3, scoring = 'neg_root_mean_squared_error'))

rfr_model = rfr.fit(x_train, y_train)
rfr_pred = rfr_model.predict(x_test)
accuracy = root_mean_squared_error(y_test, rfr_pred)

st.title("Welcome back Coach")
st.header("Let's figure out how long your player is going to be out for")
st.text("Share the injury type of your player, the body part they injured (More data needs to be collected on Severity of Injury)")

injury_type = st.selectbox('Injury Type', ['Muscle Strain', 'Strain', 'Sprain', 'Tendonitis', 'Fracture', 'Concussion', 'Soreness', 'Pain'], index = None)
body_part = st.selectbox('Body Part', ['Quadriceps', 'Knee', 'Groin', 'Hamstring', 'Shoulder', 'Wrist', 'Head', 'Lower Back', 'Ankle', 'Foot'], index = None)
#frequency = st.select_slider('Frequency', [0, 1, 2, 3, 4])

if injury_type and body_part: #and frequency:
    dict = {'Injury Type' : [injury_type],
          'Body Part' : [body_part]}
    
    df = pd.DataFrame(dict, index = None)

    ohe_data = ohe.transform(df)
    ohe_df = pd.DataFrame(ohe_data, columns=list(ohe.get_feature_names_out()))
    #missing_col = set(ohe_model3_predictors.columns) - set(ohe_df.columns)
    #ohe_df[list(missing_col)].fillna(0)

    ohe_predictors = ohe_df.drop(columns = ['Injury Type_Not Injured', 'Body Part_None'])
    #ohe_predictors['injury_number'] = frequency

    new_pred = rfr_model.predict(ohe_predictors)
    st.write(f'Based on the data you have provided, I am estimating this player to be out for about {round(new_pred[0])} days')

    