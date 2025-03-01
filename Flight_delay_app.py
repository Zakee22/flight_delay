import streamlit as st
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce

# Load the saved model and target encoder
model = pickle.load(open('model.pkl', 'rb'))
target_encoder = pickle.load(open('target_encoder.pkl', 'rb'))  # Assuming you saved the encoder

# Function for preprocessing input
def preprocess_input(Month, DayofMonth, DayOfWeek, DepTime, UniqueCarrier, Origin, Dest, Distance):
    # Convert inputs into pandas dataframe
    input_data = pd.DataFrame({
        'Month': [Month],
        'DayofMonth': [DayofMonth],
        'DayOfWeek': [DayOfWeek],
        'DepTime': [DepTime],
        'UniqueCarrier': [UniqueCarrier],
        'Origin': [Origin],
        'Dest': [Dest],
        'Distance': [Distance]
    })
    
    # Feature Engineering
    def get_time_of_day(DepTime):
        if 500 <= DepTime < 1200:
            return 'Morning'
        elif 1200 <= DepTime < 1700:
            return 'Afternoon'
        elif 1700 <= DepTime < 2100:
            return 'Evening'
        else:
            return 'Night'

    input_data['DepTimeOfDay'] = input_data['DepTime'].apply(get_time_of_day)
    
    # Distance categories
    def assign_distance(distance):
        if distance < 500:
            return 'Short'
        elif 500 <= distance < 1500:
            return 'Medium'
        else:
            return 'Long'

    input_data['DistanceCategory'] = input_data['Distance'].apply(assign_distance)

    input_data['Carrier_Origin'] = input_data['UniqueCarrier'] + '_' + input_data['Origin']
    input_data['Carrier_Dest'] = input_data['UniqueCarrier'] + '_' + input_data['Dest']

    input_data['Day_Month'] = input_data['DayOfWeek'].astype(str) + '_' + input_data['Month'].astype(str)
    
    # Apply Target Encoding for high cardinality columns
    input_data_high_cardinality_cols = ['Carrier_Origin', 'Carrier_Dest', 'Day_Month', 'UniqueCarrier', 'Origin', 'Dest']
    input_data[input_data_high_cardinality_cols] = target_encoder.transform(input_data[input_data_high_cardinality_cols])

    # One-Hot Encoding for low cardinality columns
    input_data_low_cardinality_cols = ['DepTimeOfDay', 'DistanceCategory']
    input_data = pd.get_dummies(input_data, columns=input_data_low_cardinality_cols, drop_first=False)  # Keep all dummies for now
    input_data.fillna(0, inplace=True)

    # Drop unused columns
    input_data.drop(['DepTime', 'Distance'], axis=1, inplace=True)

    # Now that the data is ready for prediction, select only the relevant features for prediction
    feature_columns = ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest', 'DepTimeOfDay_Afternoon',
                       'DepTimeOfDay_Evening', 'DepTimeOfDay_Morning', 'DepTimeOfDay_Night','DistanceCategory_Long',
                       'DistanceCategory_Medium', 'DistanceCategory_Short', 'Carrier_Origin', 
                       'Carrier_Dest', 'Day_Month']
    
    # Ensure all the required columns are present in the input data
    missing_columns = [col for col in feature_columns if col not in input_data.columns]
    for col in missing_columns:
        input_data[col] = 0  # Add missing columns with value 0

    # Ensure the input_data is in the correct column order
    input_data = input_data[feature_columns]
    
    return input_data  # Return the processed data

# Streamlit UI


st.title("Flight Delay Prediction")

# Introduction
st.markdown("""
    **Welcome to the Flight Delay Prediction app!** 
    This app uses machine learning to predict whether your flight will be delayed or arrive on time. 
    Just enter the flight details below and get your prediction!
""")

# Form layout for input
with st.form(key='flight_form'):
    st.header("Enter Flight Details")
    
    # Month selection from January to December, passing value as 1-12
    month_options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    Month = st.selectbox("Month", month_options)
    Month = month_options.index(Month) + 1  # Convert to corresponding number 1-12
    
    # DayofMonth selection from 1-31
    DayofMonth = st.selectbox("Day of Month", list(range(1, 32)))  # Dropdown for 1-31
    
    # DayOfWeek selection from Monday-Sunday, passing value as 1-7
    day_of_week_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    DayOfWeek = st.selectbox("Day of Week", day_of_week_options)
    DayOfWeek = day_of_week_options.index(DayOfWeek) + 1  # Convert to corresponding number 1-7
    
    # DepTime remains as it is
    DepTime = st.number_input("Departure Time (HHMM)", min_value=0, max_value=2359, value=800)
    
    # UniqueCarrier selection from the provided list
    carrier_options = ['AA', 'AQ', 'AS', 'B6', 'CO', 'DH', 'DL', 'EV', 'F9', 'FL', 'HA', 'HP', 'MQ', 'NW', 'OH', 'OO', 
                       'TZ', 'UA', 'US', 'WN', 'XE', 'YV']
    UniqueCarrier = st.selectbox("Unique Carrier", carrier_options)
    
    # Origin selection from the provided list
    origin_options = ['ABE', 'ABI', 'ABQ', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 
                      'AMA', 'ANC', 'APF', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR', 
                      'BHM', 'BIL', 'BIS', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRO', 'BRW', 'BTM', 'BTR', 
                      'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO', 'CHS', 'CIC', 'CID', 
                      'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'COD', 'COS', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'DAB', 
                      'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DLG', 'DLH', 'DRO', 'DSM', 'DTW', 'EGE', 'EKO', 'ELP', 
                      'ERI', 'EUG', 'EVV', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 
                      'FSM', 'FWA', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRK', 'GRR', 'GSO', 'GSP', 'GST', 'GTF', 
                      'GTR', 'GUC', 'HDN', 'HKY', 'HLN', 'HNL', 'HOU', 'HPN', 'HRL', 'HSV', 'HTS', 'HVN', 'IAD', 'IAH', 'ICT', 
                      'IDA', 'ILG', 'ILM', 'IND', 'IPL', 'ISO', 'ISP', 'ITO', 'IYK', 'JAC', 'JAN', 'JAX', 'JFK', 'JNU', 'KOA', 
                      'KTN', 'LAN', 'LAS', 'LAW', 'LAX', 'LBB', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LRD', 
                      'LSE', 'LWB', 'LWS', 'LYH', 'MAF', 'MBS', 'MCI', 'MCN', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 
                      'MGM', 'MHT', 'MIA', 'MKE', 'MLB', 'MLI', 'MLU', 'MOB', 'MOD', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP', 
                      'MSY', 'MTJ', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'OTZ', 'OXR', 'PBI', 
                      'PDX', 'PFN', 'PHF', 'PHL', 'PHX', 'PIA', 'PIE', 'PIH', 'PIT', 'PNS', 'PSC', 'PSE', 'PSG', 'PSP', 'PVD', 
                      'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RFD', 'RIC', 'RNO', 'ROA', 'ROC', 'RST', 'RSW', 'SAN', 'SAT', 'SAV', 
                      'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF', 'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 
                      'SLC', 'SMF', 'SMX', 'SNA', 'SOP', 'SPI', 'SPS', 'SRQ', 'STL', 'STT', 'STX', 'SUN', 'SWF', 'SYR', 'TEX', 
                      'TLH', 'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUP', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'VCT', 'VIS', 
                      'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'YAK', 'YUM']
    Origin = st.selectbox("Origin", origin_options)

    # Destination selection from the provided list
    Dest = st.selectbox("Destination", origin_options)
    
    # Distance remains as it is
    Distance = st.number_input("Distance (in miles)", min_value=1, max_value=10000, value=2500)

    # Button to submit the form
    submit_button = st.form_submit_button("Submit")

# If submit button is pressed
if submit_button:
    # Preprocess the input data
    processed_input = preprocess_input(Month, DayofMonth, DayOfWeek, DepTime, UniqueCarrier, Origin, Dest, Distance)

    # Make the prediction
    prediction = model.predict(processed_input)

    # Display the result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("The flight will be on time!")
    else:
        st.error("The flight will be delayed.")
