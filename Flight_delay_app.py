import streamlit as st
import pickle
import pandas as pd
import numpy as np
import category_encoders as ce
from datetime import datetime
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

# Load the saved model and target encoder
model = pickle.load(open('model.pkl', 'rb'))
target_encoder = pickle.load(open('target_encoder.pkl', 'rb'))

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Add background image (replace with your own image path)
try:
    add_bg_from_local('flight1.jpg')  # You should have a flight-themed background image
except:
    st.sidebar.warning("Background image not found. Using default background.")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 10px;
    }
    .stSelectbox, .stNumberInput, .stDateInput, .stTimeInput {
        background-color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1aumxhk {
        background-color: white;
    }
    .stForm {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

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
    input_data = pd.get_dummies(input_data, columns=input_data_low_cardinality_cols, drop_first=False)
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
        input_data[col] = 0

    # Ensure the input_data is in the correct column order
    input_data = input_data[feature_columns]
    
    return input_data

# Main content
with st.container():
    st.title("✈️ Flight Delay Prediction")
    
    # Introduction
    st.markdown("""
        <div style='background-color: rgba(255,255,255,0.7); padding: 1rem; border-radius: 10px;'>
        <h3 style='color: #0066cc;'>Welcome to the Flight Delay Prediction App!</h3>
        <p>This app uses machine learning to predict whether your flight will be delayed or arrive on time. 
        Just enter the flight details below and get your prediction!</p>
        </div>
    """, unsafe_allow_html=True)

    # Form layout for input with white background
    with st.form(key='flight_form'):
        st.markdown("""
        <style>
        div[data-testid="stForm"] {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.header("📋 Enter Flight Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Date input (calendar)
            selected_date = st.date_input("📅 Flight Date", min_value=datetime(2023, 1, 1))
            Month = selected_date.month
            DayofMonth = selected_date.day
            DayOfWeek = selected_date.weekday() + 1  # Monday=0 in datetime, so we add 1
            
            # Time input for departure time with clock-style picker
            selected_time = st.time_input("🕒 Departure Time", value=datetime.strptime("08:00", "%H:%M").time())
            DepTime = selected_time.hour * 100 + selected_time.minute
            
            # UniqueCarrier selection
            carrier_options = ['AA', 'AQ', 'AS', 'B6', 'CO', 'DH', 'DL', 'EV', 'F9', 'FL', 'HA', 'HP', 'MQ', 'NW', 'OH', 'OO', 
                             'TZ', 'UA', 'US', 'WN', 'XE', 'YV']
            UniqueCarrier = st.selectbox("🏢 Airline Carrier", carrier_options)
            
        with col2:
            # Origin selection
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
            Origin = st.selectbox("🛫 Departure Airport", origin_options)

            # Destination selection
            Dest = st.selectbox("🛬 Destination Airport", origin_options)
            
            # Distance input
            Distance = st.number_input("📏 Distance (in miles)", min_value=1, max_value=10000, value=2500)

        # Button to submit the form
        submit_button = st.form_submit_button("🚀 Predict Flight Delay")

# If submit button is pressed
if submit_button:
    # Preprocess the input data
    processed_input = preprocess_input(Month, DayofMonth, DayOfWeek, DepTime, UniqueCarrier, Origin, Dest, Distance)

    # Make the prediction
    prediction = model.predict(processed_input)

    # Display the result without balloons
    with st.spinner('Analyzing flight data...'):
        pass  # Just showing the spinner without balloons
    
    # Prediction result with solid white background
    st.markdown("""
    <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin-top: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
    <h5 style='color: #0066cc;'>📊 Prediction Result</h2>
    """, unsafe_allow_html=True)
    
    if prediction[0] == 1:
        st.markdown("""
        <div style='background-color: #e6f7e6; padding: 1rem; border-radius: 5px; border-left: 5px solid #33ffda; margin: 1rem 0;'>
        <h6 style='color: #000000;'>🎉 The flight will be on time!</h3>
        <p>You can expect your flight to depart and arrive as scheduled. Safe travels!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #ffebee; padding: 1rem; border-radius: 5px; border-left: 5px solid #F2D2BD; margin: 1rem 0;'>
        <h6 style='color: #000000;'>⚠️ The flight will likely be delayed.</h3>
        <p>Consider checking with your airline for updates and plan accordingly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Flight information with solid white background
    st.markdown(f"""
    <div style='background-color: white; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
    <h4 style='color: #0066cc;'>✈️ Flight Information</h4>
    <p><strong>Airline:</strong> {UniqueCarrier}</p>
    <p><strong>Route:</strong> {Origin} → {Dest}</p>
    <p><strong>Date:</strong> {selected_date.strftime("%B %d, %Y")}</p>
    <p><strong>Scheduled Departure:</strong> {selected_time.strftime("%I:%M %p")}</p>
    </div>
    </div>  <!-- Closing the main container -->
    """, unsafe_allow_html=True)
