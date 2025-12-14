import streamlit as st
import pickle
import pandas as pd
import base64

# --- Page Config ---
st.set_page_config(page_title="IPL Predictor", page_icon="🏏")

# --- Function for Background Video ---
def add_bg_video(video_file):
    try:
        with open(video_file, "rb") as file:
            encoded_string = base64.b64encode(file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}
        </style>
        <video autoplay muted loop id="myVideo" style="position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; opacity: 0.7;">
            <source src="data:video/mp4;base64,{encoded_string.decode()}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Could not find {video_file}. Please add a video file or check the name.")

# --- CSS for Styling ---
def add_custom_css():
    st.markdown(
    """
    <style>
    /* Text Color White */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: white !important;
        text-shadow: 2px 2px 4px #000000;
    }
    /* Input Fields White Background */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border-radius: 5px;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid white;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

# --- Run UI Setup ---
add_bg_video('background.mp4')
add_custom_css()

# --- Load Model ---
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

st.title('IPL Cricket Score Predictor 🏏')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

city = st.selectbox('Select Host City', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    # step=1 prevents decimal inputs like 184.08
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Done (Works for > 5 overs)', min_value=5.0, max_value=19.5, step=0.1)
with col5:
    wickets = st.number_input('Wickets Out', min_value=0, max_value=9, step=1)

last_five = st.number_input('Runs Scored in Last 5 Overs', min_value=0, step=1)

if st.button('Predict Score'):
    # Prepare Input
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'wickets': [wickets],
        'overs': [overs],
        'runs_last_5': [last_five]
    })

    # Predict
    result = pipe.predict(input_df)

    # Show Result
    st.header("Predicted Score: " + str(int(result[0])))
