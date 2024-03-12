# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Resale Unit', page_icon=':office:', layout='wide', initial_sidebar_state='collapsed')

# Set title of the app
st.title(':office: Predict Valuation of the Unit')

left, right = st.columns((4,4))

# Set input widgets
#st.sidebar.subheader('Select the following attributes')
num_rooms = left.selectbox('No. of Rooms', ('One Room', 'Two Room', 'Three Room', 'Four Room', 'Five Room', 'Executive',
                                 'Multigeneration'))
floor_category = right.selectbox('floor cat', (3, 2, 1))

floor_area = right.slider('What is the area of the house (sqm)',70, 200)

age_hdb = left.slider('The age of the HDB', 5, 70)
max_floor = right.slider('What is the maximum floor?', 2, 50)
mrt_distance = left.slider('Distance to nearest MRT', 0, 2500)

region = st.selectbox('Which region in Singapore?', ('North','North East','Central','West','East'))
# towns = ['Jurong West', 'Woodlands', 'Sengkang', 'Tampines', 'Yishun', 'Bedok', 'Punggol', 'Hougang', 'Ang Mo Kio',
#                  'Choa Chu Kang', 'Bukit Merah', 'Bukit Batok', 'Bukit Panjang', 'Toa Payoh', 'Pasir Ris', 'Queenstown', 
#                  'Geylang', 'Sembawang', 'Clementi', 'Jurong East', 'Kallang', 'Serangoon', 'Bishan', 'Novena', 'Marine Parade',
#                  'Outram', 'Rocher', 'Bukit Timah', 'Changi', 'Downtown Core', 'Tanglin', 'Western Water Catchment']
# towns.sort()

# township = left.selectbox('Location', towns)
# set default values
tranc_year_default = 2024
tranc_month_default = 3
multistorey_cp_default = 1
mall_within_2km_default = 1
hawker_within_2km_default = 1
bus_interchange_default = 0
floor_density_default = 0.5

# load the train model
model_file = open('hdb_model.pkl', 'rb')
model = pickle.load(model_file)

y_pred = model.predict([[floor_area, tranc_year_default, tranc_month_default, 
                        age_hdb, max_floor, multistorey_cp_default, mall_within_2km_default, 
                        hawker_within_2km_default, mrt_distance, bus_interchange_default,  floor_density_default, floor_category]])

region_to_num_dict = {'North':0.3,'North East':0.4,'Central':0.8,'West':0.1,'East':0.2}

add_on_constant = {'One Room':0.8, 'Two Room':1.4, 'Three Room':2.8, 'Four Room':3.7, 'Five Room':4.3, 'Executive':4.7,
                                 'Multigeneration':5.4}

estimated_value = round(y_pred[0] + 88888*add_on_constant[f'{num_rooms}']+288888*region_to_num_dict[f'{region}'],-3)
original_value = y_pred[0]

#r2_val = metrics.r2_score(y_test, y_pred)

# Display EDA
st.header('Estimated Valuation:')
#st.write('R2 value of this model')
#groupby_species_mean = hdb.groupby('Species').mean()
st.markdown(f'<h1 style="text-align: left;">${estimated_value:.0f}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="text-align: left;">${original_value:.0f}</h1>', unsafe_allow_html=True)

# Disclaimer
st.write('---')
st.write('**Disclaimer:**')
st.write('Please be aware that this model is still under development, and the valuations provided are for demonstration purposes only.')





