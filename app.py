import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

uploaded_file=st.file_uploader("Choose a file")
model = pk.load(open('model.pkl','rb'))

st.header('Used Car Price Prediction')

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 0, 200000)
fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())
seller_type = st.selectbox('Seller type', cars_data['seller_type'].unique())
transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())
owner = st.selectbox('Owner type', cars_data['owner'].unique())
mileage = st.slider('Car Mileage', 0, 100)
engine = st.slider('Engine CC', 0, 5000)
max_power = st.slider('Max Power', 0, 200)
seats = st.slider('No of Seats', 0, 10)

# Function to validate input data
def validate_input(km_driven, mileage, engine, max_power, seats):
    if km_driven < 0:
        st.error("Kilometers driven cannot be negative.")
        return False
    if mileage < 0:
        st.error("Mileage cannot be negative.")
        return False
    if engine <= 0:
        st.error("Engine CC must be greater than zero.")
        return False
    if max_power <= 0:
        st.error("Max power must be greater than zero.")
        return False
    if seats < 1:
        st.error("At least one seat is required.")
        return False
    return True

if st.button("Predict"):
    # Validate inputs before prediction
    if validate_input(km_driven, mileage, engine, max_power, seats):
        input_data_model = pd.DataFrame(
            [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
            columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
        )
        
        input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
           'Fourth & Above Owner', 'Test Drive Car'],
                           [1, 2, 3, 4, 5], inplace=True)
        input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
        input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
        input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
        input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
           'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
           'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
           'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
           'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                          inplace=True)

        car_price = model.predict(input_data_model)

        st.markdown(f'**Predicted Price:** Rs. {car_price[0]:,.2f}')
