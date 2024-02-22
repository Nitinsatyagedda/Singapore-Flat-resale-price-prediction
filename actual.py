import pandas as pd
def load_data():
    df1=pd.read_csv("1990-1999.csv")
    df2=pd.read_csv("2000-Feb2012.csv")
    df3=pd.read_csv("2012Mar-Dec2014.csv")
    df4=pd.read_csv("Jan2015-Dec2016.csv")
    df5=pd.read_csv("Jan2017onwards.csv")
    df=pd.concat([df1,df2,df3,df4,df5])
    return df
df=load_data()
df.info()
from sklearn.preprocessing import LabelEncoder
def preprocess_data(df):
    # Handle missing values with median
    df=df.drop('remaining_lease',axis=1)
    
    # Encode categorical variables using LabelEncoder
    cat_cols = ['month','town','flat_type','block','street_name','storey_range','flat_model','resale_price']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    outliers = (df < lower_bound) | (df > upper_bound)
    # Remove outliers
    df = df[~outliers.any(axis=1)]
    return df
df=preprocess_data(df)
from datetime import datetime
def remaining_lease(lease_commencement_date):
    current_year = datetime.now().year
    remaining_years = current_year - lease_commencement_date.dt.year
    return remaining_years
# Convert 'Lease Commencement Date' to datetime
df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'])

# Create new column 'Remaining_Lease_Years'
df['remaining_lease'] = remaining_lease(df['lease_commence_date'])

df['lease_commence_date'] = df['lease_commence_date'].dt.year.astype(int)
df.info()
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X = df[['month','town','flat_type','floor_area_sqm','flat_model','storey_range','lease_commence_date']]
y = df['resale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
def load_model():
    model=model = XGBRegressor(random_state=42)
    return model
def preprocess_input(user_input):
    cat_cols = ['month','town','flat_type','block','street_name','storey_range','flat_model','resale_price']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    preprocessed_input=pd.DataFrame(user_input,index=[0])
    return preprocessed_input
def predict_resale_price(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]
import streamlit as st
def main():
    st.title("Singapore Resale Flat Price Predictor")
    df=load_data()
    df=preprocess_data(df)

    model = load_model()

    st.sidebar.header('Enter Flat Details')   
    month = df["month"].unique()
    towns = df["town"].unique()
    flat_types = df["flat_type"].unique()
    storey_ranges = df["storey_range"].unique()
    floor_area_sqm = df["floor_area_sqm"].unique()
    flat_model=df["flat_model"].unique()
    lease_commence_date=df["lease_commence_date"].unique()
    
    present_month_year=st.sidebar.number_input("month")
    town = st.sidebar.text_input("Town")
    flat_type = st.selectbox("Flat Type",flat_types)
    storey_range = st.selectbox("Storey Range", storey_ranges)
    floor_area_sqm = st.number_input("floor_area_sqm",min_value=1.0,step=0.1)
    flat_model = st.text_input("Flat Model")
    lease_commence_date = st.number_input("lease_commence_date", min_value=1990,max_value=2024)

    input_data = {
        "month" : present_month_year,
        "town": town,
        "flat_type": flat_type,
        "storey_range": storey_range,
        "floor_area_sqm": floor_area_sqm,
        "flat_model": flat_model,
        "lease_commence_date": lease_commence_date
    }
    input_data = preprocess_input(input_data)
    #df_1 = pd.DataFrame([input_data])
    
    if st.button("Predict Resale Price"):
        prediction = predict_resale_price(model, input_data)
        prediction[0]
if __name__ == '__main__':
    main()
