import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf


stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.8)
x_test = google_data[['Close']].iloc[splitting_len:]

# Check for empty or all-NaN data
if x_test.empty or x_test['Close'].isnull().all():
    st.error("Test data is empty or contains only NaN values. Please check your data and splitting logic.")
    st.stop()

# Optionally, drop NaNs (if you want to proceed with available data)
x_test = x_test.dropna()

if len(x_test) < 100:
    st.error("Not enough data in test set after dropping NaNs. Need at least 100 rows.")
    st.stop()

def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data.index, full_data.Close, 'b', label='Close')
    plt.plot(full_data.index, values, 'orange', label='MA/Values')
    if extra_data and extra_dataset is not None:
        plt.plot(full_data.index, extra_dataset, 'g', label='Extra')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    return fig


st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])  # <-- FIXED

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data.reshape(-1, 1))  # <-- FIXED

ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data.index[:splitting_len+100], google_data.Close[:splitting_len+100], label="Data- not used")
plt.plot(ploting_data.index, ploting_data['original_test_data'], label="Original Test data")
plt.plot(ploting_data.index, ploting_data['predictions'], label="Predicted Test data")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# Show Latest Prediction
st.subheader("Latest Predicted Price")
latest_pred = ploting_data['predictions'].iloc[-1]
latest_date = ploting_data.index[-1]
st.write(f"**{stock} predicted price on {latest_date.date()}: {latest_pred:.2f}**")

# Predict price for tomorrow
st.subheader("Predicted Price for Tomorrow")
last_100 = google_data['Close'][-100:].values.reshape(-1, 1)
scaled_last_100 = scaler.transform(last_100)
scaled_last_100 = scaled_last_100.reshape(1, 100, 1)
tomorrow_pred_scaled = model.predict(scaled_last_100)
tomorrow_pred = scaler.inverse_transform(tomorrow_pred_scaled)[0][0]
st.write(f"**Predicted price for tomorrow: {tomorrow_pred:.2f}**")
