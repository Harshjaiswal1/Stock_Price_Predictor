# 📈 Stock Price Predictor App

A simple Streamlit web application that predicts stock prices using a pre-trained Keras deep learning model. This app visualizes historical trends and compares predicted stock prices against real market data.

---

## 🚀 Features

- Enter any stock ticker (e.g., `GOOG`, `AAPL`, `MSFT`) to load 20 years of historical data.
- Visualize moving averages (100, 200, and 250 days).
- Predict future stock prices using a deep learning model.
- Compare predictions with actual stock prices using dynamic plots.

---

## 📂 Project Structure

```
.
├── app.py                         # Streamlit app source code
├── Latest_stock_price_model.keras  # Pre-trained Keras model
├── requirements.txt              # Required packages
└── README.md                     # Project documentation
```

---

## 📊 Model Overview

- **Architecture**: Deep learning model built with Keras (e.g., LSTM or GRU).
- **Input**: 100-day window of normalized `Close` prices.
- **Output**: Next-day predicted price (scaled and then inverse-transformed).
- **Preprocessing**:
  - Uses `MinMaxScaler` for feature scaling.
  - Takes the last 30% of the dataset for testing and evaluation.

---

## 🧪 Technologies Used

- **Python**
- **Streamlit** for UI
- **Keras / TensorFlow** for model
- **pandas / numpy** for data handling
- **matplotlib** for plotting
- **yfinance** to fetch live historical stock data

---

## 🛠️ Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure you have the model file:**
   Place `Latest_stock_price_model.keras` in the project root.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---



## 📝 Notes

- This app uses only the `Close` price to predict future prices. It can be extended to include features like volume, open/high/low prices, or sentiment analysis.
- The model should be periodically updated to retain prediction accuracy.

---

## 📌 Example

> **Input:** `GOOG`  
> **Output:** Charts displaying real vs predicted prices, and historical trends over 20 years.

---
