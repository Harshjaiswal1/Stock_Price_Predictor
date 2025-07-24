📊 Model Overview
Architecture: Deep learning model built with Keras (e.g., LSTM or GRU).

Input: 100-day window of normalized Close prices.

Output: Next-day predicted price (scaled and then inverse-transformed).

Preprocessing:

Uses MinMaxScaler for feature scaling.

Takes the last 30% of the dataset for testing and evaluation.

🧪 Technologies Used
Python

Streamlit for UI

Keras / TensorFlow for model

pandas / numpy for data handling

matplotlib for plotting

yfinance to fetch live historical stock data

🛠️ Setup Instructions
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Make sure you have the model file:
Place Latest_stock_price_model.keras in the project root.

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py


📝 Notes
This app uses only the Close price to predict future prices. It can be extended to include features like volume, open/high/low prices, or sentiment analysis.

The model should be periodically updated to retain prediction accuracy.

📌 Example
Input: GOOG
Output: Charts displaying real vs predicted prices, and historical trends over 20 years.

📃 License
This project is open-source and available under the MIT License.
