# Predict price for tomorrow
st.subheader("Predicted Price for Tomorrow")
last_100 = google_data['Close'][-100:].values.reshape(-1, 1)
scaled_last_100 = scaler.transform(last_100)
scaled_last_100 = scaled_last_100.reshape(1, 100, 1)
tomorrow_pred_scaled = model.predict(scaled_last_100)
tomorrow_pred = scaler.inverse_transform(tomorrow_pred_scaled)[0][0]
st.write(f"**Predicted price for tomorrow: {tomorrow_pred:.2f}**")
