import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the trained model
model = load_model('stock_dl_model.keras')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'AAPL'  # Fallback default

        # Download data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.today()
        current_month_year = end.strftime("%b %Y")
        df = yf.download(stock, start=start, end=end)
        data_desc = df.describe()

        # Calculate EMAs
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Train/Test Split
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare input for model
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict on historical data
        y_predicted = model.predict(x_test)

        # Fix scaling calculation
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # === FUTURE PREDICTION ===
        n_future_days = 30
        future_predictions = []

        last_100_days = input_data[-100:]  # Shape (100, 1)
        input_seq = last_100_days.reshape(1, 100, 1)

        for _ in range(n_future_days):
            next_pred = model.predict(input_seq)[0][0]
            future_predictions.append(next_pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

        future_predictions_rescaled = np.array(future_predictions) * scale_factor
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=n_future_days, freq='B')

        # === PLOTS ===

        # EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Historical prediction
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Future prediction
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(future_dates, future_predictions_rescaled, 'b', label='Future Predicted Price')
        ax4.set_title(f"{stock} - Next {n_future_days} Day Forecast")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Predicted Price")
        ax4.legend()
        future_prediction_chart_path = "static/future_predictions.png"
        fig4.savefig(future_prediction_chart_path)
        plt.close(fig4)

        # Save dataset
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template('index.html',
                       plot_path_ema_20_50=ema_chart_path,
                       plot_path_ema_100_200=ema_chart_path_100_200,
                       plot_path_prediction=prediction_chart_path,
                       plot_path_future_prediction=future_prediction_chart_path,
                       data_desc=data_desc.to_html(classes='table table-bordered'),
                       dataset_link=csv_file_path,
                       current_month_year=current_month_year)

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
