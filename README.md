
# Stock Price Prediction Using GRU with Fourier Features
Overview
This project aims to predict the closing price of a stock (Apple Inc. - AAPL) using a GRU (Gated Recurrent Unit) neural network. The model is trained on historical stock data fetched from Yahoo Finance and incorporates Fourier features for better performance. The project also includes evaluation metrics and visualization of predictions against actual stock prices.

Evaluation Metrics
Evaluation for Timeframe 2024-01-01 to 2024-06-04:
The model's performance on the test set is as follows:

RMSE (Root Mean Squared Error): 1.98
MAE (Mean Absolute Error): 1.78
MAPE (Mean Absolute Percentage Error): 0.93%
MFE (Mean Forecast Error): -1.75
MAD (Mean Absolute Deviation): 1.78
R2 (R-squared score): -0.83


Project Structure
best_gru_model.h5: The saved GRU model with the best performance.
main.py: The main script to train, evaluate, and visualize the model's predictions.
README.md: Project documentation.
Dependencies
The following Python libraries are required:

yfinance
numpy
pandas
tensorflow
scikit-learn
matplotlib
datetime
Install the required libraries using pip:
pip install yfinance numpy pandas tensorflow scikit-learn matplotlib datetime

Data Preparation
Data Fetching
The historical stock data is fetched using the yfinance library. Data for Apple Inc. (AAPL) from January 1, 2020, to January 1, 2024, is downloaded and used for training and evaluation.

Feature Engineering
Fourier features are created to capture periodic patterns in the stock prices. These features help the model to learn and predict cyclical trends in the stock market data.

Model Training
Data Scaling
The features and 'Close' price are scaled using MinMaxScaler. This scaling is crucial for the neural network to learn effectively.

Sequence Creation
Sequences of length 100 are created from the scaled data to be used as input for the GRU model. This helps the model to capture temporal dependencies in the data.

Model Architecture
A GRU model is built and trained with various hyperparameters using a random search method to find the best model. The architecture includes two GRU layers with dropout for regularization.

model = Sequential([
    GRU(units, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(dropout_rate),  # Add dropout regularization
    GRU(units),
    Dropout(dropout_rate),  # Add dropout regularization
    Dense(1)
])

Hyperparameter Tuning
A random search method is used to find the best hyperparameters (units, batch size, learning rate, dropout rate) by evaluating the model's performance on the validation set.



Evaluation
Metrics
The model's performance is evaluated using the following metrics:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE)
Mean Forecast Error (MFE)
Mean Absolute Deviation (MAD)
R-squared (R2) score
Results
The best model is saved and its performance is evaluated on the test set. The actual and predicted prices are plotted for visual comparison.

Future Work
1. Enhance Model Architecture
Experiment with different neural network architectures like LSTM, Transformer models, or hybrid models.
Perform extensive hyperparameter tuning using techniques such as grid search or Bayesian optimization.

3. Feature Engineering
Include more features like trading volume, technical indicators (e.g., RSI, MACD), sentiment analysis from news or social media, and macroeconomic indicators.
Integrate features like lagged values, moving averages, and rolling statistics to better capture trends and seasonality.

4. Risk Management
Integrate risk management metrics like Value at Risk (VaR) and Conditional VaR to provide more comprehensive insights.
Conduct stress tests to evaluate the model's performance under extreme market conditions.

5. User Interface and Visualization
Develop interactive dashboards for visualizing predictions and model performance, allowing users to interact with and customize their views.
Implement a feedback loop where users can provide feedback on predictions, helping to continuously improve the model.
