import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance
import datetime as dt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import time
import itertools
import csv

now = time.time()
plt.style.use('fivethirtyeight')
company = ['CL=F', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS']
# Get the stock quote
for i in range(len(company)):
    filename = '%s.csv' % (company[i])
    f = open(filename, '+w')
    f.writelines(f'Date, {company[i]}(Actual Price), {company[i]}(Predict Price), {company[i]}(Difference), '
                 f'{company[i]}(percentage_actual%), {company[i]}percentage_predict(%)\n')
    f.close()

    filename1 = 'Testing/%s.csv' % (company[i])
    f1 = open(filename1, '+w')
    f1.writelines(f'Date, {company[i]}(Actual Price), {company[i]}(Predict Price)\n')
    f.close()
    data = yfinance.download(company[i], start='2012-01-01', end='2020-01-01')

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    prediction_days = 60
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaler_data)):
        x_train.append(scaler_data[x - prediction_days:x, 0])
        y_train.append(scaler_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the Neural Network Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_train, y_train), epochs=30, verbose=0, batch_size=30)

    ''' Test the Model Accuracy on Existing Data '''
    # Load Test Data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime(2021, 1, 1)
    ls_key = 'Adj Close'
    test_data = yfinance.download(company[i], start=test_start, end=test_end)
    actual_price = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    dates = []
    for x in range(len(test_data)):
        newdate = str(test_data.index[x])
        newdate = newdate[0:10]
        dates.append(newdate)
    test_data['dates'] = dates
    # Make prediction of Test Data
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict Next Day
    # real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    # real_data = np.array(real_data)
    # real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    # prediction = model.predict(real_data)
    # prediction = scaler.inverse_transform(prediction)
    # print(f"Prediction: {prediction}")
    current_time = time.time() - now
    print(current_time)

    history_dict = history.history
    # print(history_dict.keys())

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history_dict['accuracy'], label='train')
    pyplot.plot(history_dict['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    # Plot the test prediction
    plt.plot(actual_price, color="black", label=f"Actual {company[i]} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company[i]} Price")
    plt.title(f"{company[i]} Share Price")
    plt.xlabel('Time', fontsize=18)
    plt.ylabel(f"{company[i]} Price USD ($)", fontsize=18)
    plt.legend()
    plt.show()

    actual_price = list(actual_price)
    predicted_prices2 = list(itertools.chain.from_iterable(predicted_prices))
    Difference = []
    for a in range(len(actual_price)):
        for j in range(len(predicted_prices2)):
            if a == j:
                Difference.append(actual_price[a] - predicted_prices2[j])
            else:
                pass

    percentage_actual = []
    for k in range(len(actual_price)):
        if k == 0:
            percentage_actual.append(actual_price[k])
        else:
            result = (((actual_price[k] - actual_price[k - 1]) / actual_price[k]) * 100)
            percentage_actual.append(f'{result}')

    percentage_predict = []
    for m in range(len(predicted_prices2)):
        if m == 0:
            percentage_predict.append(predicted_prices2[m])
        else:
            result1 = (((predicted_prices2[m] - predicted_prices2[m - 1]) / predicted_prices2[m]) * 100)
            percentage_predict.append(f'{result1}')

    with open(filename, '+a') as f:
        writer = csv.writer(f)
        writer.writerows(zip(dates, actual_price, predicted_prices2, Difference, percentage_actual, percentage_predict))

    with open(filename1, '+a') as f1:
        writer = csv.writer(f1)
        writer.writerows(zip(dates, actual_price, predicted_prices2))
