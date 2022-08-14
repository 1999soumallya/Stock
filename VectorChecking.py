from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

import pandas as pd
import warnings
import plotly.graph_objects as go
import plotly.offline as py

warnings.filterwarnings('ignore')
CL = pd.read_csv('CL=F.csv', index_col=0, parse_dates=True, )

company = ['BPCL.NS', 'IOC.NS', 'HINDPETRO.NS']
for i in range(len(company)):
    print(f"Result for {company[i]}")
    companyName = company[i].split()
    variable = companyName[0]
    variable = pd.read_csv(f'{company[i]}.csv', index_col=0, parse_dates=True )
    variable = variable.merge(CL, on='Date')
    variable = variable.dropna()
    print(variable.info())
    # add files with CL
    ad_fuller_result_1 = adfuller(variable[f'{company[i]}(percentage_actual%)'].diff()[1:])
    print(f'{company[i]}(percentage_predict%)')
    print(f'ADF Statistic: {ad_fuller_result_1[0]}')
    print(f'p-value: {ad_fuller_result_1[1]}')
    print('\n---------------------\n')

    ad_fuller_result_2 = adfuller(variable[f'{company[i]}(percentage_predict%)'].diff()[1:])
    print(f'{company[i]}(percentage_predict%)')
    print(f'ADF Statistic: {ad_fuller_result_2[0]}')
    print(f'p-value: {ad_fuller_result_2[1]}')
    print('\n---------------------\n')

    ad_fuller_result_3 = adfuller(variable['CL=F(percentage_actual%)'].diff()[1:])
    print('CL=F(percentage_actual%)')
    print(f'ADF Statistic: {ad_fuller_result_3[0]}')
    print(f'p-value: {ad_fuller_result_3[1]}')
    print('\n---------------------\n')

    ad_fuller_result_4 = adfuller(variable['CL=F(percentage_predict%)'].diff()[1:])
    print('CL=F(percentage_predict%)')
    print(f'ADF Statistic: {ad_fuller_result_4[0]}')
    print(f'p-value: {ad_fuller_result_4[1]}')
    print('\n---------------------\n')

    print(f'CL=F(percentage_actual%) cause {company[i]}(percentage_actual%)?\n')
    print('------------------')
    granger_1 = grangercausalitytests(variable[[f'{company[i]}(percentage_actual%)', 'CL=F(percentage_actual%)']], 5)

    print(f'\n{company[i]}(percentage_actual%) causes CL=F(percentage_actual%)?\n')
    print('------------------')
    granger_2 = grangercausalitytests(variable[['CL=F(percentage_actual%)', f'{company[i]}(percentage_actual%)']], 5)

    train_df = variable[:-197]
    print('Train Data')
    print(train_df)
    # train_df = train_df.reset_index().set_index('Date',append=True)
    test_df = variable[-197:]
    print('Test Data')
    print(test_df.shape)

    # var find
    model = VAR(train_df.diff()[1:])
    sorted_order = model.select_order(maxlags=8)
    print(sorted_order.summary())

    # fit model lag 10
    var_model = VARMAX(train_df, order=(50, 0), enforce_stationarity=True)
    fitted_model = var_model.fit(disp=True)
    print(fitted_model.summary())

    # forcast
    n_forecast = 25
    predict = fitted_model.get_prediction(start=0, end=len(train_df))

    print('predict', predict)
    predictions = predict.predicted_mean
    print('prediction', predictions)

    test_vs_pred = pd.concat([test_df, predictions], axis=1)

    predictions.plot(figsize=(12, 5), kind='bar')
    train_df.plot(figsize=(12, 5), kind='bar')

    train_df = train_df.reset_index()
    train_df['date'] = pd.to_datetime(train_df["date"].dt.strftime('%d-%m-%Y'))
    print(train_df)

    graph_actual = go.Bar(y=train_df[f'{company[i]}(percentage_actual%)'], x=train_df['date'].astype(dtype=str),
                          name=f'{variable}_actual%')
    cl_actual = go.Bar(y=train_df['CL=F(percentage_actual%)'], x=train_df['date'], name='cl_actual%')
    graph_predict = go.Bar(y=predictions[f'{company[i]}(percentage_predict%)'], x=train_df['date'],
                           name=f'{variable}_predict%')
    cl_predict = go.Bar(y=predictions['CL=F(percentage_predict%)'], x=train_df['date'], name='cl_predict%')
    py.plot([graph_actual, cl_actual, graph_predict, cl_predict])
