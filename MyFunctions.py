import pandas as pd
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
from copy import deepcopy
from pandas.plotting import autocorrelation_plot
from scipy import fftpack  # Fourier fft
import sys
import math
from datetime import date
from sklearn import base  # for transfomers
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from ipywidgets import widgets, HBox, VBox, Button, Layout
from IPython.display import display
from datetime import timedelta  

# print('The scikit-learn version is {}.'.format(sklearn.__version__))
# The scikit-learn version is 0.20.3.  # need version .21

##### FEATURES #####
def drift(df):
    new = df[['Calls']].copy()
    # This will take care of drift
    new['Julian'] = new.index.to_julian_date()
    new['const'] = 1
    return new

def drift2(df):
    new = df.copy()
    # This will take care of drift
    new['Julian'] = new.index.to_julian_date()
    new['const'] = 1
    ind_drift= {'Julian', 'const'}
    
    return new, ind_drift


# def previous_info(df):
#     # Use previous data
#     df['Calls_last_1d'] = df['Calls'].shift(1)
#     # Use value from last week
#     df['Calls_last_week'] = df['Calls'].shift(7)
#     # difference vs previous week, same day
#     # df['Calls_last_7d_diff'] = df['Calls'].diff(7)
#     # Average difference vs previous day over 4 full weeks
#     df['diff1_MA3m'] = pd.Series.rolling(df['Calls'].diff(), window=28).mean()
#     df['year_avg'] = pd.Series.ewm(df['Calls'], span=365).mean()
#     return df

# def smoothed():
#     # Will take the minimum of the same day of the week from the last two weeks: 
#     # min(Monday last week, Monday from twoo weeks ago)
#     df['smooth_prev_week'] = min(df['Calls'].shift(7), df['Calls'].shift(14))
    

def previous_day(df):
    # Use previous data all at least one day old
    # Use value from previous day
    df['Calls_last_1d'] = df['Calls'].shift(1)
    # creating a feature (smoothed_prev_week) dropping really large values 
    # by taking the minimum of the same days from the two previous weeks
    # min(Monday last week, Mondays last two weeks)
#     prev_1wk =  df['Calls'].shift(7)   #18
    df['prev_1wk'] = df['Calls'].shift(7)
#     prev_2wk =  df['Calls'].shift(14)   #18
    df['prev_2wk'] =  df['Calls'].shift(14)   #18
#     smooth = np.minimum(prev_1wk,prev_2wk)   #18 series
    smooth = np.minimum(df['prev_1wk'],df['prev_2wk'])   #18 series
    df['smoothed_prev_week'] = smooth
    df['diff1_smoothed_prev_week'] = df['smoothed_prev_week'].diff()
    # Average difference vs previous day over 4 full weeks
    df['diff1_MA3m_1d'] = pd.Series.rolling(df['Calls_last_1d'].diff(), window=28).mean()
    df['year_avg_1d'] = pd.Series.ewm(df['Calls_last_1d'], span=365).mean()
    df['diff7_MA3m_7d'] = pd.Series.rolling(df['prev_1wk'].diff(7), window=28).mean()
    df['year_avg_7d'] = pd.Series.ewm(df['prev_1wk'], span=365).mean()
    df['prev_year'] = df['Calls'].shift(365)
    ind_prev_day = {'Calls_last_1d','prev_1wk','prev_2wk','smoothed_prev_week','diff1_smoothed_prev_week'\
                    ,'diff1_MA3m_1d', 'year_avg_1d', 'diff7_MA3m_7d', 'year_avg_7d','prev_year'}
    return df, ind_prev_day

# def previous_week(df):
    # Use previous data all at least one week old
    # Use value from last week
#     df['Calls_7d'] = df['Calls'].shift(7)
    # Average difference vs previous day over 4 full weeks
#     df['diff7_MA3m_7d'] = pd.Series.rolling(df['Calls_7d'].diff(7), window=28).mean()
#     df['year_avg_7d'] = pd.Series.ewm(df['Calls_7d'], span=365).mean()
#     return df

def preFourier(df):
    df['sin(year)'] = np.sin(df['Julian'] / 365.25 * 2 * np.pi)
    df['cos(year)'] = np.cos(df['Julian'] / 365.25 * 2 * np.pi)
    df['sin(sem)'] = np.sin(df['Julian'] / (365.25 / 2) * 2 * np.pi)
    df['cos(sem)'] = np.cos(df['Julian'] / (365.25 / 2) * 2 * np.pi)
    df['sin(Qs)'] = np.sin(df['Julian'] / (365.25 / 4) * 2 * np.pi)
    df['cos(Qs)'] = np.cos(df['Julian'] / (365.25 / 4) * 2 * np.pi)
    df['sin(mo)'] = np.sin(df['Julian'] / (365.25 / 12) * 2 * np.pi)
    df['cos(mo)'] = np.cos(df['Julian'] / (365.25 / 12) * 2 * np.pi)
    df['sin(wk)'] = np.sin(df['Julian'] / 7 * 2 * np.pi)
    df['cos(wk)'] = np.cos(df['Julian'] / 7 * 2 * np.pi)
    df['sin(day)'] = np.sin(df['Julian'] * 2 * np.pi)
    df['cos(day)'] = np.cos(df['Julian'] * 2 * np.pi)
    
    ind_preFou = {'sin(year)', 'cos(year)', 'sin(sem)','cos(sem)','sin(Qs)','cos(Qs)'\
                  ,'sin(mo)','cos(mo)','sin(wk)','cos(wk)','sin(day)','cos(day)'}
    return df, ind_preFou


def Calendar_Set(df):
    df['Monday'] = np.where(df.index.weekday == 0, 1, 0)   #Monday: weekday=0
    df['Tuesday'] = np.where(df.index.weekday == 1, 1, 0)
    df['Wednesday'] = np.where(df.index.weekday == 2, 1, 0)
    df['Thursday'] = np.where(df.index.weekday == 3, 1, 0)
    df['Friday'] = np.where(df.index.weekday == 4, 1, 0)
    df['Saturday'] = np.where(df.index.weekday == 5, 1, 0)
    df['Sunday'] = np.where(df.index.weekday == 6, 1, 0)  # Sunday was missing from previous version

    df['Weekday_MT'] = np.where(df.index.weekday < 4, 1, 0)
    df['Weekend_FS'] = np.where((df.index.weekday == 4) |
                                (df.index.weekday == 5), 1, 0)

    df['Jan'] = np.where(df.index.month == 1, 1, 0)
    df['Feb'] = np.where(df.index.month == 2, 1, 0)
    df['Mar'] = np.where(df.index.month == 3, 1, 0)
    df['Apr'] = np.where(df.index.month == 4, 1, 0)
    df['May'] = np.where(df.index.month == 5, 1, 0)
    df['Jun'] = np.where(df.index.month == 6, 1, 0)
    df['Jul'] = np.where(df.index.month == 7, 1, 0)
    df['Aug'] = np.where(df.index.month == 8, 1, 0)
    df['Sep'] = np.where(df.index.month == 9, 1, 0)
    df['Oct'] = np.where(df.index.month == 10, 1, 0)
    df['Nov'] = np.where(df.index.month == 11, 1, 0)
    df['Dec'] = np.where(df.index.month == 12, 1, 0)
    
    ind_cal = {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Weekday_MT','Weekend_FS'\
               ,'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'}
    
    return df, ind_cal

def add_one_dependant(df, name):
    # Create Y (dependent)
    if name == 'Calls_Y_1d':
        df['Calls_Y_1d'] = df['Calls'].shift(-1)
    elif name == 'Calls_Y_7d':
        df['Calls_Y_7d'] = df['Calls'].shift(-7)
    return df

def holidays(df):
    # 4th Thursday will always be between 22 and 28 of November
    df['Thanks_Thu'] = np.where((df.index.month == 11) & (df.index.weekday == 3) &
                            ((df.index.day >= 22) & (df.index.day <= 28)), 1, 0)
    thursdays = list(np.where(df['Thanks_Thu'] == 1)[0])
    fridays = [t + 1 for t in thursdays]
    df['Thanks_Fri'] = 0
    col_fri = df.columns.get_loc('Thanks_Fri')
    df.iloc[fridays, col_fri] = 1
    
    ind_hol = {'Thanks_Thu','Thanks_Fri'}
    
    return df, ind_hol

def clean_wind_wt2():
#     wind_path = '~/Desktop/Gina/Cursos/DataIncubator/Project/Fire/weatherData/wind2003_2019.csv'
    wind_path = './wind2003_2019.csv'
    # 22 columns, 6056 days: 01/01/2003 to 07/31/2019
    df = pd.read_csv(wind_path)
    # convert string date to timestamp  (%y= ## for year)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    # AWND: Avg wind speed (few NaN), drop FMTM and PGTM many NaN
    # WSF2 fastest 2mins speed (no NaN), WSF5 fastest 5secs speed (few NaN)
    df = df[['AWND','WSF5']]
    # keep 01/01/2003 to  06/30/2019
    df = df[df.index.date < date(2019, 7, 1)]
    df['AWND'] = df['AWND'].fillna(method='pad')  
    df['WSF5'] = df['WSF5'].fillna(df.AWND)
    return df

def clean_temp2():
#     temp_path = '~/Desktop/Gina/Cursos/DataIncubator/Project/Fire/weatherData/temp2003_2019.csv'
    temp_path = './temp2003_2019.csv'
    # 12 columns, 6052 days: 01/01/2003 to 07/31/2019
    df = pd.read_csv(temp_path)
    # convert string date to timestamp  (%y= ## for year)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    df = df[['PRCP','TMIN', 'TMAX']]
#     df = df[df.index.date < date(2019, 7, 28)]
    df = df[df.index.date < date(2019, 7, 1)]
    return df


###### FUNCTIONS #####


def metrics(test_set_name, test_set, goal, predicted):
    (test_set[goal] - test_set[predicted]).plot()
    plt.title('Residuals for ' + predicted + ' on ' + test_set_name)
    plt.show()
    print(predicted + " MSE: ", round(sklearn.metrics.mean_squared_error(test_set[goal], test_set[predicted])))
    print(predicted + " R2: ", round(sklearn.metrics.r2_score(test_set[goal], test_set[predicted]),2))
    print("Statistics for Calls on " + test_set_name)
    print((test_set[goal]).describe())
    print("Statistics for the residuals on " + test_set_name)
    print((test_set[goal] - test_set[predicted]).describe())
    test_set[goal].plot()
    test_set[predicted].plot()
    plt.title('Calls vs Predicted for ' + predicted + ' on ' + test_set_name)
    plt.show()
    return

# metrics('Test',test, 'Calls', 'all_indeps_1d')
def metrics2(test_set_name, test_set, goal, predicted):
    pd.plotting.register_matplotlib_converters(explicit=True)
    plt.subplots(figsize=(15,7))
#     (test_set[goal] - test_set[predicted]).plot()
#     plt.title('Residuals for ' + predicted + ' on ' + test_set_name)
#     plt.show()
    print("MSE final model: ", round(sklearn.metrics.mean_squared_error(test_set[goal], predicted)))
    print("R2 final model: ", round(sklearn.metrics.r2_score(test_set[goal], predicted),2))
#     date_labs = test_set.index.strftime('%m/%d/%y')  # too crowded
    plt.plot(test_set.index.date,test_set[goal], label='True Calls')
    plt.plot(test_set.index.date,predicted, label='Predicted Calls')
    plt.title('True calls vs predicted on validation set', fontsize=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.9), shadow=False, ncol=2, fontsize=14)
#     plt.show()
    return

def join(leftdf, df2):
    df = pd.DataFrame.merge(leftdf, df2, how='left', left_index=True, right_index=True)
    return df

def read_data2():
    # Bring fire calls aggregated by day
#     path = '~/Desktop/Gina/Cursos/DataIncubator/Project/Fire/byDay.csv'
    path = './byDay.csv'
    # 6052 days from 01/01/2003 to 07/27/2019
   # Data is incomplete for the month of july. Will keep until 06/30/2019
    cutoffenddate = '20190630'
    byDay = pd.read_csv(path)
    # convert string date to timestamp
    byDay['Date'] = pd.to_datetime(byDay['Date'], format="%Y-%m-%d")
    byDay.set_index('Date', inplace=True)
    byDay = byDay.loc[:cutoffenddate]
    # Will keep only calls
    byDay = byDay[['Calls']]

    # bring AWND and WSF5 
    wind = clean_wind_wt2()

    # bring PRCP, TMIN and TMAX
    temp = clean_temp2()
    
    df= join(byDay,wind)
    df = join(df,temp)
    return df

def add_features(df):
    sets = {}
    indeps = {}
    # Create dataframe with calls, julian and constant
    modelDF, sets['indeps_drift'] = drift2(df)

    # Create a non linear variable combining PRCP and WSF5:
    modelDF['RainxGusts'] = np.square(modelDF.WSF5*modelDF.PRCP)
    # Create set for independent variable names for weather
    modelDF['PRCP_cum3'] = modelDF['PRCP'].rolling(3).sum()
    sets['indeps_weather'] = {'AWND','WSF5', 'PRCP', 'TMIN', 'TMAX', 'RainxGusts', 'PRCP_cum3'}

    modelDF, sets['indeps_previous1d'] = previous_day(modelDF)
    
    modelDF, sets['indeps_fourier'] = preFourier(modelDF)
    
    modelDF, sets['indeps_calendar'] = Calendar_Set(modelDF)
    
    modelDF, sets['indeps_holidays'] = holidays(modelDF)

    return modelDF, sets


def graph_calls_vs_AA(df, num):
    # Let's show past 3 weeks and future week
    last_day = df.index[-1] + timedelta(days=1)
    last_day_nice = last_day.date()
    window = 21
    plt.subplots(figsize=(15,7))
    date_lab = df.index[-(window+365):-(365-7)].strftime('%d')  #inclusive last index
    ## BARS, before whitesmoke, wheat
    plt.bar(date_lab,df['Calls'].iloc[-(window+365):-(365-7)], color='cornsilk',\
           label='Previous Year')
    ## LINE before: dimgrey
    plt.plot(date_lab[:-7],df['Calls'].iloc[-window:], color='darkorange',lw=2.5,\
            label= 'Current: '+ str(df.index.year[-1]))
    # trying to add to the plot the forecast:
    new_x = [date_lab[-8], date_lab[-7]]
    new_y = [df['Calls'].iloc[-1], num]
    plt.plot(new_x,new_y, color='blue', linestyle='--', lw=2.5)
    windowmin = min(df['Calls'].iloc[-(window+365):-(365-7)]\
                   .append(df['Calls'].iloc[-window:]))
    windowmin = min(windowmin, num)
    windowmax = max(df['Calls'].iloc[-(window+365):-(365-7)]\
                   .append(df['Calls'].iloc[-window:]))
    windowmax = max(windowmax, num)
    plt.title('Non-medical Calls to SF Fire Department', fontsize=20)
    plt.xlabel(df.index[-(window+365)].strftime('%b-%d')+' to '+df.index[-(365-7+1)].strftime('%b-%d')\
               , fontsize=16, horizontalalignment='right', x=.97)
    plt.ylim((max(windowmin-20,0),windowmax+5))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
    # bbox_to_anchor=(0.5, -0.2)  mientras mas negativo y mas abajo [0,1] el recuadro
    # plt.legend(ncol=2)
    plt.legend(loc='upper center', bbox_to_anchor=(0.2,-0.05), shadow=False, ncol=2, fontsize=14)
    x_pos = 8.5
    y_pos = windowmax - 5
    text =  str(num)+" expected calls for "+ str(last_day_nice)
    plt.text(x_pos, y_pos, text, fontsize=18, color='blue')
    return 

def split_data(df):
# NOTE: Tried removing before jul2005, 2014, jul2019. No improvement

    cut_year = 2017
    train_time_start = '20030101'
    train_time_end = '20161231'
    test_time_start = '20170101'

    # Note: python displayed warning when not using "loc"
    # train = modelDF.loc[train_time_start:train_time_end].dropna(how='any')  # 5086
    train = df.loc[:train_time_end].dropna(how='any')  # 5086
    test = df.loc[test_time_start:].dropna(how='any')  # 931
    return train, test

def final_model(df):
    train_data, test_data = split_data(df)
    model = sklearn.linear_model.LinearRegression().fit(
        X=train_data.drop(['Calls'],axis=1)\
        , y=train_data['Calls'])
    return model, train_data, test_data

def add_new_data(row_wt, df):
    # input is a dictionary with Date, Calls, AWND, WSF5, PRCP, TMIN and TMAX
    new_row_df = pd.DataFrame(row_wt)
    new_row_df.set_index('Date', inplace=True)
    new_row_df
    df_plusone = df.append(new_row_df)
    return df_plusone


def form():
    AWND = widgets.BoundedFloatText(
    value= 12.75,
    min= 0.0,
    max = 100.0,
    description = 'Average wind speed (MPH):',
    style = {'description_width': 'initial'},
    disabled=False)

    WSF5 = widgets.BoundedFloatText(
    value= 29.1,
    min= 0.0,
    max = 200.0,
    description = 'Gusts 5 secs (MPH):',
    style = {'description_width': 'initial'},
    disabled=False)

    PRCP = widgets.BoundedFloatText(
    value= 0.0,
    min= 0.0,
    max = 10.0,
    description = 'Precipitation (IN): ',
    style = {'description_width': 'initial'},
    disabled=False)

    TMIN = widgets.BoundedFloatText(
    value= 54.0,
    min= 20.0,
    max = 100.0,
    description = 'Min temperature (F):',
    style = {'description_width': 'initial'},
    disabled=False)

    TMAX = widgets.BoundedFloatText(
    value= 64.0,
    min= 36.0,
    max = 120.0,
    description = 'Max temperature (F):',
    style = {'description_width': 'initial'},
    disabled=False)

    click = Button(
    description = 'Forecast SFFD calls',
    layout = Layout(width ='auto'))
    click.style.button_color= 'orange'
    click.style.font_weight = 'bold'
    
    widg_li = [AWND, WSF5, PRCP, TMIN, TMAX, click]

    return widg_li
    
def display_form():
    # use original df as input (before adding features)
    df = read_data2()
    date = df.index[-1] + timedelta(days=1)
    print('Last available day in our database: ', df.index[-1].date())
    print('Please enter weather forecast for ', date.date() ,': ')
    box = form()
    align_left = VBox([box[0], box[3]])
    align_center = VBox([box[1], box[4]])
    align_right = VBox([box[2], box[5]])
    display (HBox([align_left, align_center,align_right]))

    def action_click(sender):
        last_day = df.index[-1] + timedelta(days=1)
        exp_wt = {'Date': [last_day],'Calls': [None],'AWND':[box[0].value],'WSF5': [box[1].value]\
              ,'PRCP': [box[2].value],'TMIN': [box[3].value],'TMAX': [box[4].value]}
        df_plusone = add_new_data(exp_wt, df)
        print('Your input:')
        print('Date:', last_day.date(), 'AWND:', exp_wt['AWND'],'WSF5:', exp_wt['WSF5']\
              ,'PRCP:', exp_wt['PRCP'], 'TMIN:', exp_wt['TMIN'], 'TMAX:', exp_wt['TMAX'])
        print()
        df_complete, indeps = add_features(df_plusone)
        model, train_data, test_data = final_model(df_complete)
        new_info = df_complete.drop(['Calls'], axis=1)[-1:]
        forecast = model.predict(X=new_info)
        forecast_nice = int(round(forecast[0]))
#         print('The forecast is:', forecast_nice)
#         print('Expected non-medical calls to SFPD for',date.date(),': ', forecast_nice)
        graph_calls_vs_AA(df, forecast_nice)
        HBox([align_left, align_center,align_right]).close()
        return

    box[5].on_click(action_click)
    return



