import pytz
import pandas as pd 
import numpy as np
import torch 

from datetime import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
from constants import *

def load_data_with_features(filename):
    tz = pytz.timezone("America/New_York")
    df = pd.read_csv(filename, sep=" ", header=None, usecols=[1,2,3], 
        names=["time","load","temp"])
    df["time"] = df["time"].apply(dt.fromtimestamp, tz=tz)
    df["date"] = df["time"].apply(lambda x: x.date())
    df["hour"] = df["time"].apply(lambda x: x.hour)
    df.drop_duplicates("time", inplace=True)

    # Create one-day tables and interpolate missing entries
    df_load = df.pivot(index="date", columns="hour", values="load")
    df_temp = df.pivot(index="date", columns="hour", values="temp")
    df_load = df_load.transpose().fillna(method="backfill").transpose()
    df_load = df_load.transpose().fillna(method="ffill").transpose()
    df_temp = df_temp.transpose().fillna(method="backfill").transpose()
    df_temp = df_temp.transpose().fillna(method="ffill").transpose()

    holidays = USFederalHolidayCalendar().holidays(
        start='2008-01-01', end='2016-12-31').to_pydatetime()
    holiday_dates = set([h.date() for h in holidays])

    s = df_load.reset_index()["date"]
    data={"weekend": s.apply(lambda x: x.isoweekday() >= 6).values,
          "holiday": s.apply(lambda x: x in holiday_dates).values,
          "dst": s.apply(lambda x: tz.localize(
            dt.combine(x, dt.min.time())).dst().seconds > 0).values,
          "cos_doy": s.apply(lambda x: np.cos(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values,
          "sin_doy": s.apply(lambda x: np.sin(
            float(x.timetuple().tm_yday)/365*2*np.pi)).values}
    df_feat = pd.DataFrame(data=data, index=df_load.index)

    # Construct features and normalize (all but intercept)
    X = np.hstack([df_load.iloc[:-1].values,        # past load
                    df_temp.iloc[:-1].values,       # past temp
                    df_temp.iloc[:-1].values**2,    # past temp^2
                    df_temp.iloc[1:].values,        # future temp
                    df_temp.iloc[1:].values**2,     # future temp^2
                    df_temp.iloc[1:].values**3,     # future temp^3
                    df_feat.iloc[1:].values,        
                    np.ones((len(df_feat)-1, 1))]).astype(np.float64)
    # X[:,:-1] = \
    #     (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)

    Y = df_load.iloc[1:].values

    return X, Y

def get_data(): 
    X1, Y1 = load_data_with_features('data/pjm_load_data_2008-11.txt')
    X2, Y2 = load_data_with_features('data/pjm_load_data_2012-16.txt')

    X = np.concatenate((X1, X2), axis=0)
    X[:,:-1] = (X[:,:-1] - np.mean(X[:,:-1], axis=0)) / np.std(X[:,:-1], axis=0)

    Y = np.concatenate((Y1, Y2), axis=0)

    # Train, test split.
    n_tt = int(len(X) * 0.8)
    X_train, Y_train = X[:n_tt], Y[:n_tt]
    X_test, Y_test = X[n_tt:], Y[n_tt:]

    # Construct tensors (without intercepts).
    X_train_pt = torch.tensor(X_train[:,:-1], dtype=torch.float, device=DEVICE)
    Y_train_pt = torch.tensor(Y_train, dtype=torch.float, device=DEVICE)
    X_test_pt = torch.tensor(X_test[:,:-1], dtype=torch.float, device=DEVICE)
    Y_test_pt = torch.tensor(Y_test, dtype=torch.float, device=DEVICE)
    return X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt

def get_data_size(n): 
    X_train, Y_train, X_test, Y_test, X_train_pt, Y_train_pt, X_test_pt, Y_test_pt = get_data() 
    
    Y_train_full = [] 
    for i in range(Y_train.shape[0]-n): 
        cur_y = Y_train[i:i+n,:].flatten()
        Y_train_full.append(cur_y)

    Y_test_full = [] 
    for i in range(Y_test.shape[0]-n): 
        cur_y = Y_test[i:i+n,:].flatten()
        Y_test_full.append(cur_y)
    
    return torch.tensor(Y_train_full).to(DEVICE).float(), torch.tensor(Y_test_full).to(DEVICE).float()