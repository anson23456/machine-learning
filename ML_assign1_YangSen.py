import os
import pandas as pd
import numpy as np 
from collections import  Counter
from sklearn import mixture as mix
import seaborn as sns 
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


print(os.getcwd())
os.chdir('E:/python')
name=os.listdir()
name.remove('ML_assign1_YangSen.py')
data={}
for i in name:
    symbol = i [:-4]
    data[symbol]=pd.read_csv(i)
    data[symbol]=data[symbol].dropna()
    data[symbol]['Return']=np.log(data[symbol]['Adj Close']/data[symbol]['Adj Close'].shift(1))
    

for symbol in ['UPRO','TQQQ']:    
    data[symbol]=pd.merge(data[symbol],data['CNY-USD'][['Date','Return']],how='left',on=['Date'])
    data[symbol]['Return_y']=data[symbol].Return_y.fillna(0)
    data[symbol]['Return']=data[symbol]['Return_x']+data[symbol]['Return_y']
    data[symbol].drop(['Return_x', 'Return_y'], axis=1, inplace=True)    

del data['CNY-USD']

print('We use the data of {}'.format(data.keys()))
for symbol in data.keys():
    if symbol == '000300.SS':
        continue
    print('{} check class imbalance issue {}'.format(symbol,Counter(np.where(data[symbol].Return>0,1,-1))))

#%%
    
for symbol in data.keys():
    if symbol == '000300.SS':
        continue
    df=data[symbol].copy()
    df=df[df.Volume!=0]
    
    df['Pre_high']=df['High'].shift(1)
    df['Pre_low']=df['Low'].shift(1)
    df['Pre_close']=df['Close'].shift(1)
    
    feature_names=['Volume_Change','MA5','MA120','MA20','RSI','Corr','SAR','ADX','ATR','OBV']
    
    df['Volume_Change']=np.log(df.Volume/df.Volume.shift(1))
    df['MA5']=ta.MA(df.Pre_close,5)
    df['MA120']=ta.MA(df.Pre_close,120)
    df['MA20']= df.Pre_close.rolling(window=20).mean()
    df['RSI']=ta.RSI(df.Pre_close, timeperiod=14)    
    df['Corr']= df['MA20'].rolling(window=20).corr(df['Pre_close'])    
    df['SAR']=ta.SAR(np.array(df['Pre_high']),np.array(df['Pre_low']),\
                      0.2,0.2)
    df['ADX']=ta.ADX(np.array(df['Pre_high']),np.array(df['Pre_low']),\
                      np.array(df['Pre_close']), timeperiod =14)
    df['ATR']=ta.ATR(np.array(df['Pre_high']),np.array(df['Pre_low']),\
                      np.array(df['Pre_close']), timeperiod =14)
    df['OBV']=np.log(ta.OBV(df.Pre_close,df.Volume.shift(1)))
    #df[feature_names]
        
    df=df.dropna()
    
    data[symbol]=df.copy()

print('We have these features: {}'.format(feature_names))


#%%

split=int(0.7*len(df))


list_store=[]
for symbol in data.keys():
    
    df=data[symbol][feature_names+['Return']].copy()
         
    features=df.iloc[:,:-1]
    targets=df.iloc[:,-1]
    x_train, x_test,y_train,y_test=features[:split],features[split:],targets[:split],targets[split:]
    
    ss=StandardScaler()
    ss.fit(x_train)    
    x_train = pd.DataFrame(ss.transform(x_train))
    x_test = pd.DataFrame(ss.transform(x_test))
    
    date_mk=data[symbol]['Date'].copy()[split:]
    y_test.reset_index(drop=True,inplace=True)
    date_mk.reset_index(drop=True,inplace=True)
    Regime=pd.concat([x_test,y_test,date_mk],axis=1)
    
    model=LinearRegression()
    model.fit(x_train,y_train)
    
    print(pd.DataFrame(list(zip(features,model.coef_)),columns=['features','coefficients']))
        
    Regime['Predict']=model.predict(x_test)
    Regime['Hold']=np.where(Regime.Predict>0,1,0)
    Regime['Predict_return']=Regime.Return*Regime.Hold
    Regime.index=Regime.Date
    Regime.Predict_return.cumsum().plot()
    #Regime.Return.cumsum().plot()
    list_store.append(Regime.Predict_return.cumsum())



#https://plot.ly/python/line-charts/
    
np.where(df['Return']>0,1,-1)
model = LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
Regime['Predict']=model.predict(x_test)


   

        


