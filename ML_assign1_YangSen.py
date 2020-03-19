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
#读取数据
name.remove('ML_assign1_YangSen.py')
data={}
for i in name:
    symbol = i [:-4]
    data[symbol]=pd.read_csv(i)
    data[symbol]=data[symbol].dropna()
    data[symbol]['Return']=np.log(data[symbol]['Adj Close']/data[symbol]['Adj Close'].shift(1))
    
#所有美股收益率+汇率收益，即本币视为人民币
for symbol in ['UPRO','TQQQ']:    
    data[symbol]=pd.merge(data[symbol],data['CNY-USD'][['Date','Return']],how='left',on=['Date'])
    data[symbol]['Return_y']=data[symbol].Return_y.fillna(0)
    data[symbol]['Return']=data[symbol]['Return_x']+data[symbol]['Return_y']
    data[symbol].drop(['Return_x', 'Return_y'], axis=1, inplace=True)    

del data['CNY-USD']
#打印出所有使用的数据
print('We use the data of {}'.format(data.keys()))
#打印出验证数据集是否平衡
for symbol in data.keys():
    if symbol == '000300.SS':
        continue
    print('{} check class imbalance issue {}'.format(symbol,Counter(np.where(data[symbol].Return>0,1,-1))))

#%%

#所有股票分别计算features    
for symbol in data.keys():
    if symbol == '000300.SS':
        continue
    df=data[symbol].copy()
    #剔除成交量为0的数据，一共只有两条，一条是涨停，另一条应该是数据缺失
    df=df[df.Volume!=0]
    
    df['Pre_high']=df['High'].shift(1)
    df['Pre_low']=df['Low'].shift(1)
    df['Pre_close']=df['Close'].shift(1)
    df['Pre_volume']=df['Volume'].shift(1)
    
    feature_names=['Volume_Change','MA5','MA120','MA20','RSI','Corr','SAR','ADX','ATR','OBV']
    
    df['Volume_Change']=np.log(df.Volume/df.Pre_volume)
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
    df['OBV']=ta.OBV(df.Pre_close,df.Pre_volume)
   
    #将之前无法计算feature的数据删除    
    df=df.dropna()
    #替换原有的数据，重新储存
    data[symbol]=df.copy()

print('We have these features: {}'.format(feature_names))


#%%




list_store=[]
for symbol in data.keys():
    if symbol == '000300.SS':
        continue
    
    #调出所有的feature和return
    df=data[symbol][feature_names+['Return']].copy()
    
    #feature为之前所有列，return为最后一列       
    features=df.iloc[:,:-1]
    targets=df.iloc[:,-1]
    split=int(0.7*len(df))
    x_train, x_test,y_train,y_test=features[:split],features[split:],targets[:split],targets[split:]
    
    #将所有x值进行标准化
    ss=StandardScaler()
    ss.fit(x_train)    
    x_train = pd.DataFrame(ss.transform(x_train))
    x_test = pd.DataFrame(ss.transform(x_test))
    
    #保存test的值
    date_mk=data[symbol]['Date'].copy()[split:]
    y_test.reset_index(drop=True,inplace=True)
    date_mk.reset_index(drop=True,inplace=True)
    Regime=pd.concat([x_test,y_test,date_mk],axis=1)
    
    #设定模型，模型参数估计
    model=LinearRegression()
    model.fit(x_train,y_train)
    
    #打印出模型参数
    print(pd.DataFrame(list(zip(features,model.coef_)),columns=['features','coefficients']))
    
    #利用模型对test进行预测，预测为正的进行持仓
    Regime['Predict']=model.predict(x_test)
    Regime['Hold']=np.where(Regime.Predict>0,1,0)
    
    #计算单股票策略收益
    Regime['Predict_return']=Regime.Return*Regime.Hold
    Regime.index=Regime.Date
    #Regime.Predict_return.cumsum().plot()
    #Regime.Return.cumsum().plot()
    list_store.append(Regime.Predict_return.cumsum())

#所有策略收益进行集合
test=pd.concat(list_store,axis=1,sort=False).sort_index()

   



