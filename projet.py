### import 
print("Including libraries...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def waiting_time(msg):
    # input('\n...')
    print(msg)

waiting_time("Reading csv files...")

Train_data = pd.read_csv("./datasets/Train.csv")
Test_data = pd.read_csv("./datasets/Test.csv")

waiting_time("Showing train data head")
print(Train_data.head())

waiting_time("Showing data infos")
print("(Train, Test) shapes = ", end ='')
print(Train_data.shape, Test_data.shape)


waiting_time("Testing Null values")
waiting_time("***********  For Train_Data  ***********")
print(Train_data.isnull().sum())
waiting_time("***********  For Test_Data  ************")
print(Test_data.isnull().sum())

waiting_time("Cleaning Data")

Train_data['Item_Weight'].fillna(Train_data['Item_Weight'].mean(),inplace = True)
Train_data['Outlet_Size'].fillna('Missing', inplace = True)

Test_data['Item_Weight'].fillna(Test_data['Item_Weight'].mean(),inplace = True)
Test_data['Outlet_Size'].fillna('Missing', inplace = True)

waiting_time("Printing Train head again")
print(Train_data.head())


waiting_time("Data cleaning")
#There is irregularities in the Item_Fat_Content column so we need to correct it.
# for e in Train_data.columns:
#     print(e)
#     print(Train_data[e].value_counts())
print(Train_data['Item_Fat_Content'].value_counts())

waiting_time("Correcting irregularities")

Train_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
Test_data['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
Train_data['Item_Fat_Content']= Train_data['Item_Fat_Content'].astype(str)

waiting_time('dropping useless columns')
Train_data = Train_data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)
Test_data= Test_data.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)


waiting_time("Creating model")
encoder = LabelEncoder()
X = []
Y = []
features = []
for e in Train_data.columns:
    Train_data[e] = encoder.fit_transform(Train_data[e])
    if(e != "Item_Outlet_Sales"):
        features.append(e)
        Test_data[e] = encoder.fit_transform(Test_data[e])

X = Train_data[features]
# X_test = Test_data[features]
Y= Train_data['Item_Outlet_Sales']
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=22)

waiting_time("Building model")

LR = LinearRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
coef2 = pd.Series(LR.coef_,features).sort_values()

print(coef2)

waiting_time("Calculating MSE")
print(mean_squared_error(y_test, y_pred))

waiting_time("Calculating score")
R2 = r2_score(y_test,y_pred)
print(R2)

waiting_time("Optimization: cross_validation (10 times)")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LR, X,Y,cv=10)
print(scores)
print(scores.mean(), scores.std())


