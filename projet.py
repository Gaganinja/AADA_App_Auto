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


from sklearn.model_selection import cross_val_score
def multi_models(model, Xtrain, ytrain, Xtest, ytest):
    waiting_time("Fitting model")
    model[0].fit(Xtrain,ytrain)
    waiting_time("Prediction")
    y_pred = model[0].predict(Xtest)
    waiting_time("Calculating MSE")
    mse = mean_squared_error(ytest,y_pred)
    print("MSE = %.2f " % mse)
    print("R MSE = %.2f" % np.sqrt(mse))
    to_append.append(np.sqrt(mse))
    waiting_time("Calculating score")
    print("Score = %.2f" % r2_score(ytest,y_pred))
    to_append.append(r2_score(ytest,y_pred))
    scores = cross_val_score(model[0], X,Y,cv=2)
    to_append.append(scores.mean())
    # print(scores)
    print("CV K=5 Score mean %.2f" %scores.mean())
    scores = cross_val_score(model[0], X,Y,cv=4)
    to_append.append(scores.mean())
    # print(scores)
    print("CV K=10 Score mean %.2f" %scores.mean())
    

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Perceptron

models = [
    [LinearRegression(),"Linear Regression"],
    [XGBRegressor(), "XGBRegressor"],
    # [KNeighborsRegressor(), "KNeighborsRegressor"],
    # [Lasso(alpha=0.1), "Lasso"],
    # [RandomForestRegressor(),"RandomForestRegressor"],
    # [AdaBoostRegressor(),"AdaBoostRegressor"],
    # [GradientBoostingRegressor(),"GradientBoostingRegressor"],
    # [HistGradientBoostingRegressor(),"HistGradientBoostingRegressor"]
]

for elt in models:
    print("********************************  %s  *******************************" % elt[1])
    for j in [0.1,0.2,0.3]:
        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=j,random_state=22)
        print("*********** split %.2f - %.2f" %(1-j, j))
        to_append=[]
        multi_models(elt, X_train, y_train, X_test, y_test)
        elt.append(to_append)
        
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def getlists(mode):
    x = []
    y = []
    for elt in models:
        x.append(elt[1])
        y.append(elt[mode][0])
    return x,y

def getlists2(mode):
    y=[]
    for elt in models:
        y.append(elt[mode][1:3])
    return y

fig = make_subplots(rows=1,cols=2,subplot_titles=("RMSE","Accuracy"))

plot = go.Bar(x=getlists(2)[0],y=getlists(2)[1])
fig.append_trace(plot,row=1,col=1)
fig.update_layout(
    updatemenus=[
        dict(
            type = "buttons",
            direction = "down",
            buttons=list([
                dict(
                    args=[{"y":[getlists(2)[1]]}],
                    label="0.9 - 0.1",
                    method="update"
                ),
                dict(
                    args=[{"y":[getlists(3)[1]]}],
                    label="0.8 - 0.2",
                    method="update"
                ),
                dict(
                    args=[{"y":[getlists(4)[1]]}],
                    label="0.7 - 0.3",
                    method="update"
                )
            ]),
        ),
    ]
)
# plot = go.Bar(x=getlists(2)[0],y=getlists2(2)),



# fig.update_xaxes(type='category')

fig.append_trace(plot,row=1,col=2)
fig.show()
# plot.show()