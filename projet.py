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

