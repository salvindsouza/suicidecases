import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
le = LabelEncoder()
dataframe = pd.read_csv("master.csv")
import pickle


#preprocessing
dataframe.drop(['country-year','suicides/100k pop'],axis =1,inplace = True )
dataframe.rename(columns={"gdp_for_year ($) ":"gdp_for_year_usd","gdp_per_capita ($)":"gdp_per_capita_usd"},inplace=True)
dataframe["gdp_for_year_usd"] = dataframe["gdp_for_year_usd"].str.replace(',','').astype('int64')
dataframe.drop(['HDI for year'],axis = 1,inplace=True) #miss
dataframe = dataframe[dataframe['year'] != 2016]  #prior 2016
dataframe = dataframe[dataframe['suicides_no'] < 131] #outliers
#We can see that the data from 2016 only has 16 countries and actually does not include an age group (4~15 years)
X = dataframe.drop(['suicides_no','year'],axis = 1)
y = dataframe['suicides_no'].copy()
categorical_features = ['age','sex','generation','country']

#cat
mapping_dict={}
for col in categorical_features:
    X[col] = le.fit_transform(X[col])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mapping_dict[col]=le_name_mapping
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#train 
# from sklearn.tree import DecisionTreeRegressor
adareg = AdaBoostRegressor(DecisionTreeRegressor())
adareg.fit(X_train, y_train)

# Saving coloumns and model to disk
pickle.dump(adareg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print('worked')
