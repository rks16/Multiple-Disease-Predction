import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

kdf = pd.read_csv('kidney_disease.csv')

columns_to_retain = ['age','sg','bp','al','sc','hemo','pcv','classification']

# Drop unnecessary column
kdf = kdf.drop([col for col in kdf.columns if not col in columns_to_retain],axis =1)

# Drop the rows with na or missing values
kdf = kdf.dropna(axis=0)

#replacing
kdf.replace({"classification":{"notckd": 0, "ckd": 1}},inplace=True)

X = kdf.drop(columns=["classification"], axis=1)
Y = kdf["classification"]

#scaling data
scal = StandardScaler()
scal.fit(X)

# splitting data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y)

#model training
modelkd = LogisticRegression()
modelkd.fit(X,Y)

pickle.dump(modelkd,open('kid.pkl','wb'))