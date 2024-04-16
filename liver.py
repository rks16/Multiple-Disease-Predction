import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df=pd.read_csv('liver.csv')

df.dropna(inplace=True) 

X= df.drop(columns=['Gender','Dataset'], axis=1)
Y= df['Dataset']

scale = StandardScaler()
scale.fit(X)

standardized_data = scale.transform(X)

X = standardized_data
Y = df['Dataset']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,stratify=Y)

#Model training
modelLR= LogisticRegression()

modelLR.fit(X_train,Y_train)

pickle.dump(modelLR,open('lvr.pkl','wb'))