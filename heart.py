import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

heart_data = pd.read_csv('heart_disease_data.csv')

X = heart_data.drop(columns='target',axis=1)
Y= heart_data['target']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y)

classifier = LogisticRegression()

classifier.fit(X_train,Y_train)

pickle.dump(classifier,open('hrt.pkl','wb'))
# pickle.dump(classifier,open('hrt.pkl','rb'))
