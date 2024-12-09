import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('/content/diabetes.csv')
data.head()

data.shape
data.describe()

data['Outcome'].value_counts()

data.groupby('Outcome').mean()

x=data.drop(columns='Outcome',axis=1)
y=data['Outcome']

scaler=StandardScaler()
scaler.fit(x)

data=scaler.transform(x)
print(data)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
print(x.shape,x_train.shape,x_test.shape)


classifier=svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)

x_train_prediction=classifier.predict(x_train)
accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy scoreof the data :' , accuracy)

input_data=(4,110,92,0,0,37.6,0.191,30)
inout_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=inout_data_as_numpy_array.reshape(1,-1)

standardized_data=scaler.transform(input_data_reshaped)
print(standardized_data)

prediction=classifier.predict(standardized_data)
print(prediction)

if(prediction[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


import pickle

pickle.dump(classifier,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
input_data=(4,110,92,0,0,37.6,0.191,30)
inout_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=inout_data_as_numpy_array.reshape(1,-1)

standardized_data=scaler.transform(input_data_reshaped)
print(standardized_data)

prediction=model.predict(standardized_data)
print(prediction)

if(prediction[0]==0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
