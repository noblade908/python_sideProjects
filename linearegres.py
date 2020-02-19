import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn import linear_model

data = pd.read_csv("winequality-red.csv", sep = ";")
data = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide"]]
print(data.head())
predict = "citric acid"

x = np.array(data.drop([predict],1))
y = np.array(data[predict])


x_train,x_test,y_train,y_test =sklearn.model_selection.train_test_split(x,y,test_size=0.1)

linear  = linear_model.LinearRegression()
linear.fit(x_train,y_train)

acc=linear.score(x_test,y_test)
print("accuracy ",acc)
print("coef \n",linear.coef_)
print("intercept\n",linear.intercept_)


predicate = linear.predict(x_test)
for x in range(len(predict)):
    print("prediction ",predict[x]," for this ",x_test[x]," actual ",y_test[x])