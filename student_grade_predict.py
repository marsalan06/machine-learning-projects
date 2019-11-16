#import tensorflow
#import keras
import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data=pd.read_csv("student-mat.csv",sep=";")
#print(data.head())
data=data[["G1","G2","G3","Medu","Fedu","studytime"]]  #gets mentioned columns only
#print(data.head())
predict= "G3" #WANT TO KNOW THIS SO TO REMOVE FROM DATA SET #just a variable

x=np.array(data.drop([predict],1)) #creates array of column of all atributes except g3 
#1 is for axis=1 or column
y=np.array(data[predict])
#print(x)
#print(y)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
best=0
# for _ in range(30):
#     x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1) #utility of sklearn
#     #it splits data sets in to test and train sets as per need 
#     #print(x_train) #shows training data
#     linear=linear_model.LinearRegression() #create a linear regression model #initiate class
#     linear.fit(x_train,y_train) # train the model on x_train training data and match predicted g3 with y_train
#     accuracy=linear.score(x_test,y_test) #test on actual data
#     print(accuracy) #% accuracy on test set
#     #print(linear.coef_) #y=mx+c for linear line #this shows m
#     #print(linear.intercept_) #this shows c
#     if accuracy>best:
#         best= accuracy
#         with open ("studentmodel.pickle","wb") as f:
#             pickle.dump(linear,f)
# print("the best accuracy came at "+str(best))
pickle_in=open("studentmodel.pickle","rb")
linear=pickle.load(pickle_in)

predict=linear.predict(x_test)
for x in range(len(predict)):
    print(predict[x],x_test[x],y_test[x])

p="G1" #for x axis change this accordingly
style.use("ggplot") #style of plot
pyplot.scatter(data[p],data["G3"]) #x,y
pyplot.xlabel(p)#lable name of x
pyplot.ylabel("Final Grade") #lable name of y
pyplot.show() #show plot
