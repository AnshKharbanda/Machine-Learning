import numpy as np

x_train=[1,2,3,4,5]
y_train=[3,4,5,6,7]

x = [1,2,3,4,5]
y = [3,5,7,9,11]

def linear_regression(x_train,y_train,epoch:int,alpha:float):
    slope=1
    intercept=1
    for i in range(epoch):
        dm=0
        db=0
        for j in range(len(x_train)):
            dm+=(y_train[j]-x_train[j]*slope-intercept)*x_train[j]
            db+=(y_train[j]-x_train[j]*slope-intercept)
        dm=-dm*2/len(x_train)
        db=-db*2/len(x_train)
        slope=slope-alpha*dm
        intercept=intercept-alpha*db
        if i%100==0:
            print(y_train[j]-x_train[j]*slope-intercept)
    

linear_regression(x,y,5000,0.01)
