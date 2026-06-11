X = [
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [3, 4],
    [6, 7],
    [7, 6],
    [7, 8],
    [8, 7],
    [9, 8]
]

y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

from lr1 import Logisticregression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model1=Logisticregression(epoch=5000)
model2=LogisticRegression()

model1.fit(X,y)
model2.fit(X,y)

X_test = [
    [1, 1],
    [2, 2],
    [4, 3],
    [5, 5],
    [6, 6],
    [8, 8]
]

y_test = [
    0,
    0,
    0,
    1,
    1,
    1
]


y1=model1.predict(X_test)
y2=model2.predict(X_test)

print(accuracy_score(y1,y_test))
print(accuracy_score(y2,y_test))

w1,b1=model1.getparams()
w2=model2.coef_
b2=model2.intercept_

print(w1,b1)
print(w2,b2)