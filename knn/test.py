import numpy as np
from knn.knn1 import Knn
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [2, 2],
    [5, 5],
    [6, 5],
    [5, 6],
    [6, 6]
])

y_train = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])


X_test = np.array([
    [1.5, 1.5],
    [5.5, 5.5],
    [3.0, 3.0],
    [1.8, 2.0],
    [5.2, 5.0]
])

model1 = Knn(k=3)
model1.fit(X_train,y_train)
predictions1 = model1.predict(X_test)

model2=KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train,y_train)
predictions2 = model2.predict(X_test)

print(predictions1)
print(predictions2)



