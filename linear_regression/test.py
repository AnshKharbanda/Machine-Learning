import numpy as np
from mlr import Linear_Regression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.array([
    [1,2],
    [2,3],
    [3,4],
    [4,5]
])

y = 2*X[:,0] + 3*X[:,1] + 1

model=Linear_Regression(epoch=500,lr=0.02)

model.fit(X,y)

# weights=[2,3]
# intercept=1

weights,intercept=model.get_params()

print(weights,intercept)

plt.plot(model.losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

model1=LinearRegression()
model1.fit(X,y)

weight1=model1.coef_
intercept1=model1.intercept_

print(weight1,intercept1)
