import numpy as np

class Linear_Regression:
    def __init__(self,epoch=200,lr=0.01):
        self.weights_=[]
        self.intercept_=0
        self.epoch=epoch
        self.lr=lr
        self.losses=[]
        
    def fit(self,X,Y):
        X=np.asarray(X)
        Y=np.asarray(Y)
        
        n=X.shape[0]
        
        n_features=X.shape[1]
        self.weights_=np.zeros(n_features)
        
        for i in range(self.epoch):
            y_pred=np.dot(X,self.weights_)+self.intercept_
            
            dw=(-2/n)*np.dot(X.T,(Y-y_pred))
            dc=(-2/n)*np.sum(Y-y_pred)
            
            self.weights_-=self.lr*dw
            self.intercept_-=self.lr*dc
            
            loss=np.mean((Y-y_pred)**2)
            self.losses.append(loss)
            
    def predict(self,X):
        X=np.asarray(X)
        return np.dot(X,self.weights_)+self.intercept_
    
    def score(self,X,y_test):
        y_pred=self.predict(X)
        
        residual=np.sum((y_pred-y_test)**2)
        var=np.sum((y_test-np.mean(y_test))**2)
        
        r2=1-residual/var
        
        return r2
    
    def get_params(self):
        return self.weights_,self.intercept_
    
    