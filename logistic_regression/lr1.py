import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

class Logisticregression:
    def __init__(self,epoch=500,lr=0.02):
        self.w=[]
        self.b=1
        self.lr=lr
        self.epoch=epoch
        self.losses=[]
        
    def fit(self,x,y):
        x=np.asarray(x)
        y=np.asarray(y)
        
        m=x.shape[1]
        n=x.shape[0]
        
        self.w=np.zeros(m)
        
        for i in range(self.epoch):
            y_pred=sigmoid(np.dot(self.w,x.transpose())+self.b)
            
            dw=np.dot(x.transpose(),y_pred-y)/n
            db=np.sum(y_pred-y)/n
            
            self.w=self.w-self.lr*dw
            self.b=self.b-self.lr*db
            
            # to avoid adding zero
            
            eps=1e-9 
            loss=-np.dot(y,np.log(y_pred+eps))+np.dot((1-y),np.log(1-y_pred+eps))/n
            self.losses.append(loss)
     
    def predict(self,x_test):
        x_test=np.asarray(x_test)
        y_pred=sigmoid(np.dot(self.w,x_test.transpose())+self.b)
        
        return (y_pred>=0.5).astype(int)

           
    def getparams(self):
        return self.w,self.b
    
    def score(self,x_test,y_test):
        y_pred=self.predict(x_test)
        
        return np.mean(y_test==y_pred)
    