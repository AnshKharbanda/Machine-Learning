import numpy as np

class Perceptron:
    def __init__(self,epoch,lr):
        self.coef=[]
        self.epoch=epoch
        self.lr=lr
        
    def step_function(self,y):
        if y>=0:
            return 1
        return 0
        
    def fit(self,x,y):
        
        x=np.asarray(x)
        y=np.asarray(y)
        
        # add 1 at front of x
        x=np.hstack((np.ones((x.shape[0],1)),x))
        self.coef=np.ones((x.shape[1]))
        
        for i in range(self.epoch):
            random_row=np.random.randint(0,x.shape[0])
            
            y_pred=np.dot(x[random_row],self.coef)
            y_pred=self.step_function(y_pred)
            
            self.coef=self.coef+self.lr*(y[random_row]-y_pred)*x[random_row]
            