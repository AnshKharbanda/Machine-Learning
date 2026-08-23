import numpy as np

class Knn():
    def __init__(self,k):
        self.k=k
        self.x=None
        self.y=None
        
    def fit(self,x,y):
        self.x=np.asarray(x)
        
        if self.k>len(self.x) or self.k<1:
            raise ValueError("K value must be between 1 and number of training samples")
            
        self.y=np.asarray(y)
        
        if len(self.x)!=len(self.y):
            raise ValueError("Input data and output data don't match number of samples")
        
    # def _distance(self,x1,x2):
    #     return np.sum((x1-x2)**2)   as square can be compared instead of root, sqrt is unnecessary
        
    def predict(self,x_test):
        if self.x is None or self.y is None:
            raise ValueError("Model should be fit before Prediction")
        
        x_test=np.asarray(x_test)
        
        predictions=[]
        
        for test_point in x_test:
            # dist=[self._distance(test_point,i) for i in self.x]
            
            dist=np.sum((self.x-test_point)**2,axis=1)
                
            k_indices=np.argsort(dist)[:self.k]
            
            pt_freq={}
            
            for i in k_indices:
                label=self.y[i]
                pt_freq[label] = pt_freq.get(label,0)+1
                
            prediction=max(pt_freq,key=pt_freq.get)
            
            predictions.append(prediction)
            
        
        return predictions
            
    def accuracy(self,y_true,y_pred):
        if len(y_true)==0 or len(y_pred)==0:
            raise ValueError("Inputs cannot be empty")
        
        if len(y_true)!=len(y_pred):
            raise ValueError("Both entities should have same length")
        
        return np.mean(np.asarray(y_true) == np.asarray(y_pred))
        
        
        