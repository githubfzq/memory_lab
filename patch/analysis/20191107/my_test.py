import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class test_A:
    def __init__(self):
        self.x=np.arange(0,2*np.pi,np.pi/200)
        self.y=np.sin(self.x)
        self.nn=None
    def draw(self):
        plt.plot(self.x,self.y)
    def get_nn(self):
        if self.nn is None:
            result=pd.DataFrame({'A':[1,2,3],'B',[33,22,11]})
            self.nn=result
        return self.nn

class test_B(test_A):
    def __init__(self):
        super().__init__()
        self.y2=np.cos(self.x)

    def draw(self):
        super().draw()
        plt.plot(self.x,self.y2)
