from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# sets status to data
def GetStatus(test, paper):
    status = []    
    for(t,p) in zip(test, paper):
        score = 0
        if(t>65 and p>45):            
            score = 1         
        status.append(score)
    return status

# create data frame
n = 1000  # number of samples
test = np.random.randint(30, 100, n) 
paper = np.random.randint(20, 100, n)
status = np.array(GetStatus(test, paper))
df = pd.DataFrame({'test':test, 'paper':paper, 'status':status})
print(df.head(10))

# split data into test and train. x is input and y is output values to train 
array = df.values
x=array[:, 0:1]
y=array[:, 2]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

# creating model
model = LogisticRegression()
model.fit(train_x, train_y)

# score check
score = model.score(test_x, test_y)
print(score)

# predict test data and show in confusion matrix 
pred_y = model.predict(test_x)
result = confusion_matrix(test_y, pred_y)
print(result)
