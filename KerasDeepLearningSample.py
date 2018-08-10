import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# function to generate sample train and test data
def CreateSampleData(n):
    #np.random.seed(0)
    a = np.random.randint(1, 20, n)
    b = np.random.randint(1, 50, n)
    c = np.random.randint(1, 100, n)
    flag =[]
    for aa, bb, cc in zip(a, b, c):
        m = 1
        if (aa>15 and bb>30 and cc>60):
            m = 2
        elif(aa<=9 and bb<25 and cc<=35):
            m = 0
        flag.append(m)

    return a, b, c, flag

# create train data
train_n = 2000   # number of train data
a, b, c, flag = CreateSampleData(train_n)
train_x = np.column_stack((a, b, c))
train_y = np.reshape(flag, (train_n, 1))

# creat test data
test_n = 100    # number of test data
a, b, c, flag = CreateSampleData(test_n)
test_x= np.column_stack((a, b, c))
test_y=np.reshape(flag, (test_n, 1))

# building model nn model
model = Sequential()
model.add(Dense(32, input_dim = 3, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer = "adam", 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])

model.fit(train_x, train_y, epochs = 50, batch_size = 40)

# predicting a test data
pred = model.predict(test_x)
print(np.around(pred, 2))

# evaluting a model
scores = model.evaluate(test_x, test_y)
print(scores)
