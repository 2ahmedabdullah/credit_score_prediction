import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
import keras


def drop_rows_with_na(df,na_limit):
    df= df.dropna(thresh=na_limit)    
    #na_list=df.isnull().sum(axis=1)
    return df
    
def outliers(x, num_vars):
    out = []
    for j in range(0,len(num_vars)):
        z = x[num_vars[j]]
        upper_limit = z.quantile(0.99)
        lower_limit = z.quantile(0.001)
        
        z1= z>=upper_limit
        ls1= list(z1[z1].index.values)
        z2= z<lower_limit
        ls2= list(z2[z2].index.values)
        #print(ls2)
        lss = ls1+ls2
        out.append(ls1)
    return out

def filling_mean_values(df):
    df=df.fillna(df.mean())
    return df

def filling_mode_values(df):
    df=df.fillna(df.mode())
    return df

def normalize(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def drop_num_variables_with_zero_variance(df):
    idx = df.std()<0.01
    z=idx.loc[idx]  
    d1 = list(set(z.index))
    return d1


def neural_network(x_train, y_train, x_test, y_test):
    
    input_dim=125
    batch_size=256 
    epochs=50
    max_deviation=3

    model = Sequential()
    model.add(Dense(125, activation='elu', input_shape=(input_dim,)))
    model.add(Dense(60,  activation='elu'))
    model.add(Dense(30,  activation='elu'))
    model.add(Dense(15,  activation='elu'))
    model.add(Dense(8,   activation='elu'))
    model.add(Dense(1,   activation='elu'))

    model.compile(loss='mean_squared_error', optimizer = Adam())

    model1= model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                                    verbose=1, validation_data=(x_test, y_test))

    plt.plot(model1.history['loss'])
    plt.plot(model1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    y_pred = model.predict(x_test)
    pred1 = [item for sublist in y_pred for item in sublist]
    pred2= np.array(pred1)
    return pred2


    