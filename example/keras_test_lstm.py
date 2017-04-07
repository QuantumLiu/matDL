from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import time
def main(nb_batch=100,hiddensize=512,input_dim=100,timestep=10,batch_size=32,nb_epoch=1):
    x=np.random.ones(nb_batch*batch_size,timestep,input_dim).astype('float32')
    y=np.ones((nb_batch*batch_size,timestep,hiddensize))
    model = Sequential()
    model.add(LSTM(output_dim=hiddensize, input_shape=(timestep,input_dim),return_sequences=True))
    model.add(LSTM(output_dim=hiddensize,return_sequences=True))
    model.add(LSTM(output_dim=hiddensize,return_sequences=True))
    model.compile(loss='mse',optimizer='sgd')
    start=time.time()
    model.fit(x=x,y=y,batch_size=batch_size,nb_epoch=nb_epoch)
    duration=time.time()-start
    print('Duration: ',duration,' sec')
if __name__ == "__main__":
    main(nb_batch=50,hiddensize=1024,input_dim=256,timestep=10,batch_size=64,nb_epoch=3)
