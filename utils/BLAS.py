import numpy                            as np
import tensorflow                       as tf

def create_dataset(dataset, look_back=1, SeriesName =''):
    
    # Check if SeriesName exists in dataset
    if (SeriesName not in dataset.columns):
        SeriesName = dataset.columns[-1]
    
    
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-look_back):
        a = dataset.to_numpy()[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[SeriesName].to_numpy()[i + look_back])
    return np.array(dataX), np.array(dataY)




def step_decay_schedule(initial_lr=1e-1, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return tf.keras.callbacks.LearningRateScheduler(schedule)








