import keras
from keras import models
from keras import layers
import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(train_targets)

#Normierung der Daten
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

print(train_data)

#Definition des Modells
def build_model(): 
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    
    model.compile(optimizer='rmsprop', loss='mse',  metrics=['mae'])
    return model        

#Speichern der Validierungsscores nach jedem Durchlauf
k=4
num_val_samples = len(train_data) // k
num_epochs = 70
all_scores = []
all_mae_histories = []

for i in range(k):
    print('Durchlauf #', i)
    val_data = train_data[i * num_val_samples: (i + 1) *
                                        num_val_samples] 
    val_targets = train_targets[i * num_val_samples:
                          (i + 1) * num_val_samples]
    partial_train_data = np.concatenate( 
               [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()  
    history = model.fit(partial_train_data,  
                        partial_train_targets,
                        validation_data=
                        (val_data, val_targets),
                        epochs=num_epochs, batch_size=1,
                        verbose=0)
    mae_history = history.history['val_mean_absolute_error']    
    all_mae_histories.append(mae_history)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print(test_mae_score)

#Verlauf des mittleren absoluten Fehlers bei der Validierung erzeugen
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories])
                                  for i in range(num_epochs)]


#Ausgabe der Validierungsscores ohne die ersten zehn Datenpunkte
plt.plot(range(1, len(average_mae_history) + 1),
                                       average_mae_history)
plt.xlabel('Epochen')
plt.ylabel('Mittlerer absoluter Fehler Validierung')
plt.show()