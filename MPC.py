import pandas as pd

train_dataset = pd.read_csv("train.csv")

#Datapreparation Step
X_label = train_dataset.iloc[:,:20]

Y_labels = train_dataset.iloc[:,20]

#Onehot Encoding of Y_Labels (Target Outputs)
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features = [0])
Y_labels = oneHotEncoder.fit_transform(Y_labels.reshape(2000,1)).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_label, Y_labels)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

#Initialising the ANN
Nmodel = Sequential()

#Adding Dense Layers
Nmodel.add(Dense(units = X_label.shape[1], init = 'uniform', activation = 'relu', input_dim=X_label.shape[1]))
Nmodel.add(Dropout(0.1))
Nmodel.add(Dense(units = 15, init = 'uniform', activation = 'sigmoid'))
Nmodel.add(Dense(units = 10, init = 'uniform', activation = 'relu'))
Nmodel.add(Dense(units = 5, init = 'uniform', activation = 'sigmoid'))
Nmodel.add(Dense(units = Y_labels.shape[1], init = 'uniform', activation = 'softmax'))


#Nmodel.add(Activation('sigmoid', name='activation'))

Nmodel.summary()

#Compile the model
Nmodel.compile(optimizer= 'rmsprop', loss='binary_crossentropy', metrics = ['categorical_accuracy'])

#Fitting the NN model
#Nmodel.fit(X_train, y_train, epochs= 200, batch_size=32)

history = Nmodel.fit(X_train, y_train, epochs= 70, batch_size=50, validation_data=[X_test,y_test])



import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['X_train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


test_dataset = pd.read_csv("test.csv")

#data - transform to numerical form

X_pred = test_dataset.iloc[:,1:21]

y_prediction = Nmodel.predict_classes(X_pred)

y_pred = Nmodel.predict(X_pred)

Solution_frame = pd.DataFrame(y_prediction, columns= ['price_range'])

delete = test_dataset.id
Solution_frame.insert(loc=0, column='id', value=delete)
Solution_frame.to_csv("Solution.csv", index = False)
