from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense,  MaxPooling2D, Conv2D, Dropout
from keras.optimizers import SGD, Adam, rmsprop, Nadam
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2

#Datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

ytrain = to_categorical(y_train,10)
ytest = to_categorical(y_test,10)

#Initializing the CNN

model = Sequential()

#Step 1 - Convolutional
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3),activation="relu"))
#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',activation='relu'))

#Step 2 - Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout 

model.add(Dropout(0.25))

#Adding the convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

#Step 3 - Flatten
model.add(Flatten())


#Step 4 - Dense
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# compile model
# initiate RMSprop optimizer
opt = rmsprop(lr=0.0001, decay=1e-6)
optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# fit model
history = model.fit(x_train, ytrain, batch_size=32,validation_data=(x_test, ytest), epochs=100, verbose=0, shuffle=True)
# evaluate the model
_, train_acc = model.evaluate(x_train, ytrain, verbose=0)
_, test_acc = model.evaluate(x_test, ytest, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

