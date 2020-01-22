import os
import numpy as np
from tqdm import tqdm
import scipy
import matplotlib
from matplotlib import pyplot as plt
import skimage
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import cv2
from mlxtend.plotting import plot_confusion_matrix
import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation , Dropout, Flatten , Conv2D , BatchNormalization, MaxPool2D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras.constraints import maxnorm
from keras import backend as k
k.set_image_data_format('channels_first')

# As the data is already sorted and split into test , train and validation folders (using zshell) now it's just to
# feed in those directories directly

train_dir = "~/chest_xray/train"
test_dir = "~/chest_xray/test"
val_dir = "~/chest_xray/val"

# Now labels are to be extracted and images are to be preprocessed

def data_preprocess(path):
	X = []
	Y = []
	for Dir in os.listdir(path):
		if not Dir.startswith('.'):
			if Dir in ['NORMAL']:
				label = 0
			elif Dir in ['PNEUMONIA']:
				label = 1
			else:
				label = 2

			tmp = path +'/'+ Dir

			for file in tqdm(os.listdir(tmp)):
				img = cv2.imread(tmp + '/' + file)
				if img is not None:
					img = skimage.transform.resize(img, (150, 150, 3))
					img = np.asarray(img)
					X.append(img)
					Y.append(label)

	X = np.asarray(X)
	Y = np.asarray(Y)
	return X, Y

# images and labels are loaded in respective variables

X_train, Y_train = data_preprocess(train_dir)

X_test , Y_test = data_preprocess(test_dir)

X_val , Y_val = data_preprocess(val_dir)

print (X_train.shape, '/n',  X_test.shape, '/n', X_val.shape)
print (Y_train.shape, '/n', Y_test.shape, '/n', Y_val.shape)
print('Encoding labels...')

# onehot encoding labels

Y_train = to_categorical(Y_train,2)
Y_test = to_categorical(Y_test,2)
Y_val = to_categorical(Y_val,2)

print (Y_train.shape, '/n', Y_test.shape, '/n', Y_val.shape)


# as the class data is imbalenced , we are measuring precision , recall and confusion matrix plot

#callbacks used to reduce learning rate by monitoring 'val_acc'

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=1,verbose=1,min_delta=0.0001)

#using InceptionV3 weights
# Checkpoints are used to monitor and save best model and avoid val_acc drop due to overfitting

weights_path = '~/inception_v3_weights.h5'
check_point = ModelCheckpoint(weights_path,monitor='val_acuracy',verbose=1,save_best_only=True,mode='max')

#reshape data according to weights

X_train = X_train.reshape(5216,3,150,150)
X_test = X_test.reshape(624,3,150,150)
X_val = X_val.reshape(16,3,150,150)


def swish_activation(x):
	return (k.sigmoid(x)*x)

model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',padding='same',input_shape=(3,150,150)))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(3,150,150)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(96,(3,3),dilation_rate=(2,2),activation='relu',padding='same'))
model.add(Conv2D(96,(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),dilation_rate=(2,2),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),padding='valid',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64,activation=swish_activation))
model.add(Dropout(0.4))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.00005),metrics=['accuracy'])

print(model.summary())

batch_size = 256
epochs = 6

History = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),callbacks=[reduce_lr,check_point],epochs=epochs)

model.save('New_Inception.h5')

# history of model accuracy
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.savefig('model accuracy',format='png')
plt.show()

# history of model loss
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper right')
plt.savefig('model loss',format='png')
plt.show()


prediction = model.predict(X_test)
prediction = np.argmax(prediction,axis = 1)
Y_True = np.argmax(Y_test,axis=1)

ConfusionMatrix = confusion_matrix(Y_True,prediction)
fig , ax = plot_confusion_matrix(ConfusionMatrix,figsize=(5,5))
plt.savefig('confusion matrix',format='png')
plt.show()








