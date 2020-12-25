from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_epochs = 50
batch_size = 8

train_data_dir = './train_split'
val_data_dir = './val_split'

FishNames = ['Goldfish', 'Clownfish','Grass Carp','Soles','Catfish','Little Yellow Croaker','Butterfish','Snakehead']


print('Loading ResNet50 Weights ...')
ResNet50_notop = ResNet50(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = ResNet50_notop.get_layer(index = -1).output
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

ResNet50_model = Model(ResNet50_notop.input, output)

optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
ResNet50_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# 设置模型保存val_accuracy最高的
best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose = 1, save_best_only = True)

# 数据增强设置
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# 验证集图片只进行缩放
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = './aug',
        # save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = FishNames,
        class_mode = 'categorical')

history = ResNet50_model.fit_generator(
        train_generator,
        epochs = nbr_epochs,
        validation_data = validation_generator,
        callbacks = [best_model])

# 画acc和loss曲线
print(history.history)
epochs=range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs,history.history['accuracy'],'b',label='Training accuracy')
plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.savefig('./model_acc.jpg')

plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('./model_loss.jpg')