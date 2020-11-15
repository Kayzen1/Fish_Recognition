from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_width = 299
img_height = 299
batch_size = 8
nbr_test_samples = 74

FishNames = ['Goldfish', 'Clownfish','Grass Carp','Soles','Weever','Little Yellow Croaker']

root_path = os.path.dirname(os.path.realpath(__file__))

weights_path = os.path.join(root_path, './weights_valacc.h5')

test_data_dir = './test_split'

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

test_image_list = test_generator.filenames
print(test_image_list)

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)


print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,Goldfish, Clownfish,Grass Carp,Soles,Weever,Little Yellow Croaker\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 10 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
