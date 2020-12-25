from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

img_width = 299
img_height = 299
batch_size = 1
nbr_test_samples = 225

root_path = os.path.dirname(os.path.realpath(__file__))

weights_path = os.path.join(root_path, './weights.h5')

test_data_dir = './test_split'

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)

FishNames = ['Goldfish', 'Clownfish','Grass Carp','Soles','Catfish','Little Yellow Croaker','Butterfish','Snakehead']

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = FishNames,
        class_mode = 'categorical')

test_image_list = test_generator.filenames
print(len(test_image_list))

print('Loading model and weights from training process ...')
ResNet50_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = ResNet50_model.predict_generator(test_generator, nbr_test_samples)

np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)

# 输出分类的excel
print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,Goldfish, Clownfish,Grass Carp,Soles,Catfish,Little Yellow Croaker,Butterfish,Snakehead\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 10 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')

# 打印evaluate的test的准确率
test_generator.reset()
scores = ResNet50_model.evaluate_generator(test_generator,verbose=1)
print(scores)

# 打印混淆矩阵和classification report（需要修改文件夹路径和文件个数)
# y_pred = np.argmax(predictions, axis=1)
# print('Confusion Matrix')
# confusion = confusion_matrix(test_generator.classes, y_pred)
# print(confusion)
# print('Classification Report')
# target_names = ['Goldfish', 'Clownfish','Grass Carp','Soles','Catfish','Little Yellow Croaker','Butterfish','Snakehead']
# print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# 画出混淆矩阵
# plt.imshow(confusion, cmap=plt.cm.Blues)
# indices = range(len(confusion))
# plt.xticks(indices, FishNames,rotation=45)
# plt.yticks(indices, FishNames)
# plt.colorbar()
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# for first_index in range(len(confusion)):
#     for second_index in range(len(confusion[first_index])):
#         plt.text(first_index, second_index, confusion[second_index][first_index],va = 'center',ha = 'center')
 
# plt.show()