import os
import numpy as np
import shutil

np.random.seed(2016)

root_train = './train_split'
root_val = './val_split'
root_test = './test_split'

root_total = './train'

FishNames = ['goldfish', 'squid','bianbian','matou','xiaohong','crab','Larimichthys polyactis']

nbr_train_samples = 0
nbr_val_samples = 0
test_samples = 0

# Training proportion
split_proportion = 0.6

for fish in FishNames:
    if fish not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, fish))

    total_images = os.listdir(os.path.join(root_total, fish))

    nbr_train = int(len(total_images) * split_proportion)
    val_or_test = int(len(total_images) * (1-split_proportion)/2) 

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]
    val_images = total_images[nbr_train:nbr_train+val_or_test]
    test_images = total_images[nbr_train+val_or_test:]

    for img in train_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_train, fish, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if fish not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, fish))

    for img in val_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_val, fish, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

    if fish not in os.listdir(root_test):
        os.mkdir(os.path.join(root_test, fish))
    
    for img in test_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_test, fish, img)
        shutil.copy(source, target)
        test_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {},# test samples: {}'.format(nbr_train_samples, nbr_val_samples,test_samples))
