import numpy as np
import torch
from torchvision import datasets, transforms
import os
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

BATCH_SIZE = 64 #more balance for gradient updates
IMAGE_SIZE = 32 # CIFAR-10 is 32x32, better resolution than with 64x64 ( no resizing noise)
NUM_CLASSES = 10

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

#Dataset
#Muss nur delete fuer SISA geladen werden?
x_train = np.load("C:/Users/sophi/OneDrive/Documents/TH/Bachelorarbeit/Code/notebooks/data/cifar10_split10/x_delete10.npy")
y_train = np.load("C:/Users/sophi/OneDrive/Documents/TH/Bachelorarbeit/Code/notebooks/data/cifar10_split10/y_delete10.npy")

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=SEED
)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print("Number of training datapoints:", x_train.shape[0])
print("Number of validation datapoints:", x_val.shape[0])
print("Number of test datapoints:", x_test.shape[0])

#Augmentation = New variations of the training data, model must learn features more, not just plain memorization of exact patterns
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), # symmetry invariance
    layers.RandomTranslation(0.125, 0.125)  # ~4px shift for 32x32 # position invariance
])

# Konvertiere TensorFlow-Datasets zu NumPy-Arrays
def tf_dataset_to_numpy(ds):
    """tf.data.Dataset zu NumPy-Arrays."""
    X, y = [], []
    for features, labels in ds:
        X.append(features.numpy())
        y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

def train_preprocess(x, y):

    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = tf.cast(x, tf.float32)
    x = data_augmentation(x, training=True) #only applied on train data

    #preprocessing
    x = x / 255.0
    x = (x - 0.5) / 0.5   # scale to [-1, 1], normalizes pixel values into the format the ResNet model expects

    return x, y


def test_preprocess(x, y):
    x = tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = tf.cast(x, tf.float32)
    #preprocessing
    x = x / 255.0
    x = (x - 0.5) / 0.5   # scale to [-1, 1]
    return x, y


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

train_ds = train_ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=False).map(
    train_preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds=tf.data.Dataset.from_tensor_slices((x_val, y_val))

val_ds = val_ds.map(
    test_preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

test_ds = test_ds.map(
    test_preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Konvertiere zu NumPy-Arrays
x_train, y_train = tf_dataset_to_numpy(train_ds)
x_val, y_val = tf_dataset_to_numpy(val_ds)
x_test, y_test = tf_dataset_to_numpy(test_ds)

# Transponieung Daten von (N, H, W, C) zu (N, C, H, W) für PyTorch
x_train = np.transpose(x_train, (0, 3, 1, 2))  # (N, H, W, C) → (N, C, H, W)
x_val = np.transpose(x_val, (0, 3, 1, 2))  # (N, H, W, C) → (N, C, H, W)
x_test = np.transpose(x_test, (0, 3, 1, 2))    # (N, H, W, C) → (N, C, H, W)

y_train = y_train.flatten()  # (N, 1) → (N,)
y_val = y_val.flatten()  # (N, 1) → (N,)
y_test = y_test.flatten()    # (N, 1) → (N,)


def load(indices, category='train'):
    if category == 'train':
        return x_train[indices], y_train[indices]
    elif category == 'val':
        return x_val[indices], y_val[indices]
    elif category == 'test':
        return x_test[indices], y_test[indices]