import numpy as np
import tensorflow as tf

def load_data(batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    X_train = np.load("./train_x.npy")
    y_train = np.load("./train_y.npy")
    X_train = X_train.astype("float32")
    y_train = y_train.astype("float32")

    X_test = np.load("./test_x.npy")
    y_test = np.load("./test_y.npy")
    X_test = X_test.astype("float32")
    y_test = y_test.astype("float32")

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, test_dataset