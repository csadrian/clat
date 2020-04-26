import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

tf.enable_eager_execution()

MODEL_PATH = 'cifar10vgg.h5'
DATA_DIR = None
DATASET_NAME = 'cifar10'
NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)
OUT_LAYER_INDICES = [51]
BATCH_SIZE = 8
SPLIT = 'test'


def preprocess(model_path, ds):
    if model_path not in ['cifar10vgg.h5', 'cifar100vgg.h5']:
        def normalize(image, label):
            image = tf.cast(image, tf.float32)
            image = (image/127.5) - 1
            return image, label
        return ds.map(normalize)

    elif model_path == 'cifar10vgg.h5':
        mean = 120.707
        std = 64.15
    elif model_path == 'cifar100vgg.h5':
        mean = 121.936
        std = 68.389

    print('normalize - mean, std: ', mean, std)
    def normalize(image, label):
        print('normalize - mean, std: ', mean, std)
        image = tf.cast(image, tf.float32)
        return (image-mean)/(std+1e-7), label

    ds = ds.map(normalize)
    return ds


def get_dataset(dataset_name, data_dir, ds_split):
    ds_train, ds_info = tfds.load(name=dataset_name,
                                  data_dir=data_dir,
                                  split=ds_split,
                                  with_info=True,
                                  as_supervised=True)
    class_names = ds_info.features["label"].names
    assert isinstance(ds_train, tf.data.Dataset)
    return ds_train


def get_model(image_shape, num_classes, out_layer_idx, model_path=None):
    if model_path in ['cifar10vgg.h5', 'cifar100vgg.h5']:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Activation, \
            BatchNormalization, Dropout, MaxPooling2D, Flatten
        from tensorflow.keras import regularizers
        model = Sequential()
        weight_decay = 0.0005

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=image_shape,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))


        model.load_weights(model_path)
    else:
        model_path = 'VGG16'
        model = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=image_shape)

    for idx, layer in enumerate(model.layers):
        print(idx, '. ', layer.name)

    model.trainable = False
    out_layers = [model.layers[i] for i in out_layer_idx]
    #out = [tf.reduce_max(item.output, axis=(1, 2)) for item in out_layers]
    out = [item.output for item in out_layers]

    new_model = Model(inputs=model.inputs, outputs=out)
    return new_model, out_layers


def extract_featuremaps(ds_split, model_path, ds, model, out_layers, batch_size):
    ds = preprocess(model_path, ds)

    print('Extracting fetures from model.')
    outputs = [[] for i in range(len(out_layers))]
    class_labels = []

    for item in ds.batch(batch_size).repeat(1):
        images, labels = item[0].numpy(), item[1].numpy()
        class_labels.append(labels)
        model_outputs = model.predict(images)
        if len(outputs) == 1:
            model_outputs = [model_outputs]

        for idx, layer in enumerate(out_layers):
            layer_output = model_outputs[idx]
            outputs[idx].append(layer_output)

    target = np.concatenate(class_labels, axis=0)

    for i in range(len(outputs)):
        output = np.concatenate(outputs[i], axis=0)
        print('Final array shape: ', output.shape)
        print('Saving {} layer outputs.'.format(out_layers[i].name))
        np.savez('{}_{}_{}_features'.format(ds_split, model_path.split('.')[0], out_layers[i].name),
                 X=output,
                 y=target)


if __name__ == '__main__':
    ds = get_dataset(DATASET_NAME, DATA_DIR, SPLIT)
    model, out_layers = get_model(IMAGE_SHAPE, NUM_CLASSES, OUT_LAYER_INDICES, MODEL_PATH)
    extract_featuremaps(SPLIT, MODEL_PATH, ds, model, out_layers, BATCH_SIZE)
    print('fin')