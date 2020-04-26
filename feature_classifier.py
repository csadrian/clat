import matplotlib
matplotlib.use('Agg')
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Attention, Lambda
from tensorflow.keras import optimizers, regularizers, callbacks
#from keras_multi_head import MultiHeadAttention

np.set_printoptions(threshold=sys.maxsize)

import neptune
import gin
import gin.tf
from absl import flags, app

print(tf.__version__)
tf.compat.v1.enable_eager_execution()

WEIGHT_DECAY = 0.0005
NUM_CLASSES = 100

class Dataset:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def get_tuple(self):
        return (self.x_train, self.y_train, self.x_test, self.y_test)


@gin.configurable
def load_data(dataset_name='cifar100', batch_size=None, full_batches_only=True):

    if dataset_name == 'cifar100':
        train_data_file = 'train_cifar100vgg_max_pooling2d_4_features.npz'
        test_data_file = 'test_cifar100vgg_max_pooling2d_4_features.npz'
    elif dataset_name == 'cifar10':
        train_data_file = 'train_cifar10vgg_max_pooling2d_4_features.npz'
        test_data_file = 'test_cifar10vgg_max_pooling2d_4_features.npz'
    else:
        raise Exception(dataset_name + ' not available.')

    with np.load(train_data_file) as data:
        size = (data['X'].shape[0] // batch_size) * batch_size if full_batches_only else data['X'].shape
        x_train = data['X'][:size]
        y_train = data['y'][:size]

    with np.load(test_data_file) as data:
        size = (data['X'].shape[0] // batch_size) * batch_size if full_batches_only else data['X'].shape
        x_test = data['X'][:size]
        y_test = data['y'][:size]

    dataset = Dataset(x_train, y_train, x_test, y_test)

    if 'cifar100' in train_data_file:
        num_classes = 100
    else:
        num_classes = 10

    dataset.num_classes = num_classes 

    print('shapes: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return dataset


def plot_history(history, name):
    # Plot training & validation accuracy values
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(name + ' accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(name + '_accuracy.png', bbox_inches='tight')

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(name + '_loss.png', bbox_inches='tight')


class Magic(tf.keras.layers.Layer):

  def __init__(self, input_dim=512, k_dim=100, v_size=30, output_dim=100, q_proj_trainable=True, k_proj_trainable=True, v_proj_trainable=True):
    super(Magic, self).__init__()
    self.q_proj = self.add_weight(shape=(input_dim, k_dim),
                             initializer='random_normal',
                             trainable=q_proj_trainable)
    self.k_proj = self.add_weight(shape=(v_size, k_dim),
                             initializer='random_normal',
                             trainable=k_proj_trainable)
    self.v_proj = self.add_weight(shape=(v_size, output_dim * input_dim),
                             initializer='random_normal',
                             trainable=v_proj_trainable)

  def call(self, inputs):
    q = tf.matmul(inputs, self.q_proj)
    q = q / tf.linalg.norm(q, axis=0)
    #k = tf.matmul(inputs, self.k_proj)
    k = self.k_proj / tf.linalg.norm(self.k_proj, axis=0)
    v = self.v_proj
    return q, k, v



def build_model_simple_ff(num_classes):

    model = tf.keras.Sequential()
    model.add(Flatten())
    #model.add(Dense(512, kernel_regularizer=regularizers.l2(WEIGHT_DECAY)))
    #model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.build(input_shape=(1,1,512))
    return model

@gin.configurable
def build_model_meta_attention(num_classes, v_size=20, q_proj_trainable=True, k_proj_trainable=True, v_proj_trainable=True):
    a = tf.keras.Input(batch_shape=(64, 1, 1, 512))
    x = Flatten()(a)

    #x = Lambda(lambda mu: tf.matrix_diag(mu))(x)
    """
    q, k, v = Magic(v_size=80, output_dim=NUM_CLASSES)(x)
    xx = Attention()([q, v, k])
    xx_mat = tf.reshape(xx, (64, 512, 512))
    x = tf.einsum('ij,ijk->ik', x, xx_mat)

    """
    q, k, v = Magic(v_size=v_size, output_dim=num_classes, q_proj_trainable=q_proj_trainable, k_proj_trainable=k_proj_trainable, v_proj_trainable=v_proj_trainable)(x)
    xx = Attention()([q, v, k])

    scores = tf.nn.softmax(tf.matmul(q, tf.transpose(k)))
    xx_mat = tf.reshape(xx, (64, 512, num_classes))
    x = tf.einsum('ij,ijk->ik', x, xx_mat)

    z = Activation('softmax')(x)
    #z = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=a, outputs=z)
    model_scores = Model(inputs=a, outputs=scores)

    return model, model_scores

@gin.configurable
def build_model(num_classes, model_name=gin.REQUIRED):
    if model_name == 'simple_ff':
        return build_model_simple_ff(num_classes=num_classes)
    elif model_name == 'meta_attention':
        return build_model_meta_attention(num_classes=num_classes)
    else:
        raise Exception('Model not implemented: ' + str(model_name))
 
@gin.configurable
def train_model(batch_size=64, epochs_per_class=20):

    data = load_data(batch_size=batch_size)
    x_train, y_train, x_test, y_test = data.get_tuple()
    num_classes = data.num_classes

    model, model_scores = build_model(num_classes = num_classes)
    model.summary()

    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                            patience=5, min_lr=0.00001)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
                  #callbacks=[reduce_lr])

    print('# Fit model on training data')
    hist_data = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}
    for i in range(num_classes):
        print('\nFIT ON {}. CLASS\n'.format(i))
        x = x_train[(y_train == i)]
        y = y_train[(y_train == i)]

        size = x.shape[0]//batch_size * batch_size
        x = x[:size]
        y = y[:size]

        history = model.fit(x, y,
                            batch_size=batch_size,
                            epochs=epochs_per_class,
                            validation_data=(x_test, y_test))
        hist_data['acc'].extend(history.history['acc'])
        hist_data['loss'].extend(history.history['loss'])
        hist_data['val_acc'].extend(history.history['val_acc'])
        hist_data['val_loss'].extend(history.history['val_loss'])

        pred = model_scores.predict(x_test)
        for j in range(num_classes):
            counts = np.bincount(np.argmax(pred[(y_test == j)], axis=-1))
            print('counts: ', j, counts)
        #print(pred)

    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test loss, test acc: ', results)
    return hist_data


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    if use_neptune:
        neptune.init(project_qualified_name="csadrian/clat")
        print(gin.operative_config_str())
        exp = neptune.create_experiment(params={}, name="exp")
        #for tag in opts['tags'].split(','):
        #  neptune.append_tag(tag)

    history = train_model()
    plot_history(history, 'attention_icl' + FLAGS.gin_file)
    print('fin')

if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)
