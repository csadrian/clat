import matplotlib
matplotlib.use('Agg')
import os, sys, io
import PIL
import seaborn as sns; sns.set(style="ticks", font_scale=1.4, rc={"lines.linewidth": 2.5})
import matplotlib.pyplot as plt; plt.style.use('seaborn-deep')
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
from keras.datasets import cifar100
import cifar100vgg

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
def load_data(dataset_name='cifar100', batch_size=None, full_batches_only=True, train_data_file=None, test_data_file=None, num_classes=100):
    print('Loading data: {}.'.format(dataset_name))

    if dataset_name == 'features':
        pass
    elif dataset_name == 'cifar100_images':
        num_classes = 100
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = cifar100vgg.cifar100vgg.normalize(x_train, x_test)
        y_train, y_test = np.reshape(y_train, (-1,)), np.reshape(y_test, (-1,))
        size = (x_train.shape[0] // batch_size) * batch_size if full_batches_only else x_train.shape[0]
        x_train = x_train[:size]
        y_train = y_train[:size]

        size = (x_test.shape[0] // batch_size) * batch_size if full_batches_only else x_test.shape[0]
        x_test = x_test[:size]
        y_test = y_test[:size]

        #y_train = keras.utils.to_categorical(y_train, num_classes)
        #y_test = keras.utils.to_categorical(y_test, num_classes)

        #data augmentation
        """
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).

        datagen.fit(x_train)
        """
        dataset = Dataset(x_train, y_train, x_test, y_test)
        dataset.num_classes = num_classes
        return dataset
    else:
        raise Exception(dataset_name + ' not available.')

    with np.load(train_data_file) as data:
        size = (data['X'].shape[0] // batch_size) * batch_size if full_batches_only else data['X'].shape
        x_train = data['X'][:size]
        y_train = data['y'][:size]
        if 'superclass' in data.keys():
            y_train = (y_train, data['superclass'][:size])


    with np.load(test_data_file) as data:
        size = (data['X'].shape[0] // batch_size) * batch_size if full_batches_only else data['X'].shape
        x_test = data['X'][:size]
        y_test = data['y'][:size]

    dataset = Dataset(x_train, y_train, x_test, y_test)
    dataset.num_classes = num_classes

    return dataset


def fig2img(fig):
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    w, h, d = buf.shape
    return PIL.Image.frombytes("RGBA", (w, h), buf.tostring( ))


def plot_history(history, name):
    # Plot training & validation accuracy values
    fig = plt.figure(figsize=(10,7))
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(name + ' accuracy')
    plt.ylim(-0.1, 1.1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.tight_layout()

    accuracy_plot = fig2img(fig)
    neptune.send_image('accuracy', accuracy_plot)
    plt.clf()

    # Plot training & validation loss values
    fig = plt.figure(figsize=(10,7))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    fig.tight_layout()

    loss_plot = fig2img(fig)
    neptune.send_image('loss', loss_plot)
    plt.clf()


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


def build_model_cifar100vgg_with_ma(num_classes, v_size=100, q_proj_trainable=True, k_proj_trainable=True, v_proj_trainable=True):
    cifar100vgg_model = cifar100vgg.cifar100vgg(head=False)
    model = cifar100vgg_model.model
    super_head, scores = build_model_meta_attention(num_classes)
    model.summary()
    super_head.summary()

    a = tf.keras.Input(batch_shape=(64, 32, 32, 3))
    x = model(a)
    x = Flatten()(x)
    #x = Dense(512)(x)
    #x__ = Dense(num_classes)(x)
    q, k, v = Magic(v_size=v_size, output_dim=num_classes, q_proj_trainable=q_proj_trainable, k_proj_trainable=k_proj_trainable, v_proj_trainable=v_proj_trainable)(x)
    print(q, k, v)
    xx = Attention()([q, v, k])
    scores = tf.nn.softmax(tf.matmul(q, tf.transpose(k)))
    xx_mat = tf.reshape(xx, (64, 512, num_classes))
    x = tf.einsum('ij,ijk->ik', x, xx_mat)

    z = Activation('softmax')(x)
    #z = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=a, outputs=z)
    model_scores = Model(inputs=a, outputs=scores)

    return model, model_scores

    a = tf.keras.Input(batch_shape=(64, 32, 32, 3))
    x = model(a)
    print(x)
    x = super_head(x)
    s = scores(x)

    model = Model(inputs=a, outputs=x)
    model_scores = Model(inputs=a, outputs=s)

    return model, model_scores

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
    a = tf.keras.Input(batch_shape=(64, 512))
    x = a #Flatten()(a)

    #x = Lambda(lambda mu: tf.matrix_diag(mu))(x)
    """
    q, k, v = Magic(v_size=80, output_dim=NUM_CLASSES)(x)
    xx = Attention()([q, v, k])
    xx_mat = tf.reshape(xx, (64, 512, 512))
    x = tf.einsum('ij,ijk->ik', x, xx_mat)

    """
    q, k, v = Magic(v_size=v_size, output_dim=num_classes, q_proj_trainable=q_proj_trainable, k_proj_trainable=k_proj_trainable, v_proj_trainable=v_proj_trainable)(x)
    print(q, k, v)
    xx = Attention()([q, v, k], mask=None)

    scores = tf.nn.softmax(tf.matmul(q, tf.transpose(k)))
    xx_mat = tf.reshape(xx, (64, 512, num_classes))
    x = tf.einsum('ij,ijk->ik', x, xx_mat)

    z = Activation('softmax')(x)
    #z = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=a, outputs=z)
    model_scores = Model(inputs=a, outputs=scores)

    return model, model_scores


def build_model_pretrainedvgg16(num_classes, input_shape=(32,32,3)):
    base_model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    logits = Dense(num_classes)(x)
    predicted_probs = Activation('softmax')(logits)
    model = Model(inputs=base_model.input, outputs=predicted_probs)
    return model, None


@gin.configurable
def build_model(num_classes, model_name=gin.REQUIRED):
    if model_name == 'simple_ff':
        return build_model_simple_ff(num_classes=num_classes)
    elif model_name == 'meta_attention':
        return build_model_meta_attention(num_classes=num_classes)
    elif model_name == 'cifar100vgg_with_ma':
        return build_model_cifar100vgg_with_ma(num_classes=num_classes)
    elif model_name == 'pretrainedvgg16':
        return build_model_pretrainedvgg16(num_classes=num_classes)
    else:
        raise Exception('Model not implemented: ' + str(model_name))



@gin.configurable
def train_model(batch_size=64, epochs_per_split=20, num_splits=None, step_with_generated=False):

    data = load_data(batch_size=batch_size)
    x_train, y_train, x_test, y_test = data.get_tuple()
    num_classes = data.num_classes

    if isinstance(y_train, tuple) and num_splits is None:
        print('Splitting into tasks using superclasses.')
        labels_for_splitting = y_train[1]
        print('Superclass labels: ', np.unique(labels_for_splitting))
        num_splits = len(np.unique(labels_for_splitting))
        num_labels = num_splits
        y_train = y_train[0]
    elif isinstance(y_train, tuple):
        y_train = y_train[0]
        labels_for_splitting = y_train
        num_labels = num_classes
    else:
        if num_splits is None:
            num_splits = num_classes
        labels_for_splitting = y_train
        num_labels = num_classes

    print('Number of splits: ', num_splits)

    print('Train & test array shapes: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_datasets = []
    labels_in_splits = []

    if num_splits > 1:
        assert  num_labels % num_splits == 0, (
            'The number classes should be divisible by the number of splits.')
        num_concurrent_labels = num_labels // num_splits
        labels = list(range(num_labels))

        concurrent_labels = None

        for i in range(0, num_labels, num_concurrent_labels):
            concurrent_labels = labels[i:i+num_concurrent_labels]
            filter_mask = np.isin(labels_for_splitting, concurrent_labels)
            x_train_filtered = x_train[filter_mask]
            y_train_filtered = y_train[filter_mask]
            train_datasets.append((x_train_filtered, y_train_filtered))
            labels_in_splits.append(concurrent_labels)
    else:
        train_datasets = [(x_train, y_train)]
        labels_in_splits = list(range(num_labels))

    assert len(train_datasets) == num_splits, (
        'Number of training datasets must be equal to number of splits.')

    model, model_scores = build_model(num_classes = num_classes)
    model.summary()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    def generate_inputs(labels, batch_size, model, loss_object):
        label_batch = np.random.choice(labels, batch_size, p=None)
        shape = model.layers[0].input_shape[0][1:]
        image = 128 * np.ones(shape, dtype=np.uint8)
        image = tf.cast(image, tf.float32)
        gray_img = image / 255.0
        gray_img_batch = np.repeat(gray_img[np.newaxis,...], batch_size, axis=0)

        for _ in range(10):
            gray_imgs_tensor = tf.convert_to_tensor(gray_img_batch)

            with tf.GradientTape() as tape:
                tape.watch(gray_imgs_tensor)
                predicted_probs = model(gray_imgs_tensor)
                true_labels = tf.reshape(label_batch, (batch_size, 1))
                loss = loss_object(true_labels, predicted_probs)

            gradients = tape.gradient(loss, gray_imgs_tensor)
            gray_img_batch = gray_img_batch + gradients

        generated_input_batch = gray_img_batch + gradients

        generated_ds = tf.data.Dataset.from_tensor_slices((generated_input_batch, label_batch)).batch(batch_size)
        return generated_ds



    print('# Fit model on training data')
    hist_data = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}


    for i in range(num_splits):
        print('\nFIT ON {}. SPLIT\n'.format(i))
        x = train_datasets[i][0]
        y = train_datasets[i][1]

        size = x.shape[0]//batch_size * batch_size
        x = x[:size]
        y = y[:size]

        num_batches = size // batch_size
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(num_batches)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(x_test.shape[0]//batch_size)

        for epoch in range(epochs_per_split):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(images, labels)

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result() * 100,
                                  test_loss.result(),
                                  test_accuracy.result() * 100))
            hist_data['acc'].append(train_accuracy.result())
            hist_data['loss'].append(train_loss.result())
            hist_data['val_acc'].append(test_accuracy.result())
            hist_data['val_loss'].append(test_loss.result())

            if step_with_generated:
                # step with generated
                train_loss.reset_states()
                train_accuracy.reset_states()

                seen_labels = list(np.array(labels_in_splits[:num_splits]).flat)

                generated_ds = generate_inputs(seen_labels, batch_size, model, loss_object)
                for images, labels in generated_ds:
                    train_step(images, labels)

        if gin.query_parameter('build_model.model_name') in ['cifar100vgg_with_ma', 'meta_attention']:
            pred = model_scores.predict(x_test)
            all_counts = []
            for j in range(num_classes):
                counts = np.bincount(np.argmax(pred[(y_test == j)], axis=-1), minlength=gin.query_parameter('build_model_meta_attention.v_size'))
                all_counts.append(counts)
            all_counts = np.array(all_counts)
            heatmap = sns.heatmap(all_counts)
            buffer = io.StringIO()
            canvas = plt.get_current_fig_manager().canvas
            canvas.draw()
            plot = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
            neptune.send_image('heatmap_c_v', plot)
            plt.clf()
            print('sum counts per v_size: ', np.sum(all_counts, axis=0))

    print('\n# Evaluate on test data')
    test_loss.reset_states()
    test_accuracy.reset_states()
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(x_test.shape[0]//batch_size)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss.result(),
                          test_accuracy.result() * 100))
    return hist_data


def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

    use_neptune = "NEPTUNE_API_TOKEN" in os.environ
    if use_neptune:
        neptune.init(project_qualified_name="csadrian/clat")

        print(gin.operative_config_str())
        exp = neptune.create_experiment(params={}, name="exp")
    else:
        neptune.init('shared/onboarding', api_token='ANONYMOUS', backend=neptune.OfflineBackend())

    history = train_model()
    plot_history(history, FLAGS.gin_file[0].split('/')[1].split('.')[0])
    neptune.stop()
    print('fin')


if __name__ == '__main__':
    flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
    flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS

    app.run(main)
