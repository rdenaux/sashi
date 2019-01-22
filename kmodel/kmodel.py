from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
import os.path


sample_params = {
    'num_classes': 2,  # number of classes to predict
    'class_mode': 'categorical',  # type of model
    'train_loss': 'categorical_crossentropy',  # appropriate for class_mode
    'train_metrics': ['accuracy'],  # metrics to include during training
    'batch_size': 16,
    'top': {  # specifies how the top (custom) layers will look like
        'hidden_dense': [1024],  # if using hidden layers, specify dimensions
        'hidden_dropout_rate': 0.2,  # rate to use between top hidden layers
        'activation': 'softmax',  # this should be suitable for the class_mode
    },
    'top_train': {  # configure how to train the top (custom) layers
        'opt': 'adam',
        'epochs': 15,
        'checkpoint_path': 'cp.top.best.hdf5'
    },
    'fine_train': {  # configure how to train during fine-tuning
        'lr': 0.0001,  # for SGD
        'momentum': 0.9,
        'epochs': 5,
        'checkpoint_path': 'cp.fine_tuned.best.hdf5',
        'freeze_n_layers': 172  # optional, how many layers to freeze
    },
    'checkpoint_path': 'final_weights.hdf5'  # where to store model
}


def create_custom_model(custom_params):
    """Creates a custom model by extending InceptionV3

    :param custom_params: dict specifying how top layers should look like
    and more. See `sample_params` for an example.
    :returns: a dict providing access to: the base model, the full model,
    the image shape and the used custom parameters.
    :rtype: dict

    """
    base_model = InceptionV3(weights='imagenet', include_top=False)
    model = extend_base_model(base_model, custom_params)
    return {
        'base': base_model,
        'params': custom_params,
        'img_width': 299,
        'img_height': 299,
        'model': model
    }


def extend_base_model(base_model, params):
    """Creates Keras model extending a base_model with custom layers

    :param base_model: an extensible base model, see Keras applications
    :param params: dict speficying how top layers should look like.
    See `sample_params` for an example.
    :returns: a Keras model extending the `base_model` with custom layers
    :rtype: Keras Model
    """
    tparams = params['top']
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    dropout_rate = tparams.get('hidden_dropout_rate', None)
    # add hidden layers if specified (along with dropouts)
    for h_dim in tparams['hidden_dense']:
        x = Dense(h_dim, activation='relu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    # and a final logistic layer according to specification
    predictions = Dense(params['num_classes'],
                        activation=tparams['activation'])(x)
    return Model(inputs=[base_model.input], outputs=[predictions])


def train_phase(model_dict, phase, train_gen, valid_gen):
    """Fits the model according to the training parameters for the phase

    :param model_dict: as returned by `create_custom_model`
    :param phase: name of the training phase (`top` or `fine`)
    :param train_gen: training batch generator see `train_generator_for_dir`
    :param valid_gen: validation batch generator see
    `validation_generator_for_dir`. Can be `None` if not validating.
    :returns: the Keras History of the model fitting
    :rtype: Keras History

    """
    assert phase in ['top', 'fine']
    compile_for_training(model_dict, phase)
    md = model_dict
    tparams = md['params']['%s_train' % phase]
    # Callback to save model after every epoch.
    mc = ModelCheckpoint(
        tparams['checkpoint_path'],
        monitor='val_acc', verbose=0,
        save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Callback to save the TensorBoard logs.
    tb = TensorBoard(
        log_dir='./logs',
        # histogram_freq=1, # not allowed with validation generator
        write_graph=True, write_images=True)

    return md['model'].fit_generator(
        train_gen,
        steps_per_epoch=generator_steps_per_epoch(train_gen),
        epochs=tparams['epochs'],
        validation_data=valid_gen,
        validation_steps=generator_steps_per_epoch(valid_gen),
        callbacks=[mc, tb],
        verbose=1)


def compile_for_training(model_dict, phase):
    """Compiles the model so it's ready for a training phase

    :param model_dict: as returned by `create_custom_model`
    :param phase: either `top` (train only top layers) or `fine`
    (train also some intermediate layers)
    :returns: nothing, simply modifies the model
    :rtype: None

    """
    assert phase in ['top', 'fine']
    params = model_dict['params']
    t_params = params['%s_train' % phase]
    model = model_dict['model']
    if phase == 'top':
        freeze_base(model_dict)  # freeze all layers but the top
    elif phase == 'fine':
        freeze_bottom(model_dict, t_params.get('freeze_n_layers', 172))
    else:
        raise ValueError('Invalid phase %s' % phase)
    # compile the model (*after* freezing appropriate layers)
    model.compile(optimizer=optimizer(phase, t_params),
                  loss=params['train_loss'], metrics=params['train_metrics'])


def optimizer(phase, t_params):
    if phase == 'top':
        return t_params['opt']
    elif phase == 'fine':
        return SGD(lr=t_params['lr'], momentum=t_params['momentum'])
    else:
        raise ValueError('Invalid phase %s' % phase)


def freeze_base(model_dict):
    """Freezes all layers in the base model"""
    for layer in model_dict['base'].layers:
        layer.trainable = False


def freeze_bottom(model_dict, n):
    """Freezes the bottom `n` layers"""
    assert type(n) == int
    assert n > 0
    model = model_dict['model']
    assert n < len(model.layers)
    print('Freezing bottom %d layers (of %d)' % (n, len(model.layers)))
    for layer in model.layers[:n]:
        layer.trainable = False
    for layer in model.layers[n:]:
        layer.trainable = True


def load_weights(model_dict, weights_path):
    "Load weights into a model"
    model = model_dict['model']
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Checkpoint '%s' loaded." % weights_path)


# Data generator with some basic data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Data generator without augmentation (only rescaling)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def train_generator_for_dir(data_dir, model_dict):
    """Create a Keras generator suitable for training
    Data augmentation is performed during batching.

    :param data_dir: folder with subfolders for the classes and images therein
    :param model_dict: dict as returned by `create_custom_model`
    :returns: a generator for training batches suitable for training the model
    :rtype: ??
    """
    return _generator_for_dir(train_datagen, data_dir, model_dict)


def validation_generator_for_dir(data_dir, model_dict):
    """Create a Keras generator suitable for validation
    No data augmentation is performed.

    :param data_dir: folder with subfolders for the classes and images therein
    :param model_dict: dict as returned by `create_custom_model`
    :returns: a generator for batches suitable for validating the model
    :rtype: ??
    """
    return _generator_for_dir(test_datagen, data_dir, model_dict)


def _generator_for_dir(datagen, data_dir, model_dict):
    return datagen.flow_from_directory(
        data_dir,
        target_size=(model_dict['img_height'], model_dict['img_width']),
        batch_size=model_dict['params']['batch_size'],
        class_mode=model_dict['params']['class_mode'])


def generator_steps_per_epoch(gen):
    if gen is None:
        return None
    return gen.n // gen.batch_size
