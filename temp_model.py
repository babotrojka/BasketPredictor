#coding=utf-8

try:
    import datetime
except:
    pass

try:
    from pathlib import Path
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperopt import tpe, Trials
except:
    pass

try:
    from hyperopt.pyll.stochastic import lognormal
except:
    pass

try:
    from keras.optimizers import Adam, RMSprop
except:
    pass

try:
    from callbacks import TensorboardReportImages
except:
    pass

try:
    from metrics import IOU
except:
    pass

try:
    from train import make_dataset, make_dataset_split
except:
    pass

try:
    from hyperas.distributions import choice, quniform, normal
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

dataset_root = Path("../dataset/")
dataset_config = {
    "batch_size": None,
}


    split: make_dataset_split(dataset_root / split, dataset_config)
    for split in ["train", "val", "test"]
}

def keras_fmin_fnct(space):

    inputs = (tf.keras.layers.Input(shape=(224, 224, 3), name="image"),)

    backbone = tf.keras.applications.convnext.ConvNeXtTiny(
        include_top=False,
    )(inputs)
    backbone.trainable = True

    out = tf.keras.layers.GlobalAveragePooling2D()(backbone)
    out = out[0]

    num_layers = int(space['int'])
    for ind in range(num_layers):
        out = tf.keras.layers.Dense(
            units=space['units'],
            activation=space['activation']
        )

    out = tf.keras.layers.Dense(units=4, activation="sigmoid", name="bbox")(out)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    lr = space['lr']
    optimizer = space['optimizer']

    optimizer = optimizer(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=space['decay_steps'],
            decay_rate=space['decay_rate'],
        )
    )

    model.compile(
        optimizer=optimizer,
        loss={
            "bbox": "mse",
        },
        metrics=[
            IOU(),
        ],
    )

    model.fit(
        dataset["train"],
        validation_data=dataset["val"],
        batch_size=space['batch_size'],
        epochs=300,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_iou",
                mode="max",
                patience=100,
                min_delta=0.01,
            )
        ],
        verbose=0,
    )

    results = model.evaluate(
        dataset["test"],
        batch_size=64,
        verbose=0,
    )

    test_iou = results[1]

    return {
        "loss": -test_iou,
        "status": "ok",
        "model": model,
    }

def get_space():
    return {
        'int': hp.quniform('int', 1, 4, 1),
        'units': hp.choice[512, 256, 64, 16],
        'activation': hp.choice('activation', ["relu", "swish"]),
        'lr': hp.choice('lr', [1e-3, 1e-4, 1e-5]),
        'optimizer': hp.choice('optimizer', [Adam, RMSprop]),
        'decay_steps': hp.normal('decay_steps', 5600, 1400),
        'decay_rate': hp.lognormal('decay_rate', 0.1, 0.2),
        'batch_size': hp.choice('batch_size', [16, 32]),
    }
