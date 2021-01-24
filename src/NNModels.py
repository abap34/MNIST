from types import new_class
import base
import tensorflow as tf
import yaml
import wandb


class SimpleMLPRegressor(base.BaseModel):
    def __init__(self, config):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(config['model']['input_drop_rate']))
        for unit in config['model']['units']:
            model.add(tf.keras.layers.Dense(unit, activation=config['model']['activation']))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config['model']['drop_rate']))

        model.add(tf.keras.layers.Dense(1))

        if config['optimizer']['optimizer'] == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        elif config['optimizer']['optimizer'] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        
        model.compile(optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

        self.model = model

        self.config = config

    def _fit(self, train_x, train_y, val_x, val_y):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            **self.config['reduce_lr']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            **self.config['early_stopping']
        )
        history = self.model.fit(
            train_x.values,
            train_y.values,
            epochs=100000,
            validation_data=(val_x.values, val_y.values),
            batch_size=self.config['fit']['batch_size'],
            verbose=self.config['fit']['verbose'],
            callbacks=[reduce_lr, early_stopping,  wandb.keras.WandbCallback()],
        )
        return history
    
    def _predict(self, test_x):
        return self.model.predict(test_x)



class SimpleMLPClassifier(base.BaseModel):
    def __init__(self, config):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dropout(config['model']['input_drop_rate']))
        for unit in config['model']['units']:
            model.add(tf.keras.layers.Dense(unit, activation=config['model']['activation']))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(config['model']['drop_rate']))

        model.add(tf.keras.layers.Dense(config['model']['n_class'], activation='softmax'))

        if config['optimizer']['optimizer'] == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        elif config['optimizer']['optimizer'] == "adam":
            optimizer = tf.keras.optimizers.Adam(
                lr=config['optimizer']['optimizer_lr'], decay=config['optimizer']['decay']
            )
        else:
            err_msg = "ignore optimizer. passed {}".format(config['optimizer']['optimizer'])
            raise ValueError(err_msg)

        model.compile(optimizer, loss=config['model']['loss'], metrics=config['model']['metrics'])

        self.model = model

        self.config = config

    def _fit(self, train_x, train_y, val_x, val_y):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            **self.config['reduce_lr']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            **self.config['early_stopping']
        )
        history = self.model.fit(
            train_x.values,
            train_y.values,
            epochs=100000,
            validation_data=(val_x.values, val_y.values),
            batch_size=self.config['fit']['batch_size'],
            verbose=self.config['fit']['verbose'],
            callbacks=[reduce_lr, early_stopping,  wandb.keras.WandbCallback()],
        )
        return history
    
    def _predict(self, test_x):
        return self.model.predict(test_x)
