class BaseModel:
    def __init__(self, conifg):
        raise NotImplementedError

    def _fit(self, train_x, train_y, val_x, val_y):
        raise NotImplementedError

    def fit(self, train_x, train_y, val_x, val_y):
        if train_x.shape[0] != train_y.shape[0]:
            err_msg = "Incorrect size input. \n \
                       The shape of    \
                       train_x is {} \n \
                       train_y is {} \
                       val_x is {} \
                       val_y is {}".format(
                *list(map(lambda arr: arr.shape, [train_x, train_y, val_x, val_y]))
            )
            raise ValueError(err_msg)

        history = self._fit(train_x, train_y, val_x, val_y)
        return history

    def _predict(self, test_x):
        raise NotImplementedError

    def predict(self, test_x):
        pred = self._predict(test_x)
        return pred


