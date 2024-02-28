import copy

class ModelFactory:
    def __init__(self, model, model_name, X_train, y_train, X_test, y_test):
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.train_metrics = []
        self.test_metrics = []

    def deep_copy(self):
        return copy.deepcopy(self)

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_train)

    def test(self):
        return self.model.predict(self.X_test)

    # This API is added for model load cases
    def predict(self, test_data):
        self.X_test = test_data
        return self.model.predict(self.X_test)

