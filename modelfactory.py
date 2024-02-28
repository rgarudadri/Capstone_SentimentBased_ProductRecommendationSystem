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

    def train(self):
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_train)

    def test(self):
        return self.model.predict(self.X_test)

    def set_test_data(self, test_data):
        self.X_test = test_data

    # This API is added for model load cases
    def predict(self):
        return self.model.predict(self.X_test)

    def evaluate_metrics_train(self, y_pred):
        accuracy = round(accuracy_score(self.y_train, y_pred), 2)
        precision = round(precision_score(self.y_train, y_pred), 2)
        recall = round(recall_score(self.y_train, y_pred), 2)
        f1 = round(f1_score(self.y_train, y_pred), 2)
        auc_score = round(roc_auc_score(self.y_train, y_pred), 2)
        self.train_metrics.append(accuracy)
        self.train_metrics.append(precision)
        self.train_metrics.append(recall)
        self.train_metrics.append(f1)
        self.train_metrics.append(auc_score)

        print("Train Data Metrics - ", self.model_name)
        print("="*20)
        print("Accuracy:", self.train_metrics[0])
        print("Precision:", self.train_metrics[1])
        print("Recall:", self.train_metrics[2])
        print("F1 Score:", self.train_metrics[3])
        print("AUC Score:", self.train_metrics[4])

        self.get_confusion_matrix_train(y_pred)
        return

    def evaluate_metrics_test(self, y_pred):
        accuracy = round(accuracy_score(self.y_test, y_pred), 2)
        precision = round(precision_score(self.y_test, y_pred), 2)
        recall = round(recall_score(self.y_test, y_pred), 2)
        f1 = round(f1_score(self.y_test, y_pred), 2)
        auc_score = round(roc_auc_score(self.y_test, y_pred), 2)
        self.test_metrics.append(accuracy)
        self.test_metrics.append(precision)
        self.test_metrics.append(recall)
        self.test_metrics.append(f1)
        self.test_metrics.append(auc_score)

        print("Test Data Metrics - ", self.model_name)
        print("="*20)
        print("Accuracy:", self.test_metrics[0])
        print("Precision:", self.test_metrics[1])
        print("Recall:", self.test_metrics[2])
        print("F1 Score:", self.test_metrics[3])
        print("AUC Score:", self.test_metrics[4])

        self.get_confusion_matrix_test(y_pred)
        return

    def get_confusion_matrix_train(self, y_pred):
        confusion_mat = confusion_matrix(self.y_train, y_pred)
        print("="*30)
        self.plot_confusion_matrix(confusion_mat, [0, 1])
        return

    def get_confusion_matrix_test(self, y_pred):
        confusion_mat = confusion_matrix(self.y_test, y_pred)
        print("="*30)
        self.plot_confusion_matrix(confusion_mat, [0, 1])
        return

    def plot_confusion_matrix(self, data, labels):
        sns.set(color_codes=True)
        plt.title("Confusion Matrix")
        ax = sns.heatmap(data, annot=True, cmap="Blues", fmt=".1f")
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set(ylabel="True Values", xlabel="Predicted Values")
        plt.show()
        return

    def get_train_metrics(self):
        return self.train_metrics

    def get_test_metrics(self):
        return self.test_metrics
