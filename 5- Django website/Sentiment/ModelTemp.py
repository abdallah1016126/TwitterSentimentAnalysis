import numpy as np
import pickle
class layer():
    def __init__(self, n_output, n_input, type_, activation="sigmoid", parameter_initializer="uniform", target=None):
        self.n_output = n_output
        self.n_input = n_input
        self.activation = self.get_activations()[activation]
        self.act_name = activation
        self.target = target
        self.type = type_

        # Trying different weight initializations
        if parameter_initializer == "he_normal":
            self.W = np.random.randn(self.n_output, self.n_input) * np.sqrt(2 / self.n_input)
            self.b = np.random.randn(self.n_output, 1) * np.sqrt(2 / self.n_input)
        elif parameter_initializer == "normal":
            self.W = np.random.randn(0, 1, (self.n_output, self.n_input))
            self.b = np.random.randn(0, 1, (self.n_output, 1))
        elif parameter_initializer == "uniform":
            self.W = np.random.uniform(-0.05, 0.05, (self.n_output, self.n_input))
            self.b = np.random.uniform(-0.05, 0.05, (self.n_output, 1))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.Z = None
        self.X = None

    def _onehot(self, a, M):
        b = np.zeros((a.size, M), dtype='int')
        b[np.arange(a.size), a] = 1
        return b.T

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-1 * z)))

    def _diff_sigmoid(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def softmax(self, z):
        exp = np.exp(z)
        tot = exp.sum(axis=0)
        t = exp / tot
        return t

    def _diff_softmax(self, z, y):
        yhat_r = self.softmax(z)
        onehotY = self._onehot(y, z.shape[0])
        one_yi = onehotY * -1 * (1 - yhat_r)
        z = (1 - onehotY) * yhat_r
        return one_yi + z

    def get_activations(self):
        return {"softmax": self.softmax, "sigmoid": self.sigmoid}

    def get_activations_diff(self):
        return {"softmax": self._diff_softmax, "sigmoid": self._diff_sigmoid}

    def get_params(self):
        return [self.W, self.b]

    def zeroing_delta(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def _set_target(self, t):
        self.target = t

    def forward(self, input_, t=False):
        self.X = input_
        z = np.dot(self.W, self.X) + self.b
        A = self.activation(z)
        self.Z = z

        return A

    def backward(self, input_):
        if self.act_name == "softmax":
            f_dash = self._diff_softmax(self.Z, self.target)
        else:
            f_dash = self.get_activations_diff()[self.act_name](self.Z)

        e = np.ones((self.X.shape[1], 1))
        bet = input_ * f_dash

        self.dW = self.dW + np.dot(bet, self.X.T)
        self.db = self.db + np.dot(bet, e)

        return np.dot(self.W.T, bet)


class Model():
    def __init__(self, all_words, optimizer):
        self.layers = []
        self.all_words = all_words
        self.optimizer = optimizer

    def SoftmaxLogLikelihood(self, y, yhat):
        onehotY = self._onehot(y, yhat.shape[0])
        yhat_r = np.max(onehotY * yhat, axis=0, keepdims=True)
        return (1 / (y.shape[0])) * -1 * np.sum(np.log(yhat_r))

    def _onehot(self, a, M):
        b = np.zeros((a.size, M), dtype='int')
        b[np.arange(a.size), a] = 1
        return b.T

    def transform_words(self, x):
        encoding = np.zeros(len(self.all_words))
        i = 0
        for word in x:
            if word in self.all_words:
                encoding[self.all_words[word]] = 1
        return encoding.astype(np.int8)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input_):
        X_words = input_[8:, :]
        X_time = input_[:8, :]

        layer_index = 0
        while layer_index < len(self.layers):
            layer = self.layers[layer_index]
            if layer.type != "word_embedding":
                break
            layer_index += 1
            X_words = layer.forward(X_words)

        a = np.concatenate((X_words, X_time), axis=0)
        while layer_index < len(self.layers):
            layer = self.layers[layer_index]
            t = False
            a = layer.forward(a, t)
            layer_index += 1

        return a

    def backward(self, input_):
        gd = input_
        layer_index = len(self.layers) - 1

        while layer_index >= 0:
            layer = self.layers[layer_index]
            if layer.type != 'full':
                break
            layer_index -= 1
            gd = layer.backward(gd)

        nout_time = 8
        gd_words = gd[:gd.shape[0] - nout_time, :]
        gd_time = gd[gd.shape[0] - nout_time:, :]

        while layer_index >= 0:
            layer = self.layers[layer_index]
            layer_index -= 1
            gd_words = layer.backward(gd_words)

    def zeroing(self):
        for layer in self.layers:
            layer.zeroing_delta()

    def batch(self, x, y, bs):
        x = x.copy()
        y = y.copy()
        rem = x.shape[0] % bs

        for i in range(0, x.shape[0], bs):
            yield (x[i:i + bs], y[i:i + bs])

        if rem != 0:
            yield (x[x.shape[0] - rem:], y[x.shape[0] - rem:])

    def fit(self, train_data, validation_data=None, batch_size=32, epochs=5):
        x_train = train_data[0]
        y_train = train_data[1]
        no_of_batches_train = np.ceil(x_train.shape[0] / batch_size)

        if validation_data:
            x_valid = validation_data[0]
            y_valid = validation_data[1]

        for i in range(epochs):
            self.optimizer.init_params(self.layers)
            print()
            print("Epoch {}/{}".format(i + 1, epochs))
            j = 0
            k = 0
            data = self.batch(x_train, y_train, batch_size)
            losses = []

            for temp_x, temp_y in data:
                k += 1
                curr_x = temp_x.copy()
                curr_y = temp_y.copy()

                word_encodings = []
                for dic in curr_x[:, 0]:
                    word_encodings.append(self.transform_words(dic))
                words = np.array(word_encodings)
                curr_x = curr_x[:, 1:].astype(np.int8)
                curr_x = np.concatenate((curr_x, words), axis=1)

                curr_x = curr_x.T
                curr_y = curr_y.T
                y_hat = self.forward(curr_x)

                # knowing that the loss is SoftmaxLogLikelihood
                self.layers[-1]._set_target(curr_y)
                self.backward(1)

                if int(0.1 * no_of_batches_train) == (k):
                    print("=", end="")
                    k = 0

                losses.append(self.SoftmaxLogLikelihood(curr_y, y_hat))

                if j == no_of_batches_train - 1:
                    loss = sum(losses) / len(losses)
                    print()
                    print("loss: {}....".format(loss), end=" ")

                if batch_size == 1:
                    N = train_data[0].shape[0]
                else:
                    N = curr_x.shape[-1]

                self.optimizer.update(self.layers, N)
                self.zeroing()
                j += 1

                ###
            if validation_data:
                y_hat_val = self.forward(x_valid.T)
                loss_val = self.SoftmaxLogLikelihood(y_valid.T, y_hat_val)
                print("val_loss: {}....".format(loss_val), end=" ")
            ###

    def predict(self, data):

        word_encodings = []
        for dic in data[:, 0]:
            word_encodings.append(self.transform_words(dic))
        words = np.array(word_encodings)
        data = data[:, 1:].astype(np.int8)
        data = np.concatenate((data, words), axis=1)

        y_hat = self.forward(data.T)
        return y_hat.T

    def get_weights(self):
        params = []
        for layer in self.layers:
            params.append(layer.get_params())

        return params

    def save_weights(self, path):
        params = []
        for layer in self.layers:
            layer_param = [layer.W, layer.b]
            params.append(layer_param)

        # params = np.array(params)
        file = open(path, 'wb')

        # dump information to that file
        pickle.dump(params, file)

        # close the file
        file.close()

    def load_weights(self, path):
        file = open(path, 'rb')

        # dump information to that file
        params = pickle.load(file)

        # close the file
        file.close()

        i = 0
        for layer in self.layers:
            layer.set_params(params[i][0], params[i][1])
            i += 1

    def save_model(self, path):

        m = [self.layers, self.optimizer]

        file = open(path, 'wb')

        # dump information to that file
        pickle.dump(m, file)

        # close the file
        file.close()

    def load_model(self, path):
        file = open(path, 'rb')

        # dump information to that file
        m = pickle.load(file)

        # close the file
        file.close()

        self.layers = m[0]
        self.optimizer = m[2]