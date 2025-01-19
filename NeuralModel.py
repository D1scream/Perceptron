import os
import tensorflow as tf
class NeuralModelForPP:
    def __init__(self, activation='sigmoid',loss_function_name="MSE",learning_rate=0.1,X_unscaled=None,epochs=500):
        self.activation = activation
        self.epochs = epochs
        # Не нормализованные данные для вывода
        self.X_unscaled=X_unscaled
        self.learning_rate=learning_rate
        self.loss_func_name=loss_function_name
        # Словарь функций активации
        self.activation_functions = {
            'sigmoid': lambda x: 1 / (1 + tf.exp(-x)),
            'linear': lambda x: x,
            'relu': lambda x: tf.maximum(0.0, x)
        }
        if self.activation not in self.activation_functions:
            raise ValueError("wrong activation")
        # Словарь функций потерь
        self.loss_functions = {
            "MSE": self.mean_squared_error,
            "MAE": self.mean_absolute_error,
            "BinaryCrossEntropy": self.binary_cross_entropy,
            "Huber": self.huber_loss,
            "LogCosh": self.log_cosh_loss
        }
        if self.loss_func_name not in self.loss_functions:
            raise ValueError(f"Неизвестная функция потерь: {self.loss_func_name}")
        
        self.init_model_structure()
        
    def init_model_structure(self):
        input_size = 3
        hidden_layer_size = 8
        
        self.w1 = tf.Variable(tf.random.normal([input_size, hidden_layer_size], stddev=0.1), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([hidden_layer_size]), dtype=tf.float32)
        
        self.w2 = tf.Variable(tf.random.normal([hidden_layer_size, hidden_layer_size], stddev=0.1), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([hidden_layer_size]), dtype=tf.float32)
        
        self.w_output = tf.Variable(tf.random.normal([hidden_layer_size, 1], stddev=0.1), dtype=tf.float32)
        self.b_output = tf.Variable(tf.zeros([1]), dtype=tf.float32)

    def forward_pass(self, X):
        hidden_layer1 = self.activation_function(tf.matmul(X, self.w1) + self.b1)
        hidden_layer2 = self.activation_function(tf.matmul(hidden_layer1, self.w2) + self.b2)
        output_layer = self.activation_function(tf.matmul(hidden_layer2, self.w_output) + self.b_output)

        #output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer2, self.w_output) + self.b_output)))
        return output_layer
     
    def train(self,X,Y):
        # Обучение
        optimizer = tf.optimizers.Adam(self.learning_rate)
        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                predictions = self.forward_pass(X)
                loss = self.loss_func(Y, predictions)

            gradients = tape.gradient(loss, [self.w1, self.b1, self.w2, self.b2, self.w_output, self.b_output])
            optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.w2, self.b2, self.w_output, self.b_output]))

            if epoch % 100 == 0:
                print(f"{self.activation} Model, Epoch {epoch}, Loss: {loss.numpy():.4f}")

        self.evaluate(X,Y,self.X_unscaled)

    def loss_func(self, y_true, y_pred):
        return self.loss_functions[self.loss_func_name](y_true, y_pred)

    # Различные функции активации
    def mean_squared_error(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    def binary_cross_entropy(self, y_true, y_pred):
        return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-10) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-10))
    
    def mean_absolute_error(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))
    
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        condition = tf.abs(error) <= delta
        small_error_loss = 0.5 * tf.square(error)
        large_error_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.reduce_mean(tf.where(condition, small_error_loss, large_error_loss))
    
    def log_cosh_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.log(tf.cosh(y_true - y_pred)))
    
    def evaluate(self,X,Y,unscaled_data):
        predictions = self.forward_pass(X)

        print(f"\nResults with {self.activation} activation:")
        for i, prediction in enumerate(predictions.numpy()):
            print(f"Input: {[f"{x:.2f}" for x in unscaled_data[i]]}, Predicted: {prediction[0]:.2f}, Expected: {Y[i][0]:.2f}")

        print("-" * 30)

    def activation_function(self, x):
        return self.activation_functions[self.activation](x)

    def predict(self, X):
        predictions = self.forward_pass(X)
        return predictions.numpy()