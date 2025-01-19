import tensorflow as tf
import numpy as np
import os


class XORNeuralModel:
    def __init__(self, activation='sigmoid',learning_rate = 0.1,epochs=500):
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Словарь функций активации
        self.activation_functions = {
            'sigmoid': lambda x: 1 / (1 + tf.exp(-x)),
            'linear': lambda x: x,
            'relu': lambda x: tf.maximum(0.0, x)
        }
        if self.activation not in self.activation_functions:
            raise ValueError("wrong activation")
        
        hidden_layer_count = 16
        self.w1 = tf.Variable(tf.random.normal([2, hidden_layer_count], stddev=0.1), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([hidden_layer_count]), dtype=tf.float32)
        self.w_output = tf.Variable(tf.random.normal([hidden_layer_count, 1], stddev=0.1), dtype=tf.float32)
        self.b_output = tf.Variable(tf.zeros([1]), dtype=tf.float32)
        #Input Layer (2 neurons)  --> Hidden Layer ({hidden_layer_count} neurons)  --> Output Layer (1 neuron)

    def forward_pass(self, X):
        hidden_layer = self.activation_function(tf.matmul(X, self.w1) + self.b1)
        output_layer = 1 / (1 + tf.exp(-(tf.matmul(hidden_layer, self.w_output) + self.b_output)))  # Сигмоид в выходном слое
        return output_layer
    
    def binary_cross_entropy(self, y_true, y_pred):
        return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-10) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-10))
     
    def train(self):

        # Логирование
        log_dir = "logs/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        tensorboard_writer = tf.summary.create_file_writer(log_dir + self.activation)
        tf.summary.trace_on(graph=True)

        # Обучение
        optimizer = tf.optimizers.Adam(self.learning_rate)
        for epoch in range(self.epochs):
            with tf.GradientTape() as tape:
                predictions = self.forward_pass(X)
                loss = self.binary_cross_entropy(Y, predictions)
                
            gradients = tape.gradient(loss, [self.w1, self.b1, self.w_output, self.b_output])
            optimizer.apply_gradients(zip(gradients, [self.w1, self.b1, self.w_output, self.b_output]))
            
            if epoch % 100 == 0:
                print(f"{self.activation} Model, Epoch {epoch}, Loss: {loss.numpy():.4f}")

            # Логирование
            with tensorboard_writer.as_default():
                tf.summary.scalar("Loss", loss, step=epoch)
            
        with tensorboard_writer.as_default():
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=log_dir)
            
        self.evaluate()
    
    def evaluate(self):
        predictions = self.forward_pass(X)
        print(f"\nResults with {self.activation} activation:")
        for i, prediction in enumerate(predictions.numpy()):
            print(f"Input: {X[i]}, Predicted: {prediction[0]:.2f}, Expected: {Y[i][0]}")
        print("-" * 30)

    def activation_function(self, x):
        return self.activation_functions[self.activation](x)

    def predict(self, X):
        predictions = self.forward_pass(X)
        return predictions.numpy()
    
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

models = [XORNeuralModel('sigmoid'), XORNeuralModel('linear'), XORNeuralModel('relu')]

for model in models:
    model.train()