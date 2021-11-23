# Plotting
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# Parameters Managment
import yaml

# Loading dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Gradient Decent
from tensorflow.keras.optimizers import SGD

# Report Generation
from sklearn.metrics import classification_report

# Model
from keras.models import Sequential
from keras.layers.core import Dense


class mnist_ff_net:
    def __init__(self):
        print("Initialization")
        param_file = open("../params/config.yaml", 'r')
        network_params = yaml.safe_load(param_file)
        
        print(network_params)
        self.dataset_name = network_params["dataset_name"]
        self.test_dataset_size = network_params["test_dataset_size"]
        
        # Network Params
        self.num_neurons_input_layer = network_params["num_neurons_input_layer"]
        self.num_neurons_second_layer = network_params["num_neurons_second_layer"]
        self.num_neurons_third_layer = network_params["num_neurons_third_layer"]
        self.num_neurons_output_layer = network_params["num_neurons_output_layer"]
        self.hiden_layers_act_func = network_params["hiden_layers_act_func"]
        self.output_layer_act_func = network_params["output_layer_act_func"]
        self.test_dataset_size = network_params["test_dataset_size"]
        self.sgd_learning_rate = network_params["sgd_learning_rate"]
        self.training_bartch_size = network_params["training_bartch_size"]
        self.testing_batch_size = network_params["testing_batch_size"]
        self.num_epochs = network_params["num_epochs"]
        self.loss_func = network_params["loss_func"]

    def load_mnist_dataset(self):
        print("Loading mnist dataset")
        self.dataset = datasets.fetch_openml(self.dataset_name)
        return self.dataset
    
    def process_dataset(self):
        print("Processing dataset")
        self.data = self.dataset.data.astype('float') / 255.0
        (self.mnist_training_data, self.mnist_viladation_data, self.mnist_training_label, self.mnist_validation_label) = train_test_split(self.data, self.dataset.target, test_size=self.test_dataset_size)
        self.binery_label = LabelBinarizer()
        self.mnist_training_label = self.binery_label.fit_transform(self.mnist_training_label)
        self.mnist_validation_label = self.binery_label.fit_transform(self.mnist_validation_label)
        return (self.mnist_training_data, self.mnist_viladation_data, self.mnist_training_label, self.mnist_validation_label, self.binery_label)
    
    def create_ff_model(self):
        print("Creating the ff_model")
        self.ff_model = Sequential()
        self.ff_model.add(Dense(self.num_neurons_second_layer, input_shape=(self.num_neurons_input_layer,), activation=self.hiden_layers_act_func))
        self.ff_model.add(Dense(self.num_neurons_third_layer, activation=self.hiden_layers_act_func))
        self.ff_model.add(Dense(self.num_neurons_output_layer, activation=self.output_layer_act_func))
        return self.ff_model

    def train(self):
        print("Training the ff_model")
        sgd = SGD(self.sgd_learning_rate)
        self.ff_model.compile(loss=self.loss_func, optimizer=sgd, metrics=['accuracy'])
        self.H = self.ff_model.fit(self.mnist_training_data, self.mnist_training_label, validation_data=(self.mnist_viladation_data, self.mnist_validation_label), epochs=self.num_epochs, batch_size=self.training_bartch_size)

    
    def test(self):
        print("Testing the ff_model")
        self.predictions = self.ff_model.predict(self.mnist_viladation_data, batch_size=self.testing_batch_size)
        print(classification_report(self.mnist_validation_label.argmax(axis=1), self.predictions.argmax(axis=1),
                            target_names=[str(x) for x in self.binery_label.classes_]))
    
    def plot_results(self):
        print("printing results...")
        epoch_vector = np.arange(0, self.num_epochs)

        plt.figure()
        # Plotting Training LOSS
        plt.plot(epoch_vector, self.H.history['loss'], label='training_loss_value', LineWidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Validation Loss
        plt.figure()
        plt.plot(epoch_vector, self.H.history['val_loss'], label='validation_loss_value', LineWidth=2)
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Training Accuracy
        plt.figure()
        plt.plot(epoch_vector, self.H.history['accuracy'], label='training_accuracy', LineWidth=2)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Trining Accuracy')
        plt.legend()
        plt.grid()
        # plt.show()

        # Plotting Validation Accuracy
        plt.figure()
        plt.plot(epoch_vector, self.H.history['val_accuracy'], label='validation_accuracy', LineWidth=2)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()
