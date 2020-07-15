'''
MODULE: ANN.py

@Author:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano

@Contacts:
    giuseppe.dalessio@ulb.ac.be

@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
import os
import os.path
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from utilities import *


class Architecture:
    def __init__(self, X, Y, *dictionary):
        self.X = X
        self.Y = Y

        self._activation = 'relu'
        self._batch_size = 64
        self._n_epochs = 1000
        self._getNeurons = [1, 1]
        self._dropout = 0
        self._patience = 10
        self._batchNormalization = False
        self._alpha = 0.0001

        self.save_txt = True

        if dictionary:
            settings = dictionary[0]

            self._center = settings["center"]
            self._centering = settings["centering_method"]
            self._scale = settings["scale"]
            self._scaling = settings["scaling_method"]
            self._activation = settings["activation_function"]
            self._batch_size = settings["batch_size"]
            self._n_epochs = settings["number_of_epochs"]
            self._getNeurons = settings["neurons_per_layer"]
            self._dropout = settings["dropout"]
            self._patience = settings["patience"]
            self._batchNormalization = settings["batchNormalization"]
            self._alpha = settings["alpha_LR"]

            if settings["activation_function"] == 'leaky_relu':
                LR = LeakyReLU(alpha=self._alpha)
                LR.__name__ = 'relu'
                self._activation = LR

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, new_activation):
        if new_activation == 'leaky_relu':
            LR = LeakyReLU(alpha=self._alpha)
            LR.__name__ = 'relu'
            self._activation = LR
        else:
            self._activation = new_activation

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    @accepts(object, int)
    def batch_size(self, new_batchsize):
        self._batch_size = new_batchsize

        if self._batch_size <= 0:
            raise Exception("The batch size must be a positive integer. Exiting..")
            exit()

    @property
    def n_epochs(self):
        return self._n_epochs

    @n_epochs.setter
    @accepts(object, int)
    def n_epochs(self, new_epochs):
        self._n_epochs = new_epochs

        if self._n_epochs <= 0:
            raise Exception("The number of epochs must be a positive integer. Exiting..")
            exit()

    @property
    def dropout(self):
        return self._dropout

    @dropout.setter
    def dropout(self, new_value):
        self._dropout = new_value

        if self._dropout < 0:
            raise Exception("The dropout percentage must be a positive integer. Exiting..")
            exit()
        elif self._dropout >= 1:
            raise Exception("The dropout percentage must be lower than 1. Exiting..")
            exit()

    @property
    def patience(self):
        return self._patience

    @patience.setter
    @accepts(object, int)
    def patience(self, new_value):
        self._patience = new_value

        if self._patience < 0:
            raise Exception("Patience for early stopping must be a positive integer. Exiting..")
            exit()

    @property
    def batchNormalization(self):
        return self._batchNormalization

    @batchNormalization.setter
    @accepts(object, bool)
    def batchNormalization(self, new_bool):
        self._batchNormalization = new_bool

    @property
    def getNeurons(self):
        return self._getNeurons

    @getNeurons.setter
    def getNeurons(self, new_vector):
        self._getNeurons = new_vector

    @property
    def alpha_leaky(self):
        return self._alpha

    @alpha_leaky.setter
    def alpha_leaky(self, new_value):
        self._alpha = new_value


    @staticmethod
    def set_environment():
        '''
        This function creates a new folder where all the produced files
        will be saved.
        '''
        import datetime
        import sys
        import os

        now = datetime.datetime.now()
        #newDirName = "Train MLP class - " + now.strftime("%Y_%m_%d-%H%M")
        newDirName = "./model_zc/Results-ANN"
        if os.path.exists(newDirName):
            os.chdir(newDirName)
        else:
            os.mkdir(newDirName)
            os.chdir(newDirName)
        
        # try:
        #     os.mkdir(newDirName)
        #     os.chdir(newDirName)
        # except FileExistsError:
        #     pass

    @staticmethod
    def write_recap_text(neuro_number, number_batches, activation_specification):
        '''
        This function writes a txt with all the hyperparameters
        recaped, to not forget the settings if several trainings are
        launched all together.
        '''
        text_file = open("recap_training.txt", "wt")
        neurons_number = text_file.write("The number of neurons in the implemented architecture is equal to: {} \n".format(neuro_number))
        batches_number = text_file.write("The batch size is equal to: {} \n".format(number_batches))
        activation_used = text_file.write("The activation function which was used was: "+ str(activation_specification) + ". \n")
        text_file.close()


    @staticmethod
    def preprocess_training(X, centering_decision, scaling_decision, centering_method, scaling_method):

        if centering_decision and scaling_decision:
            mu, X_ = center(X, centering_method, True)
            sigma, X_tilde = scale(X_, scaling_method, True)
        elif centering_decision and not scaling_decision:
            mu, X_tilde = center(X, centering_method, True)
        elif scaling_decision and not centering_decision:
            sigma, X_tilde = scale(X, scaling_method, True)
        else:
            X_tilde = X

        return X_tilde

   
class regressor(Architecture):
    def __init__(self, X, Y, *dictionary):
        self.X = X
        self.Y = Y
        
        self._activation_output = 'linear'
        self._loss_function = 'mean_squared_error'
        self._monitor_early_stop = 'mean_squared_error'
        self._learningRate = 0.0001

        self.testProcess = False

        super().__init__(self.X, self.Y, *dictionary)

        if dictionary:
            settings = dictionary[0]

            self._center = settings["center"]
            self._centering = settings["centering_method"]
            self._scale = settings["scale"]
            self._scaling = settings["scaling_method"]
            self._activation = settings["activation_function"]
            self._batch_size = settings["batch_size"]
            self._n_epochs = settings["number_of_epochs"]
            self._getNeurons = settings["neurons_per_layer"]
            self._dropout = settings["dropout"]
            self._patience = settings["patience"]
            self._alpha = settings["alpha_LR"]
            self._activation_output = settings["activation_output"]
            self._loss_function = settings["loss_function"]
            self._monitor_early_stop = settings["monitor"]
            self._learningRate = settings["learning_rate"]

            if settings["activation_function"] == 'leaky_relu':
                LR = LeakyReLU(alpha=self._alpha)
                LR.__name__ = 'relu'
                self._activation = LR
        
  
        if len(dictionary) > 1:
            self.Z = dictionary[1]
            self.testProcess = True


    @property
    def activationOutput(self):
        return self._activation_output

    @activationOutput.setter
    def activationOutput(self, new_string):
        self._activation_output = new_string


    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, new_value):
        self._loss_function = new_value


    @property
    def monitor_early_stop(self):
        return self._monitor_early_stop

    @monitor_early_stop.setter
    def monitor_early_stop(self, new_value):
        self._monitor_early_stop = new_value


    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, new_value):
        self._learningRate = new_value


    def __set_hard_parameters(self):
        '''
        --- PRIVATE ---
        This function sets all the parameters for the neural network which should not
        be changed during the tuning.
        '''
        self.__path = os.getcwd()
        self.__optimizer = Adam(lr=self._learningRate)
        self.__metrics = ['mse', 'mae']


    def fit_network(self):
        input_dimension = self.X.shape[1]
        output_dimension = self.Y.shape[1]

        Architecture.set_environment()
        Architecture.write_recap_text(self._getNeurons, self._batch_size, self._activation)
        self.__set_hard_parameters()

        print("Preprocessing training matrix..")
        self.X_tilde = self.preprocess_training(self.X, self._center, self._scale, self._centering, self._scaling)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_tilde, self.Y, test_size=0.2)

        if self.testProcess:
            #if there is a new matrix for the test (Z), the latter has to be centered and scaleed 
            #with the training matrix preprocessing factors:
            mu = center(self.X, self._centering, return_centered_matrix=False)
            sigma = scale(self.X, self._scaling, return_scaled_matrix=False)

            self.Z_tilde = center_scale(self.Z, mu, sigma)
        
        self._layers = len(self._getNeurons)

        counter = 0

        self.model = Sequential()
        self.model.add(Dense(self._getNeurons[counter], input_dim=input_dimension, kernel_initializer='normal', activation=self._activation))
        counter += 1
        if self._dropout != 0:
            from tensorflow.python.keras.layers import Dropout
            self.model.add(Dropout(self._dropout))
            print("Dropping out some neurons...")
        if self._batchNormalization:
            print("Normalization added!")
            self.model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
        while counter < self._layers:
            self.model.add(Dense(self._getNeurons[counter], activation=self._activation))
            if self._dropout != 0:
                self.model.add(Dropout(self._dropout))
            if self._batchNormalization:
                self.model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
                counter += 1
        self.model.add(Dense(output_dimension, activation=self._activation_output))
        self.model.summary()

        earlyStopping = EarlyStopping(monitor=self._monitor_early_stop, patience=self._patience, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(filepath=self.__path+ '/best_weights2c.h5', verbose=1, save_best_only=True, monitor=self._monitor_early_stop, mode='min')
        self.model.compile(loss=self._loss_function, optimizer=self.__optimizer, metrics=[self.__metrics])
        history = self.model.fit(self.X_train, self.y_train, batch_size=self._batch_size, epochs=self._n_epochs, verbose=1, validation_data=(self.X_test, self.y_test), callbacks=[earlyStopping, mcp_save])

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.savefig('loss_history.eps')
        plt.show()

        self.model.load_weights(self.__path + '/best_weights2c.h5')

        counter_saver = 0

        test = self.model.predict(self.X_tilde)

        if self.save_txt:

            while counter_saver < self._layers:
                layer_weights = self.model.layers[counter_saver].get_weights()[0]
                layer_biases = self.model.layers[counter_saver].get_weights()[1]
                name_weights = "Weights_HL{}.txt".format(counter_saver)
                name_biases = "Biases_HL{}.txt".format(counter_saver)
                np.savetxt(name_weights, layer_weights)
                np.savetxt(name_biases, layer_biases)

                counter_saver += 1

        return test

    def predict(self):

        prediction_test = self.model.predict(self.X_test)

        return prediction_test, self.y_test

    def predict_new_matrix(self):
        if self.testProcess:
            prediction_new = self.model.predict(self.Z_tilde)

            return prediction_new
        else:
            raise Exception("Test input matrix not given. Cannot predict: exiting without error..")
            pass


def main():

    file_options = {
        "path_to_file"              : "/Users/giuseppedalessio/Dropbox/GitHub/data",
        "input_file_name"           : "X_zc.csv",
        "output_file_name"          : "Y_zc.csv"
    }

    training_options = {
        "center"                    : True,
        "centering_method"          : "MEAN",
        "scale"                     : True, 
        "scaling_method"            : "AUTO",
        "batch_size"                : 64,
        "activation_function"       : "leaky_relu",
        "number_of_epochs"          : 200,
        "neurons_per_layer"         : [10,10],
        "dropout"                   : 0,
        "patience"                  : 10,
    }

  
    X = readCSV(file_options["path_to_file"], file_options["input_file_name"])
    Y = readCSV(file_options["path_to_file"], file_options["output_file_name"])

    model = regressor(X,Y, training_options)
    '''
    model.activation_function = training_options["activation_function"]
    model.n_epochs = training_options["number_of_epochs"]
    model.batch_size = training_options["batch_size"]
    model.dropout = 0
    model.batchNormalization = True
    model.activationOutput = 'softmax'
    model.getNeurons = [256, 512]
    model.patience = 5
    '''
    yo = model.fit_network()
    predictedTest, trueTest = model.predict()


    a = plt.axes(aspect='equal')
    plt.scatter(trueTest.flatten(), predictedTest.flatten())
    plt.xlabel('Y_zc')
    plt.ylabel('Y_pred')
    lims = [np.min(trueTest), np.max(trueTest)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims, 'r')
    plt.show()


if __name__ == '__main__':
    main()
