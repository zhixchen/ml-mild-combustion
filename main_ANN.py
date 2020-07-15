'''
MODULE: main_ANN.py
​
@Author:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano
​
@Contacts:
    giuseppe.dalessio@ulb.ac.be
​
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''

import ANN as neural
from utilities import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import time
import os


def Learning():
    start_time = time.time()
    file_options = {
    "path_to_file"              : "./model_zc/",
    "input_file_name"           : "X_scaled_zc.npy",
    "output_file_name"          : "target_scaled_zc.npy",
    # Optional additional input matrix to pass through the trained net for a second test:
    "test_file_name"            : "X_scaled_zc.npy",
    }

    training_options = {
    "center"                    : True,
    "centering_method"          : "mean",
    "scale"                     : True,
    "scaling_method"            : "auto",

    "neurons_per_layer"         : [256, 512],
    "batch_size"                : 256,
    "number_of_epochs"          : 1000,

    "activation_function"       : "leaky_relu",
    "alpha_LR"                  : 0.01,
    "activation_output"         : "softmax",

    "batchNormalization"        : True,
    "dropout"                   : 0,
    "patience"                  : 10,        
    "loss_function"             : "binary_crossentropy",
    "monitor"                   : "val_loss",
    "learning_rate"             : 0.0001,
     }

    X = np.load(file_options["path_to_file"] + file_options["input_file_name"])
    Y = np.load(file_options["path_to_file"] + file_options["output_file_name"])
    Z = np.load(file_options["path_to_file"] + file_options["test_file_name"])

    # outlier removal section
    input_index = np.arange(X.shape[0])
    outlier_index = np.zeros(X.shape[0], dtype=bool)

    print("Original training dimensions: {}".format(X.shape))
    print("Original test dimensions: {}".format(Y.shape))

    X_noOUT, ___, mask = outlier_removal_leverage(X, 2, training_options["centering_method"], training_options["scaling_method"])
    Y_noOUT = np.delete(Y, mask, axis=0)
    input_index_noOUT = np.delete(input_index, mask)
    outlier_index[mask] = True

    print("Training dimensions after first outlier removal: {}".format(X_noOUT.shape))
    print("Test dimensions after first outlier removal: {}".format(Y_noOUT.shape))

    X_noOUT2, ___, mask2 = outlier_removal_orthogonal(X_noOUT, 2, training_options["centering_method"], training_options["scaling_method"])
    Y_noOUT2 = np.delete(Y_noOUT, mask2, axis=0)
    outlier_index[input_index_noOUT[mask2]] = True

    print("Training dimensions after second outlier removal: {}".format(X_noOUT2.shape))
    print("Test dimensions after second outlier removal: {}".format(Y_noOUT2.shape))

    model = neural.regressor(X_noOUT2, Y_noOUT2, training_options, Z)
    predicted_Y_noOUT2 = model.fit_network()
    predictedTest, trueTest = model.predict()

    # Test the net for an additional input matrix (Z):
    predicted_Z = model.predict_new_matrix()

    print("---Completed in %s seconds ---" % (time.time() - start_time))
    np.save('predictions_zc_noOUT', predicted_Y_noOUT2)
    np.save('predictions_zc_newInput', predicted_Z)

    # write a txt file with final results from the training
    f = open('history_final.txt', 'w+')
    f.write('completed in {:.1f} seconds \n'.format(time.time() - start_time))
    f.write('loss = {} \n'.format(model.model.history.history['loss'][-1]))
    f.write('val_loss = {} \n'.format(model.model.history.history['val_loss'][-1]))
    f.write('mae = {} \n'.format(model.model.history.history['mae'][-1]))
    f.write('mse = {} \n'.format(model.model.history.history['mse'][-1]))
    f.write('val_mse = {}'.format(model.model.history.history['val_mse'][-1]))
    f.close()

    # save the outlier indices
    np.save('outlier_index', outlier_index)

    # plotting functions

    # scatter plot
    fig = plt.figure()
    plt.axes(aspect='equal')
    plt.scatter(Y_noOUT2.flatten(), predicted_Y_noOUT2.flatten()
                , s=2, edgecolors='black', linewidths=0.1)
    plt.xlabel('Y_zc')
    plt.ylabel('Y_pred')
    lims = [np.min(predicted_Y_noOUT2), np.max(predicted_Y_noOUT2)]
    lims2 = [np.min(Y), np.max(Y)]
    _ = plt.plot(lims2, lims2, 'r')
    plt.savefig('parity_plot.png')
    plt.show()
    fig.tight_layout()
    #Changing the directory from Results-ANN to working directory
    os.chdir("../../.")

def preprocessing_zc(X_zc,target_zc,loc):
    [n_elements, n_variables] = X_zc.shape
    scal_fact = np.zeros((n_variables,1))
    x_bar = np.zeros((n_variables,1))
    X_tilde = np.zeros((n_elements,n_variables))
    for i in range(n_variables):
        x_bar[i] = np.mean(X_zc[:,i])
    for i in range(n_variables):
        scal_fact[i] = np.std(X_zc[:,i])
        
    # scale the X matrix to get the X_tilde matrix to give in input to the NN
    for i in range(n_variables):
          X_tilde[:,i] = (X_zc[:,i] - x_bar[i])/(scal_fact[i]+1e-16)
    d = np.zeros((n_elements,1));
    scaled_target = np.zeros(target_zc.shape)
    for i in range(n_elements):
         d[i] = 1
         scaled_target[i,:] = target_zc[i,:]/d[i]

    file_to_open = loc + "X_scaling_factors_zc"
    np.savez(file_to_open,x_bar=x_bar
             ,scal_fact=scal_fact)
    file_to_open = loc + "X_scaled_zc"
    np.save(file_to_open,X_tilde)
    file_to_open = loc + "target_scaled_zc"
    np.save(file_to_open,scaled_target)
