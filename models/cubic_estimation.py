"""
Network model for interpolating & extrapolating data points from a quadratic equation.

"""
from random import random
from typing import Optional

from architectures import create_elu_logistic_layers_model_linear_output
from data import Data
from network import Network, load_weights, train_one_epoch


def generate_data_cubic_equation(size: int, x_min: float, x_max: float,
                                 a: float, b: float, c: float, d: float) -> Data:
    """

    Generates sample data fulfilling f(x) = ax^3 - bx^2 + cx + d where x is the activation of a single input node
    and f(x) is the expected output.

    :param size:
    :param x_min: minimum value of x
    :param x_max: maximum value of x
    :param a: Coefficient of x^3 term
    :param b: Coefficient of x^2 term
    :param c: Coefficient of x^1 term
    :param d: Coefficient of constant term
    :return:
    """

    data = []

    for i in range(0, size):
        x = random() * (x_max - x_min) + x_min
        data.append(([x], [a * x ** 3 + b * x ** 2 + c * x + d]))

    return data


def train_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: float,
                stop_after_epoch: int, save_every_n_epochs: int = 1,
                a: float = 1, b: float = 0, c: float = 0, d: float = 0,
                training_min: float = -5, training_max: float = 3, val_min: float = -3, val_max: float = 5,
                training_data_size: int = 800, val_data_size: int = 200,
                training_iterations_per_epoch: int = 80, val_iterations_per_epoch: int = 40,
                step_size: float = 0.0005, momentum: float = 0.1, decay: float = 0.00005,
                log_level: int = 0, pause_after_iter: Optional[float] = None):
    """
    Scaffolds the network model, loads previous weights (if any), trains, validates and saves new weights & losses

    :param model_name: Used to save & load the model

    :param hidden_layers: The number of layers.

    :param nodes_per_layer: The number of fully connected nodes per layer

    :param elu_alpha: The alpha constant used in the ELU activation function for the ELU nodes.

    :param stop_after_epoch: Stop training after training this epoch number

    :param save_every_n_epochs: Only save every nth epoch (where epoch `mod` n == 0 is satisfied)

    :param training_min: Minimum value of training data

    :param training_max: Maximum value of training data

    :param val_min:  Minimum value of the training data

    :param val_max: Maximum value of the training data

    :param training_data_size: Number of training data samples to generate.

    :param val_data_size: Number of validation data samples to generate.

    :param training_iterations_per_epoch: Number of samples to train on per epoch.

    :param val_iterations_per_epoch: Number of samples to perform validation on per epoch.

    :param step_size:
            Multiple of the d(loss)/d(weight) derivative to nudge the weight by per iteration.

    :param momentum:
            How much of the previous weight update is applied to the current weight update.
            (1 --> 100%, 0 --> 0%)
            Using momentum prevents the network from getting stuck in local minimas.
            (Imagine a ball rolling down the loss function curve. If there is a pothole in the curve, momentum
            may allow the ball to not be stuck.)

    :param decay:
            How much of the previous weight to subtract the current weight by.
            (1 --> 100%, 0 --> 0%)
            This will make the weight gravitate towards zero, so that the weights won't explode to NaN.
    :return:
    """
    print('Training cubic estimation model')

    training_data = generate_data_cubic_equation(training_data_size, training_min, training_max, a, b, c, d)
    val_data = generate_data_cubic_equation(val_data_size, val_min, val_max, a, b, c, d)
    network = Network()
    create_elu_logistic_layers_model_linear_output(network, 1, 1, hidden_layers, nodes_per_layer, elu_alpha)
    prev_epoch = load_weights(network, model_name) or 0

    if prev_epoch >= stop_after_epoch:
        print(f"WARN: Not training anything! Epoch {stop_after_epoch} was already trained.")

    curr_epoch = prev_epoch + 1

    while curr_epoch <= stop_after_epoch:
        print(f"\nTraining new epoch: {curr_epoch} / {stop_after_epoch}")
        train_one_epoch(network, training_data, val_data, model_name, curr_epoch,
                        step_size, momentum, decay,
                        training_iterations_per_epoch, val_iterations_per_epoch,
                        save_weights=curr_epoch % save_every_n_epochs == 0,
                        log_level=log_level,
                        pause_after_iter=pause_after_iter)
        curr_epoch += 1


def test_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: float,
               epoch_number: Optional[int] = None):
    network = Network()
    create_elu_logistic_layers_model_linear_output(network, 1, 1, hidden_layers, nodes_per_layer, elu_alpha)

    epoch = load_weights(network, model_name, epoch_number)

    print(f"Loaded epoch {epoch} of {model_name}. Ready to test:")

    while True:
        x = float(input('Enter x: '))

        [output] = network.predict([x])

        print(f'>> f(x) = {output}')
