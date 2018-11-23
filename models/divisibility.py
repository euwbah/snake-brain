"""
Network model for divisibility testing - test if one input is perfectly divisible by the other.

"""
from random import random, randint, choice
from typing import Optional

from architectures import create_ELU_model_logistic_output
from data import Data
from network import Network, train_one_epoch, load_weights


def generate_data_divisible_check(size: int, min: float = -1, max: float = 1, max_multiple: int = 100,
                                  bidirectional: bool = True) -> Data:
    """

    Generates sample data of the following two categories in equal proportion:
        1. [x, y] -> [1] where x is a positive integer multiple of y
        2. [a, b] -> [0] where a b are random numbers of which abs(a - round_multiple(a, b)) is at least 5% of b.

        where a, b, x, y are limited to the range from `min` to `max`.

    :param size:
                The number of datasets to create.
    :param min:
                Minimum values of the input dataset
    :param max:
                Maximum values of the input dataset
    :param max_multiple:
                The highest value of abs(x / y) generated
    :param bidirectional:
                If true, Either x or y can be perfectly divisible by the other to give a [1] output.
                Otherwise, only x can be perfectly divisible by y and not the other way around.
    :return: A list of (input vector, ground truth vector) tuples
    """

    data = []
    generate_correct = True

    for i in range(0, size):
        if generate_correct:
            x = random() * (max - min) + min
            y = x / randint(1, max_multiple) * choice([-1, 1])

            if bidirectional:
                [x, y] = choice([[x, y], [y, x]])

            data.append(([x, y], [1]))
        else:
            while True:
                a = random() * (max - min) + min
                b = random() * (max - min) + min

                # Make sure values of a and b are not within a certain range of being
                # the correct answer relative to the value of b.
                if abs(a - (b * round(a / b))) / b < 0.01:
                    continue

                data.append(([a, b], [0]))
                break

        generate_correct = not generate_correct

    return data


def train_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: int, stop_after_epoch: int,
                save_every_n_epochs: int = 1,
                training_min: int = -1, training_max: int = 1, training_max_mult: int = 100,
                val_min: int = -2, val_max: int = 2, val_max_mult: int = 200,
                bidirectional: bool = True,
                training_data_size: int = 800, val_data_size: int = 200,
                training_iterations_per_epoch: int = 80, val_iterations_per_epoch: int = 40,
                step_size: float = 0.0005, momentum: float = 0.1, decay: float = 0.0005):
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

    :param training_max_mult:
            Maximum integer value of c in the training data where x = c * y & {x, y} are the two input nodes.

    :param val_min:  Minimum value of the training data

    :param val_max: Maximum value of the training data

    :param val_max_mult:
            Maximum integer value of c in the validation data where x = c * y & {x, y} are the two input nodes.

    :param bidirectional:
            If true, Either x or y can be perfectly divisible by the other to give a [1] output.
            Otherwise, only x can be perfectly divisible by y and not the other way around.
            This applies for both training and validation data.

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
    training_data = generate_data_divisible_check(training_data_size, training_min, training_max, training_max_mult, bidirectional)
    val_data = generate_data_divisible_check(val_data_size, val_min, val_max, val_max_mult, bidirectional)
    network = Network()
    create_ELU_model_logistic_output(network, 2, 1, hidden_layers, nodes_per_layer, elu_alpha)
    # Load most recently trained epoch
    # If no trained weights exist, start training from epoch 1
    prev_epoch = load_weights(network, model_name) or 0

    if prev_epoch >= stop_after_epoch:
        print(f"WARN: Not training anything! Epoch {stop_after_epoch} was already trained.")

    curr_epoch = prev_epoch + 1

    while curr_epoch <= stop_after_epoch:
        print(f"Training new epoch: {curr_epoch} / {stop_after_epoch}")
        train_one_epoch(network, training_data, val_data, model_name, curr_epoch,
                        step_size, momentum, decay,
                        training_iterations_per_epoch, val_iterations_per_epoch,
                        save_weights=curr_epoch % save_every_n_epochs == 0)
        curr_epoch += 1


def test_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: int,
               epoch_number: Optional[int] = None):

    network = Network()
    create_ELU_model_logistic_output(network, 2, 1, hidden_layers, nodes_per_layer, elu_alpha)

    epoch = load_weights(network, model_name, epoch_number)

    print(f"Loaded epoch {epoch} of {model_name}. Ready to test:")

    while True:
        a = float(input('Enter a: '))
        b = float(input('Enter b: '))

        [output] = network.predict([a, b])

        if output >= 0.5:
            print(f"Network is {round(output * 100, 2)}% sure that {a} is perfectly divisible by {b}")
        else:
            print(f"Network is {round(100 - output * 100, 2)}% sure that {a} is NOT perfectly divisible by {b}")