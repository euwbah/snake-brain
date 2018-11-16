import os
import re
from typing import Tuple, List, Optional, Dict
from random import random, randint, choice, sample

from network import Network
import node

# Data type alias
Data = List[Tuple[List[float], List[float]]]

def generate_data(size: int, min: float = -1, max: float = 1, max_multiple: int = 100) -> Data:
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
    :return: A list of (input vector, ground truth vector) tuples
    """

    data = []
    generate_correct = True

    for i in range(0, size):
        if generate_correct:
            x = random() * (max - min) + min
            y = x / randint(1, max_multiple) * choice([-1, 1])
            data.append((choice([[x, y], [y, x]]), [1]))
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

def test_forward_propagation(network: Network, data: Data):

    i1 = node.ConstantNode("i1", data[0][0][0])
    i2 = node.ConstantNode("i2", data[0][0][1])

    h1_1 = node.ELUNode("h1_1", 1)
    h1_2 = node.ELUNode("h1_2", 1)
    i1.connect(h1_1, 1)
    i1.connect(h1_2, 0)
    i2.connect(h1_1, 0)
    i2.connect(h1_2, 1)

    h2_1 = node.ELUNode("h2_1", 1)
    h2_2 = node.ELUNode("h2_2", 1)
    h1_1.connect(h2_1, 1)
    h1_1.connect(h2_2, 0)
    h1_2.connect(h2_1, 0)
    h1_2.connect(h2_2, 1)

    o1 = node.SigmoidNode("o1")
    h2_1.connect(o1, 1)
    h2_2.connect(o1, 0)

    network.register_nodes([i1, i2], [o1])

    print(network.forward_propagate())
    print(network.evaluate_loss(data[0][1]))


def test_evaluate_dloss_dactivation(network: Network, data: Data):
    ground_truths = data[0][1]
    network.evaluate_gradients(ground_truths)

def create_network_model(network: Network, n_hidden_layers: int, n_nodes_per_hidden_layer: int, elu_alpha: int = 1):
    """
    Creates, connects, and registers the nodes on the network for the task of identifying if
    i1 is perfectly divisible by i2.

    NOTE: If the network model architecture is changed, the saved weights file from the previous model will not work.
          Ensure backups are made for this function.

    :param network: The Network object
    :param n_hidden_layers:
            The number of hidden layers ('columns' of nodes that are interconnected to one another,
            between the input and output nodes)
    :param n_nodes_per_hidden_layer:
            The number of interconnected nodes in each layer.
    :return:
    """

    # Make input nodes (constant value will be assigned in Network.assign_inputs)
    i1 = node.ConstantNode("i1", 0)
    i2 = node.ConstantNode("i2", 0)

    # Make the bias node (just a node that constantly outputs '1') so that
    # through weight multiplication, nodes that takes in a bias as input will
    # be able to learn a constant offset for its values.

    bias = node.ConstantNode("bias", 1)

    # Make "hidden layer" fully connected nodes with the ELU activation function.
    # Hidden layer naming convention: h<layer index from 1>_<node index from 1>

    prev_layer_nodes = [i1, i2]

    for l in range(1, n_hidden_layers + 1):
        curr_layer_nodes = []
        for n in range(1, n_nodes_per_hidden_layer + 1):

            # Each hidden layer node is an ELU
            new_node = node.ELUNode(f"h{l}_{n}", elu_alpha)

            # Give this node a bias
            bias.connect(new_node)

            # Connect each node of the previous layer to this node
            for p in prev_layer_nodes:
                p.connect(new_node)

            curr_layer_nodes.append(new_node)

        prev_layer_nodes = curr_layer_nodes

    # Make output node
    o1 = node.SigmoidNode("o1")
    for p in prev_layer_nodes:
        p.connect(o1)

    network.register_nodes([i1, i2], [o1])


def load_weights(network: Network, model_name: str, epoch: Optional[int] = None) -> Optional[int]:
    """
    Load weights from a trained epoch. Nothing happens if no previously trained model is found.

    :param network: The Network object
    :param model_name: Name of the previously trained model
    :param epoch: A particular epoch number to load. If none specified, loads the last one.
    :return: Returns the epoch number that was loaded, or None if none exists
    """

    model_dir = os.path.join("logs", model_name)
    os.makedirs(model_dir, exist_ok=True)
    weight_files: Dict[int, str] = {} # <epoch, file name>
    epoch_used = None

    for f in os.listdir(model_dir):
        match = re.fullmatch("""E(?P<epoch_number>[0-9]+)_weights\.txt""", f)
        if match is not None:
            epoch_number = int(match.group("epoch_number"))
            weight_files[epoch_number] = f

    if epoch is None:
        if len(weight_files) != 0:
            # take the highest epoch in weight_files
            epoch_used = sorted(weight_files)[-1]
            weights_file = weight_files[epoch_used]
        else:
            # Nothing to load, no previous model exists and no explicit epoch number to load
            return
    else:
        if epoch in weight_files:
            weights_file = weight_files[epoch]
            epoch_used = epoch
        else:
            raise IndexError(f"Epoch {epoch} does not exist for model '{model_name}'")

    weights_path = os.path.join(model_dir, weights_file)
    print(f"Loaded weights from {weights_path}")
    network.load_weights(weights_path)

    return epoch_used


def train_one_epoch(network: Network, training_data: Data, validation_data: Data,
                    step_size: float, model_name: str, epoch_number: int,
                    training_iterations: Optional[int] = None,
                    validation_iterations: Optional[int] = None,
                    verbose: bool = False, pause_after_iter: bool = False):
    """
    Train the network for 1 epoch.

    Usually this means training on every sample in the dataset once, but the number of samples
    to train on can be overridden by specifying it in the iterations_per_epoch parameter.

    :param network: The Network object

    :param training_data: Training data

    :param validation_data: Validation data

    :param epoch_number: The epoch number to name the saved weight file with.

    :param training_iterations:
            The number of training data samples to train on. Defaults to len(training_data).
            Set this if training_data is huge.

    :param validation_iterations:
            The number of validation data samples to use. Defaults to len(validation_data).
            Set this if validation_data is huge.

    :param step_size: The multiplier of the d(loss)/d(weight) derivative the weights are updated by.

    :param model_name: A name used to identify this model when saving the weight files.

    :param verbose: Set True to show additional verbose logs
    :return:
    """

    training_data_subset = training_data

    if training_iterations is None or training_iterations > len(training_data):
        training_iterations = len(training_data)

    # Even if no subset specified, still use sample to randomize the order of the training data.
    training_data_subset = sample(training_data_subset, training_iterations)

    validation_data_subset = validation_data

    if validation_iterations is None or validation_iterations > len(validation_data):
        validation_iterations = len(validation_data)

    # Even if no subset specified, still use sample to randomize the order of the validation data.
    validation_data_subset = sample(validation_data_subset, validation_iterations)

    training_sample_size = len(training_data_subset)
    avg_training_loss = 0
    for i, (inputs, ground_truths) in enumerate(training_data_subset):

        print(f"\n_____________________________\nTraining iteration {i + 1} / {training_sample_size}:")
        iter_loss, avg_dloss, max_dloss = network.train_iter(inputs, ground_truths, step_size, verbose)
        print(f"Iter loss: {iter_loss}\n"
              f"Avg d(loss)/d(weight): {avg_dloss}\n"
              f"Max d(loss)/d(weight): {max_dloss}")
        avg_training_loss += iter_loss / training_sample_size

        if pause_after_iter:
            input()

    validation_sample_size = len(validation_data_subset)
    avg_validation_loss = 0
    for i, (inputs, ground_truths) in enumerate(validation_data_subset):
        print(f"\n_____________________________\nValidating iteration {i + 1} / {validation_sample_size}:")
        val_loss = network.validate(inputs, ground_truths, verbose)
        avg_validation_loss += val_loss / validation_sample_size

        if verbose:
            print(f"Val iter loss: {val_loss}")

        if pause_after_iter:
            input()

    print(f"avg. training loss: {avg_training_loss}, avg. validation loss: {avg_validation_loss}")

    weight_file = os.path.join("logs", model_name, f"E{epoch_number}_weights.txt")
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)
    network.save_weights(weight_file)

    loss_file = os.path.join("logs", model_name, "loss.txt")
    with open(loss_file, "a") as f:
        f.write(f"Epoch {epoch_number} - avg. training loss: {avg_training_loss}, "
                f"avg. val. loss: {avg_validation_loss}\n")

def train_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: int, stop_after_epoch: int):
    """
    Scaffolds the network model, loads previous weights (if any), trains, validates and saves new weights & losses

    :param model_name:
    :param stop_after_epoch: Stop training after training this epoch numer
    :return:
    """
    training_data = generate_data(800)
    val_data = generate_data(200)
    network = Network()
    create_network_model(network, hidden_layers, nodes_per_layer, elu_alpha)
    # Load most recently trained epoch
    # If no trained weights exist, start training from epoch 1
    prev_epoch = load_weights(network, model_name) or 0

    if prev_epoch >= stop_after_epoch:
        print(f"WARN: Not training anything! Epoch {stop_after_epoch} was already trained.")

    curr_epoch = prev_epoch + 1

    while curr_epoch <= stop_after_epoch:
        print(f"Training new epoch: {curr_epoch}")
        train_one_epoch(network, training_data, val_data, 0.001, model_name, curr_epoch, 80, 20)
        curr_epoch += 1

def test_model(model_name: str, hidden_layers: int, nodes_per_layer: int, elu_alpha: int, epoch_number: Optional[int] = None):

    network = Network()
    create_network_model(network, hidden_layers, nodes_per_layer, elu_alpha)

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


if __name__ == "__main__":
    # d = generate_data(10)
    # test_model('test5_5_alpha1_bidirectional', 5, 5, 1)
    train_model('test6_6_alpha1_bidirectional', 6, 6, 1, 100)

    exit(0)
