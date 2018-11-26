import sys
import os
from decimal import Decimal, Overflow
from random import sample
from time import sleep
from typing import List, Dict, Optional, Union
import re

from data import Data
from helper_functions import coerce_decimal

WEIGHT_RE = re.compile(r'\s*(?P<node>[^,]+)\s*,\s*(?P<input>[^,]+)\s*,\s*(?P<weight>.+?)\s*')
ITER_RE = re.compile(r'iter:\s*(?P<iter>[0-9]+)\s*')


class Network:
    """
    The neural network graph of nodes
    """

    def __init__(self):
        from node import Node, ConstantNode
        self.input_nodes: List[ConstantNode] = []
        self.output_nodes: List[Node] = []

        self.nodes: Dict[str, Node] = {}

        self.iter: int = 0
        """
        Iteration counter that increments after each call to train_iter().
        
        This value persists across weight saves/loads. 
        """

    def register_nodes(self, input_nodes: List['ConstantNode'], output_nodes: List['Node']):
        """
        Populates the Network structure with a dictionary of nodes by recursion of
        the node's inputs starting from the output nodes.

        Also ensures that nodes don't share an identical name.

        :param input_nodes: A list of input nodes that the network can map input data to
        :param output_nodes: A list of output nodes for activation calculation & ground-truth error evaluation
        """
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        def register_node_and_inputs(node: 'Node'):

            assert node.name not in self.nodes, \
                f"Node name conflicted: {node.name}. Ensure that node names are unique and there are no " \
                f"cyclically connected nodes."

            self.nodes[node.name] = node
            for i, _ in node.inputs.items():
                if i.name not in self.nodes or self.nodes[i.name] != i:
                    # Nodes in one layer may share the same input node, a node already registered
                    # does not need to be registered again and should not count as a node conflict.
                    register_node_and_inputs(i)

        # For each output node, add it to the node registry, for each of its inputs, add it to the registry, etc...
        for n in output_nodes:
            register_node_and_inputs(n)

    def assign_inputs(self, input_values: List[Union[float, Decimal]]):
        """
        Set the constant activation value for the network's input nodes to the values in input_values

        :param input_values:
                A list of values to set the input_nodes activations to. Values are set according to index order.
                i.e. input_nodes[idx].activation = input_values[idx]
        :return:
        """
        assert len(input_values) == len(self.input_nodes), "len(input_values) need to match len(Network.input_nodes)"

        input_values = [coerce_decimal(i) for i in input_values]

        for idx, i in enumerate(self.input_nodes):
            i.activation = input_values[idx]

    def forward_propagate(self) -> List[Decimal]:
        """
        Perform "forward propagation" recursively.

        :return: a list of activations corresponding to the activation values of the output nodes on the final layer.
        The values correspond in the same order as self.output_nodes
        """
        return [o.calculate_activation() for o in self.output_nodes]

    def evaluate_loss(self, ground_truths: List[Union[float, Decimal]]) -> Decimal:
        """
        Evaluates the loss score using the Mean-Square Error function.
        Note that this function uses the previously cached activation values. Forward propagation must
        be performed before calling this function

        :param ground_truths:
        :return:
        """
        assert len(ground_truths) == len(self.output_nodes), \
            "Number of ground truths need to match number of output nodes!"

        ground_truths = [coerce_decimal(x) for x in ground_truths]

        try:
            square_error = sum([(self.output_nodes[i].activation - g) ** 2 for i, g in enumerate(ground_truths)])
        except Overflow:
            pass

        mean_square_error = square_error / len(self.output_nodes)
        return mean_square_error

    def evaluate_dloss_doutput(self, output_node: 'Node', ground_truths: List[Union[float, Decimal]]) -> Decimal:
        """
        Evaluates d(loss) / d(output_node.activation) assuming the loss function used is MSE

        :param output_node:
        :param ground_truths: The same ground truth values that was passed to evaluate_loss
        :return: d(loss) / d(output_node.activation)
        """

        assert output_node in self.output_nodes, \
            "output_node param must be registered as an output node of the network!"

        ground_truth = ground_truths[self.output_nodes.index(output_node)]

        return 2 * (output_node.activation - ground_truth)

    def evaluate_gradients(self, ground_truths: List[Union[float, Decimal]]):
        """
        Calls calc_dloss_dactivation on all the input nodes to recursively evaluate the d(loss) / d(activation)
        derivatives on all nodes.

        NOTE: forward_propagate must be called first!
        """

        assert len(ground_truths) == len(self.output_nodes), \
            "Number of ground truths need to match number of output nodes!"

        for i in self.input_nodes:
            i.calc_dloss_dactivation(self, ground_truths, self.iter)

    def update_weights(self, step_size: float, momentum: float, decay: float, log: bool = False) -> (Decimal, Decimal):
        """
        Update weights of all registered nodes.

        NOTE: evaluate_gradients must be called first!

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
        :param log:
                Set True to show information of each weight update
        :return: (average dloss_dweight, max dloss_dweight) amongst all the nodes
        """

        dlosses: List[Decimal] = []
        max_dloss = 0

        for node in self.nodes.values():
            gradients = node.update_weights(step_size, momentum, decay, log)
            if len(gradients) > 0:
                dlosses += [abs(g) for g in gradients]
                max_gradient = max(gradients)
                if max_gradient > max_dloss:
                    max_dloss = max_gradient

        return sum(dlosses) / len(dlosses), max_dloss

    def save_weights(self, path: str):
        """
        Saves all weights & other meta information in the file at specified path

        :param path: The path of the file
        """
        with open(path, "w") as f:

            f.write(f"iter: {self.iter}\n")

            for node in self.nodes.values():
                for weight_info in node.serialize_weights():
                    f.write(weight_info + "\n")

    def load_weights(self, path: str):
        """
        Load connection weights & other meta information from the file at the specified path.
        Make sure that the nodes are connected and registered first, and that
        the network model architecture that was used to save the weights file
        is the same as this network.

        :param path: The path of the file
        """

        contents = ""

        with open(path, "r") as f:
            contents = f.read()

        lines = contents.split('\n')

        for line in lines:

            iter_match = ITER_RE.fullmatch(line)
            if iter_match is not None:
                self.iter = int(iter_match.group("iter"))
                continue

            weight_match = WEIGHT_RE.fullmatch(line)
            if weight_match is not None:
                node_name = weight_match.group("node")
                input_name = weight_match.group("input")
                weight_str = weight_match.group("weight")

                assert node_name in self.nodes, f"Node '{node_name}' not registered in the network!"
                assert input_name in self.nodes, f"Node '{input_name}' not registered in the network!"

                try:
                    weight = float(weight_str)
                    self.nodes[node_name].set_weight(self.nodes[input_name], weight)
                except ValueError:
                    raise AssertionError(f"Invalid weight file format - weight '{weight}' is not a float")

                continue

    def train_iter(self, inputs: List[Union[float, Decimal]], ground_truths: List[Union[float, Decimal]],
                   step_size: float, momentum: float, decay: float, verbose: bool = False) \
            -> (float, float, float):
        """
        Train a single iteration of the network.
        :param inputs:
        :param ground_truths:
        :param step_size:
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
        :param verbose:
        :return: (training loss, avg. dloss_dweight, max dloss_dweight)
        """

        if verbose:
            print(f"Total iterations trained: {self.iter}")
            print(f"Assigning inputs: {inputs}")
            print(f"Expecting outputs: {ground_truths}")
        self.assign_inputs(inputs)

        output_activations = self.forward_propagate()
        if verbose:
            print(f"Output activations: {output_activations}")

        loss = self.evaluate_loss(ground_truths)

        self.evaluate_gradients(ground_truths)
        avg_dloss, max_dloss = self.update_weights(step_size, momentum, decay, verbose)

        self.iter += 1

        return loss, avg_dloss, max_dloss

    def validate(self, inputs: List[Union[float, Decimal]], ground_truths: List[Union[float, Decimal]],
                 verbose: bool = False) -> Decimal:
        """
        Does a forward pass and returns the loss. No weight update.

        :param inputs:
        :param ground_truths:
        :param verbose:
        :return: the loss score
        """

        if verbose:
            print(f"Assigning inputs: {inputs}")
        self.assign_inputs(inputs)

        output_activations = self.forward_propagate()
        if verbose:
            print(f"Output activations: {output_activations}")

        loss = self.evaluate_loss(ground_truths)

        return loss

    def predict(self, inputs: List[Union[float, Decimal]]) -> List[Decimal]:
        """
        Set inputs & get the network's output

        :param inputs: Input values
        :return: Output values
        """

        self.assign_inputs(inputs)
        return self.forward_propagate()


def train_one_epoch(network: Network, training_data: Data, validation_data: Data, model_name: str, epoch_number: int,
                    step_size: float, momentum: float = 0.1, decay: float = 0.0005,
                    training_iterations: Optional[int] = None, validation_iterations: Optional[int] = None,
                    verbose: bool = False, pause_after_iter: Optional[float] = None,
                    save_weights: bool = True):
    """
    Train the network for 1 epoch.

    Usually this means training on every sample in the dataset once, but the number of samples
    to train on can be overridden by specifying it in the iterations_per_epoch parameter.

    :param network: The Network object

    :param training_data: Training data

    :param validation_data: Validation data

    :param epoch_number: The epoch number to name the saved weight file with.

    :param step_size: The multiplier of the d(loss)/d(weight) derivative the weights are updated by.

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

    :param training_iterations:
            The number of training data samples to train on. Defaults to len(training_data).
            Set this if training_data is huge.

    :param validation_iterations:
            The number of validation data samples to use. Defaults to len(validation_data).
            Set this if validation_data is huge.

    :param model_name: A name used to identify this model when saving the weight files.

    :param verbose: Set True to show additional verbose logs

    :param pause_after_iter:
            (For debugging purposes)
            If 0, wait for a newline input before moving on to the next epoch,
            else if any other number specified, wait for that number of seconds,
            otherwise if default (None), do not wait between iterations.

    :param save_weights: Set False to not save the weights to the weights file after this epoch.
    :return:
    """

    training_data_subset = training_data

    if training_iterations is None or training_iterations > len(training_data):
        training_iterations = len(training_data)

    # Even if no subset specified, `sample` should still be used to randomize the order of the training data.
    training_data_subset = sample(training_data_subset, training_iterations)

    validation_data_subset = validation_data

    if validation_iterations is None or validation_iterations > len(validation_data):
        validation_iterations = len(validation_data)

    validation_data_subset = sample(validation_data_subset, validation_iterations)

    training_sample_size = len(training_data_subset)
    avg_training_loss = 0

    # Perform training iterations
    first = True
    for i, (inputs, ground_truths) in enumerate(training_data_subset):

        iter_loss, avg_dloss, max_dloss = \
            network.train_iter(inputs, ground_truths, step_size, momentum, decay, verbose)

        print("\033[A\033[K" * (0 if first else 7))
        first = False

        print(f"\n_____________________________\nTrained iteration {i + 1} / {training_sample_size}:")

        print(f"Iter loss: {iter_loss}\n"
              f"Avg d(loss)/d(weight): {avg_dloss}\n"
              f"Max d(loss)/d(weight): {max_dloss}")
        avg_training_loss += iter_loss / training_sample_size

        if pause_after_iter == 0:
            input()
        elif pause_after_iter is not None:
            sleep(pause_after_iter)

    validation_sample_size = len(validation_data_subset)
    avg_validation_loss = 0
    # Perform validation iterations
    for i, (inputs, ground_truths) in enumerate(validation_data_subset):
        print(f"\n_____________________________\nValidating iteration {i + 1} / {validation_sample_size}:")

        val_loss = network.validate(inputs, ground_truths, verbose)

        avg_validation_loss += val_loss / validation_sample_size

        if verbose:
            print(f"Val iter loss: {val_loss}")

        if pause_after_iter:
            input()

        print("\033[A\033[K" * (5 if verbose else 4))


    print(f"avg. training loss: {avg_training_loss}, avg. validation loss: {avg_validation_loss}")

    if save_weights:
        weight_file = os.path.join("logs", model_name, f"E{epoch_number}_weights.txt")
        os.makedirs(os.path.dirname(weight_file), exist_ok=True)
        network.save_weights(weight_file)

    loss_file = os.path.join("logs", model_name, "loss.txt")
    with open(loss_file, "a") as f:
        f.write(f"Epoch {epoch_number} - avg. training loss: {avg_training_loss}, "
                f"avg. val. loss: {avg_validation_loss}\n")


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