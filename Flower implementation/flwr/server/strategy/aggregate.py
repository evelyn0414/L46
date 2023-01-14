# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Tuple

import numpy as np

from flwr.common import NDArrays


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_ap(results: List[Tuple[NDArrays, int]]) -> List[NDArrays]:
    """Compute weighted average."""
    # Select the dataset and hardcode the calculated weight matrix W from pretrained model
    datasets = ["heart_disease", "ixi", "isic2019"]
    dataset_selected = datasets[0]
    if dataset_selected == datasets[0]:
        W = np.array([[0.5,        0.20853288, 0.11731837, 0.17414875], \
                      [0.20355039, 0.5,        0.11963371, 0.17681591], \
                      [0.15743031, 0.16446689, 0.5,        0.1781028 ], \
                      [0.17842509, 0.18559211, 0.1359828,  0.5       ]])      
    elif dataset_selected == datasets[1]:
        W = np.array([[0.5,        0.20898099, 0.29101901], \
                      [0.28421542, 0.5,        0.21578458], \
                      [0.32358209, 0.17641791, 0.5       ]])
    elif dataset_selected == datasets[2]:
        W = np.array([[0.5,        0.07807627, 0.03027147, 0.05276318, 0.05035109, 0.28853799], \
                      [0.07778049, 0.5,        0.04824295, 0.15609163, 0.14048568, 0.07739925], \
                      [0.06033057, 0.0965131,  0.5,        0.13964233, 0.14403155, 0.05948244], \
                      [0.02846546, 0.08453085, 0.03780074, 0.5,        0.32116901, 0.02803395], \
                      [0.0276895,  0.07755086, 0.03974292, 0.32738033, 0.5,        0.02763638], \
                      [0.28952508, 0.07795937, 0.02994801, 0.05214111, 0.05042643, 0.5       ]])

    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute weight of each layer for different clients according to weight matrix W
    ap_w = []
    for client in range(W.shape[1]):
        layers = []
        for layer in zip(*weights):
            W_client = W[client, :]
            for i in range(np.asarray(layer).ndim - 1):
                W_client = np.expand_dims(W_client, axis=-1)
            layers.append(np.sum(np.asarray(layer) * W_client, axis=0))
        ap_w.append(layers)
    
    # Server side still uses FedAvg
    num_examples_total = sum([num_examples for _, num_examples in results])

    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    ap_w.append(weights_prime)
    return ap_w


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
    ]
    return median_w


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters
