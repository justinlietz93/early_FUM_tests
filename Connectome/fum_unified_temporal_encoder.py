# unified_temporal_encoder.py
"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""
import numpy as np
import random

class StimulusEncoder:
    """
    Translates abstract stimuli into the FUM's native language:
    spatio-temporal spike patterns. This is the FUM's Cognitive Transducer.

    Each stimulus is converted into a sequence of input current arrays,
    which can be fed to the network over multiple time steps to create a
    unique "rhythm" of neural activity.
    """
    def __init__(self, num_input_neurons: int, pattern_duration: int):
        """
        Initializes the encoder.

        Args:
            num_input_neurons (int): The number of neurons in the FUM's input layer.
            pattern_duration (int): The number of time steps each pattern should last.
        """
        self.num_input_neurons = num_input_neurons
        self.pattern_duration = pattern_duration
        self.char_to_int = {char: i for i, char in enumerate('abcdefghijklmnopqrstuvwxyz()+-*&|^PQRSTUVWXY ')}

    def _string_to_sequence(self, input_string: str) -> np.ndarray:
        """Converts a string of symbols into a sequence of integer indices."""
        return np.array([self.char_to_int.get(char, 0) for char in input_string])

    def encode(self, stimulus_type: str, stimulus: any) -> list[np.ndarray]:
        """
        Master encode function that directs to the appropriate method.

        Args:
            stimulus_type (str): 'math', 'logic', or 'graph'.
            stimulus (any): The stimulus data (string or numpy array).

        Returns:
            list[np.ndarray]: A list of current arrays, representing the pattern over time.
        """
        if stimulus_type in ['math', 'logic']:
            return self._encode_symbolic(stimulus)
        elif stimulus_type == 'graph':
            return self._encode_graph(stimulus)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    def _encode_symbolic(self, stimulus_string: str) -> list[np.ndarray]:
        """
        Encodes a symbolic string (math or logic) into a temporal pattern.

        Each character in the string is assigned to a group of input neurons,
        and they are activated sequentially to create a "rhythm".
        """
        sequence = self._string_to_sequence(stimulus_string)
        patterns = []
        
        for i in range(self.pattern_duration):
            currents = np.zeros(self.num_input_neurons)
            # Activate the neuron corresponding to the character at this point in the "rhythm"
            char_index = sequence[i % len(sequence)]
            neuron_index = char_index % self.num_input_neurons
            currents[neuron_index] = 100.0 # Strong input pulse
            patterns.append(currents)
            
        return patterns

    def _encode_graph(self, adjacency_matrix: np.ndarray) -> list[np.ndarray]:
        """
        Encodes a graph's adjacency matrix into a spatio-temporal pattern.

        The graph is "rasterized" row by row over time.
        """
        patterns = []
        num_nodes = adjacency_matrix.shape[0]

        for i in range(self.pattern_duration):
            currents = np.zeros(self.num_input_neurons)
            # Select a row of the adjacency matrix to present at this time step
            row_to_present = adjacency_matrix[i % num_nodes]
            
            # Map the row to the input neurons
            for node_idx, connection in enumerate(row_to_present):
                if connection > 0:
                    neuron_index = node_idx % self.num_input_neurons
                    currents[neuron_index] = 75.0 # Moderate input pulse
            
            patterns.append(currents)

        return patterns