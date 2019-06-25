import vco
import numpy as np
import matplotlib.pyplot as plt

class Neuron:

    def __init__(self, cell_type, matrix, weight_type, *args):
        self.cell_type = cell_type
        if weight_type == 'hand':
            weights = args[0]
            if isinstance(weights[0], tuple):
                self.weights = weights
            else:
                self._set_weight(args[0])
        self.mat = matrix
        self.hierarchy = 0
        while isinstance(matrix[0], list):
            self.hierarchy += 1
            matrix = matrix[0]
        self.inputs = []
        for weight in self.weights:
            x, y, z, w = weight
            self.inputs.append(self.mat[x][y])

    def __repr__(self):
        rs = 'Neuron [{}cell, input type = {}, number of inputs ={}]'
        return rs.format(self.cell_type, type(self.inputs[0]), len(self.weights))

    def show_plot(self, size):
        if self.hierarchy == 1:
            out_norm, out_env = vco.matrix_sum(self.mat, self.weights, size)
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
            axs[0].imshow(out_env, aspect='auto', cmap='jet', extent=(-size, size, -size, size))
            axs[0].set_title(self.cell_type + ': Envelope')
            axs[1].imshow(out_norm, aspect='auto', cmap='jet', extent=(-size, size, -size, size))
            axs[1].set_title(self.cell_type + ': Normalized Envelope')
            plt.tight_layout()

    def add_weight(self, weight):
        self.weights = [self.weights, weight]

    def get_inputs(self):
        return self.inputs

    def _set_weight(self, weights):
        self.weights = []
        for x in range(len(weights)):
            for y in range(len(weights[0])):
                if not np.isnan(weights[x, y]):
                    self.weights.append((x, y, weights[x, y], 1))


def auto_weights(start, end, matrix):
    grid_matrix = []
    jdx = 0
    for row in range(start, end):
        for firstcol in range(4):
            for cell1 in range(12):
                for cell2 in range(12):
                    for cell3 in range(12):
                        partial_matrix = np.full([6, 12], np.nan)
                        partial_matrix[row, firstcol] = cell1
                        partial_matrix[row, firstcol + 4] = cell2
                        partial_matrix[row, firstcol + 8] = cell3
                        grid_matrix.append(Neuron('grid', matrix, 'hand', partial_matrix))
                        jdx = jdx + 1
    return grid_matrix
