from grid import *
from grid import Neuron
#import pdb

N = 12
numrow = 6;
numcol = 12;

rhos = 0.14 * (np.sqrt(3) ** np.arange(numrow))
thetas = np.pi + 2.0 * np.pi * (np.arange(numcol)) / numcol

VCOmatrix = [[vco.VCO_model(N, rhos[i], thetas[j]) for j in range(numcol)] for i in range(numrow)]

# border cell (Fig. 7)
weights_border = np.full([6, 12], np.nan)
weights_border[:, 9] = [2, 3, 5, 7, 11, 9]

border_cell = Neuron('border', VCOmatrix, 'hand', weights_border)

weights = border_cell.weights
weight_number = 3
#pdb.set_trace()
neuron_list = []
x,y,z,w = weights[weight_number]
for f in np.arange(0.1, 1.1, 0.1):
    weights[weight_number] = (x, y, z, f)
    neuron_list.append(Neuron('border', VCOmatrix, 'hand', weights[:]))

ind = 1
for neuron in neuron_list:
    neuron.show_plot(5, ind)
    ind += 1
