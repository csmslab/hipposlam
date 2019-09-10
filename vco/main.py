from grid import *
from grid import Neuron
import pdb

N = 12
numrow = 6;
numcol = 12;

rhos = 0.14 * (np.sqrt(3) ** np.arange(numrow))
thetas = np.pi + 2.0 * np.pi * (np.arange(numcol)) / numcol

VCOmatrix = [[vco.VCO_model(N, rhos[i], thetas[j]) for j in range(numcol)] for i in range(numrow)]

# border cell (Fig. 7)
weights_border = np.full([6, 12], np.nan)
weights_border[:, 9] = [2, 3, 5, 7, 11, 9]

# large grid (Fig. 7)
weights_lgrid = np.full([6, 12], np.nan)
weights_lgrid[3, 0] = 9
weights_lgrid[3, 4] = 9
weights_lgrid[3, 8] = 3

# small grid (Fig. 7)
weights_sgrid = np.full([6, 12], np.nan)
weights_sgrid[3, 1] = 1
weights_sgrid[3, 5] = 9
weights_sgrid[3, 9] = 3

# place cell (Fig. 7)
weights_place = np.full([6, 12], np.nan)
rot_place = 8 * np.pi / 6.  # orientation of the tuning function is zero by default
weights_place[2, :] = [11, 11, 0, 0, 0, 0, 0, 0, 11, 10, 10, 10]
weights_place[3, :] = [10, 11, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10]
weights_place[4, :] = [10, 11, 0, 1, 2, 1, 0, 0, 10, 9, 8, 9]

# curved border (Fig. 9)
weights_cborder = np.full([6, 12], np.nan)
for col in [0, 1, 2, 3, 4, 11]:
    weights_cborder[:, col] = [1, 1, 2, 4, 7, 1]

# lumpy border (supplemental)
weights_lborder = np.full([6, 12], np.nan)
weights_lborder[:, 0] = [1, 2, 3, 5, 8, 3]
weights_lborder[2, 2] = 6
weights_lborder[3, 2] = 1

# multi-field dentate place cell in square box (supplemental)
weights_dplace = np.full([6, 12], np.nan)
weights_dplace[2, :] = [8, 0, 1, 1, 7, 4, 2, 7, 8, 8, 4, 2]
weights_dplace[3, :] = [3, 0, 3, 0, 5, 0, 1, 4, 5, 9, 9, 11]
weights_dplace[4, :] = [0, 2, 2, 10, 7, 0, 0, 5, 6, 2, 6, 4]
pdb.set_trace()
border_cell = Neuron('border', VCOmatrix, 'hand', weights_border)
lgrid_cell = Neuron('lgrid', VCOmatrix, 'hand', weights_lgrid)
sgrid_cell = Neuron('sgrid', VCOmatrix, 'hand', weights_sgrid)
place_cell = Neuron('place', VCOmatrix, 'hand', weights_place)
cborder_cell = Neuron('cborder', VCOmatrix, 'hand', weights_cborder)
lborder_cell = Neuron('lborder', VCOmatrix, 'hand', weights_lborder)
dplace_cell = Neuron('dplace', VCOmatrix, 'hand', weights_dplace)
border_cell.show_plot(5)