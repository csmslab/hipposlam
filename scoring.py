# %% ###########################################################################
# File for testing grid and place scores
# ##############################################################################
import vco
import numpy as np
import numpy.ma as ma
import numpy.random as nprd
import matplotlib.pyplot as plt

from scipy.signal                import correlate2d
from scipy.ndimage.interpolation import rotate

%matplotlib inline

# %% ###########################################################################
# Create VCO matrix (as per usual)
# ##############################################################################

numrow, numcol = 6, 12
ringsize = 12
minrho = 0.14
rotation_angle = 0
rhos = minrho * (np.sqrt(3)**(np.arange(numrow)))
thetas = np.pi + rotation_angle + 2*np.pi*(np.arange(numcol))/numcol
phz_noise = 0

VCOmat = [[vco.VCO_model(ringsize, rhos[i], thetas[j], phz_noise) \
           for j in range(numcol)] for i in range(numrow)]

# %% ###########################################################################
# 1000 random arithmetic grid cells
# ##############################################################################

n_inputs = 3
n_mats = 1000
arena = 10
hexGrid = False
w_grid = np.full([numrow,numcol,n_mats], np.nan)

for mat in range(n_mats):
    rand_row = nprd.randint(numrow)
    if hexGrid:
        rand_col = nprd.randint(4)
        cols = np.array([rand_col, rand_col+4, rand_col+8])
    else:
        cols = np.arange(numcol)
        nprd.shuffle(cols)

    for col in cols[:n_inputs]:
        w_grid[rand_row, col, mat] = nprd.randint(ringsize)

grids = np.zeros([10*arena, 10*arena, n_mats, 2])
for i in range(n_mats):
    grids[:,:,i,0], grids[:,:,i,1] = vco.theta_to_hcn(VCOmat, w_grid[:,:,i], arena)

# %% ###########################################################################
# Developing gridness score function
# ##############################################################################
def gridness_score(cell_activity,dAng=6,verbose=False):
    autocorr = correlate2d(cell_activity,cell_activity)
    # Removing center of autocorrelation
    xshape, yshape = autocorr.shape
    xc, yc, r = xshape/2, yshape/2, xshape/10
    x, y = np.meshgrid(np.arange(xshape), np.arange(yshape))
    d2 = (x - xc)**2 + (y - yc)**2
    mask = d2 < r**2
    autocorr_full = np.copy(autocorr)
    autocorr[mask] = 0

    if verbose:
        plt.figure()
        plt.subplot(131); plt.imshow(cell_activity)
        plt.subplot(132); plt.imshow(autocorr_full)
        plt.subplot(133); plt.imshow(autocorr)

    if dAng:
        angles = np.arange(0,180+dAng,dAng)
    else:
        angles = np.array([30, 60, 90, 120, 150])
    crosscorr = np.zeros(angles.shape)
    for idx, angle in enumerate(angles):
        rot_autocorr = rotate(autocorr, angle, reshape=False)
        C = np.corrcoef(np.reshape(autocorr, (1, autocorr.size)),
            np.reshape(rot_autocorr, (1, rot_autocorr.size)))
        crosscorr[idx] = C[0,1]

    if dAng:
        max_angles_i = np.array([30/dAng, 90/dAng, 150/dAng],dtype=int)
        min_angles_i = np.array([60/dAng, 120/dAng],dtype=int)
    else:
        max_angles_i = np.array([0, 2, 4])
        min_angles_i = np.array([1, 3])

    maxima = np.max(crosscorr[max_angles_i])
    minima = np.min(crosscorr[min_angles_i])
    G = minima - maxima

    return G, crosscorr, angles

# %% ###########################################################################
# Testing all randomly generated grid cells
# ##############################################################################

gridness = np.zeros(n_mats)
crosscorrs = np.zeros([n_mats,5])
for cell in range(grids.shape[2]):
    print(cell)
    gridness[cell], crosscorrs[cell,:], angles = gridness_score(grids[:,:,cell,0],dAng=0)

# %% ###########################################################################
# plot histogram
# ##############################################################################
_ = plt.hist(gridness,bins=20)
plt.scatter(range(n_mats),gridness)
# %% ###########################################################################
# checking them
# ##############################################################################
which=497
print('Gridness of cell {}: {}'.format(which, gridness[which]))
ax = plt.imshow(grids[:,:,which,0],cmap='jet')
