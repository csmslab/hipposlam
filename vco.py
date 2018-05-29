# ##############################################################################
# Import libraries, etc.
# ##############################################################################

import numpy as np
import numpy.random as nprd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
import math

# ##############################################################################
# Define VCO_model class
# ##############################################################################

class VCO_model:

    def __init__(self, N, rho, theta, phz_noise=0):
        self.N = N
        self.rho = rho
        self.theta = theta
        self.phz_noise = phz_noise
        self.cellphz = self._add_noise()

    def __repr__(self):
        rs = 'VCO [N={}, (rho, theta)=({}, {:f}), phi_n={}]'
        return rs.format(self.N, self.rho, self.theta, self.phz_noise)

    def _add_noise(self):
        cellphz = np.zeros(self.N)
        phz_int = 2.0 * np.pi / self.N
        valid = False
        while not valid:
            phase = 0
            for i in range(self.N):
                cellphz[i] = phase
                if (i==(self.N-1)):
                    if not((phase > 2*np.pi) or (phase < 2*(np.pi - phz_int))):
                        valid = True
                else:
                    noise = (2 * phz_int * (nprd.random()-0.5)) * self.phz_noise
                    phase = phase + phz_int + noise
        return cellphz


    def _add_noise_gauss(self):
        cellphz = np.zeros(self.N)
        phz_int = 2.0 * np.pi / self.N
        phz_noise = self.phz_noise * phz_int
        if not phz_noise:
            return np.arange(0,2*np.pi,phz_int)
        valid = False
        while not valid:
            #print("trying...")
            phase = 0
            for i in range(N):
                cellphz[i] = phase
                if (i==(N-1)):
                    if not((phase > 2*np.pi) or (phase < 2*(np.pi - phz_int))):
                        valid = True
                else:
                    noise = np.maximum(0,(nprd.randn() + phz_int) * phz_noise)
                    phase = phase + noise
        return cellphz

    def get_env_pol(self, cell, pol_path, F_bar=7):
        vel_term = (self.rho * pol_path[:,0])/(2 * np.pi)
        phz_term = np.cos(pol_path[:,1] + self.cellphz[cell] - self.theta)
        return F_bar + vel_term * phz_term

    def get_envelope(self, cell, x, y, F_bar=0):
        x_term = self.rho * np.cos(-self.theta) * x
        y_term = self.rho * np.sin(-self.theta) * y
        phz_term = self.cellphz[cell] + np.pi/2.0
        return F_bar + np.exp(1j * (x_term + y_term + phz_term));
    '''
    def get_env_mult(self, cell, x, y):
        x_term   = self.rho * np.cos(-self.theta) * x
        y_term   = self.rho * np.sin(-self.theta) * y
        phz_term = self.cellphz[cell] + np.pi/2.0
        cell_osc = np.exp(1j * (x_term + y_term + phz_term))
        ref_osc = 7*np.ones_like(cell_osc)
        return ref_osc + cell_osc
    '''
    def ring_activity(self, path):
        activity = np.zeros([self.cellphz.shape[0], path.shape[0]])
        for cell in range(self.cellphz.shape[0]):
            activity[cell,:] = self.cell_activity(cell, path)
        return activity

# ##############################################################################
# Various Helper Functions
# ##############################################################################

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def randwalk(v=5, nsteps=100, size=5):
    rwpath = np.ones([nsteps,2]) * (size/2.)
    for step in np.arange(1,nsteps):
        while (np.abs(rwpath[step,:])>=(size/2.)).any():
            vel = np.random.random()/v
            theta=2*math.pi*np.random.random()
            dx = vel*math.cos(theta)
            dy = vel*math.sin(theta)
            rwpath[step,:] = rwpath[step-1,:] + [dx, dy]

    return rwpath

# ##############################################################################
# Plotting functions
# ##############################################################################

def plot_weights(weights):
    (y_size, x_size) = weights.shape
    plt.figure()
    im = plt.imshow(weights,cmap='jet',origin='lower')
    ax = plt.gca()
    plt.title('Weights Matrix')
    ax.grid(which='both', color='lightgray', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-0.5,x_size,1))
    ax.set_yticks(np.arange(-0.5,y_size,1))
    ax.set_xticklabels(np.arange(0, 12, 1))
    ax.set_yticklabels(np.arange(0, 6, 1))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    plt.colorbar(im, cax=cax)

def plot_randwalk(path, envelope, arena=(5,5)):
    x   = path[:,0]
    y   = path[:,1]

    envelope = abs(envelope)
    max_env = np.max(envelope)
    env_thresh = envelope - 0.65*max_env
    env_thresh[env_thresh<0] = 0
    max_env  = np.max(env_thresh)
    env_norm = env_thresh/max_env

    # set up a list of (x,y) points
    points = np.array([x,y]).transpose().reshape(-1,1,2)
    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap('jet'))
    lc.set_array(env_norm) # color the segments by our parameter

    # plot the collection
    plt.figure()
    plt.gca().add_collection(lc) # add the collection to the plot
    plt.xlim(-arena[0], arena[0]) # line collections don't auto-scale the plot
    plt.ylim(-arena[1], arena[1])


def plot_cells(data, numcells, size=(10,10)):
    n_rc = int(np.ceil(np.sqrt(numcells)))
    fig, axes = plt.subplots(nrows=n_rc, ncols=n_rc, sharex=True, sharey=True, figsize=size)
    axes_list = [item for sublist in axes for item in sublist]

    for idx in range(numcells):
        ax = axes_list.pop(0)
        ax.imshow(data[:,:,idx],cmap='jet',extent=(-size,size,-size,size))
        ax.set_title(idx)
        ax.tick_params(
            which='both',
            bottom='off',
            left='off',
            right='off',
            top='off'
        )
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    for ax in axes_list:
        ax.remove()

    plt.tight_layout()

def plot_many(things_to_plot,size):
    number = things_to_plot.shape[2]
    n_rc = int(np.ceil(np.sqrt(number)))
    n_plots = n_rc**2
    if n_plots > number:
        n_plots = number
    
    fig, axes = plt.subplots(nrows=n_rc, ncols=n_rc, sharex=True, sharey=True, figsize=(10,10))
    axes_list = [item for sublist in axes for item in sublist]

    for idx in range(n_plots):
        ax = axes_list.pop(0)
        ax.imshow(things_to_plot[:,:,idx],cmap='jet',extent=(-size,size,-size,size))
        ax.set_title(idx)
        ax.tick_params(
            which='both',
            bottom='off',
            left='off',
            right='off',
            top='off'
        )
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    for ax in axes_list:
        ax.remove()

    plt.tight_layout()

# ##############################################################################
# Matrix Helper Functions
# ##############################################################################

def theta_to_hcn(matrix, weights, size):
    # Create mesh grid to tile space of [[-size, size],[-size, size]]
    ss = np.linspace(-size,size,10*size)
    xx, yy = np.meshgrid(ss,ss)

    # Find sum of responses from all cells in weights matrix
    env_sum = np.zeros([10*size,10*size],dtype='complex128')
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if not np.isnan(weights[i,j]):
                cell_env = matrix[i][j].get_envelope(int(weights[i,j]), xx, yy)
                env_sum = env_sum + cell_env

    env_sum = np.abs(env_sum)
    max_env = np.max(env_sum)
    thresh_env = env_sum - 0.65*max_env
    thresh_env[thresh_env<0] = 0
    max_env = np.max(thresh_env)
    norm_env = thresh_env / max_env
    return norm_env, env_sum

def theta_to_grid(matrix, weights, size,verbose=False):
    # Create mesh grid to tile space of [[-size, size],[-size, size]]
    ss = np.linspace(-size,size,10*size)
    xx, yy = np.meshgrid(ss,ss)

    # Find product of responses from all cells in weights matrix
    env_product = np.ones([10*size,10*size])
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if not np.isnan(weights[i,j]):
                cell_env = matrix[i][j].get_envelope(int(weights[i,j]), xx, yy,1)
                env_product = env_product * cell_env
                if verbose:
                    print("max: {}, min: {}".format(np.max(env_product),np.min(env_product)))

    env_product = np.abs(env_product)
    max_env = np.max(env_product)
    thresh_env = env_product - 0.75*max_env
    thresh_env[thresh_env<0] = 0
    max_env = np.max(thresh_env)
    norm_env = thresh_env / max_env
    return norm_env, env_product
