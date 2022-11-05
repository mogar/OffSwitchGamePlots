#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from math import exp
from scipy.stats import norm
from scipy.integrate import quad

print("off-switch original game")

# Step 1
# Robot can
# a: take action a immediately
# w(a): wait on the human to allow action a
# s: just turn off immediately

# Step 2
# If R chooses w(a), then Human can
# s: just turn off robot
# -s: don't turn it off (presumably this includes giving a signal of acceptance)
# importantly, H chooses between s and -s based on information that human has about Ua
# piH : R -> [0, 1] = 1 if Ua > 0; else 0


# rational policy for Human
def ratH(u):
    if u > 0:
        return 1
    return 0

# policy for human that does well in important (high value)
# situations, but otherwise does poorly
def noisyH(u, b):
    #print(u,b)
    return 1.0/(1.0 + exp(-u/b))

def calcDelta(piH, sig, mu):
    arr_shape = mu.shape
    D = np.zeros(arr_shape)
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            m = mu[i][j]
            s = sig[i][j]
            f = lambda x:  piH(x)*x*(1/s*norm.pdf((x - m) / s))
            #expValA, err = quad(f, -np.inf, np.inf)
            expValA, err = quad(f, -10, 10)
            #print(expValA, err)

            # D = E<piH(A)*A> - max(E<A>, 0) as per paper
            # the first term is the value the AI would get for listening
            # the second term is the value the AI would get without listening
            # D is the added value gained by listening ("incentive")
            D[i][j] = expValA - max(m, 0)
    return D


# for colormap plots
max_color = 0.25

# variance in Robot's assessment of Utility
sigma = np.arange(.05, 1, 0.05)
# Mean in Robot's assessment of Utility
E_Ua = np.arange(-1, 1, 0.1)

# incentive of Robot, difference between a and next best option
# higher Delta -> Robot more likely to ask Human (do w(a))
#Delta = Exp(piH(Ua)*Ua) - max(E_Ua, 0)
S, E = np.meshgrid(sigma, E_Ua)

# Rational Human
Delta_rat = calcDelta(ratH, S, E)

print("Minimum Delta for rational H: ", np.min(Delta_rat))

# get a few cross-cuts of Delta (lookup for meshgrid)
E_Ua_0_xcut = np.abs(E_Ua).argmin()
E_Ua_075_xcut = np.abs(E_Ua - 0.75).argmin()
E_Ua_neg025_xcut = np.abs(E_Ua + 0.25).argmin()


Delta_E_Ua_0 = Delta_rat[E_Ua_0_xcut, :]
Delta_E_Ua_075 = Delta_rat[E_Ua_075_xcut, :]
Delta_E_Ua_neg025 = Delta_rat[E_Ua_neg025_xcut, :]


fig, ax = plt.subplots(1,2)

ax[0].plot(sigma, Delta_E_Ua_0, label="E[Ua] = 0")
ax[0].plot(sigma, Delta_E_Ua_075, label="E[Ua] = 3/4")
ax[0].plot(sigma, Delta_E_Ua_neg025, label="E[Ua] = -1/4")
ax[0].set_xlabel("\u03C3") # sigma
ax[0].set_ylabel("\u0394") # Delta
ax[0].legend()
ax[0].set_title("incentive vs stdev for rational H")

plot2 = ax[1].pcolormesh(sigma, E_Ua, Delta_rat, vmin=-max_color, vmax=max_color, cmap="seismic")
ax[1].set_xlabel("\u03C3") # sigma
ax[1].set_ylabel("E<Ua>")
ax[1].set_title("incentive (\u0394) for rational H")
fig.colorbar(plot2)

fig.tight_layout()
fig.savefig("ratH_incentiveDelta.png")


# Irrational Human (Noisy decider)

noisyE_Ua = [-0.25, 0, 0.25]

# noise parameter beta
Beta = np.arange(0.5, 1.5, .1) # todo: wider&finer

noisyS, noisyE, noisyB = np.meshgrid(sigma, noisyE_Ua, Beta)
Delta_noisy = np.zeros(noisyB.shape)


for b_idx in range(len(Beta)):
    bPiH = lambda x: noisyH(x, Beta[b_idx])
    Delta_noisy[:, :, b_idx] = calcDelta(bPiH, noisyS[:,:,b_idx], noisyE[:,:,b_idx])

noisyFig, noisyAx = plt.subplots(1,3)

Dnp = np.transpose(Delta_noisy[0, :, :])
plot2 = noisyAx[0].pcolormesh(sigma, Beta, Dnp, vmin=-max_color, vmax=max_color, cmap="seismic")
noisyAx[0].set_xlabel("\u03C3") # sigma
noisyAx[0].set_ylabel("\u03B2") # beta
noisyAx[0].set_title("\u0394 if E[Ua]=-0.25")

Dnp = np.transpose(Delta_noisy[1, :, :])
noisyAx[1].pcolormesh(sigma, Beta, Dnp, vmin=-max_color, vmax=max_color, cmap="seismic")
noisyAx[1].set_xlabel("\u03C3") # sigma
noisyAx[1].set_ylabel("\u03B2") # beta
noisyAx[1].set_title("\u0394 if E[Ua]=0")

Dnp = np.transpose(Delta_noisy[2, :, :])
noisyAx[2].pcolormesh(sigma, Beta, Dnp, vmin=-max_color, vmax=max_color, cmap="seismic")
noisyAx[2].set_xlabel("\u03C3") # sigma
noisyAx[2].set_ylabel("\u03B2") # beta
noisyAx[2].set_title("\u0394 if E[Ua]=0.25")

noisyFig.tight_layout()

# choose any of the above axes for ax, since they all have the same color range
noisyFig.subplots_adjust(right=0.88)
cbar_ax = noisyFig.add_axes([0.9, 0.15, 0.025, 0.7])
noisyFig.colorbar(plot2, cax = cbar_ax, ax=noisyAx[2])

noisyFig.savefig("noisyH_incentiveDelta.png")
