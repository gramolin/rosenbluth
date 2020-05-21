"""A reanalysis of Rosenbluth measurements of the proton form factors
   (script to make Figure 1)
   
   Copyright (c) Alexander Gramolin, 2016
   https://github.com/gramolin/rosenbluth/
"""

import numpy as np
import pandas as pd
import radcorr as rc
import matplotlib.pyplot as plt

# Some settings for the figure:
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=19)
plt.rc('xtick.major', size=9)
plt.rc('ytick', labelsize=19)
plt.rc('ytick.major', size=9)
plt.rc('ytick.minor', size=4)
plt.figure(1, figsize=(7, 5.5))
plt.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.97)
plt.semilogy() # Log vertical scale

# The axis labels:
plt.xlabel(r"$E_3,~\mathrm{GeV}$", fontsize=20, labelpad=12)
plt.ylabel(r"$d^2 \sigma_{\mathrm{int.br.}} / (d \Omega \, d E_3),~\mathrm{GeV}^{-3} \, \mathrm{sr}^{-1}$", fontsize=20)

# Reading the Monte Carlo data obtained using the ESEPP event generator
# (see https://github.com/gramolin/esepp):
esepp = pd.read_csv('data_esepp.csv')

# Plotting the Monte Carlo data:
plt.errorbar(esepp['E3'], esepp['cross_section'], xerr=0.012, fmt='ok', markersize=6, capsize=0, linewidth=1.5)

# An object of the FormFactors class:
ff = rc.FormFactors('Dipole') # The dipole parameterization

# Setting the kinematics (E1 = 1 GeV, theta = 70 deg):
kin = rc.Kinematics(E1=1., theta=rc.DegToRad(70.))

# Radiative tail according to the soft-photon approximation:
xx = np.arange(0.005, kin.Get_E3()-0.001, 0.001)
tail_soft = rc.sigma_IntBr_soft(E3=xx, kinematics=kin, ff=ff)

# More accurate description of the radiative tail:
tail_hard = rc.sigma_IntBr(E3=xx, kinematics=kin, ff=ff)
tail_hard = tail_hard + (2.*rc.alpha/rc.pi)*(1./(kin.Get_E3() - xx))*(2.*np.log(kin.Get_eta()) + \
kin.Get_E4()*np.log(kin.Get_x())/kin.Get_p4() - 1.)*rc.sigma_Rosenbluth(E1=1., theta=rc.DegToRad(70.), ff=ff)

# Plotting the curves:
plt.plot(xx, tail_hard, '-r', linewidth=2, alpha=0.8) # Red solid line
plt.plot(xx, tail_soft, '--b', linewidth=2) # Blue dashed line

# Saving the figure to pdf and png files:
plt.savefig('fig1.pdf')
plt.savefig('fig1.png')

