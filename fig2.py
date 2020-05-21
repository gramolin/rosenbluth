"""A reanalysis of Rosenbluth measurements of the proton form factors
   (script to make Figure 2)
   
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
f, axarr = plt.subplots(3, sharex=True, figsize=(7, 15))
f.subplots_adjust(left=0.15, right=0.97, bottom=0.06, top=0.98, hspace=0.07)

# The axis labels:
axarr[0].set_ylabel(r"$G_E / G_D$", fontsize=20)
axarr[1].set_ylabel(r"$G_M / (\mu \, G_D)$", fontsize=20)
axarr[2].set_ylabel(r"$\mu \, G_E / G_M$", fontsize=20)
axarr[2].set_xlabel(r"$Q^2,~\mathrm{GeV}^2$", fontsize=20, labelpad=9)

# The "(a)", "(b)", and "(c)" labels:
axarr[0].text(0.5, 0.1, '(a)', fontsize=22)
axarr[1].text(0.5, 0.9, '(b)', fontsize=22)
axarr[2].text(0.5, 0.1, '(c)', fontsize=22)

# The axis limits:
axarr[0].set_ylim(-0.1,2.5)
axarr[1].set_ylim(0.885,1.10)
axarr[2].set_xlim(0,9.2)
axarr[2].set_ylim(-0.1,2.5)
axarr[2].set_xticks(range(10))

# Initializing some arrays:
QQ1 = np.zeros(100)
QQ2 = np.zeros(100)
fits = np.zeros((3,100))
bands = np.zeros((3,100))
kelly = np.zeros((3,100))
diff = np.zeros((3,6))
sigma = np.zeros((6,6))


"""Plotting the Kelly fit, our fit, and the confidence bands"""

# Reading the best-fit parameters and the covariance matrix:
fit = pd.read_csv('results_fit.csv')

for j in range(100):
  # Q^2 values:
  QQ1[j] = 1. + 7.83*j/99. # From 1 to 8.83 GeV^2
  QQ2[j] = 10*j/99.        # From 0 to 10 GeV^2
  
  # Tau value:
  tau = QQ1[j]/(2.*rc.m_p)**2
  
  # Kelly fit:
  kelly[0,j] = rc.GE_Kelly(QQ2[j])/rc.GD(QQ2[j])         # G_E/G_D
  kelly[1,j] = rc.GM_Kelly(QQ2[j])/(rc.mu*rc.GD(QQ2[j])) # G_M/(mu*G_D)
  kelly[2,j] = kelly[0,j]/kelly[1,j]                     # mu*G_E/G_M
  
  # Our fit:
  fits[0,j] = rc.GE_fit(QQ1[j], fit['a1'][0], fit['a2'][0], fit['a3'][0])/rc.GD(QQ1[j])
  fits[1,j] = rc.GM_fit(QQ1[j], fit['b1'][0], fit['b2'][0], fit['b3'][0])/(rc.mu*rc.GD(QQ1[j]))
  fits[2,j] = fits[0,j]/fits[1,j]
  
  # Partial derivatives of G_E/G_D with respect to a1, a2, and a3:
  diff[0,0] = -tau/(2.*fits[0,j])
  diff[0,1] = diff[0,0] * tau
  diff[0,2] = diff[0,0] * tau**2
  
  # Partial derivatives of G_M/(mu*G_D) with respect to b1, b2, and b3:
  diff[1,3] = -tau/(2.*fits[1,j])
  diff[1,4] = diff[1,3] * tau
  diff[1,5] = diff[1,3] * tau**2
  
  # Partial derivatives of mu*G_E/G_M with respect to a1, a2, and a3:
  diff[2,0] = -tau/(2.*fits[0,j]*fits[1,j])
  diff[2,1] = diff[2,0] * tau
  diff[2,2] = diff[2,0] * tau**2
  
  # Partial derivatives of mu*G_E/G_M with respect to b1, b2, and b3:
  diff[2,3] = tau*fits[0,j]/(2.*fits[1,j]**3)
  diff[2,4] = diff[2,3] * tau
  diff[2,5] = diff[2,3] * tau**2
  
  # Elements of the covariance matrix:
  sigma[0,:] = fit['a1'][4:]
  sigma[1,:] = fit['a2'][4:]
  sigma[2,:] = fit['a3'][4:]
  sigma[3,:] = fit['b1'][4:]
  sigma[4,:] = fit['b2'][4:]
  sigma[5,:] = fit['b3'][4:]
  
  # Calculation of the confidence bands:
  for i in range(3):
    for k in range(6):
      for l in range(6):
        bands[i,j] = bands[i,j] + diff[i,k]*diff[i,l]*sigma[k,l]
    bands[i,j] = np.sqrt(bands[i,j])

# Plotting panel (a):
axarr[0].plot([0,10], [1,1], ':k') # Horizontal line
axarr[0].plot(QQ2, kelly[0], '--k', linewidth=2, alpha=0.7) # Kelly fit
axarr[0].plot(QQ1, fits[0], '-r', linewidth=2, alpha=0.6)   # Our fit
axarr[0].fill_between(QQ1, fits[0] - bands[0], fits[0] + bands[0], color='r', alpha=0.4)

# Plotting panel (b):
axarr[1].plot([0,10], [1,1], ':k') # Horizontal line
axarr[1].plot(QQ2, kelly[1], '--k', linewidth=2, alpha=0.7) # Kelly fit
axarr[1].plot(QQ1, fits[1], '-r', linewidth=2, alpha=0.6)   # Our fit
axarr[1].fill_between(QQ1, fits[1] - bands[1], fits[1] + bands[1], color='r', alpha=0.4)

# Plotting panel (c):
axarr[2].plot([0,10], [1,1], ':k') # Horizontal line
axarr[2].plot(QQ2, kelly[2], '--k', linewidth=2, alpha=0.7) # Kelly fit
axarr[2].plot(QQ1, fits[2], '-r', linewidth=2, alpha=0.6)   # Our fit
axarr[2].fill_between(QQ1, fits[2] - bands[2], fits[2] + bands[2], color='r', alpha=0.4)


"""Plotting the original data points"""

# Reading input data from the file "data_unpolarized.csv":
df1 = pd.read_csv('data_unpolarized.csv')

exp1 = df1[df1['experiment']==1] # Data of Walker et al.
exp2 = df1[df1['experiment']==2] # Data of Andivahis et al.

# Panel (a):
axarr[0].errorbar(exp1['QQ'], exp1['GE'], yerr=[exp1['GE_error-'], exp1['GE_error+']], fmt='o', markersize=6, markerfacecolor='white', markeredgecolor='blue', color='blue', capsize=5)
axarr[0].errorbar(exp2['QQ'], exp2['GE'], yerr=[exp2['GE_error-'], exp2['GE_error+']], fmt='o', markersize=6, markeredgecolor='blue', color='blue', capsize=5)

# Panel (b):
axarr[1].errorbar(exp1['QQ'], exp1['GM'], yerr=[exp1['GM_error-'], exp1['GM_error+']], fmt='o', markersize=6, markerfacecolor='white', markeredgecolor='blue', color='blue', capsize=5)
axarr[1].errorbar(exp2['QQ'], exp2['GM'], yerr=[exp2['GM_error-'], exp2['GM_error+']], fmt='o', markersize=6, markeredgecolor='blue', color='blue', capsize=5)

# Panel (c):
axarr[2].errorbar(exp1['QQ'], exp1['GE_GM'], yerr=[exp1['GE_GM_error-'], exp1['GE_GM_error+']], fmt='o', markersize=6, markerfacecolor='white', markeredgecolor='blue', color='blue', capsize=5)
axarr[2].errorbar(exp2['QQ'], exp2['GE_GM'], yerr=[exp2['GE_GM_error-'], exp2['GE_GM_error+']], fmt='o', markersize=6, markeredgecolor='blue', color='blue', capsize=5)


"""Plotting the data of polarized measurements"""

# Reading input data from the file "data_polarized.csv":
df2 = pd.read_csv('data_polarized.csv')

# Combined statistical and systematic uncertainties:
df2['error'] = np.sqrt(df2['estat']**2 + df2['esyst']**2)

exp1 = df2[df2['experiment']==1] # Data of Punjabi et al. (2005)
exp2 = df2[df2['experiment']==2] # Data of Puckett et al. (2012)
exp3 = df2[df2['experiment']==3] # Data of Puckett et al. (2010)

# Plotting the data points with the corresponding error bars:
axarr[2].errorbar(exp1['QQ'], exp1['GE_GM'], yerr=exp1['error'], fmt='s', markersize=6, capsize=5, color='g')
axarr[2].errorbar(exp2['QQ'], exp2['GE_GM'], yerr=exp2['error'], fmt='^', markersize=8, capsize=5, color='g')
axarr[2].errorbar(exp3['QQ'], exp3['GE_GM'], yerr=exp3['error'], fmt='v', markersize=8, capsize=5, color='g')


"""Rosenbluth separation using the corrected cross sections"""

# Reading input data from the file "results_table.csv":
df3 = pd.read_csv('results_table.csv')

# Combined statistical and systematic uncertainties:
df3['error'] = 0.01*np.sqrt(df3['estat']**2 + df3['esyst']**2)

# Applying the new normalization factors:
for i in range(len(df3['set'])):
  if df3['set'][i] == 1:
    df3.loc[i,'sigma'] = df3['sigma'][i]*fit['n1'][0] # Set 1
  elif df3['set'][i] == 2:
    df3.loc[i,'sigma'] = df3['sigma'][i]*fit['n2'][0] # Set 2
  else:
    df3.loc[i,'sigma'] = df3['sigma'][i]*fit['n3'][0] # Set 3

# Q^2 values:
QQ_list = [1., 2.003, 2.497, 3.007, 1.75, 2.5, 3.25, 4., 5., 6., 7.]

# An array for storing results:
new_points = np.zeros((3,len(QQ_list)))

# An object of the Kinematics class:
kin = rc.Kinematics()

for i in range(len(QQ_list)):
  # Q^2 value:
  QQ = QQ_list[i]
  
  # Tau value:
  tau = QQ/(2.*rc.m_p)**2
  
  # Epsilon values (arguments to fit):
  x = df3[(df3['QQ'] - QQ)**2 < 1e-6]['epsilon'].values
  
  # Calculation of the Mott cross sections:
  sigma_Mott = np.zeros(len(x))
  for j in range(len(x)):
    kin.Set_QQ_epsilon(QQ=QQ, epsilon=x[j])
    sigma_Mott[j] = kin.Get_sigma_Mott()
  
  # Reduced cross sections (values to fit):  
  y = rc.NbToGeV(df3[(df3['QQ'] - QQ)**2 < 1e-6]['sigma'])*x*(1. + tau)/sigma_Mott
  
  # Uncertainties of the reduced cross sections:
  e = df3[(df3['QQ'] - QQ)**2 < 1e-6]['error']*y
  
  # The slope and the intercept of the linear fit:
  slope, intercept = rc.Linear_fitter(x, y, e)
  
  # Results of the Rosenbluth separation:
  new_points[0,i] = np.sqrt(slope)/rc.GD(QQ)                 # G_E/G_D
  new_points[1,i] = np.sqrt(intercept/tau)/(rc.mu*rc.GD(QQ)) # G_M/(mu*G_D)
  new_points[2,i] = new_points[0,i]/new_points[1,i]          # mu*G_E/G_M

# Panel (a):
axarr[0].plot(QQ_list[:4], new_points[0][:4], marker=(4,2,45), color='k', markeredgewidth=1, linewidth=0, markersize=12)
axarr[0].plot(QQ_list[4:], new_points[0][4:], marker=(6,2,90), color='k', markeredgewidth=1, linewidth=0, markersize=12)

# Panel (b):
axarr[1].plot(QQ_list[:4], new_points[1][:4], marker=(4,2,45), color='k', markeredgewidth=1, linewidth=0, markersize=12)
axarr[1].plot(QQ_list[4:], new_points[1][4:], marker=(6,2,90), color='k', markeredgewidth=1, linewidth=0, markersize=12)

# Panel (c):
axarr[2].plot(QQ_list[:4], new_points[2][:4], marker=(4,2,45), color='k', markeredgewidth=1, linewidth=0, markersize=12)
axarr[2].plot(QQ_list[4:], new_points[2][4:], marker=(6,2,90), color='k', markeredgewidth=1, linewidth=0, markersize=12)

# Saving the figure to pdf and png files:
plt.savefig('fig2.pdf')
plt.savefig('fig2.png')

