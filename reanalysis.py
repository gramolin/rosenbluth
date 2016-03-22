"""A reanalysis of Rosenbluth measurements of the proton form factors
   (main analysis routine)
   
   Copyright (c) Alexander Gramolin, 2016
   https://github.com/gramolin/rosenbluth/
"""

import numpy as np
import pandas as pd
import radcorr as rc

# Overall normalization uncertainties of the cross sections (%):
norm_uncert = [1.90, 1.77, 1.77] # For Sets 1, 2, and 3, respectively

# Reading the input data:
data = pd.read_csv('data_input.csv')

# Numbers of data points in each of the three sets:
N1 = len(data[data['set']==1])
N2 = len(data[data['set']==2])
N3 = len(data[data['set']==3])
print('Numbers of data points: N1 = ' + str(N1) + ', N2 = ' + str(N2) + ', N3 = ' + str(N3) + '\n')

# Objects of the Kinematics and FormFactors classes:
kin = rc.Kinematics()
ff = rc.FormFactors('Custom') # Custom parameterization

# Target material (hydrogen):
mat = rc.Material(Z=1, A=1.00794, X0=63.04)


"""Reapplying radiative corrections"""

for i in range(N1+N2+N3):
  # Setting the actual kinematics using the E1 and theta values:
  kin.Set_E1_theta(E1=data['E1'][i], theta=rc.DegToRad(data['theta'][i]))
  
  # Calculating the cut parameter, Delta E:
  DeltaE = kin.Get_DeltaE(data['WW_cut'][i])
  
  # The standard radiative corrections according to Maximon and Tjon:
  data.loc[i,'d_MTj'] = rc.delta_MaximonTjon(kinematics=kin, DeltaE=DeltaE)
  
  # An additional vacuum polarization correction:
  data.loc[i,'d_vac'] = rc.delta_vac_mu(kin.Get_QQ()) \
                      + rc.delta_vac_tau(kin.Get_QQ()) \
                      + rc.delta_vac_q(kin.Get_QQ())
  
  # An additional correction due to internal bremsstrahlung:
  data.loc[i,'d_IntBr'] = rc.delta_IntBr(DeltaE=DeltaE, kinematics=kin, ff=ff)
  
  # The external bremsstrahlung correction:
  data.loc[i,'d_ExtBr'] = rc.delta_ExtBr(DeltaE=DeltaE, kinematics=kin,
                                         t_i=0.01*data['t_i'][i], t_f=0.01*data['t_f'][i],
                                         mat_i=mat, mat_f=mat, ff=ff)
  
  # The correction factor due to ionization losses:
  data.loc[i,'C_L'] = rc.C_Landau(DeltaE=DeltaE, kinematics=kin,
                                  t_i=0.01*data['t_i'][i], t_f=0.01*data['t_f'][i],
                                  mat_i=mat, mat_f=mat)

# Old radiative correction factors:
data['C_old'] = np.exp(data['d_int1'] + data['d_int2'] + data['d_ext'])

# New radiative correction factors:
data['C_new'] = np.exp(data['d_MTj'] + data['d_vac'] + data['d_IntBr'] + data['d_ExtBr'])*data['C_L']

# Applying the new radiative corrections:
data['C_ratio'] = data['C_old']/data['C_new']
data['sigma'] = data['sigma']*data['C_ratio']

# Saving results to the file "results_table.csv":
np.savetxt('results_table.csv',
           data[['set','QQ','epsilon','d_MTj','d_vac','d_IntBr','d_ExtBr','C_L','C_new','C_ratio','sigma','estat','esyst','enorm']],
           fmt='%i,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.2f,%.2f,%.2f',
           header='set,QQ,epsilon,d_MTj,d_vac,d_IntBr,d_ExtBr,C_L,C_new,C_ratio,sigma,estat,esyst,enorm', comments='')


"""Reduced cross sections and their uncertainties"""

for i in range(N1+N2+N3):
  # Setting the nominal kinematics using the Q^2 and epsilon values:
  kin.Set_QQ_epsilon(QQ=data['QQ'][i], epsilon=data['epsilon'][i])
  
  # Reduced cross sections (dimensionless):
  data.loc[i,'sigma_red'] = data['sigma'][i]*kin.Get_epsilon()*(1. + kin.Get_tau())/rc.GeVToNb(kin.Get_sigma_Mott())

  # Normalization to the dipole form factor:
  data.loc[i,'sigma_red'] = data['sigma_red'][i]/(rc.GD(kin.Get_QQ()))**2
  
  # Combined statistical and systematic uncertainties:
  data.loc[i,'e_sigma_red'] = 0.01*np.sqrt(data['estat'][i]**2 + data['esyst'][i]**2)*data['sigma_red'][i]
  
  # Values of tau:
  data.loc[i,'tau'] = kin.Get_tau()


"""Minimization of the chi-square function using linear algebra.
   The vector x of the best-fit parameters is obtained
   after solving the matrix equation A*x = b.
"""

# An auxiliary matrix X (its columns correspond to the partial derivatives
# of the chi-square function with respect to n1, n2, n3, a1, a2, a3, b1, b2, and b3):
X = np.zeros((N1+N2+N3,9))
X[:N1,0] = data['sigma_red'][:N1]/data['e_sigma_red'][:N1]
X[N1:N1+N2,1] = data['sigma_red'][N1:N1+N2]/data['e_sigma_red'][N1:N1+N2]
X[N1+N2:,2] = data['sigma_red'][N1+N2:]/data['e_sigma_red'][N1+N2:]
X[:,3] = data['epsilon']*data['tau']/data['e_sigma_red']
X[:,4] = X[:,3]*data['tau']
X[:,5] = X[:,4]*data['tau']
X[:,6] = ((rc.mu*data['tau'])**2)/data['e_sigma_red']
X[:,7] = X[:,6]*data['tau']
X[:,8] = X[:,7]*data['tau']

# The coefficient matrix A:
A = np.dot(X.T, X)
for i in range(3):
  A[i,i] = A[i,i] + 1./(0.01*norm_uncert[i])**2

# The covariance matrix (inverse to A):
Cov = np.linalg.inv(A)

# The vector b:
y = (data['epsilon'] + (rc.mu**2)*data['tau'])/data['e_sigma_red']
b = np.dot(X.T, y)
for i in range(3):
  b[i] = b[i] + 1./(0.01*norm_uncert[i])**2

# The vector x of the best-fit parameters:
x = np.dot(Cov, b)

# Printing the best-fit parameters found:
print('Best-fit parameters:')
variables = ['n1', 'n2', 'n3', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3']
for i in range(9):
  print(variables[i] + ' = {0: .3f} +/- {1:.3f}'.format(x[i], np.sqrt(Cov[i,i])))
print('The best-fit parameters and the covariance matrix')
print('are output to the file "results_fit.csv".\n')

# Saving the best-fit parameters (the first line)
# and the covariance matrix (the next 9 lines) to the file "results_fit.csv":
np.savetxt('results_fit.csv', np.vstack((x, Cov)), delimiter=',',
	   fmt='%+.3e', header='n1,n2,n3,a1,a2,a3,b1,b2,b3', comments='')

# Calculating and printing the chi-square value achieved:
Chisq = np.dot((y - np.dot(X, x)).T, y - np.dot(X, x))
for i in range(3):
  Chisq = Chisq + ((x[i] - 1)/(0.01*norm_uncert[i]))**2
print('Chi-square value: ' + str(round(Chisq, 1)) + ' for ' + str(N1+N2+N3-9) + ' degrees of freedom')

