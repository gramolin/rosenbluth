"""Radiative corrections to electron-proton scattering: a Python library
   
   Copyright (c) Alexander Gramolin, 2016
   https://github.com/gramolin/rosenbluth/
"""

import numpy as np
from scipy import special
from scipy import integrate


"""Some mathematical and physical constants"""

pi = np.pi
m_e = 0.510998928e-3    # Electron mass (in GeV)
m_mu = 0.1056583715     # Muon mass (in GeV)
m_tau = 1.77682         # Tau lepton mass (in GeV)
m_p = 0.938272046       # Proton mass (in GeV)
mu = 2.792847356        # Magnetic moment of the proton
alpha = 1/137.035999074 # Fine-structure constant
r_e = 2.8179403267e-13  # Classical electron radius (in cm)
N_A = 6.02214129e23     # Avogadro constant (in 1/mol)


"""Unit conversion"""

def DegToRad(degree):
  """Degrees to radians conversion"""
  return degree*pi/180.

def RadToDeg(radian):
  """Radians to degrees conversion"""
  return radian*180./pi

def GeVToNb(GeV):
  """GeV^{-2} to nanobarns conversion"""
  return GeV*389379.338 # 1 GeV^{-2} = 389379.338 nanobarn

def NbToGeV(nb):
  """Nanobarns to GeV^{-2} conversion"""
  return nb/389379.338 # 1 GeV^{-2} = 389379.338 nanobarn


"""Additional mathematical functions"""

def DiLog(arg):
  """Return the dilogarithm (Spence's function) defined as
     \Phi(x) = -\int_{0}^{x} \frac{\ln|1 - u|}{u} du.
     See https://en.wikipedia.org/wiki/Spence's_function"""
  if (arg <= 1):
    return special.spence(1. - arg)
  else:
    return (pi**2)/3. - (np.log(arg)**2)/2. - special.spence(1. - 1./arg)

def Landau(arg):
  """Return the Landau distribution function"""
  def integrand(u, arg):
    return np.exp(-u*np.log(u) - arg*u)*np.sin(pi*u)
  return integrate.quad(integrand, 0, np.Inf, args=arg, epsrel=1e-12)[0]/pi

def Landau_integral(arg):
  """Return the Landau distribution function integrated from arg to infinity"""
  def integrand(u, arg):
    return np.exp(-u*np.log(u) - arg*u)*np.sin(pi*u)/u
  return integrate.quad(integrand, 0, np.Inf, args=arg, epsrel=1e-12)[0]/pi

def Linear_fitter(x, y, errors):
  """Return the slope and intercept of the linear function
     fitting the data (arguments x and values y with the corresponding errors)"""
  e2 = errors**2
  slope = (sum(x/e2)*sum(y/e2) - sum(x*y/e2)*sum(1./e2))/(sum(x/e2)**2 - sum(x*x/e2)*sum(1./e2))
  intercept = (sum(x*y/e2) - slope*sum(x*x/e2))/sum(x/e2)
  return slope, intercept


"""A class defining kinematics"""

class Kinematics:
  """A class defining kinematics"""
  def __init__(self, Z=1, M=m_p, E1=0., theta=0.):
    self.Z = Z # Charge of the target nucleus (in units of the proton charge)
    self.M = M # Mass of the target nucleus (in GeV)
    self.E1 = E1 # Beam energy (in GeV)
    self.theta = theta # Electron scattering angle (in radians)
  
  def __repr__(self):
    return 'Kinematics: Z = ' + str(self.Z) \
         + ', M = ' + str(self.M) \
         + ' GeV, E1 = ' + str(self.E1) \
         + ' GeV, theta = ' + str(RadToDeg(self.theta)) + ' deg'
  
  def Set_Z(self, Z):
    """Set the charge of the target nucleus (in units of the proton charge).
       Note that in the case of positron scattering Z should be negative!"""
    self.Z = int(round(Z))
  
  def Set_M(self, M):
    """Set the mass of the target nucleus (in GeV)"""
    if (M > 0):
      self.M = M
    else:
      print('Invalid value of M!')
  
  def Set_A(self, A):
    """Set the target nucleus mass using the atomic mass of the material.
       A is the atomic mass of the target material (in g/mol)."""
    if (A > 0):
      self.M = 0.931494061*A - abs(self.Z)*m_e
    else:
      print('Invalid value of A!')
  
  def Set_E1_theta(self, E1, theta):
    """Set kinematics using the beam energy (GeV) and scattering angle (radians)"""
    if (E1 > 0 and theta > 0 and theta <= pi):
      self.E1 = E1       # Beam energy (in GeV)
      self.theta = theta # Electron scattering angle (in radians)
    else:
      print('Invalid kinematics!')
  
  def Set_E1_QQ(self, E1, QQ):
    """Set kinematics using the beam energy (GeV) and Q^2 value (GeV^2)"""
    if (E1 > 0 and QQ > 0 and QQ < 2.*self.M*E1):
      self.E1 = E1 # Beam energy (in GeV)
      # Electron scattering angle (in radians):
      self.theta = np.arccos(1. - self.M*QQ/(E1*(2.*self.M*E1 - QQ)))
    else:
      print('Invalid kinematics!')
  
  def Set_QQ_theta(self, QQ, theta):
    """Set kinematics using the Q^2 value (GeV^2) and scattering angle (radians)"""
    if (QQ > 0 and theta > 0 and theta <= pi):
      self.theta = theta # Electron scattering angle (in radians)
      # Beam energy (in GeV):
      self.E1 = (QQ + np.sqrt(QQ**2 + (8.*QQ*self.M**2)/(1. - np.cos(theta))))/(4.*self.M)
    else:
      print('Invalid kinematics!')
  
  def Set_QQ_epsilon(self, QQ, epsilon):
    """Set kinematics using the values of Q^2 (GeV^2) and epsilon (0 < epsilon < 1)"""
    if (QQ > 0 and epsilon > 0 and epsilon < 1):
      # Electron scattering angle (in radians):
      self.theta = np.arccos((QQ/(2.*self.M**2) - 1./epsilon + 3.)/(QQ/(2.*self.M**2) + 1./epsilon + 1.))
      # Beam energy (in GeV):
      self.E1 = (QQ + np.sqrt(QQ**2 + (8.*QQ*self.M**2)/(1. - np.cos(self.theta))))/(4.*self.M)
    else:
      print('Invalid kinematics!')
  
  def Get_Z(self):
    """Return the charge of the target nucleus (in units of the proton charge)"""
    return self.Z
  
  def Get_M(self):
    """Return the mass of the target nucleus (in GeV)"""
    return self.M
  
  def Get_E1(self):
    """Return the beam energy (in GeV)"""
    return self.E1
  
  def Get_theta(self):
    """Return the electron scattering angle (in radians)"""
    return self.theta
  
  def Get_E3(self):
    """Return the energy of the scattered electron (in GeV)"""
    return self.M*self.E1/(self.M + self.E1*(1. - np.cos(self.theta)))
  
  def Get_E4(self):
    """Return the full energy of the recoil nucleus (in GeV)"""
    return self.M + self.E1 - self.Get_E3()
  
  def Get_p4(self):
    """Return the momentum of the recoil nucleus (in GeV)"""
    return np.sqrt(self.Get_E4()**2 - self.M**2)
  
  def Get_beta4(self):
    """Return the speed of the recoil nucleus in units of c (dimensionless)"""
    return self.Get_p4()/self.Get_E4()
  
  def Get_x(self):
    """Return the value of x (dimensionless)"""
    return (self.Get_E4() + self.Get_p4())/self.M
  
  def Get_eta(self):
    """Return the value of eta (dimensionless)"""
    return 1. + self.E1*(1. - np.cos(self.theta))/self.M
  
  def Get_QQ(self):
    """Return the four-momentum transfer squared (in GeV^2)"""
    M = self.M
    E1 = self.E1
    theta = self.theta
    return 2.*M*(E1**2)*(1. - np.cos(theta))/(M + E1*(1. - np.cos(theta)))
  
  def Get_tau(self):
    """Return the value of tau (dimensionless)"""
    return self.Get_QQ()/(4.*self.M**2)
  
  def Get_epsilon(self):
    """Return the virtual-photon polarization parameter (dimensionless)"""
    return 1./(1. + 2.*(1. + self.Get_tau())*np.tan(0.5*self.theta)**2)
  
  def Get_omega1(self, E3):   
    """Return the energy of the bremsstrahlung photon
       emitted along the incident electron (in GeV).
       E3 is the energy of the scattered electron (in GeV)."""
    M = self.M
    E1 = self.E1
    theta = self.theta
    R = (M + E1*(1. - np.cos(theta)))/(M - E3*(1. - np.cos(theta)))
    return R*self.Get_omega3(E3)
  
  def Get_omega3(self, E3):
    """Return the energy of the bremsstrahlung photon
       emitted along the scattered electron (in GeV).
       E3 is the energy of the scattered electron (in GeV)."""
    return self.Get_E3() - E3
  
  def Get_DeltaE(self, WW):
    """Return Delta E, the cut on the scattered electron energy (in GeV).
       WW is the cut on the missing mass squared (in GeV^2)."""
    if (WW >= self.M):
      return (WW - self.M**2)/(2.*self.Get_eta()*self.M)
    else:
      print('Invalid value of WW!')
  
  def Get_sigma_Mott(self):
    """Return the Mott differential cross section (in GeV^{-2}*sr^{-1}).
       Note that this includes the recoil factor 1/eta."""
    return (self.Z*alpha*np.cos(0.5*self.theta)/(2.*self.E1))**2 / (self.Get_eta()*np.sin(0.5*self.theta)**4)


"""Functions to calculate the proton electromagnetic form factors"""

def GD(QQ):
  """Return the dipole form factor (G_D, dimensionless)"""
  return 1./(1. + QQ/0.71)**2

def GE_Kelly(QQ, a1=-0.24, b1=10.98, b2=12.82, b3=21.97):
  """Return the electric form factor (G_E, dimensionless).
     Kelly parameterization is used, see Phys. Rev. C 70, 068202 (2004)."""
  tau = QQ/(2.*m_p)**2
  return (1. + a1*tau)/(1. + b1*tau + b2*tau**2 + b3*tau**3)

def GM_Kelly(QQ, a1=0.12, b1=10.97, b2=18.86, b3=6.55):
  """Return the magnetic form factor (G_M, dimensionless).
     Kelly parameterization is used, see Phys. Rev. C 70, 068202 (2004)."""
  tau = QQ/(2.*m_p)**2
  return mu*(1. + a1*tau)/(1. + b1*tau + b2*tau**2 + b3*tau**3)

def GE_fit(QQ, a1=0.197, a2=0.703, a3=-0.454):
  """Return the electric form factor (G_E, dimensionless).
     The custom fit is used."""
  tau = QQ/(2.*m_p)**2
  return GD(QQ)*np.sqrt(abs(1. - a1*tau - a2*tau**2 - a3*tau**3))

def GM_fit(QQ, b1=-0.444, b2=0.397, b3=-0.081):
  """Return the magnetic form factor (G_M, dimensionless).
     The custom fit is used."""
  tau = QQ/(2.*m_p)**2
  return mu*GD(QQ)*np.sqrt(abs(1. - b1*tau - b2*tau**2 - b3*tau**3))

def ff_labels(label):
  """Return the model number (integer) given its label (string)"""
  return {0: 0, '0': 0, 'dipole': 0, 'Dipole': 0, 'DIPOLE': 0,
          1: 1, '1': 1, 'kelly':  1, 'Kelly':  1, 'KELLY':  1,
          2: 2, '2': 2, 'custom': 2, 'Custom': 2, 'CUSTOM': 2
         }.get(label, 0) # Default model: 0


"""A class defining the proton electromagnetic form factors"""

class FormFactors:
  """A class defining the proton electromagnetic form factors"""
  def __init__(self, model=0):
    self.model = ff_labels(model)
  
  def __repr__(self):
    return 'Form factor model: ' + str(self.model)
  
  def Set_model(self, model):
    """Set the form factor model"""
    self.model = ff_labels(model)
    
  def Get_model(self):
    """Return the form factor model (integer)"""
    return self.model
  
  def Get_GE(self, QQ):
    """Return the Sachs electric form factor (G_E, dimensionless)"""
    if (self.model == 0): # Dipole parameterization
      return GD(QQ)
    elif (self.model == 1): # Kelly parameterization
      return GE_Kelly(QQ)
    elif (self.model == 2): # Custom fit
      return GE_fit(QQ)
  
  def Get_GM(self, QQ):
    """Return the Sachs magnetic form factor (G_M, dimensionless)"""
    if (self.model == 0): # Dipole parameterization
      return mu*GD(QQ)
    elif (self.model == 1): # Kelly parameterization
      return GM_Kelly(QQ)
    elif (self.model == 2): # Custom fit
      return GM_fit(QQ)


"""A class defining target materials"""

class Material:
  """A class defining target materials"""
  def __init__(self, Z=1, A=1.00794, X0=63.04): # Default material: hydrogen
    self.Z = int(round(Z)) # Atomic number of the material (dimensionless)
    self.A = A # Atomic mass of the material (in g/mol)
    self.X0 = X0 # Radiation length of the material (g/cm^2)
  
  def __repr__(self):
    return 'Material: Z = ' + str(self.Z) + ', A = ' + str(self.A) \
         + ' g/mol, X0 = ' + str(self.X0) + ' g/cm^2'
  
  def Set_Z_A_X0(self, Z, A, X0):
    """Set Z, A, and X0 for the material"""
    if (Z >= 1 and A > 0 and X0 > 0):
      self.Z = int(round(Z))
      self.A = A
      self.X0 = X0
    else:
      print('Invalid parameters!')
  
  def Get_Z(self):
    """Return the atomic number of the material (dimensionless)"""
    return self.Z
  
  def Get_A(self):
    """Return the atomic mass of the material (in g/mol)"""
    return self.A
  
  def Get_X0(self):
    """Return the radiation length of the material (in g/cm^2)"""
    return self.X0
  
  def Get_b(self):
    """Return b (dimensionless)"""
    return 4./3. + (4./9.)*alpha*(r_e**2)*N_A*self.Z*(self.Z + 1)*self.X0/self.A
  
  def Get_xi(self):
    """Return xi (in GeV)"""
    return 2.*pi*m_e*(r_e**2)*N_A*self.Z*self.X0/self.A


"""Radiative corrections according to Mo and Tsai
   
   See Rev. Mod. Phys. 41, 205 (1969), Eq. (II.6)
"""

def Z0_MoTsai(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^0.
     See Rev. Mod. Phys. 41, 205 (1969), Eq. (II.6)."""
  QQ = kinematics.Get_QQ()
  E1 = kinematics.Get_E1()
  E3 = kinematics.Get_E3()
  eta = kinematics.Get_eta()
  return -(alpha/pi)*(28./9. - 13.*np.log(QQ/m_e**2)/6. + (np.log(QQ/m_e**2) - 1.)*(2.*np.log(E1/DeltaE) - 3.*np.log(eta)) - DiLog((E3 - E1)/E3) - DiLog((E1 - E3)/E1))

def Z1_MoTsai(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^1.
     See Rev. Mod. Phys. 41, 205 (1969), Eq. (II.6)."""
  M = kinematics.Get_M()
  E1 = kinematics.Get_E1()
  E3 = kinematics.Get_E3()
  E4 = kinematics.Get_E4()
  eta = kinematics.Get_eta()
  return -(alpha/pi)*kinematics.Get_Z()*(2.*np.log(eta)*(2.*np.log(E1/DeltaE) - 3.*np.log(eta)) + DiLog(-(M - E3)/E1) - DiLog(M*(M - E3)/(2.*E3*E4 - M*E1)) + DiLog(2.*E3*(M - E3)/(2.*E3*E4 - M*E1)) + np.log(np.fabs((2.*E3*E4 - M*E1)/(E1*(M - 2.*E3))))*np.log(0.5*M/E3) - DiLog(-(E4 - E3)/E3) + DiLog(M*(E4 - E3)/(2.*E1*E4 - M*E3)) - DiLog(2.*E1*(E4 - E3)/(2.*E1*E4 - M*E3)) - np.log(np.fabs((2.*E1*E4 - M*E3)/(E3*(M - 2.*E1))))*np.log(0.5*M/E1) - DiLog(-(M - E1)/E1) + DiLog((M - E1)/E1) - DiLog(2.*(M - E1)/M) - np.log(np.fabs(M/(2.*E1 - M)))*np.log(0.5*M/E1) + DiLog(-(M - E3)/E3) - DiLog((M - E3)/E3) + DiLog(2.*(M - E3)/M) + np.log(np.fabs(M/(2.*E3 - M)))*np.log(0.5*M/E3))

def Z2_MoTsai(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^2.
     See Rev. Mod. Phys. 41, 205 (1969), Eq. (II.6)."""
  M = kinematics.Get_M()
  E4 = kinematics.Get_E4()
  eta = kinematics.Get_eta()
  beta4 = kinematics.Get_beta4()
  return -(alpha/pi)*(kinematics.Get_Z()**2)*(-np.log(E4/M) + np.log(M/(eta*DeltaE))*(np.log((1. + beta4)/(1. - beta4))/beta4 - 2.) + (1./beta4)*(0.5*np.log((1. + beta4)/(1. - beta4))*np.log(0.5*(E4 + M)/M) - DiLog(-np.sqrt((E4 - M)/(E4 + M))*np.sqrt((1. + beta4)/(1. - beta4))) + DiLog(np.sqrt((E4 - M)/(E4 + M))*np.sqrt((1. - beta4)/(1. + beta4))) - DiLog(np.sqrt((E4 - M)/(E4 + M))) + DiLog(-np.sqrt((E4 - M)/(E4 + M)))))

def delta_MoTsai(kinematics, DeltaE):
  """Return the full correction according to Mo and Tsai (dimensionless).
     See Rev. Mod. Phys. 41, 205 (1969), Eq. (II.6).
     kinematics is an object of the Kinematics class.
     DeltaE is the cut on the scattered electron energy (in GeV)."""
  return Z0_MoTsai(kinematics, DeltaE) + Z1_MoTsai(kinematics, DeltaE) + Z2_MoTsai(kinematics, DeltaE)


"""Radiative corrections according to Maximon and Tjon

   See Phys. Rev. C 62, 054320 (2000), Eq. (5.2)
"""

def Z0_MaximonTjon(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^0.
     See Phys. Rev. C 62, 054320 (2000), Eq. (5.2)."""
  QQ = kinematics.Get_QQ()
  E1 = kinematics.Get_E1()
  E3 = kinematics.Get_E3()
  theta = kinematics.Get_theta()
  eta = kinematics.Get_eta()
  return (alpha/pi)*(13.*np.log(QQ/m_e**2)/6. - 28./9. - (np.log(QQ/m_e**2) - 1.)*np.log(4.*E1*E3/(2.*eta*DeltaE)**2) - 0.5*np.log(eta)**2 + DiLog(np.cos(0.5*theta)**2) - (pi**2)/6.)

def Z1_MaximonTjon(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^1.
     See Phys. Rev. C 62, 054320 (2000), Eq. (5.2)."""
  QQ = kinematics.Get_QQ()
  eta = kinematics.Get_eta()
  x = kinematics.Get_x()
  return (2.*alpha/pi)*kinematics.Get_Z()*(-np.log(eta)*np.log(QQ*x/(2.*eta*DeltaE)**2) + DiLog(1. - eta/x) - DiLog(1. - 1./(eta*x)))

def Z2_MaximonTjon(kinematics, DeltaE):
  """Return the part of the radiative correction proportional to Z^2.
     See Phys. Rev. C 62, 054320 (2000), Eq. (5.2)."""
  M = kinematics.Get_M()
  QQ = kinematics.Get_QQ()
  E4 = kinematics.Get_E4()
  p4 = kinematics.Get_p4()
  eta = kinematics.Get_eta()
  x = kinematics.Get_x()
  return (alpha/pi)*(kinematics.Get_Z()**2)*((E4/p4)*(-0.5*np.log(x)**2 - np.log(x)*np.log((QQ + 4.*M**2)/M**2) + np.log(x) - DiLog(1. - 1./x**2) + 2.*DiLog(-1./x) + (pi**2)/6.) - ((E4/p4)*np.log(x) - 1.)*np.log((M/(2.*eta*DeltaE))**2) + 1.)
  
def delta_MaximonTjon(kinematics, DeltaE):
  """Return the full correction according to Maximon and Tjon (dimensionless).
     See Phys. Rev. C 62, 054320 (2000), Eq. (5.2).
     kinematics is an object of the Kinematics class.
     DeltaE is the cut on the scattered electron energy (in GeV)."""
  return Z0_MaximonTjon(kinematics, DeltaE) + Z1_MaximonTjon(kinematics, DeltaE) + Z2_MaximonTjon(kinematics, DeltaE)

def delta_2gamma(kinematics):
  """Return the difference between the Maximon-Tjon and Mo-Tsai expressions
     for the soft two-photon exchange terms (dimensionless)"""
  M = kinematics.Get_M()
  QQ = kinematics.Get_QQ()
  E1 = kinematics.Get_E1()
  E3 = kinematics.Get_E3()
  eta = kinematics.Get_eta()
  return -(alpha/pi)*kinematics.Get_Z()*(np.log(eta)*np.log((QQ**2)/(4.*(M**2)*E1*E3)) + 2.*DiLog(1. - 0.5*M/E1) - 2.*DiLog(1. - 0.5*M/E3))


"""Radiative corrections due to vacuum polarization"""

def delta_vac_lepton(QQ, mass=m_e):
  """Return the vacuum polarization correction due to lepton loops.
     mass is the electron/muon/tau mass (in GeV)."""
  return (2.*alpha/(3.*pi))*(-5./3. + (4.*mass**2)/QQ + (1. - (2.*mass**2)/QQ)*np.sqrt(1. + (4.*mass**2)/QQ)*np.log((QQ/(2.*mass)**2)*(1. + np.sqrt(1. + (4.*mass**2)/QQ))**2))

def delta_vac_mu(QQ):
  """Return the vacuum polarization correction due to muon loops"""
  return delta_vac_lepton(QQ, m_mu)

def delta_vac_tau(QQ):
  """Return the vacuum polarization correction due to tau loops"""
  return delta_vac_lepton(QQ, m_tau)

def delta_vac_q(QQ):
  """Return the vacuum polarization correction due to quark loops
     (hadronic part of the vacuum polarization)"""
  return 0.002*(1.513 + 2.822*np.log(1. + 1.218*QQ))


"""Rosenbluth differential cross section"""

def sigma_Rosenbluth(E1, theta, ff=FormFactors()):
  """Return the Rosenbluth differential cross section (in GeV^{-2}*sr^{-1}).
     E1 is the beam energy (in GeV).
     theta is the scattering angle (in radians).
     ff is an object of the FormFactors class."""
  kin = Kinematics(E1=E1, theta=theta)
  QQ = kin.Get_QQ()
  tau = kin.Get_tau()
  epsilon = kin.Get_epsilon()
  G_E = ff.Get_GE(QQ)
  G_M = ff.Get_GM(QQ)
  return (1./(epsilon*(1. + tau)))*(epsilon*G_E**2 + tau*G_M**2)*kin.Get_sigma_Mott()


"""The standard soft-photon description of internal bremsstrahlung"""

def sigma_IntBr_soft(E3, kinematics, ff=FormFactors()):
  """Return the differential cross section (in GeV^{-3}*sr^{-1})
     of internal bremsstrahlung in the soft-photon approximation.
     E3 is the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     ff is an object of the FormFactors class."""
  Z = kinematics.Get_Z()
  QQ = kinematics.Get_QQ()
  E1 = kinematics.Get_E1()
  E4 = kinematics.Get_E4()
  p4 = kinematics.Get_p4()
  theta = kinematics.Get_theta()
  eta = kinematics.Get_eta()
  x = kinematics.Get_x()
  return (2.*alpha/pi)*(1./(kinematics.Get_E3() - E3))*(np.log(QQ/m_e**2) - 1. + 2.*Z*np.log(eta) + (Z**2)*(E4*np.log(x)/p4 - 1.))*sigma_Rosenbluth(E1=E1, theta=theta, ff=ff)


"""More accurate description of hard internal bremsstrahlung"""

def sigma_IntBr(E3, kinematics, ff=FormFactors()):
  """Return the differential cross section (in GeV^{-3}*sr^{-1})
     providing a better description of hard internal bremsstrahlung.
     E3 is the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     ff is an object of the FormFactors class."""
  M = kinematics.Get_M()
  E1 = kinematics.Get_E1()
  theta = kinematics.Get_theta()
  omega1 = kinematics.Get_omega1(E3)
  omega3 = kinematics.Get_omega3(E3)
  x1 = (E1 - omega1)/E1
  x3 = E3/(E3 + omega3)
  t1 = (alpha/pi)*(0.5*(1. + x1**2)*np.log(2.*E1*E3*(1. - np.cos(theta))/m_e**2) - x1)
  t3 = (alpha/pi)*(0.5*(1. + x3**2)*np.log(2.*E1*E3*(1. - np.cos(theta))/m_e**2) - x3)
  return ((M + (E1 - omega1)*(1. - np.cos(theta)))/(M - E3*(1. - np.cos(theta))))*(t1/omega1)*sigma_Rosenbluth(E1=E1-omega1, theta=theta, ff=ff) + (t3/omega3)*sigma_Rosenbluth(E1=E1, theta=theta, ff=ff)

def delta_IntBr(DeltaE, kinematics, ff=FormFactors(), dE=1e-4):
  """Return the additional correction due to internal bremsstrahlung
     improving the description of the radiative tail (dimensionless).
     DeltaE is the cut on the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     ff is an object of the FormFactors class."""
  QQ = kinematics.Get_QQ()
  E1 = kinematics.Get_E1()
  E3_el = kinematics.Get_E3()
  theta = kinematics.Get_theta()
  def integrand(E3, arg0, arg1, arg2):
    kin = Kinematics(E1=arg0, theta=arg1)
    ff = FormFactors(arg2)
    return sigma_IntBr(E3=E3, kinematics=kin, ff=ff)
  return (2.*alpha/pi)*(np.log(QQ/m_e**2) - 1.)*np.log(dE/DeltaE) + integrate.quad(integrand, E3_el-DeltaE, E3_el-dE, args=(E1, theta, ff.Get_model()), epsrel=1e-12)[0]/sigma_Rosenbluth(E1=E1, theta=theta, ff=ff)


"""Radiative correction due to external bremsstrahlung"""

def phi(arg):
  """Return phi, the function describing the shape
     of the bremsstrahlung spectrum (dimensionless)"""
  return 1. - arg + 0.75*arg**2

def sigma_ExtBr(E3, kinematics, t_i, t_f, mat_i=Material(), mat_f=Material(), ff=FormFactors()):
  """Return the differential cross section of
     external bremsstrahlung (in GeV^{-3}*sr^{-1}).
     E3 is the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     t_i and t_f are the thicknesses of the materials traversed
     by the incident and scattered electrons (as fractions of X0).
     mat_i and mat_f are objects of the Material class.
     ff is an object of the FormFactors class."""
  M = kinematics.Get_M()
  E1 = kinematics.Get_E1()
  E3_el = kinematics.Get_E3()
  theta = kinematics.Get_theta()
  omega1 = kinematics.Get_omega1(E3)
  omega3 = kinematics.Get_omega3(E3)
  b_i = mat_i.Get_b()
  b_f = mat_f.Get_b()
  return (1./special.gamma(1. + b_i*t_i))*(1./special.gamma(1. + b_f*t_f))*((omega1/E1)**(b_i*t_i))*((omega3/E3_el)**(b_f*t_f))*(((M + (E1 - omega1)*(1. - np.cos(theta)))/(M - E3*(1. - np.cos(theta))))*(b_i*t_i/omega1)*phi(omega1/E1)*sigma_Rosenbluth(E1=E1-omega1, theta=theta, ff=ff) + (b_f*t_f/omega3)*phi(omega3/E3_el)*sigma_Rosenbluth(E1=E1, theta=theta, ff=ff))

def delta_ExtBr(DeltaE, kinematics, t_i, t_f, mat_i=Material(), mat_f=Material(), ff=FormFactors(), dE=1e-4):
  """Return the radiative correction due to external bremsstrahlung
     in the target materials (dimensionless).
     DeltaE is the cut on the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     t_i and t_f are the thicknesses of the materials traversed
     by the incident and scattered electrons (as fractions of X0).
     mat_i and mat_f are objects of the Material class.
     ff is an object of the FormFactors class."""
  E1 = kinematics.Get_E1()
  E3_el = kinematics.Get_E3()
  theta = kinematics.Get_theta()
  eta = kinematics.Get_eta()
  b_i = mat_i.Get_b()
  b_f = mat_f.Get_b()
  def integrand(E3, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    kin = Kinematics(E1=arg0, theta=arg1)
    ff = FormFactors(arg2)
    mat_i = Material(Z=arg3, A=arg4, X0=arg5)
    mat_f = Material(Z=arg6, A=arg7, X0=arg8)
    return sigma_ExtBr(E3=E3, kinematics=kin, ff=ff, mat_i=mat_i, mat_f=mat_f, t_i=arg9, t_f=arg10)
  return np.log((1./special.gamma(1. + b_i*t_i))*(1./special.gamma(1. + b_f*t_f))*(((eta**2)*dE/E1)**(b_i*t_i))*((dE/E3_el)**(b_f*t_f)) + integrate.quad(integrand, E3_el-DeltaE, E3_el-dE, args=(E1, theta, ff.Get_model(), mat_i.Get_Z(), mat_i.Get_A(), mat_i.Get_X0(), mat_f.Get_Z(), mat_f.Get_A(), mat_f.Get_X0(), t_i, t_f), epsrel=1e-12)[0]/sigma_Rosenbluth(E1=E1, theta=theta, ff=ff))


"""Radiative correction due to ionization losses (Landau straggling)"""

def C_Landau(DeltaE, kinematics, t_i, t_f, mat_i=Material(), mat_f=Material()):
  """Return the correction factor due to ionization losses
     in the target materials (dimensionless).
     DeltaE is the cut on the scattered electron energy (in GeV).
     kinematics is an object of the Kinematics class.
     t_i and t_f are the thicknesses of the materials traversed
     by the incident and scattered electrons (as fractions of X0).
     mat_i and mat_f are objects of the Material class."""
  eta = kinematics.Get_eta()
  xi_i = mat_i.Get_xi()*t_i
  xi_f = mat_f.Get_xi()*t_f
  return 1. - Landau_integral((eta**2)*DeltaE/xi_i) - Landau_integral(DeltaE/xi_f)

