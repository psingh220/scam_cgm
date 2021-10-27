"""
Module for providing boundary conditions to the cgm_model_interface module
"""

import cgm_model_interface as CMI
import numpy as np
from astropy import units as un, constants as cons

class Boundary(CMI.Boundary_Conditions):
	def __init__(self,Rout=None,nH=None,T=None,Pressure=None,Phi=None,Mdot_in=None,potential=None,Tnorm=None):
		"""
		Rout: outer boundary in kpc 
		nH: gas density at Rout in cm^-3
		T: gas temperature at Rout in keV
		Pressure: gas pressure at Rout in keV cm^-3
		Phi: gravitational potential at large distance set to zero for a bound state solution
		Mdot_in: mass inflow rate in Msun/year
		"""
		if Rout==None:
			#Use r200c and virial temperature from module HaloPotential as boundary conditions 
			#Temparature at bounday is Tnorm times virial temperature
			self.potential = potential
			self.Tnorm = 1 if Tnorm==None else Tnorm
			self.Rout = [self.potential.r200().value]*un.kpc
			self.T = [self.Tnorm*self.potential.T200().value]*un.keV
		else:
			self.Rout = Rout
			self.T = T
		self.nH = nH 
		if Pressure==None and nH!=None and T!=None:
			self.Pressure = self.nH*self.T
		else:
			self.Pressure = Pressure
		self.Phi = Phi
		self.Mdot_in = Mdot_in
	def outer_radius(self):
		return self.Rout
	def outer_density(self):
		return self.nH
	def outer_temperature(self):
		return self.T
	def outer_pressure(self):
		return self.Pressure
	def outer_phi(self):
		return self.Phi
	def outer_mdot(self):
		return self.Mdot_in
