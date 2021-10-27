"""
Module for providing metallicity to the cgm_model_interface module
"""

import cgm_model_interface as CMI
import numpy as np

class Uniform_Metallicity(CMI.Metallicity_profile):
	def __init__(self,Zuniform):
		"""
		Zuniform: Metallicity in the units of solar metallicity in numpy array format
		"""
		self.Zuniform = np.array([Zuniform])
	def Z(self,r):
		return self.Zuniform

class PowerLaw_Metallicity(CMI.Metallicity_profile):
	def __init__(self,Rcore,Zcore,slope):
		"""
		Rcore: Outer boundary in kpc 
		Zcore: Metallicity at Rcore in the units of solar metallicity
		slope: power law slope
		"""
		self.Rcore = Rcore
		self.Zcore = Zcore
		self.slope = slope
	def Z(self,r):
		return self.Zcore*(r/self.Rcore)**self.slope

class Two_PowerLaw_Metallicity(CMI.Metallicity_profile):
	def __init__(self,Rcore,Zcore,slope_in,slope_out):
		"""
		Rcore: Core radii in kpc 
		Zout: Metallicity at Rcore in the units of solar metallicity
		slope_in: power law slope inside the core
		slope_out: power law slope outside the core
		"""
		self.Rcore = Rcore
		self.Zcore = Zcore
		self.slope_in = slope_in
		self.slope_out = slope_out
	def Z(self,r):
		if r<Rcore:
			return self.Zcore*(r/self.Rcore)**self.slope_in
		else:
			return self.Zcore*(r/self.Rcore)**self.slope_out

