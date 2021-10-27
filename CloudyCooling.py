"""
Module for providing gas cooling function using Cloudy tables
"""

CoolingTableDir = 'CoolingTables/'
import cgm_model_interface as CMI
import pickle
import numpy as np
from astropy import units as un
from scipy import interpolate

class CIE(CMI.Cooling):
	"""
	Cooling function for the collisional ionisation equilibrium
	"""	
	def __init__(self):
		CoolingTab = pickle.load(open(CoolingTableDir+'cool_eff_CIE/cool_eff_CIE.pkl', 'rb'))
		self.f_Cooling = interpolate.RegularGridInterpolator((CoolingTab['Temperature'],CoolingTab['Metallicity']),CoolingTab['Cooling_Eff'],bounds_error=False, fill_value=None)
	def LAMBDA(self,T,Z):
		"""
		T: gas temperature in kelvin
		Z: metallicity
		returns cooling function in the units erg cm^3/s 
		"""
		return self.f_Cooling((T,Z))*un.erg*un.cm**3/un.s
