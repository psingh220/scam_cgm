"""
Module for providing gas cooling function using Cloudy tables
"""

CoolingTableDir = 'CoolingTables/'
CoolingTableDir_Wiersma = '/home/jonathan/Dropbox/jonathanmain/CGM/rapidCoolingCGM/published_pysrc/cooling/CoolingTables/'
import cgm_model_interface as CMI
import pickle
import glob,h5py
import scipy, numpy as np
from scipy import integrate, interpolate
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons

class CIE(CMI.Cooling):
	"""
	Cooling function for the collisional ionisation equilibrium
	"""	
	def __init__(self):
		CoolingTab = pickle.load(open(CoolingTableDir+'cool_eff_CIE/cool_eff_CIE.pkl', 'rb'))
		self.logTs = log(CoolingTab['Temperature'])
		self.f_Cooling = interpolate.RegularGridInterpolator((10.**self.logTs,CoolingTab['Metallicity']),CoolingTab['Cooling_Eff'],bounds_error=False, fill_value=None)
	def LAMBDA(self,T,Z,nH=None):
		"""
		T: gas temperature in kelvin
		Z: metallicity
		returns cooling function in the units erg cm^3/s 
		"""
		return self.f_Cooling((T,Z))*un.erg*un.cm**3/un.s
	def f_dlnLambda_dlnT(self,T,Z,nH=None):
		"""logarithmic derivative of cooling function with respect to T"""
		vals = log(self.LAMBDA(self.logTs,Z))
		dlnLambda_dlnTArr = np.gradient(vals,self.logTs)
		dlnLambda_dlnT_interpolation = interpolate.RegularGridInterpolator(self.logTs,dlnLambda_dlnTArr, bounds_error=False, fill_value=None)
		return dlnLambda_dlnT_interpolation(log(T.to('K').value))
	def f_dlnLambda_dlnrho(self,T,Z,nH=None):
		"""logarithmic derivative of cooling function with respect to rho"""
		return 0



"""
Module for providing the Wiersma et al. (2009) cooling functions to the cooling_flow module
"""

class Constant_Cooling(CMI.Cooling):
	def __init__(self,LAMBDA):
		self._LAMBDA = LAMBDA        
	def LAMBDA(self,T=None,Z=None,nH=None):
		"""cooling function"""
		return self._LAMBDA
	def f_dlnLambda_dlnT(self,T=None,Z=None,nH=None):
		"""logarithmic derivative of cooling function with respect to T"""
		return 0
	def f_dlnLambda_dlnrho(self,T=None,Z=None,nH=None):
		"""logarithmic derivative of cooling function with respect to rho"""
		return 0

class Wiersma_Cooling(CMI.Cooling):
	"""
	creates Wiersma+09 cooling function for given metallicity and redshift
	"""
	def __init__(self,Z2Zsun,z):
		fns = np.array(glob.glob(CoolingTableDir_Wiersma+'z_?.???.hdf5'))
		zs = np.array([float(fn[-10:-5]) for fn in fns])
		fn = fns[zs.argsort()][searchsortedclosest(sorted(zs), z)]

		f=h5py.File(fn,'r')

		He2Habundance = 10**-1.07 * (0.71553 + 0.28447*Z2Zsun) #Asplund+09, Groves+04
		X = (1 - 0.014*Z2Zsun) / (1.+4.*He2Habundance)
		Y = 4.*He2Habundance * X
		iHe = searchsortedclosest(f['Metal_free']['Helium_mass_fraction_bins'][:],Y)

		H_He_Cooling  = f['Metal_free']['Net_Cooling'][iHe,...]
		Tbins         = f['Metal_free']['Temperature_bins'][...]
		nHbins        = f['Metal_free']['Hydrogen_density_bins'][...]
		Metal_Cooling = f['Total_Metals']['Net_cooling'][...] * Z2Zsun    

		self.f_Cooling = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),
	                                                     Metal_Cooling+H_He_Cooling, 
	                                                bounds_error=False, fill_value=None)
		#### calculate gradients of cooling function
		X, Y = np.meshgrid(Tbins, nHbins, copy=False)
		dlogT = np.diff(log(Tbins))[0] 
		dlogn = np.diff(log(nHbins))[0] 
		vals = log(self.LAMBDA(X*un.K,Z2Zsun,Y*un.cm**-3).value)
		dlnLambda_dlnrhoArr, dlnLambda_dlnTArr = np.gradient(vals,dlogn, dlogT)    
		self.dlnLambda_dlnT_interpolation = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),dlnLambda_dlnTArr.T, bounds_error=False, fill_value=None)
		self.dlnLambda_dlnrho_interpolation = interpolate.RegularGridInterpolator((log(Tbins), log(nHbins)),dlnLambda_dlnrhoArr.T, bounds_error=False, fill_value=None)                
	def LAMBDA(self, T, Z=None, nH=None):
		"""cooling function"""
		return self.f_Cooling((log(T.to('K').value), log(nH.to('cm**-3').value))) * un.erg*un.cm**3/un.s
	def tcool(self,T,Z=None,nH=None):
		"""cooling time"""
		return 3.5 * cons.k_B * T / (nH * self.LAMBDA(T, Z,nH))
	def f_dlnLambda_dlnT(self, T, Z=None,nH=None):         
		"""logarithmic derivative of cooling function with respect to T"""
		return self.dlnLambda_dlnT_interpolation((log(T.to('K').value), log(nH.to('cm**-3').value)))
	def f_dlnLambda_dlnrho(self, T, Z=None,nH=None):
		"""logarithmic derivative of cooling function with respect to rho"""
		return self.dlnLambda_dlnrho_interpolation((log(T.to('K').value), log(nH.to('cm**-3').value)))




def searchsortedclosest(arr, val):
	if arr[0]<arr[1]:
		ind = np.searchsorted(arr,val)
		ind = minarray(ind, len(arr)-1)
		return maxarray(ind - (val - arr[maxarray(ind-1,0)] < arr[ind] - val),0)        
	else:
		ind = np.searchsorted(-arr,-val)
		ind = minarray(ind, len(arr)-1)
		return maxarray(ind - (-val + arr[maxarray(ind-1,0)] < -arr[ind] + val),0)        
def maxarray(arr, v):
	return arr + (arr<v)*(v-arr)
def minarray(arr, v):
	return arr + (arr>v)*(v-arr)

