"""
Module for accessing precipitation model
"""

import cgm_model_interface as CMI
from astropy.cosmology import Planck15 as cosmo
from astropy import units as un, constants as cons
import numpy as np
from scipy.integrate import odeint, quad

X = 0.75     #hydrogen mass fraction
mu = 4/(3+5*X) #mean molecular weight
mue = 2/(1+X) #mean molecular weight per electron
muh = 1/X #mean molecular weight per hydrogen 
mp = cons.m_p.to('g') #proton mass
kb = cons.k_B.to('erg / K') #Boltzmann constant

class pNFW(CMI.Model):
	"""interface for precipitation model"""
	def __init__(self,potential,cooling,metallicity,boundary,tcool_tff,z):
		"""
		Precipitation model specific parameters
		_______________________________________

		tcool_tff: ratio of gas cooling to free-fall time
		z: redshift of the galaxy
		"""
		self.tcool_tff = tcool_tff
		self.z = z
		self.rhocz = cosmo.critical_density(self.z)
		self.ne200 = (200*self.rhocz*cosmo.Ob0/cosmo.Om0/mue/mp).to('cm**-3')
		self.potential = potential
		self.cooling = cooling
		self.metallicity = metallicity
		self.boundary = boundary
		self.R200c = self.potential.r200()
		self.rout, self.tout = boundary.outer_radius(), boundary.outer_temperature() #outer boundary and temperature conditions
	def Tphi(self,r):
		"""
		The gravitational temperature associated with vc
		"""
		return (0.5*mu*mp*self.potential.vc(r)**2).to('erg') #ergs
	def baseline_entropy_profile(self,r):
		return 1.32*self.Tphi(self.R200c)*(r/self.R200c)**1.1/self.ne200**(2/3)
	def precipitation_entropy_profile(self,r):
		return (np.cbrt(2.*mu*mp*(self.tcool_tff*self.cooling.LAMBDA((2*self.Tphi(r)/kb).value,self.metallicity.Z(r))*r/3.)**2)).to('cm**2*erg')
	def get_entropy_profile(self,r):
		"""
		Entropy profile: see Eqn. 4, 5 and 6 from Singh+21
		Kbase: produced by cosmological structure formation
		Kpre: determined by choosing a value for the ratio tcool/tff
		"""
		return self.baseline_entropy_profile(r)+self.precipitation_entropy_profile(r)
	def dHSE(self,Pe,r_inv):
		"""
		Integrand for solving HSE
		"""
		r_inv = [r_inv]/un.kpc
		Pe = Pe*un.erg/un.cm**3
		return 2*self.Tphi(1/r_inv)/r_inv*(Pe/self.get_entropy_profile(1/r_inv))**(3/5)
	def get_electron_thermal_pressure_profile(self,r):
		"""
		precipitation-limited thermal electron pressure profile
		r: input radii in kpc
		returns pressure in ergs/cm^3
		"""
		r_inv = 1/r[::-1]
		y0 = np.sqrt(self.tout.to('erg')**5/self.get_entropy_profile(self.rout)**3)
		res = odeint(self.dHSE,y0=y0,t=r_inv)[::-1]
		return res.T[0]*un.erg/un.cm**3
	def get_temperature_profile(self,r):
		"""
		precipitation-limited temperature profile
		r: input radii in kpc
		returns temperature in keV
		"""
		return (self.get_electron_thermal_pressure_profile(r)**(2/5)*self.get_entropy_profile(r)**(3/5)).to('keV')
	def get_electron_density_profile(self,r):
		"""
		precipitation-limited electron density profile
		r: input radii in kpc
		returns density in cm^-3
		"""
		return (self.get_temperature_profile(r).to('erg')/self.get_entropy_profile(r))**(3/2)
	def get_nH(self,r):
		"""
		precipitation-limited hydrogen density profile
		r: input radii in kpc
		returns density in cm^-3        
		"""
		return self.get_electron_density_profile(r)*mue/muh
	def get_gas_thermal_pressure_profile(self,r):
		"""
		precipitation-limited thermal gas pressure profile
		r: input in kpc
		returns pressure in ergs/cm^3
		"""
		return self.get_electron_thermal_pressure_profile(r)*mue/mu
	def get_tcool2tff(self,r):
		"""
		returns the ratio of gas cooling time to free fall time which is also the input parameter of the model
		"""
		return self.tcool_tff
	def get_gas_mass_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc
		drp = rp[1:]-rp[:-1]
		dMgas = self.get_electron_density_profile(rp[1:])*rp[1:]**2*drp
		mgass = (np.nancumsum(dMgas)*mue*mp*4.*np.pi).to('M_sun')
		return np.interp(r,rp[1:],mgass)
	def get_thermal_energy_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc
		drp = rp[1:]-rp[:-1]
		dEth = self.get_gas_thermal_pressure_profile(rp[1:])*rp[1:]**2*drp
		ethermal = (3./2.*np.nancumsum(dEth)*4.*np.pi).to('erg')
		return np.interp(r,rp[1:],ethermal)
