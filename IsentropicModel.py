"""
Module for accessing isentropic model
"""

import cgm_model_interface as CMI
from astropy import units as un, constants as cons
import numpy as np
from scipy.integrate import odeint, quad

X = 0.75     #hydrogen mass fraction
mu = 4/(3+5*X) #mean molecular weight
mue = 2/(1+X) #mean molecular weight per electron
muh = 1/X #mean molecular weight per hydrogen 
mp = cons.m_p.to('g') #proton mass
kb = cons.k_B.to('erg / K') #Boltzmann constant
g1 = 5/3 # adiabatic index for thermal component
g2 = 4/3 # adiabatic index for non-thermal component

class isentropic(CMI.Model):
	"""interface for precipitation model"""
	def __init__(self,potential,cooling,metallicity,boundary,vturb,alpha_ntb):
		"""
		Isentropic model specific parameters
		____________________________________

		vturb: turbulent component
		alpha_nt: ratio of cosmic ray and magnetic field pressure to the thermal pressure (boundary condition)
		"""
		self.vturb = vturb.to('cm/s')
		self.alpha_ntb = alpha_ntb
		self.potential = potential
		self.cooling = cooling
		self.metallicity = metallicity
		self.boundary = boundary
		self.rout, self.tout, self.ngout = boundary.outer_radius(), boundary.outer_temperature(), (muh/mu)*boundary.outer_density()
		self.K_thermal = self.tout.to('erg')/self.ngout**(g1-1)/(mu*mp)**g1
		self.K_nonthermal = (self.alpha_ntb-1)*self.tout.to('erg')/self.ngout**(g2-1)/(mu*mp)**g2
	def dHSE(self,ng,r_inv):
		r_inv = [r_inv]/un.kpc
		ng = ng/un.cm**3
		tphi = (0.5*mu*mp*self.potential.vc(1/r_inv)**2).to('erg')
		return 2*tphi/((mu*mp)**2*r_inv*(self.vturb**2/(mu*mp*ng)+self.K_thermal*g1*(mu*mp*ng)**(g1-2)+self.K_nonthermal*g2*(mu*mp*ng)**(g2-2)))
	def get_ngas(self,r):
		rp = np.arange(1.,self.rout.value[0],2.)*un.kpc
		r_inv = 1./rp[::-1]
		res = odeint(self.dHSE,y0=self.ngout,t=r_inv)[::-1]
		ngass = res.T[0]/un.cm**3
		return np.interp(r,rp,ngass)
	def get_nH(self,r):
		return (mu/muh)*self.get_ngas(r)
	def get_electron_density_profile(self,r):
		return (mu/mue)*self.get_ngas(r)
	def get_gas_thermal_pressure_profile(self,r):
		return self.K_thermal*(mu*mp*self.get_ngas(r))**g1
	def get_gas_non_thermal_pressure_profile(self,r):
		return self.K_nonthermal*(mu*mp*self.get_ngas(r))**g2
	def get_gas_turbulence_pressure_profile(self,r):
		return mu*mp*self.get_ngas(r)*self.vturb**2
	def get_gas_total_pressure_profile(self,r):
		rhog = mu*mp*self.get_ngas(r)
		return self.K_thermal*rhog**g1+self.K_nonthermal*rhog**g2+rhog*self.vturb**2
	def get_thermal_temperature_profile(self,r):
		return mu*mp*self.K_thermal*(mu*mp*self.get_ngas(r))**(g1-1)		
	def get_temperature_profile(self,r):
		rhog = mu*mp*self.get_ngas(r)
		return mu*mp*(self.K_thermal*rhog**(g1-1)+self.K_nonthermal*rhog**(g2-1)+self.vturb**2)
	def get_gas_mass_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc
		drp = rp[1:]-rp[:-1]
		dMgas = self.get_ngas(rp[1:])*rp[1:]**2*drp
		mgass = (np.nancumsum(dMgas)*mu*mp*4.*np.pi).to('M_sun')
		return np.interp(r,rp[1:],mgass)
	def get_thermal_energy_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc 
		drp = rp[1:]-rp[:-1]
		dEth = self.get_gas_thermal_pressure_profile(rp[1:])*rp[1:]**2*drp
		ethermal = (3./2.*np.nancumsum(dEth)*4.*np.pi).to('erg')
		return np.interp(r,rp[1:],ethermal)
	def get_non_thermal_energy_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc 
		drp = rp[1:]-rp[:-1]
		dEnth = self.get_gas_non_thermal_pressure_profile(rp[1:])*rp[1:]**2*drp
		enonth  = (3.*np.nancumsum(dEnth)*4.*np.pi).to('erg')
		return np.interp(r,rp[1:],enonth)
	def get_turbulence_energy_profile(self,r):
		rp = np.arange(1,r[-1].value,2)*un.kpc 
		drp = rp[1:]-rp[:-1]
		dEturb = self.get_gas_turbulence_pressure_profile(rp[1:])*rp[1:]**2*drp
		eturb = (3./2.*np.nancumsum(dEturb)*4.*np.pi).to('erg')
		return np.interp(r,rp[1:],eturb)

