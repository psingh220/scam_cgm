"""
Module for accessing isentropic model
"""

import cgm_model_interface as CMI
from astropy import units as un, constants as cons
import numpy as np
from scipy.integrate import odeint

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
		self.tout, self.ngout = boundary.outer_temperature(), (muh/mu)*boundary.outer_density()
		self.K_thermal = self.tout.to('erg')/self.ngout**(g1-1)/(mu*mp)**g1
		self.K_nonthermal = (self.alpha_ntb-1)*self.tout.to('erg')/self.ngout**(g2-1)/(mu*mp)**g2
	def dHSE(self,ng,r_inv):
		r_inv = [r_inv]/un.kpc
		ng = ng/un.cm**3
		tphi = (0.5*mu*mp*self.potential.vc(1/r_inv)**2).to('erg')
		return 2*tphi/((mu*mp)**2*r_inv*(self.vturb**2/(mu*mp*ng)+self.K_thermal*g1*(mu*mp*ng)**(g1-2)+self.K_nonthermal*g2*(mu*mp*ng)**(g2-2)))
	def get_ngas(self,r):
		r_inv = 1/r[::-1]
		res = odeint(self.dHSE,y0=self.ngout,t=r_inv)[::-1]
		return res.T[0]/un.cm**3
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
	def dMgas(self,mg,r):
		r=[r]*un.kpc
		return self.get_ngas(r)*r*r*(un.kpc.to('cm'))**3
	def get_gas_mass_profile(self,r):
		y0 = 4*np.pi*mu*mp*odeint(self.dMgas,y0=0,t=[0.001,r[0].value]).T[0] #mass enclosed within r[0]
		return (4*np.pi*mu*mp*odeint(self.dMgas,y0=y0[-1],t=r).T[0]).to('M_sun')
	def dthermal_energy(self,Eth,r):
		r=[r]*un.kpc
		return self.get_gas_thermal_pressure_profile(r)*3/2*r*r*(un.kpc.to('cm'))**3
	def get_thermal_energy_profile(self,r):
		y0 = 4*np.pi*odeint(self.dthermal_energy,y0=0,t=[0.001,r[0].value]).T[0] # thermal energy at r[0]
		return 4*np.pi*odeint(self.dthermal_energy,y0=y0[-1],t=r).T[0]*un.erg
	def dnonthermal_energy(self,Enth,r):
		r=[r]*un.kpc
		return self.get_gas_non_thermal_pressure_profile(r)*3*r*r*(un.kpc.to('cm'))**3
	def get_non_thermal_energy_profile(self,r):
		y0 = 4*np.pi*odeint(self.dnonthermal_energy,y0=0,t=[0.001,r[0].value]).T[0] # non thermal energy at r[0]
		return 4*np.pi*odeint(self.dnonthermal_energy,y0=y0[-1],t=r).T[0]*un.erg
	def dturb_energy(self,Eturb,r):
		r=[r]*un.kpc
		return self.get_gas_turbulence_pressure_profile(r)*3/2*r*r*(un.kpc.to('cm'))**3
	def get_turbulence_energy_profile(self,r):
		y0 = 4*np.pi*odeint(self.dturb_energy,y0=0,t=[0.001,r[0].value]).T[0] # non thermal energy at r[0]
		return 4*np.pi*odeint(self.dturb_energy,y0=y0[-1],t=r).T[0]*un.erg
	def get_total_energy_profile(self,r):
		return self.get_thermal_energy_profile(r)+self.get_non_thermal_energy_profile(r)+self.get_turbulence_energy_profile(r)
