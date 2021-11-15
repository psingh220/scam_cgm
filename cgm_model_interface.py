"""
Module for accessing CGM thermodynamic profiles for various analytical models.
"""

######### interface base classes for potential, cooling, metallicity profile, boundary conditions and cosmology
class Potential:
        """interface for gravitational potential"""
        def vc(self,r):
                """circular velocity"""
                assert(False)

class Cooling:
	"""interface for cooling function"""
	def LAMBDA(self,T,Z,nH):
		"""cooling function"""
		assert(False)
	def f_dlnLambda_dlnT(self,T,Z,nH):
		"""logarithmic derivative of cooling function with respect to T"""
		assert(False)
	def f_dlnLambda_dlnrho(self,T,Z,nH):
		"""logarithmic derivative of cooling function with respect to rho"""
		assert(False)
	

class Metallicity_profile:
	"""interface for metallicity profile"""
	def Z(self,r):
		"""metallicity profile"""
		assert(False)

class Boundary_Conditions:
	"""interface for outer boundary conditions"""
	def outer_radius(self):
		"""radius where boundary conditions are set"""
		assert(False)
	def outer_density(self):
		"""gas density at boundary"""
		assert(False)
	def outer_temperature(self):
		"""gas temperature at boundary"""
		assert(False)
	def outer_pressure(self):
		"""gas pressure at boundary"""
		assert(False)
	def outer_phi(self):
		"""gravitational potential at inf"""
		assert(False)
	def outer_mdot(self):
		"""mass inflow rate at boundary"""
		assert(False)
class Model:
	"""interface for CGM models"""
	def __init__(self,Potential,Cooling,Metallicity,Boundary_Conditions,Model_Parameters):
		assert(False)
	'''
	Example:
	class Precipitation(Model):
		"""interface for precipitation model"""
		def __init__(self,Potential,Cooling,Metallicity,Boundary_Conditions,tcool_tff):	
			"""
			Precipitation model specific parameters
			_______________________________________

			tcool_tff: ratio of gas cooling to free-fall time
			"""
			assert(False)

	class Cooling_Flow(Model):
		"""interface for cooling flow model"""
		def __init__(self,Potential,Cooling,Metallicity,Boundary_Conditions,Mdot,R_circ):
			"""
			Cooling flow model specific parameters
			______________________________________

			Mdot (SFR) or R_sonic: Mass inflow rate
			R_circ or vphi_boundary: circularization radius where gravitational and centripetal forces balance
			"""
			assert(False)

	class Isentropic(Model):
		"""interface for isentropic model"""
		def __init__(self,Potential,Cooling,Metallicity,Boundary_Conditions,vturb,alpha_nt):
			"""
			Isentropic model specific parameters
			____________________________________

			vturb: turbulent component
			alpha_nt: ratio of cosmic ray and magnetic field pressure to the thermal pressure (boundary condition)
			"""
			assert(False)

	class Baryon_Pasting(Model):
		"""interface for baryon pasting model"""
		def __init__(self,Potential,Cooling,Metallicity,Boundary_Conditions,eps_f,eps_DM,f_star,S_star,A_nt,B_nt,gamma_nt,gamma_mod0,gamma_mod_zslope,x_break):
			"""
			Baryon pasting model specific parameters
			________________________________________

			eps_f: SN & AGN feedback efficiency
			eps_DM: energy injection efficiency from DM halo mergers
			f_star: normalization of the stellar mass fraction
			S_star: mass slope of the stellar mass fraction
			A_nt: non-thermal pressure fraction parameter
			B_nt: non-thermal pressure fraction parameter
			gamma_nt: non-thermal pressure fraction parameters
			gamma_mod0: polytropic exponent in the inner region at z=0
			gamma_mod_zslope: redshift evolution exponent of the polytropic exponent in the inner region 
			x_break: inner region in units of R500, i.e. r_inner = x_break/R500
			"""
			assert(False)
	'''
	def get_gas_mass_profile(self,r):
		"""gas mass profile"""
		assert(False)
	def get_entropy_profile(self,r):
		"""pseudo entropy profile"""
		assert(False)
	def get_temperature_profile(self,r):
		"""gas temperature profile"""
		assert(False)
	def get_electron_density_profile(self,r):
		"""electron density profile"""
		assert(False)
	def get_nH(self,r):
		"""gas density profile"""
		assert(False)
	def get_electron_thermal_pressure_profile(self,r):
		"""electron thermal pressure profile"""
		assert(False)
	def get_gas_thermal_pressure_profile(self,r):
		"""gas thermal pressure profile"""
		assert(False)
	def get_gas_non_thermal_pressure_profile(self,r):
		"""non-thermal pressure profile"""
		assert(False)
	def get_gas_turbulence_pressure_profile(self,r):
		"""turbulence pressure profile"""
		assert(False)
	def get_gas_total_pressure_profile(self,r):
		"""gas total pressure profile"""
		assert(False)
	def get_radial_velocity(self,r):
		"""radial velocity profile"""
		assert(False)
	def get_tcool(self,r):
		"""gas cooling time-scale profile"""
		assert(False)
	def get_tff(self,r):
		"""free-fall time-scale profile"""
		assert(False)
	def get_tcool2tff(self,r):
		"""gas cooling to free-fall time-scale ratio profile"""
		assert(False)
