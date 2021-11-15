"""
Module for deriving steady-state cooling flow solutions
"""
import numpy as np
import scipy, scipy.integrate
from numpy import log as ln, log10 as log, e, pi, arange, zeros
from astropy import units as un, constants as cons

import cgm_model_interface as CMI

mu = 0.62    #mean molecular weight
X = 0.75     #hydrogen mass fraction
gamma = 5/3. #adiabatic index
ne2nH = 1.2  #ratio of electrons to protons

######### interface base classes for potential and circular velocity

class Potential: 
    """interface for gravitational potential"""
    def vc(self,r): 
        """circular velocity"""
        assert(False)
    def Phi(self,r):
        """gravitational potential"""
        assert(False)
    def dlnvc_dlnR(self,r):
        """logarithmic derivative of circular velocity"""
        assert(False)

class CoolingFlow(CMI.Model):
    def __init__(self,potential,cooling,metallicity,boundary,Mdot,R_circ=None,R_sonic=None,
                 R_max=1*un.Mpc,R_min=0.1*un.kpc,
                 pr=False,return_all_results=False):
        """
        Cooling flow model specific parameters
        ______________________________________

        Mdot (SFR): Mass inflow rate
        R_sonic: Sonic point
        R_circ: circularization radius where gravitational and centripetal forces balance
        R_max: outer radius of integration
        R_min: inner radius of integration for supersonic region
        potential: gravitation potential (Potential object)
        cooling: radiative cooling (Cooling object)
        pr, return_all_results: debugging parameters
        """
        self.potential = potential
        self.cooling = cooling
        self.metallicity = metallicity
        self.boundary = boundary    
        self.Mdot = Mdot
        self.R_circ = R_circ
        self.R_sonic = R_sonic
        self.R_max = R_max
        self.R_min = R_min
        if self.R_circ!=None:
            tmp = self.shoot_from_R_circ(pr=pr,return_all_results=return_all_results)
        elif self.R_sonic!=None:
            tmp = self.shoot_from_sonic_point(pr=pr,return_all_results=return_all_results)
        if not return_all_results:
            self.res = tmp
        else:
            self.all_res = tmp           
            
    def shoot_from_R_circ(self,v0=1*un.km/un.s,max_step=0.1,T_low=1e4*un.K,T_high=1e6*un.K,
                          tol=1e-6,epsilon=0.1,terminalUnbound=True,pr=False,return_all_results=False):
        """
        Find marginally-bound solution starting at circularization radius. 
        Marginally-bound condition is satisfied via a shooting method, adjusting T at the circularization radius between consecutive integrations.
        Accepts:
        epsilon: initial radius of integration is R_circ(1+epsilon)
        v_0: velocity at R_circ(1+epsilon), should be small but positive
        max_step: minimum resolution of solution in ln(r)
        T_low, T_high: range of T(R_circ(1+epsilon)) at which to search for solution
        tol: integrations stop when consecutive integrations differ in log T(R_circ(1+epsilon)) by less than tol
        terminalUnbound: if terminalUnbound=True, integration stops when Bernoulli>0. 
        Returns:
        If return_all_results==False: a single IntegrationResult object where the integration reached R_max
        If return_all_results==True: dictionary with all integrations, regardless of stop reason. Dictionary key is T(R_circ(1+epsilon))
        """
        results = {}
        while log(T_high / T_low) > tol:
            T0 = (T_high*T_low)**0.5
            rho0 = (self.Mdot / (4*pi*self.R_circ**2 * v0)).to('g*cm**-3')
            res        = IntegrateFlowEquations(self.Mdot,T0,rho0,self.potential,self.cooling,self.metallicity,isInward=False,
                                                R_min=self.R_circ*(1+epsilon),
                                                R_max=self.R_max,terminalUnbound=terminalUnbound,
                                                issupersonic=False,
                                                minT=T_low.value/2.,R_circ=self.R_circ,max_step=max_step)
            results[T0.value] = res
            if type(res)==type(''):
                print(res)
                break 
            if pr: 
                print('Integrated with log T(R_circ)=%.2f, maximum radius reached %d kpc, stop reason: %s'%(log(T0.to('K').value),res.Rs()[-1].to('kpc').value,res.stopReason()))
            if res.stopReason() in ('sonic point', 'lowT'):
                T_low = T0
            elif res.stopReason() == 'unbound':
                T_high = T0
            elif res.stopReason()=='max R reached':            
                break #stop adjusting T
        if return_all_results:
            return results    
        if res.stopReason()=='max R reached': 
            return res
        print('no result reached maximum R, try rerunning with return_all_results=True and check intermediate solutions')

    def shoot_from_sonic_point(self,tol=1e-6,max_step=0.1,
                               epsilon=1e-3,dlnMdlnRInit=-1,return_all_results=False,
                            terminalUnbound=True,pr=False,calcInwardSolution=True,minT=2e4,x_low=1e-5, x_high = 1.):

        """
	Find transonic marginally-bound solution. 
	Integration first proceeds from the sonic point outward. Marginally-bound condition is satisfied via a shooting method, 
	adjusting T at the circularization radius between consecutive integrations. After outer subsonic region is found, solution
	is integrated inward (once) to derive supersonic part of the solution
	Accepts:
	tol: integrations stop when consecutive integrations differ in v_c^2/2 c_s^2 (R_sonic) by less than tol
	max_step: minimum resolution of solution in ln(r)
	epsilon: integration starts at R_sonic(1 +/- epsilon) for subsonic/supersonic regions
	dlnMdlnRInit: In rare cases where there are two possible transonic solutions with a subsonic outer region and a supersonic inner region, 
	the function will return the solution in which d ln Mach / d ln R is closest to this value.
	terminalUnbound: if terminalUnbound=True, integration stops when Bernoulli>0. 
	pr: whether to print out the steps of the integration
	calcInwardSolution: whether to calculate inner supersonic part of solution    
	minT: integration stops when T drops below this value. 
	x_low, x_high: range of x(R_sonic) at which to search for solution, where x = v_c^2 / (2 c_s^2). For a transonic solution x must be 0<x<=1
	Returns:
	If return_all_results==False: a single IntegrationResult object where the integration reached R_max
	If return_all_results==True: dictionary with all integrations, regardless of stop reason. Dictionary key is x, where x = v_c^2 / (2 c_s^2)
        
        **** comment 15/11/2021: this method will work only for constant metallicity profiles, otherwise another dlnT/dlnZ term needs to be added
	"""
        results = {}
        dlnMdlnRold = dlnMdlnRInit
        while x_high - x_low> tol:
            x = (x_high + x_low)/2.
            if pr: print('Integrated with v_c^2/c_s^2 (R_sonic) =%f; '%(2*x), end=' ')
            cs2_sonic_point = self.potential.vc(self.R_sonic)**2 / (2*x) 
            v_sonic_point = (cs2_sonic_point**0.5).to('km/s')
            T_sonic_point = (mu*cons.m_p*cs2_sonic_point / (gamma*cons.k_B)).to('K')
            tflow_to_tcool_sonic_point = 10/3. * (1.-x)
            rho_sonic_point = calc_rho_from_tflow2tcool(v_sonic_point, tflow_to_tcool_sonic_point, T_sonic_point, self.R_sonic, self.cooling,self.metallicity)
            if rho_sonic_point==False: #no solution found
                x_high = x
                continue                
            Mdot = 4*pi*self.R_sonic**2 * rho_sonic_point * v_sonic_point
            dlnTdlnR1, dlnTdlnR2 = calc_dlnTdlnR_at_sonic_point(self.R_sonic, x, rho_sonic_point, T_sonic_point, self.cooling, self.potential,self.metallicity,pr=pr)

            if dlnTdlnR1==None: #no solution
                x_high = x
                continue

            dlnMdlnR1, dlnMdlnR2 = [-1.5*dlnTdlnR + 3 - 5*x - 0.5*dlnTdlnR for dlnTdlnR in (dlnTdlnR1,dlnTdlnR2)]
            if np.abs(dlnMdlnR1 - dlnMdlnRold) < np.abs(dlnMdlnR2 - dlnMdlnRold): 
                dlnTdlnR = dlnTdlnR1
            else:
                dlnTdlnR = dlnTdlnR2

            dlnMdlnR = -1.5*dlnTdlnR + 3 - 5*x - 0.5*dlnTdlnR
            subsonicToSupersonic = dlnMdlnR<0 #mach number decreases outwards
            dlnvdlnR = -1.5*dlnTdlnR + 3 - 5*x
            dlnrhodlnR = -dlnvdlnR-2
            for isInward in ((False,),(False,True))[calcInwardSolution]:
                supersonic = (isInward and dlnMdlnR<0) or ((not isInward) and dlnMdlnR>0)
                direction = (1,-1)[isInward]
                T0 = T_sonic_point * (1. + direction*epsilon * dlnTdlnR)
                rho0 = rho_sonic_point * (1. + direction*epsilon * dlnrhodlnR)            
                R0 = self.R_sonic*(1+direction*epsilon)
                if not isInward:
                    res        = IntegrateFlowEquations(Mdot,T0,rho0,self.potential,self.cooling,self.metallicity,isInward=False,R_min=R0,R_max=R_max,terminalUnbound=terminalUnbound,
                                                        issupersonic=supersonic,minT=minT,max_step=max_step)                
                else: 

                    res_inward = IntegrateFlowEquations(Mdot,T0,rho0,self.potential,self.cooling,self.metallicity,isInward=True,R_min=R_min,R_max=R0,terminalUnbound=terminalUnbound,
                                                        issupersonic=supersonic,checkUnbound=False,minT=minT,max_step=max_step) 
                    res.add_inward_solution(res_inward.res) 
                if res=='starts unbound':
                    x_low = x
                    if pr: print('starts unbound')
                    break #don't run inward calculation
                if res=='starts supersonic':
                    x_high = x
                    if pr: print('starts supersonic')
                    break #don't run inward calculation
                if pr: 
                    if not isInward:
                        print('maximum r=%d kpc; stop reason: %s'%(res.Rs()[-1].to('kpc').value,res.stopReason()))
                    else:
                        print('Inward integration of supersonic part reached r = %.3f kpc'%res_inward.Rs().to('kpc').value.min())
                if res.stopReason() in ('sonic point', 'lowT'):
                    x_high = x
                    break #don't run inward calculation
                elif res.stopReason() == 'unbound':
                    x_low = x
                    break #don't run inward calculation
                elif res.stopReason()=='max R reached':
                    continue
            if res in ('starts unbound','starts supersonic'):
                continue
            dlnMdlnRold = dlnMdlnR
            results[x] = res
            if res.stopReason()=='max R reached':
                break #stop adjusting x       
        if return_all_results:
            return results    
        if res.stopReason()=='max R reached': 
            return res
        print('no result reached maximum R, try rerunning with return_all_results=True and check intermediate solutions')


    def check_if_solved(self):
        if hasattr(self,'res'): return True
        else:
            print('integration failed')
            return False
    def get_gas_mass_profile(self,r):
        """gas mass profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.Mgas()) * self.res.Mgas().unit
    def get_entropy_profile(self,r):
        """pseudo entropy profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.Ks()) * self.res.Ks().unit	    
    def get_temperature_profile(self,r):
        """gas temperature profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.Ts()) * self.res.Ts().unit * cons.k_B
    def get_electron_density_profile(self,r):
        """electron density profile"""
        return self.get_nH(r)*ne2nH 
    def get_nH(self,r):
        """gas density profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.nHs()) * self.res.nHs().unit
    def get_electron_thermal_pressure_profile(self,r):
        """electron thermal pressure profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.P2ks()) * self.res.P2ks().unit * cons.k_B
    def get_gas_thermal_pressure_profile(self,r):
        """gas thermal pressure profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.P2ks()) * self.res.P2ks().unit * cons.k_B
    def get_gas_non_thermal_pressure_profile(self,r):
        """non-thermal pressure profile"""
        return 0.
    def get_gas_turbulence_pressure_profile(self,r):
        """turbulence pressure profile"""
        return 0.
    def get_gas_total_pressure_profile(self,r):
        """gas total pressure profile"""
        return self.get_gas_thermal_pressure_profile(r) + self.get_gas_non_thermal_pressure_profile(r)
    def get_radial_velocity(self,r):
        """radial velocity profile"""
        if self.check_if_solved():
            return -np.interp(r,self.res.Rs(),self.res.vs()) * self.res.vs().unit
    def get_tcool(self,r):
        """gas cooling time-scale profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.t_cools()) * self.res.t_cools().unit
    def get_tff(self,r):
        """free-fall time-scale profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.tff()) * self.res.tff().unit
    def get_tcool2tff(self,r):
        """gas cooling to free-fall time-scale ratio profile"""
        if self.check_if_solved():
            return np.interp(r,self.res.Rs(),self.res.tcool_to_tff()) * self.res.tcool_to_tff().unit


######### steady-state equations integration
def IntegrateFlowEquations(Mdot,T0,rho0,potential,cooling,metallicity,isInward,R_min,R_max,R_circ=None,max_step=0.1,
                           atol=1e-6,rtol=1e-6,checkUnbound=True,issupersonic=False,terminalUnbound=True,minT=2e4):

    """
    Function for integrating steady-state flow equations. Called by shoot_from_R_circ() and shoot_from_sonic_point()    
    Accepts:
    Mdot, T0, rho0: hydrodynamic variables at initial radius (either R_min or R_max, depending on direction of integration)
    potential: Potential object
    cooling: Cooling object
    isInward: direction of integration (outward for subsonic part, inward for supersonic part)
    R_min, R_max: range of integration 
    max_step: minimum resolution of solution in ln(r)
    terminalUnbound, checkUnbound: if terminalUnbound=True, integration stops when Bernoulli>0. if checkUnbound==False, Bernoulli parameter is not calculated during integration
    issupersonic: whether solution is supersonic or subsoni    
    minT: integration stops when T drops below this value. 
    atol,rtol: input for scipy.integrate.solve_ivp    
    Returns:
    IntegrationResult object
    """
    def odes(ln_R, y,Mdot=Mdot,potential=potential,cooling=cooling,metallicity=metallicity,isInward=isInward,R_circ=R_circ):
        if isInward: R = e**-ln_R*un.kpc
        else:        R = e**ln_R*un.kpc
        ln_T,ln_rho = y
        rho,T=e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        nH = (X*rho/cons.m_p).to('cm**-3')

        v = (Mdot/(4*pi*R**2*rho)).to('km/s')
        cs2 = (gamma*cons.k_B * T / (mu*cons.m_p)).to('km**2/s**2')
        M = (v/cs2**0.5).to('')

        vc2 = potential.vc(R)**2
        Z = metallicity.Z(R)[0]
        
        if R_circ!=None:
            vc2 *= (1-(R_circ/R)**2)
        v_ratio = (vc2/cs2).to('')

        t_flow = (R/v).to('Gyr')
        LAMBDA = cooling.LAMBDA(T,Z,nH)
        t_cool = (rho*cs2 / (nH**2*LAMBDA) / (gamma*(gamma-1))).to('Gyr')
        t_ratio = (t_flow/t_cool).to('')

        dln_rho2dln_R =  (-t_ratio/gamma - v_ratio + 2*M**2)  / (1-M**2)
        dln_T2dln_R = t_ratio + dln_rho2dln_R*(gamma-1)

        if isInward: return -dln_T2dln_R, -dln_rho2dln_R
        else: return dln_T2dln_R, dln_rho2dln_R

    def sonic_point(ln_R, y,Mdot=Mdot,isInward=isInward,issupersonic=issupersonic): 
        if isInward: R = e**-ln_R*un.kpc
        else: R = e**ln_R*un.kpc        
        ln_T,ln_rho = y
        rho, T = e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        v = Mdot/(4*pi*R**2*rho)        
        cs2 = gamma*cons.k_B * T / (mu*cons.m_p)
        M = (v/cs2**0.5).to('')
        return M - 1
    def lowT(ln_R, y, minT=minT):
        ln_T,ln_rho = y
        T=e**ln_T*un.K
        return T.to('K').value-minT
    def unbound(ln_R, y,potential=potential,Mdot=Mdot,isInward=isInward,R_max=R_max): 
        if isInward: R = e**-ln_R*un.kpc
        else: R = e**ln_R*un.kpc    
        ln_T,ln_rho = y
        rho, T = e**ln_rho*un.g/un.cm**3, e**ln_T*un.K
        v = (Mdot/(4*pi*R**2*rho)).to('km/s').value
        cs2 = (gamma*cons.k_B * T / (mu*cons.m_p)).to('km**2/s**2').value
        B = 0.5*v**2 + cs2/(gamma-1) + potential.Phi(R).to('km**2/s**2').value
        return B
    def dummy(ln_R,y):
        return 1.
    sonic_point.terminal = True
    lowT.terminal = True
    unbound.terminal = terminalUnbound
    events = sonic_point,(dummy,unbound)[checkUnbound],(lowT,dummy)[issupersonic]

    if isInward: 
        Rrange =  R_max, R_min
        lnRrange = -ln(R_max.to('kpc').value),-ln(R_min.to('kpc').value)
    else:        
        Rrange =  R_min, R_max
        lnRrange = ln(R_min.to('kpc').value), ln(R_max.to('kpc').value)

    initVals = ln(T0/un.K), ln(rho0/(un.g*un.cm**-3))


    if not issupersonic and sonic_point(lnRrange[0], initVals)>0: return 'starts supersonic'
    if     issupersonic and sonic_point(lnRrange[0], initVals)<0: return 'starts subsonic'
    if terminalUnbound and checkUnbound and unbound(lnRrange[0], initVals)>0: return 'starts unbound'

    res = scipy.integrate.solve_ivp(odes,lnRrange,initVals,events=events,
                                    max_step=max_step,atol=atol,rtol=rtol)

    return IntegrationResult(res, Mdot, potential=potential,cooling=cooling,metallicity=metallicity,isInward=isInward)
def calc_rho_from_tflow2tcool(v, tflow2tcool, T, R, cooling,metallicity): 
    """
    Calculates rho for given v, t_flow/t_cool, T, R, and cooling function. Called by shoot_from_sonic_point.
    """

    nHs = 10.**np.arange(-7,10,0.01) * un.cm**-3
    rhos = nHs * cons.m_p / X
    Ps   = (X*mu)**-1 * nHs * cons.k_B * T
    tcools = (Ps/(2/3.) / (nHs**2*cooling.LAMBDA(T,metallicity.Z(R)[0],nHs))).to('Gyr')  
    vs = (R / (tcools * tflow2tcool)).to('km/s')

    lv = log(v.value)
    lvs = log(vs.to('km/s').value)
    arr = lvs - lv
    inds = ((np.sign(arr[1:]) * np.sign(arr[:-1])) < 0).nonzero()[0]
    if len(inds)!=1: return False
    ind = inds[0]
    log_rhos = log(rhos.to('g*cm**-3').value)
    if vs[ind]<vs[ind+1]: 
        good_lrho = np.interp(lv, lvs[ind:ind+2], log_rhos[ind:ind+2]) 
    else:
        good_lrho = np.interp(lv, lvs[ind+1:ind-1:-1], log_rhos[ind+1:ind-1:-1]) 
    return 10.**good_lrho* un.g*un.cm**-3
def calc_dlnTdlnR_at_sonic_point(R_sonic, x, rho_sonic_point, T_sonic_point, cooling, potential,metallicity,pr=True): 
    """
    Calculates dlnTdlnR at sonic point. Called by shoot_from_sonic_point().
    **** comment 15/11/2021: this method will work only for constant metallicity profiles, otherwise another dlnT/dlnZ term needs to be added
    """    
    # gradient of cooling function and potential
    Z = metallicity.Z(R_sonic)[0]
    nH_sonic_point = X*rho_sonic_point/cons.m_p
    dlnLambda_dlnT = cooling.f_dlnLambda_dlnT(T_sonic_point,Z,nH_sonic_point)
    dlnLambda_dlnrho = cooling.f_dlnLambda_dlnrho(T_sonic_point,Z,nH_sonic_point)
    dlnvc_dlnR = potential.dlnvc_dlnR(R_sonic)

    #solve quadratic equation    
    b = 29/6.*x - 17/6. + 1/3.*(1.-x)*(dlnLambda_dlnT+1.5*dlnLambda_dlnrho)
    c = 2/3.*x*dlnvc_dlnR + 5*x**2 - 13/3.*x + 2/3. - 5/3.*(1-x)**2*dlnLambda_dlnrho
    if b**2-4*c >= 0:
        return [(-b +j * (b**2-4*c)**0.5)/2. for j in (-1,1)]
    else:
        if pr: print('no transsonic solutions')
        return None, None





######### results 
class IntegrationResult:
    """
    class for accessing the integration results
    """
    eventNames = 'sonic point','unbound','lowT','max R reached'    
    def __init__(self, res, Mdot, potential, cooling, metallicity,T0factor=None,tflow2tcool0=None,isInward=False):
        self.res, self.Mdot, self.T0factor,self.tflow2tcool0,self.isInward,self.potential,self.cooling,self.metallicity = (
            res, Mdot.to('Msun/yr'), T0factor, tflow2tcool0,isInward,potential,cooling,metallicity)        
        self.inward_sonic_res = None
        self.unbound = ( len((self.Bernoulli() > 0).nonzero()[0]) or len(self.res.t_events[1]) ) #patch for cases where B=0 is not terminal        
    def add_inward_solution(self,inward_res):
        self.inward_sonic_res  = inward_res
    def Lambdas(self):
        """values of LAMBDA (cooling function) at all radii of solution"""
        return self.cooling.LAMBDA(self.Ts(),self.metallicity.Z(self.Rs()),self.nHs())            
    def Rcool(self,t):
        """
        cooling radius for a given time (e.g. Hubble time)
        """
        return 10.**np.interp(log(t.value),log(self.t_cools().value),log(self.Rs().value))*un.kpc
    def Rs(self):
        """radii of solution"""
        Rs = np.e**self.res.t
        if self.isInward: Rs = (Rs**-1.)[::-1]
        if self.inward_sonic_res !=None:
            Rs_sonic_inward = np.e**-self.inward_sonic_res.t[::-1]
            Rs = np.concatenate([Rs_sonic_inward,Rs])        
        return Rs * un.kpc
    def rhos(self):
        """densities of the solution at all radii"""
        rhos = e**self.res.y[1,:]
        if self.isInward: rhos = rhos[::-1]
        if self.inward_sonic_res !=None:
            rhos_sonic_inward = e**self.inward_sonic_res.y[1,:][::-1]
            rhos = np.concatenate([rhos_sonic_inward,rhos])                
        return rhos*un.g/un.cm**3    
    def Mgas(self):
        """cumulative gas mass of the solution at all radii"""
        dRs = np.pad((self.Rs()[2:] - self.Rs()[:-2]).to('kpc').value/2.,1,mode='constant')*un.kpc
        return ((4*np.pi*self.Rs()**2 * dRs * self.rhos()).cumsum()).to('Msun')
    def nHs(self):
        """hydrogen densities of the solution at all radii"""
        return (X*self.rhos()/cons.m_p).to('cm**-3')
    def Ts(self):
        """temperature of the solution at all radii"""
        Ts = e**self.res.y[0,:]
        if self.isInward: Ts = Ts[::-1]
        if self.inward_sonic_res !=None:
            Ts_sonic_inward = e**self.inward_sonic_res.y[0,:][::-1]
            Ts = np.concatenate([Ts_sonic_inward,Ts])        
        return Ts*un.K
    def P2ks(self):
        """pressure / k_B of the solution at all radii"""
        return (X*mu)**-1 * self.nHs() * self.Ts()
    def cs(self):
        """adiabatic sound speed of the solution at all radii"""
        return ((gamma*cons.k_B * self.Ts() / (mu*cons.m_p))**0.5).to('km/s')
    def internalEnergy(self):
        """internal energy of the solution at all radii"""
        return (gamma * (gamma-1))**-1 * self.cs()**2 
    def vc2(self):
        """square of the circular velocity of the potential at all radii"""
        return self.potential.vc(self.Rs())**2
    def tff(self):  
        """free fall time of the solution at all radii""" 
        return 2**0.5 *self.Rs() / self.vc2()**0.5
    def tcool_to_tff(self):
        """ratio of cooling time to free fall time of the solution at all radii""" 
        return self.t_cools() / self.tff()
    def Phi(self):
        """gravitational potential at all radii"""
        return self.potential.Phi(self.Rs())
    def vs(self):
        """inflow velocity of the solution at all radii"""
        return (self.Mdot / (4*pi*self.Rs()**2*self.rhos())).to('km/s')
    def Ms(self):
        """mach number of the solution at all radii"""
        return self.vs() / self.cs()
    def R_sonic(self):
        """sonic radius of the solution"""
        if log(self.Ms()[0]) > 0 and log(self.Ms()[-1]) < 0: #only supersonic to subsonic transitions
            indRsonic = ((log(self.Ms()[:-1]) * log(self.Ms()[1:]))<0).nonzero()[0]
            return self.Rs()[indRsonic]
        return None
    def Ks(self):
        """entropy of the solution at all radii"""
        return (cons.k_B * self.Ts() / (ne2nH*self.nHs())**(gamma-1)).to('keV*cm**2')
    def y_integrand(self):
        A = cons.sigma_T / (cons.m_e * cons.c**2) * cons.k_B * ne2nH
        return (A*self.nHs()*self.Ts()).to('cm**-1')
    def t_flows(self):
        """flow times (r/v) of the solution at all radii"""
        return (self.Rs()/self.vs()).to('Gyr')
    def t_cools(self):
        """cooling times of the solution at all radii"""
        return ((gamma*(gamma-1))**-1. * 
                self.rhos()*self.cs()**2 / 
                (self.nHs()**2*self.Lambdas())).to('Gyr')
    def Bernoulli(self):
        """Energy integral of the solution at all radii"""
        return (self.vs()**2/2. + 
                self.cs()**2 / (gamma-1) + 
                self.Phi()).to('km**2/s**2')
    def stopReason(self):
        """the reason the integration stopped"""
        if hasattr(self,'unbound') and self.unbound: return self.eventNames[1]
        Nevents = [len(x) for x in self.res.t_events]
        if 1 in Nevents: return self.eventNames[Nevents.index(1)]
        return self.eventNames[-1]
    def save(self,fn):
        np.savez(fn, 
                 rs_in_kpc = self.Rs().value, 
                 rhos_in_g_to_cm3 = self.rhos().value,
                 Ts_in_K = self.Ts().value,
                 vs_in_kms = self.vs())



    def sample(self,resolution,Rcirc,avoid_Rs,avoid_zs,Rres2Rcool=1.):
        """sample solution in order to create initial conditions for particle hydro simulation"""
        Mgass = self.Mgas()
        rs = self.Rs()
        Rin, Rout = 0*un.kpc, self.Rcool(10*un.Gyr)*Rres2Rcool        
        Rmax = np.interp(20*un.Gyr, (self.Rs() / self.cs()).to('Gyr').value,self.Rs().value)*un.kpc
        print(" %dr(t_cool=10Gyr) = %.0f kpc, r(t_sc=20Gyr) = %.0f kpc"%(Rres2Rcool,Rout.value, Rmax.value))
        while Rin<Rmax:            
            Min = np.interp(Rin,rs, Mgass)
            Mout = np.interp(Rout,rs, Mgass)
            dM = Mout- Min
            N = int(dM / resolution)

            N2 = 2*N
            q = np.random.random_sample(N2)
            sampled_rs     = np.interp(Min+q*dM, Mgass, rs)
            sampled_phis   = np.random.random_sample(N2) * 2 * np.pi
            sampled_thetas = np.arccos(np.random.random_sample(N2) * 2 - 1)

            sampled_Rcylinders = sampled_rs*np.sin(sampled_thetas)
            sampled_zs = sampled_rs*np.cos(sampled_thetas)
            bad_inds = (sampled_Rcylinders  < avoid_Rs) & (np.abs(sampled_zs) < avoid_zs)

            sampled_rs     = sampled_rs[~bad_inds][:N]
            sampled_phis   = sampled_phis[~bad_inds][:N]
            sampled_thetas = sampled_thetas[~bad_inds][:N]
            sampled_Rcylinders     = sampled_Rcylinders[~bad_inds][:N]
            sampled_zs     = sampled_zs[~bad_inds][:N]
            sampled_Ms = np.ones(N) * dM / N

            sampled_xs = sampled_rs*np.sin(sampled_thetas)*np.cos(sampled_phis)
            sampled_ys = sampled_rs*np.sin(sampled_thetas)*np.sin(sampled_phis)

            sampled_vrs    = np.interp(sampled_rs, rs, -self.vs())
            sampled_epsilons     = np.interp(sampled_rs, rs, self.internalEnergy())

            vcRcirc = np.interp(Rcirc, self.Rs(), self.vc2())**0.5
            sampled_vphis = ( (vcRcirc * (Rcirc / sampled_Rcylinders))                * (sampled_Rcylinders > Rcirc) + 
                              np.interp(sampled_rs,self.Rs(),self.vc2())**0.5         * (sampled_Rcylinders < Rcirc) )
            #assumes v_theta=0
            sampled_vxs = sampled_vrs * np.sin(sampled_thetas)*np.cos(sampled_phis) - sampled_vphis * np.sin(sampled_phis)
            sampled_vys = sampled_vrs * np.sin(sampled_thetas)*np.sin(sampled_phis) + sampled_vphis * np.cos(sampled_phis)
            sampled_vzs = sampled_vrs * np.cos(sampled_thetas)
            sampled_coords = np.array([sampled_xs,sampled_ys,sampled_zs]).T
            sampled_vs = np.array([sampled_vxs,sampled_vys,sampled_vzs]).T

            if Rin==0:
                fin_Ms = sampled_Ms
                fin_coords = sampled_coords
                fin_vs = sampled_vs
                fin_epsilons = sampled_epsilons
            else:            
                fin_Ms = np.concatenate([fin_Ms,sampled_Ms])  
                fin_coords = np.concatenate([fin_coords,sampled_coords],axis=0)  
                fin_vs = np.concatenate([fin_vs,sampled_vs],axis=0)  
                fin_epsilons = np.concatenate([fin_epsilons,sampled_epsilons])  
            Rin = Rout
            Rout = min(2**(0.5)*Rout,Rmax)
            resolution*=3
        return fin_Ms, fin_coords, fin_vs, fin_epsilons



