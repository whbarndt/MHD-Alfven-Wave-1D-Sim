################################
# MHD simulation:
################################
# Purpose: To simulate one of the MHD wave modes, the Alfven Wave.
################################
# Info:
# - 1-D Simulation.
# - Dependent Variables: mass density (d), flow (u), pressure (p)
# - Using the normalized set of MHD equations.
# - Using the Lax-Wendroff Discretization method (stated below)
################################
# General Lax-Wendroff Discretizations (two-step)
# q*_j+1/2 = 1/2(q_j^n + q_(j+1)^n - (delta_t/(2*delta_x))(F_(j+1)^n - F_j^n)
# q_j^(n+1) = q_j^n - (delta_t/delta_x)(F*_(j+1/2) - F*_(j-1/2))
################################

################################
# Theoretical MHD Equations:
################################
# Mass continuity equation:
# - delta_p/delta_t = div(momentum*velocity)
################################
################################
# Momentum continuity equation:
# - delta_p*(velocity)/delta_t - div[momentum*velocity + 0.5*(pressure + (mag_field^2)/(2*u0))*current - (mag_field*mag_field)/u0] = 0
################################
################################
# Magnetic Flux continuity equation:
# - delta_b/delta_t + div(velocity*magnetic_fields - magnetic_fields*velocity) = 0
################################
################################
# Again, energy is usually taken into account but not in this simulation.
################################

# - Initial Value Problem:
# (i.e.) Pluck the imaginary magnetic string

# Generalized Methodology ##################################
# ~ Initialize Variables and Domains:
# set tfinal
# set delta_t
# create time and space domains
# set initial conditions
# while t < tfinal 
#   For Mass: 
#   For Momentum: q*_j+1/2 = 1/2(q_j^n + q_(j+1)^n) - (delta_t/(2*delta_x))(F_(j+1)^n - F_j^n)
#   Divide Momentum to find velocity.
#   For Magnetic Flux: q*_j+1/2 = 1/2(q_j^n + q_(j+1)^n) - (delta_t/(2*delta_x))(F_(j+1)^n - F_j^n)
#   
#   For Mass: q_j^(n+1) = q_j^n - (delta_t/delta_x)(F*_(j+1/2) - F*_(j-1/2))
#   For Momentum: q_j^(n+1) = q_j^n - (delta_t/delta_x)(F*_(j+1/2) - F*_(j-1/2))
#   For Magnetic Flux: q_j^(n+1) = q_j^n - (delta_t/delta_x)(F*_(j+1/2) - F*_(j-1/2))
############################################################

# Program ##################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MHD_sim_var:
  def __init__(self, gauss_offset, stand_dev, gauss_amp):
      self.xt = np.arange(-1, xfinal+dx,dx) # 1-D Space (x)
      self.xt0 = gauss_offset               # Offset of the Initial Gaussian
      self.h = stand_dev                    # Standard Deviation of the Initial Gaussian                        
      self.A = gauss_amp                    # Amplitude of the Initial Gaussian
      # Animation Variables
      self.x_momentum = []
      self.y_momentum = []
      self.z_momentum = []
      self.x_mag_field = []
      self.y_mag_field = []
      self.z_mag_field = []
      self.tarr = []
      # Main MHD Variables
      self.d = np.ones(len(self.xt))        # Density
      self.dux = np.zeros(len(self.xt))     # Momentum
      self.duy = np.zeros(len(self.xt))               
      self.duz = np.zeros(len(self.xt))
      self.ux = np.zeros(len(self.xt))      # Velocity
      self.uy = np.zeros(len(self.xt))               
      self.uz = np.zeros(len(self.xt))          
      self.Bx = np.zeros(len(self.xt))      # Magnetic Field
      self.By = np.zeros(len(self.xt))                
      self.Bz = np.ones(len(self.xt))
      self.p = 0                            # Pressure
      # Flux Variables
      self.m_flx = np.zeros(len(self.xt))
      self.x_mom_flx = np.zeros(len(self.xt))
      self.y_mom_flx = np.zeros(len(self.xt))
      self.z_mom_flx = np.zeros(len(self.xt))
      self.x_mag_flx = np.zeros(len(self.xt))
      self.y_mag_flx = np.zeros(len(self.xt))
      self.z_mag_flx = np.zeros(len(self.xt))

# Initializes a Gaussian curve on the z-dimension of the Magnetic Field.
def initialize_simulation(mhd):
  mhd.dux = mhd.A*np.exp(-(mhd.xt-mhd.xt0)**2/mhd.h**2) # ???

# General Lax-Wendroff Discretizations (two-step)
def lax_wendroff_half(q, fl, dt, dx):
  qhalf = 0.5*(q + np.roll(q, -1)) - (0.5*dt/dx)*(np.roll(fl, -1) - fl)
  return qhalf

def lax_wendroff_full(qhalf, fl, dt, dx):
  qtemp = qhalf - (dt/dx)*(fl - np.roll(fl, 1))
  return qtemp

def calculate_flux_terms(bucket):
  bucket.m_flx = mass_flux(bucket)
  bucket.x_mom_flx = x_momentum_flux(bucket)
  bucket.y_mom_flx = y_momentum_flux(bucket)
  bucket.z_mom_flx = z_momentum_flux(bucket)
  bucket.x_mag_flx = x_mag_flux(bucket)
  bucket.y_mag_flx = y_mag_flux(bucket)
  bucket.z_mag_flx = z_mag_flux(bucket)

# Main Solver
def solver(mhd, tfinal, dt, dx):
  t = 0
  while t < tfinal:
    # Calculate Flux Terms
    calculate_flux_terms(mhd)

    # Container for the first half of Lax Wendroff
    mhd_half = MHD_sim_var(mhd.xt0, mhd.h, mhd.A)
    
    # Density
    mhd_half.d = lax_wendroff_half(mhd.d, mhd.m_flx, dt, dx)
    # Momentum
    mhd_half.dux = lax_wendroff_half(mhd.dux, mhd.x_mom_flx, dt, dx)
    mhd_half.duy = lax_wendroff_half(mhd.duy, mhd.y_mom_flx, dt, dx)
    mhd_half.duz = lax_wendroff_half(mhd.duz, mhd.z_mom_flx, dt, dx)
    # Divide to find velocity 
    mhd_half.ux = np.divide(mhd_half.dux, mhd_half.d)
    mhd_half.uy = np.divide(mhd_half.duy, mhd_half.d)
    mhd_half.uz = np.divide(mhd_half.duz, mhd_half.d)
    # Magnetic Field
    mhd_half.Bx = lax_wendroff_half(mhd.Bx, mhd.x_mag_flx, dt, dx)
    mhd_half.By = lax_wendroff_half(mhd.By, mhd.y_mag_flx, dt, dx)
    mhd_half.Bz = lax_wendroff_half(mhd.Bz, mhd.z_mag_flx, dt, dx)
    
    # Second half of Lax Wendroff
    calculate_flux_terms(mhd_half)

    # Density
    dtemp = lax_wendroff_full(mhd.d, mhd_half.m_flx, dt, dx)
    # Momentum
    duxtemp = lax_wendroff_full(mhd.dux, mhd_half.x_mom_flx, dt, dx)
    duytemp = lax_wendroff_full(mhd.duy, mhd_half.y_mom_flx, dt, dx)
    duztemp = lax_wendroff_full(mhd.duz, mhd_half.z_mom_flx, dt, dx)
    # Divide to find velocity 
    uxtemp = np.divide(duxtemp, dtemp)
    uytemp = np.divide(duytemp, dtemp)
    uztemp = np.divide(duztemp, dtemp)
    # Magnetic Field
    Bxtemp = lax_wendroff_full(mhd.Bx, mhd_half.x_mag_flx, dt, dx)
    Bytemp = lax_wendroff_full(mhd.By, mhd_half.y_mag_flx, dt, dx)
    Bztemp = lax_wendroff_full(mhd.Bz, mhd_half.z_mag_flx, dt, dx)

    mhd.d = dtemp
    mhd.dux = duxtemp
    mhd.duy = duytemp
    mhd.duz = duztemp
    mhd.ux = uxtemp
    mhd.uy = uytemp
    mhd.uz = uztemp 
    mhd.Bx = Bxtemp
    mhd.By = Bytemp
    mhd.Bz = Bztemp

    mhd.x_momentum.append(mhd.dux)
    mhd.y_momentum.append(mhd.duy)
    mhd.z_momentum.append(mhd.duz)
    mhd.x_mag_field.append(mhd.Bx)
    mhd.y_mag_field.append(mhd.By)
    mhd.z_mag_field.append(mhd.Bz)
    mhd.tarr.append(t)

    t += dt
  # Plot the wave
  #plt.figure(0)
  #plt.plot(mhdsim.xt, mhdsim.dux)
  #plt.xlabel("x")
  #plt.ylabel("x_momentum_flux")
  #plt.figure(1)
  #plt.plot(mhdsim.xt, mhdsim.Bx)
  #plt.xlabel("x")
  #plt.ylabel("x_momentum_flux")
  #plt.show()  

# Mass Continuity Equation
def mass_flux(mhd):
  return mhd.d*mhd.uz

# X-Momentum Continuity Equation
def x_momentum_flux(mhd):    
  return mhd.d*mhd.ux*mhd.uz - mhd.Bz*mhd.Bx

# Y-Momentum Continuity Equation
def y_momentum_flux(mhd):    
  return mhd.d*mhd.uz*mhd.uy - mhd.Bz*mhd.By

 # Z-Momentum Continuity Equation
def z_momentum_flux(mhd):
  return mhd.d*mhd.uz**2 + 0.5*(mhd.Bx**2 + mhd.By**2 + mhd.Bz**2) - mhd.Bz**2

# X-Magnetic Flux Continuity Equation
def x_mag_flux(mhd):        
  return mhd.uz*mhd.Bx - mhd.Bz*mhd.ux 

# Y-Magnetic Flux Continuity Equation
def y_mag_flux(mhd):        
  return mhd.uz*mhd.By - mhd.Bz*mhd.uy 

# Z-Magnetic Flux Continuity Equation
def z_mag_flux(mhd):        
  return mhd.uz*mhd.Bz - mhd.Bz*mhd.uz 

# Animate the simulation (Taken from Lab 9)
def animate(mhd, label, dim, nstep):          
  fig, ax = plt.subplots()
  line, = ax.plot([],[],'r')
  ax.set_ylim(-mhd.A/2, mhd.A)
  ax.set_xlim(mhd.xt.min(), mhd.xt.max())
  ax.set_xlabel('x')
  ax.set_ylabel(label)
  def update_line(i):
    line.set_ydata(data[i])
    line.set_xdata(mhd.xt)
    return line,
  data = []  
  for i in range(0, len(mhd.tarr), nstep):
    data.append(dim[i])
  nfrm = int(len(data))
  ani = animation.FuncAnimation(fig, update_line, frames=nfrm, interval=1, blit=True, repeat=True, cache_frame_data=True)
  plt.show()
  return ani

# -------------------------------------------------------------
# Declaration of Program Parameters: 
xfinal = 1
dx = 0.01
c = 0.1       # Courant Number
va = 1        # Alfven Speed
dt = c*dx/va
tfinal = 2000*dt
# Gaussian variables
h = 10.0*dx
gauss_amp = 0.1
xt0 = 0
# Animation variable
nstep = 5

# Main Program 
mhdsim = MHD_sim_var(xt0, h, gauss_amp)
initialize_simulation(mhdsim)
solver(mhdsim, tfinal, dt, dx)
#xm_ani = animate(mhdsim, 'x-momentum', mhdsim.x_momentum, nstep)
#xm_ani
#ym_ani = animate(mhdsim, 'y-momentum', mhdsim.y_momentum, nstep)
#ym_ani
#zm_ani = animate(mhdsim, 'z-momentum', mhdsim.z_momentum, nstep)
#zm_ani
#xb_ani = animate(mhdsim, 'x-mag_field', mhdsim.x_mag_field, nstep)
#xb_ani
#yb_ani = animate(mhdsim, 'y-mag_field', mhdsim.y_mag_field, nstep)
#yb_ani
#zb_ani = animate(mhdsim, 'z-mag_field', mhdsim.z_mag_field, nstep)
#zb_ani
#--------------------------------------------------------------
# For possible later use...
# Extras #

#def div(f):
#    """
#    Computes the divergence of the vector field f, corresponding to dFx/dx + dFy/dy + ...
#    :param f: List of ndarrays, where every item of the list is one dimension of the vector field
#    :return: Single ndarray of the same shape as each of the items in f, which corresponds to a scalar field
#    Credit to @Daniel on StackOverflow.
#    """
#    num_dims = len(f)
#    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])

