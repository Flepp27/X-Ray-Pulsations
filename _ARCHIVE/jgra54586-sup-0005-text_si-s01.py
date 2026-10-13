#Purpose: Read in time series. Compute the Rayleigh test. Evaluate significance from Monte Carlo simulation.
#Category: Time series analysis
#Authors: Christian Knigge, Caitriona Jackman (c.jackman@soton.ac.uk)
#Version released September 18th 2018, University of Southampton

import numpy as np
from astropy.io import ascii
from matplotlib import pyplot as plt
from numpy.random import poisson 
import time
from distutils.version import LooseVersion

try:
    import numba
except ImportError:
    use_numba = False
    print
    print ('Numba is not installed; falling back on standard python')
else:
    if (LooseVersion(numba.__version__) > LooseVersion('0.38')):
        use_numba = True
        print
        print ('Numba is installed; using jit to speed up code')
        @numba.jit(nopython=True)     
        def rayleigh_power(omega, tf):
            s2 = 0.0*omega
            c2 = 0.0*omega
            pow = 0.0*omega
            nfreq = omega.size
            for i in range(nfreq):
                s2[i] = np.sum(np.sin(omega[i]*tf))**2.0
                c2[i] = np.sum(np.cos(omega[i]*tf))**2.0
                pow = (s2+c2)/np.float(tf.size)
            return pow
    else:
        use_numba = False
        print
        print ('Numba is installed, but version is old; falling back to standard python')
    

#start of main program    
start_clock = time.clock()

plotit = True      #create an interactive plot of the power spectrum?
verify = False      #if True, use a second method to calculate all power spectra and test for discrepancies
                    #this is a debugging tool and is slow, obviously...
ntest = 10000       #set the number of runs for Monte Carlo testing
startp = 2.0      #smallest period to search for (using Rayleigh test)
endp = 100.0      #largest period to search for (using Rayleigh test)
nstep = 1500.0       #number of frequency bins to search over (using Rayleigh test)
time_convert=True   #if you want to convert the input time from seconds into minutes
time_restrict=True  #if you want to restrict the time to a manual selection (e.g. X-ray hot spot)
tstart = 104.0      #the start time for N Hot spot 1 for ObsID 2519 from Jackman paper
tend = 454.0        #the end time for N Hot spot 1 for ObsID 2519 from Jackman paper
method = 2          #Method 1 has MC simulations start with Poissonian distribution of points.
                    #Method 2 has number of points for MC sim (nfake) is just same as input t

#Read in unbinned, time-tagged data
all = ascii.read("reg_np_nobin_2519_double_header.txt",header_start=0,data_start=1)
t=all['t']
t = t - t[0]    #Gets time from start of array (in seconds). *This assumes your input array is increasing in time.
if time_convert:
    t = t / 60.0    #converts the above time into minutes
if time_restrict:
    t=t[(t>=tstart)&(t<=tend)]  #restricts to a specific hot spot start and end time

#Run Rayleigh test on input data to extract highest power and best period

startf = 1./endp
endf = 1./startp
f = np.linspace(startf, endf, nstep)    #equal linearly spaced frequency grid
#Note: Be very careful when selecting the number of steps and the range of the frequency grid: Avoid undersampling, and ensure each spectral peak is fully resolved with at least 2 points
p = 1./f #convert to period...
omega = 2.0*np.pi*f #...and angular frequency

if (use_numba):
    pow = rayleigh_power(omega, t)
else:
    s2 = np.asarray([ (np.sum(np.sin(omega_i*t)))**2.0 for omega_i in omega],float,omega.size)
    c2 = np.asarray([ (np.sum(np.cos(omega_i*t)))**2.0 for omega_i in omega],float,omega.size)
    pow = (s2+c2)/np.float(t.size)


if (verify):
    #use alternative method to check (this is slow, obviously)
    #
    pow2 = np.asarray(map(lambda omega_i: (np.sum(np.cos(omega_i*t))**2.0 + \
                                           np.sum(np.sin(omega_i*t))**2.0) / np.float(t.size), omega))
    test = pow-pow2
    if (np.abs(test).max() > 1.e-10):
        print ('Max power difference between methods is in excess of 1e-10: '), np.abs(test).max()

fbest = f[np.argmax(pow)]
bestperiod_data = p[np.argmax(pow)]
maxpower_data =pow.max()

print
print ('Period corresponding to highest power = '), bestperiod_data, ('min')
print ('Max Power = '), maxpower_data
print

if (plotit):
    plt.plot(p, pow,'-k', lw=1)
    plt.scatter(p, pow, s=20, c='k')
    plt.ylabel('Rayleigh Power')
    plt.xlabel('Period (min)')
    plt.show()

    plt.plot(f, pow,'-k', lw=1)
    plt.scatter(f, pow, s=20, c='k')
    plt.ylabel('Rayleigh Power')
    plt.xlabel('Frequency (1/min)')
    plt.show()

    
#now we run Monte Carlo simulations to estimate significance
dt = tend-tstart #this is the interval of time over which we randomly spread the photons for MC simulation

maxpow_fake = np.zeros(ntest)
pbest_fake = np.zeros(ntest)
    
for ii in np.arange(ntest):

    if (method == 1):   #number of events drawn from Poisson distribution
        lam = np.float(t.size)
        nfake = poisson(lam, size=ntest)
        #each of the fake time arrays for the MC sim has a slightly different number of entries
    if (method == 2):  #fixed number of events
        nfake = np.full(ntest, t.size, dtype=int)
        #each of the fake time arrays for the MC sim has the exact same number of entries
        
    if (ii%1000 == 0) :
        print ('Done '), ii,(' of '), ntest, ('iterations')

    
    tf = tstart + np.sort(dt * np.random.uniform(low=0.0, high=1.0, size=nfake[ii]))
    #create the mock time series:
    #   nfake randomly (uniformly) distributed events across the same time interval as the actual data

        
        
    if (use_numba):
        powf = rayleigh_power(omega, tf)
    else: 
        s2 = np.asarray([ (np.sum(np.sin(omega_i*tf)))**2.0 for omega_i in omega],float,omega.size)
        c2 = np.asarray([ (np.sum(np.cos(omega_i*tf)))**2.0 for omega_i in omega],float,omega.size)
        powf = (s2+c2)/np.float(np.size(tf))

    if (verify):
        #use alternative method to check (this is slow, obviously)
        #
        powf2 = np.asarray(map(lambda omega_i: (np.sum(np.cos(omega_i*tf))**2.0 + \
                                               np.sum(np.sin(omega_i*tf))**2.0) / np.float(tf.size), omega))
        test = powf-powf2
        if (np.abs(test).max() > 1.e-10):
            print ('Max power difference between methods is in excess of 1e-10: '), np.abs(test).max()
        
        
    #save the highest peak and the associated period from each iteration
    maxpow_fake[ii] = powf.max()
    pbest_fake[ii]  = p[np.argmax(powf)]
        
fake_succ = 0
for ii in range(ntest):
    if (maxpow_fake[ii] > maxpower_data):
        fake_succ = fake_succ + 1
            
pvalue = float(fake_succ)/np.float(ntest)
    
print ('Number of Fake Data sets with Power > Real Power = '), fake_succ
print ('P-value (Fake Power >= Real Power) = '), pvalue
print    
print

stop_clock = time.clock()
timer = stop_clock - start_clock
print ('Code Execution Time: '), timer, (' s')
