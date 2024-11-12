import os
import numpy as np
import zodiax as zdx
import jax.numpy as jnp
import jax
import jaxoplanet
from jaxoplanet.orbits.keplerian import Central, Body, System
from jaxoplanet.units import unit_registry as ureg
import numpyro
import matplotlib.pyplot as plt
import multiprocessing
import os
import glob
import pandas as pd
from astropy.table import Table
import numpyro_ext.distributions, numpyro_ext.optim
from jax import config
import seaborn as sns
import scipy.optimize 
from astropy.io import fits
from astropy.time import Time
import glob
import astropy.units as u
from astropy.timeseries import LombScargle
import matplotlib.font_manager as font_manager
import arviz as az
import scipy
from numpyro.optim import _NumPyroOptim
from numpyro import distributions as dist
from numpyro_ext import distributions as distx
print(jax.devices(backend='gpu'))
config.update("jax_enable_x64", True)
print('64 bit, gpu')
numpyro.set_platform('gpu')
yr = 365.25
#cores = 1
#numpyro.set_host_device_count(cores)

print('SITES LIMITED RUN')
def estimate_frequencies(
    x, y, fmin=None, fmax=None, max_peaks=3, oversample=4.0, optimize_freq=True):
    tmax = x.max()
    tmin = x.min()
    dt = np.median(np.diff(x))
    df = 1.0 / (tmax - tmin)
    ny = 0.5 / dt

    if fmin is None:
        fmin = df
    if fmax is None:
        fmax = ny

    freq = np.arange(fmin, fmax, df / oversample)
    power = LombScargle(x, y).power(freq)

    # Find peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power) - 1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for j in range(max_peaks):
        i = peak_inds[0]
        freq0 = freq[i]
        alias = 2.0 * ny - freq0

        m = np.abs(freq[peak_inds] - alias) > 25 * df
        m &= np.abs(freq[peak_inds] - freq0) > 25 * df

        peak_inds = peak_inds[m]
        peaks.append(freq0)
    peaks = np.array(peaks)

    if optimize_freq:

        def chi2(nu):
            arg = 2 * np.pi * nu[None, :] * x[:, None]
            D = np.concatenate([np.cos(arg), np.sin(arg), np.ones((len(x), 1))], axis=1)

            # Solve for the amplitudes and phases of the oscillations
            DTD = np.matmul(D.T, D)
            DTy = np.matmul(D.T, y[:, None])
            w = np.linalg.solve(DTD, DTy)
            model = np.squeeze(np.matmul(D, w))

            chi2_val = np.sum(np.square(y - model))
            return chi2_val
        print(peaks)
        res = scipy.optimize.minimize(chi2, peaks, method="L-BFGS-B")
        return res.x
    else:
        return peaks
def get_weights(time,flux,freq,norm=True):
    """Calculates the amplitudes of each frequency, returning
    an array of amplitudes. This is useful for 
    calculating the weighted average time delay.
    
    Returns
    -------
    weights
        (potentially) normalised amplitudes of each frequency
    """
    weights = np.zeros(len(freq))
    for i, f in enumerate(freq):
        model = LombScargle(time, flux)
        sc = model.power(f, method="fast", normalization="psd")
        fct = np.sqrt(4.0 / len(time))
        weights[i] = np.sqrt(sc) * fct
    if norm:
        weights /= np.max(weights)
    return weights
def dft_phase(x, y, freq):
    """ 
    Discrete fourier transform to calculate the ASTC phase
    given x, y, and an array of frequencies
    
    Parameters
    ----------
        x : `array`
            Array in which to calculate 
        y : `array`
    
    Returns:
    ----------
        phase : `list`
            A list of phases for the given frequencies
    """

    freq = np.asarray(freq)
    x = np.array(x)
    y = np.array(y)
    phase = []
    for f in freq:
        expo = 2.0 * np.pi * f * x
        ft_real = np.sum(y * np.cos(expo))
        ft_imag = np.sum(y * np.sin(expo))
        phase.append(np.arctan2(ft_imag, ft_real))
    return phase
def get_window_tds(time,flux, freq, segment_size):
    """
    Calculates the time delay signal, splitting the light curve into 
    chunks of width segment_size. A smaller segment size will increase
    the scatter of the time delay signal, especially for low frequencies.
    
    Parameters
    ----------
    segment_size : `float`
        Segment size in which to separate the light curve, in units of
        the light curve time. For example, the default segment size of 10 
        will separate a 1000 d long light curve in 100 segments of 10 d
        each.
    
    Returns
    -------
    time_midpoint : `numpy.ndarray`
        Midpoints of time for each segment in the light curve
    time_delay: `numpy.ndarray`
        Values of the extracted time delay in each segment.
    """
    uHz_conv = 1e-6 * 24 * 60 * 60
    time_0 = time[0]
    time_slice, mag_slice, phase = [], [], []
    time_delays, time_midpoints = [], []

    # Iterate over lightcurve
    for t, y in zip(time, flux):
        time_slice.append(t)
        mag_slice.append(y)

        # In each segment
        if t - time_0 > segment_size:
            # Append the time midpoint
            time_midpoints.append(np.mean(time_slice))

            # And the phases for each frequency
            phase.append(dft_phase(time_slice, mag_slice, freq))
            time_0 = t
            time_slice, mag_slice = [], []

    phase = np.array(phase).T
    # Phase wrapping patch
    for ph, f in zip(phase, freq):
        ph = np.unwrap(ph)
        ph -= np.mean(ph)

        td = ph / (2 * np.pi * (f / uHz_conv * 1e-6))
        time_delays.append(td)
    time_delays = np.array(time_delays).T
    return np.array(time_midpoints), np.array(time_delays)
def plot_time_delay(time, flux, freq, ax=None, segment_size=10, show_weighted=False, **kwargs):
    """Plots the time delay. **kwargs go into `get_time_delay`.
    """
    t0s, time_delay = get_window_tds(time, flux,freq, segment_size, **kwargs)
    if ax is None:
        fig, ax = plt.subplots()

    colors = np.array(sns.color_palette("Blues", n_colors=len(freq)))[::-1]
    for td, color in zip(time_delay.T, colors):
        ax.plot(t0s, td, c=color)

    if show_weighted:
        ax.plot(
            t0s,
            np.average(time_delay, axis=1, weights=get_weights(time,flux,freq)),
            c=[0.84627451, 0.28069204, 0.00410611],
        )
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("Time delay [s]")
    ax.set_xlim(t0s[0], t0s[-1])
    return ax
class Orbit(zdx.Base):
    """
    This class defines an orbit model which solves the time delay equations
        for given input parameters and times. The orbit model can return
        either a synthetic light curve composed of `freq` sinusoids which are
        phase modulated with the orbital parameters, or instead can return a
        synthetic time delay curve. `Orbit` will also solve radial velocity
        curves given the same parameters.
    
    Args:
        period (pymc3.model.FreeRV, optional): Orbital period tensor. Defaults to None.
        lighttime (pymc3.model.FreeRV, optional): Projected semi-major axis tensor. Defaults to None.
        freq (array or pymc3.model.FreeRV, optional): Frequencies used in the model. Defaults to None.
        eccen (pymc3.model.FreeRV, optional): Eccentricity tensor. Defaults to None.
        omega (pymc3.model.FreeRV, optional): Periapsis tensor. Defaults to None.
        phi (pymc3.model.FreeRV, optional): Phase of periapsis tensor. Defaults to None.
    """
    period: jnp.ndarray
    lighttime: jnp.ndarray
    freq: jnp.ndarray
    eccen: jnp.ndarray
    omega: jnp.ndarray
    phi: jnp.ndarray
    def __init__(
        self, period, lighttime, freq, eccen, omega, phi
    ):
        self.period = period
        self.lighttime = lighttime
        self.omega = omega
        self.eccen = eccen
        self.phi = phi

        self.freq = freq

    def get_time_delay(self, time):
        """Calculates the time delay for the given time values.
        
        Args:
            time (array): Time values at which to calculate tau.
        
        Returns:
            array: Time delay values for each `time`
        """
        # Mean anom
        M = 2.0 * jnp.pi * time / self.period - self.phi

        # Negative psi to agree with Hey+2019. Implies closest star has negative
        # time delay
        if self.eccen is None:
            psi = -jnp.sin(M)
        else:
            sinf, cosf = jaxoplanet.core.kepler(M, self.eccen + jnp.zeros_like(M))
            psi = (
                -1
                * (1 - jnp.square(self.eccen))
                * (sinf*jnp.cos(self.omega) + cosf*jnp.sin(self.omega))
                / (1 + self.eccen * cosf)
            )

        tau = (self.lighttime / 86400) * psi[:, None]
        return tau
    def get_lightcurve_model(self, time, flux):
            """Calculates a synthetic light curve given the orbital parameters of 
            the `Orbit` object and supplied times and fluxes. The `orbit.freq` are
            phase modulated with binary motion.
            
            Args:
                time (array): Time-stamps
                flux (array): Flux values for each `time`
            
            Returns:
                array: Synthetic light curve
            """
            tau = self.get_time_delay(time)
            arg = 2.0 * jnp.pi * self.freq * (time[:, None] - tau)
            D = jnp.concatenate((jnp.cos(arg), jnp.sin(arg)), axis=-1)
            w = jax.scipy.linalg.solve(jnp.dot(D.T, D), jnp.dot(D.T, flux))
            #print(w)
            lc_model = jnp.dot(D, w)
            return lc_model


def td_plotter(window_time, window_td, med_td_time, med_td, taus=None, alpha=1.0, ax=None):
    plt.rcParams['axes.edgecolor']='black'
    plt.rcParams['axes.labelcolor']='black'
    plt.rcParams['xtick.color']='black'
    plt.rcParams['ytick.color']='black'
    plt.rcParams['text.color']='black'
    font_dir = ['/home/u1154594/tess_shortlist_pm_search_reverse/font/']
    for font in font_manager.findSystemFonts(font_dir):
        font_manager.fontManager.addfont(font)
    plt.rcParams['font.family'] = 'HelveticaNeue'

    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    plt.rcParams['axes.grid.which'] = 'both'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    if not ax:
        fig = plt.figure(figsize=(8,4),tight_layout = True)
        gs = fig.add_gridspec(1,1)
        ax = gs.subplots()
    if alpha == 1.0:
        ax.plot(med_td_time,med_td,color="#7139AC",lw=1)
    else:
        ax.plot(med_td_time,med_td,color="#7139AC",lw=0.1,alpha=0.5)

    ax.set_xlabel('Time (BJTD-2457000)')
    ax.set_ylabel('Time delay (s)')
    ax.scatter(window_time,window_td, s=3, color='grey')
    return ax
def make_phase(times, t0, P):
    phase_offset = 0.25
    phase = (((times - t0) / P) + phase_offset) % 1
    return (phase - phase_offset) * P * 24
# for the TESS phase plot
def phase_bin(times_lc, flux,stepsize=(11/60)):
    steps  = np.arange(-4, 4, stepsize)
    length = len(steps)-1
    flux_binned     = np.zeros(length)
    flux_binned_err = np.zeros(length)
    for i in range(length):
        lim_1 = times_lc < steps[i]
        lim_2 = steps[i+1] < times_lc
        box   = np.where(~(lim_1 + lim_2))[0]
        
        flux_binned[i]     = np.median(flux[box])
        flux_binned_err[i] = np.std(flux[box]) / np.sqrt(len(flux[box]))
    
    return (steps[:-1] + (stepsize/2)), flux_binned, flux_binned_err


def jaxstrom(tic):
    spoc_path = 'TESS_lc'
    print('hello2')
    lclist = glob.glob(os.path.join(spoc_path,'*'+str(tic)+'*'))
    print(lclist)
    lclist.sort(key=lambda x: int(x.split('-')[1][1:]))
    lclist = [str(i)+'/'+str(i.split('/')[-1])+'_lc.fits' for i in lclist]
    print(lclist)
    if not lclist:
        return 0
    lk_lclist = []
    lctime,lcflux,lcfluxerr = jnp.array([]),jnp.array([]),jnp.array([])
    for lc in lclist:
            try:
                print(lc)
                hdul = fits.open(lc)
                print(hdul[0].header["Sector"])
                quality_flag = jnp.array(hdul[1].data['QUALITY']>0).astype('bool')
                flux = hdul[1].data['PDCSAP_FLUX'].astype('float64')[~quality_flag]
                flux_err = hdul[1].data['PDCSAP_FLUX_ERR'].astype('float64')[~quality_flag]
                time = hdul[1].data['TIME'].astype('float64')[~quality_flag]
                nanmask = jnp.isnan(flux) | jnp.isnan(time)
                flux = (flux[~nanmask])
                flux_err = flux_err[~nanmask]
                time = time[~nanmask]
                medflux = jnp.nanmedian(flux)
                flux = ((flux/medflux)) - 1.0
                flux_err = (flux_err)/medflux
                outlier_mask = (jnp.abs(flux-jnp.median(flux)) > 5*jnp.nanstd(flux))
                flux = (flux[~outlier_mask])*1e3
                flux_err = (flux_err[~outlier_mask])*1e3
                time = (time[~outlier_mask])
                lctime = jnp.concatenate((lctime,time))
                lcflux = jnp.concatenate((lcflux,flux))
                lcfluxerr = jnp.concatenate((lcfluxerr,flux_err))
                hdul.close()
            except Exception as e:
                #raise
                print('Failed to open fits lightcurves for '+ str(tic))
                print(e)
    print(lctime)
    print(lcflux)
    print(lcfluxerr)

    est_freqs = estimate_frequencies(jnp.array(lctime), jnp.array(lcflux), fmin=5, fmax=70, max_peaks=3, optimize_freq=True)
    peaks = est_freqs
    true_params = {
    'logP': jnp.log(1279.264843290876),
    'logasini': jnp.log(407.85743603961475),
    'omega': -0.20443811352052782,
    'eccen': 0.10631335768678031,#jnp.median(samples["eccen"][nstart:]),
    "mean": jnp.mean(lcflux),
    'phi': 1.2076862858385957,
    'nu': peaks,
    }
    print(est_freqs)
    print(peaks)
    orbit_test = Orbit(period=jnp.exp(true_params['logP']),
                      lighttime=jnp.exp(true_params['logasini']),
                      omega=true_params['omega'],
                      eccen=true_params['eccen'],#jnp.median(samples["eccen"][nstart:]),
                      phi=true_params['phi'],
                      freq=peaks,
                     )
    time_seg = lctime
    flux_seg = lcflux
    fold_seg = time_seg % (3/est_freqs[0])
    binned_flux,binned_seg,_ = scipy.stats.binned_statistic(fold_seg,flux_seg,'mean',bins=200)

    fig = plt.figure(figsize=(10,4),tight_layout = True)
    ax = fig.add_subplot(111)
    fine_time_seg = np.linspace(min(time_seg),max(time_seg),500)
    ax.scatter(fold_seg,orbit_test.get_lightcurve_model(time_seg, flux_seg),color="grey",s=1,alpha=0.05)
    ax.set_xlabel('Relative Time (Days)',fontsize=15)
    ax.set_ylabel('Relative Brightness (ppt)',fontsize=15)
    #plt.plot(fold_seg,orbit_test.get_lightcurve_model(fold_seg, flux_seg),color="#7139AC",alpha=1.0, lw = 3)
    ax.scatter(binned_seg[:-1],binned_flux,c="#7139AC",s=10,zorder=10)
    plt.savefig('lc_test_fold.png',dpi=500, transparent=True)
    plt.close('all')
    period_guess = 1300
    print(jnp.median(lcflux))
    print(jnp.mean(lcflux))
    plt.scatter(lctime,lcflux,s=1,alpha=0.1,color='blue')
    plt.savefig('alpha_pic_lc.png',dpi=300)

    #Open IAD astrometry data
    header_name = 'RAdeg        DEdeg        Plx      pm_RA    pm_DE    e_RA   e_DE   e_Plx  e_pmRA e_pmDE dpmRA  dpmDE  e_dpmRA  e_dpmDE  ddpmRA  ddpmDE  e_ddpmRA  e_ddpmDE  upsRA   upsDE   e_upsRA  e_upsDE  var'.split()
    full_data = pd.read_csv('H032607.d',header=None,names=header_name,delim_whitespace=True,skiprows=lambda x: x != 10)
    plx = float(full_data['Plx']*1e-3) #get the mean hipparcos parallax from the header data
    pm_ra = float(full_data['pm_RA']*1e-3) #get the mean hipparcos pm_ra from the header data
    pm_dec = float(full_data['pm_DE']*1e-3) #get the mean hipparcos pm_dec from the header data
    del_ra = float(full_data['RAdeg'])
    header = 'IORB   EPOCH    PARF    CPSI    SPSI     RES   SRES'.split()
    data = pd.read_csv('H032607.d',header=None,names=header,delim_whitespace=True,comment='#')
    t_as = jnp.array(Time(jnp.array(data["EPOCH"]+1991.25).flatten(), format='decimalyear').jd)
    t0_as =  Time(1991.25, format='decimalyear').jd # J1991.25
    #t_as -= t0_as
    gof_ext = 0.07
    w = jnp.array(data['RES']*1e-3) # along-scan position, arcsec
    sres = jnp.array(data['SRES']) # uncertainty in the along-scan position, arcsec
    DOF = len(t_as) - 6
    disper_ext = 12.0

    renorm_factor = (gof_ext * jnp.sqrt(2/(9*DOF)) + 1 - (2/(9*DOF)))**3
    sig_w = renorm_factor * np.sqrt(sres**2 - disper_ext**2)*1e-3
    pf = jnp.array(data['PARF']) # parallax factor

    spsi = jnp.array(data['CPSI']) # sin of scan position angle #CHANGE BACK TO CPSI
    cpsi = jnp.array(data['SPSI']) # cos of scan position angle #CHANGE BACK TO SPSI
    def single_star_soln(t, delta_ra, delta_dec, pm_ra, pm_dec, parallax, pf, spsi,cpsi):
        single_ra = (pm_ra*t/yr)*cpsi + delta_ra + parallax*pf*cpsi
        single_dec = (pm_dec*t/yr)*spsi + delta_dec + parallax*pf*spsi
        tot = ((pm_ra*t/yr)*spsi + (pm_dec*t/yr)*cpsi) + parallax*pf
        return tot
    tot_meansoln = single_star_soln(t_as-t0_as, 0.0, 0.0, pm_ra, pm_dec, plx, pf, spsi, cpsi)
    def mlr(m):
        return 4.85 - 14.2*jnp.log10(m) + 14.1*jnp.log10(m)**2 - 9.99*jnp.log10(m)**3 + 2.66*jnp.log10(m)**4
    def mlrv(m):  
        return 10**(0.4*(mlr(1.0)-mlr(m)))
    def model(t, yerr, y=None):
            ### Proper motion + parallax modeling ###
        t_ptv,t_as = t
        lc, w = y
        lc_err, sig_w = yerr
        # Stellar position offest
        delta_ra = numpyro.sample('delta_ra',dist.Normal(0., 0.2)) # ra*cos(dec), arcsec
        delta_dec = numpyro.sample('delta_dec',dist.Normal(0., 0.2)) # dec, arcsec

        # Stellar proper motion
        pm_ra = numpyro.sample('pm_ra',dist.Normal(0., 0.2)) # ra*cos(dec), arcsec/yr
        pm_dec = numpyro.sample('pm_dec',dist.Normal(0., 0.2)) # dec, arcsec/yr
        # Parallax
        log_parallax = numpyro.sample('log_parallax', dist.Normal(jnp.log(0.033), 5.0)) #arcsec
        parallax = numpyro.deterministic('parallax',jnp.exp(log_parallax)) # arcsec
        # Astrometric jitter
        log_sigma = numpyro.sample('log_sigma',dist.Normal(0.0,1.0))
        # Model prediction
        def single_star_as(t_as, delta_ra, delta_dec, pm_ra, pm_dec, parallax, pf, spsi,cpsi):
            single_ra = numpyro.deterministic('single_ra', pm_ra*t_as/yr + delta_ra + parallax*pf*spsi)
            single_dec = numpyro.deterministic('single_dec', pm_dec*t_as/yr + delta_dec + parallax*pf*cpsi)
            return (pm_ra*t_as/yr+delta_ra)*spsi + (pm_dec*t_as/yr+delta_dec)*cpsi + (parallax*pf)
        
        model_single = numpyro.deterministic('model_single', single_star_as(t_as-t0_as, delta_ra, delta_dec, pm_ra, pm_dec, parallax, pf, spsi,cpsi))
        ### Binary modeling ###
        # Companion mass
        log_mp = numpyro.sample("log_mp", dist.Uniform(jnp.log(1e-3), jnp.log(1.59))) # solar masses
        mp = numpyro.deterministic('mp', jnp.exp(log_mp))

        #stellar mass
        ms = numpyro.sample('ms',dist.Normal(1.6,0.065))

        # Period
        #log_p = numpyro.sample('log_p', dist.Uniform(jnp.log(1e2),jnp.log(1e4))) # days
        bounded_logP = numpyro.distributions.Normal(jnp.log(1400), 20.)
        bounded_logP.support = numpyro.distributions.constraints.interval(jnp.log(50), jnp.log(3000))
        logP = numpyro.sample("logP", bounded_logP)
        period = numpyro.deterministic("period", jnp.exp(logP))
        ecc = numpyro.sample("eccen", numpyro.distributions.Uniform(0.01, 0.9))
        omega = numpyro.sample('omega', distx.Angle())#numpyro.deterministic('omega', jnp.arctan2(hk[1], hk[0]))
        # Time of periastron passage
        #phase_peri = numpyro.sample('phase_peri', dist.Uniform(-0.5,0.5))
        phi = numpyro.sample("phi", numpyro_ext.distributions.Angle())
        phase_peri = numpyro.deterministic('phase_peri', phi/(2*jnp.pi))#(((phi*period/(2*jnp.pi) - 2457000 + t0_as)) - 6*period)/period)
        tp = numpyro.deterministic('tp', phase_peri*period + 2457000)
        # Eccentricity and argument of periastron
        #hk = numpyro.sample('hk', distx.UnitDisk())

        # Inclination
        cosi = numpyro.sample('cosi', dist.Uniform(-1., 1.))
        i = numpyro.deterministic('i', jnp.arccos(cosi))
        sini3 = numpyro.deterministic('sini3', jnp.sin(i)**3)

        # Longitude of ascending node
        long_peri = numpyro.sample('long_peri', distx.Angle())
        i_ratio = numpyro.deterministic('i_ratio', 1.0 - ((((mp/ms)**3.5)*(1+mp/ms))/((mp/ms)*(1. + (mp/ms)**3.5))))#1.0 - (((mlrv(mp)/mlrv(ms))*(1+mp/ms))/((mp/ms)*(1. + mlrv(mp)/mlrv(ms)))))
        # Define the orbit
        rstar = 1.0
        host = Central(mass=ms,radius=rstar)
        system = System(host).add_body(period=period,mass=mp,time_peri=tp,inclination=i,eccentricity=ecc,omega_peri=omega,asc_node=long_peri,parallax=parallax)
        def binary_as(t_as, system):
            xs, ys, zs = system.central_position(t_as)
            ra = numpyro.deterministic('ra', ys.magnitude[0]*i_ratio)
            dec = numpyro.deterministic('dec', xs.magnitude[0]*i_ratio)
            # ys in RA direction and xs in Dec direction
            return (ys.magnitude[0]*spsi + xs.magnitude[0]*cpsi)*i_ratio
        
        model_binary = numpyro.deterministic('model_binary', binary_as(t_as, system))
        # Likelihood
        # For plotting
        t_plot = jnp.linspace(0, 5000, 1000)
        xs_plot, ys_plot, _ = system.central_position(t_plot)
        ra_plot = numpyro.deterministic('ra_plot', ys_plot.magnitude[0]*i_ratio)
        dec_plot = numpyro.deterministic('dec_plot', xs_plot.magnitude[0]*i_ratio)


        log_lc = numpyro.sample("log_lc", numpyro.distributions.Normal(jnp.log(jnp.nanstd(lc)), 1))
        mass_func = numpyro.deterministic("mass_func",(mp**3 * sini3)/((mp+ms)**2))
        #bounded_logasini = numpyro.distributions.Normal(jnp.log(500.), 5)
        #bounded_logasini.support = numpyro.distributions.constraints.interval(jnp.log(10), jnp.log(1000))
        #logasini = numpyro.sample("logasini", bounded_logasini)
        #asini = numpyro.deterministic("asini", jnp.exp(logasini))
        #lognu = numpyro.sample('lognu',numpyro.distributions.Normal(jnp.log(peaks), 0.001))
        nu = numpyro.sample("nu", numpyro.distributions.Uniform(peaks-0.0001,peaks+0.0001))
        mean = numpyro.sample("mean", numpyro.distributions.Normal(jnp.mean(lc), 1))

        asini = numpyro.deterministic('asini',((((period*86400)**2 * 6.67e-11 * (mass_func*1.989e30))/(4*jnp.pi**2))**(1/3))/(2.99792e8))

        orbit = Orbit(period=period, lighttime=asini, omega=omega, eccen=ecc, phi=phi, freq=nu)
        y_pred = orbit.get_lightcurve_model(t_ptv, lc) + mean
        #numpyro.deterministic("light_curve", y_pred)
        numpyro.sample("lc_logl", numpyro.distributions.Normal(y_pred, jnp.exp(log_lc)), obs=lc)
        numpyro.sample('hip_logl', dist.Normal(model_single + model_binary, jnp.sqrt(sig_w**2 + jnp.exp(log_sigma)**2)), obs=w)


    init_params = {
    "logP": jnp.log(period_guess),
    "phi": 0.05,
    "nu": peaks,
    "mean": jnp.mean(lcflux),
    "omega": 0.1,
    "eccen": 0.2,
    "delta_ra": 0.0,
    "delta_dec": 0.0,
    "pm_ra": 0.0,
    "pm_dec": 0.0,
    "log_parallax": jnp.log(0.033),
    "log_sigma": 0.0,
    "log_mp": jnp.log(0.1),
    "ms": 1.6,
    "mp": 0.9,
    "sini3": 0.8,
    "long_peri": 0.0,
    
    } 
    run_optim = numpyro_ext.optim.optimize(
        model, 
        optimizer=numpyro.optim.Adam(step_size=0.01), num_steps=1000, return_info=True,
        init_strategy=numpyro.infer.init_to_value(values=init_params)
    )
    opt_params, status = run_optim(jax.random.PRNGKey(2), [jnp.array(lctime),t_as], [jnp.array(lcfluxerr),sig_w], y=[jnp.array(lcflux),w+tot_meansoln])
    opt_params["period"] = jnp.exp(opt_params["logP"])
    #opt_params["asini"] = jnp.exp(opt_params["logasini"])
    #opt_params['nu'] = jnp.exp(opt_params["lognu"])
    for k, v in opt_params.items():
        if k in ["obs", "_b"]:
            continue
        print(f"{k}: {v}")
    print("optimizaton complete")


    #sample
    nstart=2000
    nsamples=4000
    sampler = numpyro.infer.MCMC(
    numpyro.infer.NUTS(
        model,
        dense_mass=True,
        regularize_mass_matrix=False,
        init_strategy=numpyro.infer.init_to_value(values=opt_params),
    ),
    num_warmup=nstart,
    num_samples=nsamples,
    num_chains=4,
    progress_bar=True,
    )
    sampler.run(jax.random.PRNGKey(1), [jnp.array(lctime),t_as], [jnp.array(lcfluxerr),sig_w], y=[jnp.array(lcflux),w+tot_meansoln])
    med_td_time = jnp.linspace(jnp.min(lctime), jnp.max(lctime), 10000)
    samples = sampler.get_samples()
    sampler.print_summary()
    print('------------------------------------------------------')
    sampler.print_summary(exclude_deterministic=False)

    nstart=2000
    nsamples=4000
    med_td_time = jnp.linspace(jnp.min(lctime), jnp.max(lctime), 10000)
    #plotting
    orbit_med = Orbit(period=jnp.median(jnp.exp(samples["logP"][nstart:])),
                      lighttime=jnp.median(samples["asini"][nstart:]),
                      omega=jnp.median(samples["omega"][nstart:]),
                      eccen=jnp.median(samples["eccen"][nstart:]),
                      phi=jnp.median(samples["phi"][nstart:]),
                      freq=jnp.median(samples["nu"][nstart:]),
                     )

    '''
    orbit_med = Orbit(period=opt_params['period'],
                      lighttime=opt_params['asini'],
                      omega=opt_params['omega'],
                      eccen=opt_params['eccen'],#jnp.median(samples["eccen"][nstart:]),
                      phi=opt_params['phi'],
                      freq=opt_params['nu'],
                     )
    med_td_time = jnp.linspace(jnp.min(lctime), jnp.max(lctime), 10000)
    '''
    med_td = orbit_med.get_time_delay(med_td_time)*86400
    fn = scipy.interpolate.interp1d(med_td_time,med_td.flatten()-np.median(med_td.flatten()))
    t,td = get_window_tds(lctime,lcflux, est_freqs, segment_size=5.0)
    window_td = np.average(td, axis=1, weights=get_weights(lctime,lcflux,est_freqs))

    z = np.polyfit(t,np.array(window_td-np.median(window_td)-fn(t)),1)
    resids = window_td-np.median(window_td)-z[0]*t-z[1] - fn(t)
    resids = resids - np.median(resids)
    outliers = jnp.abs(resids) >= 2.5*scipy.stats.median_abs_deviation(resids)
    z = jnp.polyfit(t,np.array(window_td-np.median(window_td)-fn(t)),1)
    z_2 = jnp.polyfit(t[~outliers],np.array(window_td-np.median(window_td)-fn(t))[~outliers],1)
    window_time_final = t[~outliers]
    window_td_final = np.array(window_td-np.median(window_td)-z_2[0]*t-z_2[1])[~outliers]
    ax = td_plotter(window_time_final,window_td_final, med_td_time,med_td.flatten()-np.median(med_td.flatten()))

    for i in range(50):
        random_idx = np.random.randint(nstart,nsamples+nstart-1)
        sampled_orbit = Orbit(
            period=jnp.exp(samples["logP"][random_idx]),
            lighttime=samples["asini"][random_idx],
            omega=samples["omega"][random_idx],
            eccen=samples["eccen"][random_idx],
            phi=samples["phi"][random_idx],
            freq=samples["nu"][random_idx],
        )
        sampled_td = sampled_orbit.get_time_delay(med_td_time)*86400
        ax = td_plotter([],[], med_td_time,sampled_td.flatten()-np.median(med_td.flatten()),alpha=0.1,ax=ax)
    
    plt.title("Alpha Pictoris Phase Modulations")
    plt.savefig(f'{tic}_gpu_test_sampled.png',dpi=500)
    plt.title("Alpha Pictoris")
    plt.savefig(f'{tic}_gpu.png',dpi=500)
    cpus = jax.devices("cpu")
    sampler._states = jax.device_put(sampler._states, cpus[0])
    sampler._kwargs = jax.device_put(sampler._kwargs, cpus[0])
    inf_data = az.from_numpyro(sampler)
    plt.close('all')
    fig = plt.figure()

    plt.plot(0, 0, marker='x', c='k')

    ra_plot = inf_data.posterior['ra_plot'].values
    dec_plot = inf_data.posterior['dec_plot'].values
    # 300 draws from the posterior
    for i in range(50):
        m = np.random.randint(0, ra_plot.shape[0])
        n = np.random.randint(0, ra_plot.shape[1])
        plt.plot(ra_plot[m,n,:], dec_plot[m,n,:], lw=0.5, c="#7139AC", alpha=0.5)

    # plot "data"

    jitter = np.exp(np.median(inf_data.posterior.log_sigma.values.flatten()))
    delta_ra_mean = np.median(inf_data.posterior.delta_ra.values.flatten())
    delta_dec_mean = np.median(inf_data.posterior.delta_dec.values.flatten())
    pm_ra_mean = np.median(inf_data.posterior.pm_ra.values.flatten())
    pm_dec_mean = np.median(inf_data.posterior.pm_dec.values.flatten())
    plx_mean = np.median(inf_data.posterior.parallax.values.flatten())

    tot_meanposition = single_star_soln(t_as-t0_as, delta_ra_mean, delta_dec_mean, pm_ra_mean, pm_dec_mean, plx_mean, pf, spsi, cpsi)
    mean_data = w+tot_meansoln-tot_meanposition

    as_pred_total = inf_data.posterior['model_single'].values + inf_data.posterior['model_binary'].values
    q16_total, q50_total, q84_total = np.percentile(as_pred_total, [16, 50, 84], axis=(0, 1))
    total_residual = w-q50_total+tot_meansoln

    mean_ra_data = total_residual*spsi + np.median(inf_data.posterior.ra.values,axis=[1,0])
    mean_dec_data = total_residual*cpsi + np.median(inf_data.posterior.dec.values,axis=[1,0])
    plt.plot(mean_ra_data, mean_dec_data, marker='o', linestyle=' ', c='k', ms=2)
    fine_t = np.linspace(0,5000,5000)
    var_names = ['delta_ra', 'delta_dec', 'pm_ra', 'pm_dec', 'log_parallax', 'parallax', 'log_sigma', 'log_mp', 'mp', 'ms','period', 'eccen', 'omega', 'cosi', 'i','i_ratio', 'long_peri', 'tp']
    summary = az.summary(inf_data, var_names=var_names, round_to=8)
    print('------------------------------------------------------')
    print(summary)
    means = summary['mean']
    mper, mtp, mi, mecc, momega, mcosi, mlong_peri, mmp, mi_ratio, mms = means['period'], means['tp'], means['i'], means['eccen'], means['omega'], means['cosi'], means['long_peri'], means['mp'],means['i_ratio'],means['ms']
    host = Central(mass=mms,radius=1.0)
    msystem = System(host).add_body(period=mper,mass=mmp,time_peri=mtp,inclination=mi,eccentricity=mecc,omega_peri=momega,asc_node=mlong_peri,parallax=plx_mean)
    dec_fine, ra_fine, _ = msystem.central_position(fine_t)
    plt.plot(ra_fine.magnitude[0]*mi_ratio,dec_fine.magnitude[0]*mi_ratio, c="#7139AC",lw=1)
    for i in range(len(t_as)):
        this_ra = inf_data.posterior.ra.values[:,:,i].flatten()
        this_dec = inf_data.posterior.dec.values[:,:,i].flatten()

        # median position and uncertainty out of model
        med_ra = np.median(this_ra)
        med_dec = np.median(this_dec)

        sd_ra = np.sqrt(sig_w[i]**2 + jitter**2)*spsi[i]
        sd_dec = np.sqrt(sig_w[i]**2 + jitter**2)*cpsi[i]
        plt.plot([mean_ra_data[i], med_ra], [mean_dec_data[i], med_dec], c='k', lw=0.4)      
        #plt.plot([med_ra - sd_ra, med_ra + sd_ra], [med_dec - sd_dec, med_dec + sd_dec], c='k', lw=0.5)      
        #plt.plot(med_ra, med_dec, marker='o', linestyle=' ', c='r', ms=2)

    plt.xlabel('$\Delta\\alpha \cos \delta$ (arcsec)')
    plt.ylabel('$\Delta \delta$ (arcsec)')

    plt.gca().invert_xaxis()
    plt.gca().set_aspect('equal')
    #plt.legend()
    plt.title("Alpha Pictoris Astrometry")
    plt.savefig('alpha_pic_orbit.png', dpi=300,bbox_inches='tight')
    #inf_data.to_netcdf(f'{tic}_gpu_test.h5')

    #plt.savefig(f'{tic}_hipparcos_gpu_test.png',dpi=500)
if __name__ == '__main__':
    tic = '160582982'
    tic = '167602316'
    jaxstrom(tic)

