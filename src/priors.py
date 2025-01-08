import pickle
import json
import pymc as pm
import numpy as np
import astropy.units as u
from astropy.table import Table
from isochrones import get_ichrone
from binarysedfit import misc_utils
import logging

# Constants
G = 2942.206218  # Gravitational constant in R_sun^3/(days^2 * M_sun)
gsun = G * 9.319541  # Surface gravity of the Sun in solar units
sigma_boltz = 5.670374419e-8 * u.W / u.m**2 / u.K**4
Z_SOLAR = 0.02  # Solar metallicity for Z -> [Fe/H] conversion
R_V = 3.1  # Reddening law constant

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Priors:
    def __init__(self, params_dict, trace_dict, amortized=False):
        self.params_dict = params_dict
        self.trace_dict = trace_dict
        self.amortized = amortized
        self.params_list = list(params_dict.keys())
        self.data_dict = None
        self.lookup_table = None
        self.isochrone = None
        self.trace = None
        self.model = None
        self.rng = np.random.default_rng(0)
        self.tracks = get_ichrone("mist", tracks=True)
        logger.info("Priors class initialized.")

    def read_files(self, sim_file, lookup_file, isochrone_file, iso_header_start=0):
        """
        Load simulation, lookup, and isochrone files.
        """
        logger.info("Reading input files.")
        with open(sim_file, "r") as f:
            self.data_dict = json.load(f)
        self.lookup_table = Table.read(lookup_file, format="ascii")
        self.isochrone = Table.read(isochrone_file, format="ascii", header_start=iso_header_start)
        logger.info("Files successfully loaded.")

    def create_dists(self):
        """
        Create distributions for priors using PyMC.
        """
        logger.info("Creating prior distributions.")
        with pm.Model() as model:
            msum = pm.TruncatedNormal("msum", mu=2, sigma=7, lower=0.8, upper=24)
            q = pm.Triangular("q", lower=0.05, upper=1.0, c=1)
            m1 = msum / (q + 1)
            m2 = q * m1

            log_age = pm.TruncatedNormal("log_age", mu=9.5, sigma=2, lower=6, upper=-2.4 * np.log10(m1) + 10.3)
            metallicity = pm.Uniform("metallicity", lower=0.001, upper=0.06)
            ebv = pm.Uniform("ebv", lower=0, upper=3)
            incl = pm.TruncatedNormal("incl", mu=85, sigma=15, lower=45, upper=90)
            rsumfrac = pm.TruncatedNormal("rsumfrac", mu=0.35, sigma=0.2, lower=np.sin(np.radians(90 - incl)), upper=0.7)
            ecc = pm.TruncatedNormal("ecc", mu=0, sigma=0.2, lower=0, upper=1 - rsumfrac)
            per0 = pm.Uniform("per0", lower=0, upper=360)
            distance = pm.TruncatedNormal("distance", mu=4000, sigma=3000, lower=10, upper=14000)

            self.trace = pm.sample(
                draws=self.trace_dict["DRAWS"],
                chains=self.trace_dict["CHAINS"],
                target_accept=0.9,
                random_seed=1,
            )
            self.model = model
            pickle.dump(
                {"model": model, "trace": self.trace},
                open(self.trace_dict["FILE_PATH"], "wb"),
                protocol=4,
            )
        logger.info("Distributions created and saved.")

    def prior_sample(self, n):
        """
        Sample parameters from prior distributions.
        """
        logger.info(f"Sampling {n} prior distributions.")
        samples = pm.sample_posterior_predictive(self.trace, model=self.model, var_names=self.params_list, samples=n)
        sampled_dict = {param: samples[param] for param in self.params_list}
        logger.info("Prior samples generated.")
        return sampled_dict

    def _isochrone_interpolate(self, mass, log_age, feh):
        """
        Interpolate over isochrones to get stellar parameters based on mass.
        """
        logger.info(f"Interpolating isochrone for mass={mass}, log_age={log_age}, feh={feh}.")
        eeps = self.tracks.get_eep(mass, log_age, feh)
        logTeff, radius, logL = self.tracks.interp_value([mass, eeps, feh], ["logTeff", "radius", "logL"]).T
        teff = 10 ** logTeff
        return teff, radius, logL

    def _calculate_requiv(self, teff, lum):
        """
        Calculate stellar radius using the Stefan-Boltzmann law.
        """
        requiv = np.sqrt(lum * u.W / (4 * np.pi * sigma_boltz * (teff * u.K) ** 4))
        return requiv.to(u.R_sun).value

    def _z_to_feh(self, z):
        """
        Convert metallicity Z to [Fe/H].
        """
        return np.log10(z / Z_SOLAR)

    def log_prior(self, thetas):
        """
        Compute log prior probability of a parameter set.
        """
        try:
            logp = self.model.logp(thetas)
            return logp
        except Exception as e:
            logger.error(f"Error computing log prior: {e}")
            return -np.inf

    def generate_seds(self, samples, filters_dict):
        """
        Generate SEDs for sampled parameters.
        """
        logger.info("Generating SEDs.")
        seds = []
        for sample in samples:
            teff1, lum1, logg1 = self._isochrone_interpolate(sample["m1"], sample["log_age"], self._z_to_feh(sample["metallicity"]))
            teff2, lum2, logg2 = self._isochrone_interpolate(sample["m2"], sample["log_age"], self._z_to_feh(sample["metallicity"]))
            sed = misc_utils.binary_star_kurucz_sed(teff1, lum1, logg1, teff2, lum2, logg2, sample["distance"], sample["metallicity"], sample["ebv"], R_V)
            seds.append(sed)
        logger.info("SED generation completed.")
        return seds
