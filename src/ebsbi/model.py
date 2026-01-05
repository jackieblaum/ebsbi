import phoebe
import numpy as np
import scipy.stats
import eclipsebin as ebin

import matplotlib.pyplot as plt
import binarysed

from .constants import TWIG_DICT, C_LIGHT, Z_SOLAR, R_V, STEFAN_BOLTZ, G_SUN
from .stellar_utils import z_to_feh, interpolate_isochrone
from .observational import CadenceNoiseSampler, survey_noise_process

# =======================
# MODEL CLASS
# =======================

class EBModel:
    """
    Encapsulates the PHOEBE eclipsing binary model and related utilities.
    """

    def __init__(self, eb_path, params_dict, phase_bank, noise_bank, amortized=False, rng=None, system_weights="length"):
        self.eb_path = eb_path
        self.params_dict = params_dict
        self.amortized = amortized
        self.rng = rng
        self.sed_engine = binarysed.SED(None)

        self.cadence_noise_sampler = CadenceNoiseSampler(
            phase_bank,
            noise_bank,
            log_noise=True,
            seed=rng,
            system_weights=system_weights,
        )

        self._lc_system_ids = {}  # maps dataset_label -> (survey, system_idx)
        self._lc_dataset_labels = []

    def _passband_to_survey(self, passband):
        """
        Map a PHOEBE passband string to a survey name used in the phase/noise banks.
        Adjust this logic to match your actual passband naming.
        """
        pb = str(passband)
        if "TESS" in pb:
            return "TESS"
        if "KEPLER" in pb:
            return "Kepler"
        if "GAIA" in pb:
            return "Gaia"
        elif pb=='SDSS:g':
            return 'ASASSN_g'
        elif pb=='Johnson:V':
            return 'ASASSN_V'
        elif pb=='ZTF:g':
            return 'ZTF_zg'
        elif pb=='ZTF:r':
            return 'ZTF_zr'
        # Fallback: tweak as needed
        return "TESS"

    def create_phoebe_bundle(self, theta, passbands=['SDSS:g', 'Johnson:V', 'ZTF:g', 'ZTF:r']):
        b = phoebe.default_binary()
        self._set_phoebe_parameters(b, theta)
        self._configure_phoebe(b, theta, passbands)
        return b

    def get_lc_system_ids(self):
        """
        Return mapping from dataset_label -> (survey, system_idx)
        for the most recently created PHOEBE bundle.
        """
        return dict(self._lc_system_ids)

    def choose_nbins_from_npoints(self, N, nbins_max=200, nbins_min=10, points_per_bin=5):
        """
        Choose nbins so eclipsebin's constraint N >= points_per_bin * nbins is satisfied.
        """
        N = int(N)
        if N <= 0:
            return 0  # caller should handle
        nb = N // points_per_bin
        nb = min(nb, int(nbins_max))
        nb = max(nb, int(nbins_min))
        # If even nbins_min violates the constraint, signal "can't eclipsebin"
        if N < points_per_bin * nbins_min:
            return 0
        return nb


    def nbi_simulator(self, theta, path):
        """
        Simulator wrapper for NBI.

        For a single parameter vector `theta`, this:
          - builds the PHOEBE model
          - generates a (binned) light curve
          - generates an SED
          - builds a metadata vector
          - creates masks (all ones for now; can be randomized later)
          - saves everything to `path` as a single .npy object

        Parameters
        ----------
        theta : array-like, shape (D,)
            Parameter vector (one sample).
        path : str
            File path to save the simulated data.

        Returns
        -------
        bool
            True if simulation succeeded, False otherwise (e.g., PHOEBE failure).
        """
        # Turn 1D theta into the dict form your helpers expect
        theta = np.atleast_2d(theta)
        theta_dict = {
            name: val for name, val in zip(self.params_dict.keys(), np.array(theta).T)
        }

        try:
            # -------------------------
            # 1) Light curve(s)
            # -------------------------
            # For now: one LC channel produced by PHOEBE
            # (Later return a list of LCs here.)
            lc_phase, lc_flux, lc_err = self.generate_light_curve(theta, nbins=200)

            # -------------------------
            # 2) SED
            # -------------------------
            wavelengths, sed_flux, sed_err = self.generate_sed(theta)
            sed_flux = np.asarray(sed_flux, dtype=np.float32)
            sed_err  = np.asarray(sed_err,  dtype=np.float32)

            # -------------------------
            # 3) Metadata
            # -------------------------
            meta = []
            # meta.append(float(theta_dict["period"]))       # period not in params_dict
            meta.append(float(theta_dict["distance"]))
            meta = np.asarray(meta, dtype=np.float32)

            # -------------------------
            # 4) Masks (all ones for now)
            # -------------------------
            mask_lc = [np.ones_like(ch, dtype=np.float32) for ch in lc_flux]
            mask_sed = np.ones_like(sed_flux, dtype=np.float32)
            mask_meta = np.ones_like(meta, dtype=np.float32)


            # -------------------------
            # 5) Pack into a single object and save
            # -------------------------
            lc_system_ids = dict(self._lc_system_ids)

            x_obj = {
                "lc_phase": lc_phase,
                "lc": lc_flux,
                "lc_err": lc_err,
                "sed": sed_flux,
                "sed_err": sed_err,
                "meta": meta,
                "mask_lc": mask_lc,
                "mask_sed": mask_sed,
                "mask_meta": mask_meta,
                "lc_system_ids": lc_system_ids,
                "lc_dataset_labels": list(self._lc_dataset_labels),
            }

            np.save(path, x_obj, allow_pickle=True)
            return True

        except Exception as e:
            print(f"nbi_simulator() failed for path={path}, returning False")
            print(e)
            # Save a sentinel so BaseContainer can skip or treat as bad
            x_obj = None
            np.save(path, x_obj, allow_pickle=True)
            return False

    def _bin_with_eclipsebin(self, phases, fluxes, fluxerrs, nbins=200,
                         fraction_in_eclipse=0.5, atol_primary=0.001, atol_secondary=0.05,
                         plot=False):
        binner = ebin.EclipsingBinaryBinner(
            phases, fluxes, fluxerrs,
            nbins=nbins,
            fraction_in_eclipse=fraction_in_eclipse,
            atol_primary=atol_primary,
            atol_secondary=atol_secondary,
        )
        bin_centers, bin_means, bin_errs = binner.bin_light_curve(plot=plot)
        return (np.asarray(bin_centers, dtype=np.float32),
                np.asarray(bin_means, dtype=np.float32),
                np.asarray(bin_errs, dtype=np.float32))

    def generate_light_curve(self, theta, name=None, nbins=200):
        """
        Generate synthetic light curves from PHOEBE model with normalized fluxes.

        Parameters
        ----------
        theta : array-like or dict
            Parameter vector or dictionary. If array, will be converted to dict
            using self.params_dict.keys(). Should contain all required PHOEBE
            parameters (period, q, teff1, teff2, r1, r2, etc.).
        name : str, optional
            Name for the light curve (currently unused).
        nbins : int, default=200
            Maximum number of bins for eclipsebin. Actual number may be lower
            based on data density (see choose_nbins_from_npoints).

        Returns
        -------
        phases_passbands : list of ndarray
            Phase arrays for each passband, shape (nbins_eff,) each.
        fluxes_passbands : list of ndarray
            Normalized flux arrays for each passband, shape (nbins_eff,) each.
            **Fluxes are normalized to median=1 for each passband.**
        errs_passbands : list of ndarray
            Flux uncertainty arrays for each passband, shape (nbins_eff,) each.
            **Errors are scaled by the same normalization factor as fluxes.**

        Notes
        -----
        - Phases are sorted in ascending order
        - Binning is performed using eclipsebin with adaptive bin selection
        - Noise is sampled from cadence_noise_sampler based on survey/system
        - Each passband is normalized independently to median flux = 1.0
        - Error propagation: if flux is divided by median M, errors are also
          divided by M to maintain correct fractional uncertainties

        Examples
        --------
        >>> theta = {'period': 2.5, 'q': 0.8, 'teff1': 6000, ...}
        >>> phases, fluxes, errs = model.generate_light_curve(theta, nbins=200)
        >>> np.median(fluxes[0])  # Should be 1.0
        1.0
        """
        theta_dict = {name: val for name, val in zip(self.params_dict.keys(), np.array(theta).T)}
        b = self.create_phoebe_bundle(theta_dict)
        b.run_compute()

        phases_passbands = []
        fluxes_passbands = []
        errs_passbands   = []

        for dataset_label in self._lc_dataset_labels:
            phases = b.get_value(f"compute_phases@dataset@{dataset_label}")
            fluxes = b.get_value(f"fluxes@model@{dataset_label}")

            idx = np.argsort(phases)
            phases = np.asarray(phases[idx], dtype=np.float32)
            fluxes = np.asarray(fluxes[idx], dtype=np.float32)

            # get prototype system for this dataset and sample *unbinned* sigmas
            survey, system_idx = self._lc_system_ids[dataset_label]
            sigmas = self.cadence_noise_sampler.sample_noise_for_system(
                survey, system_idx, n=len(fluxes), rng=self.rng
            ).astype(np.float32)

            N = len(phases)
            nbins_eff = self.choose_nbins_from_npoints(N, nbins_max=nbins, nbins_min=10, points_per_bin=5)

            # Normalize unbinned fluxes to help eclipsebin find eclipse boundaries
            # eclipsebin expects flux baseline near 1.0
            flux_median_unbinned = np.median(fluxes)
            fluxes_for_binning = fluxes / flux_median_unbinned
            sigmas_for_binning = sigmas / flux_median_unbinned

            # bin to <= nbins and propagate sigmas -> binned sigmas via eclipsebin
            ph_b, fl_b, er_b = self._bin_with_eclipsebin(
                phases, fluxes_for_binning, sigmas_for_binning, nbins=nbins_eff, plot=False
            )

            # Normalize fluxes to median=1 and scale errors accordingly
            # Even though unbinned data was normalized, binning may change the median slightly
            flux_median = np.median(fl_b)
            fl_b_normalized = fl_b / flux_median
            er_b_normalized = er_b / flux_median

            phases_passbands.append(ph_b)
            fluxes_passbands.append(fl_b_normalized.astype(np.float32))
            errs_passbands.append(er_b_normalized.astype(np.float32))

        return phases_passbands, fluxes_passbands, errs_passbands

    def generate_sed(self, theta, sed_frac_err=0.02):
        theta_dict = {name: val for name, val in zip(self.params_dict.keys(), np.array(theta).T)}

        teff1 = theta_dict["teff1"][0]
        r1    = theta_dict["r1"][0]
        teff2 = theta_dict["teff2"][0]
        r2    = theta_dict["r2"][0]
        msum  = theta_dict["msum"][0]
        q     = theta_dict["q"][0]
        m1 = msum / (q + 1)
        m2 = q * m1
        g1 = m1 * G_SUN / (r1**2)
        g2 = m2 * G_SUN / (r2**2)
        logg1 = np.log10(g1)
        logg2 = np.log10(g2)
        dist = theta_dict["distance"][0]
        ebv  = theta_dict["ebv"][0]

        fluxes, wavelengths = self.sed_engine.create_apparent_sed(
            None, teff1, teff2, r1, r2, logg1, logg2, dist=dist, ebv=ebv
        )
        sed_flux = np.asarray(fluxes, dtype=np.float32)
        wavelengths = np.asarray(wavelengths, dtype=np.float32)

        # "reported" uncertainties for SED points (same length as sed_flux)
        sed_err = (sed_frac_err * np.abs(sed_flux)).astype(np.float32)

        return wavelengths, sed_flux, sed_err

    # def noise_fn(self, flux, params):
    #     noise_level=0.01
    #     noise = np.random.normal(0, noise_level, size=flux.shape)
    #     return flux * (1 + noise), params


    def log_likelihood(self, observed_flux_file, model_flux_file, error_scale=0.01):
        try:
            model_flux = np.load(model_flux_file)
        except Exception:
            return -np.inf
        
        try:
            observed_flux = np.load(observed_flux_file)
        except Exception:
            return -np.inf
        
        try:
            model_flux = np.asarray(model_flux, dtype=float)
        except Exception:
            return -np.inf

        err = error_scale * model_flux
        chi2 = np.sum(((observed_flux - model_flux) / err) ** 2)
        return -0.5 * chi2

    def _set_phoebe_parameters(self, b, theta):
        b.flip_constraint('mass@primary', solve_for='period@binary')
        b.flip_constraint('requivsumfrac@binary', solve_for='sma@binary')

        # PHOEBE doesn't have a parameter corresponding to msum, so calculate m1
        msum = float(theta['msum'])
        q = float(theta['q'])
        m1 = msum / (q + 1)
        m1_twig = TWIG_DICT.get('m1')
        b.set_value(m1_twig, value=m1)
        print(f'Set {m1_twig}')

        # Binary stars should have approximately same metallicity
        feh = float(theta['metallicity'])
        b.set_value('abun@primary', value=feh)
        b.set_value('abun@secondary', value=feh)

        for param, value in theta.items():
            if param not in ['mag', 'times', 'msum', 'metallicity', 'log_age', 'ebv']:
                twig = TWIG_DICT.get(param)
                if twig:
                    b.set_value(twig, value=float(value))
                    print(f'Set {twig}')

    def _configure_phoebe(self, b, theta, passbands):
        # Gravity darkening, irradiation, limb darkening logic
        for param_name, star in zip(['teff1', 'teff2'], ['primary', 'secondary']):
            teff = theta[param_name]
            if teff <= 3700:
                gravb = 0.32
            elif 3700 < teff < 7000:
                poly = np.polynomial.polynomial.Polynomial([-5.00784, 27.9838, -46.9701, 25.5398])
                gravb = float(poly(float(np.squeeze(teff)) * 1e-4))
            elif 7000 <= teff < 8000:
                gravb = 0.9
            else:
                gravb = 1.0
                b.set_value(f'irrad_frac_refl_bol@{star}', value=1.0)

            b.set_value(f'gravb_bol@{star}', value=gravb)

        self._lc_system_ids = {}
        self._lc_dataset_labels = []

        for i, passband in enumerate(passbands):
            survey = self._passband_to_survey(passband)

            system_idx = self.cadence_noise_sampler.choose_system(
                survey,
                rng=self.rng,
            )

            phases = self.cadence_noise_sampler.sample_phases_for_system(
                survey,
                system_idx,
                rng=self.rng,
                sort=True,
            )

            dataset_label = f"lc_{survey}_{i}"
            self._lc_dataset_labels.append(dataset_label)

            b.add_dataset(
                'lc',
                compute_phases=phases,
                passband=f"{passband}",
                dataset=dataset_label,
            )

            self._lc_system_ids[dataset_label] = (survey, system_idx)

        # Adjust atmosphere and limb darkening for edge cases
        for star in ['primary', 'secondary']:
            logg = b.get_value(f'logg@{star}@component')
            teff = b.get_value(f'teff@{star}@component')

            if logg > 5 or teff > 8000:
                b.set_value(f'ld_mode@{star}', 'manual')
                b.set_value(f'ld_mode_bol@{star}', 'manual')
                b.set_value(f'atm@{star}', 'blackbody')

            if teff < 3500:
                b.set_value(f'ld_mode@{star}', 'manual')
                b.set_value(f'ld_mode_bol@{star}', 'manual')
                b.set_value(f'atm@{star}', 'extern_atmx')

    # def read_files(self, sim_file, lookup_file, isochrone_file, iso_header_start=0):
    #     """
    #     Load simulation, lookup, and isochrone files.
    #     """
    #     logger.info("Reading input files.")
    #     with open(sim_file, "r") as f:
    #         self.data_dict = json.load(f)
    #     self.lookup_table = Table.read(lookup_file, format="ascii")
    #     self.isochrone = Table.read(isochrone_file, format="ascii", header_start=iso_header_start)
    #     logger.info("Files successfully loaded.")

def instrumental_noise(cadence_noise_sampler, *, nbins=200, rng=None):
    """
    Returns a function process(x, y) that NBI can call,
    with cadence_noise_sampler already bound.
    """
    if rng is None:
        rng = np.random.default_rng()

    return partial(
        survey_noise_process,
        cadence_noise_sampler=cadence_noise_sampler,
        nbins=nbins,
        rng=rng,
    )