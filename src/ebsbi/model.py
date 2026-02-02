import phoebe
import numpy as np
import scipy.stats
import eclipsebin as ebin
from functools import partial

import matplotlib.pyplot as plt

from .constants import TWIG_DICT, C_LIGHT, Z_SOLAR, R_V, STEFAN_BOLTZ, G_SUN
from .stellar_utils import z_to_feh, interpolate_isochrone
from .observational import CadenceNoiseSampler, survey_noise_process, bin_light_curve
from .phoebe_wrapper import PhoebeWrapper

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
        self.rng = np.random.default_rng(rng)

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
        Choose nbins ensuring sufficient points per bin for robust binning.

        This enforces EBSBI's binning quality constraint: each bin should have
        at least `points_per_bin` data points on average.

        Note: EBSBI uses realistic survey cadences from phase/noise banks.
        - If a passband has <= nbins observations, data is kept unbinned
        - If a passband has > nbins observations, eclipsebin bins it
        - eclipsebin uses min_total_points=10 (relaxed constraint)
        - When very sparse (avg points/bin < 2), eclipsebin uses adaptive binning
        This function calculates appropriate nbins when you need to ensure
        binning quality for a specific observation count.

        Parameters
        ----------
        N : int
            Number of data points
        nbins_max : int, default=200
            Maximum bins to request
        nbins_min : int, default=10
            Minimum bins to request
        points_per_bin : int, default=5
            Target average points per bin

        Returns
        -------
        int
            Chosen nbins, or 0 if insufficient data (N < points_per_bin * nbins_min)
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

    def simulate_gaia_parallax(self, sed_flux, distance_pc, gaia_coverage=0.7, filter_mask=None):
        """
        Simulate a Gaia parallax measurement with realistic uncertainties.

        Based on Gaia DR3 parallax uncertainty model as a function of G magnitude.
        Gaia DR3 has ~70% sky coverage and provides parallaxes for G < 21 mag.

        Parameters
        ----------
        sed_flux : array
            SED fluxes in erg/cm²/s for all filters
        distance_pc : float
            True distance in parsecs
        gaia_coverage : float
            Probability that Gaia observed this system (default 0.7)
        filter_mask : array, optional
            Boolean mask indicating which filters are valid (1=valid, 0=NaN)

        Returns
        -------
        parallax_obs : float
            Observed parallax in mas (or NaN if not in Gaia)
        parallax_err : float
            Parallax uncertainty in mas (or NaN if not in Gaia)
        distance_obs : float
            Distance derived from parallax in pc (or NaN if not in Gaia)
        distance_err : float
            Distance uncertainty in pc (or NaN if not in Gaia)
        has_gaia : bool
            Whether this system has Gaia data
        """
        # Check if system is in Gaia coverage
        if self.rng is not None:
            has_gaia = self.rng.random() < gaia_coverage
        else:
            has_gaia = np.random.random() < gaia_coverage

        if not has_gaia:
            return np.nan, np.nan, np.nan, np.nan, False

        # Estimate Gaia G magnitude from total SED flux
        # Pick a valid optical filter to use as G-band proxy
        # Prefer middle of SED (optical range), but fall back to any valid filter

        g_flux = None
        if len(sed_flux) > 0:
            if filter_mask is not None:
                # Find valid filters
                valid_indices = np.where(filter_mask > 0)[0]
                if len(valid_indices) > 0:
                    # Prefer middle filter (likely optical)
                    mid_idx = len(valid_indices) // 2
                    g_flux = sed_flux[valid_indices[mid_idx]]
            else:
                # No mask provided, use middle filter (old behavior)
                mid_idx = len(sed_flux) // 2
                g_flux = sed_flux[mid_idx]

        # Convert flux to AB magnitude
        if g_flux is not None and g_flux > 0 and not np.isnan(g_flux):
            # AB mag = -2.5 * log10(flux) - 48.60 (for flux in erg/cm²/s/Hz)
            # Rough conversion for broadband flux
            g_mag = -2.5 * np.log10(g_flux) - 10.0  # Empirical offset
            # Clamp to reasonable range
            g_mag = np.clip(g_mag, 8.0, 21.0)
        else:
            # If flux is invalid or no valid filters, assume faint star
            g_mag = 18.0

        # Gaia DR3 parallax uncertainty model
        # Based on empirical relations: σ_ϖ ≈ 0.02 * 10^(0.4*(G-15)) mas
        # This gives:
        #   G=12: σ ≈ 0.013 mas (bright stars)
        #   G=15: σ ≈ 0.02 mas  (good precision)
        #   G=18: σ ≈ 0.063 mas (moderate precision)
        #   G=20: σ ≈ 0.20 mas  (faint stars)

        if g_mag < 21.0:
            # Good Gaia parallax
            parallax_err = 0.02 * 10**(0.4 * (g_mag - 15.0))  # mas

            # Add magnitude-dependent systematic floor
            parallax_err = np.sqrt(parallax_err**2 + 0.01**2)  # Add 0.01 mas systematic
        else:
            # Too faint for good Gaia parallax
            return np.nan, np.nan, np.nan, np.nan, False

        # True parallax
        parallax_true = 1000.0 / distance_pc  # Convert pc to mas

        # Add Gaussian noise
        if self.rng is not None:
            parallax_obs = parallax_true + self.rng.normal(0, parallax_err)
        else:
            parallax_obs = parallax_true + np.random.normal(0, parallax_err)

        # Convert back to distance with proper error propagation
        if parallax_obs > 0:
            distance_obs = 1000.0 / parallax_obs  # pc
            # Error propagation: σ_d = d² * σ_ϖ / ϖ (for small relative errors)
            # More accurate: σ_d = (1000/ϖ²) * σ_ϖ
            distance_err = (1000.0 / parallax_obs**2) * parallax_err  # pc
        else:
            # Negative parallax (can happen with noise) - system too faint/distant
            return np.nan, np.nan, np.nan, np.nan, False

        return parallax_obs, parallax_err, distance_obs, distance_err, True


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

        Metadata Structure
        ------------------
        The metadata vector contains:
          [0] period - Known orbital period (days)
          [1] distance_gaia - Noisy Gaia distance (pc) or NaN
          [2] distance_err - Gaia distance uncertainty (pc) or NaN
          [3] parallax_gaia - Noisy Gaia parallax (mas) or NaN
          [4] parallax_err - Gaia parallax uncertainty (mas) or NaN
          [5:5+N] n_obs - Pre-binning observation counts for N passbands

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
            # generate_light_curve() now returns unbinned data
            # We explicitly bin here using bin_light_curve() from observational.py
            nbins = 200
            lc_phase_unbinned, lc_flux_unbinned, lc_err_unbinned, n_obs = self.generate_light_curve(theta, nbins=nbins)

            # Bin each passband if L > nbins
            lc_phase = []
            lc_flux = []
            lc_err = []

            for phases, fluxes, sigmas in zip(lc_phase_unbinned, lc_flux_unbinned, lc_err_unbinned):
                L = len(phases)
                if L > nbins:
                    # Bin using centralized function from observational.py
                    ph_b, fl_b, er_b = bin_light_curve(
                        phases, fluxes, sigmas,
                        nbins=nbins,
                        fraction_in_eclipse=0.5,
                        atol_primary=0.001,
                        atol_secondary=0.05,
                        plot=False
                    )
                    lc_phase.append(ph_b)
                    lc_flux.append(fl_b)
                    lc_err.append(er_b)
                else:
                    # Already at or below target: use as-is
                    lc_phase.append(phases)
                    lc_flux.append(fluxes)
                    lc_err.append(sigmas)

            # Align phases using eclipse minima
            # Find the phase of minimum flux (primary eclipse) in each passband
            eclipse_midpoints = []
            for phases, fluxes, sigmas in zip(lc_phase, lc_flux, lc_err):
                # Find the phase at minimum flux
                min_idx = np.argmin(fluxes)
                eclipse_phase = phases[min_idx]
                eclipse_midpoints.append(eclipse_phase)

            # Use first passband's eclipse phase as reference
            reference_eclipse = eclipse_midpoints[0]

            # Shift all passbands to align with reference
            lc_phase_aligned = []
            for i, (phases, eclipse_phase) in enumerate(zip(lc_phase, eclipse_midpoints)):
                # Calculate shift to align this passband's eclipse with reference
                shift = eclipse_phase - reference_eclipse
                aligned_phases = (phases - shift) % 1.0
                lc_phase_aligned.append(aligned_phases)

            lc_phase = lc_phase_aligned

            # -------------------------
            # 2) SED
            # -------------------------
            wavelengths, sed_flux, sed_err, sed_mask = self.generate_sed(theta)
            sed_flux = np.asarray(sed_flux, dtype=np.float32)
            sed_err  = np.asarray(sed_err,  dtype=np.float32)
            sed_mask = np.asarray(sed_mask, dtype=np.float32)

            # -------------------------
            # 3) Metadata
            # -------------------------
            meta = []

            # Period is known from observations (not inferred)
            # Extracted from PHOEBE bundle (may be computed from other parameters)
            meta.append(float(self.period))

            # Simulate Gaia parallax measurement with realistic uncertainties
            distance_true = float(theta_dict["distance"])
            parallax_obs, parallax_err, distance_obs, distance_err, has_gaia = \
                self.simulate_gaia_parallax(sed_flux, distance_true, gaia_coverage=0.7, filter_mask=sed_mask)

            # Add Gaia measurements (NaN if not in Gaia)
            meta.append(distance_obs)      # Observed distance from Gaia (pc)
            meta.append(distance_err)      # Distance uncertainty (pc)
            meta.append(parallax_obs)      # Observed parallax (mas)
            meta.append(parallax_err)      # Parallax uncertainty (mas)

            # Add pre-binning observation counts for each passband
            # This tells the network about data quality/cadence
            for n in n_obs:
                meta.append(float(n))

            meta = np.asarray(meta, dtype=np.float32)

            # -------------------------
            # 4) Masks
            # -------------------------
            mask_lc = [np.ones_like(ch, dtype=np.float32) for ch in lc_flux]
            # mask_sed comes from generate_sed (indicates valid filters)
            mask_meta = np.ones_like(meta, dtype=np.float32)


            # -------------------------
            # 5) Pack into a single object and save
            # -------------------------
            lc_system_ids = dict(self._lc_system_ids)

            x_obj = {
                "lc_phase": lc_phase,
                "lc": lc_flux,
                "lc_err": lc_err,
                "lc_phase_unbinned": lc_phase_unbinned,  # Save unbinned data too
                "lc_unbinned": lc_flux_unbinned,
                "lc_err_unbinned": lc_err_unbinned,
                "sed": sed_flux,
                "sed_err": sed_err,
                "sed_wavelengths": wavelengths,
                "meta": meta,
                "mask_lc": mask_lc,
                "mask_sed": sed_mask,
                "mask_meta": mask_meta,
                "lc_system_ids": lc_system_ids,
                "lc_dataset_labels": list(self._lc_dataset_labels),
            }

            np.save(path, x_obj, allow_pickle=True)
            return True

        except Exception as e:
            print(f"nbi_simulator() failed for path={path}, returning False")
            print(e)
            import traceback
            traceback.print_exc()
            # Save a sentinel so BaseContainer can skip or treat as bad
            x_obj = None
            np.save(path, x_obj, allow_pickle=True)
            return False


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
        n_obs_passbands : list of int
            Number of pre-binning observations for each passband.
            Useful for understanding data quality/cadence.

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
        self.bundle = self.create_phoebe_bundle(theta_dict)

        # Extract period from bundle (may be computed from other parameters)
        self.period = self.bundle.get_value('period@binary@component')

        self.bundle.run_compute()

        phases_passbands = []
        fluxes_passbands = []
        errs_passbands   = []
        n_obs_passbands  = []  # Track pre-binning observation counts

        for dataset_label in self._lc_dataset_labels:
            phases = self.bundle.get_value(f"compute_phases@dataset@{dataset_label}")
            fluxes = self.bundle.get_value(f"fluxes@model@{dataset_label}")

            idx = np.argsort(phases)
            phases = np.asarray(phases[idx], dtype=np.float32)
            fluxes = np.asarray(fluxes[idx], dtype=np.float32)

            # Sample error bars from noise bank
            # (no binning here - that's handled by observational.py)
            survey, system_idx = self._lc_system_ids[dataset_label]
            sigmas = self.cadence_noise_sampler.sample_noise_for_system(
                survey, system_idx, n=len(fluxes), rng=self.rng
            ).astype(np.float32)

            # Add random noise to fluxes (drawn from error distribution)
            # sigmas are fractional errors from noise bank, convert to absolute before adding
            noise = self.rng.normal(0, sigmas * fluxes)
            fluxes = fluxes + noise

            N = len(phases)
            n_obs_passbands.append(N)  # Store pre-binning count for metadata

            # Normalize fluxes to median=1 (helps with eclipsebin later)
            flux_median = np.median(fluxes)
            fluxes_normalized = fluxes / flux_median

            # Normalize sigmas too (fractional errors remain fractional after normalization)
            # So sigmas stay the same, but we compute absolute errors on normalized fluxes
            sigmas_normalized = sigmas * fluxes_normalized

            phases_passbands.append(phases)
            fluxes_passbands.append(fluxes_normalized.astype(np.float32))
            errs_passbands.append(sigmas_normalized.astype(np.float32))

        return phases_passbands, fluxes_passbands, errs_passbands, n_obs_passbands

    def generate_sed(self, theta, sed_frac_err=0.02):
        """
        Generate SED (Spectral Energy Distribution) for given parameters.

        SED is computed at phase 0.25 (quadrature) to ensure both stars are
        fully visible and uneclipsed regardless of orbital inclination.

        Parameters
        ----------
        theta : array
            Parameter vector containing system parameters
        sed_frac_err : float, optional
            Fractional uncertainty for SED fluxes (default: 0.02 = 2%)

        Returns
        -------
        wavelengths : array [23]
            Effective wavelengths for all canonical filters (nm)
            NaN for filters that couldn't be computed for this system
        sed_flux : array [23]
            Fluxes in W/m² at phase 0.25 (quadrature, out of eclipse)
            NaN for invalid filters
        sed_err : array [23]
            Flux uncertainties in W/m²
            NaN for invalid filters
        filter_mask : array [23]
            Boolean mask: 1 = valid filter, 0 = invalid/NaN filter
        """
        theta_dict = {name: val for name, val in zip(self.params_dict.keys(), np.array(theta).T)}

        # Extract parameters needed for PHOEBE
        teff1 = theta_dict["teff1"][0]
        r1    = theta_dict["r1"][0]
        teff2 = theta_dict["teff2"][0]
        r2    = theta_dict["r2"][0]
        msum  = theta_dict["msum"][0]
        q     = theta_dict["q"][0]
        incl   = theta_dict["incl"][0]
        dist = theta_dict["distance"][0]
        ebv  = theta_dict["ebv"][0]

        # Period is computed by PHOEBE from other parameters
        # Use self.period if already set (from generate_light_curve), otherwise extract from bundle
        if hasattr(self, 'period'):
            period = self.period
        else:
            # Create temporary bundle to extract period
            temp_bundle = self.create_phoebe_bundle(theta_dict, passbands=[])
            period = temp_bundle.get_value('period@binary@component')

        # Calculate individual masses
        m1 = msum / (q + 1)
        m2 = q * m1

        # Create PHOEBE wrapper instance
        phoebe_wrapper = PhoebeWrapper(
            teff1=teff1,
            teff2=teff2,
            requiv1=r1,
            requiv2=r2,
            mass1=m1,
            mass2=m2,
            period=period,
            incl=incl,
            distance=dist,
            ebv=ebv
        )

        # Compute SED at phase 0.25 (out of eclipse)
        result = phoebe_wrapper.compute_sed(phases=[0.25])

        # Extract wavelengths and fluxes
        # result['fluxes'] is [n_phases, n_filters], we want [n_filters] for phase 0
        wavelengths = np.asarray(result['wavelengths'], dtype=np.float32)
        sed_flux = np.asarray(result['fluxes'][0], dtype=np.float32)

        # Get filter validity mask (True/1 = valid, False/0 = NaN/invalid)
        filter_mask = np.asarray(result['filter_mask'], dtype=np.float32)

        # "reported" uncertainties for SED points (same length as sed_flux)
        sed_err = (sed_frac_err * np.abs(sed_flux)).astype(np.float32)

        return wavelengths, sed_flux, sed_err, filter_mask

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
            if param not in ['mag', 'times', 'msum', 'metallicity', 'log_age', 'ebv', 'period']:
                twig = TWIG_DICT.get(param)
                if twig:
                    b.set_value(twig, value=float(value))
                    print(f'Set {twig}')

    def _configure_phoebe(self, b, theta, passbands):
        """
        Configure PHOEBE bundle with passbands and compute phases.

        Uses realistic phase sampling from phase_bank/noise_bank.
        """
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

        # Sample phases from each survey's phase bank and add datasets
        for i, passband in enumerate(passbands):
            survey = self._passband_to_survey(passband)

            system_idx = self.cadence_noise_sampler.choose_system(
                survey,
                rng=self.rng,
            )

            # Sample phases from cadence_noise_sampler to get realistic observation times
            sampled_phases = self.cadence_noise_sampler.sample_phases_for_system(
                survey,
                system_idx,
                rng=self.rng,
                sort=True,
            )

            # Add dataset with sampled phases
            dataset_label = f"lc_{survey}_{i}"
            self._lc_dataset_labels.append(dataset_label)
            b.add_dataset(
                'lc',
                compute_phases=sampled_phases,
                passband=f"{passband}",
                dataset=dataset_label,
            )
            self._lc_system_ids[dataset_label] = (survey, system_idx)

        # Adjust atmosphere and limb darkening for edge cases
        for star in ['primary', 'secondary']:
            logg = b.get_value(f'logg@{star}@component')
            teff = b.get_value(f'teff@{star}@component')

            if logg > 5 or teff > 8000:
                b.set_value_all(f'ld_mode@{star}', 'manual')
                b.set_value_all(f'ld_mode_bol@{star}', 'manual')
                b.set_value(f'atm@{star}', 'blackbody')

            if teff < 3500:
                b.set_value_all(f'ld_mode@{star}', 'manual')
                b.set_value_all(f'ld_mode_bol@{star}', 'manual')
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