import warnings
from pathlib import Path

import phoebe
import numpy as np
import scipy.stats
import eclipsebin as ebin
from functools import partial
from astropy import units as u

import matplotlib.pyplot as plt

from .constants import TWIG_DICT, C_LIGHT, Z_SOLAR, R_V, STEFAN_BOLTZ, G_SUN
from .stellar_utils import z_to_feh, interpolate_isochrone
from .observational import CadenceNoiseSampler, survey_noise_process, bin_light_curve
from .phoebe_wrapper import PhoebeWrapper

# =======================
# MODEL CLASS
# =======================

C_LIGHT_SI = 2.99792458e8  # m/s

class EBModel:
    """
    Encapsulates the PHOEBE eclipsing binary model and related utilities.

    Parameters
    ----------
    eb_path : str or Path
        Path to eclipsing binary data
    params_dict : dict
        Dictionary of parameter names and ranges
    phase_bank : dict, optional
        Phase data for cadence sampling
    noise_bank : dict, optional
        Noise data for cadence sampling
    rng : int or RandomState, optional
        Random number generator seed or instance
    save_binning_plots : bool, optional
        If True, save diagnostic plots during light curve binning.
        Default: False
    plot_output_dir : str or Path, optional
        Base directory for saving plots. Plots saved to
        <plot_output_dir>/binning_plots/. Required if save_binning_plots=True.
    amortized : bool, optional
        Whether to use amortized inference. Default: False
    system_weights : str, optional
        Weighting scheme for system selection. Default: "length"
    """

    def __init__(
        self,
        eb_path,
        params_dict,
        phase_bank=None,
        noise_bank=None,
        rng=None,
        save_binning_plots=False,
        plot_output_dir=None,
        amortized=False,
        system_weights="length",
    ):
        self.eb_path = eb_path
        self.params_dict = params_dict
        self.amortized = amortized
        self.rng = np.random.default_rng(rng)

        if phase_bank is not None and noise_bank is not None:
            self.cadence_noise_sampler = CadenceNoiseSampler(
                phase_bank,
                noise_bank,
                log_noise=True,
                seed=rng,
                system_weights=system_weights,
            )
        else:
            self.cadence_noise_sampler = None

        self._lc_system_ids = {}  # maps dataset_label -> (survey, system_idx)
        self._lc_dataset_labels = []
        self._pb_cache = {}

        # Plot saving configuration
        self.save_binning_plots = save_binning_plots
        self.plot_output_dir = Path(plot_output_dir) if plot_output_dir else None

        # Validate plot configuration
        if self.save_binning_plots and self.plot_output_dir is None:
            warnings.warn(
                "save_binning_plots=True but plot_output_dir not provided. "
                "Disabling plot saving."
            )
            self.save_binning_plots = False

        # Initialize sample tracking for plot naming
        self._current_sample_id = 0

    def _pb_pivot_and_dnu(self, passband, lambda_kind="pivot"):
        """
        Return (lambda_nm, dnu_Hz) for a PHOEBE passband, cached.
        We normalize the passband transmission so amplitude doesn't matter.
        """
        key = (str(passband), lambda_kind)
        if key in self._pb_cache:
            return self._pb_cache[key]

        pb = phoebe.get_passband(passband)

        wl_m = np.asarray(pb.ptf_table["wl"], dtype=float)  # meters
        T    = np.asarray(pb.ptf_table["fl"], dtype=float)  # dimensionless

        ok = np.isfinite(wl_m) & np.isfinite(T) & (wl_m > 0) & (T > 0)
        wl_m = wl_m[ok]
        T    = T[ok]

        if wl_m.size < 3:
            self._pb_cache[key] = (np.nan, np.nan)
            return self._pb_cache[key]

        # Normalize transmission so peak=1 (CRITICAL!)
        Tpeak = np.nanmax(T)
        if not np.isfinite(Tpeak) or Tpeak <= 0:
            self._pb_cache[key] = (np.nan, np.nan)
            return self._pb_cache[key]
        Tn = T / Tpeak

        # Representative wavelength (pivot or mean), using normalized T
        if lambda_kind == "pivot":
            num = np.trapz(wl_m * Tn, wl_m)
            den = np.trapz(Tn / wl_m, wl_m)
            lam_m = np.sqrt(num / den)
        elif lambda_kind == "mean":
            lam_m = np.trapz(wl_m * Tn, wl_m) / np.trapz(Tn, wl_m)
        else:
            raise ValueError("lambda_kind must be 'pivot' or 'mean'")

        lam_nm = (lam_m * u.m).to_value(u.nm)

        # Effective rectangular width in FREQUENCY:
        # Δν_eff = ∫ T(ν) dν (with T normalized, amplitude invariant)
        nu = C_LIGHT_SI / wl_m
        order = np.argsort(nu)
        dnu_Hz = float(np.trapz(Tn[order], nu[order]))

        self._pb_cache[key] = (float(lam_nm), float(dnu_Hz))
        return self._pb_cache[key]


    def bandflux_Wm2_to_fnu_Jy(self, bandflux_W_m2, filter_mask=None, lambda_kind="pivot"):
        bandflux_W_m2 = np.asarray(bandflux_W_m2, dtype=float)
        n = len(PhoebeWrapper.CANONICAL_FILTERS)

        lam_nm = np.full(n, np.nan)
        fnu_Jy = np.full(n, np.nan)

        if filter_mask is None:
            mask = np.isfinite(bandflux_W_m2) & (bandflux_W_m2 > 0)
        else:
            mask = np.asarray(filter_mask, dtype=bool) & np.isfinite(bandflux_W_m2) & (bandflux_W_m2 > 0)

        c = C_LIGHT_SI  # m/s

        for i, f in enumerate(PhoebeWrapper.CANONICAL_FILTERS):
            if not mask[i]:
                continue

            pb = phoebe.get_passband(f)
            wl_m = np.asarray(pb.ptf_table["wl"], dtype=float)  # meters
            T    = np.asarray(pb.ptf_table["fl"], dtype=float)  # dimensionless

            # representative wavelength for plotting (pivot or mean)
            Tpeak = np.nanmax(T)
            if not np.isfinite(Tpeak) or Tpeak <= 0:
                continue

            if lambda_kind == "pivot":
                num = np.trapz(wl_m * T, wl_m)
                den = np.trapz(T / wl_m, wl_m)
                lam_m = np.sqrt(num / den)
            elif lambda_kind == "mean":
                lam_m = np.trapz(wl_m * T, wl_m) / np.trapz(T, wl_m)
            else:
                raise ValueError("lambda_kind must be 'pivot' or 'mean'")

            # ---- KEY PART: solve for band-averaged Fnu from the integral ----
            denom = np.trapz(T * (c / (wl_m**2)), wl_m)  # units: 1/s
            if not np.isfinite(denom) or denom <= 0:
                continue

            fnu_W_m2_Hz = bandflux_W_m2[i] / denom  # W m^-2 Hz^-1
            lam_nm[i] = lam_m * 1e9
            fnu_Jy[i] = fnu_W_m2_Hz / 1e-26

        return lam_nm.astype(np.float32), fnu_Jy.astype(np.float32)



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

        # Extract period and incl from bundle (may be computed from other parameters)
        self.period = self.bundle.get_value('period@binary@component')
        self.incl = self.bundle.get_value('incl@binary@component')

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

        # Extract distance and ebv (not in PHOEBE bundle)
        dist = theta_dict["distance"][0]
        ebv  = theta_dict["ebv"][0]

        # Use existing bundle if available (from generate_light_curve), otherwise create temporary bundle
        if hasattr(self, 'bundle'):
            bundle = self.bundle
        else:
            # Create temporary bundle for SED computation
            bundle = self.create_phoebe_bundle(theta_dict, passbands=[])

        # Create PHOEBE wrapper instance with bundle
        phoebe_wrapper = PhoebeWrapper(
            bundle=bundle,
            distance=dist,
            ebv=ebv
        )

        # Compute SED at phase 0.25 (out of eclipse)
        result = phoebe_wrapper.compute_sed(phases=[0.25])

        # Extract wavelengths and fluxes
        # result['fluxes'] is [n_phases, n_filters], we want [n_filters] for phase 0
        wavelengths = np.asarray(result['wavelengths'], dtype=np.float32)
        sed_band_Wm2 = np.asarray(result["fluxes"][0], dtype=np.float32)
        filter_mask  = np.asarray(result["filter_mask"], dtype=np.float32)

        # Convert to f_nu (Jy) + get pivot wavelengths (nm) from passbands
        wavelengths_nm, sed_fnu_Jy = self.bandflux_Wm2_to_fnu_Jy(
            sed_band_Wm2,
            filter_mask=filter_mask,
            lambda_kind="pivot",
        )

        # Update mask based on conversion success
        good = np.isfinite(sed_fnu_Jy) & (sed_fnu_Jy > 0) & (filter_mask > 0)
        sed_mask = good.astype(np.float32)

        # Simple fractional errors in f_nu space
        sed_err = (sed_frac_err * np.abs(sed_fnu_Jy)).astype(np.float32)

        return wavelengths_nm, sed_fnu_Jy, sed_err, sed_mask

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

        # Convert cosi to incl (PHOEBE expects inclination in degrees)
        if 'cosi' in theta:
            cosi = float(theta['cosi'])
            incl_rad = np.arccos(cosi)
            incl_deg = np.degrees(incl_rad)
            incl_twig = TWIG_DICT.get('incl')
            b.set_value(incl_twig, value=incl_deg)
            print(f'Set {incl_twig} from cosi={cosi:.4f} -> incl={incl_deg:.2f} deg')

        for param, value in theta.items():
            if param not in ['mag', 'times', 'msum', 'metallicity', 'log_age', 'ebv', 'period', 'cosi']:
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