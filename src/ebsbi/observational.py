import copy
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import eclipsebin as ebin


def _copy_axes_content(src_ax, dst_ax):
    """
    Copy plot elements from source axes to destination axes.

    Copies lines, collections, labels, limits, and styling to create
    a duplicate of the source axes content.

    Parameters
    ----------
    src_ax : matplotlib.axes.Axes
        Source axes to copy from
    dst_ax : matplotlib.axes.Axes
        Destination axes to copy to

    Notes
    -----
    This function is used internally to combine multiple matplotlib
    figures into a single multi-panel figure for saving.
    """
    # Copy plot elements
    for line in src_ax.get_lines():
        line_copy = copy.copy(line)
        line_copy.figure = None
        line_copy.axes = None
        dst_ax.add_line(line_copy)

    for collection in src_ax.collections:
        coll_copy = copy.copy(collection)
        coll_copy.figure = None
        coll_copy.axes = None
        dst_ax.add_collection(coll_copy)

    # Copy labels and titles
    dst_ax.set_title(src_ax.get_title(), fontsize=src_ax.title.get_fontsize())
    dst_ax.set_xlabel(src_ax.get_xlabel(), fontsize=src_ax.xaxis.label.get_fontsize())
    dst_ax.set_ylabel(src_ax.get_ylabel(), fontsize=src_ax.yaxis.label.get_fontsize())

    # Copy limits and scales
    dst_ax.set_xlim(src_ax.get_xlim())
    dst_ax.set_ylim(src_ax.get_ylim())
    dst_ax.set_xscale(src_ax.get_xscale())
    dst_ax.set_yscale(src_ax.get_yscale())

    # Copy grid
    gridlines = src_ax.xaxis.get_gridlines()
    if gridlines:
        dst_ax.grid(gridlines[0].get_visible())

    # Copy legend if present
    legend = src_ax.get_legend()
    if legend:
        dst_ax.legend(loc=legend._loc)


class Empirical1DSampler:
    """
    Empirical 1D distribution using inverse-CDF sampling.
    Can be used for phases, noise, etc.
    """
    def __init__(self, samples, *, clip=None):
        """
        Parameters
        ----------
        samples : array-like
            1D array of samples (e.g., phases or log10(sigmas)).
        clip : tuple or None
            Optional (min, max) to clip values before building the CDF.
        """
        x = np.asarray(samples).ravel()
        x = x[np.isfinite(x)]

        if clip is not None:
            lo, hi = clip
            x = x[(x >= lo) & (x <= hi)]

        if x.size < 2:
            raise ValueError("Need at least 2 samples to build empirical CDF.")

        # Sort samples
        self.x_sorted = np.sort(x)

        # Empirical CDF values in (0, 1); use midpoints to avoid extremes
        n = self.x_sorted.size
        self.u = (np.arange(n) + 0.5) / n  # 0.5/n, 1.5/n, ..., (n-0.5)/n

        # Build inverse CDF interpolator
        self.inv_cdf = interp1d(
            self.u,
            self.x_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=(self.x_sorted[0], self.x_sorted[-1]),
            assume_sorted=True,
        )

    def sample(self, size=None, rng=None):
        """
        Draw samples using inverse-CDF sampling.

        Parameters
        ----------
        size : int or tuple, optional
            Number/shape of samples to draw.
        rng : np.random.Generator or int, optional
            Random number generator or seed.

        Returns
        -------
        samples : ndarray
        """
        if rng is None:
            rng = np.random.default_rng()
        elif isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(rng)

        u = rng.uniform(0.0, 1.0, size=size)
        return self.inv_cdf(u)


class CadenceNoiseSampler:
    """
    Survey-aware sampler that:
      - Mirrors per-system cadence density and LC lengths.
      - Samples phases via empirical inverse-CDF (no exact phase replicates).
      - Samples per-point noise from empirical noise distributions.
      - Uses the *same system* for phases and noise to preserve correlations.

    Parameters
    ----------
    phase_bank : dict
        {survey_name: list of 1D phase arrays}
        For each survey, phase_bank[survey][i] is phases for system i.
        Phases can be in any range; will be wrapped into [0, 1).
    noise_bank : dict
        {survey_name: list of 1D noise arrays (e.g., sigma_flux)}.
        For each survey, noise_bank[survey][i] must correspond to the
        same system as phase_bank[survey][i].
    log_noise : bool, default True
        If True, build noise distribution in log10 space.
    seed : int or None
        Optional global seed for internal RNG.
    system_weights : {"uniform", "length"}, default "uniform"
        How to choose which system to prototype from:
          - "uniform": each system equally likely.
          - "length": probability ∝ number of points in that system.
    """

    def __init__(
        self,
        phase_bank,
        noise_bank,
        log_noise=True,
        seed=None,
        system_weights="uniform",
    ):
        self.phase_bank = phase_bank
        self.noise_bank = noise_bank
        self.log_noise = log_noise
        self.system_weights = system_weights

        self.rng = np.random.default_rng(seed)

        # Per-survey, per-system samplers
        self.system_phase_samplers = {}   # survey -> list[Empirical1DSampler]
        self.system_noise_samplers = {}   # survey -> list[callable(size, rng)]
        self.system_lengths = {}          # survey -> array of N_i
        self.system_probs = {}            # survey -> array of p_i

        self._build_paired_samplers()

    # -----------------------------
    # Internal builders
    # -----------------------------
    def _build_paired_samplers(self):
        surveys = set(self.phase_bank.keys()).union(self.noise_bank.keys())

        for survey in surveys:
            phase_list = self.phase_bank.get(survey, None)
            noise_list = self.noise_bank.get(survey, None)

            if phase_list is None or noise_list is None:
                raise ValueError(
                    f"Survey '{survey}' missing in either phase_bank or noise_bank."
                )

            if len(phase_list) != len(noise_list):
                raise ValueError(
                    f"Survey '{survey}': phase_bank and noise_bank lengths differ "
                    f"({len(phase_list)} vs {len(noise_list)}). "
                    "They must correspond system-by-system."
                )

            phase_samplers = []
            noise_samplers = []
            lengths = []

            for ph_arr, no_arr in zip(phase_list, noise_list):
                ph_arr = np.asarray(ph_arr, dtype=float).ravel()
                ph_arr = ph_arr[np.isfinite(ph_arr)]
                if ph_arr.size == 0:
                    continue

                # Wrap phases into [0, 1)
                ph_arr = ph_arr - np.floor(ph_arr)

                no_arr = np.asarray(no_arr, dtype=float).ravel()
                no_arr = no_arr[np.isfinite(no_arr) & (no_arr > 0)]
                if no_arr.size == 0:
                    continue

                # Build per-system phase sampler
                ph_sampler = Empirical1DSampler(ph_arr, clip=(0.0, 1.0))
                phase_samplers.append(ph_sampler)
                lengths.append(ph_arr.size)

                # Build per-system noise sampler (possibly in log space)
                if self.log_noise:
                    log_vals = np.log10(no_arr)
                    no_sampler = Empirical1DSampler(log_vals)

                    def _noise_sample(size=None, rng=None, _sampler=no_sampler):
                        log_s = _sampler.sample(size=size, rng=rng)
                        return 10.0 ** log_s
                else:
                    lin_sampler = Empirical1DSampler(no_arr)

                    def _noise_sample(size=None, rng=None, _sampler=lin_sampler):
                        return _sampler.sample(size=size, rng=rng)

                noise_samplers.append(_noise_sample)

            if not phase_samplers:
                raise ValueError(f"No valid systems for survey '{survey}'.")

            lengths = np.asarray(lengths, dtype=int)

            # How to weight systems when selecting them
            if self.system_weights == "length":
                w = lengths.astype(float)
                probs = w / w.sum()
            elif self.system_weights == "uniform":
                probs = np.ones_like(lengths, dtype=float) / lengths.size
            else:
                raise ValueError(
                    "system_weights must be 'uniform' or 'length', "
                    f"got '{self.system_weights}'."
                )

            self.system_phase_samplers[survey] = phase_samplers
            self.system_noise_samplers[survey] = noise_samplers
            self.system_lengths[survey] = lengths
            self.system_probs[survey] = probs

    # -----------------------------
    # Helpers
    # -----------------------------
    def _get_rng(self, rng):
        if rng is None:
            return self.rng
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(rng)
        return rng

    def _choose_system_index(self, survey, rng=None):
        if survey not in self.system_probs:
            raise KeyError(f"No system probabilities for survey '{survey}'.")
        rng = self._get_rng(rng)
        probs = self.system_probs[survey]
        return int(rng.choice(np.arange(probs.size), p=probs))

    # -----------------------------
    # Public API
    # -----------------------------
    def sample_cadence_and_noise(self, survey, rng=None, sort=True):
        """
        Sample phases AND noise for one synthetic light curve for a given survey,
        using the same real system as a prototype.

        Parameters
        ----------
        survey : str
            Survey name (key in phase_bank / noise_bank).
        rng : np.random.Generator or int or None
        sort : bool
            Sort phases if True.

        Returns
        -------
        phases : ndarray
            Phases in [0, 1), length N_i where i is the chosen system.
        sigmas : ndarray
            Per-point noise values, same length as phases.
        """
        rng = self._get_rng(rng)

        if survey not in self.system_phase_samplers:
            raise KeyError(f"No samplers found for survey '{survey}'.")

        # 1. Choose a system index for this survey
        k = self._choose_system_index(survey, rng=rng)

        # 2. Get its length
        n_points = int(self.system_lengths[survey][k])

        # 3. Sample phases from that system's phase sampler
        phases = self.system_phase_samplers[survey][k].sample(
            size=n_points, rng=rng
        )
        phases = phases - np.floor(phases)
        if sort:
            phases = np.sort(phases)

        # 4. Sample noise from that system's noise sampler
        sigmas = self.system_noise_samplers[survey][k](size=n_points, rng=rng)

        return phases, sigmas

    def sample_phases(self, survey, rng=None, sort=True):
        """
        Convenience: just phases (paired-by-system).
        """
        phases, _ = self.sample_cadence_and_noise(survey, rng=rng, sort=sort)
        return phases

    def sample_noise(self, survey, rng=None):
        """
        Convenience: just noise (paired-by-system).
        """
        _, sigmas = self.sample_cadence_and_noise(survey, rng=rng, sort=True)
        return sigmas

    def choose_system(self, survey, rng=None):
        """
        Public wrapper to choose a system index for a given survey.
        """
        return self._choose_system_index(survey, rng=rng)

    def sample_phases_for_system(self, survey, system_idx, rng=None, sort=True):
        """
        Sample phases using a specific prototype system.
        """
        rng = self._get_rng(rng)

        phase_sampler = self.system_phase_samplers[survey][system_idx]
        n_points = int(self.system_lengths[survey][system_idx])

        phases = phase_sampler.sample(size=n_points, rng=rng)
        phases = phases - np.floor(phases)  # wrap to [0, 1)
        if sort:
            phases = np.sort(phases)
        return phases

    def sample_noise_for_system(self, survey, system_idx, rng=None, n=None):
        """
        Sample noise using a specific prototype system.
        If n is provided, draw n samples from that system's empirical noise distribution.
        If n is None, use the prototype system's native length.
        """
        rng = self._get_rng(rng)

        if n is None:
            n = int(self.system_lengths[survey][system_idx])

        noise_sampler = self.system_noise_samplers[survey][system_idx]
        sigmas = noise_sampler(size=int(n), rng=rng)
        return sigmas


def _wrap_width(lo, hi):
    """Width on a unit circle for bounds in [0,1)."""
    if lo is None or hi is None:
        return np.nan
    lo = float(lo); hi = float(hi)
    w = hi - lo
    if w < 0:
        w += 1.0
    return w

def _eclipse_width_from_bounds(bounds):
    """
    eclipsebin's get_eclipse_boundaries returns something like (left, right)
    (sometimes nested/array-like). This tries to robustly extract two numbers.
    """
    if bounds is None:
        return np.nan
    # common cases: tuple/list/np array
    try:
        if len(bounds) >= 2 and np.isscalar(bounds[0]) and np.isscalar(bounds[1]):
            return _wrap_width(bounds[0], bounds[1])
        # sometimes it might return dict-like or nested; handle simple nesting
        if len(bounds) == 1 and hasattr(bounds[0], "__len__") and len(bounds[0]) >= 2:
            return _wrap_width(bounds[0][0], bounds[0][1])
    except Exception:
        pass
    return np.nan


def detect_eclipse_edges_slope(
    phases,
    fluxes,
    sigmas=None,
    smoothing_window=None,
    slope_threshold_percentile=93.0,
    return_threshold_fraction=0.1,
    min_eclipse_depth=0.02,
    plot=False
):
    """
    Detect eclipse boundaries using slope/derivative-based edge detection.
    
    This method is robust to ellipsoidal variations because it detects edges
    based on local slope changes rather than absolute flux levels.
    
    Parameters
    ----------
    phases : array
        Phase values (should be sorted and in [0, 1))
    fluxes : array
        Flux values (normalized to median~1 recommended)
    sigmas : array, optional
        Per-point uncertainties (used for weighting if provided)
    smoothing_window : int, optional
        Window size for Savitzky-Golay smoothing (must be odd).
        If None, auto-selects based on data density (~5% of points, minimum 5)
    slope_threshold_percentile : float, default=90.0
        Percentile of absolute slopes to use as threshold for detecting
        steep ingress/egress. Higher = more selective.
    return_threshold_fraction : float, default=0.1
        Fraction of maximum slope to use for defining boundary.
        Lower = boundaries closer to eclipse center.
    min_eclipse_depth : float, default=0.01
        Minimum flux drop to consider as eclipse (relative to local baseline)
    plot : bool, default=False
        Whether to create diagnostic plots
    
    Returns
    -------
    eclipse_boundaries : list of tuples
        List of (ingress_phase, egress_phase) for each detected eclipse.
        Empty list if no eclipses detected.
    diagnostics : dict
        Dictionary with diagnostic information:
        - 'slopes': derivative values
        - 'smoothed_fluxes': smoothed flux values
        - 'threshold': slope threshold used
        - 'ingress_candidates': phase indices of ingress candidates
        - 'egress_candidates': phase indices of egress candidates
    """
    phases = np.asarray(phases, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    
    if len(phases) < 10:
        return [], {'error': 'Insufficient data points'}
    
    # Ensure sorted
    idx = np.argsort(phases)
    phases = phases[idx]
    fluxes = fluxes[idx]
    
    # Auto-select smoothing window if not provided
    if smoothing_window is None:
        n_points = len(phases)
        window = max(5, int(0.05 * n_points))
        # Must be odd for Savitzky-Golay
        if window % 2 == 0:
            window += 1
        smoothing_window = min(window, n_points - 2)
        if smoothing_window < 5:
            smoothing_window = 5
    
    # Smooth the light curve to reduce noise
    try:
        smoothed_fluxes = savgol_filter(fluxes, smoothing_window, 3)
    except Exception:
        # Fallback to simple moving average if Savitzky-Golay fails
        from scipy.ndimage import uniform_filter1d
        smoothed_fluxes = uniform_filter1d(fluxes, size=smoothing_window)
    
    # Compute derivative (slope) using finite differences
    # Handle phase wrapping at boundaries
    dphase = np.diff(phases)
    dflux = np.diff(smoothed_fluxes)
    
    # Handle phase wrapping: if gap is large, don't compute slope across it
    median_dphase = np.median(dphase[dphase > 0])
    large_gap = dphase > 10 * median_dphase
    
    slopes = np.zeros_like(phases)
    slopes[1:] = dflux / (dphase + 1e-10)  # Avoid division by zero
    # Set slopes to zero where there are large gaps (indexing matches slopes[1:])
    slopes[1:][large_gap] = 0.0
    
    # Compute absolute slopes for thresholding
    abs_slopes = np.abs(slopes)
    
    # Determine threshold based on percentile
    valid_slopes = abs_slopes[abs_slopes > 0]
    if len(valid_slopes) == 0:
        return [], {'error': 'No valid slopes computed'}
    
    slope_threshold = np.percentile(valid_slopes, slope_threshold_percentile)
    return_threshold = slope_threshold * return_threshold_fraction
    
    # Find regions with steep negative slopes (ingress) and positive slopes (egress)
    # Ingress: steep negative slope
    ingress_mask = slopes < -slope_threshold
    # Egress: steep positive slope
    egress_mask = slopes > slope_threshold
    
    # Find eclipse candidates by looking for ingress-egress pairs
    eclipse_boundaries = []
    
    # Simple algorithm: find ingress points, then look for corresponding egress
    ingress_indices = np.where(ingress_mask)[0]
    egress_indices = np.where(egress_mask)[0]
    
    if len(ingress_indices) == 0 or len(egress_indices) == 0:
        return [], {
            'slopes': slopes,
            'smoothed_fluxes': smoothed_fluxes,
            'threshold': slope_threshold,
            'ingress_candidates': ingress_indices,
            'egress_candidates': egress_indices
        }
    
    # Pair up ingress and egress points
    i = 0
    while i < len(ingress_indices):
        ingress_idx = ingress_indices[i]
        
        # Find next egress after this ingress
        egress_candidates = egress_indices[egress_indices > ingress_idx]
        
        if len(egress_candidates) > 0:
            egress_idx = egress_candidates[0]
            
            # Check if this is a real eclipse (flux drops significantly)
            eclipse_region = smoothed_fluxes[ingress_idx:egress_idx+1]
            if len(eclipse_region) > 0:
                local_baseline = np.median([
                    np.median(smoothed_fluxes[max(0, ingress_idx-10):ingress_idx]),
                    np.median(smoothed_fluxes[egress_idx:min(len(fluxes), egress_idx+10)])
                ])
                min_flux = np.min(eclipse_region)
                depth = (local_baseline - min_flux) / local_baseline
                
                if depth >= min_eclipse_depth:
                    # Refine boundaries: find where slope crosses return_threshold
                    # Ingress: find where slope becomes less negative
                    ingress_refined = ingress_idx
                    for j in range(ingress_idx, max(0, ingress_idx-20), -1):
                        if abs_slopes[j] < return_threshold:
                            ingress_refined = j
                            break
                    
                    # Egress: find where slope becomes less positive
                    egress_refined = egress_idx
                    for j in range(egress_idx, min(len(phases), egress_idx+20)):
                        if abs_slopes[j] < return_threshold:
                            egress_refined = j
                            break
                    
                    eclipse_boundaries.append((
                        phases[ingress_refined],
                        phases[egress_refined]
                    ))
                    
                    # Skip to after this egress
                    i = np.searchsorted(ingress_indices, egress_idx, side='right')
                    continue
        
        i += 1
    
    diagnostics = {
        'slopes': slopes,
        'smoothed_fluxes': smoothed_fluxes,
        'threshold': slope_threshold,
        'return_threshold': return_threshold,
        'ingress_candidates': ingress_indices,
        'egress_candidates': egress_indices
    }
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            # Top: original and smoothed light curve
            axes[0].plot(phases, fluxes, 'k.', alpha=0.3, label='Original', markersize=2)
            axes[0].plot(phases, smoothed_fluxes, 'b-', label='Smoothed', linewidth=1.5)
            for ingress, egress in eclipse_boundaries:
                axes[0].axvspan(ingress, egress, alpha=0.2, color='red', label='Eclipse' if eclipse_boundaries.index((ingress, egress)) == 0 else '')
            axes[0].set_ylabel('Flux')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Middle: slopes
            axes[1].plot(phases, slopes, 'g-', linewidth=1)
            axes[1].axhline(slope_threshold, 'r--', label=f'Threshold ({slope_threshold:.4f})')
            axes[1].axhline(-slope_threshold, 'r--')
            axes[1].axhline(return_threshold, 'b--', label=f'Return threshold ({return_threshold:.4f})')
            axes[1].axhline(-return_threshold, 'b--')
            axes[1].set_ylabel('Slope (dFlux/dPhase)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Bottom: absolute slopes with candidates
            axes[2].plot(phases, abs_slopes, 'purple', linewidth=1)
            if len(ingress_indices) > 0:
                axes[2].plot(phases[ingress_indices], abs_slopes[ingress_indices], 'ro', 
                           markersize=8, label='Ingress candidates')
            if len(egress_indices) > 0:
                axes[2].plot(phases[egress_indices], abs_slopes[egress_indices], 'go', 
                           markersize=8, label='Egress candidates')
            axes[2].axhline(slope_threshold, 'r--', label='Threshold')
            axes[2].set_ylabel('|Slope|')
            axes[2].set_xlabel('Phase')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    return eclipse_boundaries, diagnostics


def detect_eclipse_edges_curvature(
    phases,
    fluxes,
    sigmas=None,
    smoothing_window=None,
    curvature_threshold_percentile=93.0,
    min_eclipse_depth=0.02,
    plot=False
):
    """
    Detect eclipse boundaries using curvature (second derivative) detection.
    
    This method identifies eclipse edges by finding points of maximum curvature,
    which helps distinguish sharp eclipse edges from gradual baseline variations.
    
    Parameters
    ----------
    phases : array
        Phase values (should be sorted and in [0, 1))
    fluxes : array
        Flux values (normalized to median~1 recommended)
    sigmas : array, optional
        Per-point uncertainties (not currently used, kept for API compatibility)
    smoothing_window : int, optional
        Window size for Savitzky-Golay smoothing (must be odd).
        If None, auto-selects based on data density
    curvature_threshold_percentile : float, default=90.0
        Percentile of absolute curvature to use as threshold
    min_eclipse_depth : float, default=0.01
        Minimum flux drop to consider as eclipse
    plot : bool, default=False
        Whether to create diagnostic plots
    
    Returns
    -------
    eclipse_boundaries : list of tuples
        List of (ingress_phase, egress_phase) for each detected eclipse
    diagnostics : dict
        Dictionary with diagnostic information
    """
    phases = np.asarray(phases, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)
    
    if len(phases) < 10:
        return [], {'error': 'Insufficient data points'}
    
    # Ensure sorted
    idx = np.argsort(phases)
    phases = phases[idx]
    fluxes = fluxes[idx]
    
    # Auto-select smoothing window
    if smoothing_window is None:
        n_points = len(phases)
        window = max(5, int(0.05 * n_points))
        if window % 2 == 0:
            window += 1
        smoothing_window = min(window, n_points - 2)
        if smoothing_window < 5:
            smoothing_window = 5
    
    # Smooth the light curve
    try:
        smoothed_fluxes = savgol_filter(fluxes, smoothing_window, 3)
    except Exception:
        from scipy.ndimage import uniform_filter1d
        smoothed_fluxes = uniform_filter1d(fluxes, size=smoothing_window)
    
    # Compute first derivative
    dphase = np.diff(phases)
    dflux = np.diff(smoothed_fluxes)
    median_dphase = np.median(dphase[dphase > 0])
    large_gap = dphase > 10 * median_dphase
    
    slopes = np.zeros_like(phases)
    slopes[1:] = dflux / (dphase + 1e-10)
    slopes[large_gap] = 0.0
    
    # Compute second derivative (curvature)
    dslope = np.diff(slopes)
    # dphase has length n-1, matching dslope length
    # Use dphase directly for second derivative calculation
    
    curvature = np.zeros_like(phases)
    curvature[1:] = dslope / (dphase + 1e-10)
    
    abs_curvature = np.abs(curvature)
    
    # Find threshold
    valid_curvature = abs_curvature[abs_curvature > 0]
    if len(valid_curvature) == 0:
        return [], {'error': 'No valid curvature computed'}
    
    curvature_threshold = np.percentile(valid_curvature, curvature_threshold_percentile)
    
    # Find high curvature points (potential edges)
    high_curvature = abs_curvature > curvature_threshold
    
    # Look for ingress (negative curvature spike) and egress (positive curvature spike)
    ingress_candidates = np.where((curvature < -curvature_threshold))[0]
    egress_candidates = np.where((curvature > curvature_threshold))[0]
    
    # Similar pairing logic as slope method
    eclipse_boundaries = []
    
    if len(ingress_candidates) > 0 and len(egress_candidates) > 0:
        i = 0
        while i < len(ingress_candidates):
            ingress_idx = ingress_candidates[i]
            egress_candidates_after = egress_candidates[egress_candidates > ingress_idx]
            
            if len(egress_candidates_after) > 0:
                egress_idx = egress_candidates_after[0]
                
                # Check eclipse depth
                eclipse_region = smoothed_fluxes[ingress_idx:egress_idx+1]
                if len(eclipse_region) > 0:
                    local_baseline = np.median([
                        np.median(smoothed_fluxes[max(0, ingress_idx-10):ingress_idx]),
                        np.median(smoothed_fluxes[egress_idx:min(len(fluxes), egress_idx+10)])
                    ])
                    min_flux = np.min(eclipse_region)
                    depth = (local_baseline - min_flux) / local_baseline
                    
                    if depth >= min_eclipse_depth:
                        eclipse_boundaries.append((
                            phases[ingress_idx],
                            phases[egress_idx]
                        ))
                        i = np.searchsorted(ingress_candidates, egress_idx, side='right')
                        continue
            
            i += 1
    
    diagnostics = {
        'curvature': curvature,
        'smoothed_fluxes': smoothed_fluxes,
        'threshold': curvature_threshold,
        'ingress_candidates': ingress_candidates,
        'egress_candidates': egress_candidates
    }
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            axes[0].plot(phases, fluxes, 'k.', alpha=0.3, markersize=2)
            axes[0].plot(phases, smoothed_fluxes, 'b-', linewidth=1.5)
            for ingress, egress in eclipse_boundaries:
                axes[0].axvspan(ingress, egress, alpha=0.2, color='red')
            axes[0].set_ylabel('Flux')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(phases, curvature, 'g-', linewidth=1)
            axes[1].axhline(curvature_threshold, 'r--')
            axes[1].axhline(-curvature_threshold, 'r--')
            axes[1].set_ylabel('Curvature')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(phases, abs_curvature, 'purple', linewidth=1)
            if len(ingress_candidates) > 0:
                axes[2].plot(phases[ingress_candidates], abs_curvature[ingress_candidates], 'ro', markersize=8)
            if len(egress_candidates) > 0:
                axes[2].plot(phases[egress_candidates], abs_curvature[egress_candidates], 'go', markersize=8)
            axes[2].axhline(curvature_threshold, 'r--')
            axes[2].set_ylabel('|Curvature|')
            axes[2].set_xlabel('Phase')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")
    
    return eclipse_boundaries, diagnostics


def bin_light_curve(
    phases,
    fluxes,
    sigmas,
    nbins=200,
    fraction_in_eclipse=0.5,
    atol_primary=0.001,
    atol_secondary=0.05,
    plot=False,
    use_edge_detection=False,
    edge_detection_method='slope',
    edge_detection_kwargs=None
):
    """
    Bin a light curve using eclipsebin.

    This is the single source of truth for light curve binning in the codebase.
    Used by both:
    - nbi_simulator() when generating pre-binned training samples
    - survey_noise_process() when generating data on-the-fly

    Parameters
    ----------
    phases : array
        Phase values (will be wrapped to [0, 1))
    fluxes : array
        Flux values (should be normalized to median~1 for best eclipsebin performance)
    sigmas : array
        Per-point uncertainties (absolute units matching fluxes)
    nbins : int
        Target number of bins (note: eclipsebin returns nbins+1, so we request nbins-1)
    fraction_in_eclipse : float
        Fraction of points expected to be in eclipse
    atol_primary : float
        Tolerance for primary eclipse detection
    atol_secondary : float
        Tolerance for secondary eclipse detection
    plot : bool
        Whether to plot the binning (for debugging)
    use_edge_detection : bool, default=False
        If True, use slope/curvature-based edge detection to improve eclipse
        boundary detection. This is more robust to ellipsoidal variations.
    edge_detection_method : str, default='slope'
        Method to use: 'slope' or 'curvature'
    edge_detection_kwargs : dict, optional
        Additional keyword arguments to pass to edge detection function

    Returns
    -------
    bin_phases : array
        Binned phase centers
    bin_fluxes : array
        Binned flux values
    bin_sigmas : array
        Binned uncertainties (propagated)
    """
    phases = np.asarray(phases, dtype=np.float32)
    fluxes = np.asarray(fluxes, dtype=np.float32)
    sigmas = np.asarray(sigmas, dtype=np.float32)

    # Wrap phases to [0, 1)
    phases = phases - np.floor(phases)

    # Sort by phase
    idx = np.argsort(phases)
    phases = phases[idx]
    fluxes = fluxes[idx]
    sigmas = sigmas[idx]

    # Optional: Use edge detection to improve eclipse boundary detection
    if use_edge_detection:
        if edge_detection_kwargs is None:
            edge_detection_kwargs = {}
        
        if edge_detection_method == 'slope':
            eclipse_boundaries, edge_diagnostics = detect_eclipse_edges_slope(
                phases, fluxes, sigmas, plot=plot, **edge_detection_kwargs
            )
        elif edge_detection_method == 'curvature':
            eclipse_boundaries, edge_diagnostics = detect_eclipse_edges_curvature(
                phases, fluxes, sigmas, plot=plot, **edge_detection_kwargs
            )
        else:
            raise ValueError(f"Unknown edge_detection_method: {edge_detection_method}")
        
        # Note: Currently, eclipsebin doesn't accept pre-detected boundaries,
        # so we just use the standard binning. The edge detection can be used
        # for validation or as a fallback. Future enhancement could modify
        # eclipsebin parameters based on detected boundaries.
        if plot and len(eclipse_boundaries) > 0:
            print(f"Edge detection found {len(eclipse_boundaries)} eclipse(s):")
            for i, (ingress, egress) in enumerate(eclipse_boundaries):
                print(f"  Eclipse {i+1}: ingress={ingress:.4f}, egress={egress:.4f}, width={_wrap_width(ingress, egress):.4f}")

    # Request nbins-1 because eclipsebin returns nbins+1 centers
    nbins_request = max(1, nbins - 1)

    # For very large datasets, use a more conservative fraction_in_eclipse to prevent
    # eclipsebin's retry logic from hitting the floating-point bug that allows
    # fraction_in_eclipse to go to ~0 (which causes pd.qcut(q=0) to return NaN)
    if len(phases) > 1500:
        # Use smaller fraction to leave more room for retries without hitting 0
        fraction_in_eclipse_adj = min(fraction_in_eclipse, 0.2)
        if plot:
            print(f"Large dataset ({len(phases)} points): using fraction_in_eclipse={fraction_in_eclipse_adj}")
    else:
        fraction_in_eclipse_adj = fraction_in_eclipse

    # Bin with eclipsebin (handles both normal and sparse data internally)
    # Retry with progressively smaller nbins if binning fails
    max_retries = 4
    nbins_to_try = nbins_request
    last_error = None

    for attempt in range(max_retries):
        try:
            binner = ebin.EclipsingBinaryBinner(
                phases,
                fluxes,
                sigmas,
                nbins=nbins_to_try,
                fraction_in_eclipse=fraction_in_eclipse_adj,
                atol_primary=atol_primary,
                atol_secondary=atol_secondary,
            )
            bin_phases, bin_fluxes, bin_sigmas = binner.bin_light_curve(plot=plot)

            # Success! Break out of retry loop
            if attempt > 0 and plot:
                print(f"✓ Binning succeeded with {len(bin_phases)} bins (requested {nbins_to_try + 1})")
            break

        except ValueError as e:
            last_error = e
            # Reduce nbins for next attempt: try 70%, 50%, 35%, 25% of original
            reduction_factors = [0.70, 0.50, 0.35, 0.25]
            nbins_to_try = max(10, int(nbins_request * reduction_factors[attempt]))

            if attempt < max_retries - 1:
                if plot:
                    print(f"Binning failed, retrying with nbins={nbins_to_try + 1} "
                          f"({int(reduction_factors[attempt] * 100)}% of original)")
            else:
                # All retries exhausted
                raise ValueError(
                    f"Could not bin light curve after {max_retries} attempts. "
                    f"Last error: {last_error}"
                )

    # Ensure correct output types
    bin_phases = np.asarray(bin_phases, dtype=np.float32).ravel()
    bin_fluxes = np.asarray(bin_fluxes, dtype=np.float32).ravel()
    bin_sigmas = np.asarray(bin_sigmas, dtype=np.float32).ravel()

    return bin_phases, bin_fluxes, bin_sigmas


def survey_noise_process(
    x,
    y,
    cadence_noise_sampler,
    rng=None,
    nbins=200,
    fraction_in_eclipse=0.5,
    atol_primary=0.001,
    atol_secondary=0.05,
    ):
    """
    NBI 'process' function: takes dict x from simulator and returns (channels, y).

    channels = [lc_ch0, lc_ch1, ..., sed, meta]
    - each lc_ch: (2, nbins) float32  -> [flux, sigma]
    - sed:        (2, n_sed) if sed_err exists else (1, n_sed)
    - meta:       (1, n_meta_aug)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Handle case where x is a 0-d numpy array containing a dict
    # (happens when NBI loads from .npy files without .item())
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    lc_flux_list  = x["lc"]              # list of flux arrays
    lc_phase_list = x["lc_phase"]        # list of phase arrays
    lc_err_list   = x.get("lc_err", None)  # list of sigma arrays (preferred)
    lc_system_ids = x.get("lc_system_ids", {})
    dataset_labels = x.get("lc_dataset_labels", None)

    # Preserve the simulator's ordering (critical!)
    if dataset_labels is None:
        dataset_labels = list(lc_system_ids.keys())
        if not dataset_labels:
            # fallback: assume lists are already aligned
            dataset_labels = [f"lc_{i}" for i in range(len(lc_flux_list))]

    lc_channels = []

    for ch_idx, dataset_label in enumerate(dataset_labels):
        flux   = np.asarray(lc_flux_list[ch_idx], dtype=np.float32).ravel()
        phases = np.asarray(lc_phase_list[ch_idx], dtype=np.float32).ravel()

        # Prefer saved per-point uncertainties (these should correspond to flux/phases)
        if lc_err_list is not None:
            sig = np.asarray(lc_err_list[ch_idx], dtype=np.float32).ravel()
        else:
            # fallback: sample from the prototype system (keeps your earlier logic alive)
            if dataset_label in lc_system_ids:
                survey, system_idx = lc_system_ids[dataset_label]
                sig = cadence_noise_sampler.sample_noise_for_system(
                    survey, system_idx, n=len(flux), rng=rng
                ).astype(np.float32)
            else:
                # last-resort fallback
                sig = (0.01 * np.ones_like(flux)).astype(np.float32)

        # length safety
        L = min(len(flux), len(phases), len(sig))
        flux, phases, sig = flux[:L], phases[:L], sig[:L]

        # Add scatter consistent with the *reported* sigmas
        flux_noisy = flux + rng.normal(0.0, sig, size=L).astype(np.float32)

        # ---- binning: only if needed (avoid double-binning) ----
        # Bin if we have more points than target
        # Skip if already at or below target (pre-binned or sparse data)
        if L > nbins:
            # Bin using centralized function
            _, bin_means, bin_sig = bin_light_curve(
                phases,
                flux_noisy,
                sig,
                nbins=nbins,
                fraction_in_eclipse=fraction_in_eclipse,
                atol_primary=atol_primary,
                atol_secondary=atol_secondary,
                plot=False
            )
        else:
            # Already at or below target: use as-is
            bin_means = flux_noisy.astype(np.float32)
            bin_sig   = sig.astype(np.float32)

        # sanitize NaNs
        bin_means_safe = np.where(np.isfinite(bin_means), bin_means, 0.0).astype(np.float32)
        bin_sig_safe   = np.where(np.isfinite(bin_sig) & (bin_sig > 0), bin_sig, 0.0).astype(np.float32)

        # LC channel: (2, nbins)
        lc_ch = np.stack([bin_means_safe, bin_sig_safe], axis=0).astype(np.float32)
        lc_channels.append(lc_ch)

    # Metadata (no augmentation needed)
    meta = np.asarray(x["meta"], dtype=np.float32).ravel()[None, :]

    # SED: include uncertainties if present
    sed = np.asarray(x["sed"], dtype=np.float32).ravel()
    sed_err = x.get("sed_err", None)
    if sed_err is not None:
        sed_err = np.asarray(sed_err, dtype=np.float32).ravel()
        Ls = min(len(sed), len(sed_err))
        sed, sed_err = sed[:Ls], sed_err[:Ls]
        sed_noisy = sed + rng.normal(0.0, sed_err, size=Ls).astype(np.float32)
        sed_ch = np.stack([sed_noisy.astype(np.float32), sed_err.astype(np.float32)], axis=0)
    else:
        sed_ch = sed[None, :].astype(np.float32)

    channels = lc_channels + [sed_ch, meta]
    return channels, y