import pymc as pm
import numpy as np
import xarray as xr
import hashlib
import pytensor.tensor as pt
import pytensor
from scipy.stats import truncnorm
import pandas as pd


from .stellar_utils import interpolate_isochrone, z_to_feh, roche_lobe_scaled_radius
from .constants import G_SUN, Z_SOLAR

class EBPriors:
    """
    Defines prior distributions and sampling methods for eclipsing binary parameters,
    supporting truncated normal, uniform, triangular, and normal distributions.
    """

    def __init__(self, params_dict, labels_dict, tracks, rng=None):
        """
        Args:
            params_dict (dict): Dictionary of parameter settings from the config file.
        """
        self.params_dict = params_dict
        self.labels_dict = labels_dict
        self.distributions = None
        self.tracks = tracks
        self.rng = rng
        self.logp_fn = None
        self.pymc_rvs = None
        self.pymc_names = None
        self.keys_derived = ['teff1', 'r1', 'log_lum1', 'teff2', 'r2', 'log_lum2', 'log_age']
        self.mist_df_reindexed = self._reindex_mist()

        self._create_distributions()

    def _reindex_mist(self):
        df = self.tracks.model_grid.df.copy()
        df.index.set_names(["feh", "mass", "eeps"], inplace=True)
        df = df.rename_axis(["feh_index", "mass_index", "eeps_index"])
        mass_col = 'mass'
        df_reset = df.reset_index()
        df_reset["current_mass"] = df_reset[mass_col].astype(float)
        df_reset["feh"] = df_reset["feh"].astype(float)
        df_reset["eep"] = df_reset["eep"].astype(float)
        df_cur = df_reset.set_index(["feh", "current_mass", "eep"]).sort_index()
        if not df_cur.index.is_unique:
            df_cur = (
                df_cur[~df_cur.index.duplicated(keep="first")]
                .sort_index()
            )
        return df_cur

    def _get_distribution(self, name, settings, dependent_lower=None, dependent_upper=None, dependent_mu=None):
        dist_info = settings.get('DISTRIBUTION', [])
        dist_type = dist_info[0].lower() if dist_info else 'normal'
        dist_params = dist_info[1:]

        lower_bound = dependent_lower if dependent_lower is not None else settings.get('LOWER', None)
        upper_bound = dependent_upper if dependent_upper is not None else settings.get('UPPER', None)

        if dist_type == 'normal':
            if len(dist_params) < 2:
                raise ValueError(f"Missing mean or std for normal distribution of '{name}'")
            mean, std = dist_params
            if dependent_mu is not None:
                mean = dependent_mu
            return pm.Normal(name, mu=mean, sigma=std)

        elif dist_type == 'lognormal':
            if len(dist_params) < 2:
                raise ValueError(f"Missing mean or std for lognormal distribution of '{name}'")
            mean, std = dist_params
            if dependent_mu is not None:
                mean = dependent_mu
            return pm.LogNormal(name, mu=mean, sigma=std)

        elif dist_type == 'uniform':
            return pm.Uniform(name, lower=lower_bound, upper=upper_bound)

        elif dist_type == 'triangular':
            if not dist_params:
                raise ValueError(f"Missing mode for triangular distribution of '{name}'")
            mode = dist_params[0]
            return pm.Triangular(name, lower=lower_bound, c=mode, upper=upper_bound)

        elif dist_type == 'halfnormal':
            if not dist_params:
                raise ValueError(f"Missing scale for halfnormal distribution of '{name}'")
            scale = dist_params[0]
            return pm.TruncatedNormal(name, mu=0, sigma=scale, lower=lower_bound, upper=upper_bound)

        elif dist_type == 'truncnorm':
            if not dist_params:
                raise ValueError(f"Missing scale for truncnorm distribution of '{name}'")
            scale = dist_params[0]
            return pm.TruncatedNormal(name, mu=0, sigma=scale, lower=lower_bound, upper=upper_bound)

        elif dist_type == 'gamma':
            if not dist_params:
                raise ValueError(f"Missing shape and scale for gamma distribution of '{name}'")
            shape, scale = dist_params
            return pm.Gamma(name, alpha=shape, beta=scale)

        elif dist_type == 'beta':
            if not dist_params:
                raise ValueError(f"Missing shape and scale for beta distribution of '{name}'")
            shape, scale = dist_params
            return pm.Beta(name, alpha=shape, beta=scale)

        elif dist_type == 'exponnorm':
            if not dist_params:
                raise ValueError(f"Missing shape and scale for exponnorm distribution of '{name}'")
            nu, mu, sigma = dist_params
            return pm.ExGaussian(name, mu=mu, sigma=sigma, nu=nu)

        else:
            raise ValueError(f"Unknown distribution type '{dist_type}' for parameter '{name}'")

    def _create_distributions(self):
        """
        Create prior distributions for each parameter using the specified type and settings.
        """
        with pm.Model() as model:

            msum = self._get_distribution('msum', self._require_param('msum'))
            q = self._get_distribution('q', self._require_param('q'))

            m1 = msum / (q + 1)
            m2 = q * m1

            # feh_min = self.tracks.fehs.min()
            # feh_max = self.tracks.fehs.max()
            log_age_unit = self._get_distribution('log_age_unit', self._require_param('log_age'))
            # rescale to physical log-age range
            lo, hi = 6.5, 10.13
            log_age = pm.Deterministic("log_age", lo + log_age_unit * (hi - lo))

            metallicity = self._get_distribution('metallicity', self._require_param('metallicity'))
            ebv = self._get_distribution('ebv', self._require_param('ebv'))
            cosi = self._get_distribution('cosi', self._require_param('cosi'))
            incl = pm.Deterministic("incl", pt.arccos(cosi) * 180 / np.pi)
            per0 = self._get_distribution('per0', self._require_param('per0'))

            # # degrees -> radians
            # cos_i = pt.sin((90 - incl) * (2*pt.pi)/360)          # == cos(incl) where incl is in degrees
            # sinw  = pt.sin(per0 * (2*pt.pi)/360)                 # per0 is ω in degrees

            # fac = (1 - ecc**2) / (1 + pt.abs(ecc * sinw))
            # rsumfrac_lower = cos_i * fac
            # # Roche-lobe sum at periastron (q-aware)
            # def RLfrac(q):
            #     return 0.49*q**(2/3) / (0.6*q**(2/3) + pt.log1p(q**(1/3)))

            # RL1 = RLfrac(q)
            # RL2 = RLfrac(1/q)                # <-- important!
            # rsumfrac_upper = (RL1 + RL2) * (1 - ecc)
            rsumfrac = self._get_distribution('rsumfrac', self._require_param('rsumfrac'))
            log_ecc_upper = pt.log(1 - rsumfrac)
            log_ecc = self._get_distribution('log_ecc', self._require_param('ecc'), dependent_upper=log_ecc_upper)
            ecc = pm.Deterministic("ecc", pt.exp(log_ecc))
            distance = self._get_distribution('distance', self._require_param('distance'))

            # eps = 1e-9
            # # Geometry-enforcing potentials
            # pm.Potential("geom_eclipse",
            #     pt.switch(rsumfrac > rsumfrac_lower + eps, 0.0, -np.inf))

            # # Non-contact at periastron (your original intent):
            # pm.Potential("no_contact_periastron",
            #     pt.switch(ecc < 1 - rsumfrac - eps, 0.0, -np.inf))

            self.distributions = model

        label_keys = list(self.labels_dict.keys())
        self.pymc_rvs = [self.distributions[k] for k in label_keys if k not in self.keys_derived]
        self.pymc_names = [rv.name for rv in self.pymc_rvs]

        self._attach_joint_logp_functions()

    def _attach_joint_logp_functions(self):
        """
        Build joint logp function that takes **constrained** values
        for each free RV in the current model.
        """

        m = self.distributions

        # Create a symbolic input (constrained space) per free RV.
        # Assumes scalar RVs (shape=()). If any are arrays, swap to pt.vector/pt.matrix accordingly.
        name_to_sym = {}
        logp_terms = []
        with m:
            for rv in self.pymc_rvs:
                sym = pt.scalar(rv.name + "_in")  # constrained scalar input
                name_to_sym[rv.name] = sym
                # pm.logp(rv, sym) constructs the correct logp including transforms/Jacobians
                logp_terms.append(pm.logp(rv, sym))

            joint_logp_sym = pt.add(*logp_terms) if len(logp_terms) else pt.as_tensor_variable(0.0)

        # Compile joint logp: order of inputs is deterministic over free_RVs
        inputs_in_order = [name_to_sym[rv.name] for rv in self.pymc_rvs]
        self._logp_input_order = [rv.name for rv in self.pymc_rvs]

        self._joint_logp_sym = joint_logp_sym
        _logp_fn_raw = pytensor.function(inputs_in_order, joint_logp_sym)

        # Nice wrappers so you can pass kwargs or a dict
        def _prep_args(kwargs_or_dict):
            if isinstance(kwargs_or_dict, dict):
                kwargs = kwargs_or_dict
            else:
                kwargs = dict(kwargs_or_dict)
            # Build args in the right order
            args = []
            for name in self._logp_input_order:
                if name not in kwargs:
                    raise KeyError(f"Missing value for parameter '{name}'")
                args.append(np.asarray(kwargs[name], dtype=float))
            return args

        def logp_fn(**kwargs):
            args = _prep_args(kwargs)
            return float(_logp_fn_raw(*args))

        self.logp_fn = logp_fn

    def _require_param(self, key):
        if key not in self.params_dict:
            raise ValueError(f"Missing required parameter '{key}' in config.")
        return self.params_dict[key]

    def _mist_bounds_check(self, m1, m2, log_age, feh):

        if np.isnan(log_age):
            print('log_age is NaN')
            return False
        
        mist_min_mass = self.tracks.masses.min()
        mist_max_mass = self.tracks.masses.max()
        if m1 < mist_min_mass or m1 > mist_max_mass:
            print('m1 out of MIST bounds')
            return False
        if m2 < mist_min_mass or m2 > mist_max_mass:
            print('m2 out of MIST bounds')
            return False

        # mist_min_age = self.tracks.minage
        # mist_max_age = self.tracks.maxage
        # if log_age < mist_min_age or log_age > mist_max_age:
        #     print('log_age out of MIST bounds:', log_age, mist_min_age, mist_max_age)
        #     return False

        mist_min_feh = self.tracks.fehs.min()
        mist_max_feh = self.tracks.fehs.max()
        if feh < mist_min_feh or feh > mist_max_feh:
            print('feh out of MIST bounds')
            return False
        
        return True
    
    def _bounds_for_sample(self, tracks, m, log_age, feh, iso_accuracy=0.1):
        # Interpolate (these often come back as length-1 arrays)
        eeps = tracks.get_eep(m, log_age, feh)
        logTeff, r_interp, logL_interp = tracks.interp_value(
            [m, eeps, feh], ['logTeff', 'radius', 'logL']
        ).T

        # Scalarize
        logTeff     = float(np.asarray(logTeff).ravel()[0])
        r_interp    = float(np.asarray(r_interp).ravel()[0])
        logL_interp = float(np.asarray(logL_interp).ravel()[0])

        f = iso_accuracy / 2.0

        # Teff, radius: multiplicative bands in linear space
        teff_c = 10.0**logTeff
        teff_lo = teff_c * (1.0 - f)
        teff_hi = teff_c * (1.0 + f)

        r_lo = r_interp * (1.0 - f)
        r_hi = r_interp * (1.0 + f)

        # logL: do multiplicative band in L, then convert back to log10(L)
        L_c   = 10.0**logL_interp
        L_lo  = L_c * (1.0 - f)
        L_hi  = L_c * (1.0 + f)
        logL_lo = float(np.log10(L_lo))
        logL_hi = float(np.log10(L_hi))

        # Ensure proper ordering in all cases
        teff_lo, teff_hi = (min(teff_lo, teff_hi), max(teff_lo, teff_hi))
        r_lo,    r_hi    = (min(r_lo, r_hi),       max(r_lo, r_hi))
        logL_lo, logL_hi = (min(logL_lo, logL_hi), max(logL_lo, logL_hi))

        return (teff_lo, teff_hi), (r_lo, r_hi), (logL_lo, logL_hi)

    
    def _bounds_for_system(self, msum, q, log_age, feh, tracks, iso_accuracy=0.1):
        m1 = msum / (q + 1)
        m2 = q * m1

        teff1_bounds, r1_bounds, logL1_bounds = self._bounds_for_sample(tracks, m1, log_age, feh, iso_accuracy)
        teff2_bounds, r2_bounds, logL2_bounds = self._bounds_for_sample(tracks, m2, log_age, feh, iso_accuracy)

        return (teff1_bounds, r1_bounds, logL1_bounds), (teff2_bounds, r2_bounds, logL2_bounds)

    
    def _logpdf_uniform_from_bounds(self, value, lower, upper):
        low  = float(lower)
        high = float(upper)
        if not np.isfinite(low) or not np.isfinite(high):
            return -np.inf
        if high < low:  # fix inverted bounds defensively
            low, high = high, low
        if not (high > low):  # zero-width or invalid band
            return -np.inf
        if low <= float(value) <= high:
            return -np.log(high - low)
        return -np.inf

    
    def compute_logpdf_all_derived(self, derived, core_params, tracks, iso_accuracy=0.1):
        """
        Parameters:
        - derived: dict with teff1, teff2, r1, r2, log_lum1, log_lum2 arrays
        - core_params: dict with msum, q, log_age, metallicity arrays
        - tracks: isochrone model
        """
        total_logp = 0
        msum = core_params['msum']
        q = core_params['q']
        feh = core_params['metallicity']
        m1 = msum / (q + 1) 

        min_age, median_age, max_age = self._get_age_bounds(m1, feh)
        star1_bounds, star2_bounds = self._bounds_for_system(msum, q, derived['log_age'], feh, tracks, iso_accuracy)
        teff1_bounds, r1_bounds, logL1_bounds = star1_bounds
        teff2_bounds, r2_bounds, logL2_bounds = star2_bounds

        # logpdfs
        total_logp += self._logpdf_uniform_from_bounds(derived['teff1'], teff1_bounds[0], teff1_bounds[1])
        total_logp += self._logpdf_uniform_from_bounds(derived['teff2'], teff2_bounds[0], teff2_bounds[1])
        total_logp += self._logpdf_uniform_from_bounds(derived['r1'], r1_bounds[0], r1_bounds[1])
        total_logp += self._logpdf_uniform_from_bounds(derived['r2'], r2_bounds[0], r2_bounds[1])
        total_logp += self._logpdf_uniform_from_bounds(derived['log_lum1'], logL1_bounds[0], logL1_bounds[1])
        total_logp += self._logpdf_uniform_from_bounds(derived['log_lum2'], logL2_bounds[0], logL2_bounds[1])
        total_logp += self._logpdf_log_age(derived['log_age'], min_age, median_age, max_age)

        return total_logp
    

    def _compile_logp_function(self, model):
        """
        Compiles a symbolic logp function from a PyMC model with conditional priors.

        Args:
            model (pm.Model): The PyMC model containing your priors.

        Returns:
            function: Callable that accepts keyword arguments for each variable and returns logp.
        """
        # Extract untransformed RVs and their names
        inputs = []
        values = []

        for rv in model.free_RVs:
            # Create a symbolic input for the untransformed variable
            input_var = pt.scalar(name=rv.name)
            inputs.append(input_var)

            # Substitute the RV in the model with the symbolic input
            values.append((rv, input_var))

        # Compute symbolic logp expression
        logp_expr = model.logp({rv: val for rv, val in values})

        # Compile function for fast evaluation
        logp_fn = pm.compile_pymc(inputs, logp_expr)

        return logp_fn

        
    # def logpdf(self, sample):
    #     """
    #     Parameters:
    #     - samples: list of lists (each inner list is a full parameter vector)
    #             order is given by self.labels_dict.keys()

    #     Returns:
    #     - logp_total: array of joint log-probabilities
    #     """

    #     param_keys = list(self.params_dict.keys())
    #     label_keys = list(self.labels_dict.keys())

    #     # Step 1: Convert list-of-lists to dict-of-lists for derived quantity function
    #     core_params = {k: sample[i] for i, k in enumerate(label_keys)}

    #     # Step 2: Compute derived quantities - should check if these stay the same
    #     # derived = self._compute_derived_quantities(core_params, n_samples)
    #     keys_to_copy = ['teff1', 'r1', 'log_lum1', 'teff2', 'r2', 'log_lum2']
    #     derived = {key: core_params[key] for key in keys_to_copy}

    #     # Step 3: Compute logp of the primary parameters using PyMC
    #     sample_dict = dict(zip(param_keys, sample))
    #     pymc_rvs = [self.distributions[k] for k in label_keys if k not in derived]
    #     pymc_vals = [sample_dict[k] for k in label_keys if k not in derived]
    #     print("Computing logp for PyMC RVs...")
    #     # logp_fn = self.distributions.logp(pymc_rvs, pymc_vals)
    #     # Use model.named_vars to map user-facing names to internal names
    #     compiled_input = {rv.name: float(val) for rv, val in zip(pymc_rvs, pymc_vals)}

    #     logp_pymc = self.logp_fn(**compiled_input)

    #     print("Computing logp for conditional priors...")
    #     # Step 4: Compute logp of the derived quantities
    #     logp_derived = self.compute_logpdf_all_derived(
    #         derived=derived,
    #         core_params=core_params,
    #         tracks=self.tracks,
    #         iso_accuracy=0.1
    #     )

    #     print(logp_pymc, logp_derived)

    #     # Step 5: Total logp
    #     logp_total = logp_pymc + logp_derived

    #     return logp_total
    
    def logpdf(self, samples):
        """
        Parameters
        ----------
        samples : array-like, shape (n_samples, d) or (d,)
            Each row is a full parameter vector in the order given by self.labels_dict.keys()

        Returns
        -------
        logp_total : ndarray, shape (n_samples,)
            Joint log-probabilities per sample.
        """
        print('logpdf samples:', samples)

        # Order & indexing helpers
        label_keys = list(self.labels_dict.keys())
        idx_by_key = {k: i for i, k in enumerate(label_keys)}

        # Which keys are "derived" (i.e., not free PyMC RVs)
        keys_derived = ['teff1', 'r1', 'log_lum1', 'teff2', 'r2', 'log_lum2', 'log_age']

        # Names of PyMC free RVs (exclude derived)
        # (Using the model to get the canonical RV names)
        pymc_rvs   = [self.distributions[k] for k in label_keys if k not in keys_derived]
        pymc_names = [rv.name for rv in pymc_rvs]

        # Normalize input to (n, d)
        X = np.asarray(samples, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        n, d = X.shape
        if d != len(label_keys):
            raise ValueError(f"Each sample must have {len(label_keys)} parameters (got {d}).")

        print('logpdf samples after normalization: ', X)
        out = np.empty(n, dtype=float)

        for row_idx in range(n):
            row = X[row_idx]

            # Full dict of core params for this row (constrained space)
            core_params = {k: float(row[idx_by_key[k]]) for k in label_keys}

            # Subset passed to PyMC joint prior
            compiled_input = {name: core_params[name] for name in pymc_names}

            # Joint logp of PyMC RVs (handles interval/log transforms internally)
            lp_pymc = self.logp_fn(**compiled_input)

            # Derived-quantity logp (kept per-row unless your function supports batch)
            derived = {k: core_params[k] for k in keys_derived if k in core_params}
            lp_derived = self.compute_logpdf_all_derived(
                derived=derived,
                core_params=core_params,
                tracks=self.tracks,
                iso_accuracy=0.1,
            )

            out[row_idx] = lp_pymc + lp_derived

        return out


    # def _get_age_bounds(self, target_masses, target_fehs, mtol=0.1, fehtol=0.25):
    #     """
    #     For each (mass, feh) in target_masses/target_fehs, return
    #     (min_age, median_age, max_age) from tracks.model_grid.df.
    #     """

    #     df_cur = self.mist_df_reindexed
    #     fehs   = df_cur.index.get_level_values("feh").to_numpy().astype(float)
    #     masses = df_cur.index.get_level_values("current_mass").to_numpy().astype(float)

    #     min_ages = []
    #     median_ages = []
    #     max_ages = []
    #     for m, feh in zip(np.atleast_1d(target_masses), np.atleast_1d(target_fehs)):
    #         mask = (np.abs(masses - m) <= mtol) & (np.abs(fehs - feh) <= fehtol)
    #         rows = df_cur[mask]

    #         if rows.empty:
    #             print('no rows found for m, feh:', m, feh)
    #             min_ages.append(np.nan)
    #             median_ages.append(np.nan)
    #             max_ages.append(np.nan)
    #         else:
    #             ages = rows['age'].to_numpy()
    #             min_ages.append(ages.min())
    #             median_ages.append(np.median(ages))
    #             max_ages.append(ages.max())

    #     return np.array(min_ages), np.array(median_ages), np.array(max_ages)  # shape (N, 3)

    # ---- Build a lightweight cache ONCE ----
    def _build_mist_track_cache(self, df):
        """
        Returns a list of per-track dicts with arrays for fast selection.
        Requires columns: ['feh','initial_mass','current_mass','age'].
        'age' must be log10(age/yr).
        """
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        req = {'feh','initial_mass','current_mass','age'}
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        tracks = []
        for (feh, im), g in df.groupby(['feh','initial_mass'], sort=False):
            a = g['age'].to_numpy(float)
            m = g['current_mass'].to_numpy(float)
            good = np.isfinite(a) & np.isfinite(m)
            if good.sum() < 2:
                continue
            # Sort by age so time increases
            o = np.argsort(a[good])
            logA = a[good][o]
            mcur = m[good][o]
            tracks.append({
                'feh': float(feh),
                'initial_mass': float(im),
                'logA': logA,                # log10(age/yr), increasing
                'mcur': mcur,                # current_mass at those ages
                'mmin': float(np.min(mcur)),
                'mmax': float(np.max(mcur)),
            })
        return tracks

    # Attach a cache once (e.g., after you load self.mist_df_reindexed)
    def _ensure_cache(self):
        if not hasattr(self, '_mist_tracks_cache'):
            self._mist_tracks_cache = self._build_mist_track_cache(self.mist_df_reindexed)

    # ---- FAST bounds using the cache ----
    def _get_age_bounds(self, target_masses, target_fehs, mtol=0.1, fehtol=0.25):
        """
        For each (mass, feh), compute (min_age, median_age, max_age) in log10(years)
        by aggregating ages ONLY from within individual tracks that pass through
        |current_mass - m| <= mtol at |feh - feh_target| <= fehtol.
        """

        self._ensure_cache()
        tracks = self._mist_tracks_cache

        tm = np.atleast_1d(target_masses).astype(float)
        tz = np.atleast_1d(target_fehs).astype(float)

        mins, meds, maxs = [], [], []

        for m_t, feh_t in zip(tm, tz):
            ages_hit = []

            # Pre-filter tracks by metallicity and whether they can ever reach m_t
            for tr in tracks:
                if abs(tr['feh'] - feh_t) > fehtol:
                    continue
                # Quick reject if m_t is outside (mmin-mtol, mmax+mtol)
                if m_t < (tr['mmin'] - mtol) or m_t > (tr['mmax'] + mtol):
                    continue

                mcur = tr['mcur']
                # Boolean select rows inside the mass band for THIS track
                sel = np.abs(mcur - m_t) <= mtol
                if np.any(sel):
                    ages_hit.append(tr['logA'][sel])

            if len(ages_hit) == 0:
                # No track crosses the mass band at this [Fe/H]
                mins.append(np.nan); meds.append(np.nan); maxs.append(np.nan)
                continue

            hits = np.concatenate(ages_hit)
            mins.append(np.nanmin(hits))
            meds.append(np.nanmedian(hits))
            maxs.append(np.nanmax(hits))

        return np.array(mins), np.array(meds), np.array(maxs)


    
    def _sample_log_age(self, lo, med, hi, sigma=0.2, size=1, rng=None):
        if np.isnan(lo) or np.isnan(med) or np.isnan(hi):
            print('lo/med/hi is nan')
            return np.full(size, np.nan)
        rng = rng or np.random.default_rng()
        a, b = (lo - med)/sigma, (hi - med)/sigma

        if not (np.isfinite(a) and np.isfinite(b) and (b > a)):
            print('a/b is nan or b <= a')
            return np.full(size, np.nan)

        return rng.uniform(low=lo, high=hi, size=size)
        # return truncnorm(a, b, loc=med, scale=sigma).rvs(size=size, random_state=rng)
    
    def _logpdf_log_age(self, x, lo, med, hi, sigma=0.2):
        low  = float(lo)
        high = float(hi)
        if not np.isfinite(low) or not np.isfinite(high):
            return -np.inf
        if high < low:  # fix inverted bounds defensively
            low, high = high, low
        if not (high > low):  # zero-width or invalid band
            return -np.inf
        a, b = (lo - med)/sigma, (hi - med)/sigma
        return truncnorm(a, b, loc=med, scale=sigma).logpdf(x)

    def _compute_derived_quantities(self, samples, n_samples):

        # log_age_list = []
        teff1_list, teff2_list = [], [] 
        r1_list, r2_list = [], [] 
        log_lum1_list, log_lum2_list = [], [] 

        for i in range(n_samples): 
            print(i)
            msum = samples['msum'][i] 
            log_age = samples['log_age'][i]
            q = samples['q'][i] 
            feh = samples['metallicity'][i] 
            m1 = msum / (q + 1) 
            m2 = q * m1 
            # min_age1, median_age, max_age1 = self._get_age_bounds(m1, feh)
            # min_age2, _, max_age2 = self._get_age_bounds(m2, feh)
            # min_age = max(min_age1, min_age2)
            # max_age = min(max_age1, max_age2)
            # seed0 = self._get_deterministic_rng([m1, feh])
            # rng0 = np.random.default_rng(seed0)
            # log_age = self._sample_log_age(min_age, median_age, max_age, rng=rng0)[0]

            in_range = self._mist_bounds_check(m1, m2, log_age, feh) 
            if not in_range: 
                teff1, r1, log_lum1 = np.nan, np.nan, np.nan 
                teff2, r2, log_lum2 = np.nan, np.nan, np.nan 
            else: 
                seed1 = self._get_deterministic_rng([m1, log_age, feh])
                rng1 = np.random.default_rng(seed1)
                teff1, r1, log_lum1 = interpolate_isochrone(self.tracks, m1, log_age, feh, rng=rng1)
                seed2 = self._get_deterministic_rng([m2, log_age, feh])
                rng2 = np.random.default_rng(seed2)
                teff2, r2, log_lum2 = interpolate_isochrone(self.tracks, m2, log_age, feh, rng=rng2) 
                
            # log_age_list.append(log_age)
            teff1_list.append(teff1) 
            r1_list.append(r1) 
            log_lum1_list.append(log_lum1) 
            teff2_list.append(teff2) 
            r2_list.append(r2) 
            log_lum2_list.append(log_lum2) 
            
        return { 
                #'log_age': np.array(log_age_list),
                'teff1': np.array(teff1_list), 
                'r1': np.array(r1_list), 
                'log_lum1': np.array(log_lum1_list), 
                'teff2': np.array(teff2_list), 
                'r2': np.array(r2_list), 
                'log_lum2': np.array(log_lum2_list), }
    
    def _cut_unphysical_samples(self, samples):

        print('Making cuts...')

        # Binary parameters
        q = samples['q']
        ecc = samples['ecc']
        rsumfrac = samples['rsumfrac']
        msum = samples['msum']
        m1 = msum / (q + 1)
        m2 = q * m1
        
        # Primary stellar parameters
        r1 = samples['r1']
        log_lum1 = samples['log_lum1']
        teff1 = samples['teff1']
        g1 = m1*G_SUN / (r1**2)
        logg1 = np.log10(g1)

        # Secondary stellar parameters
        r2 = samples['r2']
        log_lum2 = samples['log_lum2']
        teff2 = samples['teff2']
        g2 = m2*G_SUN / (r2**2)
        logg2 = np.log10(g2)

        sma = (r1+r2)/rsumfrac

        # cuts = ((logg1 > 5) | (logg2 > 5) | 
        #         (np.isnan(teff1)) | (np.isnan(teff2)) | 
        #         (teff1 < 3500) | (teff2 < 3500) |
        #         ((logg1 < 3.5) & (log_lum1 > np.log10(500))) | 
        #         ((logg2 < 3.5) & (log_lum2 > np.log10(500)))
        #         | (log_age > 10.3) | (r1/roche_lobe_scaled_radius(q)/sma/(1-ecc) >= 1) 
        #         | (r2/roche_lobe_scaled_radius(q)/sma/(1-ecc) >= 1))
        # n_systems_cut = np.count_nonzero(cuts)
        # print('{} systems cut.'.format(n_systems_cut))

        # Individual rejection conditions
        c1 = logg1 > 5
        c2 = logg2 > 5
        c3 = np.isnan(teff1)
        c4 = np.isnan(teff2)
        c5 = teff1 < 3500
        c6 = teff2 < 3500
        c7 = (logg1 < 3.5) & (log_lum1 > np.log10(500))
        c8 = (logg2 < 3.5) & (log_lum2 > np.log10(500))
        c9 = (r1 / (roche_lobe_scaled_radius(q) * sma * (1 - ecc))) >= 1
        c10 = (r2 / (roche_lobe_scaled_radius(1/q) * sma * (1 - ecc))) >= 1   # <-- note 1/q for secondary

        # Stack into a dict for reporting
        conditions = {
            "logg1 > 5": c1,
            "logg2 > 5": c2,
            "NaN teff1": c3,
            "NaN teff2": c4,
            "teff1 < 3500": c5,
            "teff2 < 3500": c6,
            "giant primary too luminous": c7,
            "giant secondary too luminous": c8,
            "primary Roche overflow": c9,
            "secondary Roche overflow": c10,
        }

        # Combine them
        cuts = np.zeros_like(logg1, dtype=bool)
        for reason, mask in conditions.items():
            cuts |= mask

        # Report summary
        n_systems_cut = np.count_nonzero(cuts)
        print(f"{n_systems_cut} systems cut.")
        for reason, mask in conditions.items():
            n_reason = np.count_nonzero(mask)
            if n_reason > 0:
                print(f"  {reason}: {n_reason}")

        samples_cut = {key: val[~cuts] for key, val in samples.items()}

        return samples_cut

    def _get_deterministic_rng(self, param_vector):
        """
        Create a deterministic RNG based on the parameter vector.
        """
        key = "_".join(f"{x:.6e}" for x in param_vector)
        hash_hex = hashlib.sha256(key.encode()).hexdigest()
        seed = int(hash_hex[:8], 16)  # Take first 8 hex digits → 32-bit seed
        return np.random.default_rng(seed)

    def sample(self, n_samples=1):
        """
    Sample parameter values from the configured prior distributions,
    computing derived quantities, filtering for physical systems,
    and ensuring exactly n_samples remain.

    Args:
        n_samples (int): Number of valid physical samples to return.
        rng (np.random.Generator, optional): Random number generator for reproducibility.

    Returns:
        dict: Dictionary of sampled parameter values (including derived quantities).
    """
        collected_samples = None
        collected_count = 0

        batch_size = max(int(n_samples), 10)

        while collected_count < n_samples:
            print(collected_count)
            # Use a fresh seed per batch from the provided rng for reproducibility
            random_seed = self.rng.integers(0, 2**32 - 1)

            with self.distributions:
                prior = pm.sample_prior_predictive(samples=batch_size, random_seed=random_seed)

            samples = {key: prior.prior[key][0] for key in prior.prior.keys()}
            derived = self._compute_derived_quantities(samples, batch_size)  # Pass rng downstream
            samples.update(derived)

            physical_samples = self._cut_unphysical_samples(samples)

            n_new = len(next(iter(physical_samples.values())))  # survivors

            if n_new > 0:
                if collected_samples is None:
                    collected_samples = {key: val for key, val in physical_samples.items()}
                else:
                    for key in collected_samples.keys():
                        collected_samples[key] = np.concatenate([collected_samples[key], physical_samples[key]])

                collected_count = len(next(iter(collected_samples.values())))

            if n_new == 0:
                batch_size *= 2  # Increase batch size if no survivors

        # Trim to exact number requested
        final_samples = {key: val[:n_samples] for key, val in collected_samples.items()}

        # Build list of sample vectors for each parameter
        theta_sample_list = []
        for var in self.labels_dict.keys():
            val = final_samples[var]

            if isinstance(val, xr.DataArray):
                val = val.values
                if val.ndim == 0:
                    theta_sample_list.append([float(val)])  # scalar → list
                else:
                    theta_sample_list.append(list(val))     # array → list
            elif isinstance(val, (float, int)):
                theta_sample_list.append([val])             # scalar → list
            else:
                theta_sample_list.append(list(val))         # assume iterable

        # Transpose to shape (n_samples, n_parameters)
        theta_sample_array = np.array(theta_sample_list).T

        return theta_sample_array
