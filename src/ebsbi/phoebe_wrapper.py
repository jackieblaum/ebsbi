"""
PHOEBE wrapper for generating binary star SEDs at specified orbital phases.
"""
import numpy as np
import phoebe
from astropy import units as u
import extinction


class PhoebeWrapper:
    """
    Wrapper for PHOEBE to compute SEDs for eclipsing binary systems.

    Attributes
    ----------
    bundle : phoebe.Bundle
        PHOEBE bundle containing the binary system configuration

    Class Attributes
    ----------------
    CANONICAL_FILTERS : list
        Fixed list of all 23 supported passbands in canonical order.
        SEDs always return arrays with this ordering for ML compatibility.
    """

    # Canonical filter order (fixed for all systems)
    CANONICAL_FILTERS = [
        'GALEX:FUV',
        'GALEX:NUV',
        'Johnson:B',
        'Johnson:U',
        'Johnson:V',
        'Pan-Starrs:g',
        'Pan-Starrs:i',
        'Pan-Starrs:r',
        'Pan-Starrs:w',
        'Pan-Starrs:y',
        'Pan-Starrs:z',
        'SDSS:g',
        'SDSS:i',
        'SDSS:r',
        'SDSS:u',
        'SDSS:z',
        'Gaia:BP',
        'Gaia:G',
        'Gaia:RP',
        '2MASS:J',
        '2MASS:H',
        '2MASS:Ks',
        'WISE:W1',
        # 'WISE:W2',
    ]

    def __init__(self, bundle, distance, ebv=0.0):
        """
        Initialize PHOEBE wrapper with existing bundle.

        Parameters
        ----------
        bundle : phoebe.Bundle
            Pre-configured PHOEBE bundle with binary system parameters
        distance : float
            Distance to system (pc)
        ebv : float, optional
            Color excess E(B-V) for extinction (default: 0.0)
        """
        self.bundle = bundle
        self.distance = distance
        self.ebv = ebv

    def _apply_extinction(self, fluxes_array):
        """
        Apply Fitzpatrick99 extinction to band-integrated fluxes.

        Computes band-averaged attenuation coefficients by integrating
        the F99 extinction curve across each filter's transmission curve.

        Parameters
        ----------
        fluxes_array : ndarray, shape [n_phases, n_filters]
            Intrinsic fluxes from PHOEBE in W/m²

        Returns
        -------
        fluxes_ext : ndarray, shape [n_phases, n_filters]
            Extincted fluxes in W/m²

        Raises
        ------
        ValueError
            If E(B-V) < 0 or wavelengths outside F99 range (910-60000 Å)
            or filter has non-positive integral response

        Notes
        -----
        Uses Fitzpatrick99 extinction law with R_V = 3.1 (standard MW ISM).
        Extinction is applied as: F_obs = F_int × 10^(-0.4 × A_eff)
        where A_eff is computed by averaging attenuation factors across
        the filter bandpass (not averaging A(λ) directly).
        """
        # No-op cases
        if self.ebv is None or self.ebv == 0:
            return fluxes_array

        # Validation
        if self.ebv < 0:
            raise ValueError(f"E(B-V) must be non-negative, got {self.ebv}")

        # Compute A(V) with fixed R_V for standard Milky Way ISM
        r_v = 3.1
        a_v = r_v * self.ebv

        # Compute per-band attenuation coefficients
        att_band = np.full(len(self.CANONICAL_FILTERS), np.nan)

        for i, filt in enumerate(self.CANONICAL_FILTERS):
            pb = phoebe.get_passband(filt)
            wl = np.asarray(pb.ptf_table["wl"], float) * 1e10   # m → Å
            R  = np.asarray(pb.ptf_table["fl"], float)          # transmission

            # Sort by wavelength and clip response to non-negative
            idx = np.argsort(wl)
            wl, R = wl[idx], R[idx]
            R = np.clip(R, 0.0, None)

            # Validate wavelength range for F99 (910-60000 Å)
            if np.any(wl < 910) or np.any(wl > 60000):
                raise ValueError(
                    f"Filter {filt} has wavelengths outside F99 range (910-60000 Å)"
                )

            # Validate filter integral
            den = np.trapz(R, wl)
            if den <= 0:
                raise ValueError(
                    f"Filter {filt} has non-positive integral response"
                )

            # Compute extinction curve A(λ) using F99
            A_lam = extinction.fitzpatrick99(wl, a_v, r_v=r_v)

            # Convert to attenuation factors (linear, not magnitude)
            att_lam = 10**(-0.4 * A_lam)

            # Band-averaged attenuation (transmission-weighted)
            att_eff = np.trapz(att_lam * R, wl) / den

            # Clip to prevent numerical issues
            att_eff = np.clip(att_eff, 1e-300, 1.0)

            att_band[i] = att_eff

        # Apply extinction (broadcast over phase dimension)
        return fluxes_array * att_band[None, :]

    def compute_sed(self, phases=None, wavelength_min=1000, wavelength_max=100000,
                    wavelength_num=1000):
        """
        Compute passband fluxes at specified phases.

        Note: wavelength_min, wavelength_max, wavelength_num are ignored.
        Kept for API compatibility but not used in photometric-only mode.

        Parameters
        ----------
        phases : list or array, optional
            Orbital phases to compute (0.0 = superior conjunction). If None,
            a random phase is generated per survey and used for all passbands
            in that survey.
        wavelength_min : float, optional
            Ignored (kept for API compatibility)
        wavelength_max : float, optional
            Ignored (kept for API compatibility)
        wavelength_num : int, optional
            Ignored (kept for API compatibility)

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'phases': input phases (or per-filter phases if phases=None)
            - 'wavelengths': array of 23 filter effective wavelengths (nm)
            - 'fluxes': 2D array [n_phases, 23] (W/m²)
                NaN for filters that couldn't be computed for this system
            - 'filter_mask': boolean array [23] indicating valid filters
            - 'filter_names': list of 23 canonical filter names

            Always returns fixed shape for ML compatibility. Invalid filters
            (e.g., those with extreme stellar temps) are filled with NaN.
        """
        import numpy as np
        from astropy import units as u

        # Set distance in bundle with proper units
        # PHOEBE requires astropy units; without units it interprets as meters
        self.bundle.set_value('distance', value=self.distance * u.pc)

        default_phases_by_filter = None
        if phases is None:
            rng = np.random.default_rng()
            survey_phases = {}
            default_phases_by_filter = []
            for filt in self.CANONICAL_FILTERS:
                survey = filt.split(':', 1)[0]
                if survey not in survey_phases:
                    survey_phases[survey] = rng.uniform(0.0, 1.0)
                default_phases_by_filter.append(survey_phases[survey])
            phases = np.asarray(default_phases_by_filter, dtype=float)

        phase_count = 1 if default_phases_by_filter is not None else len(phases)
        # Build fixed-size wavelength and bandwidth arrays (23 filters)
        # Fill with NaN for invalid filters
        eff_wavelengths = np.full(len(self.CANONICAL_FILTERS), np.nan)
        filter_mask = np.zeros(len(self.CANONICAL_FILTERS), dtype=bool)

        # Add light curve datasets ONLY for valid filters
        nonempty_filters = False
        for filt_idx, filt in enumerate(self.CANONICAL_FILTERS):
            try:
                pb = phoebe.get_passband(filt)
                if filt.startswith('Johnson:') or filt.startswith('SDSS:') or filt.startswith('Pan-Starrs:'):
                    eff_wavelengths[filt_idx] = pb.effwl / 10.0
                elif filt.startswith('WISE:W1') or filt.startswith('WISE:W2'):
                    eff_wavelengths[filt_idx] = pb.effwl * 10**3
                else:
                    eff_wavelengths[filt_idx] = pb.effwl
                filter_mask[filt_idx] = True
                # Replace colons and hyphens in filter names for valid dataset labels
                dataset_label = f'lc_{filt}'.replace(':', '_').replace('-', '_')
                if default_phases_by_filter is not None:
                    compute_phases = [default_phases_by_filter[filt_idx]]
                else:
                    compute_phases = phases
                self.bundle.add_dataset(
                    'lc',
                    compute_phases=compute_phases,
                    passband=filt,
                    dataset=dataset_label,
                    pblum_mode='absolute'  # Set pblum independently for each star
                )
                nonempty_filters = True
            except Exception as e:
                print(f"Error adding dataset {filt}: {e}")

        if not nonempty_filters:
            return {
                'phases': phases,
                'wavelengths': eff_wavelengths,
                'fluxes': np.full((phase_count, len(self.CANONICAL_FILTERS)), np.nan),
                'filter_mask': np.zeros(len(self.CANONICAL_FILTERS), dtype=bool),
                'filter_names': self.CANONICAL_FILTERS,
            }

        try:
            self.bundle.compute_pblums()
            self.bundle.run_compute()
        except Exception as e:
            print(f"Error computing pblums: {e}")
            return {
                'phases': phases,
                'wavelengths': eff_wavelengths,
                'fluxes': np.full((phase_count, len(self.CANONICAL_FILTERS)), np.nan),
                'filter_mask': np.zeros(len(self.CANONICAL_FILTERS), dtype=bool),
                'filter_names': self.CANONICAL_FILTERS,
            }

        # Extract fluxes: build [n_phases, 23] array with NaN for invalid filters
        # PHOEBE returns fluxes in W/m²
        fluxes_array = np.full((phase_count, len(self.CANONICAL_FILTERS)), np.nan)

        for filt_idx, filt in enumerate(self.CANONICAL_FILTERS):
            if not filter_mask[filt_idx]:
                continue

            dataset_label = f'lc_{filt}'.replace(':', '_').replace('-', '_')

            vals = self.bundle.get_value(
                'fluxes',
                dataset=dataset_label,
                context='model'
            )  # expected shape (n_phases,)

            vals = np.asarray(vals, dtype=float)  # W/m^2
            if phase_count == 1:
                fluxes_array[0, filt_idx] = vals[0]
            else:
                fluxes_array[:, filt_idx] = vals

        filter_mask = filter_mask & np.all(np.isfinite(fluxes_array), axis=0)

        return {
            'phases': phases,
            'wavelengths': eff_wavelengths,
            'fluxes': fluxes_array,
            'filter_mask': filter_mask,
            'filter_names': self.CANONICAL_FILTERS,
        }
