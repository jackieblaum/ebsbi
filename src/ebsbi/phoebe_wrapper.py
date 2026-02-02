"""
PHOEBE wrapper for generating binary star SEDs at specified orbital phases.
"""
import numpy as np
import phoebe
from astropy import units as u


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
        '2MASS:K',
        'WISE:W1',
        'WISE:W2',
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

    def compute_sed(self, phases, wavelength_min=1000, wavelength_max=100000,
                    wavelength_num=1000):
        """
        Compute passband fluxes at specified phases.

        Note: wavelength_min, wavelength_max, wavelength_num are ignored.
        Kept for API compatibility but not used in photometric-only mode.

        Parameters
        ----------
        phases : list or array
            Orbital phases to compute (0.0 = superior conjunction)
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
            - 'phases': input phases
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
        # Build fixed-size wavelength and bandwidth arrays (23 filters)
        # Fill with NaN for invalid filters
        eff_wavelengths = np.full(len(self.CANONICAL_FILTERS), np.nan)
        filter_mask = np.zeros(len(self.CANONICAL_FILTERS), dtype=bool)

        # Add light curve datasets ONLY for valid filters
        nonempty_filters = False
        for filt_idx, filt in enumerate(self.CANONICAL_FILTERS):
            try:
                pb = phoebe.get_passband(filt)
                eff_wavelengths[filt_idx] = (pb.effwl * pb.wlunits).to(u.nm).value
                filter_mask[filt_idx] = True
                # Replace colons and hyphens in filter names for valid dataset labels
                dataset_label = f'lc_{filt}'.replace(':', '_').replace('-', '_')
                self.bundle.add_dataset(
                    'lc',
                    compute_phases=phases,
                    passband=filt,
                    dataset=dataset_label,
                    pblum_mode='absolute'  # Set pblum independently for each star
                )
                nonempty_filters = True
            except Exception:
                pass

        if not nonempty_filters:
            return {
                'phases': phases,
                'wavelengths': eff_wavelengths,
                'fluxes': np.full((len(phases), len(self.CANONICAL_FILTERS)), np.nan),
                'filter_mask': np.zeros(len(self.CANONICAL_FILTERS), dtype=bool),
                'filter_names': self.CANONICAL_FILTERS,
            }

        try:
            self.bundle.compute_pblums()
            self.bundle.run_compute()
        except Exception:
            return {
                'phases': phases,
                'wavelengths': eff_wavelengths,
                'fluxes': np.full((len(phases), len(self.CANONICAL_FILTERS)), np.nan),
                'filter_mask': np.zeros(len(self.CANONICAL_FILTERS), dtype=bool),
                'filter_names': self.CANONICAL_FILTERS,
            }

        # Extract fluxes: build [n_phases, 23] array with NaN for invalid filters
        # PHOEBE returns fluxes in W/m²
        fluxes_array = np.full((len(phases), len(self.CANONICAL_FILTERS)), np.nan)

        for filt_idx, filt in enumerate(self.CANONICAL_FILTERS):
            if not filter_mask[filt_idx]:
                continue

            dataset_label = f'lc_{filt}'.replace(':', '_').replace('-', '_')

            vals = self.bundle.get_value(
                'fluxes',
                dataset=dataset_label,
                context='model'
            )  # expected shape (n_phases,)

            fluxes_array[:, filt_idx] = np.asarray(vals, dtype=float)  # W/m^2

        filter_mask = filter_mask & np.all(np.isfinite(fluxes_array), axis=0)

        return {
            'phases': phases,
            'wavelengths': eff_wavelengths,
            'fluxes': fluxes_array,
            'filter_mask': filter_mask,
            'filter_names': self.CANONICAL_FILTERS,
        }
