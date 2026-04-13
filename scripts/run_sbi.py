import os
import sys
import numpy as np
import nbi
from isochrones import get_ichrone

from ebsbi.model import EBModel
from ebsbi.priors import EBPriors
from ebsbi.config import Config
from ebsbi.engine import create_engine

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="isochrones")


def main(config_path):

    rng = np.random.default_rng(seed=42)

    # Load configuration
    conf = Config(config_path)

    tracks = get_ichrone('mist', tracks=True)

    # Initialize Priors
    priors = EBPriors(conf.params_dict, tracks, rng=rng)

    # Sample parameters
    theta_sample = priors.sample(n_samples=1)

    # Initialize EB Model
    model = EBModel(eb_path=conf.eb_path, 
                    params_dict=conf.labels_dict, 
                    times=conf.params_dict['times']['VALUE'])

    # Generate noiseless light curve
    lc_flux = model.generate_light_curve(theta_sample, name='sample1')

    if lc_flux is None:
        raise RuntimeError("PHOEBE model failed to compute. Check input parameters.")

    # # Generate SED fluxes
    # sed_fluxes = {}
    # for filt_name, filt_props in conf.filters_dict.items():
    #     distance_pc = theta_sample.get('dist', 1000)  # Fallback distance in parsecs
    #     mag = theta_sample.get('mag', filt_props.get('MAG', 10.0))
    #     sed_fluxes[filt_name] = model.generate_sed(theta_sample, 
    #                                                distance_pc, 
    #                                                zero_point=filt_props.get('ZEROPOINT', 25.0), 
    #                                                lambda_eff=filt_props.get('LAMBDA_EFF', 0.64))

    # Add instrumental noise
    lc_noisy = model.instrumental_noise(lc_flux)

    # sed_noisy = {band: flux * (1 + np.random.normal(0, 0.02)) for band, flux in sed_fluxes.items()}

    # Featurizer setup
    # featurizer = nbi.get_featurizer(
    #     conf.nbi_dict.get('FEATURIZER', 'resnetrnn'),
    #     1,
    #     conf.nbi_dict.get('FEAT_DIM', 128),
    #     depth=conf.nbi_dict.get('FEAT_DEPTH', 4)
    # )

    # Create NBI Engine
    engine = create_engine(conf, model, priors)

    # Run Inference
    engine.fit(
        x_obs=lc_noisy,
        noise=model.instrumental_noise,
        log_like=model.log_likelihood,
        n_sims=conf.nbi_dict['N_SIMS'],
        n_rounds=conf.nbi_dict['N_ROUNDS'],
        n_epochs=conf.nbi_dict['N_EPOCHS'],
        batch_size=conf.nbi_dict['BATCH_SIZE'],
        f_val=conf.nbi_dict['F_VAL'],
        lr=conf.nbi_dict['LR'],
        min_lr=conf.nbi_dict['MIN_LR'],
        early_stop_patience=conf.nbi_dict['EARLY_STOP_PATIENCE'],  # fixed the key name
        neff_stop=conf.nbi_dict['NEFF_STOP'],
        y_true=theta_sample
        )

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'docs/config_linear_times_sequential.yml'
    main(config_path)
