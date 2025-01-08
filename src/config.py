import os
import yaml
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self, config_path):
        """
        Initialize configuration with default values and load from the provided YAML file.
        """
        self.mcmc = False
        self.sequential = False
        self.real_data = False
        self.sim_file = None
        self.lookup_file = None
        self.isochrone_file = None
        self.times = np.array([])
        self.params_dict = {}
        self.trace_dict = {}
        self.nbi_dict = {}
        self.eb_path = None
        self.eb_filename = None
        self.mcmc_figpath = None
        self.mcmc_bundlepath = None

        self.load_config(config_path)

    def load_config(self, config_path):
        """
        Load configuration from a YAML file and validate contents.
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loading configuration from {config_path}.")
                self._parse_boolean_flags(config)
                self._load_files(config)
                self._validate_dicts(config)
                self._create_directories(config)
                logger.info("Configuration successfully loaded.")
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise

    def _parse_boolean_flags(self, config):
        """
        Parse and set boolean flags for MCMC, SEQUENTIAL, and REAL_DATA.
        """
        self.mcmc = self._parse_bool(config.get("MCMC", False))
        self.sequential = self._parse_bool(config.get("SEQUENTIAL", False))
        self.real_data = self._parse_bool(config.get("REAL_DATA", False))

    @staticmethod
    def _parse_bool(value):
        """
        Convert a value to boolean.
        """
        return str(value).lower() in ["true", "1", "yes"]

    def _load_files(self, config):
        """
        Load file paths and validate their existence.
        """
        self.sim_file = self._validate_file(config.get("SIM_FILE"), "Simulation file")
        self.lookup_file = self._validate_file(config.get("LOOKUP_FILE"), "Lookup file")
        self.isochrone_file = self._validate_file(config.get("ISOCHRONE_FILE"), "Isochrone file")
        self.times = self._load_numpy_file(config.get("TIMES_FILE"), "Times file")

    @staticmethod
    def _validate_file(file_path, file_description):
        """
        Check if a file exists, and return its path if valid.
        """
        if file_path and os.path.exists(file_path):
            logger.info(f"{file_description} loaded from {file_path}.")
            return file_path
        else:
            logger.warning(f"{file_description} not found at {file_path}.")
            return None

    @staticmethod
    def _load_numpy_file(file_path, file_description):
        """
        Load a NumPy file and handle errors.
        """
        if file_path and os.path.exists(file_path):
            try:
                data = np.load(file_path)
                logger.info(f"{file_description} loaded successfully.")
                return data
            except Exception as e:
                logger.error(f"Error loading {file_description}: {e}")
        else:
            logger.warning(f"{file_description} not found.")
        return np.array([])

    def _validate_dicts(self, config):
        """
        Validate dictionaries in the configuration.
        """
        self.params_dict = self._validate_dict(config.get("PARAMS"), "PARAMS")
        self.trace_dict = self._validate_dict(config.get("TRACE"), "TRACE")
        self.nbi_dict = self._validate_dict(config.get("NBI"), "NBI")

    @staticmethod
    def _validate_dict(value, name):
        """
        Validate that a configuration value is a dictionary.
        """
        if isinstance(value, dict):
            logger.info(f"{name} successfully loaded.")
            return value
        else:
            logger.error(f"{name} must be a dictionary.")
            raise TypeError(f"{name} must be a dictionary.")

    def _create_directories(self, config):
        """
        Create directories specified in the configuration.
        """
        self.eb_path = self._create_directory(config.get("EB_PATH"), "EB_PATH")
        self.mcmc_figpath = self._create_directory(config.get("MCMC_FIGPATH"), "MCMC_FIGPATH")
        self.mcmc_bundlepath = self._create_directory(config.get("MCMC_BUNDLEPATH"), "MCMC_BUNDLEPATH")
        self.eb_filename = config.get("EB_FILENAME")
        if self.eb_path and self.eb_filename:
            eb_full_path = os.path.join(self.eb_path, self.eb_filename)
            if not os.path.exists(eb_full_path):
                logger.warning(f"EB file not found at {eb_full_path}.")

    @staticmethod
    def _create_directory(path, description):
        """
        Create a directory if it doesn't exist.
        """
        if path:
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"{description} directory created at {path}.")
                return path
            except Exception as e:
                logger.error(f"Error creating {description} directory: {e}")
                return None
        else:
            logger.warning(f"{description} directory not specified.")
            return None

    def get_param_vals(self):
        """
        Return parameter values from params_dict.
        """
        return {param: subdict.get("VALUE") for param, subdict in self.params_dict.items() if "VALUE" in subdict}
