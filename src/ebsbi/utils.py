"""Utility functions for ebsbi."""
import unicodedata


def sanitize_passband_label(label):
    """
    Convert passband labels to filesystem-safe names.

    Replaces colons, hyphens, slashes, and other special characters
    with underscores. Removes leading/trailing whitespace, path
    traversal sequences, and dangerous Unicode characters.
    Converts to lowercase.

    Parameters
    ----------
    label : str
        Passband label (e.g., 'TESS:T', 'Gaia:G', 'Pan-STARRS:g')

    Returns
    -------
    str
        Sanitized label safe for filenames (e.g., 'tess_t', 'gaia_g')

    Raises
    ------
    TypeError
        If label is not a string
    ValueError
        If label is empty or results in empty filename after sanitization

    Examples
    --------
    >>> sanitize_passband_label('TESS:T')
    'tess_t'
    >>> sanitize_passband_label('Pan-STARRS:g')
    'pan_starrs_g'
    >>> sanitize_passband_label(' Gaia:G ')
    'gaia_g'
    """
    if not isinstance(label, str):
        raise TypeError(f"Expected string, got {type(label).__name__}")

    # Remove leading/trailing whitespace
    label = label.strip()

    if not label:
        raise ValueError("Label cannot be empty or whitespace-only")

    # Normalize Unicode to remove combining characters and convert to NFKC form
    sanitized = unicodedata.normalize('NFKC', label)

    # Remove Unicode format and control characters (security risk)
    sanitized = ''.join(
        char for char in sanitized
        if unicodedata.category(char) not in ['Cf', 'Cc']
    )

    # Convert to lowercase
    sanitized = sanitized.lower()

    # Replace filesystem-unsafe characters with underscores
    # Covers: : - / \ * ? " < > | and all whitespace
    unsafe_chars = [':', '-', '/', '\\', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r', '\v', '\f']
    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove any path traversal sequences
    sanitized = sanitized.replace('..', '')

    # Remove leading dots (hidden files on Unix)
    sanitized = sanitized.lstrip('.')

    # Collapse multiple underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    if not sanitized:
        raise ValueError(f"Label '{label}' results in empty filename after sanitization")

    return sanitized
