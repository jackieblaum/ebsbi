import pytest
from ebsbi.utils import sanitize_passband_label


def test_sanitize_passband_label_tess():
    """Test TESS:T conversion."""
    assert sanitize_passband_label('TESS:T') == 'tess_t'


def test_sanitize_passband_label_gaia():
    """Test Gaia:G conversion."""
    assert sanitize_passband_label('Gaia:G') == 'gaia_g'


def test_sanitize_passband_label_panstarrs():
    """Test Pan-STARRS:g with hyphen."""
    assert sanitize_passband_label('Pan-STARRS:g') == 'pan_starrs_g'


def test_sanitize_passband_label_lowercase():
    """Test already lowercase input."""
    assert sanitize_passband_label('custom_band') == 'custom_band'


def test_sanitize_passband_label_multiple_special():
    """Test multiple special characters."""
    assert sanitize_passband_label('HST-WFC3:F814W') == 'hst_wfc3_f814w'


def test_sanitize_passband_label_empty():
    """Test empty string raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        sanitize_passband_label('')


def test_sanitize_passband_label_whitespace():
    """Test whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        sanitize_passband_label('   ')


def test_sanitize_passband_label_none():
    """Test None input raises TypeError."""
    with pytest.raises(TypeError):
        sanitize_passband_label(None)


def test_sanitize_passband_label_path_traversal():
    """Test path traversal sequences are removed."""
    result = sanitize_passband_label('../../TESS:T')
    assert '..' not in result
    assert '/' not in result


def test_sanitize_passband_label_slashes():
    """Test forward and backslashes replaced."""
    assert '/' not in sanitize_passband_label('test/with/slashes')
    assert '\\' not in sanitize_passband_label('test\\with\\backslashes')


def test_sanitize_passband_label_special_chars():
    """Test special characters replaced."""
    result = sanitize_passband_label('TESS*?<>|:T')
    assert all(c not in result for c in '*?<>|:')


def test_sanitize_passband_label_strips_whitespace():
    """Test leading/trailing whitespace removed."""
    assert sanitize_passband_label(' TESS:T ') == 'tess_t'


def test_sanitize_passband_label_tabs():
    """Test tab characters are removed."""
    result = sanitize_passband_label('test\ttab')
    assert '\t' not in result


def test_sanitize_passband_label_newlines():
    """Test newline characters are removed."""
    result = sanitize_passband_label('test\nline')
    assert '\n' not in result


def test_sanitize_passband_label_unicode_format():
    """Test Unicode format characters are removed."""
    result = sanitize_passband_label('test\u200btest')  # zero-width space
    assert '\u200b' not in result


def test_sanitize_passband_label_rtl_override():
    """Test right-to-left override is removed."""
    result = sanitize_passband_label('test\u202etest')
    assert '\u202e' not in result


def test_sanitize_passband_label_combining_marks():
    """Test combining marks are removed."""
    result = sanitize_passband_label('test\u0301')  # Combining acute
    assert '\u0301' not in result


def test_sanitize_passband_label_private_use():
    """Test private use area characters removed."""
    result = sanitize_passband_label('test\ue000')
    assert '\ue000' not in result
