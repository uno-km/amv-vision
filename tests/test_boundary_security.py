import pytest
from termux_vision.vlm.cache import ModelCacheManager, validate_model_id, require_path_within_root

def test_model_id_path_traversal_is_rejected():
    """Verify that malicious model IDs with path traversal are strictly rejected."""
    bad_ids = [
        "../../etc/passwd",
        "../models",
        "/absolute/path",
        "model; rm -rf /",
        "model\0nullbyte",
        "-leadinghyphen",
        "",
        "   "
    ]
    for bad_id in bad_ids:
        with pytest.raises(ValueError):
            validate_model_id(bad_id)

def test_require_path_within_root_rejects_escape(tmp_path):
    """Verify that paths resolving outside allowed_root raise ValueError."""
    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    
    # Direct escape
    with pytest.raises(ValueError):
        require_path_within_root(str(outside), str(root))
        
    # Root itself when allow_root_itself=False
    with pytest.raises(ValueError):
        require_path_within_root(str(root), str(root), allow_root_itself=False)
