"""
CGM-WGP: Crop Growth Model + Whole Genome Prediction.

Mechanistic temperature-driven flowering time prediction
combined with genomic (GBLUP) predictions.
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies on simple `import cgm_wgp`
def __getattr__(name):
    """Lazy-load commonly used symbols."""
    _PUBLIC = {
        # config
        "PROJECT_ROOT": ("cgm_wgp.config", "PROJECT_ROOT"),
        "OUTPUT_DIR": ("cgm_wgp.config", "OUTPUT_DIR"),
        "load_crop_config": ("cgm_wgp.config", "load_crop_config"),
        "create_run_dir": ("cgm_wgp.config", "create_run_dir"),
        "resolve_run": ("cgm_wgp.config", "resolve_run"),
        "finalize_run": ("cgm_wgp.config", "finalize_run"),
        "get_output_path": ("cgm_wgp.config", "get_output_path"),
        # mechanistic model
        "cardinal_beta": ("cgm_wgp.mech_fit", "cardinal_beta"),
        "predict_flowering_dap": ("cgm_wgp.mech_fit", "predict_flowering_dap"),
        "load_and_build_dicts": ("cgm_wgp.mech_fit", "load_and_build_dicts"),
        # gblup
        "load_evd": ("cgm_wgp.gblup", "load_evd"),
    }
    if name in _PUBLIC:
        module_path, attr = _PUBLIC[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'cgm_wgp' has no attribute {name!r}")
