# compat_jax_xla_shim.py
def apply():
    import importlib
    import jax
    try:
        from jax.core import pytype_aval_mappings as _pam
        xla = importlib.import_module("jax.interpreters.xla")
        if not hasattr(xla, "pytype_aval_mappings"):
            setattr(xla, "pytype_aval_mappings", _pam)
    except Exception:
        # 失敗しても致命化しない（古いJAXでは既に存在する可能性あり）
        pass