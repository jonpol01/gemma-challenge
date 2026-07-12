"""Minimal sitecustomize: prometheus 500-fix + activate serve_patch_inmem (no engine yet)."""
def _patch_prometheus():
    try:
        import prometheus_fastapi_instrumentator as p
        p.Instrumentator.instrument = lambda s,*a,**k: s
        p.Instrumentator.expose = lambda s,*a,**k: s
    except Exception:
        pass
    try:
        import prometheus_fastapi_instrumentator.routing as r
        o=getattr(r,"_get_route_name",None)
        if o is not None and not getattr(o,"_g",False):
            def g(scope,routes):
                try: return o(scope,routes)
                except AttributeError: return None
            g._g=True; r._get_route_name=g
    except Exception:
        pass
_patch_prometheus()
import serve_patch_inmem  # noqa
