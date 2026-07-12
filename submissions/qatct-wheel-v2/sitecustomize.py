"""Minimal sitecustomize (auto-loaded via PYTHONPATH): neutralize the
prometheus-fastapi-instrumentator so vLLM 0.22.x's _IncludedRouter (no .path)
stops 500-ing every request incl. the /v1/models readiness probe.
HTTP-metrics-only; inference / greedy-identity / PPL unaffected."""
def _patch():
    try:
        import prometheus_fastapi_instrumentator as p
        p.Instrumentator.instrument = lambda self, *a, **k: self
        p.Instrumentator.expose = lambda self, *a, **k: self
    except Exception:
        pass
    try:
        import prometheus_fastapi_instrumentator.routing as r
        o = getattr(r, "_get_route_name", None)
        if o is not None and not getattr(o, "_g", False):
            def g(scope, routes):
                try:
                    return o(scope, routes)
                except AttributeError:
                    return None
            g._g = True
            r._get_route_name = g
    except Exception:
        pass
_patch()
