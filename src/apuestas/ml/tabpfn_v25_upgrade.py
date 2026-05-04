"""TabPFN v2.5 upgrade shim — Sprint 14 #140.

TabPFN-2.5 (arXiv 2511.08667, nov 2025) soporta 50k filas y 2000 features
(vs 10k/500 de v1). Este wrapper hace fallback: usa v2.5 si disponible,
sino v1 legacy (apuestas.ml.tabpfn_stacker).

Dependencies:
  pip install tabpfn==2.5.0  (cuando disponible en PyPI; ahora es preview)

Uso:
    from apuestas.ml.tabpfn_v25_upgrade import TabPFNv25Stacker
    stacker = TabPFNv25Stacker()
    stacker.fit(X_train, y_train)
    proba = stacker.predict_proba(X_test)

Beneficio: mejor calibración + handles datasets 5× mayores que v1.
"""

from __future__ import annotations

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


def check_tabpfn_version() -> tuple[str | None, bool]:
    """Returns (version, is_v2_plus)."""
    try:
        import tabpfn as _tp

        ver = getattr(_tp, "__version__", "unknown")
        is_v2 = ver >= "2.0.0"
        return ver, is_v2
    except ImportError:
        return None, False


class TabPFNv25Stacker:
    """Auto-select v2.5 vs v1 legacy. Compat con sklearn."""

    def __init__(self, max_samples: int = 50_000, **kwargs):
        self.max_samples = max_samples
        self._impl = None
        self._version = None

    def _init_impl(self):
        ver, is_v2 = check_tabpfn_version()
        self._version = ver
        if is_v2:
            try:
                from tabpfn import TabPFNClassifier  # type: ignore

                self._impl = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                logger.info("tabpfn_v25.loaded", version=ver)
                return
            except Exception as exc:
                logger.warning("tabpfn_v25.init_fail", error=str(exc)[:80])
        # Fallback v1
        try:
            from apuestas.ml.tabpfn_stacker import TabPFNStacker

            self._impl = TabPFNStacker()
            logger.info("tabpfn_v25.fallback_v1", version=ver)
        except Exception as exc:
            logger.warning("tabpfn_v25.no_impl", error=str(exc)[:80])
            self._impl = None

    def fit(self, X, y):
        if self._impl is None:
            self._init_impl()
        if self._impl is None:
            return self
        # v2.5 acepta hasta 50k; truncar si excede
        n = len(y)
        if n > self.max_samples:
            idx = list(range(self.max_samples))
            X = X[idx] if hasattr(X, "__getitem__") else X
            y = y[idx] if hasattr(y, "__getitem__") else y
        self._impl.fit(X, y)
        return self

    def predict_proba(self, X):
        if self._impl is None:
            import numpy as np

            return np.full((len(X), 2), 0.5)
        return self._impl.predict_proba(X)

    def predict(self, X):
        if self._impl is None:
            import numpy as np

            return np.zeros(len(X))
        return self._impl.predict(X)


__all__ = ["TabPFNv25Stacker", "check_tabpfn_version"]
