"""FT-Transformer para NBA (feature-rich, n>2000) — Sprint 11 Fase H.

Paper: Gorishniy 2023, "Revisiting Deep Learning Models for Tabular Data"
(NeurIPS 2021 + 2023 benchmarks). FT-Transformer supera GBDT solo cuando
n>2000 Y d>50. NBA cumple ambos (2500 games/season, 80+ features).

Arquitectura:
1. Feature tokenizer: cada feature → embedding de dim=32
2. Multi-head self-attention (4 heads, 3 layers)
3. Classification head sobre token [CLS]

Requiere PyTorch. Si no está disponible o falla, `fit` retorna None y
el ensemble cae al stacker GBDT estándar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apuestas.obs.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FTTransformerClassifier:
    """Wrapper sklearn-compat de FT-Transformer.

    Mantiene implementación modesta (sin rtdl ni librerías extra). Si
    torch no está disponible, fit() loguea warning y no entrena; el caller
    debe chequear `is_fitted`.
    """

    n_features: int
    d_token: int = 32
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    n_epochs: int = 50
    learning_rate: float = 5e-4
    batch_size: int = 128
    device: str = "cpu"
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _feature_mean: np.ndarray | None = field(default=None, init=False)
    _feature_std: np.ndarray | None = field(default=None, init=False)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _build_module(self):  # type: ignore[no-untyped-def]
        import torch
        from torch import nn

        self._torch = torch

        class _FeatureTokenizer(nn.Module):
            def __init__(self, n_feat: int, d: int):
                super().__init__()
                self.weights = nn.Parameter(torch.randn(n_feat, d) * 0.02)
                self.bias = nn.Parameter(torch.zeros(n_feat, d))
                self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)

            def forward(self, x):
                tokens = x.unsqueeze(-1) * self.weights + self.bias
                cls = self.cls.expand(x.size(0), -1, -1)
                return torch.cat([cls, tokens], dim=1)

        class _FTTransformer(nn.Module):
            def __init__(
                self,
                n_feat: int,
                d: int,
                n_heads: int,
                n_layers: int,
                dropout: float,
            ):
                super().__init__()
                self.tokenizer = _FeatureTokenizer(n_feat, d)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=n_heads,
                    dim_feedforward=d * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.ReLU(),
                    nn.Linear(d, 1),
                )

            def forward(self, x):
                tokens = self.tokenizer(x)
                h = self.encoder(tokens)
                cls = h[:, 0, :]
                return self.head(cls).squeeze(-1)

        return _FTTransformer(
            n_feat=self.n_features,
            d=self.d_token,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FTTransformerClassifier:
        try:
            import torch
        except ImportError:
            logger.warning("ft_transformer.torch_unavailable")
            return self

        self._feature_mean = X.mean(axis=0)
        self._feature_std = X.std(axis=0) + 1e-6
        X_norm = (X - self._feature_mean) / self._feature_std

        device = torch.device(self.device)
        model = self._build_module().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        n = X_t.size(0)
        model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                optim.zero_grad()
                logits = model(X_t[idx])
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * len(idx)
            if epoch % 10 == 0:
                logger.info("ft_transformer.epoch", epoch=epoch, loss=epoch_loss / n)
        model.eval()
        self._model = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Fallback: 50/50 si no se pudo entrenar
            return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])

        import torch

        X_norm = (X - self._feature_mean) / self._feature_std
        X_t = torch.tensor(X_norm, dtype=torch.float32, device=self._torch.device(self.device))
        with torch.no_grad():
            logits = self._model(X_t).cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


__all__ = ["FTTransformerClassifier"]
