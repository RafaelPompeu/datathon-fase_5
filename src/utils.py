"""Utilidades compartilhadas para o pipeline do Datathon."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "decision"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

POSITIVE_STATUSES: set[str] = {
    "Aprovado",
    "Proposta Aceita",
    "Documentação PJ",
    "Documentação CLT",
}

LOG_LEVEL = logging.INFO


def setup_logging(level: int = LOG_LEVEL) -> None:
    """Configura logging padronizado para todo o projeto."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_directories(directories: Iterable[Path]) -> None:
    """Garante que diretórios necessários existam."""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
