"""Schémas de réponse de l'API (contrat d'interface D11)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    category: str
    method: str
    score: float = Field(description="Score d'anomalie image (plus haut = plus suspect)")
    threshold: float = Field(description="Seuil utilisé pour la décision")
    threshold_source: Literal["default", "calibrated"]
    decision: Literal["normal", "anomaly"]
    anomaly_map: str = Field(description="Carte de chaleur superposée à l'image, PNG en base64")
    processed_image: Optional[str] = Field(
        None, description="Image après changement d'éclairage simulé (PNG base64), si demandé")
    applied_shift: Optional[str] = None
    applied_severity: Optional[int] = None
    inference_time_ms: float
    model_backend: str
    model_version: str


class CalibrateResponse(BaseModel):
    category: str
    method: str
    n_images: int
    alpha: float
    new_threshold: float
    previous_threshold: float
    scores_min: float
    scores_max: float
    calibrated_at: str
    note: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_backend: str
    model_version: str
    categories: list[str]
    methods: list[str]
    calibration_pool: dict[str, Optional[int]] = Field(
        default_factory=dict, description="Par catégorie : nombre de photos MVTec réservées à la recalibration")
    examples_source: dict[str, str] = Field(
        default_factory=dict, description="Par catégorie : 'mvtec' (vraies photos) ou 'synthetic' (pièces dessinées)")


class ErrorResponse(BaseModel):
    detail: str


class DriftPoint(BaseModel):
    shift: Optional[str] = Field(None, description="None = lumière normale")
    severity: Optional[int] = None
    score: float
    alert: bool = Field(description="True si le score dépasse la limite de la méthode")


class DriftSeries(BaseModel):
    method: str
    threshold: float
    threshold_source: Literal["default", "calibrated"]
    points: list[DriftPoint]
    error: Optional[str] = Field(None, description="Renseigné si la méthode n'est pas disponible")


class DriftResponse(BaseModel):
    category: str
    series: list[DriftSeries]
