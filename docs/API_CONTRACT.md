# Contrat d'API — LightShift-PatchCore (livrable D11)

> À valider par les trois membres. Toute modification doit être notée ici.

Documentation interactive : `http://localhost:8000/docs` (générée automatiquement par FastAPI).

## Vue d'ensemble

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/` | GET | Interface web (banc d'inspection) |
| `/health` | GET | L'API et le modèle sont-ils prêts ? |
| `/predict` | POST | Analyser une image → score, décision, carte d'anomalie |
| `/calibrate` | POST | Brique B3 : recalculer le seuil avec des images normales d'un nouvel éclairage |
| `/calibrate/{category}` | DELETE | Revenir au seuil par défaut |
| `/drift` | POST | Vue de dérive : score d'une image sous les 16 conditions de lumière, pour les deux méthodes |
| `/examples/{category}` | GET | Pièce d'exemple pour la démo : vraie photo MVTec (si le vrai modèle et le dataset sont là) ou pièce dessinée. `kind=good|bad|calibration`, `seed`, `defect` (optionnel). En-têtes `X-Example-Source`, `X-Example-Name`, `X-Example-Defect` |
| `/metrics` | GET | Compteurs (analyses, décisions, latence moyenne, **p50 et p95**, seuils actuels) |

Valeurs autorisées :
- `category` : `metal_nut`, `screw`, `transistor`
- `method` : `baseline` (PatchCore original), `lightshift` (méthode du membre 2)
- `shift` : `brightness`, `contrast`, `gamma`, `color_temperature`, `shading` — `severity` : 1, 2, 3

## POST /predict

Requête `multipart/form-data` :

| Champ | Obligatoire | Description |
|---|---|---|
| `image` | oui | Fichier PNG ou JPG (max 10 Mo) |
| `category` | oui | Catégorie de la pièce |
| `method` | non (`baseline`) | Méthode de détection |
| `shift`, `severity` | non | **Démo uniquement** : applique un changement d'éclairage avant l'analyse (mêmes transformations que `src/augmentation/lighting_shifts.py`) |

Réponse 200 :

```json
{
  "category": "screw",
  "method": "baseline",
  "score": 0.734,
  "threshold": 0.5,
  "threshold_source": "default",
  "decision": "anomaly",
  "anomaly_map": "<PNG base64 : carte de chaleur superposée à l'image>",
  "processed_image": null,
  "applied_shift": null,
  "applied_severity": null,
  "inference_time_ms": 412.3,
  "model_backend": "patchcore",
  "model_version": "patchcore-wrn50-l2l3-cs0.1-k9"
}
```

Règle de décision : `decision = "anomaly"` si `score > threshold` (inégalité stricte).
`threshold_source` vaut `calibrated` si un seuil a été recalculé via `/calibrate`.

## POST /calibrate (brique B3)

| Champ | Obligatoire | Description |
|---|---|---|
| `images` | oui (plusieurs) | Images **normales** prises sous le nouvel éclairage |
| `category` | oui | Catégorie |
| `alpha` | non (0.1) | Taux de fausses alertes visé (0 < alpha < 1) |
| `method` | non (`baseline`) | Méthode dont on recalibre le seuil |
| `shift`, `severity` | non | **Démo uniquement** : applique ce changement d'éclairage aux images avant le calcul |

Calcul (identique à B3) : scores triés s(1) ≤ … ≤ s(n), `k = ceil((n+1)(1-alpha))`, seuil = s(k).
Si `n` est trop petit (`k > n`), l'API refuse avec une erreur 422 qui indique le nombre minimum
(9 images pour alpha = 0.1, 19 pour alpha = 0.05).

Réponse 200 :

```json
{
  "category": "screw", "method": "baseline", "n_images": 20, "alpha": 0.1,
  "new_threshold": 0.62, "previous_threshold": 0.5,
  "scores_min": 0.41, "scores_max": 0.71,
  "calibrated_at": "2026-11-22T11:00:00+00:00",
  "note": "Garantie : FPR <= alpha en moyenne si ..."
}
```

Les seuils recalibrés sont enregistrés dans `app_state/thresholds.json` et survivent à un redémarrage.

⚠️ Les images envoyées à `/calibrate` ne doivent pas être réutilisées pour évaluer le modèle.

## POST /drift (vue de dérive des scores)

Requête `multipart/form-data` : `image` (PNG/JPG) et `category`.

L'image est analysée sous la lumière normale puis sous les 15 conditions de l'expérience
(5 lumières × 3 intensités), par les deux méthodes. Réponse :

```json
{
  "category": "screw",
  "series": [
    {
      "method": "baseline", "threshold": 0.5, "threshold_source": "default", "error": null,
      "points": [
        {"shift": null, "severity": null, "score": 0.41, "alert": false},
        {"shift": "brightness", "severity": 1, "score": 0.63, "alert": true}
      ]
    },
    {"method": "lightshift", "...": "..."}
  ]
}
```

Si une méthode n'est pas encore disponible (LightShift avant la livraison du membre 2),
`points` est vide et `error` explique pourquoi.

## Scores du vrai PatchCore

Par défaut (`SCORE_MODE=raw`), l'API renvoie les **scores bruts** de PatchCore, avant le
post-processor d'anomalib, et la limite par défaut est la limite brute apprise par anomalib
(`post_processor.image_threshold`). Le post-processor normalise et **coupe** les scores dans [0, 1] :
sous un fort changement de lumière, toutes les images vaudraient 1,0 et la recalibration (B3) ne
pourrait plus rien distinguer. `SCORE_MODE=normalized` rétablit les scores normalisés (limite 0,5).

## GET /metrics

```json
{
  "predict_requests": 42, "predict_errors": 0,
  "decisions": {"normal": 30, "anomaly": 12}, "calibrations": 2,
  "mean_inference_ms": 31.4,
  "latency_ms": {"p50": 28.0, "p95": 55.2, "window": 42},
  "thresholds": {"baseline:screw": {"threshold": 0.62, "source": "calibrated"}}
}
```

Les percentiles portent sur les 500 dernières analyses (`window`). Ils répondent à la question
QR5 du sujet (coût en latence).

## Codes d'erreur

| Code | Cas |
|---|---|
| 400 | Fichier vide, illisible, pas une image, trop lourd, trop d'images |
| 422 | Paramètre invalide (catégorie, méthode, shift, sévérité, alpha) ou pas assez d'images de calibration |
| 501 | Méthode pas encore implémentée (`lightshift` avant la livraison du membre 2) |
| 503 | Modèle indisponible (checkpoints absents) |

Toutes les erreurs renvoient `{"detail": "message lisible"}`.

## Ce que le membre 2 doit fournir pour brancher sa méthode

Dans `app/models/patchcore_backend.py`, méthode `_predict_lightshift(image_rgb, category)`, renvoyer :

```python
Prediction(score=float, anomaly_map=np.ndarray)  # carte (H, W) float
```

- `image_rgb` : `np.ndarray` uint8 de forme (H, W, 3), en RGB.
- `score` : plus haut = plus suspect, sur la **même échelle** que le seuil par défaut renvoyé
  par `default_threshold()` (ou bien fournir aussi ce seuil).
- Indiquer les fichiers nécessaires (checkpoints, statistiques AdaBN/CORAL, memory bank B1).
