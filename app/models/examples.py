"""Pièces d'exemple synthétiques (écrou, vis, transistor), saines ou abîmées.

Elles servent à la démo et au modèle de démonstration : elles ne remplacent pas MVTec AD.
"""

import math

import cv2
import numpy as np

SIZE = 320


def _transform(points, cx, cy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[cx + x * c - y * s, cy + x * s + y * c] for x, y in points], dtype=np.int32)


def _rect(w, h, x0=0.0, y0=0.0):
    return [(x0 - w / 2, y0 - h / 2), (x0 + w / 2, y0 - h / 2), (x0 + w / 2, y0 + h / 2), (x0 - w / 2, y0 + h / 2)]


def _circle_pts(r, x0=0.0, y0=0.0, n=48, ry=None):
    ry = r if ry is None else ry
    return [(x0 + r * math.cos(2 * math.pi * k / n), y0 + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


def draw_part(category: str, defect: bool, seed: int = 1) -> np.ndarray:
    """Renvoie une image RGB uint8 (320, 320, 3)."""
    rng = np.random.default_rng(seed * 7919 + (31 if defect else 0))
    jit = lambda a: float(rng.uniform(-a / 2, a / 2))
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    fill = lambda pts, col: cv2.fillPoly(img, [pts], col, lineType=cv2.LINE_AA)

    if category == "metal_nut":
        img[:] = (110, 115, 119)
        cx, cy, a = SIZE / 2 + jit(8), SIZE / 2 + jit(8), jit(0.3)
        hexa = [(104 * math.cos(math.pi / 3 * i), 104 * math.sin(math.pi / 3 * i)) for i in range(6)]
        fill(_transform(hexa, cx, cy, a), (176, 182, 187))
        shade = np.zeros((SIZE, SIZE), np.float32)
        cv2.circle(shade, (int(cx - 30), int(cy - 30)), 70, 1.0, -1)
        shade = cv2.GaussianBlur(shade, (0, 0), 30)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(mask, [_transform(hexa, cx, cy, a)], 1)
        img = np.clip(img + (shade * mask)[..., None] * 40, 0, 255).astype(np.uint8)
        fill = lambda pts, col: cv2.fillPoly(img, [pts], col, lineType=cv2.LINE_AA)
        fill(_transform(_circle_pts(62), cx, cy, a), (164, 171, 176))
        fill(_transform(_circle_pts(40), cx, cy, a), (60, 66, 70))
        if defect:
            p = _transform([(48, -70), (84, -18)], cx, cy, a)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (43, 47, 51), 3, cv2.LINE_AA)
            fill(_transform(_circle_pts(7, -64, 36), cx, cy, a), (51, 56, 60))

    elif category == "screw":
        img[:] = (154, 158, 160)
        cx, cy, a = SIZE / 2 + jit(10), SIZE / 2 + jit(10), -0.6 + jit(0.25)
        fill(_transform(_rect(36, 190, 0, -15), cx, cy, a), (195, 199, 202))
        for y in range(-100, 76, 9):
            p = _transform([(-18, y), (18, y + 5)], cx, cy, a)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (132, 139, 144), 2, cv2.LINE_AA)
        fill(_transform([(-18, 80), (0, 108), (18, 80)], cx, cy, a), (185, 190, 193))
        fill(_transform(_circle_pts(44, 0, -114, ry=16), cx, cy, a), (213, 216, 218))
        fill(_transform(_rect(44, 4, 0, -114), cx, cy, a), (126, 133, 138))
        if defect:
            fill(_transform(_rect(12, 18, 12, 19), cx, cy, a), (74, 80, 84))
            fill(_transform(_circle_pts(5, -10, -60), cx, cy, a), (74, 80, 84))

    else:  # transistor
        img[:] = (111, 122, 107)
        cx, cy, a = SIZE / 2 + jit(8), SIZE / 2 - 30 + jit(8), jit(0.12)
        legs = [-34, 34] if defect else [-34, 0, 34]
        for x in legs:
            fill(_transform(_rect(10, 110, x, 95), cx, cy, a), (185, 190, 193))
        if defect:
            fill(_transform(_transform(_rect(10, 60, 0, 30), 0, 44, 0.9).tolist(), cx, cy, a), (185, 190, 193))
        fill(_transform(_rect(116, 104, 0, -8), cx, cy, a), (36, 40, 43))
        fill(_transform(_rect(116, 30, 0, -71), cx, cy, a), (168, 174, 178))
        fill(_transform(_circle_pts(9, 0, -71), cx, cy, a), (111, 122, 107))

    noise = rng.normal(0, 3, size=(SIZE, SIZE, 1))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
