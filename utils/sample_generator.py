"""Sample portrait synthesis and testing fixtures for OpenCV Computer Vision."""

import cv2
import numpy as np


def _draw_realistic_face(
    canvas: np.ndarray,
    center: tuple[int, int],
    scale: float = 1.0,
    smiling: bool = True,
    skin_tone: tuple[int, int, int] = (158, 186, 218),  # BGR
) -> None:
    """Draws a Haar-cascade compatible shaded human face on a BGR canvas."""
    cx, cy = center
    head_w = int(95 * scale)
    head_h = int(125 * scale)

    # Hair / crown background
    cv2.ellipse(
        canvas,
        (cx, cy - int(18 * scale)),
        (int(head_w * 1.06), int(head_h * 1.04)),
        0,
        0,
        360,
        (35, 40, 48),
        -1,
    )

    # Skin tone with gradient
    base_b, base_g, base_r = skin_tone
    for dy in range(-head_h, head_h):
        ratio = (dy + head_h) / max(1, (2 * head_h))
        half_x = int(head_w * np.sqrt(max(0.0, 1.0 - (dy / head_h) ** 2)))
        if half_x <= 0:
            continue
        shade = 1.0 - 0.18 * abs(ratio - 0.42)
        b = int(np.clip(base_b * shade, 50, 245))
        g = int(np.clip(base_g * shade, 70, 250))
        r = int(np.clip(base_r * shade, 90, 255))
        cv2.line(canvas, (cx - half_x, cy + dy), (cx + half_x, cy + dy), (b, g, r), 1)

    # Forehead highlight
    cv2.ellipse(
        canvas,
        (cx, cy - int(42 * scale)),
        (int(58 * scale), int(24 * scale)),
        0,
        0,
        360,
        (int(min(255, base_b + 25)), int(min(255, base_g + 25)), int(min(255, base_r + 25))),
        -1,
    )

    # Cheek highlights
    for side in (-1, 1):
        cv2.ellipse(
            canvas,
            (cx + side * int(42 * scale), cy + int(12 * scale)),
            (int(26 * scale), int(18 * scale)),
            0,
            0,
            360,
            (int(min(255, base_b + 20)), int(min(255, base_g + 20)), int(min(255, base_r + 20))),
            -1,
        )

    # Eye socket regions
    eye_y = cy - int(14 * scale)
    eye_dx = int(34 * scale)
    for side in (-1, 1):
        ex = cx + side * eye_dx
        # Socket shadow
        cv2.ellipse(
            canvas,
            (ex, eye_y),
            (int(22 * scale), int(13 * scale)),
            0,
            0,
            360,
            (95, 112, 135),
            -1,
        )
        # Eyebrow arch
        cv2.ellipse(
            canvas,
            (ex, eye_y - int(14 * scale)),
            (int(22 * scale), int(5 * scale)),
            0,
            180,
            360,
            (40, 46, 56),
            max(2, int(4 * scale)),
        )
        # Eye sclera
        cv2.ellipse(
            canvas,
            (ex, eye_y),
            (int(14 * scale), int(7 * scale)),
            0,
            0,
            360,
            (225, 230, 235),
            -1,
        )
        # Iris & Pupil
        cv2.circle(canvas, (ex, eye_y), max(3, int(6 * scale)), (52, 62, 78), -1)
        cv2.circle(canvas, (ex, eye_y), max(2, int(3 * scale)), (18, 20, 24), -1)
        # Specular light
        cv2.circle(
            canvas,
            (ex - max(1, int(2 * scale)), eye_y - max(1, int(2 * scale))),
            max(1, int(1.5 * scale)),
            (250, 250, 255),
            -1,
        )

    # Bright nose bridge
    cv2.ellipse(
        canvas,
        (cx, cy + int(4 * scale)),
        (int(10 * scale), int(28 * scale)),
        0,
        0,
        360,
        (int(min(255, base_b + 30)), int(min(255, base_g + 30)), int(min(255, base_r + 30))),
        -1,
    )
    # Nostrils
    cv2.ellipse(
        canvas,
        (cx, cy + int(30 * scale)),
        (int(16 * scale), int(6 * scale)),
        0,
        0,
        360,
        (110, 128, 152),
        -1,
    )

    # Mouth
    mouth_y = cy + int(56 * scale)
    if smiling:
        cv2.ellipse(
            canvas,
            (cx, mouth_y - int(4 * scale)),
            (int(30 * scale), int(14 * scale)),
            0,
            10,
            170,
            (70, 75, 115),
            -1,
        )
        cv2.ellipse(
            canvas,
            (cx, mouth_y - int(2 * scale)),
            (int(22 * scale), int(6 * scale)),
            0,
            0,
            180,
            (235, 235, 240),
            -1,
        )
        cv2.ellipse(
            canvas,
            (cx, mouth_y + int(6 * scale)),
            (int(26 * scale), int(8 * scale)),
            0,
            0,
            180,
            (95, 105, 150),
            max(2, int(3 * scale)),
        )
    else:
        cv2.line(
            canvas,
            (cx - int(24 * scale), mouth_y),
            (cx + int(24 * scale), mouth_y),
            (80, 85, 120),
            max(2, int(3 * scale)),
        )
        cv2.ellipse(
            canvas,
            (cx, mouth_y + int(6 * scale)),
            (int(18 * scale), int(6 * scale)),
            0,
            0,
            180,
            (95, 105, 145),
            max(2, int(3 * scale)),
        )

    # Hair fringe overlay
    cv2.ellipse(
        canvas,
        (cx, cy - int(92 * scale)),
        (int(85 * scale), int(38 * scale)),
        0,
        0,
        180,
        (30, 35, 42),
        -1,
    )
    cv2.ellipse(
        canvas,
        (cx - int(55 * scale), cy - int(45 * scale)),
        (int(26 * scale), int(55 * scale)),
        15,
        0,
        360,
        (32, 36, 44),
        -1,
    )
    cv2.ellipse(
        canvas,
        (cx + int(55 * scale), cy - int(45 * scale)),
        (int(26 * scale), int(55 * scale)),
        -15,
        0,
        360,
        (32, 36, 44),
        -1,
    )


def generate_sample_portrait(scene: str = "single") -> np.ndarray:
    """Generates an RGB synthetic image with detectable faces for testing."""
    if scene == "duo":
        width, height = 750, 480
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            t = y / height
            canvas[y, :] = (int(45 + 30 * t), int(35 + 25 * t), int(30 + 20 * t))
        _draw_realistic_face(canvas, center=(250, 240), scale=1.05, smiling=True, skin_tone=(155, 182, 215))
        _draw_realistic_face(canvas, center=(520, 250), scale=0.98, smiling=False, skin_tone=(135, 165, 198))
    elif scene == "crowd":
        width, height = 800, 500
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            t = y / height
            canvas[y, :] = (int(30 + 35 * t), int(38 + 30 * t), int(42 + 25 * t))
        _draw_realistic_face(canvas, center=(180, 250), scale=0.88, smiling=True)
        _draw_realistic_face(canvas, center=(400, 220), scale=1.0, smiling=True)
        _draw_realistic_face(canvas, center=(620, 260), scale=0.92, smiling=False)
    else:  # "single"
        width, height = 500, 520
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            t = y / height
            canvas[y, :] = (int(40 + 35 * t), int(32 + 28 * t), int(28 + 22 * t))
        _draw_realistic_face(canvas, center=(250, 260), scale=1.18, smiling=True)

    # Add very subtle Gaussian blur for natural camera feel
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
