import colorsys
import hashlib

from PySide6.QtGui import QColor


FRAME_RGB_COLORS = {
    "pelvis": (0.05, 0.85, 0.25),
    "torso": (0.00, 0.72, 0.85),
    "left_foot": (0.18, 0.42, 1.00),
    "right_foot": (0.85, 0.20, 0.95),
    "left_hand": (0.95, 0.70, 0.10),
    "right_hand": (0.95, 0.15, 0.12),
}


def _generated_rgb_for_frame(frame_name):
    digest = hashlib.sha256(str(frame_name or "frame").encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.64 + (digest[2] / 255.0) * 0.22
    value = 0.78 + (digest[3] / 255.0) * 0.16
    return colorsys.hsv_to_rgb(hue, saturation, value)


def rgb_for_frame(frame_name):
    return FRAME_RGB_COLORS.get(frame_name, _generated_rgb_for_frame(frame_name))


def qt_color_for_frame(frame_name):
    return QColor.fromRgbF(*rgb_for_frame(frame_name))


def gl_color_for_frame(frame_name):
    return rgb_for_frame(frame_name)
