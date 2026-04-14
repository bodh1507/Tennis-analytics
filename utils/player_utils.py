import numpy as np


def get_center_of_bbox(bbox):
    """Return center (x, y) of a bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def get_bbox_height(bbox):
    """Return height of bounding box"""
    return bbox[3] - bbox[1]


def get_bbox_width(bbox):
    """Return width of bounding box"""
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    """
    Return bottom-center of bounding box.
    Better than center for projecting player position onto court.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def measure_distance(p1, p2):
    """Euclidean distance between two (x, y) points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def measure_xy_distance(p1, p2):
    """Return (dx, dy) between two points"""
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
