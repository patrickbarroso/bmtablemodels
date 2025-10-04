#calcula IOU entre dois bounding box
def calculate_iou(bbox1, bbox2):
    """Calcula o IoU (Intersection over Union) entre dois bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


bbox1 = [97, 1257, 1613, 1502]
bbox2 = [103,1259,1595, 1505]

print(calculate_iou(bbox1, bbox2))