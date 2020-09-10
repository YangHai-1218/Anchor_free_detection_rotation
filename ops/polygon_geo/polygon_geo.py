from . import polygon_geo_cpu


def polygon_iou(poly1, poly2):
    '''
    poly1: det bboxes shape (N,9) the last one is score, ndarray
    poly2: gt bboxes shape (M,8), ndarray
    '''
    return polygon_geo_cpu.polygon_iou(poly1, poly2)
