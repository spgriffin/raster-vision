import copy
import json

import numpy as np

from rastervision.core.box import Box
from rastervision.data import ObjectDetectionLabels
from rastervision.utils.files import file_to_str


def boxes_to_geojson(boxes, class_ids, crs_transformer, class_map,
                     scores=None):
    """Convert boxes and associated data into a GeoJSON dict.

    Args:
        boxes: list of Box in pixel row/col format.
        class_ids: list of int (one for each box)
        crs_transformer: CRSTransformer used to convert pixel coords to map
            coords in the GeoJSON
        class_map: ClassMap used to infer class_name from class_id
        scores: optional list of floats (one for each box)


    Returns:
        dict in GeoJSON format
    """
    features = []
    for box_ind, box in enumerate(boxes):
        polygon = box.geojson_coordinates()
        polygon = [list(crs_transformer.pixel_to_map(p)) for p in polygon]

        class_id = int(class_ids[box_ind])
        class_name = class_map.get_by_id(class_id).name
        score = 0.0
        if scores is not None:
            score = scores[box_ind]

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            },
            'properties': {
                'class_id': class_id,
                'class_name': class_name,
                'score': score
            }
        }
        features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}


def add_classes_to_geojson(geojson, class_map):
    """Add missing class_names and class_ids from label GeoJSON."""
    geojson = copy.deepcopy(geojson)
    features = geojson['features']

    for feature in features:
        properties = feature.get('properties', {})
        if 'class_id' not in properties:
            if 'class_name' in properties:
                properties['class_id'] = \
                    class_map.get_by_name(properties['class_name']).id
            elif 'label' in properties:
                # label is considered a synonym of class_name for now in order
                # to interface with Raster Foundry.
                properties['class_id'] = \
                    class_map.get_by_name(properties['label']).id
                properties['class_name'] = properties['label']
            else:
                # if no class_id, class_name, or label, then just assume
                # everything corresponds to class_id = 1.
                class_id = 1
                class_name = class_map.get_by_id(class_id).name
                properties['class_id'] = class_id
                properties['class_name'] = class_name

        feature['properties'] = properties

    return geojson


def load_label_store_json(uri):
    """Load JSON for LabelStore.

    Returns JSON for uri
    """
    return json.loads(file_to_str(uri))


def geojson_to_object_detection_labels(geojson_dict,
                                       crs_transformer,
                                       extent=None):
    """Convert GeoJSON to ObjectDetectionLabels object.

    If extent is provided, filter out the boxes that lie "more than a little
    bit" outside the extent.

    Args:
        geojson_dict: dict in GeoJSON format
        crs_transformer: used to convert map coords in geojson to pixel coords
            in labels object
        extent: Box in pixel coords

    Returns:
        ObjectDetectionLabels
    """
    features = geojson_dict['features']
    boxes = []
    class_ids = []
    scores = []

    def polygon_to_label(polygon, crs_transformer):
        polygon = [crs_transformer.map_to_pixel(p) for p in polygon]
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        boxes.append(Box(ymin, xmin, ymax, xmax))

        properties = feature['properties']
        class_ids.append(properties['class_id'])
        scores.append(properties.get('score', 1.0))

    for feature in features:
        geom_type = feature['geometry']['type']
        coordinates = feature['geometry']['coordinates']
        if geom_type == 'MultiPolygon':
            for polygon in coordinates:
                polygon_to_label(polygon[0], crs_transformer)
        elif geom_type == 'Polygon':
            polygon_to_label(coordinates[0], crs_transformer)
        else:
            raise Exception(
                'Geometries of type {} are not supported in object detection \
                labels.'.format(geom_type))

    if len(boxes):
        boxes = np.array([box.npbox_format() for box in boxes], dtype=float)
        class_ids = np.array(class_ids)
        scores = np.array(scores)
        labels = ObjectDetectionLabels(boxes, class_ids, scores=scores)
    else:
        labels = ObjectDetectionLabels.make_empty()

    if extent is not None:
        labels = ObjectDetectionLabels.get_overlapping(
            labels, extent, ioa_thresh=0.8, clip=True)
    return labels


def color_to_triple(color: str) -> Tuple[int, int, int]:
    """Given a PIL ImageColor string, return a triple of integers
    representing the red, green, and blue values.

    Args:
         color: A PIL ImageColor string

    Returns:
         An triple of integers

    """
    # TODO: doc none behavior
    if color is None:
        r = np.random.randint(0, 0x100)
        g = np.random.randint(0, 0x100)
        b = np.random.randint(0, 0x100)
        return (r, g, b)
    else:
        return ImageColor.getrgb(color)


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    triple = color_to_triple(color)
    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def rgb_to_int_array(rgb_array):
    r = np.array(rgb_array[:, :, 0], dtype=np.uint32) * (1 << 16)
    g = np.array(rgb_array[:, :, 1], dtype=np.uint32) * (1 << 8)
    b = np.array(rgb_array[:, :, 2], dtype=np.uint32) * (1 << 0)
    return r + g + b


class SegmentationClassTransformer():
    def __init__(self, class_map):
        color_to_class = dict(
            [(item.color, item.class_id) for item in class_map.get_items()])

        # color int to class
        color_int_to_class = dict(
            zip([color_to_integer(c) for c in color_to_class.keys()],
                color_to_class.values()))

        def color_int_to_class_fn(color: int) -> int:
            return color_int_to_class.get(color, 0x00)

        self.transform_color_int_to_class = \
            np.vectorize(color_int_to_class_fn, otypes=[np.uint8])

        # class to color triple
        class_to_color_triple = dict(
            zip(color_to_class.values(),
                [color_to_triple(c) for c in color_to_class.keys()]))

        def class_to_channel_color(channel: int, class_id: int) -> int:
            """Given a channel (red, green, or blue) and a class, return the
            intensity of that channel.

            Args:
                 channel: An integer with value 0, 1, or 2
                      representing the channel.
                 class_id: The class id represented as an integer.
            Returns:
                 The intensity of the channel for the color associated
                      with the given class.
            """
            default_triple = (0x00, 0x00, 0x00)
            return class_to_color_triple.get(class_id, default_triple)[channel]

        class_to_r = np.vectorize(
            lambda c: class_to_channel_color(0, c), otypes=[np.uint8])
        class_to_g = np.vectorize(
            lambda c: class_to_channel_color(1, c), otypes=[np.uint8])
        class_to_b = np.vectorize(
            lambda c: class_to_channel_color(2, c), otypes=[np.uint8])
        self.transform_class_to_color = [class_to_r, class_to_g, class_to_b]

    def rgb_to_class(self, rgb_labels):
        color_int_labels = rgb_to_int_array(rgb_labels)
        class_labels = self.transform_color_int_to_class(color_int_labels)
        return class_labels

    def class_to_rgb(self, class_labels):
        rgb_labels = np.empty(class_labels.shape + (3, ))
        for chan in range(3):
            class_to_channel_color = self.transform_class_to_color[chan]
            rgb_labels[:, :, chan] = class_to_channel_color(class_labels)
        return rgb_labels
