import os
import tempfile
from typing import (Dict, List, Tuple, Union)
from urllib.parse import urlparse

import numpy as np
import rasterio

from rastervision.builders import raster_source_builder
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.core.raster_source import RasterSource
from rastervision.protos.raster_source_pb2 import (RasterSource as
                                                   RasterSourceProto)
from rastervision.utils.files import (get_local_path, make_dir, sync_dir)
from rastervision.utils.misc import SegmentationClassTransformer
from rastervision.data.label import SemanticSegmentationLabels


class SemanticSegmentationRasterStore(LabelStore):
    """A prediction label store for segmentation raster files.
    """

    def __init__(self, uri, class_map):
        self.uri = uri
        self.class_transfomer = SegmentationClassTransformer(class_map)

    def get_labels(self):
        """Get all labels.

        Returns:
             np.ndarray
        """
        source = raster_source_builder.build(self.uri)
        rgb_labels = source.get_raw_image_array()
        return self.class_transformer.transform(rgb_labels)

    def save(self, labels):
        """Save.

        Args:
           labels - Labels to be saved, the type of which will be dependant on the type
                    of task.
        """
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        local_path = get_local_path(self.uri, temp_dir)
        make_dir(local_path, use_dirname=True)

        windows = list(map(lambda pair: pair[0], labels.get_label_pairs()))
        xmax = max(map(lambda w: w.xmax, windows))
        ymax = max(map(lambda w: w.ymax, windows))

        # https://github.com/mapbox/rasterio/blob/master/docs/quickstart.rst
        # https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
        with rasterio.open(
                local_path,
                'w',
                driver='GTiff',
                height=ymax,
                width=xmax,
                count=3,
                dtype=np.uint8) as dataset:
            for (window, class_labels) in self.label_pairs:
                window = (window.ymin, window.ymax), (window.xmin, window.xmax)
                rgb_labels = self.class_transformer.class_to_rgb(class_labels)
                for chan in range(3):
                    dataset.write_band(
                        chan + 1, rgb_labels[chan, :, :], window=window)

        # sync to s3
        # TODO: why not just copy individual file?
        if urlparse(self.sink).scheme == 's3':
            local_dir = os.path.dirname(local_path)
            remote_dir = os.path.dirname(self.uri)
            sync_dir(local_dir, remote_dir, delete=False)

    def empty_labels(self):
        """Produces an empty Labels"""
        return SemanticSegmentationLabels()
