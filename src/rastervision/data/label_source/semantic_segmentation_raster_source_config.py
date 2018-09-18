from copy import deepcopy

import rastervision as rv
from rastervision.data.label_source import (LabelSourceConfig,
                                            LabelSourceConfigBuilder,
                                            SemanticSegmentationRasterSource)
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg


class SemanticSegmentationRasterSourceConfig(LabelSourceConfig):
    def __init__(self, src, raster_class_map: Dict[str, int] = {}):
        super().__init__(source_type=rv.SEMANTIC_SEGMENTATION_RASTER_SOURCE)
        self.src = src
        self.raster_class_map = raster_class_map

    def to_proto(self):
        msg = super().to_proto()
        opts = LabelSourceConfigMsg.SemanticSegmentationRasterSource(
            src=self.src.to_proto(), raster_class_map=raster_class_map)
        msg.semantic_segmentation_raster_source.CopyFrom(opts)
        return msg

    def create_source(self, tmp_dir):
        return SemanticSegmentationRasterSource(self.src,
                                                self.raster_class_map)

    def preprocess_command(self, command_type, experiment_config, context=[]):
        if context is None:
            context = []
        context = context + [self]
        io_def = rv.core.CommandIODefinition()

        b = self.to_builder()

        (new_raster_source,
         sub_io_def) = self.raster_source.preprocess_command(
             command_type, experiment_config, context)
        io_def.merge(sub_io_def)
        b = b.with_raster_source(new_raster_source)

        return (b.build(), io_def)


class SemanticSegmentationRasterSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'src': prev.src,
                'raster_class_map': prev.raster_class_map
            }

        super().__init__(SemanticSegmentationRasterSourceConfig, config)

    def from_proto(self, msg):
        b = SemanticSegmentationRasterSourceConfigBuilder()

        return b \
            .with_raster_source(msg.semantic_segmentation_raster_source.src) \
            .with_raster_class_map(
                msg.semantic_segmentation_raster_source.raster_class_map)

    # TODO handle src being string or rastersource
    def with_raster_source(self, src):
        b = deepcopy(self)
        b.config['src'] = src
        return b

    def with_raster_class_map(self, raster_class_map):
        b = deepcopy(self)
        b.config['raster_class_map'] = raster_class_map
        return b
