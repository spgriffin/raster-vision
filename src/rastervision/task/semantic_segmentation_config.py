from copy import deepcopy
from typing import (List, Dict, Tuple, Union)

import rastervision as rv
from rastervision.task import SemanticSegmentation
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.task import (TaskConfig, TaskConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg
from rastervision.protos.class_item_pb2 import ClassItem as ClassItemMsg


class SemanticSegmentationConfig(TaskConfig):
    class ChipOptions:
        def __init__(self,
                     target_classes=None,
                     debug_chip_probability=0.25,
                     negative_survival_probability=1.0,
                     number_of_chips=1000,
                     ioa_threshold=2048):
            self.target_classes = target_classes
            self.debug_chip_probability = debug_chip_probability
            self.negative_survival_probability = negative_survival_probability
            self.number_of_chips = number_of_chips
            self.ioa_threshold = ioa_threshold

    def __init__(self,
                 class_map,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 debug=True,
                 chip_size=300,
                 chip_options=None):
        super().__init__(rv.SEMANTIC_SEGMENTATION, predict_batch_size,
                         predict_package_uri, debug)
        self.class_map = class_map
        self.chip_size = chip_size
        self.chip_options = chip_options

    def create_task(self, backend):
        return SemanticSegmentation(self, backend)

    def to_proto(self):
        msg = super().to_proto()
        chip_options = TaskConfigMsg.SemanticSegmentationConfig.ChipOptions(
            debug_chip_probability=self.chip_options.debug_chip_probability,
            negative_survival_probability=self.chip_options.
            negative_survival_probability,
            number_of_chips=self.chip_options.number_of_chips,
            ioa_threshold=self.chip_options.ioa_threshold,
            target_classes=self.chip_options.target_classes)

        conf = TaskConfigMsg.SemanticSegmentationConfig(
            chip_size=self.chip_size,
            class_items=classes_to_class_items(self.class_map),
            chip_options=chip_options)
        msg.MergeFrom(TaskConfigMsg(semantic_segmentation_config=conf))

        return msg

    def preprocess_command(self, command_type, experiment_config,
                           context=None):
        return (self, rv.core.CommandIODefinition())


class SemanticSegmentationConfigBuilder(TaskConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'predict_batch_size': prev.predict_batch_size,
                'predict_package_uri': prev.predict_package_uri,
                'debug': prev.debug,
                'class_map': prev.class_map,
                'chip_size': prev.chip_size,
                'chip_options': prev.chip_options
            }
        super().__init__(SemanticSegmentationConfig, config)

    def from_proto(self, msg):
        conf = msg.semantic_segmentation_config
        b = SemanticSegmentationConfigBuilder()

        return b.with_classes(list(conf.class_items)) \
                .with_predict_batch_size(msg.predict_batch_size) \
                .with_predict_package_uri(msg.predict_package_uri) \
                .with_debug(msg.debug) \
                .with_chip_size(conf.chip_size) \
                .with_chip_options(
                    target_classes=conf.chip_options.target_classes,
                    debug_chip_probability=conf.chip_options.debug_chip_probability,
                    negative_survival_probability=conf.chip_options.negative_survival_probability,
                    number_of_chips=conf.chip_options.number_of_chips,
                    ioa_threshold=conf.chip_options.ioa_threshold)

    def with_classes(
            self, classes: Union[ClassMap, List[str], List[ClassItemMsg], List[
                ClassItem], Dict[str, int], Dict[str, Tuple[int, str]]]):
        """Set the classes for this task.

            Args:
                classes: Either a list of class names, a dict which
                         maps class names to class ids, or a dict
                         which maps class names to a tuple of (class_id, color),
                         where color is a PIL color string.
        """
        b = deepcopy(self)
        b.config['class_map'] = construct_class_map(classes)
        return b

    def with_chip_size(self, chip_size):
        """Set the chip_size for this task.

            Args:
                chip_size: Integer value chip size
        """
        b = deepcopy(self)
        b.config['chip_size'] = chip_size
        return b

    def with_chip_options(self,
                          target_classes=None,
                          debug_chip_probability=0.25,
                          negative_survival_probability=1.0,
                          number_of_chips=1000,
                          ioa_threshold=2048):
        """Sets semantic segmentation configurations for the Chip command

           Args:
            target_classes:
            debug_chip_probability:
            negative_survival_probability:
            number_of_chips:
            ioa_threshold:
        """
        b = deepcopy(self)

        b.config['chip_options'] = SemanticSegmentationConfig.ChipOptions(
            target_classes=target_classes,
            debug_chip_probability=debug_chip_probability,
            negative_survival_probability=negative_survival_probability,
            number_of_chips=number_of_chips,
            ioa_threshold=ioa_threshold)
        return b
