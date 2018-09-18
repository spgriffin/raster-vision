import unittest

import numpy as np

from rastervision.data.label_source.utils import (
    SegmentationClassTransformer, color_to_triple)


class TestSegmentationClassTransformer(unittest.TestCase):
    def setUp(self):
        color_to_class_id = {'red': 1, 'green': 2, 'blue': 3}
        self.transformer = SegmentationClassTransformer(color_to_class_id)

        self.rgb_image = np.zeros((1, 3, 3))
        self.rgb_image[0, 0, :] = color_to_triple('red')
        self.rgb_image[0, 1, :] = color_to_triple('green')
        self.rgb_image[0, 2, :] = color_to_triple('blue')

        self.class_image = np.array([[1, 2, 3]])

    def test_rgb_to_class(self):
        class_image = self.transformer.rgb_to_class(self.rgb_image)
        expected_class_image = self.class_image
        np.testing.assert_array_equal(class_image, expected_class_image)

    def test_class_to_rgb(self):
        rgb_image = self.transformer.class_to_rgb(self.class_image)
        expected_rgb_image = self.rgb_image
        np.testing.assert_array_equal(rgb_image, expected_rgb_image)


if __name__ == '__main__':
    unittest.main()
