from rastervision.data.label import Labels


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced labels.
    """

    def __init__(self):
        self.label_pairs = []

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        self.label_pairs.extend(other.label_pairs)

    def add_label_pair(self, window, label_array):
        self.label_pairs.append((window, label_array))

    def get_label_pairs(self):
        return self.label_pairs
