import sys

sys.path.append("../")
from pipeline import Pipeline
import os
import unittest


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline(both_box_types=True)
        self.pipeline.train_val_test_split()

    def test_allocation(self):
        for image_path, label_path in zip(
            self.pipeline.train_data, self.pipeline.labels_train
        ):
            self.assertEqual(
                os.path.splitext(os.path.basename(image_path))[0],
                os.path.splitext(os.path.basename(label_path))[0],
            )

        for image_path, label_path in zip(
            self.pipeline.val_data, self.pipeline.labels_val
        ):
            self.assertEqual(
                os.path.splitext(os.path.basename(image_path))[0],
                os.path.splitext(os.path.basename(label_path))[0],
            )

        for image_path, label_path in zip(
            self.pipeline.test_data, self.pipeline.labels_test
        ):
            self.assertEqual(
                os.path.splitext(os.path.basename(image_path))[0],
                os.path.splitext(os.path.basename(label_path))[0],
            )


if __name__ == "__main__":
    unittest.main()
