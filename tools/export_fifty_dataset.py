import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.utils.random as four

# The Dataset or DatasetView containing the samples you wish to export
dataset = foz.load_zoo_dataset(
              "open-images-v7",
            #   split="validation",
              label_types=["detections"],
              classes=["Vehicle registration plate"],
              dataset_dir='/root/vol/datasets/fiftyone',
            #   download_if_necessary=False
          )
          
plate_view = (
    dataset
    .select_fields("ground_truth")
    .filter_labels("ground_truth", F("label") == "Vehicle registration plate")
    .map_labels("ground_truth", {"Vehicle registration plate": "licence"})
    # .sort_by(F("ground_truth.detections").length(), reverse=True)
)

train, test, val = four.random_split(plate_view, [6872, 426, 859])
print(len(train), len(test), len(val))

# The directory to which to write the exported dataset
export_dir = "/root/vol/datasets/open_image_v7_VOC/"

# The name of the sample field containing the label that you wish to export
# Used when exporting labeled datasets (e.g., classification or detection)
label_field = "ground_truth"  # for example

# The type of dataset to export
# Any subclass of `fiftyone.types.Dataset` is supported
dataset_type = fo.types.VOCDetectionDataset  # for example
# dataset_type = fo.types.COCODetectionDataset  # for example

# Export the dataset
train.export(
    export_dir=export_dir+"train",
    dataset_type=dataset_type,
    label_field=label_field,
)
test.export(
    export_dir=export_dir+"test",
    dataset_type=dataset_type,
    label_field=label_field,
)
val.export(
    export_dir=export_dir+"val",
    dataset_type=dataset_type,
    label_field=label_field,
)