# Mostly stolen from https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation#note-on-custom-data

from typing import List
from datasets import Dataset, DatasetDict, Image, Array2D, Array3D # type: ignore
import os
import tifffile
import numpy as np

def create_bactsam_dataset():
  image_paths_train = [os.getcwd() + "/dataset/images/" + f for f in os.listdir(os.getcwd() + "/dataset/images") if f.endswith(".tif")]
  label_paths_train = [os.getcwd() + "/dataset/masks/" + f for f in os.listdir(os.getcwd() + "/dataset/masks") if f.endswith(".tif")]

  image_paths_validation = [os.getcwd() + "/dataset/validate/images/" + f for f in os.listdir(os.getcwd() + "/dataset/validate/images") if f.endswith(".tif")]
  label_paths_validation = [os.getcwd() + "/dataset/validate/masks/" + f for f in os.listdir(os.getcwd() + "/dataset/validate/masks") if f.endswith(".tif")]

  def create_dataset(image_paths: List[str], label_paths: List[str]) -> Dataset:
      images = []
      for img in sorted(image_paths):
        tif = tifffile.TiffFile(img)
        image = tif.asarray()
        images.append(image)
        image1 = np.array([np.rot90(image[i]) for i in range(3)])
        image2 = np.array([np.rot90(image1[i]) for i in range(3)])
        image3 = np.array([np.rot90(image2[i]) for i in range(3)])
        images.append(image1)
        images.append(image2)
        images.append(image3)

      labels = []
      for lbl in sorted(label_paths):
        tif = tifffile.TiffFile(lbl)
        image = tif.asarray()
        labels.append(image)
        image1 = np.rot90(image)
        image2 = np.rot90(image1)
        image3 = np.rot90(image2)
        labels.append(image1)
        labels.append(image2)
        labels.append(image3)

      images = np.array(images)
      labels = np.array(labels)
      images = images.astype("int16")
      labels = labels.astype("int16")

      dataset: Dataset = Dataset.from_dict({ # type: ignore
        "image": images,
        "label": labels
      })
      dataset = dataset.cast_column("image", Array3D(shape=(3,256,256), dtype="int16")) # type: ignore
      dataset = dataset.cast_column("label", Array2D(shape=(256,256), dtype="int16")) # type: ignore

      return dataset

  train_dataset = create_dataset(image_paths_train, label_paths_train)
  validation_dataset = create_dataset(image_paths_validation, label_paths_validation)

  dataset = DatasetDict({
      "train": train_dataset,
      "validation": validation_dataset,
    }
  )

  dataset.save_to_disk("./bact_dataset") # type: ignore


if __name__ == "__main__":
  create_bactsam_dataset()