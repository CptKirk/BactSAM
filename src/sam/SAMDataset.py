from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as Dataset # type: ignore
from transformers import SamProcessor # type: ignore
import numpy as np

class SAMDataset(TorchDataset): # type: ignore
  def __init__(self, dataset: Dataset, processor: SamProcessor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self) -> int:
    return len(self.dataset)

  def __getitem__(self, idx: int): # type: ignore
    item = self.dataset[idx] # type: ignore
    image = item["image"] # type: ignore
    ground_truth_mask = np.array(item["label"]) # type: ignore

    # get bounding box prompt
    prompt = [0, 0, 255, 255]

    np_image = np.array(image)

    # prepare image and prompt for the model
    inputs = self.processor(np_image, input_boxes=[[prompt]], return_tensors="pt") # type: ignore

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()} # type: ignore

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs # type: ignore