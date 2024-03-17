from statistics import mean

import monai
from datasets import Dataset  # type: ignore
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SamModel, SamProcessor, SamImageProcessor  # type: ignore

from src.sam.SAMDataset import SAMDataset  # type: ignore

# Heavily inspired by: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
def train_bactsam_model(num_epochs:int = 1, batch_size:int = 1):
  # Load the SAM model
  processor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore
  processor = SamProcessor(SamImageProcessor(
    do_resize=False,
    do_rescale=False,
    # do_normalize=False,
    # do_pad=False,
    do_convert_rgb=False,
  ))
  model = SamModel.from_pretrained("facebook/sam-vit-base") # type: ignore
  # model = SamModel(SamConfig(**model_config))
  for name, param in model.named_parameters(): # type: ignore
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"): # type: ignore
      param.requires_grad_(False) # type: ignore

  # Load our custom dataset
  raw_dataset = Dataset.load_from_disk("./bact_dataset/train") # type: ignore
  train_dataset = SAMDataset(dataset=raw_dataset, processor=processor)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # type: ignore

  # Set various optimizer configurations
  # Note: Hyperparameter tuning could improve performance here
  optimizer = AdamW(model.mask_decoder.parameters(), lr=1e-8, weight_decay=1e-4) # type: ignore
  seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean') # type: ignore

  # Train the model
  device = "cuda" if cuda.is_available() else "cpu"
  model.to(device) # type: ignore
  model.train() # type: ignore
  for epoch in range(num_epochs):
      epoch_losses = []
      for batch in tqdm(train_dataloader): # type: ignore
        # forward pass
        outputs = model( # type: ignore
          pixel_values=batch["pixel_values"].to(device),
          input_boxes=batch["input_boxes"].to(device),
          multimask_output=False
        )

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1) # type: ignore
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1)) # type: ignore

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward() # type: ignore

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item()) # type: ignore

      print(f'EPOCH: {epoch}')
      print(f'Mean loss: {mean(epoch_losses)}')

  model.save_pretrained("./model") # type: ignore

if __name__ == "__main__":
  train_bactsam_model()