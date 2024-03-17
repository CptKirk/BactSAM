from transformers import SamModel, SamProcessor, SamImageProcessor  # type: ignore
import numpy as np
from torch import cuda
import torch
from PIL import Image

def run_inference(images = []):
    assert len(images) == 64, "Only one image (64 patches) is supported for the moment."

    device = "cuda" if cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base") # type: ignore
    processor = SamProcessor(SamImageProcessor(
        do_resize=False,
        do_rescale=False,
        # do_normalize=False,
        # do_pad=False,
        do_convert_rgb=False,
    ))
    model = SamModel.from_pretrained("./model") # type: ignore
    model.to(device)
    model.eval()
    cnt = 1
    for image in images:
        np_image = np.array(image)
        inputs = processor(
            np_image,
            input_boxes=[[[0, 0, 255, 255]]],
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"].to(device),
                input_boxes=inputs["input_boxes"].to(device),
                multimask_output=False
            )

        mask = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )

        img_mask = Image.fromarray(mask[0].squeeze(1).squeeze().cpu().numpy())
        img_mask.save(f"./outputs/mask_{str(cnt).zfill(3)}.png")
        cnt += 1