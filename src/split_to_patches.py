# import numpy as np
import tifffile

metadata = {
  'Channels': {
    'C:0': {'Name': 'C:0', 'ID': 'Channel:0:0', 'SamplesPerPixel': 1},
    'C:1': {'Name': 'C:1', 'ID': 'Channel:0:1', 'SamplesPerPixel': 1},
    'C:2': {'Name': 'C:2', 'ID': 'Channel:0:2', 'SamplesPerPixel': 1}
  },
}

def split_image_into_patches(input_file):
  patch_size=256
# Read the image
  with tifffile.TiffFile(input_file) as tif:
      image = tif.asarray()

  image = image.astype("int16")

  # Determine the shape and orientation of the image
  if image.ndim == 2:  # Grayscale image
      channels, height, width = 1, *image.shape
  elif image.shape[0] == 3 or image.shape[0] == 1:  # Channels first (e.g., 3, H, W)
      channels, height, width = image.shape
  elif image.shape[2] == 3 or image.shape[2] == 1:  # Channels last (e.g., H, W, 3)
      height, width, channels = image.shape
  else:
      raise ValueError("Image format not recognized")

  # Calculate the number of patches in each dimension
  n_patches_x = width // patch_size
  n_patches_y = height // patch_size

  patches = []

  # Loop over the image to extract patches
  for i in range(n_patches_y):
    for j in range(n_patches_x):
      # Extract the patch based on the orientation
      if channels == 1 or image.shape[2] == channels:  # Channels last or grayscale
          patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
      else:  # Channels first
          patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
      patches.append(patch)

  return patches

def split_to_patches(num_train_images:int = 6, num_validate_images:int = 1):
  for i in range(1, num_train_images + 1):
    image_path = f"./raw/time_steps/bact_sub_{str(i).zfill(3)}.tif"
    patches = split_image_into_patches(image_path)
    for idx, patch in enumerate(patches):
      tifffile.imwrite(f'./dataset/images/bact_sub_{str(i).zfill(3)}_{str(idx+1).zfill(3)}.tif', patch, ome=True, metadata=metadata)

  # for i in range(NUM_TRAIN_IMAGES):
    image_path = f"./ground_truth/masks/bact_sub_{str(i).zfill(3)}.tif"
    patches = split_image_into_patches(image_path)
    for idx, patch in enumerate(patches):
      tifffile.imwrite(f'./dataset/masks/bact_sub_{str(i).zfill(3)}_{str(idx+1).zfill(3)}.tif', patch)


  for i in range(1, num_validate_images + 1):
    image_path = f"./raw/time_steps/bact_sub_{str(num_train_images + i).zfill(3)}.tif"
    patches = split_image_into_patches(image_path)
    for idx, patch in enumerate(patches):
      tifffile.imwrite(f'./dataset/validate/images/bact_sub_{str(num_train_images + i).zfill(3)}_{str(idx+1).zfill(3)}.tif', patch, ome=True, metadata=metadata)

    image_path = f"./ground_truth/masks/bact_sub_{str(num_train_images + i).zfill(3)}.tif"
    patches = split_image_into_patches(image_path)
    for idx, patch in enumerate(patches):
      tifffile.imwrite(f'./dataset/validate/masks/bact_sub_{str(num_train_images + i).zfill(3)}_{str(idx+1).zfill(3)}.tif', patch)