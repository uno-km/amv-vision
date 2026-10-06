import numpy as np

def extract_patches(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Extract non-overlapping patches from an image for Vision Transformer (ViT) input.
    Input image shape: (H, W, C) or (H, W) or (B, H, W, C)
    Output shape: (N_patches, patch_size * patch_size * C) or (B, N_patches, patch_dim)
    """
    has_batch = (image.ndim == 4)
    if not has_batch:
        img_batched = image[np.newaxis, ...]
    else:
        img_batched = image

    b, h, w = img_batched.shape[:3]
    c = 1 if img_batched.ndim == 3 else img_batched.shape[3]

    if h % patch_size != 0 or w % patch_size != 0:
        # Pad image to multiple of patch_size
        pad_h = (patch_size - (h % patch_size)) % patch_size
        pad_w = (patch_size - (w % patch_size)) % patch_size
        if img_batched.ndim == 4:
            img_batched = np.pad(img_batched, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        else:
            img_batched = np.pad(img_batched, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
        h, w = img_batched.shape[1:3]

    num_patches_y = h // patch_size
    num_patches_x = w // patch_size
    num_patches = num_patches_y * num_patches_x
    patch_dim = patch_size * patch_size * c

    if c == 1 and img_batched.ndim == 3:
        img_batched = img_batched[..., np.newaxis]

    # Shape transformation: (B, num_y, patch_size, num_x, patch_size, C) -> (B, num_y, num_x, patch_size, patch_size, C)
    reshaped = img_batched.reshape(b, num_patches_y, patch_size, num_patches_x, patch_size, c)
    transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
    patches = transposed.reshape(b, num_patches, patch_dim)

    if not has_batch:
        return patches[0]
    return patches

def reconstruct_from_patches(patches: np.ndarray, image_shape: tuple, patch_size: int = 16) -> np.ndarray:
    """
    Reconstruct spatial image from flattened ViT patches.
    patches shape: (N_patches, patch_dim)
    image_shape: (H, W, C) or (H, W)
    """
    h, w = image_shape[:2]
    c = 1 if len(image_shape) == 2 else image_shape[2]
    num_patches_y = h // patch_size
    num_patches_x = w // patch_size

    # (num_y, num_x, patch_size, patch_size, c)
    reshaped = patches.reshape(num_patches_y, num_patches_x, patch_size, patch_size, c)
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    img = transposed.reshape(h, w, c)
    if len(image_shape) == 2:
        return img[:, :, 0]
    return img
