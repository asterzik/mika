import numpy as np
import torch
import hyde


def hsi_tensor_from_images(images):
    """
    Transforms grayscale images to tensor representation that hyde can work with."""

    if not images:
        raise ValueError("The input images list is empty.")

    # Check that all images have the same shape and type
    first_shape = images[0].shape
    for img in images:
        if img.shape != first_shape:
            raise ValueError(
                f"All images must have the same shape. Found shape: {img.shape}."
            )

    # Ensure the images are of a proper numeric type
    if not np.issubdtype(images[0].dtype, np.number):
        raise TypeError("All images must be numeric types.")

    stacked_images = np.stack(images, axis=0)  # shape (wl, height, width)
    # normalized_stacked_images = stacked_images.astype(np.float32) / 255.0
    # TODO: My datatype is int, but somehow if I transform it to normalized like the comments of the package want I just get bullshit
    tensor = torch.tensor(stacked_images, dtype=torch.float32)
    # Move the tensor to the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = tensor.to(device)

    # Now the tensor is ready to be passed into HyRes
    return tensor


def smooth_hsi_tensor(input_tensor, denoising_method):
    """
    Applies the HyRes smoothing to the input tensor.

    Parameters:
    - input_tensor (torch.Tensor): Input tensor (shape: (Wavelengths, Height, Width)).

    Returns:
    - torch.Tensor: Smoothed output tensor.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if denoising_method == "HyRes":
        denoising = hyde.HyRes()
    elif denoising_method == "HyMinor":
        denoising = hyde.HyMiNoR()
    elif denoising_method == "FastHyDe":
        denoising = hyde.FastHyDe()
    elif denoising_method == "L1HyMixDe":
        denoising = hyde.L1HyMixDe()
    elif denoising_method == "WSRRR":
        denoising = hyde.WSRRR()
    elif denoising_method == "OTVCA":
        denoising = hyde.OTVCA()
    elif denoising_method == "FORPDN":
        denoising = hyde.FORPDN_SURE()
    result = denoising(input_tensor)

    return result


def hsi_tensor_to_images(tensor):
    """
    Converts a tensor representation back to individual grayscale images.

    Parameters:
    - tensor (torch.Tensor): Tensor representation (shape: (Wavelengths, Height, Width)).

    Returns:
    - list of np.ndarray: List of grayscale images (shape: (Height, Width)).
    """
    array = tensor.detach().cpu().numpy()  # Correctly detach before moving to CPU

    # Save each channel (wavelength) as a separate image
    num_channels = array.shape[0]  # C is the number of channels (wavelengths)
    # Reformat to integers
    # array = array * 255.0
    # array = array.astype(np.uint8)

    images = []
    for i in range(num_channels):
        # Get the i-th channel (spectral band)
        img = array[i]  # Shape: (H, W)

        images.append(img)  # Append the image to the list

    return images


def denoise_hsi_images(images, denoising_method):
    """
    Denoises hyperspectral images using the HyRes model.

    Parameters:
    - images (list of np.ndarray): List of grayscale images (shape: (H, W)).

    Returns:
    - list of np.ndarray: Denoised grayscale images (shape: (H, W)).
    """
    if not images:
        raise ValueError("The input images list is empty.")

    tensor = hsi_tensor_from_images(images)
    tensor = smooth_hsi_tensor(tensor, denoising_method)
    return hsi_tensor_to_images(tensor)
