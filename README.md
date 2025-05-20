# Project Setup

## 1. Setting Up the Environment

1. Install Miniconda or Anaconda if you don’t have it already and activate it with
   ```bash
   conda activate
   ```
3. Clone or download this repository.
4. Create the environment by running:

    ```bash
    conda env create -f environment.yml
    ```

5. Activate the environment:

    ```bash
    conda activate mika
    ```

6. Then run the application:

    ```bash
    python app.py
    ```

## 2. CUDA-Enabled OpenCV (Optional for Speed)

The environment provided uses CPU-only OpenCV by default. If you want to enable GPU acceleration with CUDA, you need to install CUDA-enabled OpenCV in the environment manually.

To check if OpenCV is using CUDA, run the following in Python:

```python
import cv2
print(cv2.getBuildInformation())
```

Look for `CUDA: YES` in the output.

---

## Notes

- The wavelengths and frames are extracted from the image names. The pattern matching is done in the function `load_images` in `image_processing/circle_detection.py`.
- If you want, you can set a `default_path` to an image folder that gets opened automatically upon startup in `app.py`, but this is not necessary.
- It's only possible to assign `n` groups with `n = number of spots * 2`. This is currently hardcoded. If more groups are necessary, this can be changed.
