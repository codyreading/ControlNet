import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from annotator.util import resize_image, HWC3


model_hed = None


def hed(img):
    img = HWC3(img)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return result

def main():
    # Your main code goes here
    DATA_PATH = Path("data/OmniObject3D/OpenXDLab___OmniObject3D-New/raw/blender_renders_24_views/")
    NAME = "teapot_001"

    image_path = DATA_PATH / "img" / NAME
    sketch_path = DATA_PATH / "sketch" / NAME
    sketch_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(image_path.iterdir()))
    image_paths = [path for path in image_paths if path.suffix == ".png"]

    for image_file in tqdm(image_paths, desc="Processing sketches"):
        sketch_file = sketch_path / image_file.name

        image = np.array(Image.open(image_file))[:, :, :3]
        sketch = hed(image)

        sketch = Image.fromarray(sketch)
        sketch.save(sketch_file)



if __name__ == "__main__":
    main()