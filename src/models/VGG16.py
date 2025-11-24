import dpl.functions as F
import dpl.layers as L
from dpl import ndarray, Variable
from PIL.Image import Image
import numpy as np


class VGG16(L.Layer):
    WEIGHTS_PATH = (
        "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"
    )
    LABELS_PATH = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"

    def __init__(self, pretrained: bool = False):
        super().__init__()

        self.conv1_1 = L.Conv2d(64, kernel_size=3, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            import os
            import urllib.request

            # Create cache directory
            cache_dir = os.path.expanduser("~/.cache/dpl")
            os.makedirs(cache_dir, exist_ok=True)

            # Extract filename from URL
            filename = "vgg16.npz"
            cache_path = os.path.join(cache_dir, filename)

            # Download if not cached
            if not os.path.exists(cache_path):
                print(f"Downloading weights from {VGG16.WEIGHTS_PATH}...")
                urllib.request.urlretrieve(VGG16.WEIGHTS_PATH, cache_path)
                print(f"Weights cached at {cache_path}")
            else:
                print(f"Using cached weights from {cache_path}")

            self.load_weights(cache_path)

    def forward(self, *xs: Variable) -> Variable:
        (x,) = xs
        x = F.relu(self.conv1_1.apply(x))
        x = F.relu(self.conv1_2.apply(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1.apply(x))
        x = F.relu(self.conv2_2.apply(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1.apply(x))
        x = F.relu(self.conv3_2.apply(x))
        x = F.relu(self.conv3_3.apply(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1.apply(x))
        x = F.relu(self.conv4_2.apply(x))
        x = F.relu(self.conv4_3.apply(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1.apply(x))
        x = F.relu(self.conv5_2.apply(x))
        x = F.relu(self.conv5_3.apply(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc6.apply(x))
        x = F.relu(self.fc7.apply(x))
        x = self.fc8.apply(x)
        return x

    @staticmethod
    def preprocess(image: Image, size=(224, 224)) -> ndarray:
        image = image.convert("RGB")
        if size:
            image = image.resize(size)
        image_data = np.asarray(image, dtype=np.float32)
        image_data = image_data[:, :, ::-1]  # RGB -> BGR
        image_data -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
        image_data = image_data.transpose(2, 0, 1)  # HWC -> CHW
        return image_data

    def apply(self, x: Variable) -> Variable:
        result = super().__call__(x)
        assert isinstance(result, Variable)
        return result

    @staticmethod
    def get_labels() -> dict[int, str]:
        """Download and cache ImageNet labels."""
        import os
        import urllib.request
        import json

        # Create cache directory
        cache_dir = os.path.expanduser("~/.cache/dpl")
        os.makedirs(cache_dir, exist_ok=True)

        labels_cache_path = os.path.join(cache_dir, "imagenet_labels.json")

        # Download if not cached
        if not os.path.exists(labels_cache_path):
            print(f"Downloading ImageNet labels from {VGG16.LABELS_PATH}...")
            with urllib.request.urlopen(VGG16.LABELS_PATH) as response:
                content = response.read().decode("utf-8")
                # Parse the Python dict format
                labels = eval(content)
            # Save as JSON for faster loading next time
            with open(labels_cache_path, "w") as f:
                json.dump(labels, f)
            print(f"Labels cached at {labels_cache_path}")
        else:
            print(f"Using cached labels from {labels_cache_path}")
            with open(labels_cache_path, "r") as f:
                labels = json.load(f)

        # Convert string keys to int keys
        return {int(k): v for k, v in labels.items()}
