# %%
from dpl import no_grad, as_variable
from models import VGG16
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = "../data/zebra.jpg"

image = Image.open(image_path)
plt.imshow(image)
plt.title(f"Image: {image_path}")
plt.show()

model = VGG16(pretrained=True)

x = model.preprocess(image)
x = x[np.newaxis]
x = as_variable(x)

with no_grad():
    y = model.apply(x)

predict_id = int(np.argmax(y.data_required))
print(f"Predicted class ID: {predict_id}")

# Get and display the label
labels = VGG16.get_labels()
predicted_label = labels[predict_id]
print(f"Predicted label: {predicted_label}")
