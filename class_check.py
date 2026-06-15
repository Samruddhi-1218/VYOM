# check_classes.py

import numpy as np

classes = np.load(
    "models/label_encoder_classes.npy",
    allow_pickle=True
)

print(f"\nClasses ({len(classes)}):\n")

for i, cls in enumerate(classes):
    print(f"{i}: {cls}")