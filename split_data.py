import os
import shutil
import random

# Paths
SOURCE_DIR = "Thyroid Data"
DEST_DIR = "data"

# Split ratio
TRAIN_SPLIT = 0.8

classes = ["benign", "malignant"]

for cls in classes:
    os.makedirs(os.path.join(DEST_DIR, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "test", cls), exist_ok=True)

    images = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)

    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Copy train images
    for img in train_images:
        src = os.path.join(SOURCE_DIR, cls, img)
        dst = os.path.join(DEST_DIR, "train", cls, img)
        shutil.copyfile(src, dst)

    # Copy test images
    for img in test_images:
        src = os.path.join(SOURCE_DIR, cls, img)
        dst = os.path.join(DEST_DIR, "test", cls, img)
        shutil.copyfile(src, dst)

print("✅ Data split completed!")