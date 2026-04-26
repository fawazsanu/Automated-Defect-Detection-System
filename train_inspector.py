"""
Automated Defect Detection — Model Training
============================================
Dataset : NEU Metal Surface Defects Database (NEU-DET)
          https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
Model   : MobileNetV2 (transfer learning) + custom classification head
Task    : 6-class surface defect classification from bounding-box crops
Classes : crazing, inclusion, patches, pitted_surface, rolled-in-scale, scratches

Approach
--------
Rather than training a full object detector (which requires significantly
more compute), we:
  1. Parse VOC-format XML annotations to extract bounding box coordinates
  2. Crop each annotated defect region from the source image
  3. Train a MobileNetV2 classifier on the cropped defect patches
  4. At inference time (scan_part.py), a sliding window locates the defect
     and the classifier identifies its type

This produces a system that both localises and classifies defects.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────
IMAGE_SIZE   = 128   # Reduced from 224 to fit in RAM; MobileNetV2 supports 96+ px
BATCH_SIZE   = 32
EPOCHS       = 30    # EarlyStopping will halt before this if needed
LEARNING_RATE = 1e-4

TRAIN_IMG_DIR  = 'NEU-DET/train/images'
TRAIN_ANN_DIR  = 'NEU-DET/train/annotations'
VAL_IMG_DIR    = 'NEU-DET/validation/images'
VAL_ANN_DIR    = 'NEU-DET/validation/annotations'

CLASSES = ['crazing', 'inclusion', 'patches',
           'pitted_surface', 'rolled-in_scale', 'scratches']
NUM_CLASSES = len(CLASSES)

# ── 1. Parse annotations and crop defect patches ─────────────────────

def parse_annotations(img_dir, ann_dir):
    """
    Parse VOC XML annotations and return cropped defect images with labels.
    Each bounding box is extracted as a separate training sample.
    """
    crops, labels = [], []

    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.xml')])
    print(f"  Parsing {len(ann_files)} annotation files from {ann_dir}...")

    for ann_file in ann_files:
        tree = ET.parse(os.path.join(ann_dir, ann_file))
        root = tree.getroot()

        filename = root.find('filename').text.strip()
        base_name = os.path.splitext(filename)[0]

        # Search flat folder first, then class subfolders
        img_path = None
        candidates = [
            os.path.join(img_dir, filename),
            os.path.join(img_dir, base_name + '.jpg'),
            os.path.join(img_dir, base_name + '.png'),
        ]
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break

        if img_path is None:
            # Search one level of subfolders (per-class folders)
            for subfolder in os.listdir(img_dir):
                subfolder_path = os.path.join(img_dir, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue
                for ext in [filename, base_name + '.jpg', base_name + '.png']:
                    candidate = os.path.join(subfolder_path, ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break

        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for obj in root.findall('object'):
            label = obj.find('name').text.strip().lower()
            if label not in CLASSES:
                continue

            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text))
            ymin = int(float(bbox.find('ymin').text))
            xmax = int(float(bbox.find('xmax').text))
            ymax = int(float(bbox.find('ymax').text))

            # Add padding around the crop for context
            h, w = img_rgb.shape[:2]
            pad = 10
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(w, xmax + pad)
            ymax = min(h, ymax + pad)

            crop = img_rgb[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            crops.append(crop_resized)
            labels.append(label)

    return np.array(crops), np.array(labels)


def augment_image(img):
    """Apply random augmentation to a single image array."""
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    # Random vertical flip
    if np.random.rand() > 0.5:
        img = np.flipud(img)
    # Random rotation (0, 90, 180, 270)
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    # Random brightness adjustment
    factor = np.random.uniform(0.8, 1.2)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img


def prepare_dataset(crops, labels, le, augment=False):
    """Normalise images and encode labels."""
    X = crops.astype('float32') / 255.0

    if augment:
        augmented_X, augmented_y = [], []
        for img, label in zip(crops, labels):
            augmented_X.append(img)
            augmented_y.append(label)
            # Add 1 augmented version of each sample
            for _ in range(1):
                augmented_X.append(augment_image(img.copy()))
                augmented_y.append(label)
        X = np.array(augmented_X, dtype='float32') / 255.0
        labels = np.array(augmented_y)

    y = le.transform(labels)
    y_cat = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    return X, y_cat


# ── 2. Build model ────────────────────────────────────────────────────

def build_model():
    """
    MobileNetV2 backbone (pretrained on ImageNet) with a custom
    classification head. The backbone is initially frozen; we fine-tune
    the top layers after initial convergence.
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Freeze backbone initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model


# ── 3. Evaluation utilities ───────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix — NEU-DET Defect Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {save_path}")


def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training history saved to {save_path}")


# ── 4. Main training pipeline ─────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "=" * 60)
    print("  NEU-DET Surface Defect Classifier — Training Pipeline")
    print("=" * 60)

    # --- Parse data ---
    print("\n[1/5] Parsing training annotations...")
    train_crops, train_labels = parse_annotations(TRAIN_IMG_DIR, TRAIN_ANN_DIR)
    print(f"  Training samples: {len(train_crops)}")

    print("\n[2/5] Parsing validation annotations...")
    val_crops, val_labels = parse_annotations(VAL_IMG_DIR, VAL_ANN_DIR)
    print(f"  Validation samples: {len(val_crops)}")

    # Class distribution
    print("\n  Class distribution (training):")
    for cls in CLASSES:
        count = np.sum(train_labels == cls)
        print(f"    {cls:<20} {count} samples")

    # --- Encode labels ---
    le = LabelEncoder()
    le.fit(CLASSES)

    print("\n[3/5] Preparing datasets (with augmentation on training set)...")
    X_train, y_train = prepare_dataset(train_crops, train_labels, le, augment=True)
    X_val, y_val     = prepare_dataset(val_crops, val_labels, le, augment=False)
    print(f"  Training set  : {X_train.shape[0]} samples after augmentation")
    print(f"  Validation set: {X_val.shape[0]} samples")

    # --- Build model ---
    print("\n[4/5] Building MobileNetV2 model...")
    model, base_model = build_model()
    print(f"  Total parameters     : {model.count_params():,}")
    print(f"  Trainable parameters : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    # --- Phase 1: Train head only ---
    print("\n  Phase 1: Training classification head (backbone frozen)...")
    callbacks_phase1 = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1),
        ModelCheckpoint('defect_inspector_best.keras',
                        monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # --- Phase 2: Fine-tune top layers of backbone ---
    print("\n  Phase 2: Fine-tuning top 30 layers of MobileNetV2...")
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1),
        ModelCheckpoint('defect_inspector_best.keras',
                        monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # --- Evaluation ---
    print("\n[5/5] Evaluating on validation set...")
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=le.classes_))

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Final Validation Loss     : {val_loss:.4f}")

    # --- Save outputs ---
    model.save('defect_inspector.keras')
    print("\n  Model saved to defect_inspector.keras")

    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("  Label encoder saved to label_encoder.pkl")

    plot_confusion_matrix(y_true, y_pred, le.classes_)
    plot_training_history(history1)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)
