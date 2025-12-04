import os 
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
from PIL import Image

# ---------------- CONFIG ---------------- #
model_path = "models/xray_mobilenetv2.h5"
valid_df = "MURA-v1.1/valid_image_paths.csv"   # your CSV
image_list = []
output_dir = "heatmaps"

target_size = (224, 224)
num_process = 5
heatmap_threshold_percentile = 85
# ----------------------------------------- #

os.makedirs(output_dir, exist_ok=True)


# ---------------- MODEL LOADING ---------------- #
def load_model(model_path=model_path):
    print("Loading model from:", model_path)
    model = keras.models.load_model(model_path)
    model.summary()
    return model


# ---------------- IMAGE LOADING ---------------- #
def load_image(path, target_size=target_size):
    img = Image.open(path).convert("RGB")
    orig = img.copy()
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return orig, np.expand_dims(arr, axis=0)


# ---------------- GRAD-CAM ---------------- #
def make_gradcam_heatmap(model, img_tensor, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break

    if last_conv_layer_name is None:
        raise ValueError("No conv layer found. Set last_conv_layer manually.")

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]     # binary classification output

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    heatmap = cv.resize(heatmap, (target_size[1], target_size[0]))
    return heatmap


# ---------------- SAVE HEATMAP ---------------- #
def save_heatmap(orig_pil, heatmap, out_basename):
    orig_resized = orig_pil.resize(target_size)
    orig_arr = np.array(orig_resized)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv.applyColorMap(heatmap_uint8, cv.COLORMAP_JET)
    heatmap_color = cv.cvtColor(heatmap_color, cv.COLOR_BGR2RGB)

    overlay = cv.addWeighted(orig_arr.astype(np.uint8), 0.6,
                             heatmap_color.astype(np.uint8), 0.4, 0)

    heatmap_path = os.path.join(output_dir, f"{out_basename}_heatmap.png")
    overlay_path = os.path.join(output_dir, f"{out_basename}_overlay.png")

    Image.fromarray(heatmap_color).save(heatmap_path)
    Image.fromarray(overlay).save(overlay_path)

    return heatmap_path, overlay_path


# ---------------- HEATMAP EXPLANATION ---------------- #
def heatmap_explanation(heatmap, prediction_score,
                        threshold_percentile=heatmap_threshold_percentile):

    h, w = heatmap.shape
    thresh = np.percentile(heatmap, threshold_percentile)

    mask = heatmap >= thresh
    if mask.sum() == 0:
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        mask[max_pos] = True

    ys, xs = np.where(mask)
    y_center = ys.mean()
    x_center = xs.mean()

    vertical = "upper" if y_center < h / 3 else ("lower" if y_center > 2 * h / 3 else "central")
    horizontal = "left" if x_center < w / 3 else ("right" if x_center > 2 * w / 3 else "center")

    if vertical == "central" and horizontal == "center":
        loc = "central region"
    else:
        loc = f"{vertical}-{horizontal} area"

    explanation = (f"The model predicted a score of {prediction_score:.2f}. "
                   f"Activation is focused in the {loc}.")
    return explanation


# ---------------- PROCESS IMAGE ---------------- #
def process_image(path, model, last_conv_layer_name=None):
    base = os.path.splitext(os.path.basename(path))[0]

    orig_pil, tensor = load_image(path)
    preds = model.predict(tensor)

    pred_score = float(preds[0][0])
    label = "Abnormal" if pred_score > 0.5 else "Normal"

    heatmap = make_gradcam_heatmap(model, tensor, last_conv_layer_name)
    heatmap_path, overlay_path = save_heatmap(orig_pil, heatmap, base)
    explanation = heatmap_explanation(heatmap, pred_score)

    save_csv = os.path.join(output_dir, "explanations.csv")

    if not os.path.exists(save_csv):
        with open(save_csv, "w") as f:
            f.write("basename,label,pred_score,heatmap_path,overlay_path,explanation\n")

    with open(save_csv, "a", encoding="utf-8") as f:
        safe_expl = explanation.replace('"', '""')
        f.write(f"{base},{label},{pred_score:.4f},{heatmap_path},{overlay_path},\"{safe_expl}\"\n")

    return {
        "image": path,
        "label": label,
        "score": pred_score,
        "heatmap": heatmap_path,
        "overlay": overlay_path,
        "explanation": explanation
    }


# ---------------- READ CSV + COLLECT MURA IMAGES ---------------- #
def load_mura_paths(valid_df):
    image_paths = []

    try:
        df = pd.read_csv(valid_df)
    except:
        df = pd.read_csv(valid_df, header=None)

    if "image_path" in df.columns:
        raw_paths = df["image_path"].astype(str).tolist()
    else:
        raw_paths = df.iloc[:, 0].astype(str).tolist()

    allowed = (".png", ".jpg", ".jpeg")

    for p in raw_paths:
        p = p.strip().strip(",").strip()

        # if row points to a study folder
        if os.path.isdir(p):
            imgs = []
            for ext in allowed:
                imgs.extend(sorted(glob.glob(os.path.join(p, f"*{ext}"))))

            if len(imgs) == 0:
                print("[WARN] No images in folder:", p)
            else:
                image_paths.extend(imgs)

        # if row is a file path
        elif os.path.isfile(p) and p.lower().endswith(allowed):
            image_paths.append(p)

        else:
            # try resolving relative to CSV folder
            candidate = os.path.join(os.path.dirname(valid_df), p)
            if os.path.isdir(candidate):
                for ext in allowed:
                    image_paths.extend(sorted(glob.glob(os.path.join(candidate, f"*{ext}"))))
            elif os.path.isfile(candidate):
                image_paths.append(candidate)
            else:
                print("[WARN] Cannot resolve path:", p)

    return list(dict.fromkeys(image_paths))  # remove duplicates


# ---------------- MAIN ---------------- #
def main():
    model = load_model(model_path)

    # auto-detect last conv layer
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            last_conv = layer.name
            break

    print("Using last conv layer:", last_conv)

    # load dataset image paths
    image_paths = load_mura_paths(valid_df)

    if len(image_paths) == 0:
        print("No valid image paths found. Exiting.")
        sys.exit(0)

    img_paths = image_paths[:num_process]

    print("\nProcessing", len(img_paths), "images...\n")

    for img in img_paths:
        print("->", img)
        try:
            process_image(img, model, last_conv)
        except Exception as e:
            print("[ERROR]", e)

    print("\nDone! Heatmaps saved in:", output_dir)


if __name__ == "__main__":
    main()
