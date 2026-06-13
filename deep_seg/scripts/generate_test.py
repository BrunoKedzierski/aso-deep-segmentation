from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# =========================================================
# Config
# =========================================================
SEED = 42
TEST_FRAC = 0.10

RAW_ROOT = Path("data/raw/lars_data")
PROCESSED_ROOT = Path("data/processed/lars_data")

rng = np.random.default_rng(SEED)

# =========================================================
# Helpers
# =========================================================
def list_images(split_root: Path):
    image_dir = split_root / "images"
    return sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))

def list_pngs(dir_path: Path):
    return sorted(dir_path.glob("*.png")) if dir_path.exists() else []

def ensure_test_dirs(root: Path):
    test_root = root / "test"
    (test_root / "images").mkdir(parents=True, exist_ok=True)
    (test_root / "semantic_masks").mkdir(parents=True, exist_ok=True)
    (test_root / "panoptic_masks").mkdir(parents=True, exist_ok=True)

def build_train_df(processed_root: Path):
    train_root = processed_root / "train"

    image_paths = list_images(train_root)
    sem_paths = list_pngs(train_root / "semantic_masks")
    pan_paths = list_pngs(train_root / "panoptic_masks")

    sem_lookup = {p.stem: p for p in sem_paths}
    pan_lookup = {p.stem: p for p in pan_paths}

    rows = []
    missing_sem = []
    missing_pan = []

    for img_path in image_paths:
        stem = img_path.stem
        sem_path = sem_lookup.get(stem)
        pan_path = pan_lookup.get(stem)

        if sem_path is None:
            missing_sem.append(img_path.name)
            continue

        if pan_path is None:
            missing_pan.append(img_path.name)

        rows.append({
            "stem": stem,
            "image_path": str(img_path),
            "semantic_mask_path": str(sem_path),
            "panoptic_mask_path": str(pan_path) if pan_path is not None else None,
        })

    df = pd.DataFrame(rows)

    print(f"Processed train paired samples: {len(df)}")
    print(f"Missing semantic masks: {len(missing_sem)}")
    print(f"Missing panoptic masks: {len(missing_pan)}")

    return df

# =========================================================
# Safety checks
# =========================================================
if not RAW_ROOT.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW_ROOT}")

if PROCESSED_ROOT.exists():
    raise FileExistsError(
        f"Processed dataset already exists: {PROCESSED_ROOT}\n"
        f"Delete it first if you want to rebuild it."
    )

PROCESSED_ROOT.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# Step 1: Copy raw -> processed as-is
# =========================================================
print("Copying raw dataset to processed...")
shutil.copytree(RAW_ROOT, PROCESSED_ROOT)
print(f"Copied to: {PROCESSED_ROOT}")

# =========================================================
# Step 2: Build test from processed/train
# =========================================================
df_train = build_train_df(PROCESSED_ROOT)

n_test = max(1, int(round(len(df_train) * TEST_FRAC)))
test_indices = rng.choice(df_train.index.to_numpy(), size=n_test, replace=False)

df_train = df_train.copy()
df_train["new_split"] = "train"
df_train.loc[test_indices, "new_split"] = "test"

df_test = df_train[df_train["new_split"] == "test"].reset_index(drop=True)

print(f"Moving {len(df_test)} samples from processed/train -> processed/test")

ensure_test_dirs(PROCESSED_ROOT)

# =========================================================
# Step 3: Move selected files from train to test
# =========================================================
for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Moving test samples"):
    img_src = Path(row["image_path"])
    sem_src = Path(row["semantic_mask_path"])
    pan_src = Path(row["panoptic_mask_path"]) if pd.notna(row["panoptic_mask_path"]) else None

    img_dst = PROCESSED_ROOT / "test" / "images" / img_src.name
    sem_dst = PROCESSED_ROOT / "test" / "semantic_masks" / sem_src.name

    shutil.move(str(img_src), str(img_dst))
    shutil.move(str(sem_src), str(sem_dst))

    if pan_src is not None and pan_src.exists():
        pan_dst = PROCESSED_ROOT / "test" / "panoptic_masks" / pan_src.name
        shutil.move(str(pan_src), str(pan_dst))

# =========================================================
# Step 4: Save manifests
# =========================================================
df_train.assign(final_split=df_train["new_split"]).to_csv(
    PROCESSED_ROOT / "split_manifest.csv", index=False
)

df_test.to_csv(PROCESSED_ROOT / "test_manifest.csv", index=False)

print("\nDone.")
print(f"Raw remains untouched: {RAW_ROOT}")
print(f"Processed ready at   : {PROCESSED_ROOT}")
print(f"Train kept in place  : {len(df_train[df_train['new_split'] == 'train'])}")
print(f"Test moved from train: {len(df_test)}")