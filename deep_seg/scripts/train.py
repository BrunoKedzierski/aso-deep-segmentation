import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pathlib import Path
from torchinfo import summary
import wandb
from deep_seg.scripts.water_segmentation_model import WaterSegmentationModel
from deep_seg.scripts.lars_binary_dataset import WaterBinaryDataset

EPOCHS = 10
RAW_ROOT = Path("data/raw/lars_data")
PROCESSED_ROOT = Path("data/processed/lars_data")

def main():
    wandb.login()
    batch_size = 16
    img_size = 640

    train_dataset = WaterBinaryDataset(PROCESSED_ROOT, "train", img_size)
    valid_dataset = WaterBinaryDataset(PROCESSED_ROOT, "val", img_size)
    test_dataset  = WaterBinaryDataset(PROCESSED_ROOT, "test", img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    t_max = EPOCHS * len(train_loader)

    model = WaterSegmentationModel(
        "Unet",
        "resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        out_classes=1,
        t_max=t_max,
        trainable_backbone_layers=2,
        lr=2e-4,
    )

    summary(
        model,
        input_size=(batch_size, 3, img_size, img_size),
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )

    wandb_logger = WandbLogger(
        project="water-segmentation",
        name="unet-resnet34-freeze2",
        log_model=False,
    )

    wandb_logger.experiment.config.update({
        "epochs": EPOCHS,
        "batch_size": batch_size,
        "img_size": img_size,
        "lr": 2e-4,
        "arch": "Unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "trainable_backbone_layers": 2,
    })

    checkpoint_cb = ModelCheckpoint(
        monitor="valid_dataset_iou",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{valid_dataset_iou:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    trainer.test(model, dataloaders=test_loader)
    wandb.finish()

if __name__ == "__main__":
    main()