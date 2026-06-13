

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from torchinfo import summary

from deep_seg.scripts.water_segmentation_model import WaterSegmentationModel
from deep_seg.scripts.lars_binary_dataset import WaterBinaryDataset
EPOCHS = 10
RAW_ROOT = Path("data/raw/lars_data")
PROCESSED_ROOT = Path("data/processed/lars_data")


def main():
    batch_size = 16
    img_size = 640
    
    train_dataset = WaterBinaryDataset(PROCESSED_ROOT, "train", img_size)
    valid_dataset = WaterBinaryDataset(PROCESSED_ROOT, "val", img_size)
    test_dataset  = WaterBinaryDataset(PROCESSED_ROOT, "test", img_size)

    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )



    T_MAX = EPOCHS * len(train_dataset)


    model = WaterSegmentationModel("Unet", "resnet34", encoder_weights="imagenet",in_channels=3, out_classes=1, t_max=T_MAX, trainable_backbone_layers =2)
  
    summary(model, input_size=(batch_size, 3, img_size, img_size), col_names = ["input_size", "output_size", "num_params", "trainable"])
  
    # print(model.model.encoder)
    # print(model.model.decoder)
    # print(model.model.segmentation_head)

    # def count_params(m):
    #     total = sum(p.numel() for p in m.parameters())
    #     trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    #     return total, trainable

    # enc_total, enc_trainable = count_params(model.model.encoder)
    # dec_total, dec_trainable = count_params(model.model.decoder)

    # print(f"Encoder: total={enc_total:,}, trainable={enc_trainable:,}")
    # print(f"Decoder: total={dec_total:,}, trainable={dec_trainable:,}")



    # trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1,
    #     num_sanity_val_steps=0)

    # trainer.fit(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=valid_loader

    # )


if __name__ == "__main__":
    main()