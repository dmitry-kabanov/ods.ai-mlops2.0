python src/data/make_dataset.py \
    data/raw/caltech_birds2010 \
    data/interim/train/marker \
    data/interim/val/marker \
    data/processed/train/marker \
    data/processed/val/marker \
    --batch_size=64

python src/models/train_model.py \
    data/interim/train/marker \
    data/interim/val/marker \
    data/processed/train/marker \
    data/processed/val/marker \
    models/birds_mobilenetv2.h5 \
    --batch_size=64