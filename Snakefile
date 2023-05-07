BATCH_SIZE = 64


rule make_dataset:
    input:
        "data/raw/caltech_birds2010"
    output:
        "data/interim/train/marker",
        "data/interim/val/marker",
        "data/processed/train/marker",
        "data/processed/val/marker"
    shell:
        "python src/data/make_dataset.py {input} {output} --batch_size={BATCH_SIZE}"


rule train_model:
    input:
        "data/interim/train/marker",
        "data/interim/val/marker",
        "data/processed/train/marker",
        "data/processed/val/marker",
    output:
        "models/birds_mobilenetv2.h5"
    shell:
        "python src/models/train_model.py {input} {output} --batch_size={BATCH_SIZE}"
