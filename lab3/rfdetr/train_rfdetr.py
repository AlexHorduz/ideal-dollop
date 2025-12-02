from rfdetr import RFDETRBase, RFDETRNano

model = RFDETRNano(num_classes=15)

model.train(
    dataset_dir="dataset_coco",
    epochs=5,
    batch_size=1,
    grad_accum_steps=16,
    lr=1e-4,
    output_dir="runs/rfdetr"
)
