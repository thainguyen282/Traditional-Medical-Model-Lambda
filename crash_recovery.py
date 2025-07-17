from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    every_n_epochs=1,
    save_top_k=3,  # Save best 3 models
)

trainer = Trainer(
    accelerator="gpu",
    strategy="ddp",
    callbacks=[checkpoint_callback],
    max_epochs=100,
    resume_from_checkpoint="checkpoints/last.ckpt"  # Auto-resumes if exists
)
trainer.fit(model)