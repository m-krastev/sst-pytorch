import os

import torch
from torch import nn
from torch import optim

from utils import TreeDatasetPL, parser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class BOW(nn.Module):
    """A simple bag-of-words model"""

    def __init__(self, vocab_size, embedding_dim, vocab):
        super(BOW, self).__init__()
        self.vocab = vocab

        # this is a trainable look-up table with word embeddings
        self.embed = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=vocab.w2i["<pad>"]
        )

        # this is a trainable bias term
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        logits = embeds.sum(dim=1) + self.bias
        return logits


class BOWLightning(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, vocab, lr=0.001):
        super().__init__()

        self.save_hyperparameters(ignore=["vocab"])

        self.model = BOW(vocab_size, embedding_dim, vocab)
        self.loss = nn.CrossEntropyLoss()
        self.losses = []

    def training_step(self, batch):
        x, targets = batch
        logits = self.model(x)
        loss = self.loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, targets = batch
        logits = self.model(x)
        loss = self.loss(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).sum().float() / targets.size(0)
        self.log("val_acc", acc, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "val_acc": acc}

    def test_step(self, batch):
        x, targets = batch
        logits = self.model(x)
        loss = self.loss(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).sum().float() / targets.size(0)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_loss", loss, on_epoch=True)
        return {"loss": loss, "test_acc": acc}

    def on_test_end(self):
        print(f"Test accuracy: {self.trainer.callback_metrics['test_acc']:.2%}")
        super().on_test_end()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer


def main():
    args = parser()

    if args.debug:
        print(args)

    # Set the random seed manually for reproducibility.
    pl.seed_everything(args.seed)

    i2t = args.classes
    t2i = {p: i for i, p in enumerate(i2t)}  # noqa: F841

    # Load the dataset
    loader = TreeDatasetPL(
        batch_size=args.batch_size, num_workers=args.num_workers, lower=args.lower
    )
    loader.prepare_data()

    # Load the model
    if args.checkpoint:
        lightning_model = BOWLightning.load_from_checkpoint(
            args.checkpoint, vocab=loader.vocab
        )
    else:
        lightning_model = BOWLightning(len(loader.vocab.w2i), len(i2t), loader.vocab, args.lr)

    os.makedirs(args.model_dir, exist_ok=True)
    bestmodel_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        filename="BOW-{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        dirpath=os.path.join(args.model_dir, "checkpoints"),
    )
    trainer = pl.Trainer(
        accelerator=args.device,
        max_epochs=args.epochs,
        callbacks=[bestmodel_callback],
    )
    if args.evaluate:
        trainer.test(lightning_model, loader.test_dataloader())
    else:
        # Training code + testing
        trainer.fit(lightning_model, loader.train_dataloader(), loader.val_dataloader())

        lightning_model = BOWLightning.load_from_checkpoint(
            bestmodel_callback.best_model_path, vocab=loader.vocab
        )

        trainer.test(lightning_model, loader.test_dataloader())


if __name__ == "__main__":
    main()
