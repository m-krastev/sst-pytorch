import os

from torch import nn
from torch import optim

from utils import (
    SST_PL,
    Vocabulary,
    initialize_vocabulary,
    load_embeddings,
    parser,
    print_parameters,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class DeepCBOW(nn.Module):
    """Continuous bag-of-words model"""

    def __init__(
        self,
        vocab: Vocabulary,
        vectors=None,
        output_dim=5,
        hiddens=[100, 100],
        embedding_dim=300,
        train_embeddings=False,
    ):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab.i2w)

        # this is a trainable look-up table with word embeddings
        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(
                vectors,
                freeze=not train_embeddings,
                padding_idx=self.vocab.w2i["<pad>"],
            )
        else:
            self.embed = nn.Embedding(
                self.vocab_size, embedding_dim, padding_idx=self.vocab.w2i["<pad>"]
            )

        hiddens = [self.embed.weight.shape[1]] + hiddens
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
            layers.append(nn.Tanh())

        if len(layers) > 0:
            layers.pop()  # no activation on last layer
        layers.append(nn.Linear(hiddens[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        # this looks up the embeddings for each word ID in inputs
        # the result is a sequence of word embeddings
        embeds = self.embed(inputs)

        logits = self.layers(embeds)
        logits = logits.sum(dim=1)

        return logits


class DeepCBOWLightning(pl.LightningModule):
    def __init__(
        self,
        embedding_dim,
        vocab,
        vectors=None,
        output_dim=5,
        lr=0.001,
        train_embeddings=False,
        hiddens=[100, 100],
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["vocab", "vectors"])

        self.model = DeepCBOW(
            vocab,
            vectors,
            output_dim=output_dim,
            embedding_dim=embedding_dim,
            train_embeddings=train_embeddings,
            hiddens=hiddens,
        )
        self.loss = nn.CrossEntropyLoss()

        print_parameters(self.model)

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
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)
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

    print(args)

    # Set the random seed manually for reproducibility.
    pl.seed_everything(args.seed)

    i2t = args.classes
    t2i = {p: i for i, p in enumerate(i2t)}  # noqa: F841

    vocab, vectors = None, None
    if args.embeddings_type:
        print(f"Loading embeddings: {args.embeddings_type}")
        embeddings_path = load_embeddings(args.embeddings_type, args.data_dir)
        vocab, vectors = initialize_vocabulary(embeddings_path)

    # Load the dataset
    loader = SST_PL(
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lower=args.lower,
    )
    loader.prepare_data()

    # Load the model
    if args.checkpoint:
        lightning_model = DeepCBOWLightning.load_from_checkpoint(
            args.checkpoint, vocab=loader.vocab, vectors=vectors
        )
    else:
        lightning_model = DeepCBOWLightning(
            args.embedding_dim,
            loader.vocab,
            vectors=vectors,
            output_dim=len(i2t),
            lr=args.lr,
            train_embeddings=args.train_embeddings,
            hiddens=args.hidden_dims,
        )

    model_name = f"{lightning_model.model.__class__.__name__}-{args.embedding_type or 'custom'}{'-ft' if args.train_embeddings else ''}"
    os.makedirs(args.model_dir, exist_ok=True)
    bestmodel_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        filename=f"{model_name}-{{epoch}}-{{val_loss:.2f}}-{{val_acc:.2f}}",
        dirpath=os.path.join(args.model_dir, "checkpoints"),
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=args.model_dir, name=model_name)
    trainer = pl.Trainer(
        accelerator=args.device,
        max_epochs=args.epochs,
        callbacks=[bestmodel_callback],
        logger=logger,
        enable_progress_bar=args.debug,
    )

    if args.evaluate:
        trainer.test(lightning_model, loader.test_dataloader())
    else:
        # Training code + testing
        trainer.fit(
            lightning_model,
            loader.train_dataloader(),
            loader.val_dataloader(),
            ckpt_path=args.checkpoint,
        )

        trainer.test(lightning_model, loader.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    main()
