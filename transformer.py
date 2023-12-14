import os

import torch
from torch import nn
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import RobertaTokenizer, RobertaModel
from utils import (
    Example,
    examplereader,
    load_dataset,
    parser,
    print_parameters,
)

from torch.utils.data import Dataset


class SST_Roberta(Dataset):
    def __init__(self, data: list[Example], max_len=256, berta_model="roberta-base"):
        super().__init__()
        self.data = data
        self.max_len = max_len

        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            berta_model,
            unk_token="<unk>",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = " ".join(self.data[index].tokens)

        inputs = self.tokenizer(text, return_token_type_ids=True, padding="max_length")
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "mask": torch.tensor(mask, dtype=torch.int64),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int64),
            "targets": torch.tensor(self.data[index].label, dtype=torch.int64),
        }


class SST_RobertaPL(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=0,
        lower=False,
        data_dir="data",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lower = lower
        self.data_dir = data_dir

    def prepare_data(self):
        datasets = load_dataset(self.data_dir)

        self.train_data = SST_Roberta(
            list(examplereader(datasets["train"], lower=self.lower))
        )
        self.val_data = SST_Roberta(
            list(examplereader(datasets["dev"], lower=self.lower))
        )
        self.test_data = SST_Roberta(
            list(examplereader(datasets["test"], lower=self.lower))
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class RobertaSST(nn.Module):
    def __init__(
        self, output_dim=5, hidden_dim=512, dropout=0.32, model_name="roberta-base"
    ):
        super().__init__()
        self.model = RobertaModel.from_pretrained(model_name)
        # freeze model
        # for param in self.model.parameters():
        #     param.requires_grad_(False)

        self.linear = nn.Linear(768, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.model(ids, mask, token_type_ids)

        hidden_state = outputs[0]
        outputs = hidden_state[:, 0]

        outputs = self.linear(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        logits = self.linear2(outputs)
        return logits

class RobertaLightning(pl.LightningModule):
    def __init__(
        self,
        lr=0.00001,
        hidden_layer=512,
        output_dim=5,
        dropout=0.32,
        model_name="roberta-base",
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = RobertaSST(
            output_dim=output_dim,
            hidden_dim=hidden_layer,
            dropout=dropout,
            model_name=model_name,
        )
        self.loss = nn.CrossEntropyLoss()
        print_parameters(self.model)

    def training_step(self, batch):
        ids = batch["ids"]
        mask = batch["mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]

        logits = self.model(ids, mask, token_type_ids)

        loss = self.loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        ids = batch["ids"]
        mask = batch["mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]

        logits = self.model(ids, mask, token_type_ids)
        loss = self.loss(logits, targets)
        acc = (logits.argmax(dim=-1) == targets).sum().float() / targets.size(0)
        self.log(
            "val_acc", acc, prog_bar=True, on_epoch=True, batch_size=targets.size(0)
        )
        self.log("val_loss", loss, on_epoch=True, batch_size=targets.size(0))
        return {"loss": loss, "val_acc": acc}

    def test_step(self, batch):
        ids = batch["ids"]
        mask = batch["mask"]
        token_type_ids = batch["token_type_ids"]
        targets = batch["targets"]

        logits = self.model(ids, mask, token_type_ids)
        loss = self.loss(logits, targets)

        acc = (logits.argmax(dim=-1) == targets).sum().float() / targets.size(0)
        self.log("test_acc", acc, on_epoch=True, batch_size=targets.size(0))
        self.log("test_loss", loss, on_epoch=True, batch_size=targets.size(0))
        return {"loss": loss, "test_acc": acc}

    def on_test_end(self):
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

    loader = SST_RobertaPL(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lower=args.lower,
        data_dir=args.data_dir,
    )
    loader.prepare_data()

    # Load the model
    if args.checkpoint:
        lightning_model = RobertaLightning.load_from_checkpoint(args.checkpoint)
    else:
        lightning_model = RobertaLightning(
            output_dim=len(i2t),
            lr=args.lr,
            model_name=args.model_name,
        )

    # Prepare a callback to save the best model
    model_name = lightning_model.model.__class__.__name__ + "-" + args.model_name.split("-")[-1]
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
