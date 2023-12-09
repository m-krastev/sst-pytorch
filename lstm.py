import math
import os

from torch import nn
from torch import optim
import torch

from utils import (
    TreeDatasetPL,
    initialize_vocabulary,
    load_embeddings,
    parser,
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MyLSTMCell(nn.Module):
    """Our own LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.linear_ii = nn.Linear(self.input_size, self.hidden_size)
        self.linear_hi = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_if = nn.Linear(self.input_size, self.hidden_size)
        self.linear_hf = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_ig = nn.Linear(self.input_size, self.hidden_size)
        self.linear_hg = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_io = nn.Linear(self.input_size, self.hidden_size)
        self.linear_ho = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        i = self.sigmoid(self.linear_ii(input_) + self.linear_hi(prev_h))
        f = self.sigmoid(self.linear_if(input_) + self.linear_hf(prev_h))
        g = self.tanh(self.linear_ig(input_) + self.linear_hg(prev_h))
        o = self.sigmoid(self.linear_io(input_) + self.linear_ho(prev_h))

        c = f * prev_c + i * g
        h = o * self.tanh(c)

        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size
        )


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(
        self,
        vocab,
        embedding_dim,
        hidden_dim,
        output_dim,
        vectors=None,
        train_embeddings=False,
    ):
        super(LSTMClassifier, self).__init__()
        
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(hidden_dim, output_dim)  # explained later
        )
        
        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(
                vectors, freeze=train_embeddings, padding_idx=vocab.w2i["<pad>"]
            )
        else:
            self.embed = nn.Embedding(
                len(vocab.w2i), embedding_dim, padding_idx=vocab.w2i["<pad>"]
            )
    
    def forward(self, x, *args):
        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)

        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)

        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)

        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.

            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)  # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]

            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)

            outputs = outputs.contiguous()
            outputs = outputs.masked_fill_(pad_positions, 0.0)

            mask = x != 1  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)  # [B, 1]

            indexes = (lengths - 1) + torch.arange(
                B, device=x.device, dtype=x.dtype
            ) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits


class LSTMLightning(pl.LightningModule):
    def __init__(
        self,
        embedding_dim,
        vocab,
        vectors,
        output_dim=5,
        lr=0.001,
        hidden_dim = 100,
        train_embeddings=False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["vocab", "vectors"])

        self.model = LSTMClassifier(
            vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            vectors = vectors,
            train_embeddings=train_embeddings,
        )
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

    if args.debug:
        print(args)

    # Set the random seed manually for reproducibility.
    pl.seed_everything(args.seed)

    i2t = args.classes
    t2i = {p: i for i, p in enumerate(i2t)}  # noqa: F841

    # Load the embeddings
    embeddings_path = load_embeddings(args.embeddings_type, args.data_dir)
    vocab, vectors = initialize_vocabulary(embeddings_path)

    loader = TreeDatasetPL(
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lower=args.lower,
        data_dir=args.data_dir,
    )
    loader.prepare_data()

    # Load the model
    if args.checkpoint:
        lightning_model = LSTMLightning.load_from_checkpoint(
            args.checkpoint, vocab=vocab, vectors=None
        )
    else:
        lightning_model = LSTMLightning(
            args.embedding_dim,
            vocab,
            vectors,
            len(i2t),
            args.lr,
            args.hidden_dim,
            args.train_embeddings,
        )

    # Prepare a callback to save the best model
    model_name = lightning_model.model.__class__.__name__
    os.makedirs(args.model_dir, exist_ok=True)
    bestmodel_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        filename=f"{model_name}-{{epoch}}-{{val_loss:.2f}}-{{val_acc:.2f}}",
        dirpath=os.path.join(args.model_dir, "checkpoints"),
    )
    
    logger = pl.loggers.TensorBoardLogger(
        save_dir=args.model_dir, name=model_name
    )
    trainer = pl.Trainer(
        accelerator=args.device,
        max_epochs=args.epochs,
        callbacks=[bestmodel_callback],
        logger=logger,
    )

    if args.evaluate:
        trainer.test(lightning_model, loader.test_dataloader())
    else:
        # Training code + testing
        trainer.fit(lightning_model, loader.train_dataloader(), loader.val_dataloader())

        lightning_model = LSTMLightning.load_from_checkpoint(
            bestmodel_callback.best_model_path, vocab=vocab, vectors=None
        )

        trainer.test(lightning_model, loader.test_dataloader())


if __name__ == "__main__":
    main()
