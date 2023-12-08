import math
import os

import torch
from torch import nn
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import (
    TreeDatasetTreePL,
    load_embeddings,
    initialize_vocabulary,
    parser,
    print_parameters,
)

from trainingutils import batch, unbatch

SHIFT = 0
REDUCE = 1


class TreeLSTMCell(nn.Module):
    """A Binary Tree LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(TreeLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
        self.dropout_layer = nn.Dropout(p=0.25)

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hx_l, hx_r, mask=None):
        """
        hx_l is ((batch, hidden_size), (batch, hidden_size))
        hx_r is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h_l, prev_c_l = hx_l  # left child
        prev_h_r, prev_c_r = hx_r  # right child

        # B = prev_h_l.size(0)

        # we concatenate the left and right children
        # you can also project from them separately and then sum
        children = torch.cat([prev_h_l, prev_h_r], dim=1)

        # project the combined children into a 5D tensor for i,fl,fr,g,o
        # this is done for speed, and you could also do it separately
        proj = self.reduce_layer(children)  # shape: B x 5D

        # each shape: B x D
        i, fl, fr, g, o = torch.chunk(proj, 5, dim=-1)

        # that was literally in the provided code...
        c = fl * prev_c_l + fr * prev_c_r + i * g
        h = o * torch.tanh(c)
        return h, c

    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size
        )


class TreeLSTM(nn.Module):
    """Encodes a sentence using a TreeLSTMCell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(TreeLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.reduce = TreeLSTMCell(input_size, hidden_size)

        # project word to initial c
        self.proj_x = nn.Linear(input_size, hidden_size)
        self.proj_x_gate = nn.Linear(input_size, hidden_size)

        self.buffers_dropout = nn.Dropout(p=0.5)

    def forward(self, x, transitions):
        """
        WARNING: assuming x is reversed!
        :param x: word embeddings [B, T, E]
        :param transitions: [2T-1, B]
        :return: root states
        """

        # B = x.size(0)  # batch size
        # T = x.size(1)  # time

        # compute an initial c and h for each word
        # Note: this corresponds to input x in the Tai et al. Tree LSTM paper.
        # We do not handle input x in the TreeLSTMCell itself.
        buffers_c = self.proj_x(x)
        buffers_h = buffers_c.tanh()
        buffers_h_gate = self.proj_x_gate(x).sigmoid()
        buffers_h = buffers_h_gate * buffers_h

        # concatenate h and c for each word
        buffers = torch.cat([buffers_h, buffers_c], dim=-1)

        # D = buffers.size(-1) // 2

        # we turn buffers into a list of stacks (1 stack for each sentence)
        # first we split buffers so that it is a list of sentences (length B)
        # then we split each sentence to be a list of word vectors
        buffers = buffers.split(1, dim=0)  # Bx[T, 2D]
        buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]  # BxTx[2D]

        # create B empty stacks
        stacks = [[] for _ in buffers]

        # t_batch holds 1 transition for each sentence
        for t_batch in transitions:
            child_l = []  # contains the left child for each sentence with reduce action
            child_r = []  # contains the corresponding right child

            # iterate over sentences in the batch
            # each has a transition t, a buffer and a stack
            for transition, buffer, stack in zip(t_batch, buffers, stacks):
                if transition == SHIFT:
                    stack.append(buffer.pop())
                elif transition == REDUCE:
                    assert (
                        len(stack) >= 2
                    ), "Stack too small! Should not happen with valid transition sequences"
                    child_r.append(stack.pop())  # right child is on top
                    child_l.append(stack.pop())

            # if there are sentences with reduce transition, perform them batched
            if child_l:
                reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
                for transition, stack in zip(t_batch, stacks):
                    if transition == REDUCE:
                        stack.append(next(reduced))

        final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
        final = torch.cat(final, dim=0)  # tensor [B, D]

        return final


class TreeLSTMClassifier(nn.Module):
    """Encodes sentence with a TreeLSTM and projects final hidden state"""

    def __init__(
        self,
        vocab,
        embedding_dim,
        hidden_dim,
        output_dim,
        vectors=None,
        train_embeddings=False,
    ):
        super(TreeLSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.treelstm = TreeLSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(hidden_dim, output_dim, bias=True)
        )

        if vectors is not None:
            self.embed = nn.Embedding.from_pretrained(
                vectors, freeze=train_embeddings, padding_idx=vocab.w2i["<pad>"]
            )
        else:
            self.embed = nn.Embedding(
                len(vocab.w2i), embedding_dim, padding_idx=vocab.w2i["<pad>"]
            )

    def forward(self, x):
        # x is a pair here of words and transitions; we unpack it here.
        # x is batch-major: [B, T], transitions is time major [2T-1, B]
        x, transitions = x
        emb = self.embed(x)

        # we use the root/top state of the Tree LSTM to classify the sentence
        root_states = self.treelstm(emb, transitions)

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(root_states)
        return logits


class TreeLSTMLightning(pl.LightningModule):
    def __init__(
        self,
        embedding_dim,
        vocab,
        vectors,
        output_dim=5,
        lr=0.001,
        hidden_dim=100,
        train_embeddings=False,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["vocab", "vectors"])

        self.model = TreeLSTMClassifier(
            vocab,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            vectors=vectors,
            train_embeddings=train_embeddings,
        )
        
        print_parameters(self.model)
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
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=targets.size(0))
        self.log("val_loss", loss, on_epoch=True, batch_size=targets.size(0))
        return {"loss": loss, "val_acc": acc}

    def test_step(self, batch):
        x, targets = batch
        logits = self.model(x)
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

    if args.debug:
        print(args)

    # Set the random seed manually for reproducibility.
    pl.seed_everything(args.seed)

    i2t = args.classes
    t2i = {p: i for i, p in enumerate(i2t)}  # noqa: F841

    # Load the embeddings
    embeddings_path = load_embeddings(args.embeddings_type, args.data_dir)
    vocab, vectors = initialize_vocabulary(embeddings_path)

    loader = TreeDatasetTreePL(
        vocab=vocab,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lower=args.lower,
        data_dir=args.data_dir,
    )
    loader.prepare_data()

    # Load the model
    if args.checkpoint:
        lightning_model = TreeLSTMLightning.load_from_checkpoint(
            args.checkpoint, vocab=vocab, vectors=None
        )
    else:
        lightning_model = TreeLSTMLightning(
            args.embedding_dim,
            vocab,
            vectors,
            len(i2t),
            args.lr,
            args.hidden_dim,
            args.train_embeddings,
        )

    # Prepare a callback to save the best model
    os.makedirs(args.model_dir, exist_ok=True)
    bestmodel_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        filename="TreeLSTM-{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        dirpath=os.path.join(args.model_dir, "checkpoints"),
    )

    trainer = pl.Trainer(
        accelerator=args.device,
        max_epochs=args.epochs,
        callbacks=[bestmodel_callback],
        enable_progress_bar=args.debug,
    )

    if args.evaluate:
        trainer.test(lightning_model, loader.test_dataloader())
    else:
        # Training code + testing
        trainer.fit(lightning_model, loader.train_dataloader(), loader.val_dataloader())

        lightning_model = TreeLSTMLightning.load_from_checkpoint(
            bestmodel_callback.best_model_path, vocab=vocab, vectors=None
        )

        trainer.test(lightning_model, loader.test_dataloader())


if __name__ == "__main__":
    main()
