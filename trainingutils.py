import torch
from utils import pad
import random
import numpy as np
import time


def prepare_example(example, vocab, device="cpu"):
    """
    Map tokens to their IDs for a single example
    """
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = torch.tensor([[vocab.w2i.get(t, 0) for t in example.tokens]], dtype=torch.int64)
    x = x.to(device)
    y = torch.tensor([example.label], dtype=torch.int64)
    y = y.to(device)

    return x, y


def get_examples(data, shuffle=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example


def batch(states):
    """
    Turns a list of states into a single tensor for fast processing.
    This function also chunks (splits) each state into a (h, c) pair"""
    return torch.cat(states, 0).chunk(2, 1)


def unbatch(state):
    """
    Turns a tensor back into a list of states.
    First, (h, c) are merged into a single state.
    Then the result is split into a list of sentences.
    """
    return torch.split(torch.cat(state, 1), 1, 0)


def get_minibatch(data, batch_size=32, shuffle=True):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def prepare_minibatch(mb, vocab, device="cpu"):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    x = torch.tensor(x, dtype=torch.int64).to(device)
    y = torch.tensor([ex.label for ex in mb], dtype=torch.int64).to(device)
    return x, y


def prepare_treelstm_minibatch(mb, vocab, device):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.
    """
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = torch.tensor(
        [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb],
        dtype=torch.int64,
    ).to(device)
    y = torch.tensor([ex.label for ex in mb], dtype=torch.int64).to(device)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y


def simple_evaluate(model, data, prep_fn=prepare_example, device="cpu", **kwargs):
    """Accuracy of a model on given data set."""

    correct = 0
    total = 0
    model.eval()  # disable dropout (explained later)

    for example in data:
        # convert the example input and label to PyTorch tensors
        x, target = prep_fn(example, model.vocab)

        # forward pass without backpropagation (no_grad)
        # get the output from the neural network for input x
        with torch.no_grad():
            logits = model(x)

        # get the prediction
        prediction = logits.argmax(dim=-1)

        # add the number of correct predictions to the total correct
        correct += (prediction == target).sum().item()
        total += 1

    return correct, total, correct / float(total)


def evaluate_minibatch(
    model,
    data,
    batch_fn=get_minibatch,
    prep_fn=prepare_treelstm_minibatch,
    batch_size=32,
    **kwargs,
):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x)

        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)


def train_model(
    model,
    optimizer,
    train_data,
    test_data,
    dev_data,
    num_iterations=10000,
    print_every=1000,
    eval_every=1000,
    batch_fn=get_examples,
    prep_fn=prepare_example,
    eval_fn=simple_evaluate,
    batch_size=1,
    eval_batch_size=None,
    device="cpu",
    model_dir="save/",
    **kwargs,
):
    """Train a model."""

    iter_i = 0
    train_loss = 0.0
    print_num = 0
    start = time.time()
    criterion = torch.nn.CrossEntropyLoss()  # loss function
    best_eval = 0.0
    best_iter = 0

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []
    model.to(device)

    if eval_batch_size is None:
        eval_batch_size = batch_size

    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):
            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)

            B = targets.size(0)  # later we will use B examples per update

            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()

            # backward pass (tip: check the Introduction to PyTorch notebook)

            # erase previous gradients
            optimizer.zero_grad()

            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                print(
                    "Iter %r: loss=%.4f, time=%.2fs"
                    % (iter_i, train_loss, time.time() - start)
                )
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.0

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(
                    model,
                    dev_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                accuracies.append(accuracy)
                print("iter %r: dev acc=%.4f" % (iter_i, accuracy))

                # save best model parameters
                if accuracy > best_eval:
                    print("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = f"{model_dir}/{model.__class__.__name__}.pt"
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter,
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("Done training")

                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = "{}.pt".format(model.__class__.__name__)
                ckpt = torch.load(model_dir + path)
                model.load_state_dict(ckpt["state_dict"])

                _, _, train_acc = eval_fn(
                    model,
                    train_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                _, _, dev_acc = eval_fn(
                    model,
                    dev_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                _, _, test_acc = eval_fn(
                    model,
                    test_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )

                print(
                    "best model iter {:d}: "
                    "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                        best_iter, train_acc, dev_acc, test_acc
                    )
                )

                return losses, accuracies


def evaluate(
    model,
    data,
    batch_fn=get_minibatch,
    prep_fn=prepare_treelstm_minibatch,
    batch_size=16,
):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x)

        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)
