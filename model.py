import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

from evaluation import compute_classifier_metrics

CHAIN_MAX_LEN = 150
EMBEDDING_DIM = 512


with open("data/embeddings.p", "rb") as f:
    embedding_dict = pickle.load(f)


def encode_seq(sequence):
    encoded = torch.zeros((CHAIN_MAX_LEN, EMBEDDING_DIM))
    try:
        encoded[:len(sequence)] = embedding_dict[sequence]
    except KeyError:
        from igfold import IgFoldRunner
        igfold = IgFoldRunner()
        encoded[:len(sequence)] = igfold.embed(sequences={"H": sequence}).bert_embs.squeeze(0)
    return encoded


def encode_batch(batch_of_sequences):
    encoded_seqs = [encode_seq(seq) for seq in batch_of_sequences]
    seq_lens = [len(seq) for seq in batch_of_sequences]
    return torch.stack(encoded_seqs), torch.as_tensor(seq_lens)


def generate_mask(input_tensor, cdrs):
    mask = torch.ones_like(input_tensor, dtype=torch.bool)
    for i, cdr in enumerate(cdrs):
        for j, c in enumerate(cdr):
            if c == 0.:
                mask[i][j, :] = False
    return mask


class Parabert(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, n_hidden_cells=2 * EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=n_hidden_cells, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(n_hidden_cells * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, mask, lengths):
        # Masking input_tensor with CDRs
        o = input_tensor * mask
        # Packing sequences to remove padding
        packed_seq = pack_padded_sequence(o, lengths, batch_first=True, enforce_sorted=False)
        o_packed, _ = self.lstm(packed_seq)
        # Re-pad sequences before prediction of probabilities
        o, _ = pad_packed_sequence(o_packed, batch_first=True, total_length=CHAIN_MAX_LEN)
        # Predict probabilities
        return self.sigmoid(self.fc(o))


def train(model, train_dl, val_dl, device, epochs=40, cv=None):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    loss_fn = nn.BCELoss()

    # Initialize early stopping parameters
    patience = 10
    early_stopping_counter = 0
    best_validation_loss = float('inf')

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    history = {'train_loss': [], 'val_loss': []}
    tresh = 0.0

    for epoch in tqdm(range(epochs)):
        model.train()

        train_loss = 0.0

        for batch in train_dl:
            chains, labels, cdrs = batch

            optimizer.zero_grad()

            sequences, lengths = encode_batch(chains)
            # Generate a mask for the input
            m = generate_mask(sequences, cdrs).to(device)
            sequences = sequences.to(device)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64).cpu()

            loss = loss_fn(out, labels)
            train_loss += loss.data.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        train_loss = train_loss / len(train_dl)
        val_loss, tresh = evaluate(model, val_dl, device, epoch, cv=cv)

        # Check if the validation loss has improved
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Adjust learning rate with the scheduler
        scheduler.step(val_loss)

        # Check if we should early stop
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

        print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % (epoch + 1, epochs, train_loss, val_loss))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return history, tresh


def evaluate(model, loader, device, epoch, threshold=None, cv=None):
    loss_fn = nn.BCELoss()
    model.eval()
    model = model.to(device)

    val_loss = 0.0
    all_outs = []
    all_lengths = []
    all_labels = []
    all_cdrs = []

    with torch.no_grad():
        for batch in loader:
            chains, labels, cdrs = batch

            sequences, lengths = encode_batch(chains)
            # Generate a mask for the input
            m = generate_mask(sequences, cdrs).to(device)
            sequences = sequences.to(device)
            probabilities = model(sequences, m, lengths)
            out = probabilities.squeeze(2).type(torch.float64).cpu()

            loss = loss_fn(out, labels)
            val_loss += loss.data.item()

            all_outs.append(out)
            all_lengths.extend(lengths)
            all_labels.append(labels)
            all_cdrs.append(cdrs)

    return val_loss / len(loader),\
        compute_classifier_metrics(torch.cat(all_outs),
                                   torch.cat(all_labels),
                                   all_lengths,
                                   torch.cat(all_cdrs),
                                   epoch,
                                   threshold,
                                   cv)
