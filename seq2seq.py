import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence,
)
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, List, Tuple
import pathlib


class EncoderRNN(nn.Module):
    """
    Encoder RNN using GRUs for the Seq2Seq model.

    Parameters
    ----------
    input_dim : int
        The number of input features for each sequence.
    hidden_dim : int
        The number of features to linearly map the input features to. This is
        upscaled to four times the number of features inside the GRU network
        itself.
    layer_dim : int
        The number of layers in the GRU network.
    dropout_p : float, optional
        The probability of dropout in the encoder. Default is 0.1.
    bidirectional : bool, optional
        Whether or not the encoder is bidirectional. Default is False.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, layer_dim: int, dropout_p: float = 0.1, bidirectional: bool = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bidirectional = bidirectional

        self.lin = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=4 * hidden_dim,
            num_layers=layer_dim,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, x: torch.tensor, lens: torch.tensor
    ) -> Tuple[PackedSequence, torch.tensor]:
        """
        The forward pass through the encoder network.

        Parameters
        ----------
        x : torch.tensor
            The input sequence(s) to the encoder.
        lens: torch.tensor
            The unpadded length(s) of the sequence(s) to allow for correct
            packing of sequences.

        Returns
        -------
        outputs : torch.nn.utils.rnn.PackedSequence
            The outputs of the encoder as a packed sequence.
        hidden : torch.tensor
            The final hidden state of the encoder.
        """
        x = self.lin(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        outputs, hidden = self.gru(x)

        return outputs, hidden


class DecoderRNN(nn.Module):
    """
    Decoder RNN using GRUs for he Seq2Seq model.

    Parameters
    ----------
    input_dim : int
        The input features of the sequence.
    hidden_dim : int
        The number of features to linearly map the input features to. Like the
        encoder, this is upscaled by a multiple of four in the GRU network.
    layer_dim : int
        The number of layers to use in the GRU network.
    classes_dim : int
        The number of output dimensions.
    dropout_p : float, optional
        The dropout probability. Default is 0.1.
    device : str, optional
        The device to load the initial decoder input to. Default is "cuda:0".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        classes_dim: int,
        dropout_p: float = 0.1,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classes_dim = classes_dim
        self.device = device

        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, 4 * hidden_dim, layer_dim, batch_first=True, dropout=dropout_p
        )
        self.lin2 = nn.Linear(4 * hidden_dim, classes_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, encoder_outputs: torch.tensor, encoder_hidden: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        The forward pass of the decoder.

        Parameters
        ----------
        encoder_outputs : torch.tensor
            The outputs of the encoder.
        encoder_hidden : torch.tensor
            The final hidden state of the encoder.

        Returns
        -------
        decoder_outputs : torch.tensor
            The decoder outputs logits.
        decoder_hidden : torch.tensor
            The final decoder hidden state.
        """
        batch_size = encoder_outputs.data.size(0)
        seq_len = encoder_outputs.data.size(1)

        decoder_input = torch.zeros(
            batch_size, 1, 1, dtype=torch.float, device=self.device
        )
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for _ in range(seq_len):
            decoder_input = self.lin1(decoder_input)
            decoder_input = self.act(decoder_input)
            decoder_input = self.dropout(decoder_input)
            decoder_output, decoder_hidden = self.gru(decoder_input, decoder_hidden)
            decoder_output = self.lin2(decoder_output)
            decoder_outputs.append(decoder_output)

            _, decoder_input = torch.max(decoder_output, dim=-1)
            decoder_input = decoder_input.unsqueeze(-1).float()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden


class SeqDataset(Dataset):
    """
    Custom dataset for use with sequence data.

    Parameters
    ----------
    data : torch.tensor
        The input sequences.
    labels : torch.tensor, optional
        The output sequences. Default is None.
    """

    def __init__(
        self, data: torch.tensor, labels: Optional[torch.tensor] = None
    ) -> None:
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.tensor, torch.tensor], torch.tensor]:
        if self.labels:
            return self.data[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


def collate_train(batch: SeqDataset) -> Tuple[torch.tensor, torch.tensor, List]:
    """
    The function used for collating data per batch for the sequences for
    training including padding the inputs and outputs to the longest sequence
    length and returning a list of the lengths of the sequences. This is used as
    the argument to `collate_fn` in PyTorch's DataLoader.

    Parameters
    ----------
    batch : SeqDataset
        A batch of data from the dataset.

    Returns
    -------
    padded_data : torch.tensor
        The padded input sequences.
    padded_target : torch.tensor
        The padded output sequences.
    lens : list
        The lengths of each of the unpadded sequences.
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    lens = list(map(len, data))

    padded_data = pad_sequence(data, batch_first=True, padding_value=-100)
    padded_target = pad_sequence(target, batch_first=True, padding_value=-100)

    return padded_data, padded_target, lens


def collate_inf(batch: SeqDataset) -> Tuple[torch.tensor, List]:
    """
    The function used for collating data per batch for the sequences for
    inference including padding the inputs to the longest sequence length and
    returning a list of the lengths of the sequences. This is used as the
    argument to `collate_fn` in PyTorch's DataLoader.

    Parameters
    ----------
    batch : SeqDataset
        A batch of data from the dataset.

    Returns
    -------
    padded_data : torch.tensor
        The padded input sequences.
    lens : list
        The lengths of each of the unpadded sequences.
    """
    data = batch
    lens = list(map(len, data))

    padded_data = pad_sequence(data, batch_first=True, padding_value=-100)

    return padded_data, torch.tensor(lens)


class Seq2SeqInference:
    """
    The class for doing inference with the Seq2Seq model.

    Parameters
    ----------
    enc_model : torch.nn.Module
        The model for the encoder half of the Seq2Seq model.
    dec_model : torch.nn.Module
        The model for the decoder half of the Seq1Seq model.
    batch_size : int
        The batch size to use for inference.
    device_id : torch.device or str or int, Optional
        The device to perform the inference on. Default is 0, so will use device `"cuda:0"`.
    """

    def __init__(
        self,
        enc_model: EncoderRNN,
        dec_model: DecoderRNN,
        batch_size: int,
        device_id: Union[torch.device, str, int] = 0,
    ) -> None:
        if isinstance(device_id, str):
            self.device = torch.device(device_id)
        elif isinstance(device_id, int):
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = device_id

        self.enc = enc_model
        self.dec = dec_model

        self.batch_size = batch_size

    def load_checkpoint(self, filename: pathlib.Path) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if filename.is_file():
            print(f"=> loading checkpoint at {str(filename)}")
            chkpt = torch.load(filename, map_location=self.device)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.enc.load_state_dict(chkpt["model1_state_dict"])
            self.dec.load_state_dict(chkpt["model2_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {str(filename)} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {str(filename)}")

    def load_data(self, data: List) -> None:
        """
        This class method loads the data to be classified into a DataLoader.

        Parameters
        ----------
        data : list
            The data to be loaded into the DataLoader. Must be a list to account
            for possible different sequence lengths.
        """
        tensor_data = list(map(lambda x: torch.tensor(x).unsqueeze(-1).float(), data))

        dataset = SeqDataset(tensor_data)

        self.loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_inf
        )

    def chic_actress(self) -> Tuple[List, List]:
        """
        This class method performs inference on the data in the DataLoader.

        Returns
        -------
        seq_labels : list
            The predicted output sequences with the extra padding removed.
        seq_probs : list
            The predicted probabilities of each class for the output sequence
            with the extra padding removed.
        """
        self.enc.eval()
        self.dec.eval()
        seq_probs = []
        seq_labels = []

        with torch.no_grad():
            for data, lens in self.loader:
                data = data.float().to(self.device)

                encoder_outputs, encoder_hidden = self.enc(data, lens)
                encoder_outputs = pad_packed_sequence(encoder_outputs, batch_first=True)[
                    0
                ]
                if self.enc.bidirectional:
                    _, _, dh = encoder_outputs.shape
                    encoder_outputs = encoder_outputs[:,:,:dh//2] + encoder_outputs[:,:,dh//2:]
                    encoder_hidden = encoder_hidden[:self.enc.layer_dim] + encoder_hidden[self.enc.layer_dim:]
                decoder_outputs, _ = self.dec(encoder_outputs, encoder_hidden)
                decoder_outputs = torch.softmax(decoder_outputs, dim=-1)

                _, predicted = torch.max(decoder_outputs.data, -1)

                for k in range(predicted.size(0)):
                    seq_probs.append(decoder_outputs[k, : lens[k]])
                    seq_labels.append(predicted[k, : lens[k]])

        return seq_labels, seq_probs