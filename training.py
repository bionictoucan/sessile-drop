import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Optional, Callable, Union, Dict, Tuple
import os
from time import time
from tqdm import tqdm

pt_vibrant = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "BBBBBB",
}


class Trainer:
    """
    This is the default class for defining trainers to teach neural networks
    different tasks.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to train.
    optimiser : torch.optim.Optimizer
        The optimiser to use during training.
    loss_fn : Callable
        The loss function to use during training.
    no_of_epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    data_pth : str
        The path to the training/validation data.
    save_dir : str, optional
        The directory to save the trained models to. Default is ``"./"`` -- the
        :abbr:`CWD (current working directory)`.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        The way to adaptively change the learning rate while training. Default
        is None. For examples on how to adaptively change the learning rate
        please see
        `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.
    device_id : int or str, optional
        Which device to perform training on. Providing an integer will point to
        the GPU with that specific ID whereas the string option can be used to
        also specify training on the CPU. When using ``nn.DataParallel`` for the
        model, setting ``device_id = "cuda"`` uses all GPUs. Default is 0 -- use
        the GPU corresponding to ID ``"cuda:0"``.
    data_parallel : bool, optional
        Whether or not to use multiple GPUs for training. Default is ``False``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        no_of_epochs: int,
        batch_size: int,
        data_pth: Optional[str] = None,
        save_dir: str = "./",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device_id: Union[int, str] = 0,
        data_parallel: bool = False,
    ) -> None:

        self.device = torch.device(
            f"cuda:{device_id}" if type(device_id) == int else device_id
        )

        if data_parallel:
            self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(self.device)

        self.optimiser = optimiser

        self.loss_fn = loss_fn

        self.total_epochs = no_of_epochs
        self.batch_size = batch_size

        self.data_pth = data_pth

        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.scheduler = scheduler

        self.current_epoch = 0

    def load_data(self) -> None:
        """
        User-defined data loading instance method.
        This should populate self.train_in, self.train_out, self.val_in and self.val_out.
        """

        raise NotImplementedError("This must be user-defined!!")

    def checkpoint(self, add_info: Optional[Dict] = None) -> None:
        """
        This class method creates a checkpoint for the current epoch.

        Parameters
        ----------
        add_info : dict, optional
            An additional information to add to the checkpoint e.g. scheduler
            state dictionary. Default is ``None``.
        """

        if isinstance(self.model, torch.nn.DataParallel):
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
        else:
            self.chkpt = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

        if isinstance(add_info, dict):
            self.chkpt.update(add_info)

    def save_checkpoint(self, custom_path: Optional[str] = None) -> None:
        """
        This class method saves the current checkpoint to the save directory
        defined when instantiating the class.

        Parameters
        ----------
        custom_path : str, optional
            Path to save the model to. Default is ``None``.
        """

        current_epoch = str(self.current_epoch).zfill(3)
        if isinstance(custom_path, str):
            save_pth = f"{custom_path}{current_epoch}.pth"
        else:
            save_pth = f"{self.save_dir}{current_epoch}.pth"

        if not hasattr(self, "chkpt"):
            self.checkpoint()
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename: str) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if os.path.isfile(filename):
            print(f"=> loading checkpoint at {filename}")
            chkpt = torch.load(filename)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.model.load_state_dict(chkpt["model_state_dict"])
            self.optimiser.load_state_dict(chkpt["optimiser_state_dict"])

            def_keys = [
                "epoch",
                "train_losses",
                "val_losses",
                "model_state_dict",
                "optimiser_state_dict",
            ]
            if len(chkpt.keys()) != 5:
                self.add_info = {}
                for key in chkpt.keys():
                    if key not in def_keys:
                        self.add_info.update({key: chkpt[key]})
            if self.scheduler:
                self.scheduler.load_state_dict(self.add_info["scheduler_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {filename} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {filename}")


class BinaryTrainer(Trainer):
    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (data, labels) in enumerate(tqdm(train_loader)):
            data, labels = data.float().to(self.device), labels.float().to(self.device)

            self.optimiser.zero_grad()
            output = self.model(data).squeeze()
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            output = (
                output.squeeze()
            )  # gets rid of the second dimension which is always 1
            output = F.sigmoid(output)  # finds the probability from the logit
            predicted = torch.tensor([1 if x >= 0.5 else 0 for x in output]).to(
                self.device
            )
            total += labels.size(0)
            correct += (
                predicted == labels.squeeze()
            ).sum()  # produces a boolean tensor which when summed evaluates True to 1 and False to 0.

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for data, labels in val_loader:
                data, labels = data.float().to(self.device), labels.float().to(
                    self.device
                )
                output = self.model(data).squeeze()

                loss = self.loss_fn(output, labels)
                batch_losses.append(loss.item())

                output = (
                    output.squeeze()
                )  # gets rid of the second dimension which is always 1
                output = F.sigmoid(output)  # finds the probability from the logit
                predicted = torch.tensor([1 if x >= 0.5 else 0 for x in output]).to(
                    self.device
                )
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum()

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def shady_guy(self, load=None, load_pth="."):
        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)

        # initialisation of the plotting environment
        fig = plt.figure(figsize=(6, 6))
        train_ax = fig.add_subplot(2, 1, 1)
        val_ax = train_ax.twinx()
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.tick_params(axis="x", labelbottom=False)
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        train_ax_perc = fig.add_subplot(2, 1, 2)
        val_ax_perc = train_ax_perc.twinx()
        train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
        train_ax_perc.set_xlabel("Number of Epochs")
        val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()

        train_losses, val_losses, train_perc, val_perc = [], [], [], []

        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1

            tl, tp = self.train(train_loader=self.train_loader)
            train_losses.append(tl.item())
            train_perc.append(tp)

            vl, vp = self.validation(val_loader=self.val_loader)
            val_losses.append(vl.item())
            val_perc.append(vp)
            t_now = round(time() - t_init, 3)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl)
                else:
                    self.scheduler.step()

            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_perc = train_perc
            self.val_perc = val_perc

            if self.scheduler:
                self.checkpoint(
                    add_info={
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "train_perc": self.train_perc,
                        "val_perc": self.val_perc,
                    }
                )
            else:
                self.checkpoint(
                    add_info={"train_perc": self.train_perc, "val_perc": self.val_perc}
                )
            self.save_checkpoint()

            # plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.tick_params(axis="x", labelbottom=False)
            train_ax_perc.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax_perc.set_ylabel(
                "Training Percentage [%]", color=pt_vibrant["cyan"]
            )
            val_ax_perc.set_ylabel(
                "Validation Percentage [%]", color=pt_vibrant["magenta"]
            )
            train_ax.plot(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.plot(val_losses, color=pt_vibrant["magenta"], marker="o")
            train_ax_perc.plot(train_perc, color=pt_vibrant["cyan"], marker="o")
            val_ax_perc.plot(val_perc, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()


class MultiTrainer(Trainer):
    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (data, labels) in enumerate(tqdm(train_loader)):
            data, labels = data.float().to(self.device), labels.long().to(self.device)

            self.optimiser.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimiser.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            _, predicted = torch.max(
                output.data, 1
            )  # output will have dimensions (batch_size, output_features) with the maximum taken along the output_features axis i.e. the probability distribution of classes, predicted is populated by the indices of the maximum in this dimension -- the class label.
            total += labels.size(0)
            correct += (
                predicted == labels
            ).sum()  # produces a boolean tensor which when summed evaluates True to 1 and False to 0.

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for data, labels in val_loader:
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                loss = self.loss_fn(output, labels)
                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def shady_guy(self, load=None, load_pth="."):
        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)

        # initialisation of the plotting environment
        fig = plt.figure(figsize=(6, 6))
        train_ax = fig.add_subplot(2, 1, 1)
        val_ax = train_ax.twinx()
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.tick_params(axis="x", labelbottom=False)
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        train_ax_perc = fig.add_subplot(2, 1, 2)
        val_ax_perc = train_ax_perc.twinx()
        train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
        train_ax_perc.set_xlabel("Number of Epochs")
        val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()

        train_losses, val_losses, train_perc, val_perc = [], [], [], []

        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1

            tl, tp = self.train(train_loader=self.train_loader)
            train_losses.append(tl.item())
            train_perc.append(tp)

            vl, vp = self.validation(val_loader=self.val_loader)
            val_losses.append(vl.item())
            val_perc.append(vp)
            t_now = round(time() - t_init, 3)

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl)
                else:
                    self.scheduler.step()

            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_perc = train_perc
            self.val_perc = val_perc

            if self.scheduler:
                self.checkpoint(
                    add_info={
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "train_perc": self.train_perc,
                        "val_perc": self.val_perc,
                    }
                )
            else:
                self.checkpoint(
                    add_info={"train_perc": self.train_perc, "val_perc": self.val_perc}
                )
            self.save_checkpoint()

            # plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.tick_params(axis="x", labelbottom=False)
            train_ax_perc.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax_perc.set_ylabel(
                "Training Percentage [%]", color=pt_vibrant["cyan"]
            )
            val_ax_perc.set_ylabel(
                "Validation Percentage [%]", color=pt_vibrant["magenta"]
            )
            train_ax.plot(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.plot(val_losses, color=pt_vibrant["magenta"], marker="o")
            train_ax_perc.plot(train_perc, color=pt_vibrant["cyan"], marker="o")
            val_ax_perc.plot(val_perc, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()


class TwoModelTrainer:
    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        optimiser1: torch.optim.Optimizer,
        optimiser2: torch.optim.Optimizer,
        loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
        no_of_epochs: int,
        batch_size: int,
        data_pth: Optional[str] = None,
        save_dir: str = "./",
        scheduler1: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler2: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device_id: Union[int, str] = 0,
        data_parallel: bool = False,
    ) -> None:

        self.device = torch.device(
            f"cuda:{device_id}" if type(device_id) == int else device_id
        )

        if data_parallel:
            self.model1 = torch.nn.DataParallel(model1)
            self.model2 = torch.nn.DataParallel(model2)
        else:
            self.model1 = model1
            self.model2 = model2
        self.model1.to(self.device)
        self.model2.to(self.device)

        self.optimiser1 = optimiser1
        self.optimiser2 = optimiser2

        self.loss_fn = loss_fn

        self.total_epochs = no_of_epochs
        self.batch_size = batch_size

        self.data_pth = data_pth

        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2

        self.current_epoch = 0

    def load_data(self) -> None:
        """
        User-defined data loading instance method.
        This should populate self.train_in, self.train_out, self.val_in and self.val_out.
        """

        raise NotImplementedError("This must be user-defined!!")

    def checkpoint(self, add_info: Optional[Dict] = None) -> None:
        """
        This class method creates a checkpoint for the current epoch.

        Parameters
        ----------
        add_info : dict, optional
            An additional information to add to the checkpoint e.g. scheduler
            state dictionary. Default is ``None``.
        """

        if isinstance(self.model1, torch.nn.DataParallel):
            self.chkpt = {
                "epoch": self.current_epoch,
                "model1_state_dict": self.model1.module.state_dict(),
                "optimiser1_state_dict": self.optimiser1.state_dict(),
                "model2_state_dict": self.model2.module.state_dict(),
                "optimiser2_state_dict": self.optimiser2.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
        else:
            self.chkpt = {
                "epoch": self.current_epoch,
                "model1_state_dict": self.model1.state_dict(),
                "optimiser1_state_dict": self.optimiser1.state_dict(),
                "model2_state_dict": self.model2.state_dict(),
                "optimiser2_state_dict": self.optimiser2.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }

        if isinstance(add_info, dict):
            self.chkpt.update(add_info)

    def save_checkpoint(self, custom_path: Optional[str] = None) -> None:
        """
        This class method saves the current checkpoint to the save directory
        defined when instantiating the class.

        Parameters
        ----------
        custom_path : str, optional
            Path to save the model to. Default is ``None``.
        """

        current_epoch = str(self.current_epoch).zfill(3)
        if isinstance(custom_path, str):
            save_pth = f"{custom_path}{current_epoch}.pth"
        else:
            save_pth = f"{self.save_dir}{current_epoch}.pth"

        if not hasattr(self, "chkpt"):
            self.checkpoint()
        torch.save(self.chkpt, save_pth)

    def load_checkpoint(self, filename: str) -> None:
        """
        This class method loads a checkpoint for the model.

        Parameters
        ----------
        filename : str
            The path to the model to load.
        """

        if os.path.isfile(filename):
            print(f"=> loading checkpoint at {filename}")
            chkpt = torch.load(filename)
            self.current_epoch = chkpt["epoch"]
            self.train_losses = chkpt["train_losses"]
            self.val_losses = chkpt["val_losses"]
            self.model1.load_state_dict(chkpt["model1_state_dict"])
            self.optimiser1.load_state_dict(chkpt["optimiser1_state_dict"])
            self.model2.load_state_dict(chkpt["model2_state_dict"])
            self.optimiser2.load_state_dict(chkpt["optimiser2_state_dict"])

            def_keys = [
                "epoch",
                "train_losses",
                "val_losses",
                "model1_state_dict",
                "optimiser1_state_dict",
                "model2_state_dict",
                "optimiser2_state_dict",
            ]
            if len(chkpt.keys()) != 7:
                self.add_info = {}
                for key in chkpt.keys():
                    if key not in def_keys:
                        self.add_info.update({key: chkpt[key]})
            if self.scheduler1:
                self.scheduler1.load_state_dict(self.add_info["scheduler1_state_dict"])
            if self.scheduler2:
                self.scheduler2.load_state_dict(self.add_info["scheduler2_state_dict"])

            train_l = self.train_losses[-1]
            val_l = self.val_losses[-1]
            print(
                f"=> loaded checkpoint at {filename} at epoch {self.current_epoch} with training loss {train_l} and validation loss {val_l}."
            )
        else:
            print(f"=> no checkpoint found at {filename}")


class Seq2SeqTrainer(TwoModelTrainer):
    def train(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        The function used to train the classifier model.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The data to train the model on wrapped up in a `PyTorch DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the training data.
        """
        self.model1.train()
        self.model2.train()

        batch_losses = []
        total, correct = 0.0, 0.0
        for j, (data, labels, lens) in enumerate(tqdm(train_loader)):
            data, labels = data.float().to(self.device), labels.long().to(self.device)

            self.optimiser1.zero_grad()
            self.optimiser2.zero_grad()
            encoder_outputs, encoder_hidden = self.model1(data, lens)
            encoder_outputs = pad_packed_sequence(encoder_outputs, batch_first=True)[0]
            if hasattr(self.model1, "bidirectional"):
                if self.model1.bidirectional:
                    #if the encoder is bidirectional, add the bidirectional
                    #encodings together from
                    #https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/5
                    #outputs to GRU have dims (batch_size, seq_lens, dh) and
                    #hidden have dims (2*num_layers, dh//2)
                    _, _, dh = encoder_outputs.shape
                    encoder_outputs = encoder_outputs[:, :, :dh//2] + encoder_outputs[:, :, dh//2:]
                    encoder_hidden = encoder_hidden[:self.model1.layer_dim] + encoder_hidden[self.model1.layer_dim:]
            decoder_outputs, _ = self.model2(encoder_outputs, encoder_hidden)

            loss = self.loss_fn(
                decoder_outputs.view(-1, decoder_outputs.size(-1)), labels.view(-1)
            )
            loss.backward()
            self.optimiser1.step()
            self.optimiser2.step()

            batch_losses.append(loss.item())

            # work out the % correct per epoch accumulatively
            _, predicted = torch.max(
                decoder_outputs.data, -1
            )  # output will have dimensions (batch_size, output_features) with the maximum taken along the output_features axis i.e. the probability distribution of classes, predicted is populated by the indices of the maximum in this dimension -- the class label.
            for k in range(predicted.size(0)):
                total += labels[k, : lens[k]].view(-1).size(0)
                correct += (
                    predicted[k, : lens[k]].view(-1) == labels[k, : lens[k]].view(-1)
                ).sum()

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def validation(
        self, val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """
        The function used to validate the classifier model.

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            The data to validate the model on wrapped up in a `PyTorch
            DataLoader
            <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_.

        Returns
        -------
         : float
            The average of the batch losses for one training epoch.
         : float
            The percentage correct on the validation data.
        """
        self.model1.eval()
        self.model2.eval()

        total, correct = 0.0, 0.0
        with torch.no_grad():
            batch_losses = []
            for data, labels, lens in val_loader:
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                encoder_outputs, encoder_hidden = self.model1(data, lens)
                encoder_outputs = pad_packed_sequence(
                    encoder_outputs, batch_first=True
                )[0]
                if hasattr(self.model1, "bidirectional"):
                    if self.model1.bidirectional:
                        #if the encoder is bidirectional, add the bidirectional
                        #encodings together from
                        #https://discuss.pytorch.org/t/about-bidirectional-gru-with-seq2seq-example-and-some-modifications/15588/5
                        #outputs to GRU have dims (batch_size, seq_lens, dh) and
                        #hidden have dims (2*num_layers, dh//2)
                        _, _, dh = encoder_outputs.shape
                        encoder_outputs = encoder_outputs[:, :, :dh//2] + encoder_outputs[:, :, dh//2:]
                        encoder_hidden = encoder_hidden[:self.model1.layer_dim] + encoder_hidden[self.model1.layer_dim:]
                decoder_outputs, _ = self.model2(encoder_outputs, encoder_hidden)

                _, predicted = torch.max(decoder_outputs.data, -1)

                loss = self.loss_fn(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)), labels.view(-1)
                )

                for k in range(predicted.size(0)):
                    total += labels[k, : lens[k]].view(-1).size(0)
                    correct += (
                        predicted[k, : lens[k]].view(-1)
                        == labels[k, : lens[k]].view(-1)
                    ).sum()

                batch_losses.append(loss.item())

        return torch.mean(torch.tensor(batch_losses)), float((correct / total) * 100)

    def shady_guy(self, load=None, load_pth="."):
        if load:
            print("=> a model is being loaded.")
            self.load_checkpoint(load_pth)

        # initialisation of the plotting environment
        fig = plt.figure(figsize=(6, 6))
        train_ax = fig.add_subplot(2, 1, 1)
        val_ax = train_ax.twinx()
        train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
        train_ax.tick_params(axis="x", labelbottom=False)
        val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
        train_ax_perc = fig.add_subplot(2, 1, 2)
        val_ax_perc = train_ax_perc.twinx()
        train_ax_perc.set_ylabel("Training Percentage [%]", color=pt_vibrant["cyan"])
        train_ax_perc.set_xlabel("Number of Epochs")
        val_ax_perc.set_ylabel("Validation Percentage [%]", color=pt_vibrant["magenta"])
        fig.show()
        fig.canvas.draw()

        train_losses, val_losses, train_perc, val_perc = [], [], [], []

        t_init = time()
        for n in range(self.total_epochs):
            if n != 0:
                self.current_epoch += 1
            if n == 0 and load:
                self.current_epoch += 1

            tl, tp = self.train(train_loader=self.train_loader)
            train_losses.append(tl.item())
            train_perc.append(tp)

            vl, vp = self.validation(val_loader=self.val_loader)
            val_losses.append(vl.item())
            val_perc.append(vp)
            t_now = round(time() - t_init, 3)

            if self.scheduler1:
                if isinstance(self.scheduler1, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler1.step(vl)
                else:
                    self.scheduler1.step()

            if self.scheduler2:
                if isinstance(self.scheduler2, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler2.step(vl)
                else:
                    self.scheduler2.step()

            self.train_losses = train_losses
            self.val_losses = val_losses
            self.train_perc = train_perc
            self.val_perc = val_perc

            if self.scheduler1 and self.scheduler2:
                self.checkpoint(
                    add_info={
                        "scheduler1_state_dict": self.scheduler1.state_dict(),
                        "scheduler2_state_dict": self.scheduler2.state_dict(),
                        "train_perc": self.train_perc,
                        "val_perc": self.val_perc,
                    }
                )
            elif self.scheduler1 and not self.scheduler2:
                self.checkpoint(
                    add_info={
                        "scheduler1_state_dict": self.scheduler1.state_dict(),
                        "tran_perc": self.train_perc,
                        "val_perc": self.val_perc,
                    }
                )
            elif not self.scheduler1 and self.scheduler2:
                self.checkpoint(
                    add_info={
                        "scheduler2_state_dict": self.scheduler2.state_dict(),
                        "train_perc": self.train_perc,
                        "val_perc": self.val_perc,
                    }
                )
            else:
                self.checkpoint(
                    add_info={"train_perc": self.train_perc, "val_perc": self.val_perc}
                )
            self.save_checkpoint()

            # plot the results
            fig.suptitle(f"Time elapsed {t_now}s after epoch {self.current_epoch}")
            train_ax.set_ylabel("Training Loss", color=pt_vibrant["cyan"])
            train_ax.tick_params(axis="x", labelbottom=False)
            train_ax_perc.set_xlabel("Number of Epochs")
            val_ax.set_ylabel("Validation Loss", color=pt_vibrant["magenta"])
            train_ax_perc.set_ylabel(
                "Training Percentage [%]", color=pt_vibrant["cyan"]
            )
            val_ax_perc.set_ylabel(
                "Validation Percentage [%]", color=pt_vibrant["magenta"]
            )
            train_ax.plot(train_losses, color=pt_vibrant["cyan"], marker="o")
            val_ax.plot(val_losses, color=pt_vibrant["magenta"], marker="o")
            train_ax_perc.plot(train_perc, color=pt_vibrant["cyan"], marker="o")
            val_ax_perc.plot(val_perc, color=pt_vibrant["magenta"], marker="o")
            fig.canvas.draw()
