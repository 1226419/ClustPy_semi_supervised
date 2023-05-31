import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep.autoencoders.resnet_ae_modules import resnet18_encoder, resnet18_decoder, resnet50_encoder, \
    resnet50_decoder
from clustpy.deep.autoencoders.flexible_autoencoder import FullyConnectedBlock
from clustpy.deep._utils import detect_device

_VALID_CONV_MODULES = {
    "resnet18": {
        "enc": resnet18_encoder,
        "dec": resnet18_decoder,
    },
    "resnet50": {
        "enc": resnet50_encoder,
        "dec": resnet50_decoder,
    },
}

class ConvolutionalAutoencoder(torch.nn.Module):
    """
    A flexible convolutional autoencoder.
    
    Attributes
    ----------
    conv_encoder: convolutional encoder part of the autoencoder
    conv_decoder: convolutional decoder part of the autoencoder
    fc_encoder : fully connected encoder part of the autoencoder, together with conv_encoder is responsible for embedding data points (class is FullyConnectedBlock)
    fc_decoder : fully connected decoder part of the autoencoder, together with conv_decoder is responsible for reconstructing data points from the embedding (class is FullyConnectedBlock)
    fitted  : boolean value indicating whether the autoencoder is already trained.

    References
    ----------
    Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>
    E.g. Ballard, Dana H. "Modular learning in neural networks." Aaai. Vol. 647. 1987.
    """

    def __init__(self, input_height, fc_layers, conv_encoder="resnet18", conv_decoder=None, activation_fn=torch.nn.ReLU, 
                 fc_decoder_layers=None, decoder_output_fn=None, pretrained_encoder_weights=None, pretrained_decoder_weights=None, **fc_kwargs):
        """
        Create an instance of a convolutional autoencoder.

        Parameters
        ----------
        input_height: height of the images for the decoder
        fc_layers : list of the different layer sizes from flattened convolutional layer input to embedding. First input needs to be 512.
                 If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
        activation_fn: activation function from torch.nn, default=torch.nn.ReLU, set the activation function for the hidden layers, if None then it will be linear.
        bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
        fc_decoder_layers : list, default=None, list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers.
        decoder_output_fn : activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear.
                            e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1.
        pretrained_encoder_weights : weights from torch.vision.models, default=None, indicates whether pretrained resnet weights should be used for the encoder.
        pretrained_decoder_weights : weights from torch.vision.models, default=None, indicates whether pretrained resnet weights should be used for the decoder.
        fc_kwargs : additional parameters for FullyConnectedBlock
        """
        super().__init__()
        self.fitted = False
        self.input_height = input_height
        self.device = detect_device()
        if fc_layers[0] not in [512, 2048]:
            raise ValueError(f"First input in fc_layers needs to be 512 or 2048, but is fc_layers[0] = {fc_layers[0]}")
        if fc_decoder_layers is None:
            fc_decoder_layers = fc_layers[::-1]
        if (fc_layers[-1] != fc_decoder_layers[0]):
            raise ValueError(
                f"Innermost hidden layer and first decoder layer do not match, they are {layers[-1]} and {fc_decoder_layers[0]} respectively.")
        if (fc_layers[0] != fc_decoder_layers[-1]):
            raise ValueError(
                f"Output and input dimension do not match, they are {fc_layers[0]} and {fc_decoder_layers[-1]} respectively.")

        # Setup convolutional encoder and decoder
        if conv_encoder in _VALID_CONV_MODULES:
            self.conv_encoder = _VALID_CONV_MODULES[conv_encoder]["enc"](first_conv=True, maxpool1=True, pretrained_weights=pretrained_encoder_weights)

            if conv_decoder is None:
                conv_decoder = conv_encoder
                self.conv_decoder = _VALID_CONV_MODULES[conv_decoder]["dec"](latent_dim=fc_decoder_layers[-1],
                                                                             input_height=self.input_height,
                                                                             first_conv=True, maxpool1=True,
                                                                             pretrained_weights=pretrained_decoder_weights)
            elif conv_decoder in _VALID_CONV_MODULES:
                self.conv_decoder = _VALID_CONV_MODULES[conv_decoder]["dec"](latent_dim=fc_decoder_layers[-1], input_height=self.input_height, first_conv=True, maxpool1=True, pretrained_weights=pretrained_decoder_weights)
            else:
                raise ValueError(f"value for conv_decoder={conv_decoder} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")
        else:
            raise ValueError(f"value for conv_encoder={conv_encoder} is not valid. Has to be one of {list(_VALID_CONV_MODULES.keys())}")
        print("conv_encoder", self.conv_decoder)
        print("type of decoder", type(self.conv_decoder))
        # Initialize encoder
        self.fc_encoder = FullyConnectedBlock(layers=fc_layers, activation_fn=activation_fn, output_fn=None, **fc_kwargs)
        # Inverts the list of layers to make symmetric version of the encoder
        self.fc_decoder = FullyConnectedBlock(layers=fc_decoder_layers, activation_fn=activation_fn, output_fn=decoder_output_fn, **fc_kwargs)
        self.to(self.device)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the encoder function to x.

        Parameters
        ----------
        x : input data point, can also be a mini-batch of points
        
        Returns
        -------
        embedded : the embedded data point with dimensionality embedding_size
        """
        z = self.conv_encoder(x)
        return self.fc_encoder(z)

    def decode(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Apply the decoder function to embedded.

        Parameters
        ----------
        embedded: embedded data point, can also be a mini-batch of embedded points
        
        Returns
        -------
        reconstruction: returns the reconstruction of a data point
        """
        x = self.fc_decoder(embedded)

        return self.conv_decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies both encode and decode function.
        The forward function is automatically called if we call self(x).

        Parameters
        ----------
        x : input data point, can also be a mini-batch of embedded points
        
        Returns
        -------
        reconstruction: returns the reconstruction of a data point
        """

        embedded = self.encode(x)
        reconstruction = self.decode(embedded)

        return reconstruction

    def loss(self, batch_data, loss_fn, device):
        """
        Calculate the loss of a single batch of data.

        Parameters
        ----------
        batch_data : torch.Tensor, the samples
        loss_fn : torch.nn, loss function to be used for reconstruction

        Returns
        -------
        loss: returns the reconstruction loss of the input sample
        """
        batch_data = batch_data[1].to(device)
        reconstruction = self.forward(batch_data)

        loss = loss_fn(reconstruction, batch_data)
        return loss
    
    def loss_view_invariance(self, batch_data, loss_fn, device):
        """
        Calculate the loss of a single batch of data by reconstructing the .

        Parameters
        ----------
        batch_data : torch.Tensor, the samples
        loss_fn : torch.nn, loss function to be used for reconstruction
        device : device to train on

        Returns
        -------
        loss: returns the reconstruction loss of the input sample
        """
        side = batch_data[0].to(device)
        upper = batch_data[1].to(device)
        
        reconstruction = self.forward(side)
        loss = loss_fn(reconstruction, upper)
        return loss

    def evaluate(self, dataloader, loss_fn, device=torch.device("cpu")):
        """
        Evaluates the autoencoder.
        
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader, dataloader to be used for training
        loss_fn : torch.nn, loss function to be used for reconstruction
        device : torch.device, default=torch.device('cpu'), device to be trained on
        
        Returns
        -------
        loss: returns the reconstruction loss of all samples in dataloader
        """
        with torch.no_grad():
            self.eval()
            loss = 0
            for batch in dataloader:
                #batch_data = batch[1].to(device)
                loss += self.loss(batch, loss_fn, device)
                #loss += self.loss(batch_data, loss_fn, device)
            loss /= len(dataloader)
        return loss

    def fit(self, n_epochs, lr, batch_size=128, data=None, data_eval=None, dataloader=None, evalloader=None,
            optimizer_class=torch.optim.Adam, loss_fn=torch.nn.MSELoss(), patience=5, scheduler=None,
            scheduler_params=None, device=torch.device("cpu"), model_path=None, print_step=0):
        """
        Trains the autoencoder in place.
        
        Parameters
        ----------
        n_epochs : int, number of epochs for training
        lr : float, learning rate to be used for the optimizer_class
        batch_size : int, default=128
        data : np.ndarray, default=None, train data set. If data is passed then dataloader can remain empty
        data_eval : np.ndarray, default=None, evaluation data set. If data_eval is passed then evalloader can remain empty.
        dataloader : torch.utils.data.DataLoader, default=None, dataloader to be used for training
        evalloader : torch.utils.data.DataLoader, default=None, dataloader to be used for evaluation, early stopping and learning rate scheduling if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        optimizer_class : torch.optim, default=torch.optim.Adam, optimizer to be used
        loss_fn : torch.nn, default=torch.nn.MSELoss(), loss function to be used for reconstruction
        patience : int, default=5, patience parameter for EarlyStopping
        scheduler : torch.optim.lr_scheduler, default=None, learning rate scheduler that should be used. 
                    If torch.optim.lr_scheduler.ReduceLROnPlateau is used then the behaviour is matched by providing the validation_loss calculated based on samples from evalloader.
        scheduler_params : dict, default=None, dictionary of the parameters of the scheduler object
        device : torch.device, default=torch.device('cpu'), device to be trained on
        model_path : str, default=None, if specified will save the trained model to the location. If evalloader is used, then only the best model w.r.t. evaluation loss is saved.
        print_step : int, default=0, specifies how often the losses are printed. If 0, no prints will occur
        
        Raises
        ----------
        ValueError: data cannot be None if dataloader is None
        ValueError: evalloader cannot be None if scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        if dataloader is None:
            if data is None:
                raise ValueError("data must be specified if dataloader is None")
            dataloader = get_dataloader(data, batch_size, True)
        # evalloader has priority over data_eval
        if evalloader is None:
            if data_eval is not None:
                evalloader = get_dataloader(data_eval, batch_size, False)
        params_dict = {'params': self.parameters(), 'lr': lr}
        optimizer = optimizer_class(**params_dict)

        early_stopping = EarlyStopping(patience=patience)
        if scheduler is not None:
            scheduler = scheduler(optimizer=optimizer, **scheduler_params)
            # Depending on the scheduler type we need a different step function call.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                eval_step_scheduler = True
                if evalloader is None:
                    raise ValueError(
                        "scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, but evalloader is None. Specify evalloader such that validation loss can be computed.")
            else:
                eval_step_scheduler = False
        best_loss = np.inf
        # training loop
        for epoch_i in range(n_epochs):
            print(epoch_i)
            self.train()
            for batch in dataloader:

                #batch_data = batch[1].to(device)
                loss = self.loss(batch, loss_fn, device)
                #loss = self.loss(batch_data, loss_fn, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                print(f"Epoch {epoch_i}/{n_epochs - 1} - Batch Reconstruction loss: {loss.item():.6f}")

            if scheduler is not None and not eval_step_scheduler:
                scheduler.step()
            # Evaluate autoencoder
            if evalloader is not None:
                # self.evaluate calls self.eval()
                val_loss = self.evaluate(dataloader=evalloader, loss_fn=loss_fn, device=device)
                if print_step > 0 and ((epoch_i - 1) % print_step == 0 or epoch_i == (n_epochs - 1)):
                    print(f"Epoch {epoch_i} EVAL loss total: {val_loss.item():.6f}")
                early_stopping(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_i
                    # Save best model
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)

                if early_stopping.early_stop:
                    if print_step > 0:
                        print(f"Stop training at epoch {best_epoch}")
                        print(f"Best Loss: {best_loss:.6f}, Last Loss: {val_loss:.6f}")
                    break
                if scheduler is not None and eval_step_scheduler:
                    scheduler.step(val_loss)
        # Save last version of model
        if evalloader is None and model_path is not None:
            torch.save(self.state_dict(), model_path)
        # Autoencoder is now pretrained
        self.fitted = True
        self.optimizer = optimizer
        return self