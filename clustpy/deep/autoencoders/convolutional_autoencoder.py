import torch
import numpy as np
from clustpy.deep._early_stopping import EarlyStopping
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep.autoencoders.resnet_ae_modules import resnet18_encoder, resnet18_decoder, resnet50_encoder, \
    resnet50_decoder
from clustpy.deep.autoencoders.flexible_autoencoder import FullyConnectedBlock, FlexibleAutoencoder
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


class ConvolutionalAutoencoder(FlexibleAutoencoder):
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

    def __init__(self, input_height, fc_layers, conv_encoder="resnet18", conv_decoder=None, batch_norm: bool = False,
                 dropout: float = None,
                 activation_fn: torch.nn.Module = torch.nn.ReLU, bias: bool = True,
                 fc_decoder_layers=None, decoder_output_fn=None, pretrained_encoder_weights=None,
                 pretrained_decoder_weights=None, reusable: bool = True, **fc_kwargs):
        """
        Create an instance of a convolutional autoencoder.

        Parameters
        ----------
        input_height: height of the images for the decoder
        fc_layers : list of the different layer sizes from flattened convolutional layer input to embedding. First input needs to be 512.
                 If decoder_layers are not specified then the decoder is symmetric and goes in the same order from embedding to input.
        conv_encoder: architecture of convolutional encoder
        conv_decoder:  architecture of convolutional dencoder, if None is conv_encoder reversed
        batch_norm : bool
            Set True if you want to use torch.nn.BatchNorm1d (default: False)
        dropout : float
            Set the amount of dropout you want to use (default: None)
        activation_fn: activation function from torch.nn, default=torch.nn.ReLU, set the activation function for the hidden layers, if None then it will be linear.
        bias : bool, default=True, set False if you do not want to use a bias term in the linear layers
        fc_decoder_layers : list, default=None, list of different layer sizes from embedding to output of the decoder. If set to None, will be symmetric to layers.
        decoder_output_fn : activation function from torch.nn, default=None, set the activation function for the decoder output layer, if None then it will be linear.
                            e.g. set to torch.nn.Sigmoid if you want to scale the decoder output between 0 and 1.
        pretrained_encoder_weights : weights from torch.vision.models, default=None, indicates whether pretrained resnet weights should be used for the encoder.
        pretrained_decoder_weights : weights from torch.vision.models, default=None, indicates whether pretrained resnet weights should be used for the decoder.
        reusable : bool
            If set to true, deep clustering algorithms will optimize a copy of the autoencoder and not the autoencoder itself.
            Ensures that the same autoencoder can be used by multiple deep clustering algorithms.
            As copies of this object are created, the memory requirement increases (default: True)
        fc_kwargs : additional parameters for FullyConnectedBlock

        """
        if fc_layers[0] not in [512, 2048]:
            raise ValueError(f"First input in fc_layers needs to be 512 or 2048, but is fc_layers[0] = {fc_layers[0]}")

        super(ConvolutionalAutoencoder, self).__init__(fc_layers, batch_norm, dropout, activation_fn, bias,
                                              fc_decoder_layers, decoder_output_fn, reusable)
        self.fitted = False
        self.input_height = input_height
        self.device = detect_device()

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

