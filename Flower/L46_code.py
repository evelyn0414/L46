## Import relevant modules
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy, AUROC
from tqdm import trange
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
import random

print("Imported modules.")



## Select the dataset
datasets = ["heart_disease", "ixi" ,"isic2019"]
dataset_selected = datasets[0]
if dataset_selected == datasets[0]:
    from flamby.datasets.fed_heart_disease import FedHeartDisease as FedDataset
    from flamby.datasets.fed_heart_disease import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED,
        BaselineLoss,
        NUM_CLIENTS,
        Optimizer,
        get_nb_max_rounds
    )

elif dataset_selected == datasets[1]:
    from flamby.datasets.fed_ixi import FedIXITiny as FedDataset
    from flamby.datasets.fed_ixi import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED,
        BaselineLoss,
        NUM_CLIENTS,
        Optimizer,
        get_nb_max_rounds
    )
    def metric(output, target, epsilon=1e-9):
        SPATIAL_DIMENSIONS = 2, 3, 4
        p0 = output
        g0 = target
        p1 = 1 - p0
        g1 = 1 - g0
        tp = (p0 * g0).sum(dim=SPATIAL_DIMENSIONS)
        fp = (p0 * g1).sum(dim=SPATIAL_DIMENSIONS)
        fn = (p1 * g0).sum(dim=SPATIAL_DIMENSIONS)
        num = 2 * tp
        denom = 2 * tp + fp + fn + epsilon
        dice_score = num / denom
        return torch.mean(dice_score).item()

elif dataset_selected == datasets[2]:
    from flamby.datasets.fed_isic2019 import FedIsic2019 as FedDataset
    from flamby.datasets.fed_isic2019 import (
        BATCH_SIZE,
        LR,
        NUM_EPOCHS_POOLED,
        BaselineLoss,
        NUM_CLIENTS,
        Optimizer,
        get_nb_max_rounds
    )
    def metric(logits, y_true):
        y_true = torch.reshape(y_true, (-1,))
        preds = torch.argmax(logits, dim=1)
        return balanced_accuracy_score(y_true.cpu().detach().numpy(), preds.cpu().detach().numpy())

print("Batch size = ", BATCH_SIZE)
print("Optimizer = ", Optimizer)
print("Learning rate = ", LR)
print("Local iteration = ", NUM_EPOCHS_POOLED)
print("Global round = ", get_nb_max_rounds(NUM_EPOCHS_POOLED))



## Set FL training options
def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class option:
  def __init__(self):
    self.alg = 'FedAvg'  # [FedAvg | FedBN | FedProx | FedAP]
    self.dataset = dataset_selected  # [heart_disease | ixi | isic2019]
    self.device = 'cuda'  # [cuda | cpu]
    self.batch = BATCH_SIZE
    self.iters = NUM_EPOCHS_POOLED
    self.round = get_nb_max_rounds(NUM_EPOCHS_POOLED)
    self.optimizer = Optimizer
    self.lr = LR
    if dataset_selected == "heart_disease":
      self.metric = Accuracy(task="binary")
    elif dataset_selected == "ixi":
      self.metric = metric
    elif dataset_selected == "isic2019":
      self.metric = metric #AUROC(task="multiclass", num_classes=8)
    self.loss = BaselineLoss()
    self.n_clients = NUM_CLIENTS
    self.nosharebn = True if self.alg in ['FedBN', 'FedAP'] else False
    self.mu = 1e-2 # only for FedProx
    self.seed = 0


args = option()
args.random_state = np.random.RandomState(1)
set_random_seed(args.seed)
print("Strategy is: ", args.alg)



## Load the dataset and conduct Federated Learning dataset partition
def global_test_dataset():
    return [
      torch.utils.data.DataLoader(
      FedDataset(train=False, pooled=True),
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=0,
      drop_last=True
    )
  ]

def local_val_datasets():
  return [
    torch.utils.data.DataLoader(
        FedDataset(center=i, train=False, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
        )
    for i in range(NUM_CLIENTS)
  ]

def local_train_datasets():
  return [
    torch.utils.data.DataLoader(
        FedDataset(center=i, train=True, pooled=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
        )
    for i in range(NUM_CLIENTS)
  ]

def client_datasets(cid):
  train_loaders = local_train_datasets()
  val_loaders = local_val_datasets()
  return train_loaders[cid], val_loaders[cid]

def server_dataset():
  test_loader = global_test_dataset()
  return test_loader[0]

# train_loaders = local_train_datasets()
# val_loaders = local_val_datasets()
# test_loader = global_test_dataset()
# test_loader = test_loader[0]



## Observe size of the dataset
# assert args.n_clients == len(train_loaders)

# print("Number of clients: ", len(val_loaders))
# for i, client in enumerate(val_loaders):
#   print("client", i+1, "number of batches" , len(client))
#   for data in client:
#     print(data[0].shape, data[1].shape)
#     print(data[0], data[1])



## Centralized Federated Learning using Flower framework
## Import Flower relevant modules
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from flwr.common.typing import Parameters
from tqdm import tqdm



## Create neural network model
from efficientnet_pytorch import EfficientNet


class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        normalization: Optional[str] = None,
        kernel_size: int = 3,
        activation: Optional[str] = "ReLU",
        preactivation: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        block = nn.ModuleList()

        dilation = 1 if dilation is None else dilation
        if padding:
            total_padding = kernel_size + 2 * (dilation - 1) - 1
            padding = total_padding // 2

        class_name = "Conv{}d".format(dimensions)
        conv_class = getattr(nn, class_name)
        no_bias = not preactivation and (normalization is not None)
        conv_layer = conv_class(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=not no_bias,
        )

        norm_layer = None
        if normalization is not None:
            class_name = "{}Norm{}d".format(normalization.capitalize(), dimensions)
            norm_class = getattr(nn, class_name)
            num_features = in_channels if preactivation else out_channels
            norm_layer = norm_class(num_features)

        activation_layer = None
        if activation is not None:
            activation_layer = getattr(nn, activation)()

        if preactivation:
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)
            self.add_if_not_none(block, conv_layer)
        else:
            self.add_if_not_none(block, conv_layer)
            self.add_if_not_none(block, norm_layer)
            self.add_if_not_none(block, activation_layer)

        dropout_layer = None
        if dropout:
            class_name = "Dropout{}d".format(dimensions)
            dropout_class = getattr(nn, class_name)
            dropout_layer = dropout_class(p=dropout)
            self.add_if_not_none(block, dropout_layer)

        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

    @staticmethod
    def add_if_not_none(module_list, module):
        if module is not None:
            module_list.append(module)


# Decoding

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = ("nearest", "linear", "bilinear", "bicubic", "trilinear")


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        num_decoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = DecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
        self,
        in_channels_skip_connection: int,
        dimensions: int,
        upsampling_type: str,
        normalization: Optional[str],
        preactivation: bool = True,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.residual = residual

        if upsampling_type == "conv":
            in_channels = out_channels = 2 * in_channels_skip_connection
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, out_channels
            )
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
        in_channels_first = in_channels_skip_connection * (1 + 2)
        out_channels = in_channels_skip_connection
        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def forward(self, skip_connection, x):
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x

    def center_crop(self, skip_connection, x):
        skip_shape = torch.tensor(skip_connection.shape)
        x_shape = torch.tensor(x.shape)
        crop = skip_shape[2:] - x_shape[2:]
        half_crop = crop // 2
        # If skip_connection is 10, 20, 30 and x is (6, 14, 12)
        # Then pad will be (-2, -2, -3, -3, -9, -9)
        pad = -torch.stack((half_crop, half_crop)).t().flatten()
        skip_connection = F.pad(skip_connection, pad.tolist())
        return skip_connection


def get_upsampling_layer(upsampling_type: str) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = 'Upsampling type is "{}"' " but should be one of the following: {}"
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    upsample = nn.Upsample(scale_factor=2, mode=upsampling_type, align_corners=False)
    return upsample


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = "ConvTranspose{}d".format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == "linear":
        if dimensions == 2:
            upsampling_type = "bilinear"
        elif dimensions == 3:
            upsampling_type = "trilinear"
    return upsampling_type


# Encoding


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        pooling_type: str,
        num_encoding_blocks: int,
        normalization: Optional[str],
        preactivation: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        initial_dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        is_first_block = True
        for _ in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            if dimensions == 2:
                in_channels = out_channels_first
                out_channels_first = in_channels * 2
            elif dimensions == 3:
                in_channels = 2 * out_channels_first
                out_channels_first = in_channels
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        for encoding_block in self.encoding_blocks:
            x, skip_connnection = encoding_block(x)
            skip_connections.append(skip_connnection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        dimensions: int,
        normalization: Optional[str],
        pooling_type: Optional[str],
        preactivation: bool = False,
        is_first_block: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = "zeros",
        activation: Optional[str] = "ReLU",
        dilation: Optional[int] = None,
        dropout: float = 0,
    ):
        super().__init__()

        self.preactivation = preactivation
        self.normalization = normalization

        self.residual = residual

        if is_first_block:
            normalization = None
            preactivation = None
        else:
            normalization = self.normalization
            preactivation = self.preactivation

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if dimensions == 2:
            out_channels_second = out_channels_first
        elif dimensions == 3:
            out_channels_second = 2 * out_channels_first
        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            preactivation=self.preactivation,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsample = None
        if pooling_type is not None:
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        if self.downsample is None:
            return x
        else:
            skip_connection = x
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(
    dimensions: int, pooling_type: str, kernel_size: int = 2
) -> nn.Module:
    class_name = "{}Pool{}d".format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)


class Baseline(nn.Module):
    if args.dataset == datasets[0]:
        def __init__(self, input_dim=13, output_dim=1):
            super(Baseline, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)
            self.linear1 = torch.nn.Linear(input_dim, 32)
            self.bn = torch.nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.linear2 = torch.nn.Linear(32, output_dim)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(self.bn(x))
            x = self.linear2(x)
            return torch.sigmoid(x)
            # return torch.sigmoid(self.linear(x))

    elif args.dataset == datasets[1]:
        def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 3,
            num_encoding_blocks: int = 3,
            out_channels_first_layer: int = 8,
            normalization: Optional[str] = "batch",
            pooling_type: str = "max",
            upsampling_type: str = "linear",
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 1,
            padding_mode: str = "zeros",
            activation: Optional[str] = "PReLU",
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            monte_carlo_dropout: float = 0,
        ):
            super().__init__()
            self.CHANNELS_DIMENSION = 1
            depth = num_encoding_blocks - 1

            # Force padding if residual blocks
            if residual:
                padding = 1

            # Encoder
            self.encoder = Encoder(
                in_channels,
                out_channels_first_layer,
                dimensions,
                pooling_type,
                depth,
                normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                initial_dilation=initial_dilation,
                dropout=dropout,
            )

            # Bottom (last encoding block)
            in_channels = self.encoder.out_channels
            if dimensions == 2:
                out_channels_first = 2 * in_channels
            else:
                out_channels_first = in_channels

            self.bottom_block = EncodingBlock(
                in_channels,
                out_channels_first,
                dimensions,
                normalization,
                pooling_type=None,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.encoder.dilation,
                dropout=dropout,
            )

            # Decoder
            if dimensions == 2:
                power = depth - 1
            elif dimensions == 3:
                power = depth
            in_channels = self.bottom_block.out_channels
            in_channels_skip_connection = out_channels_first_layer * 2**power
            num_decoding_blocks = depth
            self.decoder = Decoder(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                num_decoding_blocks,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                initial_dilation=self.encoder.dilation,
                dropout=dropout,
            )

            # Monte Carlo dropout
            self.monte_carlo_layer = None
            if monte_carlo_dropout:
                dropout_class = getattr(nn, "Dropout{}d".format(dimensions))
                self.monte_carlo_layer = dropout_class(p=monte_carlo_dropout)

            # Classifier
            if dimensions == 2:
                in_channels = out_channels_first_layer
            elif dimensions == 3:
                in_channels = 2 * out_channels_first_layer
            self.classifier = ConvolutionalBlock(
                dimensions, in_channels, out_classes, kernel_size=1, activation=None
            )

        def forward(self, x):
            skip_connections, encoding = self.encoder(x)
            encoding = self.bottom_block(encoding)
            x = self.decoder(skip_connections, encoding)
            if self.monte_carlo_layer is not None:
                x = self.monte_carlo_layer(x)
            x = self.classifier(x)
            x = F.softmax(x, dim=self.CHANNELS_DIMENSION)
            return x

    elif args.dataset == datasets[2]:
        """Baseline model
        We use the EfficientNets architecture that many participants in the ISIC
        competition have identified to work best.
        See here the [reference paper](https://arxiv.org/abs/1905.11946)
        Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
        [pytorch reimplementation of EfficientNets]
        (https://github.com/lukemelas/EfficientNet-PyTorch).
        """

        def __init__(self, pretrained=False, arch_name="efficientnet-b0"):
            super(Baseline, self).__init__()
            self.pretrained = pretrained
            self.base_model = (
                EfficientNet.from_pretrained(arch_name)
                if pretrained
                else EfficientNet.from_name(arch_name)
            )
            # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
            nftrs = self.base_model._fc.in_features
            print("Number of features output by EfficientNet", nftrs)
            self.base_model._fc = nn.Linear(nftrs, 8)

        def forward(self, image):
            out = self.base_model(image)
            return out

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        if args.nosharebn == True:
          return [val.cpu().numpy() for name, val in self.state_dict().items() if (('bn' not in name) or ('nor' not in name))]
        else:
          return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        if args.nosharebn == True:
          keys = [k for k in self.state_dict().keys() if (('bn' not in k) or ('nor' not in k))]
          params_dict = zip(keys, weights)
          state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
          self.load_state_dict(state_dict, strict=False)
        else:
          layer_dict = {}
          for k, v in zip(self.state_dict().keys(), weights):
            if v.ndim != 0:
              layer_dict[k] = torch.Tensor(v)
          state_dict = OrderedDict(layer_dict)
          self.load_state_dict(state_dict, strict=True)


def train(
    net: Baseline,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    proximal_term: float,
    log_progress: bool = True):
    # Define loss and optimizer
    criterion = args.loss
    optimizer = args.optimizer(net.parameters(), lr=args.lr)

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_correct, n_samples = 0.0, 0.0, 0
    pbar = tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f'TRAIN') if log_progress else iter(cycle(trainloader))

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        outputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(outputs)
        if args.alg == "FedProx":
            criterion_2 = nn.Identity()
            loss = criterion(outputs, labels) + criterion_2(proximal_term)
            #print("Total loss = ", loss)
            #print("Single loss = ", criterion(outputs, labels))
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        acc = args.metric(outputs, labels.type(torch.int))
        n_samples += labels.size(0)
        total_correct += acc * labels.size(0)

        if log_progress:
            pbar.set_postfix({
                "train_loss": total_loss/n_samples, 
                "train_acc": total_correct/n_samples
            })
    if log_progress:
        print("\n")
    
    return total_loss/n_samples, total_correct/n_samples, n_samples   
  

def test(
    net: Baseline,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True):
    """Evaluates the network on test data."""
    criterion = args.loss
    total_loss, total_correct, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            outputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(outputs)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            acc = args.metric(outputs, labels.type(torch.int))
            n_samples += labels.size(0)
            total_correct += acc * labels.size(0)
    if log_progress:
        print("\n")
    
    return total_loss/n_samples, total_correct/n_samples, n_samples



## Create Flower custom client
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
    GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays


class FL_Client(fl.client.Client):
    def __init__(self, args, client_datasets, cid: str, log_progress: bool = False):
        self.cid = cid
        self.args = args
        self.client_datasets = client_datasets
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = Baseline()

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, ins: GetPropertiesIns):
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(self, ins: GetParametersIns):
        return GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(self.net.get_weights()))

    def set_parameters(self, parameters):
        if args.alg == "FedAP":
            if isinstance(parameters, Parameters):
                self.net.set_weights(parameters_to_ndarrays(parameters))
                return parameters_to_ndarrays(parameters)
            else:
                self.net.set_weights(parameters_to_ndarrays(parameters[int(self.cid)]))
                return parameters_to_ndarrays(parameters[int(self.cid)])
        else:
            self.net.set_weights(parameters_to_ndarrays(parameters))
            return parameters_to_ndarrays(parameters)

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        trainloader, _ = self.client_datasets(int(self.cid))
        num_iterations = fit_params.config["num_iterations"]
        global_parameters = self.set_parameters(fit_params.parameters)
        
        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_acc, num_examples = train(self.net, trainloader, device=self.device, \
            num_iterations=num_iterations, proximal_term=0.0, log_progress=self.log_progress)

        # Proximal term calculation for FedProx strategy and re-train
        if self.args.alg == 'FedProx':
            local_parameters = parameters_to_ndarrays(self.get_parameters(fit_params.config).parameters)
            proximal_term = 0.0
            for w, w_t in zip(local_parameters, global_parameters):
                proximal_term += args.mu / 2 * np.sqrt(np.sum(np.power((w - w_t), 2)))
            #print("Proximal term = ", proximal_term)
            
            self.net.to("cpu")
            self.set_parameters(fit_params.parameters)
            self.net.to(self.device)
            train_loss, train_acc, num_examples = train(self.net, trainloader, device=self.device, \
                num_iterations=num_iterations, proximal_term=proximal_term, log_progress=self.log_progress)
        
        print(f"Client {self.cid}: training round complete, {num_examples} examples processed")

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config).parameters, 
            num_examples=num_examples, 
            metrics={"loss": train_loss, "accuracy": train_acc})

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        _, valloader = self.client_datasets(int(self.cid))
        loss, accuracy, num_examples = test(self.net, valloader, device=self.device, log_progress=self.log_progress)

        print(f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={accuracy:.4f}")
        # Return evaluation information
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss, num_examples=num_examples, 
            metrics={"accuracy": accuracy})



## Create server-side evaluation and experiment
import functools
from flwr.server.strategy import FedAvg, FedAP
from flwr.server.app import ServerConfig


def print_model_layers(model):
    print(model)
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def serverside_eval(server_round, parameters: NDArrays, config, server_dataset):
    """An evaluation function for centralized/serverside evaluation over the entire test set."""
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = Baseline()
    #print_model_layers(model)

    model.set_weights(parameters)
    model.to(device)

    testloader = server_dataset()
    loss, accuracy, n_samples = test(model, testloader, device=device, log_progress=False)

    print(f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")
    return loss, {"accuracy": accuracy}


def start_experiment(
    args,
    client_datasets,
    server_dataset,
    num_rounds=args.round, 
    alg=args.alg,
    client_pool_size=args.n_clients, 
    num_iterations=args.iters, 
    fraction_fit=1.0,
    min_fit_clients=1,
    batch_size=args.batch):
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Configure the strategy
    def fit_config(server_round: int):
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # Strategy selection
    if alg in ['FedAvg', 'FedBN', 'FedProx']:
      strategy = FedAvg(
          fraction_fit=fraction_fit,
          fraction_evaluate=fraction_fit,
          min_fit_clients=min_fit_clients,
          min_evaluate_clients=min_fit_clients,
          min_available_clients=client_pool_size,  # all clients should be available
          on_fit_config_fn=fit_config,
          on_evaluate_config_fn=(lambda r: {"batch_size": 100}),
          evaluate_fn=functools.partial(serverside_eval, server_dataset=server_dataset),
          accept_failures=False,
      )
    elif alg == 'FedAP':
      strategy = FedAP(
          fraction_fit=fraction_fit,
          fraction_evaluate=fraction_fit,
          min_fit_clients=min_fit_clients,
          min_evaluate_clients=min_fit_clients,
          min_available_clients=client_pool_size,  # all clients should be available
          on_fit_config_fn=fit_config,
          on_evaluate_config_fn=(lambda r: {"batch_size": 100}),
          evaluate_fn=functools.partial(serverside_eval, server_dataset=server_dataset),
          accept_failures=False,
      )

    print(f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool.")
    print(f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}.")


    def client_fn(cid: str):
        """Creates a federated learning client"""
        return FL_Client(args, client_datasets, cid, log_progress=False)


    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds), strategy=strategy)
    
    print(history)

    return history



## Start federated training and inference
start_experiment(args=args, client_datasets=client_datasets, server_dataset=server_dataset, \
                 fraction_fit=1.0, min_fit_clients=1)