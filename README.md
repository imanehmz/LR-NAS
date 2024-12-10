# BioNAS
NAS for bio-inspired learning rules, we incorporate different feedback alignment techniques from [Biotorch](https://github.com/jsalbert/biotorch)

# Reproducing experiments

To search on cifar10, run 
```
python train_search.py --gpu=0 --batch_size=BATCH_SIZE --epochs=50
```

To train the resulting architecture which can be found in the log file after searching, run:

```
python train.py PARAMS
```

## Adding new layers with new learning rules:
To add a Convolutional layers trained with hebbian learning, we make changes in the `operations.py` file, we add the layer declaration and name it `Hebbian_conv`

In the `genotypes.py` file, we add `Hebbian_conv` layer to the search space.

Code: 

```
class HebbianConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, mode="hebbian", affine=True):
        super(HebbianConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.mode = mode

        self.hebbian_weights = torch.zeros_like(self.conv.weight, requires_grad=False)

        # Scaling factor for Hebbian update
        self.hebbian_scale = 1e-4

    def forward(self, x):
        batch_size, _, height, width = x.shape

        if self.mode == "hebbian":
            with torch.no_grad():
                if self.hebbian_weights.device != x.device:
                    self.hebbian_weights = self.hebbian_weights.to(x.device)

                input_unfolded = nn.functional.unfold(
                    x,
                    kernel_size=self.conv.kernel_size[0],  # Assuming square kernels
                    padding=self.conv.padding[0],
                    stride=self.conv.stride[0],
                )  # Shape: (batch_size, in_channels * kernel_size^2, out_height * out_width)

                output = self.conv(x)  # Shape: (batch_size, out_channels, out_height, out_width)

                # Flatten
                out_height, out_width = output.shape[2:]
                output_flat = output.permute(0, 2, 3, 1).reshape(-1, self.conv.out_channels)
                # Shape: (batch_size * out_height * out_width, out_channels)

                input_flat = input_unfolded.permute(0, 2, 1).reshape(-1, input_unfolded.shape[1])
                # Shape: (batch_size * out_height * out_width, in_channels * kernel_size^2)

                # Normalize input and output to prevent exploding updates
                input_flat = nn.functional.normalize(input_flat, dim=1)
                output_flat = nn.functional.normalize(output_flat, dim=1)

                hebbian_update = torch.einsum("bi,bo->io", input_flat, output_flat)
                # Shape: (in_channels * kernel_size^2, out_channels)

                hebbian_update = hebbian_update.view_as(self.hebbian_weights)

                hebbian_update = hebbian_update.to(self.hebbian_weights.device)

                hebbian_update = self.hebbian_scale * hebbian_update
                self.hebbian_weights.add_(hebbian_update)

                # Regularize weights to prevent explosion
                self.hebbian_weights.data.clamp_(-1.0, 1.0)

                self.conv.weight.data.copy_(self.hebbian_weights)

        x = self.conv(x)
        x = self.bn(x)
        return x


OPS['hebbian_conv'] = lambda C, stride, affine: HebbianConv(C, C, kernel_size=3, stride=stride, padding=1, affine=affine, mode='hebbian')


```


The baseline code has been borrowed from [DARTS](https://github.com/quark0/darts) and [EG-NAS](https://github.com/caicaicheng/EG-NAS), by changing the operations.
