
# based SleepStagerChambon2018 class

class FeatureModel(nn.Module):
    def __init__(self, return_feats=True):
        super().__init__()
    
    def forward(self, x):
        """
        Forward pass
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

