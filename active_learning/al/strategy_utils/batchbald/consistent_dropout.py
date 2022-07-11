import torch
import types


class BayesianModule(torch.nn.Module):
    """A module that we can sample multiple times from given a single input batch.

    To be efficient, the module allows for a part of the forward pass to be deterministic.
    """

    k = None

    def __init__(self):
        super().__init__()

    # Returns B x n x output
    def forward(self, input_B: torch.Tensor, k: int):
        BayesianModule.k = k

        mc_input_BK = BayesianModule.mc_tensor(input_B, k)
        mc_output_BK = self.mc_forward_impl(mc_input_BK)
        mc_output_B_K = BayesianModule.unflatten_tensor(mc_output_BK, k)
        return mc_output_B_K

    def mc_forward_impl(self, mc_input_BK: torch.Tensor):
        return mc_input_BK

    @staticmethod
    def unflatten_tensor(input: torch.Tensor, k: int):
        input = input.view([-1, k] + list(input.shape[1:]))
        return input

    @staticmethod
    def flatten_tensor(mc_input: torch.Tensor):
        return mc_input.flatten(0, 1)

    @staticmethod
    def mc_tensor(input: torch.tensor, k: int):
        mc_shape = [input.shape[0], k] + list(input.shape[1:])
        return input.unsqueeze(1).expand(mc_shape).flatten(0, 1)


class _ConsistentMCDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )

        self.p = p
        # Sometimes one dropout layer can be used both for 3d tensors and 4d tensors (e.g. in XLNet)
        self.mask = {"3d": None, "4d": None}
        self.shape = None

    def extra_repr(self):
        return "p={}".format(self.p)

    def reset_mask(self):
        self.mask = {"3d": None, "4d": None}

    def train(self, mode=True):
        self.training = mode
        if not mode:
            self.reset_mask()

    def _create_mask(
        self,
        input: torch.Tensor = None,
        shape: torch.Size = None,
        device: torch.device = None,
    ):
        assert ((shape is not None) and (device is not None)) or (
            input is not None
        ), "Either input or (shape and device) should be specified!"
        if input is not None:
            device = input.device
            if len(input.shape) == 3:
                mask_shape = [1, 512, 4096]
            elif len(input.shape) == 4:
                mask_shape = [1, 16, 512, 512]
            else:
                raise ValueError("Invalid shape of input!")
        else:
            mask_shape = [1] + list(shape)
        mask = torch.empty(mask_shape, dtype=torch.bool, device=device).bernoulli_(
            self.p
        )
        return mask

    def forward(self, input: torch.Tensor):
        if self.p == 0.0:
            return input

        self.shape = input.shape
        num_dim = len(input.shape)
        if self.training:
            # Create a new mask on each call and for each batch element.
            mask = self._create_mask(input)
        else:
            key = f"{num_dim}d"
            if self.mask[key] is None:
                # print('recreating mask', self)
                # Recreate mask.
                self.mask[key] = self._create_mask(input)

            mask = self.mask[key]

        truncated_mask = self.truncate_mask(mask)
        mc_output = input.masked_fill(truncated_mask, 0) / (1 - self.p)
        return mc_output

    def truncate_mask(self, mask):
        slicers = [slice(0, 1)] + [slice(0, x) for x in self.shape[1:]]
        truncated_mask = mask[slicers]
        return truncated_mask


def make_dropouts_consistent(model):
    for mod in model.modules():
        if type(mod).__name__ == "Dropout":
            for method in [
                "__init__",
                "extra_repr",
                "reset_mask",
                "train",
                "_create_mask",
                "forward",
                "truncate_mask",
            ]:
                setattr(
                    mod,
                    method,
                    types.MethodType(getattr(_ConsistentMCDropout, method), mod),
                )
            mod.mask = None
