import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, dim_model: int, module_list: nn.ModuleList):
        super().__init__()
        self.dim_model = dim_model

        self.module_list = module_list
        self.num_modules = len(module_list)

        self.axial_ratio = nn.Parameter(
            torch.tensor([[1], [0], [0]], dtype=torch.float32).repeat(1, dim_model)
        )
        self.coronal_ratio = nn.Parameter(
            torch.tensor([[0], [1], [0]], dtype=torch.float32).repeat(1, dim_model)
        )
        self.sagittal_ratio = nn.Parameter(
            torch.tensor([[0], [0], [1]], dtype=torch.float32).repeat(1, dim_model)
        )

        self.router_idx = -1

    def forward(self, x):
        assert self.router_idx in [0, 1, 2]

        axial_out = self.module_list[0](x)
        axial_out = axial_out * self.axial_ratio[self.router_idx]
        coronal_out = self.module_list[1](x)
        coronal_out = coronal_out * self.coronal_ratio[self.router_idx]
        sagittal_out = self.module_list[2](x)
        sagittal_out = sagittal_out * self.sagittal_ratio[self.router_idx]

        return axial_out + coronal_out + sagittal_out


class Dummy(nn.Module):  # NOTE: just for testing
    def __init__(self, dim_model):
        super().__init__()
        self.linear = nn.Linear(256, dim_model)

    def forward(self, x):
        return self.linear(x)


class Wrapper(nn.Module):  # NOTE: just for testing
    def __init__(self, router: Router):
        super().__init__()
        self.router = router


def set_router_idx(model: nn.Module, idx: int):
    if isinstance(model, Router):
        model.router_idx = idx
    for c in model.children():
        if isinstance(c, Router):
            c.router_idx = idx
        elif len(list(c.children())) > 0:
            set_router_idx(c, idx)


if __name__ == "__main__":
    dim_model = 512
    dummy_1 = Dummy(dim_model)
    dummy_2 = Dummy(dim_model)
    dummy_3 = Dummy(dim_model)
    dummy_list = nn.ModuleList([dummy_1, dummy_2, dummy_3])
    router = Router(dim_model, dummy_list)
    model = Wrapper(router)

    x = torch.rand(1, 256)
    set_router_idx(model, 1)
    y = router(x)
    print(x.shape, y.shape)
