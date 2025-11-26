import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, dim_model: int, module_list: nn.ModuleList):
        super().__init__()
        self.dim_model = dim_model

        self.module_list = module_list
        self.num_modules = len(module_list)

        self.ratio = nn.Parameter(
            torch.tensor([[1], [0], [0]], dtype=torch.float32).repeat(1, dim_model)
        )

    def forward(self, x):
        axial_out = self.module_list[0](x)
        axial_out = axial_out * self.ratio[0]
        coronal_out = self.module_list[1](x)
        coronal_out = coronal_out * self.ratio[1]
        sagittal_out = self.module_list[2](x)
        sagittal_out = sagittal_out * self.ratio[2]

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


if __name__ == "__main__":
    dim_model = 512
    dummy_1 = Dummy(dim_model)
    dummy_2 = Dummy(dim_model)
    dummy_3 = Dummy(dim_model)
    dummy_list = nn.ModuleList([dummy_1, dummy_2, dummy_3])
    router = Router(dim_model, dummy_list)
    model = Wrapper(router)

    x = torch.rand(1, 256)
    y = router(x)
    print(x.shape, y.shape)
