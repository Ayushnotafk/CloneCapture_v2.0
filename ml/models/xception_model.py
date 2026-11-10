import torch
import torch.nn as nn
import timm


class XceptionModel(nn.Module):

    def __init__(self, num_classes=2, pretrained=True):

        super().__init__()

        self.model = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = XceptionModel(pretrained=True).to(device)

    sample = torch.randn(2, 3, 224, 224).to(device)

    output = model(sample)

    print("Device:", device)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", total_params)