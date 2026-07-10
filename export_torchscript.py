"""Export BiRefNet's main inference graph to TorchScript via tracing.

Image normalization and the final sigmoid are intentionally left out, since
those run outside the model in the deployment target. The exported module
takes a single already-normalized image batch and returns the model's raw
(pre-sigmoid) prediction map.

The traced graph calls torchvision::deform_conv2d (from the ASPPDeformable
decoder attention), a custom op registered by torchvision. Anywhere the
exported module is loaded, `import torchvision` must happen before
`torch.jit.load`, or loading fails with "Unknown builtin op:
torchvision::deform_conv2d".
"""

import argparse

import torch
import torch.nn as nn

from models.birefnet import BiRefNet


class BiRefNetJIT(nn.Module):
    def __init__(self, birefnet: BiRefNet):
        super().__init__()
        self.birefnet = birefnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # eval-mode forward returns a list; the last entry is the final prediction.
        return self.birefnet(x)[-1]


def export(repo_id: str, output_path: str, height: int, width: int, batch_size: int) -> None:
    model = BiRefNet.from_pretrained(repo_id)
    model.eval()

    wrapped = BiRefNetJIT(model)
    wrapped.eval()

    example = torch.randn(batch_size, 3, height, width)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example, check_trace=True)

    torch.jit.save(traced, output_path)
    print(f"Saved traced model to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export BiRefNet's inference graph to TorchScript.")
    parser.add_argument('--repo_id', type=str, default='zhengpeng7/BiRefNet', help='HuggingFace Hub repo id to load weights from')
    parser.add_argument('--output', type=str, default='birefnet.pt', help='Output path for the traced TorchScript module')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    export(args.repo_id, args.output, args.height, args.width, args.batch_size)
