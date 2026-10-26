"""Compatibility for an obsolete torchvision empty-tensor API used by pinned upstream."""
import torch
import torchvision.ops.misc


if not hasattr(torchvision.ops.misc, "_NewEmptyTensorOp"):
    class _NewEmptyTensorOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, shape):
            ctx.old_shape = tensor.shape
            return tensor.new_empty(shape)

        @staticmethod
        def backward(ctx, gradient):
            return gradient.new_empty(ctx.old_shape), None

    torchvision.ops.misc._NewEmptyTensorOp = _NewEmptyTensorOp
