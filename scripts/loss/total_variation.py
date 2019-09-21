"""Total variation for pytorch


This function is based on Tensorflow implementation.
https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/ops/image_ops_impl.py#L3085-L3154
"""

import torch
from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss


class TotalVariationMSELoss(torch.nn.Module):
    def __init__(self, is_mean_reduction=False):
        super(TotalVariationMSELoss, self).__init__()
        self.total_variation_loss = TotalVariation(
            is_mean_reduction=is_mean_reduction)

    def forward(self, input, reference):
        input_tv = self.total_variation_loss(input)
        reference_tv = self.total_variation_loss(reference)
        return mse_loss(input_tv, reference_tv, reduction='mean')


class TotalVariationL1Loss(torch.nn.Module):
    def __init__(self, *, reduction='mean', is_mean_reduction=False):
        super(TotalVariationL1Loss, self).__init__()
        self._total_variation_loss = TotalVariation(
            is_mean_reduction=is_mean_reduction)
        self._reduction = reduction

    def forward(self, input_, reference):
        input_tv = self._total_variation_loss(input_)
        reference_tv = self._total_variation_loss(reference)
        return l1_loss(input_tv, reference_tv, reduction=self._reduction)


class TotalVariation(torch.nn.Module):
    """Calculate the total variation for one or batch tensor.

    The total variation is the sum of the absolute differences for neighboring
    pixel-values in the input images.

    Example:
    >>> import torch
    >>> loss_ = TotalVariation()

    >>> # Example for 2-dimensional tensor.
    >>> tensor_ = torch.arange(0, 2.5, 0.1, requires_grad=True).reshape(5, 5)
    >>> tensor_.shape
    torch.Size([5, 5])
    >>> loss_(tensor_)
    tensor(12., grad_fn=<AddBackward0>)

    >>> # Example for 3-dimensional tensor.
    >>> tensor_ = torch.arange(0, 2.5, 0.1, requires_grad=True).reshape(1, 5, 5)
    >>> tensor_.shape
    torch.Size([1, 5, 5])
    >>> loss_(tensor_)
    tensor(12., grad_fn=<AddBackward0>)

    >>> # Example for 4-dimensional tensor.
    >>> tensor_ = (
    ...     torch.arange(0, 10.0, 0.1, requires_grad=True).reshape(4, 1, 5, 5)
    ... )
    >>> tensor_.shape
    torch.Size([4, 1, 5, 5])
    >>> loss_(tensor_)
    tensor([12.0000, 12.0000, 12.0000, 12.0000], grad_fn=<AddBackward0>)

    >>> # Example for 4-dimensional tensor with `is_mean_reduction=True`.
    >>> loss_ = TotalVariation(is_mean_reduction=True)
    >>> tensor_ = (
    ...     torch.arange(0, 10.0, 0.1, requires_grad=True).reshape(4, 1, 5, 5)
    ... )
    >>> loss_(tensor_)
    tensor([0.6000, 0.6000, 0.6000, 0.6000], grad_fn=<AddBackward0>)
    """

    def __init__(self, *, is_mean_reduction: bool = False) -> None:
        """Constructor.

        Args:
            is_mean_reduction (bool, optional):
                When `is_mean_reduction` is True, the sum of the output will be
                divided by the number of elements those used
                for total variation calculation. Defaults to False.
        """
        super(TotalVariation, self).__init__()
        self._is_mean = is_mean_reduction

    def forward(self, tensor_: Tensor) -> Tensor:
        return self._total_variation(tensor_)

    def _total_variation(self, tensor_: Tensor) -> Tensor:
        """Calculate total variation.

        Args:
            tensor_ (Tensor): input tensor must be the any following shapes:
                - 2-dimensional: [height, width]
                - 3-dimensional: [channel, height, width]
                - 4-dimensional: [batch, channel, height, width]

        Raises:
            ValueError: Input tensor is not either 2, 3 or 4-dimensional.

        Returns:
            Tensor: the output tensor shape depends on the size of the input.
                - Input tensor was 2 or 3 dimensional
                    return tensor as a scalar
                - Input tensor was 4 dimensional
                    return tensor as an array
        """
        ndims_ = tensor_.dim()

        if ndims_ == 2:
            y_diff = tensor_[1:, :] - tensor_[:-1, :]
            x_diff = tensor_[:, 1:] - tensor_[:, :-1]
        elif ndims_ == 3:
            y_diff = tensor_[:, 1:, :] - tensor_[:, :-1, :]
            x_diff = tensor_[:, :, 1:] - tensor_[:, :, :-1]
        elif ndims_ == 4:
            y_diff = tensor_[:, :, 1:, :] - tensor_[:, :, :-1, :]
            x_diff = tensor_[:, :, :, 1:] - tensor_[:, :, :, :-1]
        else:
            raise ValueError(
                'Input tensor must be either 2, 3 or 4-dimensional.')

        sum_axis = tuple({abs(x) for x in range(ndims_ - 3, ndims_)})
        y_denominator = (
            y_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )
        x_denominator = (
            x_diff.shape[sum_axis[0]::].numel() if self._is_mean else 1
        )

        return (
            torch.sum(torch.abs(y_diff), dim=sum_axis) / y_denominator
            + torch.sum(torch.abs(x_diff), dim=sum_axis) / x_denominator
        )


if __name__ == '__main__':
    import doctest
    doctest.testmod()
