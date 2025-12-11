import torch
import torch.nn.functional as F

def unfold_naive(x, kernel_size, stride=1, padding=0):
    """
    Naive Python implementation of torch.nn.functional.unfold.
    x: (B, C, H, W)
    Returns: (B, C*K*K, P)
    """
    B, C, H, W = x.shape
    K_h, K_w = kernel_size

    # Pad input manually
    x_pad = torch.nn.functional.pad(
        x, (padding, padding, padding, padding)
    )
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    # Compute output spatial size
    H_out = (H_pad - K_h) // stride + 1
    W_out = (W_pad - K_w) // stride + 1
    P = H_out * W_out

    # Allocate output tensor
    patches = torch.zeros((B, C * K_h * K_w, P), dtype=x.dtype)

    # Fill patches
    p = 0
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            w_start = w * stride

            patch = x_pad[:, :, h_start:h_start + K_h, w_start:w_start + K_w]
            patches[:, :, p] = patch.reshape(B, -1)
            p += 1

    return patches

class Conv2DFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_batch, kernel, stride=1, padding=1):
        # Save inputs for backward
        ctx.stride = stride
        ctx.padding = padding
        ctx.save_for_backward(input_batch, kernel)

        # Unfold input
        B, C_in, H_in, W_in = input_batch.shape
        C_out, _, K_h, K_w = kernel.shape

        U = F.unfold(input_batch, kernel_size=(K_h, K_w),
                     padding=padding, stride=stride)
        # U: (B, C_in*K_h*K_w, P)

        # Reshape for batched matrix multiply
        U_t = U.transpose(1, 2)        # (B, P, K)
        W = kernel.reshape(C_out, -1).t()   # (K, C_out)

        Y = torch.matmul(U_t, W)       # (B, P, C_out)

        # reshape to output
        H_out = (H_in + 2*padding - K_h) // stride + 1
        W_out = (W_in + 2*padding - K_w) // stride + 1
        Y = Y.transpose(1, 2).reshape(B, C_out, H_out, W_out)

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        input_batch, kernel = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding
        B, C_in, H_in, W_in = input_batch.shape
        C_out, _, K_h, K_w = kernel.shape

        # dY shape: (B, C_out, H_out, W_out)
        dY = grad_output

        # reshape for matmul
        dY = dY.reshape(B, C_out, -1).transpose(1, 2)  # (B, P, C_out)

        # kernel flattened
        W = kernel.reshape(C_out, -1).t()  # (K, C_out)

        # dU = dY * W^T
        dU = torch.matmul(dY, W.t())  # (B, P, K)
        dU = dU.transpose(1, 2)       # (B, K, P)

        # fold to get dX
        dX = F.fold(
            dU,
            output_size=(H_in, W_in),
            kernel_size=(K_h, K_w),
            padding=padding,
            stride=stride
        )

        # compute dW
        U = F.unfold(input_batch, kernel_size=(K_h, K_w),
                     padding=padding, stride=stride)  # (B, K, P)
        U_t = U.transpose(1, 2)  # (B, P, K)

        # dW = sum_b U^T @ dY
        dY_flat = grad_output.reshape(B, C_out, -1).transpose(1, 2)  # (B, P, C_out)
        dW = torch.matmul(U_t.transpose(1, 2), dY_flat)  # (B, K, C_out)
        dW = dW.sum(dim=0).t().reshape(C_out, C_in, K_h, K_w)

        return dX, dW, None, None


