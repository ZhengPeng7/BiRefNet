"""This module adds ONNX conversion of `deform_conv2d`.

This module implements Deformable Convolution v2,
described in a paper, `Deformable ConvNets v2: More Deformable, Better Results
<https://arxiv.org/abs/1811.11168>`, using ONNX operators.
The implementation is straightforward, but may not be very efficient.

This exporter requires opset version 12 to support the following operators:
  - Clip:
    It can accept tensor(int64) from version 12.
  - GatherND:
    It can support batch_dims from version 12.
"""

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx import symbolic_helper as sym_help
try:
    from torch.onnx._type_utils import JitScalarType
except ImportError:
    JitScalarType = None

__all__ = ["register_deform_conv2d_onnx_op"]

onnx_opset_version = 12


def add(g, lhs, rhs):
    return g.op("Add", lhs, rhs)


def sub(g, lhs, rhs):
    return g.op("Sub", lhs, rhs)


def mul(g, lhs, rhs):
    return g.op("Mul", lhs, rhs)


def reshape(g, x, shape):
    if isinstance(shape, list):
        shape = tensor(g, shape, dtype=torch.int64)
    return g.op("Reshape", x, shape)


def slice(g, x, axes, starts, ends, *, steps=None):
    axes = tensor(g, axes, dtype=torch.int64)
    starts = tensor(g, starts, dtype=torch.int64)
    ends = tensor(g, ends, dtype=torch.int64)
    if steps is not None:
        steps = tensor(g, steps, dtype=torch.int64)
        return g.op("Slice", x, starts, ends, axes, steps)
    else:
        return g.op("Slice", x, starts, ends, axes)


def unsqueeze(g, input, dims):
    return sym_help._unsqueeze_helper(g, input, axes_i=dims)

def get_tensor_dim_size(tensor, dim):
    
    tensor_dim_size = sym_help._get_tensor_dim_size(tensor, dim)
    if tensor_dim_size == None and (dim == 2 or dim == 3):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()

        tensor_dim_size = x_strides[2] if dim == 3 else x_strides[1] // x_strides[2]
    elif tensor_dim_size == None and (dim == 0):
        import typing
        from torch import _C

        x_type = typing.cast(_C.TensorType, tensor.type())
        x_strides = x_type.strides()
        tensor_dim_size = x_strides[3]

    return tensor_dim_size




def tensor(g, value, dtype):
    return g.op("Constant", value_t=torch.tensor(value, dtype=dtype))


def calculate_p_0(dcn_params):
    """
    Calculate p_0 value in equation (1) in the paper.

    Args:
        dcn_params: parameters for deform_conv2d.

    Returns:
        torch.Tensor[1, 1, kernel_area_size, 2, out_h, out_w]
    """
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    stride_h = dcn_params["stride_h"]
    stride_w = dcn_params["stride_w"]
    K = dcn_params["kernel_area_size"]
    additional_pad_h = dcn_params["additional_pad_h"]
    additional_pad_w = dcn_params["additional_pad_w"]

    p_0_y, p_0_x = torch.meshgrid(torch.arange(0, h * stride_h, stride_h),
                                  torch.arange(0, w * stride_w, stride_w))
    p_0_y = p_0_y.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_y += additional_pad_h
    p_0_x = p_0_x.view(1, 1, 1, 1, h, w).repeat(1, 1, K, 1, 1, 1)
    p_0_x += additional_pad_w
    return torch.cat([p_0_y, p_0_x], dim=3)


def calculate_p_k(dcn_params):
    """
    Calculate p_k value in equation (1) in the paper.

    Args:
        dcn_params: parameters for deform_conv2d.

    Returns:
        torch.Tensor[1, 1, kernel_area_size, 2, 1, 1]
    """
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]
    dilation_h = dcn_params["dilation_h"]
    dilation_w = dcn_params["dilation_w"]
    K = dcn_params["kernel_area_size"]

    p_k_y, p_k_x = torch.meshgrid(
        torch.arange(0, kernel_h * dilation_h, step=dilation_h),
        torch.arange(0, kernel_w * dilation_w, step=dilation_w),
    )
    p_k_y = p_k_y.reshape(1, 1, K, 1, 1, 1)
    p_k_x = p_k_x.reshape(1, 1, K, 1, 1, 1)
    return torch.cat([p_k_y, p_k_x], dim=3)


def calculate_p(g, dcn_params, offset):
    """
    Calculate p_0 + p_k + Delta(p_k) in equation (1) in the paper.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        offset: Delta(p_k) in the paper.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        The shape is (b, group, K, 2, out_h, out_w).
    """
    b = dcn_params["batch"]
    K = dcn_params["kernel_area_size"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    group = dcn_params["n_offset_grps"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    offset = reshape(g, offset, [b, group, K, 2, h, w])

    p_0 = calculate_p_0(dcn_params)
    p_k = calculate_p_k(dcn_params)
    p = p_0 + p_k
    p = add(g, tensor(g, p.tolist(), dtype=offset_dtype), offset)
    # => p.shape is (b, group, K, 2, h, w)
    return p


def calculate_p_floor(g, dcn_params, p):
    """
    Calculate floor of p.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        p: Coords for sampling points of DCN.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        The shape is (b, group, K, 2, out_h, out_w).
        Note that the data type is not integer but float.
    """
    p_floor = g.op("Floor", p)
    return p_floor


def calculate_p_tlbr(g, dcn_params, p_floor):
    """
    Calculate floor and ceil of p.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        p_floor: Floored coords for sampling points of DCN.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        A dict, {"t": p_t, "l", p_l, "b": p_b, "r": p_r}, which contains
        "t"op, "l"eft, "b"ottom, and "r"ight coordinates around p.
        The shape of p_t, ..., p_r is (b, group, K, 1, out_h, out_w).
    """
    h = dcn_params["in_h"]
    w = dcn_params["in_w"]
    index_dtype_onnx = dcn_params["index_dtype_onnx"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    p_floor = g.op("Cast", p_floor, to_i=index_dtype_onnx)
    one = tensor(g, 1, dtype=index_dtype_pytorch)

    p_t = slice(g, p_floor, [3], [0], [1])
    p_l = slice(g, p_floor, [3], [1], [2])
    p_b = add(g, p_t, one)
    p_r = add(g, p_l, one)

    # Clip out-of-bounds coords.
    # Clipped coords point to padding area, which is filled with 0.
    p_t = g.op("Clip", p_t, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, h - 1, dtype=index_dtype_pytorch))
    p_l = g.op("Clip", p_l, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, w - 1, dtype=index_dtype_pytorch))
    p_b = g.op("Clip", p_b, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, h - 1, dtype=index_dtype_pytorch))
    p_r = g.op("Clip", p_r, tensor(g, 0, dtype=index_dtype_pytorch),
               tensor(g, w - 1, dtype=index_dtype_pytorch))
    return {
        "t": p_t,
        "l": p_l,
        "b": p_b,
        "r": p_r,
    }


def calculate_weight(g, dcn_params, p, p_floor):
    """
    Calculate weight value for bilinear interpolation.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        p: Coords for sampling points.
            The shape is (b, group, K, 2, out_h, out_w).
        p_floor: Floored coords for sampling points.
            The shape is (b, group, K, 2, out_h, out_w).

    Returns:
        A dict, {"tl": weight_tl, "br": weight_br, ..., "tr": weight_tr},
        which contains weights for "t"op-"l"eft, "b"ottom-"r"ight, ....
        The shape of weight_tl is (b, group, 1, K, out_h, out_w).
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    offset_dtype = dcn_params["offset_dtype_pytorch"]

    one = tensor(g, 1, dtype=offset_dtype)

    diff = sub(g, p, p_floor)
    diff_y = slice(g, diff, [3], [0], [1])
    diff_x = slice(g, diff, [3], [1], [2])
    diff_y_inv = sub(g, one, diff_y)
    diff_x_inv = sub(g, one, diff_x)

    # bilinear kernel (b, group, K, 1, h, w)
    # (1 - (p_x - p_l)) * (1 - (p_y - p_t))
    weight_tl = mul(g, diff_x_inv, diff_y_inv)
    # (p_x - p_l) * (p_y - p_t)
    weight_br = mul(g, diff_x, diff_y)
    # (1 - (p_x - p_l)) * (p_y - p_t)
    weight_bl = mul(g, diff_x_inv, diff_y)
    # (p_x - p_l) * (1 - (p_y - p_t))
    weight_tr = mul(g, diff_x, diff_y_inv)

    weights = {
        "tl": weight_tl,
        "br": weight_br,
        "bl": weight_bl,
        "tr": weight_tr,
    }
    weights = {
        key: reshape(g, weight, [b, group, 1, K, h, w])
        for key, weight in weights.items()
    }
    return weights


def reshape_input_for_gather_elements(g, dcn_params, input):
    """
    Reshape input for gather_elements function.

    Even if no padding is specified, 1 padding is always added
    to ensure that out-of-bounds index can be handled correctly.

    This function also transpose input tensor, so that "GatherND"
    can easily gather all data in a channel.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, in_ch, in_h, in_w)

    Returns:
        The shape is (b, group, ch_per_group, in_h, in_w).
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    in_h = dcn_params["in_h"]
    in_w = dcn_params["in_w"]
    pad_h = dcn_params["padding_h"]
    pad_w = dcn_params["padding_w"]
    additional_pad_h = dcn_params["additional_pad_h"]
    additional_pad_w = dcn_params["additional_pad_w"]

    pad_size = [
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
        0,
        0,
        (pad_h + additional_pad_h),
        (pad_w + additional_pad_w),
    ]
    pad = tensor(g, pad_size, dtype=torch.int64)
    input = g.op("Pad", input, pad, mode_s="constant")
    input = reshape(g, input, [b, group, ch, in_h, in_w])
    return input


def gather_elements(g, dcn_params, input, p_y, p_x):
    """
    Gather elements specified by p_y and p_x using GatherElements operator.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_y: y coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).
        p_x: x coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    in_h = dcn_params["in_h"]
    in_w = dcn_params["in_w"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]
    index_dtype_pytorch = dcn_params["index_dtype_pytorch"]

    p_y = reshape(g, p_y, [b, group, 1, K * out_h * out_w])
    p_x = reshape(g, p_x, [b, group, 1, K * out_h * out_w])
    p_y = g.op("Mul", p_y, tensor(g, in_w, dtype=index_dtype_pytorch))
    index = g.op("Add", p_y, p_x)
    shape = [b, group, ch, K * out_h * out_w]
    index = g.op("Expand", index, tensor(g, shape, dtype=torch.int64))

    input = reshape(g, input, [b, group, ch, in_h * in_w])

    v = g.op("GatherElements", input, index, axis_i=3)
    # => v.shape is (b, group, ch_per_group, K * out_h * out_w)
    v = reshape(g, v, [b, group, ch, K, out_h, out_w])

    return v


def gather_nd(g, dcn_params, input, p_y, p_x):
    """
    Gather elements specified by p_y and p_x using GatherND.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_y: y coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).
        p_x: x coordinates of sampling points.
            The shape is (b, group, K, 1, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    ch = dcn_params["in_ch_per_group"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]

    p_y = reshape(g, p_y, [b, group, K * out_h * out_w, 1])
    p_x = reshape(g, p_x, [b, group, K * out_h * out_w, 1])
    index = g.op("Concat", p_y, p_x, axis_i=3)
    # => index.shape is (b, group, K * out_h * out_w, 2)

    input = g.op("Transpose", input, perm_i=[0, 1, 3, 4, 2])
    # => input.shape is (b, group, in_h, in_w, ch_per_group)
    v = g.op("GatherND", input, index, batch_dims_i=2)
    # => v.shape is (b, group, K * out_h * out_w, ch)
    if dcn_params["option"]["enable_openvino_patch"]:
        # OpenVINO 2021.4 has a bug related to shape of the output of GatherND.
        v = reshape(g, v, [b, group, K * out_h * out_w, ch])
    v = g.op("Transpose", v, perm_i=[0, 1, 3, 2])
    v = reshape(g, v, [b, group, ch, K, out_h, out_w])
    return v


def gather_elements_tlbr(g, dcn_params, input, p_tlbr):
    """
    Gather elements specified by p_tlbr.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        input: input tensor.
            The shape is (b, group, ch_per_group, in_h, in_w).
        p_tlbr: A dict, {"t": p_t, "l", p_l, "b": p_b, "r": p_r},
            which contains "t"op, "l"eft, "b"ottom, and "r"ight
            coordinates around p.
            The shape of p_t, ..., p_r is (b, group, K, 1, out_h, out_w).

    Returns:
        A dict, {"tl": v_tl, "br": v_br, ..., "tr": v_tr}, which contains
        gathred elements.
        The shape of v_tl is (b, group, ch_per_group, K, out_h, out_w).
    """
    tlbr = ["tl", "br", "bl", "tr"]
    v_tlbr = {}
    for key in tlbr:
        key_y = key[0]  # "t" or "b"
        key_x = key[1]  # "l" or "r"
        p_y = p_tlbr[key_y]
        p_x = p_tlbr[key_x]
        if dcn_params["option"]["use_gathernd"]:
            v = gather_nd(g, dcn_params, input, p_y, p_x)
        else:
            v = gather_elements(g, dcn_params, input, p_y, p_x)
        v_tlbr[key] = v
    return v_tlbr


def calculate_weighted_sum(g, dcn_params, v_tlbr, weight_tlbr):
    """
    Calculate sum of weighted tensors.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        v_tlbr: a dict, {"tl": v_tl, "br": v_br, ..., "tr": v_tr}, which
            contains gathred elements.
            The shape of v_tl is (b, group, ch_per_group, K, out_h, out_w).
        weight_tlbr: a dict, {"tl": weight_tl, "br": weight_br, ...},
            which contains weights for "t"op-"l"eft, "b"ottom-"r"ight, ....
            The shape of weight_tl is (b, group, 1, K, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).
    """
    weighted_v_list = [mul(g, weight_tlbr[key], v_tlbr[key]) for key in v_tlbr]
    v = g.op("Sum", *weighted_v_list)
    return v


def apply_mask(g, dcn_params, v, mask):
    """
    Apply mask tensor.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        v: input tensor.
            The shape is (b, group, ch_per_group, K, out_h, out_w).
        mask: mask tensor.
            The shape is (b, group * K, out_h, out_w).

    Returns:
        The shape is (b, group, ch_per_group, K, out_h, out_w).
    """
    b = dcn_params["batch"]
    group = dcn_params["n_offset_grps"]
    out_h = dcn_params["out_h"]
    out_w = dcn_params["out_w"]
    K = dcn_params["kernel_area_size"]

    mask = reshape(g, mask, [b, group, 1, K, out_h, out_w])
    v = mul(g, v, mask)
    return v


def reshape_v_for_conv(g, dcn_params, v):
    """
    Reshape v for convolution.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        v: a reshaped tensor.
            The shape is (b, group, ch_per_group, K, out_h, out_w).

    Returns:
        The shape is (b, in_ch, out_h * kernel_h, out_w * kernel_w).
    """
    b = dcn_params["batch"]
    h = dcn_params["out_h"]
    w = dcn_params["out_w"]
    ch = dcn_params["in_ch"]
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]

    v = reshape(g, v, [b, ch, kernel_h, kernel_w, h, w])
    v = g.op("Transpose", v, perm_i=[0, 1, 4, 2, 5, 3])
    return reshape(g, v, [b, ch, h * kernel_h, w * kernel_w])


def apply_conv(g, dcn_params, v, weight):
    """
    Apply convolution.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        v: input tensor.
            The shape is (b, in_ch, out_h * kernel_h, out_w * kernel_w).
        weight: weight for convolution.
            The shape is (out_ch, ch_per_group, kernel_h, kernel_w).

    Returns:
        The shape is (b, out_ch, out_h, out_w).
    """
    weight_groups = dcn_params["n_weight_grps"]
    kernel_h = dcn_params["kernel_h"]
    kernel_w = dcn_params["kernel_w"]

    v = g.op("Conv",
             v,
             weight,
             group_i=weight_groups,
             kernel_shape_i=[kernel_h, kernel_w],
             strides_i=[kernel_h, kernel_w])
    return v


def apply_bias(g, dcn_params, v, bias):
    """
    Apply bias parameter.

    Args:
        g: graph object.
        dcn_params: parameters for deform_conv2d.
        v: input tensor.
            The shape is (b, out_ch, out_h, out_w).
        bias: bias tensor.
            The shape is (out_ch,).

    Returns:
        The shape is (b, out_ch, out_h, out_w).
    """
    bias = unsqueeze(g, bias, [0, 2, 3])
    v = add(g, v, bias)
    return v


def create_dcn_params(input, weight, offset, mask, bias, stride_h, stride_w,
                      pad_h, pad_w, dilation_h, dilation_w, n_weight_grps,
                      n_offset_grps, use_mask, option):
    """
    Manage parameters for DeformConv2d.
    """
    additional_pad_h = additional_pad_w = 0
    if pad_h == 0:
        additional_pad_h = 1
    if pad_w == 0:
        additional_pad_w = 1

    batch = get_tensor_dim_size(input, 0)
    in_ch = get_tensor_dim_size(input, 1)
    in_h = get_tensor_dim_size(input, 2) + 2 * (pad_h + additional_pad_h)
    in_w = get_tensor_dim_size(input, 3) + 2 * (pad_w + additional_pad_w)
    in_ch_per_group = in_ch // n_offset_grps

    out_ch = get_tensor_dim_size(weight, 0)
    kernel_h = get_tensor_dim_size(weight, 2)
    kernel_w = get_tensor_dim_size(weight, 3)
    kernel_area_size = kernel_h * kernel_w

    out_h = get_tensor_dim_size(offset, 2)
    out_w = get_tensor_dim_size(offset, 3)

    if JitScalarType is not None and hasattr(JitScalarType, "from_value"):
        # 2.0 and later
        scalar_type = JitScalarType.from_value(offset)
        offset_dtype_onnx = scalar_type.onnx_type()
        offset_dtype_pytorch = scalar_type.dtype()

        scalar_type = JitScalarType.from_dtype(torch.int64)
        index_dtype_onnx = scalar_type.onnx_type()
        index_dtype_pytorch = scalar_type.dtype()
    else:
        offset_dtype = sym_help._try_get_scalar_type(offset)
        offset_dtype_onnx = sym_help.cast_pytorch_to_onnx[offset_dtype]
        dtype_idx = sym_help.scalar_type_to_onnx.index(offset_dtype_onnx)
        offset_dtype_pytorch = sym_help.scalar_type_to_pytorch_type[dtype_idx]

        index_dtype = "Long"
        index_dtype_onnx = sym_help.cast_pytorch_to_onnx[index_dtype]
        dtype_idx = sym_help.scalar_type_to_onnx.index(index_dtype_onnx)
        index_dtype_pytorch = sym_help.scalar_type_to_pytorch_type[dtype_idx]

    dcn_params = {
        # batch and kernel
        "batch": batch,
        "kernel_h": kernel_h,
        "kernel_w": kernel_w,
        "kernel_area_size": kernel_area_size,

        # input size
        "in_ch": in_ch,
        "in_ch_per_group": in_ch_per_group,
        "in_h": in_h,
        "in_w": in_w,

        # output size
        "out_ch": out_ch,
        "out_h": out_h,
        "out_w": out_w,

        # other parameters
        "stride_h": stride_h,
        "stride_w": stride_w,
        "dilation_h": dilation_h,
        "dilation_w": dilation_w,
        "n_offset_grps": n_offset_grps,
        "n_weight_grps": n_weight_grps,

        # offset data type
        "offset_dtype_onnx": offset_dtype_onnx,
        "offset_dtype_pytorch": offset_dtype_pytorch,

        # index data type
        "index_dtype_onnx": index_dtype_onnx,
        "index_dtype_pytorch": index_dtype_pytorch,

        # padding
        "padding_h": pad_h,
        "padding_w": pad_w,
        "additional_pad_h": additional_pad_h,
        "additional_pad_w": additional_pad_w,

        "option": option,
    }
    return dcn_params


def deform_conv2d_func(use_gathernd, enable_openvino_patch):
    @sym_help.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i",
                         "i", "i", "b")
    def deform_conv2d(g, input, weight, offset, mask, bias, stride_h, stride_w,
                      pad_h, pad_w, dilation_h, dilation_w, n_weight_grps,
                      n_offset_grps, use_mask):
        option = {
            "use_gathernd": use_gathernd,
            "enable_openvino_patch": enable_openvino_patch,
        }
        dcn_params = create_dcn_params(input, weight, offset, mask, bias,
                                       stride_h, stride_w, pad_h, pad_w,
                                       dilation_h, dilation_w, n_weight_grps,
                                       n_offset_grps, use_mask, option)

        p = calculate_p(g, dcn_params, offset)
        p_floor = calculate_p_floor(g, dcn_params, p)
        p_tlbr = calculate_p_tlbr(g, dcn_params, p_floor)
        weight_tlbr = calculate_weight(g, dcn_params, p, p_floor)

        input = reshape_input_for_gather_elements(g, dcn_params, input)
        v_tlbr = gather_elements_tlbr(g, dcn_params, input, p_tlbr)

        v = calculate_weighted_sum(g, dcn_params, v_tlbr, weight_tlbr)

        if use_mask:
            v = apply_mask(g, dcn_params, v, mask)

        v = reshape_v_for_conv(g, dcn_params, v)
        v = apply_conv(g, dcn_params, v, weight)
        v = apply_bias(g, dcn_params, v, bias)
        return v

    return deform_conv2d


def register_deform_conv2d_onnx_op(use_gathernd=True,
                                   enable_openvino_patch=False):
    """
    Register ONNX operator for torchvision::deform_conv2d.

    Args:
        use_gathernd: If True, use GatherND. Otherwise use GatherElements.
        enable_openvino_patch: If True, enable patch for OpenVINO.
            Otherwise, disable it.
    """
    register_custom_op_symbolic(
        'torchvision::deform_conv2d',
        deform_conv2d_func(use_gathernd, enable_openvino_patch),
        onnx_opset_version)
