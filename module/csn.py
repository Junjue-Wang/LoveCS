import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
import itertools


class _CSNorm(Module):
    '''
    This implementation refers to TransNorm (https://github.com/thuml/TransNorm/blob/master/src/trans_norm.py).
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, affine_tar=False, track_running_stats=True):
        super(_CSNorm, self).__init__()
        self.affine_tar = affine_tar
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.source=True
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight_source = Parameter(torch.Tensor(num_features))
            self.bias_source = Parameter(torch.Tensor(num_features))
            if self.affine_tar:
                self.weight_target = Parameter(torch.Tensor(num_features))
                self.bias_target = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight_source', None)
            self.register_parameter('bias_source', None)
            if self.affine_tar:
                self.register_parameter('weight_target', None)
                self.register_parameter('bias_target', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))
            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)
        self.reset_parameters()

    def training_source(self):
        self.source = True

    def training_target(self):
        self.source = False


    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)
        if self.affine:
            self.weight_source.data.uniform_()
            self.bias_source.data.zero_()
            if self.affine_tar:
                self.weight_target.data.uniform_()
                self.bias_target.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def _load_from_state_dict_from_pretrained_model(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if 'source' in key or 'target' in key:
                key = key[:-7]
                print(key)
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

    def _load_from_state_dict_from_restored_model(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        r"""Copies restored parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)


    def forward(self, input):
        self._check_input_dim(input)
        if self.training :  ## train mode
            if self.source:
                ## 1. Domain Specific Mean and Variance.
                z_source = F.batch_norm(
                    input, self.running_mean_source, self.running_var_source, self.weight_source, self.bias_source,
                    self.training, self.momentum, self.eps)
                return z_source
            else:
                z_target = F.batch_norm(
                    input, self.running_mean_target, self.running_var_target, self.weight_source, self.bias_source,
                    self.training, self.momentum, self.eps)
                return z_target


        else:  ##test mode
            if self.source:
                z = F.batch_norm(
                    input, self.running_mean_source, self.running_var_source, self.weight_source, self.bias_source,
                    self.training, self.momentum, self.eps)
            else:
                z = F.batch_norm(
                    input, self.running_mean_target, self.running_var_target, self.weight_source, self.bias_source,
                    self.training, self.momentum, self.eps)
            return z

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, affine_tar={affine_tar} ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
        if prefix + 'running_mean_source' not in state_dict:
            self._load_from_state_dict_from_pretrained_model(
                state_dict, prefix, metadata, strict,
                missing_keys, unexpected_keys, error_msgs)
        else:
            self._load_from_state_dict_from_restored_model(
                state_dict, prefix, metadata, strict,
                missing_keys, unexpected_keys, error_msgs)

class CSNorm1d(_CSNorm):
    r"""Applies Cross-Sensor Normalization over a 2D or 3D input (a mini-batch of 1D inputs
    with additional channel dimension) as described in the paper
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.CSNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.CSNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class CSNorm2d(_CSNorm):
    r"""Applies Cross-Sensor Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.CSNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.CSNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CSNorm3d(_CSNorm):
    r"""Applies Cross-Sensor Normalization over a 5D input (a mini-batch of 3D inputs
    with additional channel dimension) as described in the paper
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, D, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    Examples::
        >>> # With Learnable Parameters
        >>> m = nn.CSNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.CSNorm3d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))



def change_csn(net:nn.Module, source=True):
    '''
    Args:
        net: during the training, the model changes as the input source/target changes
        source: True - > source data, False -> target data

    Returns:

    '''
    if source:
        for m in net.modules():
            if isinstance(m, _CSNorm):
                m.training_source()
    else:
        for m in net.modules():
            if isinstance(m, _CSNorm):
                m.training_target()


def copy_targetcsn(net:nn.Module):
    """

    Args:
        net: transferring the source weights for target norm weights

    Returns:

    """
    for m in net.modules():
        if isinstance(m, _CSNorm):
            if m.affine_tar:
                m.weight_target.data = m.weight_source.data.clone().detach()
                m.bias_target.data = m.bias_source.data.clone().detach()
            m.running_mean_target.data = m.running_mean_source.data.clone()
            m.running_var_target.data = m.running_var_source.data.clone()


def replace_bn_with_csn(module:nn.Module, affine_tar=False):
    '''
    Args:
        module: the original model with BNs

    Returns:
        module: the model with the CSNs
    '''
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = CSNorm2d(module.num_features, module.eps, module.momentum, module.affine, affine_tar=affine_tar)
        if module.affine:
            with torch.no_grad():
                module_output.weight_source = module.weight.data.clone().detach()
                module_output.bias_source = module.bias.data.clone().detach()
        if affine_tar:
            with torch.no_grad():
                module_output.weight_target.data = module.weight.data.clone().detach()
                module_output.bias_target.data = module.bias.data.clone().detach()
        module_output.running_mean_source = module.running_mean.data.clone()
        module_output.running_var_source = module.running_var.data.clone()
        module_output.running_mean_target = module.running_mean.data.clone()
        module_output.running_var_target = module.running_var.data.clone()
        module_output.num_batches_tracked = module.num_batches_tracked.data.clone()

    for name, child in module.named_children():
        module_output.add_module(name, replace_bn_with_csn(child))
    del module
    return module_output