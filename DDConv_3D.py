"""
Dynamic Deformable Convolution
"""
import torch
from torch import nn
import torch.nn.functional as F
from thop import *
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair,_triple
from torch.nn.parameter import Parameter

device = torch.device("cpu" )

class DDConv_3D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DDConv_3D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ConstantPad3d(padding,value=0)
        self.conv = SConv3D(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = SConv3D(inc, 3*kernel_size*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = SConv3D(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype).to(device)

        p = p.contiguous().permute(0, 2, 3, 4, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:2*N], 0, x.size(3)-1),torch.clamp(p[..., 2*N:], 0, x.size(4)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_lt[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_rb =  (1 + (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rb[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_rb[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lb[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_lb[..., 2*N:].type_as(p) - p[..., 2*N:]))
        g_rt = (1 + (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:2*N].type_as(p) - p[..., N:2*N])) *(1 + (q_rt[..., 2*N:].type_as(p) - p[..., 2*N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z= torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y),torch.flatten(p_n_z)], 0)
        p_n = p_n.view(1, 3*N, 1, 1,1).type(dtype)

        return p_n

    def _get_p_0(self, h, w,d, N, dtype):
        p_0_x, p_0_y,p_0_z = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            torch.arange(1, d*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0_z = torch.flatten(p_0_z).view(1, 1, h, w,d).repeat(1, N, 1, 1,1)
        p_0 = torch.cat([p_0_x, p_0_y,p_0_z], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w,d = offset.size(1)//3, offset.size(2), offset.size(3),offset.size(4)

        p_n = self._get_p_n(N, dtype).to(device)
        p_0 = self._get_p_0(h, w,d, N, dtype).to(device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w,d, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:2*N]+q[..., 2*N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1,-1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w,d, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w,d, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w,d*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks,d*ks)
        return x_offset

class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.sigmoid(x)


class SConv3D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=8, dropout_rate=0.2):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(SConv3D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool3d, output_size=(1, 1,1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

      
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        
        return F.conv3d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _,_ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)

            routing_weights = self._routing_fn(pooled_inputs)

            
            kernels = torch.sum(routing_weights[:, None, None, None, None,None] * self.weight, 0)
            
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)
if __name__ == "__main__":
    with torch.no_grad():
        input = torch.rand(2, 16, 112, 112, 80).to("cpu")
        model = DDConv_3D(16,16).to("cpu")

        out_result = model(input)
        print(out_result.shape)

        flops, params = profile(model, (input,))

        print("-" * 50)
        print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
        print('Params = ' + str(params / 1000 ** 2) + ' M')
