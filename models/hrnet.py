import numbers
import collections
import logging
import torch
from torch import nn
from torch.nn import functional as F
from models.mobilenet_base import _make_divisible
from models.mobilenet_base import ConvBNReLU
from models.mobilenet_base import get_active_fn
from models.mobilenet_base import InvertedResidualChannels, InvertedResidualChannelsFused
from mmseg.utils import resize
import json
from utils import distributed as udist
from ops import Voxelization, nms_cuda
import numpy as np
from model.anchors import Anchors, anchor_target, anchors2bboxes
from point_utils import limit_period
import uni3dm.uni3d as models

__all__ = ['HighResolutionNet']

# from uni3dm import create_uni3d
checkpoint_kwparams = None
# checkpoint_kwparams = json.load(open('checkpoint.json'))


class InvertedResidual(InvertedResidualChannels):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 expand_ratio,
                 kernel_sizes,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 **kwargs):

        def _expand_ratio_to_hiddens(expand_ratio):
            if isinstance(expand_ratio, list):
                assert len(expand_ratio) == len(kernel_sizes)
                expand = True
            elif isinstance(expand_ratio, numbers.Number):
                expand = expand_ratio != 1
                expand_ratio = [expand_ratio for _ in kernel_sizes]
            else:
                raise ValueError(
                    'Unknown expand_ratio type: {}'.format(expand_ratio))
            hidden_dims = [int(round(inp * e)) for e in expand_ratio]
            return hidden_dims, expand

        hidden_dims, expand = _expand_ratio_to_hiddens(expand_ratio)
        if checkpoint_kwparams:
            assert oup == checkpoint_kwparams[0][0]
            if udist.is_master():
                logging.info('loading: {} -> {}, {} -> {}'.format(
                    hidden_dims, checkpoint_kwparams[0][4], kernel_sizes, checkpoint_kwparams[0][3]))
            hidden_dims = checkpoint_kwparams[0][4]
            kernel_sizes = checkpoint_kwparams[0][3]
            checkpoint_kwparams.pop(0)

        super(InvertedResidual,
              self).__init__(inp,
                             oup,
                             stride,
                             hidden_dims,
                             kernel_sizes,
                             expand,
                             active_fn=active_fn,
                             batch_norm_kwargs=batch_norm_kwargs)
        self.expand_ratio = expand_ratio

def get_block_wrapper(block_str):
    """Wrapper for MobileNetV2 block.
    Use `expand_ratio` instead of manually specified channels number."""

    assert block_str == 'InvertedResidualChannels'
    return InvertedResidual

class ParallelModule(nn.Module):
    def __init__(self,
                 num_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 num_blocks=[2, 2],
                 num_channels=[32, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6')):
        super(ParallelModule, self).__init__()

        self.num_branches = num_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes

        self._check_branches(
            num_branches, num_blocks, num_channels)
        self.branches = self._make_branches(
            num_branches, block, num_blocks, num_channels)

    def _check_branches(self, num_branches, num_blocks, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logging.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logging.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(0, num_blocks[branch_index]):
            layers.append(
                block(
                    num_channels[branch_index],
                    num_channels[branch_index],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        return x

class FuseModule(nn.Module):
    '''
        Consistent with HRNET:
        1. self.use_hr_format, eg: fuse 3 branches, and then add 4th branch from 3rd branch. (default fuse 4 branches)
        2. use_hr_format, if the channels are the same and stride==1, use None rather than fuse. (default, always fuse)
            and use convbnrelu, and kernel_size=1 when upsample.
            also control the relu here (last layer no relu)
        3. self.in_channels_large_stride, use 16->16->64 instead of 16->32->64 for large stride. (default, True)
        4. The only difference in self.use_hr_format when adding a branch:
            is we use add 4th branch from 3rd branch, add 5th branch from 4rd branch
            hrnet use add 4th branch from 3rd branch, add 5th branch from 3rd branch (2 conv layers)
            actually only affect 1->2 stage
            can be hard coded: self.use_hr_format = self.use_hr_format and not(out_branches == 2 and in_branches == 1)
        5. hrnet have a fuse layer at the end, we remove it
    '''
    def __init__(self,
                 in_branches=1,
                 out_branches=2,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 in_channels=[16],
                 out_channels=[16, 32],
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6'),
                 use_hr_format=False,
                 only_fuse_neighbor=True,
                 directly_downsample=True):
        super(FuseModule, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.only_fuse_neighbor = only_fuse_neighbor
        self.in_channels_large_stride = True  # see 3.
        if only_fuse_neighbor:
            self.use_hr_format = out_branches > in_branches
            # w/o self, are two different flags. (see 1.)
        else:
            self.use_hr_format = out_branches > in_branches and \
                                 not (out_branches == 2 and in_branches == 1)  # see 4.

        self.relu = self.active_fn()
        if use_hr_format:
            block = ConvBNReLU  # See 2.

        fuse_layers = []
        for i in range(out_branches if not self.use_hr_format else in_branches):
            fuse_layer = []
            for j in range(in_branches):
                if only_fuse_neighbor:
                    if j < i - 1 or j > i + 1:
                        fuse_layer.append(None)
                        continue
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        block(
                            in_channels[j],
                            out_channels[i],
                            expand_ratio=self.expand_ratio,
                            kernel_sizes=self.kernel_sizes,
                            stride=1,
                            batch_norm_kwargs=self.batch_norm_kwargs,
                            active_fn=self.active_fn if not use_hr_format else None,
                            kernel_size=1  # for hr format
                        ),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    if use_hr_format and in_channels[j] == out_channels[i]:
                        fuse_layer.append(None)
                    else:
                        fuse_layer.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=1,
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.active_fn if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                else:
                    downsamples = []
                    if directly_downsample:
                        downsamples.append(
                            block(
                                in_channels[j],
                                out_channels[i],
                                expand_ratio=self.expand_ratio,
                                kernel_sizes=self.kernel_sizes,
                                stride=2 ** (i - j),
                                batch_norm_kwargs=self.batch_norm_kwargs,
                                active_fn=self.active_fn if not use_hr_format else None,
                                kernel_size=3  # for hr format
                            ))
                    else:
                        for k in range(i - j):
                            if self.in_channels_large_stride:
                                if k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not use_hr_format else None,
                                            kernel_size=3  # for hr format
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            in_channels[j],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn,
                                            kernel_size=3  # for hr format
                                        ))
                            else:
                                if k == 0:
                                    downsamples.append(
                                        block(
                                            in_channels[j],
                                            out_channels[j + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not (use_hr_format and i == j + 1) else None,
                                            kernel_size=3  # for hr format
                                        ))
                                elif k == i - j - 1:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[i],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn if not use_hr_format else None,
                                            kernel_size=3  # for hr format
                                        ))
                                else:
                                    downsamples.append(
                                        block(
                                            out_channels[j + k],
                                            out_channels[j + k + 1],
                                            expand_ratio=self.expand_ratio,
                                            kernel_sizes=self.kernel_sizes,
                                            stride=2,
                                            batch_norm_kwargs=self.batch_norm_kwargs,
                                            active_fn=self.active_fn,
                                            kernel_size=3  # for hr format
                                        ))
                    fuse_layer.append(nn.Sequential(*downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        if self.use_hr_format:
            for branch in range(in_branches, out_branches):
                fuse_layers.append(nn.ModuleList([block(
                    out_channels[branch - 1],
                    out_channels[branch],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn,
                    kernel_size=3  # for hr format
                )]))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        if not self.only_fuse_neighbor:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                y = self.fuse_layers[i][0](x[0]) if self.fuse_layers[i][0] else x[0]  # hr_format, None
                for j in range(1, self.in_branches):
                    if self.fuse_layers[i][j]:
                        y = y + self.fuse_layers[i][j](x[j])
                    else:  # hr_format, None
                        y = y + x[j]
                x_fuse.append(self.relu(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        else:
            for i in range(len(self.fuse_layers) if not self.use_hr_format else self.in_branches):
                flag = 1
                for j in range(i-1, i+2):
                    if 0 <= j < self.in_branches:
                        if flag:
                            y = self.fuse_layers[i][j](x[j]) if self.fuse_layers[i][j] else x[j]  # hr_format, None
                            flag = 0
                        else:
                            if self.fuse_layers[i][j]:
                                y = y + resize(
                                    self.fuse_layers[i][j](x[j]),
                                    size=y.shape[2:],
                                    mode='bilinear',
                                    align_corners=False)
                            else:  # hr_format, None
                                y = y + x[j]
                x_fuse.append(self.relu(y))
            if self.use_hr_format:
                for branch in range(self.in_branches, self.out_branches):
                    x_fuse.append(self.fuse_layers[branch][0](x_fuse[branch - 1]))
        return x_fuse

class HeadModule(nn.Module):
    def __init__(self,
                 pre_stage_channels=[16, 32, 64, 128],
                 head_channels=None,  # [32, 64, 128, 256],
                 last_channel=1024,
                 avg_pool_size=7,
                 block=get_block_wrapper('InvertedResidualChannels'),
                 expand_ratio=6,
                 kernel_sizes=[3, 5, 7],
                 batch_norm_kwargs=None,
                 active_fn=get_active_fn('nn.ReLU6'),
                 concat_head_for_cls=False):
        super(HeadModule, self).__init__()

        self.active_fn = active_fn
        self.batch_norm_kwargs = batch_norm_kwargs
        self.expand_ratio = expand_ratio
        self.kernel_sizes = kernel_sizes
        self.avg_pool_size = avg_pool_size
        self.concat_head_for_cls = concat_head_for_cls

        # Increasing the #channels on each resolution
        if head_channels:
            incre_modules = []
            for i, channels in enumerate(pre_stage_channels):
                incre_module = block(
                    pre_stage_channels[i],
                    head_channels[i],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=1,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
                incre_modules.append(incre_module)
            self.incre_modules = nn.ModuleList(incre_modules)
        else:
            head_channels = pre_stage_channels
            self.incre_modules = []

        if not self.concat_head_for_cls:
            # downsampling modules
            downsamp_modules = []
            for i in range(len(pre_stage_channels) - 1):
                downsamp_module = block(
                    head_channels[i],
                    head_channels[i + 1],
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    stride=2,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
                downsamp_modules.append(downsamp_module)
            self.downsamp_modules = nn.ModuleList(downsamp_modules)
        else:
            self.downsamp_modules = []

        self.final_layer = ConvBNReLU(
            head_channels[-1] if not self.concat_head_for_cls else sum(head_channels),
            last_channel,
            kernel_size=1,
            batch_norm_kwargs=batch_norm_kwargs,
            active_fn=active_fn)

    def forward(self, x_list):

        # print("############### len of x list",len(x_list))
        # for i in range(4):
        #     print(x_list[i].shape)
        # print("########################################")


        if self.concat_head_for_cls:
            if self.incre_modules:
                for i in range(len(x_list)):
                    x_list[i] = self.incre_modules[i](x_list[i])
            x_incre = [resize(input=x,
                              size=x_list[-1].shape[2:],
                              mode='bilinear',
                              align_corners=False) for x in x_list]
            x = torch.cat(x_incre, dim=1)
        else:
            if self.incre_modules:
                x = self.incre_modules[0](x_list[0])
                for i in range(len(self.downsamp_modules)):
                    x = self.incre_modules[i + 1](x_list[i + 1]) \
                        + self.downsamp_modules[i](x)
            else:
                x = x_list[0]
                for i in range(len(self.downsamp_modules)):
                    x = x_list[i + 1] \
                        + self.downsamp_modules[i](x)

        x = self.final_layer(x)

        # assert x.size()[2] == self.avg_pool_size

        if torch._C._get_tracing_state():
            x = x.flatten(start_dim=2).mean(dim=2)
        else:
            x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)
        return x

## for point pillar (transform intial data, neck, head)
class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (max_voxel, num_points, c), coors_out: (max_voxel, 3)
            # num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        return pillars, coors_batch, npoints_per_pillar

class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas

class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        # print("@@@@@@", len(in_channels))
        # print(in_channels)
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)

        out = torch.cat(outs, dim=1)
        
        return out

class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred

class uni3d_args():
    def __init__(self):
        self.pc_model = "eva_giant_patch14_560.m30m_ft_in22k_in1k"
        self.pretrained_pc=""
        self.drop_path_rate=0.20
        self.pc_feat_dim = 1408           # Inference: 1408
        self.embed_dim = 1024             # Inference: 1024
        self.group_size = 64              # Inference: 64
        self.num_group = 512              # Inference: 512
        self.pc_encoder_dim = 512
        self.patch_dropout = 0.5

class HighResolutionNet(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 input_size=224,
                 input_stride=4,
                 input_channel=[16, 16],
                 last_channel=1024,
                 head_channels=None,
                 bn_momentum=0.1,
                 bn_epsilon=1e-5,
                 dropout_ratio=0.2,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 width_mult=1.0,
                 round_nearest=8,
                 expand_ratio=4,
                 kernel_sizes=[3, 5, 7],
                 inverted_residual_setting=None,
                 task='classification',
                 align_corners=False,
                 start_with_atomcell=False,
                 fcn_head_for_seg=False,
                 initial_for_heatmap=False,
                 # some parameter for 3d object detection
                 nclasses=3, 
                 mode = 'train',
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=32,
                 max_voxels=(16000, 40000),
                 **kwargs):
        super(HighResolutionNet, self).__init__()
        
        batch_norm_kwargs = {
            'momentum': bn_momentum,
            'eps': bn_epsilon
        }
        
        # some parameter and function for 3d object detection
        self.nclasses = nclasses
        
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)
        
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=9, 
                                            out_channel=128)




        ###########################################################
        self.avg_pool_size = input_size // 32
        self.input_stride = input_stride
        self.input_channel = [_make_divisible(item * width_mult, round_nearest) for item in input_channel]
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = get_active_fn(active_fn)
        self.kernel_sizes = kernel_sizes
        self.expand_ratio = expand_ratio
        self.task = task
        self.align_corners = align_corners
        self.initial_for_heatmap = initial_for_heatmap
        
        # uni3d ###########################
        u3d_args = uni3d_args()
        
        self.uni3_model = models.create_uni3d(u3d_args)
        checkpoint = torch.load("./checkpoint/model.pt", map_location='cpu')
        sd = checkpoint['module']
        # if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        #     sd = {k[len('module.'):]: v for k, v in sd.items()}
        self.uni3_model.load_state_dict(sd)
        self.uni3_model.eval()
        ###################################
        self.fusion_mlp = nn.Linear(1408, 3348)
        self.fusion_pixel_shuffle = nn.PixelShuffle(4)
        

        self.block = get_block_wrapper(block)
        self.inverted_residual_setting = inverted_residual_setting
        self.mode = mode
        downsamples = []
        
        if self.input_stride > 1:
            downsamples.append(ConvBNReLU(
                128,
                input_channel[0],
                kernel_size=3,
                stride=2,
                batch_norm_kwargs=self.batch_norm_kwargs,
                active_fn=self.active_fn))
        
        # if self.input_stride > 2:
        #     if start_with_atomcell:
        #         downsamples.append(InvertedResidual(input_channel[0],
        #                                             input_channel[0],
        #                                             1,
        #                                             1,
        #                                             [3],
        #                                             self.active_fn,
        #                                             self.batch_norm_kwargs))
        #     downsamples.append(ConvBNReLU(
        #         input_channel[0],
        #         input_channel[1],
        #         kernel_size=3,
        #         stride=2,
        #         batch_norm_kwargs=self.batch_norm_kwargs,
        #         active_fn=self.active_fn))
        # print("########$$$$",len(downsamples))
        # exit()
        self.downsamples = nn.Sequential(*downsamples)

        features = [] # 在中間的部份加入 fusion module 和 parallel module
        
        # for index in range(len(inverted_residual_setting)):
        #     in_channels = [input_channel[1]] if index == 0 else inverted_residual_setting[index - 1][-1]
        #     print("#####", in_channels)
        # exit()

        for index in range(len(inverted_residual_setting)):
            in_branches = 1 if index == 0 else inverted_residual_setting[index - 1][0]
            in_channels = [input_channel[1]] if index == 0 else inverted_residual_setting[index - 1][-1]
            features.append(
                FuseModule(
                    in_branches=in_branches,
                    out_branches=inverted_residual_setting[index][0],
                    in_channels=in_channels,
                    out_channels=inverted_residual_setting[index][-1],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
            features.append(
                ParallelModule(
                    num_branches=inverted_residual_setting[index][0],
                    num_blocks=inverted_residual_setting[index][1],
                    num_channels=inverted_residual_setting[index][2],
                    block=self.block,
                    expand_ratio=self.expand_ratio,
                    kernel_sizes=self.kernel_sizes,
                    batch_norm_kwargs=self.batch_norm_kwargs,
                    active_fn=self.active_fn)
            )
        
        if self.task == "3dclassification":

            features.append(
                # Neck(in_channels=[64, 128, 256], 
                #      upsample_strides=[1, 2, 4], 
                #      out_channels=[128, 128, 128])

                Neck(in_channels=[18, 36, 72, 144], 
                     upsample_strides=[1, 2, 4, 8], 
                     out_channels=[128, 128, 128, 128])
            )
            # 定義 classifier
            self.classifier = Head(in_channel=512, n_anchors=2*nclasses, n_classes=nclasses)
            # self.classifier = Head(in_channel=512, n_anchors=2*nclasses, n_classes=nclasses)
            
            ranges = [[0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                    [0, -39.68, -1.78, 69.12, 39.68, -1.78]]
            sizes = [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]]
            
            rotations=[0, 1.57]
            self.anchors_generator = Anchors(ranges=ranges, 
                                            sizes=sizes, 
                                            rotations=rotations)
            self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
            ]

            # val and test
            self.nms_pre = 100
            self.nms_thr = 0.01
            self.score_thr = 0.1
            self.max_num = 50
        
        self.features = nn.Sequential(*features) # 將每層整合起來成為 feature layer

        self.init_weights()

    def init_weights(self):
        if udist.is_master():
            logging.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if not self.initial_for_heatmap:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_named_block_list(self):
        """Get `{name: module}` dictionary for all inverted residual blocks."""
        blocks = list(self.features.named_children())
        all_cells = []
        for name, block in blocks:
            if isinstance(block, ParallelModule):
                parallel_module = block.branches
                for i, parallel_branch in enumerate(parallel_module):
                    for j, cell in enumerate(parallel_branch):
                        all_cells.append(
                            ('features.{}.branches.{}.{}'.format(name, i, j), cell))
            if isinstance(block, FuseModule):
                fuse_module = block.fuse_layers
                for i, fuse_branch in enumerate(fuse_module):
                    for j, fuse_path in enumerate(fuse_branch):
                        if isinstance(fuse_path, self.block):
                            all_cells.append(
                                ('features.{}.fuse_layers.{}.{}'.format(name, i, j), fuse_path))
                        if isinstance(fuse_path, nn.Sequential):
                            for k, cell in enumerate(fuse_path):
                                if isinstance(cell, self.block):
                                    all_cells.append(
                                        ('features.{}.fuse_layers.{}.{}.{}'.format(name, i, j, k), cell))
            if isinstance(block, HeadModule):
                incre_module = block.incre_modules
                downsample_module = block.downsamp_modules
                for i, cell in enumerate(incre_module):
                    if isinstance(cell, self.block):
                        all_cells.append(
                            ('features.{}.incre_modules.{}'.format(name, i), cell))
                for i, cell in enumerate(downsample_module):
                    if isinstance(cell, self.block):
                        all_cells.append(
                            ('features.{}.downsamp_modules.{}'.format(name, i), cell))
        for name, block in self.named_children():
            if isinstance(block, self.block):
                all_cells.append(
                    ('{}'.format(name), block))

        return collections.OrderedDict(all_cells)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        upsampled_inputs = [
            resize(
                input=x,
                size=inputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)
        inputs = self.transform(inputs)
        return inputs

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return [], [], []
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }
        return result

    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    # batch_pts should be list of tensor
    def forward(self, batched_pts, batched_gt_bboxes=None, batched_gt_labels=None, mode='train'):
        
        batch_size = len(batched_pts)
        # print(batch_size)
        
        # uni3d_feature = [self.uni3_model.encode_pc(batched_pts[i])[0] for i in range(batch_size)]
        # uni3d_feature = torch.cat(uni3d_feature, dim=0)
        # # print("uni3d encode feature: ", uni3d_feature.shape)
        # uni3d_tmp_feature = [self.uni3_model.encode_pc(batched_pts[i])[1] for i in range(batch_size)]
        # uni3d_tmp_feature = torch.cat(uni3d_tmp_feature, dim=0)
        # # print("uni3d encode tmp_feature: ", uni3d_tmp_feature.shape)
        # uni3d_tmp_feature = self.fusion_mlp(uni3d_tmp_feature)
        # uni3d_tmp_feature = uni3d_tmp_feature.view(batch_size, 257, 62, 54)
        # uni3d_tmp_feature = uni3d_tmp_feature[:,:256,:,:]
        # uni3d_tmp_feature = self.fusion_pixel_shuffle(uni3d_tmp_feature)
        # # print("important:",uni3d_tmp_feature.shape)

        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar) # 1, 128, 496, 432
        # print("pillar_features shape: ", pillar_features.shape)
        # hr-nas
        x = self.downsamples(pillar_features)
        # print("hr-nas downsample: ", x.shape)
        x = self.features([x])
        # print("hr-nas feature", x.shape)
        # x = torch.cat([x, uni3d_tmp_feature], dim = 1)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.classifier(x) 

        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]

        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   

Model = HighResolutionNet
