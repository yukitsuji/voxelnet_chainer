#/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import functools
import numpy as np
import os
import sys
import subprocess
import time

try:
    import matplotlib.pyplot as plt
except:
    pass

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.transforms import resize
from chainercv.utils import download_model
from voxelnet.models.spatial_dropout import spatial_dropout
from voxelnet.models.feature_to_voxel import feature_to_voxel

from data_util.kitti_util.cython_util.nms_3d import nms_3d
from voxelnet.models.utils import create_timer, print_timer
from voxelnet.models.active_bn import BatchNormalization as BN


class BasicModel(chainer.Chain):
    def __init__(self):
        super(BasicModel, self).__init__()

    def __call__(self, x, counter, indexes, gt_prob, gt_reg, gt_obj_for_reg,
                 area_mask, batch, n_no_empty):
        """
           Args:
               x (ndarray): Shape is (Batch * ave(K), 7, t).
                            each set has (xi, yi, zi, ri, xi −vx, yi −vy, zi −vz).
                            vx, vy, vz is local mean at each voxel.
               indexes (ndarray): Shape is (Batch * K, 3). 3 is (d, h, w).
               gt_prob (ndarray): Shape is (Batch, H, W) or (Batch, H, W, 2).
               gt_reg (ndarray): Shape is (Batch, 8, H, W).
               area_mask (ndarray): Shape is (Batch, H, W)
           Return:
               loss (Variable).
        """
        with self.xp.cuda.Device(chainer.cuda.get_device_from_array(x)):
            x = self.feature_net(x, n_no_empty)
            x = feature_to_voxel(x, indexes, self.k, self.d, self.h, self.w, batch)
            x = self.middle_conv(x)
            pred_prob, pred_reg = self.rpn(x)
            prob_loss, reg_loss = self.binary_cross_entropy(pred_prob, gt_prob,
                                                            pred_reg, gt_reg,
                                                            gt_obj_for_reg,
                                                            area_mask)
            total_loss = prob_loss * self.p + reg_loss
            chainer.report({'loss': total_loss}, self)
            chainer.report({'prob_loss': prob_loss}, self)
            chainer.report({'reg_loss': reg_loss}, self)
            return total_loss

    def binary_cross_entropy(self, pred_prob, gt_prob, pred_reg, gt_reg,
                             gt_obj_for_reg, area_mask):
        """
           Args:
               pred_prob: Shape is (Batch, 1, H, W)
               pred_reg: Shape is (Batch, 8, H, W)
               gt_prob: Shape is (Batch, H, W) # 0 or 1
               gt_reg: Shape is (Batch, 8, H, W) x, y, z, l, w, h, rotate, head of rotate
               mask: Shape is (Batch, 1, H, W). Mask is a search area of model.
        """
        batch, _, h, w = pred_prob.shape
        gt_prob = gt_prob.reshape(batch, h, w)
        gt_reg = gt_reg.reshape(pred_reg.shape)
        gt_obj_for_reg = gt_obj_for_reg.reshape(batch, h, w)
        area_mask = area_mask.reshape(batch, h, w)

        pred_prob = F.sigmoid(pred_prob)
        positive_loss = 0
        loc_loss = 0

        if self.aug_gt: # set surrounding area as True
            gt_prob = gt_obj_for_reg
        num_positive = (gt_prob != 0).sum()

        if num_positive:
            # Regression loss
            loc_loss = F.sum(F.huber_loss(pred_reg[:, :7], gt_reg[:, :7], 1, reduce='no'), axis=1)
            rotate_head = F.sigmoid(pred_reg[:, 7])
            rotate_head_loss = -F.log(rotate_head + 1e-5) * gt_reg[:, 7] - F.log((1 - rotate_head) + 1e-5) * (1 - gt_reg[:, 7])
            loc_loss += rotate_head_loss
            loc_loss = F.sum(loc_loss * gt_obj_for_reg) / num_positive

            # Positive confidence loss
            positive_loss = F.sum(-F.log(pred_prob[:, 0] + 1e-5) * gt_prob) / num_positive

        # Negative confidence loss
        negative_mask = (gt_prob == 0) * area_mask
        num_negative = negative_mask.sum()
        negative_loss = F.sum(-F.log((1 - pred_prob[:, 0]) + 1e-5)  * negative_mask) / num_negative
        conf_loss = self.alpha * positive_loss + self.beta * negative_loss
        chainer.report({'posi': positive_loss}, self)
        chainer.report({'nega': negative_loss}, self)
        return conf_loss, loc_loss

    def decoder(self, pred_reg, anchor, anchor_size, xp=np):
        pred_reg[:, 0] = pred_reg[:, 0] * anchor_size[2] + anchor[:, 0]
        pred_reg[:, 1] = pred_reg[:, 1] * anchor_size[1] + anchor[:, 1]
        pred_reg[:, 2] = pred_reg[:, 2] * anchor_size[0] + anchor[:, 2]
        pred_reg[:, 3] = xp.exp(pred_reg[:, 3]) * anchor_size[2] # pred_length 奥行き
        pred_reg[:, 4] = xp.exp(pred_reg[:, 4]) * anchor_size[1] # pred_width
        pred_reg[:, 5] = xp.exp(pred_reg[:, 5]) * anchor_size[0] # pred_height
        rotate_head = np.where(pred_reg[:, 7] >= 0 , 1, -1) # rotate angle
        pred_reg[:, 6] = rotate_head * pred_reg[:, 6] * (np.pi / 2) # pred_rotate
        return pred_reg

    # def softmax_cross_entropy(self, pred_prob, gt_prob, pred_reg, gt_reg):
    #     """
    #        Args:
    #            pred_prob: Shape is (Batch, 2, H, W)
    #            pred_reg: Shape is (Batch, 7, H, W)
    #            gt_prob: Shape is (Batch, H, W) # 0 or 1
    #            gt_reg: Shape is (Batch, 7, H, W)
    #     """
    #     num_positive = gt_prob.sum()
    #     batch, _, h, w = pred_prob.shape
    #     gt_prob = gt_prob.reshape(batch, h, w)
    #     gt_reg = gt_reg.reshape(pred_reg.shape)
    #     loc_loss = F.sum(F.huber_loss(pred_reg, gt_reg, 1, reduce='no'), axis=1)
    #     loc_loss = F.sum(loc_loss * gt_prob) / num_positive
    #     exp_prob = F.exp(pred_prob)
    #     exp_prob = exp_prob / F.broadcast_to(F.sum(exp_prob, axis=1, keepdims=True), (batch, 2, h, w))
    #     positive_loss = F.sum(-F.log(exp_prob[:, 0] + 1e-5) * gt_prob) / num_positive
    #     num_negative = (1 - gt_prob).sum()
    #     negative_loss = F.sum(-F.log(exp_prob[:, 1] + 1e-5) * (1 - gt_prob)) / num_negative
    #     conf_loss = self.alpha * positive_loss + self.beta * negative_loss
    #     chainer.report({'posi': positive_loss}, self)
    #     chainer.report({'nega': negative_loss}, self)
    #     return conf_loss, loc_loss

    def inference(self, x, counter, indexes, batch, n_no_empty, area_mask,
                  config=None, thres_prob=0.996, nms_thresh=0.0,
                  anchor_size=None, anchor_center=None, anchor=None):
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            sum_time = 0
            start, stop = create_timer()
            x = self.feature_net(x)
            x = feature_to_voxel(x, indexes, self.k, self.d, self.h, self.w, batch)
            x = self.middle_conv(x)
            pred_prob, pred_reg = self.rpn(x)
            print_timer(start, stop, "## Sum of execution time: ")
            s = time.time()
            pred_reg = self.xp.transpose(pred_reg, (0, 2, 3, 1)).data[0]
            pred_prob = pred_prob[0, 0].data
            candidate = F.sigmoid(pred_prob).data * area_mask > thres_prob
            pred_prob = pred_prob[candidate]
            pred_reg = pred_reg[candidate]
            pred_prob = chainer.cuda.to_cpu(pred_prob)
            pred_reg = chainer.cuda.to_cpu(pred_reg)
            candidate = chainer.cuda.to_cpu(candidate)
            anchor = anchor[candidate]
            pred_reg = self.decoder(pred_reg, anchor, anchor_size, xp=np)
            sort_index = np.argsort(pred_prob)[::-1]
            pred_reg = pred_reg[sort_index]
            pred_prob = pred_prob[sort_index]
            result_index = nms_3d(pred_reg,
                                  pred_prob,
                                  nms_thresh)
            print("Post-processing", time.time() - s)
        return pred_reg[result_index][:, :7], pred_prob[result_index]

    def visualize(self, pred_reg, gt_reg, pred_prob, gt_prob, area_mask,
                  l_rotate=None, g_rotate=None, resolution=None, voxel_shape=None,
                  x_range=None, y_range=None, z_range=None, t=35, thres_t=None,
                  anchor_size=(1.56, 1.6, 3.9), anchor_center=(-1.0, 0., 0.),
                  fliplr=False, n_class=20, scale_label=1, thres_p=0.998,
                  **kwargs):
        """
           gt_prob: shape is (Batch, H, W)
           gt_reg: shape is (Batch, 7, H, W)
           pred_prob: shape is (Batch, 1, H, W)
           pred_reg: shape is (Batch, 7, H, W)
        """
        d, h, w = voxel_shape
        d_res, h_res, w_res = resolution
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        anchor_z, anchor_y, anchor_x = anchor_center
        anchor_h, anchor_w, anchor_l = anchor_size

        pred_reg = pred_reg.data
        pred_prob = pred_prob.data[:, 0]
        gt_prob = gt_prob
        batch, _, h, w = pred_reg.shape
        gt_prob = gt_prob.reshape(batch, h, w).astype('bool')
        gt_reg = gt_reg.reshape(pred_reg.shape)
        pred_reg = self.xp.transpose(pred_reg, (0, 2, 3, 1))
        gt_reg = self.xp.transpose(gt_reg, (0, 2, 3, 1))

        # anchor = self.xp.zeros((batch, h, w, 2))
        # x_array = F.broadcast_to(self.xp.arange(0, x_max - x_min, w_res*scale_label), (batch, h, w))
        # y_array = F.broadcast_to(self.xp.arange(0, y_max - y_min, h_res*scale_label)[self.xp.newaxis, :, self.xp.newaxis], (batch, h, w))
        # anchor[:, :, :, 0] = x_array.data
        # anchor[:, :, :, 1] = y_array.data

        # thres = F.sigmoid(pred_prob).data > thres_p
        # img = np.zeros((h, w, 3), dtype="f")
        # thres_cpu = chainer.cuda.to_cpu(thres)
        # # print(thres.shape, thres_cpu.shape)
        # img[thres_cpu[0]] = 1
        #
        # img = img * chainer.cuda.to_cpu(area_mask.transpose(1, 2, 0))

        # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))
        # ax1.imshow(img)
        # ax2.imshow(chainer.cuda.to_cpu(gt_prob.astype("f")[0]))
        # plt.show()
        # pred_center_x = pred_reg[thres][:, 0] * anchor_l + anchor[thres][:, 0]
        # pred_center_y = pred_reg[:, :, :, 1] * anchor_w + anchor[:, :, :, 1]
        # pred_length = self.xp.exp(pred_reg[:, :, :,  3]) * anchor_l
        # pred_width = self.xp.exp(pred_reg[:, :, :,  4]) * anchor_w
        # pred_rotate = pred_reg[:, :, :, 6] * 3.14160
        #
        # gt_center_x = gt_reg[:, :, :, 0] * anchor_l + anchor[:, :, :, 0]
        # gt_center_y = gt_reg[:, :, :, 1] * anchor_w + anchor[:, :, :, 1]
        # gt_length = self.xp.exp(gt_reg[:, :, :, 3]) * anchor_l
        # gt_width = self.xp.exp(gt_reg[:, :, :, 4]) * anchor_w
        # gt_rotate = gt_reg[:, :, :, 6] * 3.14160
        #
        # true_prob = self.xp.mean(F.sigmoid(pred_prob[gt_prob]).data)
        # false_prob = 1 - self.xp.mean(F.sigmoid(pred_prob[~gt_prob]).data)

        anchor = self.xp.zeros((batch, h, w, 3), dtype="f")
        x_array = F.broadcast_to(self.xp.arange(0, x_max - x_min, w_res*scale_label), (batch, h, w))
        y_array = F.broadcast_to(self.xp.arange(0, y_max - y_min, h_res*scale_label)[self.xp.newaxis, :, self.xp.newaxis], (batch, h, w)) + y_min
        anchor[:, :, :, 0] = x_array.data
        anchor[:, :, :, 1] = y_array.data
        anchor[:, :, :, 2] = anchor_center[0]
        gt_reg = gt_reg[0] #self.xp.transpose(gt_reg, (0, 2, 3, 1)).data[0]
        gt_prob = gt_prob[0].astype("f")
        candidate = gt_prob * area_mask[0] > 0.5
        gt_prob = gt_prob[candidate]
        gt_reg = gt_reg[candidate]
        gt_prob = chainer.cuda.to_cpu(gt_prob)
        gt_reg = chainer.cuda.to_cpu(gt_reg)
        candidate = chainer.cuda.to_cpu(candidate)
        anchor = chainer.cuda.to_cpu(anchor)
        anchor = anchor[0][candidate]
        gt_reg = self.decoder(gt_reg, anchor, anchor_size, xp=np)
        sort_index = np.argsort(gt_prob)[::-1]
        gt_reg = gt_reg[sort_index]
        gt_prob = gt_prob[sort_index]
        thres_prob = 0.5
        result_index = nms_3d(gt_reg,
                              gt_prob,
                              0.0)
        print(gt_reg[result_index])
        gt_reg[result_index][:, :7]
        print("####################################")

    def viz_input(self, x):
        input_x = chainer.cuda.to_cpu(x.data.astype("f")[0])
        input_x = input_x.max(axis=(0, 1))
        input_x[input_x != 0] = 1
        plt.imshow(input_x[::-1])
        plt.show()

    def predict(self, x, counter, indexes, gt_prob, gt_reg, area_mask, batch,
                n_no_empty, config=None):
        with chainer.using_config('train', False), \
                 chainer.function.no_backprop_mode():
            sum_time = 0
            start, stop = create_timer()
            x = self.feature_net(x)
            sum_time += print_timer(start, stop, sentence="feature net")
            start, stop = create_timer()
            y = feature_to_voxel(x, indexes, self.k, self.d, self.h, self.w, batch)
            sum_time += print_timer(start, stop, sentence="feature_to_voxel")
            start, stop = create_timer()
            x = self.middle_conv(y)
            sum_time += print_timer(start, stop, sentence="middle_conv")
            start, stop = create_timer()
            pred_prob, pred_reg = self.rpn(x)
            sum_time += print_timer(start, stop, sentence="rpn")
            print("## Sum of execution time: ", sum_time)
            # self.viz_input(y)
            if config is not None:
                print("#####   Visualize   #####")
                self.visualize(pred_reg, gt_reg, pred_prob, gt_prob, area_mask,
                               **config)


class FeatureVoxelNet(chainer.Chain):

    """Feature Learning Network"""

    def __init__(self, out_ch=128):
        super(FeatureVoxelNet, self).__init__(
            conv1 = L.ConvolutionND(1, 7, 16, 1, nobias=True),
            conv2 = L.ConvolutionND(1, 32, 64, 1, nobias=True),
            conv3 = L.ConvolutionND(1, 128, out_ch, 1, nobias=True),
	        bn1 = BN(16), #L.BatchNormalization(16),
	        bn2 = BN(64), #L.BatchNormalization(64),
	        bn3 = BN(out_ch)) #L.BatchNormalization(out_ch))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch * K, 7, t).
                            each set has (xi, yi, zi, intensity, xi−vx, yi−vy, zi−vz).
                            vx, vy, vz is local mean at each voxel.
           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        n_batch, n_channels, n_points = x.shape
        # mask = F.max(x, axis=(1, 2), keepdims=True).data != 0
        mask = F.max(x, axis=1, keepdims=True).data != 0
        active_length = 0 #mask.sum()

        # Convolution1D -> BN -> relu -> pool -> concat
        h = F.relu(self.bn1(self.conv1(x), active_length, mask))
        global_feat = F.max_pooling_nd(h, n_points)
        # Shape is (Batch, channel, points)
        global_feat_expand = F.tile(global_feat, (1, 1, n_points))
        h = F.concat((h, global_feat_expand))
        h *= mask

        h = F.relu(self.bn2(self.conv2(h), active_length, mask))
        global_feat = F.max_pooling_nd(h, n_points)
        global_feat_expand = F.tile(global_feat, (1, 1, n_points))
        h = F.concat((h, global_feat_expand))
        h *= mask

        # h = F.relu(self.bn3(self.conv3(h), active_length))
        h = self.conv3(h)
        # h *= mask
        return F.squeeze(F.max_pooling_nd(h, n_points))


class MiddleLayers(chainer.Chain):

    """Convolution Middle Layers."""
    def __init__(self, in_ch=128, out_ch=64):
        super(MiddleLayers, self).__init__(
            conv1 = L.ConvolutionND(3, in_ch, 32, (3, 1, 1), (2, 1, 1), (0, 0, 0), nobias=True),
            conv2 = L.ConvolutionND(3, 32, 64, (1, 3, 3), (1, 1, 1), (0, 1, 1), nobias=True),
            conv3 = L.ConvolutionND(3, 64, 32, (3, 1, 1), 1, (0, 0, 0), nobias=True),
            conv4 = L.ConvolutionND(3, 32, 64, (1, 3, 3), 1, (0, 1, 1), nobias=True),
            conv5 = L.ConvolutionND(3, 64, out_ch, (2, 3, 3), (1, 1, 1), (0, 1, 1), nobias=True),
            bn1 = L.BatchNormalization(32),
            bn2 = L.BatchNormalization(64),
            bn3 = L.BatchNormalization(32),
            bn4 = L.BatchNormalization(64),
            bn5 = L.BatchNormalization(out_ch))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        return h


class RegionProposalNet(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=64):
        super(RegionProposalNet, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 64, 3, 2, 1, nobias=True), # 0.4 # 3
            conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 5
            conv1_3 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 7
            conv1_4 = L.Convolution2D(64, 128, 3, 1, 1, nobias=True), # 9 = 3.6m

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True), # 13
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 17
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 21
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 25
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 29
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 33

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True), # 41
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 49
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 57
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 65
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 73
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 81 * 0.4 = 32.4m

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            prob_conv = L.Convolution2D(768, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(768, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(64),
            bn1_2 = L.BatchNormalization(64),
            bn1_3 = L.BatchNormalization(64),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            x = x[:, :, 0]

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn2_1(self.conv2_1(out1)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))
        h = F.relu(self.bn2_4(self.conv2_4(h)))
        h = F.relu(self.bn2_5(self.conv2_5(h)))
        out2 = F.relu(self.bn2_6(self.conv2_6(h)))
        h = F.relu(self.bn3_1(self.conv3_1(out2)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.relu(self.bn3_5(self.conv3_5(h)))
        out3 = F.relu(self.bn3_6(self.conv3_6(h)))
        out1 = F.relu(self.bn_out1(self.deconv1(out1)))
        out2 = F.relu(self.bn_out2(self.deconv2(out2)))
        out3 = F.relu(self.bn_out3(self.deconv3(out3)))
        h = F.concat((out1, out2, out3), axis=1)
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class LightVoxelnet(BasicModel):

    """Voxelnet original Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class FeatureVoxelNet_v2(chainer.Chain):

    """Feature Learning Network"""

    def __init__(self, out_ch=128):
        super(FeatureVoxelNet_v2, self).__init__(
            conv1 = L.ConvolutionND(1, 7, out_ch, 1, nobias=True))
            # conv2 = L.ConvolutionND(1, 32, 64, 1, nobias=True),
            # conv3 = L.ConvolutionND(1, 128, out_ch, 1, nobias=True))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch * K, 7, t).
                            each set has (xi, yi, zi, ri, xi −vx, yi −vy, zi −vz).
                            vx, vy, vz is local mean at each voxel.
           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        n_batch, n_channels, n_points = x.shape
        h = F.relu(self.conv1(x))
        return F.squeeze(F.max_pooling_nd(h, n_points))


class LightVoxelnet_v2(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v2, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet_v2()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class RegionProposalNet_v3(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=64):
        super(RegionProposalNet_v3, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 64, 3, 2, 1, nobias=True), # 0.4 # 3
            conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 5
            conv1_3 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 7
            conv1_4 = L.Convolution2D(64, 128, 3, 1, 1, nobias=True), # 9 = 3.6m

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True), # 13
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 17
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 21
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 25
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 29
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 33

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True), # 41
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 49
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 57
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 65
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 73
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 81 * 0.4 = 32.4m

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            prob_conv = L.Convolution2D(512, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(512, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(64),
            bn1_2 = L.BatchNormalization(64),
            bn1_3 = L.BatchNormalization(64),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            x = x[:, :, 0]

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn2_1(self.conv2_1(out1)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))
        h = F.relu(self.bn2_4(self.conv2_4(h)))
        h = F.relu(self.bn2_5(self.conv2_5(h)))
        out2 = F.relu(self.bn2_6(self.conv2_6(h)))
        out1 = F.relu(self.bn_out1(self.deconv1(out1)))
        out2 = F.relu(self.bn_out2(self.deconv2(out2)))
        h = F.concat((out1, out2), axis=1)
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class LightVoxelnet_v3(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v3, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet_v3()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class RegionProposalNet_v4(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=64):
        super(RegionProposalNet_v4, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 64, 3, 2, 1, nobias=True), # 0.4 # 3
            conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 5
            conv1_3 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 7
            conv1_4 = L.Convolution2D(64, 128, 3, 1, 1, nobias=True), # 9 = 3.6m

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True), # 13
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 17
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 21
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 25
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 29
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 33

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True), # 41
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 49
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 57
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 65
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 73
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 81 * 0.4 = 32.4m

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            prob_conv = L.Convolution2D(256, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(256, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(64),
            bn1_2 = L.BatchNormalization(64),
            bn1_3 = L.BatchNormalization(64),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            x = x[:, :, 0]

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn_out1(self.deconv1(out1)))
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class LightVoxelnet_v4(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v4, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet_v4()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class RegionProposalNet_v5(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=64):
        super(RegionProposalNet_v5, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 64, 3, 2, 1, nobias=True), # 0.4 # 3
            conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 5
            conv1_3 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 7
            conv1_4 = L.Convolution2D(64, 128, 3, 1, 1, nobias=True), # 9 = 3.6m

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True), # 13
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 17
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 21
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 25
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 29
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 33

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True), # 41
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 49
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 57
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 65
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 73
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 81 * 0.4 = 32.4m

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            conv4 = L.Convolution2D(768, 768, 3, 1, 1, nobias=True),
            bn4 = L.BatchNormalization(768),

            prob_conv = L.Convolution2D(768, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(768, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(64),
            bn1_2 = L.BatchNormalization(64),
            bn1_3 = L.BatchNormalization(64),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            x = x[:, :, 0]

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn2_1(self.conv2_1(out1)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))
        h = F.relu(self.bn2_4(self.conv2_4(h)))
        h = F.relu(self.bn2_5(self.conv2_5(h)))
        out2 = F.relu(self.bn2_6(self.conv2_6(h)))
        h = F.relu(self.bn3_1(self.conv3_1(out2)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.relu(self.bn3_5(self.conv3_5(h)))
        out3 = F.relu(self.bn3_6(self.conv3_6(h)))
        out1 = F.relu(self.bn_out1(self.deconv1(out1)))
        out2 = F.relu(self.bn_out2(self.deconv2(out2)))
        out3 = F.relu(self.bn_out3(self.deconv3(out3)))
        h = F.concat((out1, out2, out3), axis=1)
        h = F.relu(self.bn4(self.conv4(h)))
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class LightVoxelnet_v5(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v5, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet_v5()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class FeatureVoxelNet_v6(chainer.Chain):

    """Feature Learning Network"""

    def __init__(self, out_ch=128):
        super(FeatureVoxelNet_v6, self).__init__(
            conv1 = L.ConvolutionND(1, 7, 32, 1, nobias=True),
	        conv2 = L.ConvolutionND(1, 64, out_ch, 1),
            # conv3 = L.ConvolutionND(1, 128, out_ch, 1, nobias=True),
	        bn1 = L.BatchNormalization(32))
	        # bn2 = L.BatchNormalization(out_ch))
	        # bn3 = L.BatchNormalization(out_ch))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch * K, 7, t).
                            each set has (xi, yi, zi, ri, xi −vx, yi −vy, zi −vz).
                            vx, vy, vz is local mean at each voxel.
           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        n_batch, n_channels, n_points = x.shape
        # mask = F.max(x, axis=(1, 2), keepdims=True).data != 0
        mask = F.max(x, axis=1, keepdims=True).data != 0
        active_length = 0 #mask.sum()

        # Convolution1D -> BN -> relu -> pool -> concat
        h = F.relu(self.bn1(self.conv1(x), active_length, mask))
        global_feat = F.max_pooling_nd(h, n_points)
        # Shape is (Batch, channel, points)
        global_feat_expand = F.tile(global_feat, (1, 1, n_points))
        h = F.concat((h, global_feat_expand))
        h *= mask

        h = self.conv2(h)
        return F.squeeze(F.max_pooling_nd(h, n_points))


class LightVoxelnet_v6(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v6, self).__init_()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet_v6()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class RegionProposalNet_v7(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=64):
        super(RegionProposalNet_v7, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 64, 3, 2, 1, nobias=True), # 0.4 # 3
            conv1_2 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 5
            conv1_3 = L.Convolution2D(64, 64, 3, 1, 1, nobias=True), # 7
            conv1_4 = L.Convolution2D(64, 128, 3, 1, 1, nobias=True), # 9 = 3.6m

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True), # 13
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 17
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 21
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 25
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 29
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True), # 33

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True), # 41
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 49
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 57
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 65
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 73
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True), # 81 * 0.4 = 32.4m

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            conv4 = L.Convolution2D(512, 768, 3, 1, 1, nobias=True),
            bn4 = L.BatchNormalization(768),

            prob_conv = L.Convolution2D(768, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(768, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(64),
            bn1_2 = L.BatchNormalization(64),
            bn1_3 = L.BatchNormalization(64),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).

           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            x = x[:, :, 0]

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn2_1(self.conv2_1(out1)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))
        h = F.relu(self.bn2_4(self.conv2_4(h)))
        h = F.relu(self.bn2_5(self.conv2_5(h)))
        out2 = F.relu(self.bn2_6(self.conv2_6(h)))
        out1 = F.relu(self.bn_out1(self.deconv1(out1)))
        out2 = F.relu(self.bn_out2(self.deconv2(out2)))
        h = F.concat((out1, out2), axis=1)
        h = F.relu(self.bn4(self.conv4(h)))
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class LightVoxelnet_v7(BasicModel):

    """Light-Weight VoxelNet Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(LightVoxelnet_v7, self).__init__()
        with self.init_scope():
            self.feature_net = FeatureVoxelNet()
            self.middle_conv = MiddleLayers()
            self.rpn = RegionProposalNet_v7()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)


class OrigFeatureVoxelNet(chainer.Chain):

    """Feature Learning Network"""

    def __init__(self, out_ch=128):
        super(OrigFeatureVoxelNet, self).__init__(
            conv1 = L.ConvolutionND(1, 7, 16, 1, nobias=True),
	        conv2 = L.ConvolutionND(1, 32, 64, 1, nobias=True),
            conv3 = L.ConvolutionND(1, 128, out_ch, 1),
	        bn1 = BN(16), #L.BatchNormalization(16),
	        bn2 = BN(64)) #L.BatchNormalization(64),
	        # bn3 = BN(out_ch)) #L.BatchNormalization(out_ch))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch * K, 7, t).
                            each set has (xi, yi, zi, ri, xi −vx, yi −vy, zi −vz).
                            vx, vy, vz are local mean at each voxel.
           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        n_batch, n_channels, n_points = x.shape
        # mask = F.max(x, axis=(1, 2), keepdims=True).data != 0
        mask = F.max(x, axis=1, keepdims=True).data != 0
        active_length = 0 #mask.sum()

        # Convolution1D -> BN -> relu -> pool -> concat
        h = F.relu(self.bn1(self.conv1(x), active_length, mask))
        global_feat = F.max_pooling_nd(h, n_points)
        # Shape is (Batch, channel, points)
        global_feat_expand = F.tile(global_feat, (1, 1, n_points))
        h = F.concat((h, global_feat_expand))
        h *= mask

        h = F.relu(self.bn2(self.conv2(h), active_length, mask))
        global_feat = F.max_pooling_nd(h, n_points)
        global_feat_expand = F.tile(global_feat, (1, 1, n_points))
        h = F.concat((h, global_feat_expand))
        h *= mask

        # h = F.relu(self.bn3(self.conv3(h), active_length))
        h = self.conv3(h)
        # h *= mask
        return F.squeeze(F.max_pooling_nd(h, n_points))


class OrigMiddleLayers(chainer.Chain):

    """Convolution Middle Layers."""
    def __init__(self, in_ch=128, out_ch=64):
        super(OrigMiddleLayers, self).__init__(
            conv1 = L.ConvolutionND(3, in_ch, 64, 3, (2, 1, 1), (1, 1, 1), nobias=True),
            conv2 = L.ConvolutionND(3, 64, 64, 3, 1, (0, 1, 1), nobias=True),
            conv3 = L.ConvolutionND(3, 64, out_ch, 3, (2, 1, 1), (1, 1, 1), nobias=True),
            bn1 = L.BatchNormalization(64),
            bn2 = L.BatchNormalization(64),
            bn3 = L.BatchNormalization(out_ch))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).
           Return:
               y (ndarray): Shape is (Batch * K, 128)
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        return F.relu(self.bn3(self.conv3(h)))


class OrigRegionProposalNet(chainer.Chain):

    """Region Proposal Network"""

    def __init__(self, in_ch=128):
        super(OrigRegionProposalNet, self).__init__(
            conv1_1 = L.Convolution2D(in_ch, 128, 3, 2, 1, nobias=True),
            conv1_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv1_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv1_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),

            conv2_1 = L.Convolution2D(128, 128, 3, 2, 1, nobias=True),
            conv2_2 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv2_3 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv2_4 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv2_5 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),
            conv2_6 = L.Convolution2D(128, 128, 3, 1, 1, nobias=True),

            conv3_1 = L.Convolution2D(128, 256, 3, 2, 1, nobias=True),
            conv3_2 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True),
            conv3_3 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True),
            conv3_4 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True),
            conv3_5 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True),
            conv3_6 = L.Convolution2D(256, 256, 3, 1, 1, nobias=True),

            deconv1 = L.Deconvolution2D(128, 256, 3, 1, 1, nobias=True),
            deconv2 = L.Deconvolution2D(128, 256, 2, 2, 0, nobias=True),
            deconv3 = L.Deconvolution2D(256, 256, 4, 4, 0, nobias=True),

            prob_conv = L.Convolution2D(768, 1, 1, 1, 0, nobias=False),
            reg_conv = L.Convolution2D(768, 8, 1, 1, 0, nobias=False),

            bn1_1 = L.BatchNormalization(in_ch),
            bn1_2 = L.BatchNormalization(128),
            bn1_3 = L.BatchNormalization(128),
            bn1_4 = L.BatchNormalization(128),
            bn2_1 = L.BatchNormalization(128),
            bn2_2 = L.BatchNormalization(128),
            bn2_3 = L.BatchNormalization(128),
            bn2_4 = L.BatchNormalization(128),
            bn2_5 = L.BatchNormalization(128),
            bn2_6 = L.BatchNormalization(128),
            bn3_1 = L.BatchNormalization(256),
            bn3_2 = L.BatchNormalization(256),
            bn3_3 = L.BatchNormalization(256),
            bn3_4 = L.BatchNormalization(256),
            bn3_5 = L.BatchNormalization(256),
            bn3_6 = L.BatchNormalization(256),
            bn_out1 = L.BatchNormalization(256),
            bn_out2 = L.BatchNormalization(256),
            bn_out3 = L.BatchNormalization(256))

    def __call__(self, x, *args):
        """
           Args:
               x (ndarray): Shape is (Batch, C, D, H, W).
           Return:
               y (ndarray): regression map and probability score map
        """
        if x.ndim == 5:
            batch, C, _, H, W = x.shape
            x = F.reshape(x, (batch, -1, H, W))

        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.relu(self.bn1_3(self.conv1_3(h)))
        out1 = F.relu(self.bn1_4(self.conv1_4(h)))
        h = F.relu(self.bn2_1(self.conv2_1(out1)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.relu(self.bn2_3(self.conv2_3(h)))
        h = F.relu(self.bn2_4(self.conv2_4(h)))
        h = F.relu(self.bn2_5(self.conv2_5(h)))
        out2 = F.relu(self.bn2_6(self.conv2_6(h)))
        h = F.relu(self.bn3_1(self.conv3_1(out2)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.relu(self.bn3_5(self.conv3_5(h)))
        out3 = F.relu(self.bn3_6(self.conv3_6(h)))
        out1 = F.relu(self.bn_out1(self.deconv1(out1)))
        out2 = F.relu(self.bn_out2(self.deconv2(out2)))
        out3 = F.relu(self.bn_out3(self.deconv3(out3)))
        h = F.concat((out1, out2, out3), axis=1)
        prob = self.prob_conv(h)
        reg = self.reg_conv(h)
        return prob, reg


class OrigVoxelnet(BasicModel):

    """Voxelnet original Implementation"""

    def __init__(self, config, pretrained_model=None):
        super(OrigVoxelnet, self).__init__()
        with self.init_scope():
            self.feature_net = OrigFeatureVoxelNet()
            self.middle_conv = OrigMiddleLayers()
            self.rpn = OrigRegionProposalNet()

        self.k = config['k']
        self.d = config['d']
        self.h = config['h']
        self.w = config['w']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.p = config['p']
        self.num_negative = None
        self.aug_gt = False if not 'aug_gt' in config else config['aug_gt']

        if pretrained_model['download']:
            if not os.path.exists(pretrained_model['download'].split("/")[-1]):
                subprocess.call(['wget', pretrained_model['download']])

        if pretrained_model['path']:
            chainer.serializers.load_npz(pretrained_model['path'], self,
                                         strict=False)
