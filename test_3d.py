"""Train and val."""
import logging
import os
import time
from tqdm import tqdm
import torch
import subprocess

from utils.config import FLAGS, _ENV_EXPAND
from utils.common import get_params_by_name
from utils.common import set_random_seed
from utils.common import create_exp_dir
from utils.common import setup_logging
from utils.common import save_status
from utils.common import get_device
from utils.common import extract_item
from utils.common import get_data_queue_size
from utils.common import bn_calibration
from utils import dataflow
from utils import optim
from utils import distributed as udist
from utils import prune
from mmseg import seg_dataflow
from mmseg.loss import CrossEntropyLoss, JointsMSELoss, accuracy_keypoint
from kitti_dataset import Kitti, get_dataloader
import models.mobilenet_base as mb
import common as mc
from mmseg.validation import SegVal, keypoint_val
from point_loss import Loss
import numpy as np
from point_utils import setup_seed, keep_bbox_from_image_range, \
    keep_bbox_from_lidar_range, write_pickle, write_label, \
    iou2d, iou3d_camera, iou_bev, bbox3d2corners_camera, points_camera2image, vis_img_3d
import cv2



def shrink_model(model_wrapper,
                 ema,
                 optimizer,
                 prune_info,
                 threshold=1e-3,
                 ema_only=False):
    r"""Dynamic network shrinkage to discard dead atomic blocks.

    Args:
        model_wrapper: model to be shrinked.
        ema: An instance of `ExponentialMovingAverage`, could be None.
        optimizer: Global optimizer.
        prune_info: An instance of `PruneInfo`, could be None.
        threshold: A small enough constant.
        ema_only: If `True`, regard an atomic block as dead only when
            `$$\hat{alpha} \le threshold$$`. Otherwise use both current value
            and momentum version.
    """
    model = mc.unwrap_model(model_wrapper)
    for block_name, block in model.get_named_block_list().items():  # inverted residual blocks
        assert isinstance(block, mb.InvertedResidualChannels)
        masks = [
            bn.weight.detach().abs() > threshold
            for bn in block.get_depthwise_bn()
        ]
        if ema is not None:
            masks_ema = [
                ema.average('{}.{}.weight'.format(
                    block_name, name)).detach().abs() > threshold
                for name in block.get_named_depthwise_bn().keys()
            ]
            if not ema_only:
                masks = [
                    mask0 | mask1 for mask0, mask1 in zip(masks, masks_ema)
                ]
            else:
                masks = masks_ema
        block.compress_by_mask(masks,
                               ema=ema,
                               optimizer=optimizer,
                               prune_info=prune_info,
                               prefix=block_name,
                               verbose=False)

    if optimizer is not None:
        assert set(optimizer.param_groups[0]['params']) == set(
            model.parameters())

    mc.model_profiling(model,
                       FLAGS.image_size,
                       FLAGS.image_size,
                       num_forwards=0,
                       verbose=False)
    if udist.is_master():
        logging.info('Model Shrink to FLOPS: {}'.format(model.n_macs))
        logging.info('Current model: {}'.format(mb.output_network(model)))


def get_prune_weights(model, use_transformer=False):
    """Get variables for pruning."""
    # ['features.2.ops.0.1.1.weight', 'features.2.ops.1.1.1.weight', 'features.2.ops.2.1.1.weight'...]
    if use_transformer:
        return get_params_by_name(mc.unwrap_model(model), FLAGS._bn_to_prune_transformer.weight)
    return get_params_by_name(mc.unwrap_model(model), FLAGS._bn_to_prune.weight)


@udist.master_only
def summary_bn(model, prefix):
    """Summary BN's weights."""
    weights = get_prune_weights(model)
    for name, param in zip(FLAGS._bn_to_prune.weight, weights):
        mc.summary_writer.add_histogram(
            '{}/{}/{}'.format(prefix, 'bn_scale', name), param.detach(),
            FLAGS._global_step)
    if len(FLAGS._bn_to_prune.weight) > 0:
        mc.summary_writer.add_histogram(
            '{}/bn_scale/all'.format(prefix),
            torch.cat([weight.detach() for weight in weights]),
            FLAGS._global_step)


@udist.master_only
def log_pruned_info(model, flops_pruned, infos, prune_threshold):
    """Log pruning-related information."""
    if udist.is_master():
        logging.info('Flops threshold: {}'.format(prune_threshold))
        for info in infos:
            if FLAGS.prune_params['logging_verbose']:
                logging.info(
                    'layer {}, total channel: {}, pruned channel: {}, flops'
                    ' total: {}, flops pruned: {}, pruned rate: {:.3f}'.format(
                        *info))
            mc.summary_writer.add_scalar(
                'prune_ratio/{}/{}'.format(prune_threshold, info[0]), info[-1],
                FLAGS._global_step)
        logging.info('Pruned model: {}'.format(
            prune.output_searched_network(model, infos, FLAGS.prune_params)))

    flops_remain = model.n_macs - flops_pruned
    if udist.is_master():
        logging.info(
            'Prune threshold: {}, flops pruned: {}, flops remain: {}'.format(
                prune_threshold, flops_pruned, flops_remain))
        mc.summary_writer.add_scalar('prune/flops/{}'.format(prune_threshold),
                                     flops_remain, FLAGS._global_step)

def do_eval(det_results, gt_results, CLASSES, saved_path):
    '''
    det_results: list,
    gt_results: dict(id -> det_results)
    CLASSES: dict
    '''
    print(len(det_results), len(gt_results))

    assert len(det_results) == len(gt_results)
    f = open(os.path.join(saved_path, 'eval_results.txt'), 'w')

    # 1. calculate iou
    ious = {
        'bbox_2d': [],
        'bbox_bev': [],
        'bbox_3d': []
    }
    ids = list(sorted(gt_results.keys()))
    for id in ids:
        gt_result = gt_results[id]['annos']
        det_result = det_results[id]

        # 1.1, 2d bboxes iou
        gt_bboxes2d = gt_result['bbox'].astype(np.float32)
        det_bboxes2d = det_result['bbox'].astype(np.float32)
        iou2d_v = iou2d(torch.from_numpy(gt_bboxes2d).cuda(), torch.from_numpy(det_bboxes2d).cuda())
        ious['bbox_2d'].append(iou2d_v.cpu().numpy())

        # 1.2, bev iou
        gt_location = gt_result['location'].astype(np.float32)
        gt_dimensions = gt_result['dimensions'].astype(np.float32)
        gt_rotation_y = gt_result['rotation_y'].astype(np.float32)
        det_location = det_result['location'].astype(np.float32)
        det_dimensions = det_result['dimensions'].astype(np.float32)
        det_rotation_y = det_result['rotation_y'].astype(np.float32)

        gt_bev = np.concatenate([gt_location[:, [0, 2]], gt_dimensions[:, [0, 2]], gt_rotation_y[:, None]], axis=-1)
        det_bev = np.concatenate([det_location[:, [0, 2]], det_dimensions[:, [0, 2]], det_rotation_y[:, None]], axis=-1)
        iou_bev_v = iou_bev(torch.from_numpy(gt_bev).cuda(), torch.from_numpy(det_bev).cuda())
        ious['bbox_bev'].append(iou_bev_v.cpu().numpy())

        # 1.3, 3dbboxes iou
        gt_bboxes3d = np.concatenate([gt_location, gt_dimensions, gt_rotation_y[:, None]], axis=-1)
        det_bboxes3d = np.concatenate([det_location, det_dimensions, det_rotation_y[:, None]], axis=-1)
        iou3d_v = iou3d_camera(torch.from_numpy(gt_bboxes3d).cuda(), torch.from_numpy(det_bboxes3d).cuda())
        ious['bbox_3d'].append(iou3d_v.cpu().numpy())

    MIN_IOUS = {
        'Pedestrian': [0.5, 0.5, 0.5],
        'Cyclist': [0.5, 0.5, 0.5],
        'Car': [0.7, 0.7, 0.7]
    }
    MIN_HEIGHT = [40, 25, 25]

    overall_results = {}
    for e_ind, eval_type in enumerate(['bbox_2d', 'bbox_bev', 'bbox_3d']):
        eval_ious = ious[eval_type]
        eval_ap_results, eval_aos_results = {}, {}
        for cls in CLASSES:
            eval_ap_results[cls] = []
            eval_aos_results[cls] = []
            CLS_MIN_IOU = MIN_IOUS[cls][e_ind]
            for difficulty in [0, 1, 2]:
                # 1. bbox property
                total_gt_ignores, total_det_ignores, total_dc_bboxes, total_scores = [], [], [], []
                total_gt_alpha, total_det_alpha = [], []
                for id in ids:
                    gt_result = gt_results[id]['annos']
                    det_result = det_results[id]

                    # 1.1 gt bbox property
                    cur_gt_names = gt_result['name']
                    cur_difficulty = gt_result['difficulty']
                    gt_ignores, dc_bboxes = [], []
                    for j, cur_gt_name in enumerate(cur_gt_names):
                        ignore = cur_difficulty[j] < 0 or cur_difficulty[j] > difficulty
                        if cur_gt_name == cls:
                            valid_class = 1
                        elif cls == 'Pedestrian' and cur_gt_name == 'Person_sitting':
                            valid_class = 0
                        elif cls == 'Car' and cur_gt_name == 'Van':
                            valid_class = 0
                        else:
                            valid_class = -1
                        
                        if valid_class == 1 and not ignore:
                            gt_ignores.append(0)
                        elif valid_class == 0 or (valid_class == 1 and ignore):
                            gt_ignores.append(1)
                        else:
                            gt_ignores.append(-1)
                        
                        if cur_gt_name == 'DontCare':
                            dc_bboxes.append(gt_result['bbox'][j])
                    total_gt_ignores.append(gt_ignores)
                    total_dc_bboxes.append(np.array(dc_bboxes))
                    total_gt_alpha.append(gt_result['alpha'])

                    # 1.2 det bbox property
                    cur_det_names = det_result['name']
                    cur_det_heights = det_result['bbox'][:, 3] - det_result['bbox'][:, 1]
                    det_ignores = []
                    for j, cur_det_name in enumerate(cur_det_names):
                        if cur_det_heights[j] < MIN_HEIGHT[difficulty]:
                            det_ignores.append(1)
                        elif cur_det_name == cls:
                            det_ignores.append(0)
                        else:
                            det_ignores.append(-1)
                    total_det_ignores.append(det_ignores)
                    total_scores.append(det_result['score'])
                    total_det_alpha.append(det_result['alpha'])

                # 2. calculate scores thresholds for PR curve
                tp_scores = []
                for i, id in enumerate(ids):
                    cur_eval_ious = eval_ious[i]
                    gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                    scores = total_scores[i]

                    nn, mm = cur_eval_ious.shape
                    assigned = np.zeros((mm, ), dtype=np.bool_)
                    for j in range(nn):
                        if gt_ignores[j] == -1:
                            continue
                        match_id, match_score = -1, -1
                        for k in range(mm):
                            if not assigned[k] and det_ignores[k] >= 0 and cur_eval_ious[j, k] > CLS_MIN_IOU and scores[k] > match_score:
                                match_id = k
                                match_score = scores[k]
                        if match_id != -1:
                            assigned[match_id] = True
                            if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                tp_scores.append(match_score)
                total_num_valid_gt = np.sum([np.sum(np.array(gt_ignores) == 0) for gt_ignores in total_gt_ignores])
                score_thresholds = get_score_thresholds(tp_scores, total_num_valid_gt)    
            
                # 3. draw PR curve and calculate mAP
                tps, fns, fps, total_aos = [], [], [], []

                for score_threshold in score_thresholds:
                    tp, fn, fp = 0, 0, 0
                    aos = 0
                    for i, id in enumerate(ids):
                        cur_eval_ious = eval_ious[i]
                        gt_ignores, det_ignores = total_gt_ignores[i], total_det_ignores[i]
                        gt_alpha, det_alpha = total_gt_alpha[i], total_det_alpha[i]
                        scores = total_scores[i]

                        nn, mm = cur_eval_ious.shape
                        assigned = np.zeros((mm, ), dtype=np.bool_)
                        for j in range(nn):
                            if gt_ignores[j] == -1:
                                continue
                            match_id, match_iou = -1, -1
                            for k in range(mm):
                                if not assigned[k] and det_ignores[k] >= 0 and scores[k] >= score_threshold and cur_eval_ious[j, k] > CLS_MIN_IOU:
    
                                    if det_ignores[k] == 0 and cur_eval_ious[j, k] > match_iou:
                                        match_iou = cur_eval_ious[j, k]
                                        match_id = k
                                    elif det_ignores[k] == 1 and match_iou == -1:
                                        match_id = k

                            if match_id != -1:
                                assigned[match_id] = True
                                if det_ignores[match_id] == 0 and gt_ignores[j] == 0:
                                    tp += 1
                                    if eval_type == 'bbox_2d':
                                        aos += (1 + np.cos(gt_alpha[j] - det_alpha[match_id])) / 2
                            else:
                                if gt_ignores[j] == 0:
                                    fn += 1
                            
                        for k in range(mm):
                            if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                fp += 1
                        
                        # In case 2d bbox evaluation, we should consider dontcare bboxes
                        if eval_type == 'bbox_2d':
                            dc_bboxes = total_dc_bboxes[i]
                            det_bboxes = det_results[id]['bbox']
                            if len(dc_bboxes) > 0:
                                ious_dc_det = iou2d(torch.from_numpy(det_bboxes), torch.from_numpy(dc_bboxes), metric=1).numpy().T
                                for j in range(len(dc_bboxes)):
                                    for k in range(len(det_bboxes)):
                                        if det_ignores[k] == 0 and scores[k] >= score_threshold and not assigned[k]:
                                            if ious_dc_det[j, k] > CLS_MIN_IOU:
                                                fp -= 1
                                                assigned[k] = True
                            
                    tps.append(tp)
                    fns.append(fn)
                    fps.append(fp)
                    if eval_type == 'bbox_2d':
                        total_aos.append(aos)

                tps, fns, fps = np.array(tps), np.array(fns), np.array(fps)

                recalls = tps / (tps + fns)
                precisions = tps / (tps + fps)
                for i in range(len(score_thresholds)):
                    precisions[i] = np.max(precisions[i:])
                
                sums_AP = 0
                for i in range(0, len(score_thresholds), 4):
                    sums_AP += precisions[i]
                mAP = sums_AP / 11 * 100
                eval_ap_results[cls].append(mAP)

                if eval_type == 'bbox_2d':
                    total_aos = np.array(total_aos)
                    similarity = total_aos / (tps + fps)
                    for i in range(len(score_thresholds)):
                        similarity[i] = np.max(similarity[i:])
                    sums_similarity = 0
                    for i in range(0, len(score_thresholds), 4):
                        sums_similarity += similarity[i]
                    mSimilarity = sums_similarity / 11 * 100
                    eval_aos_results[cls].append(mSimilarity)

        print(f'=========={eval_type.upper()}==========')
        print(f'=========={eval_type.upper()}==========', file=f)
        for k, v in eval_ap_results.items():
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
            print(f'{k} AP@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        if eval_type == 'bbox_2d':
            print(f'==========AOS==========')
            print(f'==========AOS==========', file=f)
            for k, v in eval_aos_results.items():
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
                print(f'{k} AOS@{MIN_IOUS[k][e_ind]}: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        
        overall_results[eval_type] = np.mean(list(eval_ap_results.values()), 0)
        if eval_type == 'bbox_2d':
            overall_results['AOS'] = np.mean(list(eval_aos_results.values()), 0)
    
    print(f'\n==========Overall==========')
    print(f'\n==========Overall==========', file=f)
    for k, v in overall_results.items():
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}')
        print(f'{k} AP: {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
    f.close()

def get_score_thresholds(tp_scores, total_num_valid_gt, num_sample_pts=41):
    score_thresholds = []
    tp_scores = sorted(tp_scores)[::-1]
    cur_recall, pts_ind = 0, 0
    for i, score in enumerate(tp_scores):
        lrecall = (i + 1) / total_num_valid_gt
        rrecall = (i + 2) / total_num_valid_gt

        if i == len(tp_scores) - 1:
            score_thresholds.append(score)
            break

        if (lrecall + rrecall) / 2 < cur_recall:
            continue

        score_thresholds.append(score)
        pts_ind += 1
        cur_recall = pts_ind / (num_sample_pts - 1)
    return score_thresholds


def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  lr_scheduler,
                  ema,
                  rho_scheduler,
                  meters,
                  max_iter=None,
                  phase='train',
                  val_dataset = None):
    
    """Run one epoch."""
    assert phase in [
        'train', 'val', 'test', 'bn_calibration'
    ] or phase.startswith(
        'prune'), "phase not be in train/val/test/bn_calibration/prune."
    
    # 看現在的 phase 是否為 train
    train = phase == 'train'
    
    if train:
        model.train()
    else:
        model.eval()
    
    if phase == 'bn_calibration':
        model.apply(bn_calibration)

    # 需檢查此 dataloader 是否有 sampler 
    if not FLAGS.use_hdfs:
        if FLAGS.use_distributed:
            loader.sampler.set_epoch(epoch)
        
    # for i, data_dict in enumerate(loader):
    #     for key in data_dict:
    #         print(key)
    #     exit()

    results = None
    data_iterator = iter(loader)
    
    if not FLAGS.use_hdfs:
        # if FLAGS.use_distributed:
        #     if FLAGS.dataset == 'coco':
        #         data_fetcher = dataflow.DataPrefetcherKeypoint(data_iterator)
        #     else:
        #         data_fetcher = dataflow.DataPrefetcher(data_iterator)
        # else:
        #     logging.warning('Not use prefetcher')
        data_fetcher = data_iterator
    
    if FLAGS.dataset == 'kitti':
        format_results = {}
    
    image_count = 0

    for batch_idx, data in enumerate(data_fetcher):    
        # 從 data_fetcher 中拿出資料
        
        if FLAGS.dataset == 'kitti':
            # 將資料移動到 cuda
            for key in data:
                for j, item in enumerate(data[key]):
                    if torch.is_tensor(item):
                        data[key][j] = data[key][j].cuda()
            batched_pts = data['batched_pts']
            batched_gt_bboxes = data['batched_gt_bboxes']
            batched_labels = data['batched_labels']
            batched_difficulty = data['batched_difficulty']
            
            
        else:
            input, target = data

        # if batch_idx > 400:
        #     break
        # used for bn calibration
        
        if max_iter is not None:
            assert phase == 'bn_calibration'
            if batch_idx >= max_iter:
                break
        
        if FLAGS.dataset != 'kitti':
            target = target.cuda(non_blocking=True)
        
        if train:
            optimizer.zero_grad()
            rho = rho_scheduler(FLAGS._global_step)
            
            if FLAGS.dataset == 'kitti':
                # 在這邊去計算 loss
                nclass = 3
                loss_func = Loss()

                bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = model(batched_pts = batched_pts, 
                                                                    batched_gt_bboxes = batched_gt_bboxes, 
                                                                    batched_gt_labels = batched_labels)
                
                bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, nclass)
                bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
                bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

                batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
                batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
                batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
                batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)

                pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < nclass)
                bbox_pred = bbox_pred[pos_idx]
                batched_bbox_reg = batched_bbox_reg[pos_idx]
                # sin(a - b) = sin(a)*cos(b) - cos(a)*sin(b)
                bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
                batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
                bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
                batched_dir_labels = batched_dir_labels[pos_idx]

                num_cls_pos = (batched_bbox_labels < nclass).sum()
                bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
                batched_bbox_labels[batched_bbox_labels < 0] = nclass
                batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

                loss_dict = loss_func(bbox_cls_pred=bbox_cls_pred,
                                  bbox_pred=bbox_pred,
                                  bbox_dir_cls_pred=bbox_dir_cls_pred,
                                  batched_labels=batched_bbox_labels, 
                                  num_cls_pos=num_cls_pos, 
                                  batched_bbox_reg=batched_bbox_reg, 
                                  batched_dir_labels=batched_dir_labels)
                
                loss = loss_dict['total_loss']
                # break

            else:
                loss = mc.forward_loss(model, criterion, input, target, meters, task=FLAGS.model_kwparams.task, distill=FLAGS.distill)
            
            # 要進行 prune 的話才會跑這一區
            # if FLAGS.prune_params['method'] is not None:
            #     loss_l2 = optim.cal_l2_loss(model, FLAGS.weight_decay,
            #                                 FLAGS.weight_decay_method)  # manual weight decay
            #     loss_bn_l1 = prune.cal_bn_l1_loss(get_prune_weights(model),
            #                                       FLAGS._bn_to_prune.penalty, rho)
            #     if FLAGS.prune_params.use_transformer:

            #         transformer_weights = get_prune_weights(model, True)
            #         loss_bn_l1 += prune.cal_bn_l1_loss(transformer_weights,
            #                                            FLAGS._bn_to_prune_transformer.penalty, rho)

            #         transformer_dict = []
            #         for name, weight in zip(FLAGS._bn_to_prune_transformer.weight, transformer_weights):
            #             transformer_dict.append(sum(weight > FLAGS.model_shrink_threshold).item())
            #         FLAGS._bn_to_prune_transformer.add_info_list('channels', transformer_dict)
            #         FLAGS._bn_to_prune_transformer.update_penalty()
            #         if udist.is_master() and FLAGS._global_step % FLAGS.log_interval == 0:
            #             logging.info(transformer_dict)
            #             # logging.info(FLAGS._bn_to_prune_transformer.penalty)

            #     meters['loss_l2'].cache(loss_l2)
            #     meters['loss_bn_l1'].cache(loss_bn_l1)
            #     loss = loss + loss_l2 + loss_bn_l1
            
            meters['loss'].cache(loss)
            
            loss.backward()
            
            if FLAGS.use_distributed:
                udist.allreduce_grads(model)

            if FLAGS._global_step % FLAGS.log_interval == 0:
                results = mc.reduce_and_flush_meters(meters)
                if udist.is_master():
                    logging.info('Epoch {}/{} Iter {}/{} Lr: {} {}: '.format(
                        epoch, FLAGS.num_epochs, batch_idx, len(loader), optimizer.param_groups[0]["lr"], phase)
                                 + ', '.join('{}: {:.4f}'.format(k, v)
                                             for k, v in results.items()))
                    for k, v in results.items():
                        mc.summary_writer.add_scalar('{}/{}'.format(phase, k),
                                                     v, FLAGS._global_step)

            if udist.is_master(
            ) and FLAGS._global_step % FLAGS.log_interval == 0:
                mc.summary_writer.add_scalar('train/learning_rate',
                                             optimizer.param_groups[0]['lr'],
                                             FLAGS._global_step)
                
                # if FLAGS.prune_params['method'] is not None:
                #     mc.summary_writer.add_scalar('train/l2_regularize_loss',
                #                                  extract_item(loss_l2),
                #                                  FLAGS._global_step)
                #     mc.summary_writer.add_scalar('train/bn_l1_loss',
                #                                  extract_item(loss_bn_l1),
                #                                  FLAGS._global_step)
                
                mc.summary_writer.add_scalar('prune/rho', rho,
                                             FLAGS._global_step)
                mc.summary_writer.add_scalar(
                    'train/current_epoch',
                    FLAGS._global_step / FLAGS._steps_per_epoch,
                    FLAGS._global_step)
                if FLAGS.data_loader_workers > 0:
                    mc.summary_writer.add_scalar(
                        'data/train/prefetch_size',
                        get_data_queue_size(data_iterator), FLAGS._global_step)

            if udist.is_master(
            ) and FLAGS._global_step % FLAGS.log_interval_detail == 0:
                summary_bn(model, 'train')

            optimizer.step()
            if FLAGS.lr_scheduler == 'poly':
                optim.poly_learning_rate(optimizer,
                                         FLAGS.lr,
                                         epoch * FLAGS._steps_per_epoch + batch_idx + 1,
                                         FLAGS.num_epochs * FLAGS._steps_per_epoch)
            else:
                lr_scheduler.step()
            if FLAGS.use_distributed and FLAGS.allreduce_bn:
                udist.allreduce_bn(model)
            FLAGS._global_step += 1

            # NOTE: after steps count update
            if ema is not None:
                model_unwrap = mc.unwrap_model(model)
                ema_names = ema.average_names()
                params = get_params_by_name(model_unwrap, ema_names)
                for name, param in zip(ema_names, params):
                    ema(name, param, FLAGS._global_step)
            
        else:
            
            if FLAGS.dataset == 'kitti':  # validate for pointpillar
                pcd_limit_range = np.array([0, -40, -3, 70.4, 40, 0.0], dtype=np.float32)
                
                CLASSES = Kitti.CLASSES
                LABEL2CLASSES = {v:k for k, v in CLASSES.items()}
                
                batch_results = model(batched_pts = batched_pts, batched_gt_bboxes = batched_gt_bboxes, batched_gt_labels = batched_labels, mode = 'val')
                
                # print('Predicting and Formatting the results.')
                
                for j, result in enumerate(batch_results):
                    format_result = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'dimensions': [],
                        'location': [],
                        'rotation_y': [],
                        'score': []
                    }
                    
                    calib_info = data['batched_calib_info'][j]
                    tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
                    r0_rect = calib_info['R0_rect'].astype(np.float32)
                    P2 = calib_info['P2'].astype(np.float32)
                    image_shape = data['batched_img_info'][j]['image_shape']
                    idx = data['batched_img_info'][j]['image_idx']
                    path_name = data['batched_img_info'][j]['image_path']
                    
                    img = cv2.imread(os.path.join('./kitti/',path_name))
                    
                    # print(data['batched_img_info'][j])
                    # exit()

                    result_filter = keep_bbox_from_image_range(result, tr_velo_to_cam, r0_rect, P2, image_shape)
                    result_filter = keep_bbox_from_lidar_range(result_filter, pcd_limit_range)
            
                    lidar_bboxes = result_filter['lidar_bboxes']
                    labels, scores = result_filter['labels'], result_filter['scores']
                    bboxes2d, camera_bboxes = result_filter['bboxes2d'], result_filter['camera_bboxes']
                    

                    # 存下視覺化影像：
                    bboxes_corners = bbox3d2corners_camera(camera_bboxes)
                    image_points = points_camera2image(bboxes_corners, P2)
                    img = vis_img_3d(img, image_points, labels, rt=True)

                    cv2.imwrite(os.path.join(FLAGS.log_dir, 'img', path_name.split('/')[-1]), img)
                    image_count+=1

                    
                    for lidar_bbox, label, score, bbox2d, camera_bbox in \
                        zip(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes):
                        format_result['name'].append(LABEL2CLASSES[label])
                        format_result['truncated'].append(0.0)
                        format_result['occluded'].append(0)
                        alpha = camera_bbox[6] - np.arctan2(camera_bbox[0], camera_bbox[2])
                        format_result['alpha'].append(alpha)
                        format_result['bbox'].append(bbox2d)
                        format_result['dimensions'].append(camera_bbox[3:6])
                        format_result['location'].append(camera_bbox[:3])
                        format_result['rotation_y'].append(camera_bbox[6])
                        format_result['score'].append(score)
                    
                    write_label(format_result, os.path.join(FLAGS.log_dir, f'{idx:06d}.txt'))
                    format_results[idx] = {k:np.array(v) for k, v in format_result.items()}
                
            else:
                mc.forward_loss(model, criterion, input, target, meters, task=FLAGS.model_kwparams.task, distill=False)


    if FLAGS.dataset == 'kitti' and phase == 'val':
        write_pickle(format_results, os.path.join(FLAGS.log_dir, 'results.pkl'))
        print('Evaluating.. Please wait several seconds.')
        do_eval(format_results, val_dataset.data_infos, CLASSES, FLAGS.log_dir)


    if not train: 
        results = mc.reduce_and_flush_meters(meters)
        # 處理文字的輸出
        if udist.is_master():
            logging.info(
                'Epoch {}/{} {}: '.format(epoch, FLAGS.num_epochs, phase)
                + ', '.join(
                    '{}: {:.4f}'.format(k, v) for k, v in results.items()))
            for k, v in results.items():
                mc.summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                             FLAGS._global_step)
                
    return results


def train_val_test():
    """Train and val."""
    torch.backends.cudnn.benchmark = True  # For acceleration

    # 初始化 model
    model, model_wrapper = mc.get_model()
    ema = mc.setup_ema(model)

    # 處理 loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    criterion_smooth = optim.CrossEntropyLabelSmooth(
        FLAGS.model_kwparams['num_classes'],
        FLAGS['label_smoothing'],
        reduction='mean').cuda()
    
    if FLAGS.dataset == 'ade20k' or FLAGS.dataset == 'cityscapes': # model.task == 'segmentation':
        criterion = CrossEntropyLoss().cuda()
        criterion_smooth = CrossEntropyLoss().cuda()
    
    if FLAGS.dataset == 'coco':
        criterion = JointsMSELoss(use_target_weight=True).cuda()
        criterion_smooth = JointsMSELoss(use_target_weight=True).cuda()

    if FLAGS.get('log_graph_only', False):
        if udist.is_master():
            _input = torch.zeros(1, 3, FLAGS.image_size,
                                 FLAGS.image_size).cuda()
            _input = _input.requires_grad_(True)
            if isinstance(model_wrapper, (torch.nn.DataParallel, udist.AllReduceDistributedDataParallel)):
                mc.summary_writer.add_graph(model_wrapper.module, (_input,), verbose=True)
            else:
                mc.summary_writer.add_graph(model_wrapper, (_input,), verbose=True)
        return

    # 是否使用預訓練權重
    if FLAGS.pretrained:
        print("use pretrain")
        checkpoint = torch.load(FLAGS.pretrained,
                                map_location=lambda storage, loc: storage)
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        # update keys from external models
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if (hasattr(FLAGS, 'pretrained_model_remap_keys')
                and FLAGS.pretrained_model_remap_keys):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                if udist.is_master():
                    logging.info('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        if udist.is_master():
            logging.info('Loaded model {}.'.format(FLAGS.pretrained))
    
    # 初始化優化器
    optimizer = optim.get_optimizer(model_wrapper, FLAGS)

    # 從預訓練權重開始訓練
    if FLAGS.resume:
        print("Resume")
        # 載入checkpoint
        checkpoint = torch.load(os.path.join(FLAGS.resume,
                                             'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper = checkpoint['model'].cuda()
        model = model_wrapper.module

        optimizer = checkpoint['optimizer']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # model_wrapper.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        
        if ema:
            # ema.load_state_dict(checkpoint['ema'])
            ema = checkpoint['ema'].cuda()
            ema.to(get_device(model))
        
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS, last_epoch=(last_epoch + 1) * FLAGS._steps_per_epoch)
        lr_scheduler.last_epoch = (last_epoch + 1) * FLAGS._steps_per_epoch
        best_val = extract_item(checkpoint['best_val'])
        train_meters, val_meters = checkpoint['meters']
        FLAGS._global_step = (last_epoch + 1) * FLAGS._steps_per_epoch
        if udist.is_master():
            logging.info('Loaded checkpoint {} at epoch {}.'.format(
                FLAGS.resume, last_epoch))
    
    else: # 不是從預訓練權重開始訓練> 設定一些 optimizor 參數
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        # last_epoch = lr_scheduler.last_epoch
        
        last_epoch = -1
        best_val = 1.
        
        if not FLAGS.distill:
            train_meters = mc.get_meters('train', FLAGS.prune_params['method'])
            val_meters = mc.get_meters('val')
        
        else:
            train_meters = mc.get_distill_meters('train', FLAGS.prune_params['method'])
            val_meters = mc.get_distill_meters('val')
        
        if FLAGS.model_kwparams.task == 'segmentation':
            best_val = 0.
            if not FLAGS.distill:
                train_meters = mc.get_seg_meters('train', FLAGS.prune_params['method'])
                val_meters = mc.get_seg_meters('val')
            else:
                train_meters = mc.get_seg_distill_meters('train', FLAGS.prune_params['method'])
                val_meters = mc.get_seg_distill_meters('val')
        FLAGS._global_step = 0

    if not FLAGS.resume and udist.is_master():
        logging.info(model_wrapper)
    assert FLAGS.profiling, '`m.macs` is used for calculating penalty'
    
    # if udist.is_master():
    #     model.apply(lambda m: print(m))
    
    if FLAGS.profiling:
        if 'gpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=False)
    
    
    # 處理資料集以及loader
    if FLAGS.dataset == 'cityscapes':
        (train_set, val_set, test_set) = seg_dataflow.cityscapes_datasets(FLAGS)
        segval = SegVal(num_classes=19)
        
    elif FLAGS.dataset == 'ade20k':
        (train_set, val_set, test_set) = seg_dataflow.ade20k_datasets(FLAGS)
        segval = SegVal(num_classes=150)
    
    elif FLAGS.dataset == 'coco':
        (train_set, val_set, test_set) = seg_dataflow.coco_datasets(FLAGS)
        segval = None
    
    elif FLAGS.dataset == 'kitti':    
        train_set   = Kitti(data_root='./kitti/', split='train')
        val_set     = Kitti(data_root='./kitti/', split='val')
        test_set    = Kitti(data_root='./kitti/', split='test')
        segval = None

    else: # classification
        (train_transforms, val_transforms, test_transforms) = dataflow.data_transforms(FLAGS)
        (train_set, val_set, test_set) = dataflow.dataset(train_transforms,
                                                          val_transforms,
                                                          test_transforms, FLAGS)
        segval = None
    
    (train_loader, calib_loader, val_loader, test_loader) = dataflow.data_loader(train_set, val_set, test_set, FLAGS)
    
    # get bn's weights
    if FLAGS.prune_params.use_transformer:
        FLAGS._bn_to_prune, FLAGS._bn_to_prune_transformer = prune.get_bn_to_prune(model, FLAGS.prune_params)
    else:
        FLAGS._bn_to_prune = prune.get_bn_to_prune(model, FLAGS.prune_params)

    rho_scheduler = prune.get_rho_scheduler(FLAGS.prune_params,
                                            FLAGS._steps_per_epoch)

    # 如果只有需要 validate 的話
    # if FLAGS.test_only and (test_loader is not None):
    if udist.is_master():
        logging.info('Start testing.')
    test_meters = mc.get_meters('test')
    validate(last_epoch, calib_loader, val_loader, criterion, val_meters,
                model_wrapper, ema, 'val', segval, val_set)
    return

    # already broadcast by AllReduceDistributedDataParallel
    # optimizer load same checkpoint/same initialization

    if udist.is_master():
        logging.info('Start training.')

    # for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        
    #     # train > 跑一個 epoch
    #     results = run_one_epoch(epoch,
    #                             train_loader,
    #                             model_wrapper,
    #                             criterion_smooth,
    #                             optimizer,
    #                             lr_scheduler,
    #                             ema,
    #                             rho_scheduler,
    #                             train_meters,
    #                             phase='train')
        
    #     # 如果現在的epoch 達到特定的epoch 數就會進行 validation
    #     if (epoch + 1) % FLAGS.eval_interval == 0:
    #         print("start validation")
    #         # 進行validation
    #         results, model_eval_wrapper = validate(epoch, calib_loader, val_loader,
    #                                                criterion, val_meters,
    #                                                model_wrapper, ema, 'val', segval, val_set)
    #         # 是否要進行 pruning
    #         if FLAGS.prune_params['method'] is not None and FLAGS.prune_params['bn_prune_filter'] is not None:
    #             prune_threshold = FLAGS.model_shrink_threshold  # 1e-3
    #             masks = prune.cal_mask_network_slimming_by_threshold(
    #                 get_prune_weights(model_eval_wrapper), prune_threshold)  # get mask for all bn weights (depth-wise)
    #             FLAGS._bn_to_prune.add_info_list('mask', masks)
    #             flops_pruned, infos = prune.cal_pruned_flops(FLAGS._bn_to_prune)
    #             log_pruned_info(mc.unwrap_model(model_eval_wrapper), flops_pruned,
    #                             infos, prune_threshold)
    #             if not FLAGS.distill:
    #                 if flops_pruned >= FLAGS.model_shrink_delta_flops \
    #                         or epoch == FLAGS.num_epochs - 1:
    #                     ema_only = (epoch == FLAGS.num_epochs - 1)
    #                     shrink_model(model_wrapper, ema, optimizer, FLAGS._bn_to_prune,
    #                                  prune_threshold, ema_only)
            
    #         model_kwparams = mb.output_network(mc.unwrap_model(model_wrapper))
            
    #         # 只有 master 的 process 會進行此動作:存模型權重
    #         if udist.is_master(): 
    #             if FLAGS.model_kwparams.task == 'classification' and results['top1_error'] < best_val:
    #                 best_val = results['top1_error']
    #                 logging.info('New best validation top1 error: {:.4f}'.format(best_val))
    #                 save_status(model_wrapper, model_kwparams, optimizer, ema,
    #                             epoch, best_val, (train_meters, val_meters),
    #                             os.path.join(FLAGS.log_dir, 'best_model'))

    #             elif FLAGS.model_kwparams.task == 'segmentation' and FLAGS.dataset != 'coco' and results[
    #                 'mIoU'] > best_val:
    #                 best_val = results['mIoU']
    #                 logging.info('New seg mIoU: {:.4f}'.format(best_val))
    #                 save_status(model_wrapper, model_kwparams, optimizer, ema,
    #                             epoch, best_val, (train_meters, val_meters),
    #                             os.path.join(FLAGS.log_dir, 'best_model'))
                    
    #             elif FLAGS.dataset == 'coco' and results > best_val:
    #                 best_val = results
    #                 logging.info('New Result: {:.4f}'.format(best_val))
    #                 save_status(model_wrapper, model_kwparams, optimizer, ema,
    #                             epoch, best_val, (train_meters, val_meters),
    #                             os.path.join(FLAGS.log_dir, 'best_model'))

    #             # save latest checkpoint
    #             save_status(model_wrapper, model_kwparams, optimizer, ema, epoch,
    #                         best_val, (train_meters, val_meters),
    #                         os.path.join(FLAGS.log_dir, 'latest_checkpoint'))

    return


def validate(epoch, calib_loader, val_loader, criterion, val_meters,
             model_wrapper, ema, phase, segval=None, val_set=None):
    """Calibrate and validate."""
    assert phase in ['test', 'val']
    model_eval_wrapper = mc.get_ema_model(ema, model_wrapper)

    # bn_calibration
    # if FLAGS.prune_params['method'] is not None:
    #     if FLAGS.get('bn_calibration', False):
    #         if not FLAGS.use_distributed:
    #             logging.warning(
    #                 'Only GPU0 is used when calibration when use DataParallel')
    #         with torch.no_grad():
    #             _ = run_one_epoch(epoch,
    #                               calib_loader,
    #                               model_eval_wrapper,
    #                               criterion,
    #                               None,
    #                               None,
    #                               None,
    #                               None,
    #                               val_meters,
    #                               max_iter=FLAGS.bn_calibration_steps,
    #                               phase='bn_calibration', val_dataset=val_set)
    #         if FLAGS.use_distributed:
    #             udist.allreduce_bn(model_eval_wrapper)

    # val
    with torch.no_grad():
        if FLAGS.model_kwparams.task == 'segmentation':
            if FLAGS.dataset == 'coco':
                results = 0
                if udist.is_master():
                    results = keypoint_val(val_set, val_loader, model_eval_wrapper.module, criterion)
            else:
                assert segval is not None
                results = segval.run(epoch,
                                     val_loader,
                                     model_eval_wrapper.module if FLAGS.single_gpu_test else model_eval_wrapper,
                                     FLAGS)
        else:
            results = run_one_epoch(epoch,
                                    val_loader,
                                    model_eval_wrapper,
                                    criterion,
                                    None,
                                    None,
                                    None,
                                    None,
                                    val_meters,
                                    phase=phase, val_dataset=val_set)
    summary_bn(model_eval_wrapper, phase)
    return results, model_eval_wrapper


def main():
    """Entry."""
    NUM_IMAGENET_TRAIN = 1281167
    if FLAGS.dataset == 'cityscapes':
        NUM_IMAGENET_TRAIN = 2975
    elif FLAGS.dataset == 'ade20k':
        NUM_IMAGENET_TRAIN = 20210
    elif FLAGS.dataset == 'coco':
        NUM_IMAGENET_TRAIN = 149813
    mc.setup_distributed(NUM_IMAGENET_TRAIN)

    if FLAGS.net_params and FLAGS.model_kwparams.task == 'segmentation':
        tag, input_channels, block1, block2, block3, block4, last_channel = FLAGS.net_params.split('-')
        input_channels = [int(item) for item in input_channels.split('_')]
        block1 = [int(item) for item in block1.split('_')]
        block2 = [int(item) for item in block2.split('_')]
        block3 = [int(item) for item in block3.split('_')]
        block4 = [int(item) for item in block4.split('_')]
        last_channel = int(last_channel)

        inverted_residual_setting = []
        for item in [block1, block2, block3, block4]:
            for _ in range(item[0]):
                inverted_residual_setting.append(
                    [item[1], item[2:-int(len(item) / 2 - 1)], item[-int(len(item) / 2 - 1):]])

        FLAGS.model_kwparams.input_channel = input_channels
        FLAGS.model_kwparams.inverted_residual_setting = inverted_residual_setting
        FLAGS.model_kwparams.last_channel = last_channel

    if udist.is_master():
        FLAGS.log_dir = '{}/{}'.format(FLAGS.log_dir,
                                       time.strftime("%Y%m%d-%H%M%S"))
        # yapf: disable
        # create_exp_dir(FLAGS.log_dir, FLAGS.config_path, blacklist_dirs=['exp', '.git', 'pretrained', 'tmp', 'deprecated', 'bak', 'output'])
        # yapf: enable
        setup_logging(FLAGS.log_dir)
        for k, v in _ENV_EXPAND.items():
            logging.info('Env var expand: {} to {}'.format(k, v))
        logging.info(FLAGS)

    set_random_seed(FLAGS.get('random_seed', 0))
    with mc.SummaryWriterManager():
        train_val_test()


if __name__ == "__main__":
    main()