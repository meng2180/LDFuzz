# _*_coding:utf-8_*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import pickle
from functools import partial

import numpy as np
import numba
from torch import no_grad
from torch.cuda import empty_cache

import shapely
from shapely.geometry import Polygon

from pcdet.utils import box_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils

from _lib.queue.queue import FuzzQueue
from _others.velodyne_mutators import Mutator
from utils import config, DUMPS_utlis, model_utils
from utils.coverage_utils import cal_single_c1, cal_single_c2, get_gt_polygon, get_scene_graph_encode


@numba.jit(nopython=True)
def boxes_bev_centerpoint_distance(boxes_a: np.ndarray, boxes_b: np.ndarray):
    point_a = boxes_a[:, :2]
    point_b = boxes_b[:, :2]
    ans = np.zeros((point_a.shape[0], point_b.shape[0]), dtype=np.float32)
    for index, b in enumerate(point_b):
        x = point_a - b
        x = x**2
        x = x.sum(axis=-1)
        x = np.sqrt(x)
        ans[:, index] = x
    return ans


def lidar_mutation_functionV1(dataset_name, criteria=None):

    mutator = Mutator(dataset_name, criteria)
    METHOD = mutator.generate_method_list()

    def mutate(batch, method_list=None):
        if method_list is None:
            method_list = np.array(config.object_level + config.scene_level)
        elif not isinstance(method_list, np.ndarray):
            method_list = np.array(method_list)

        method_list = np.random.permutation(method_list)

        for method in method_list:
            if method in config.scene_level and batch[
                    'weather_type'] in config.scene_level:
                continue

            batch['method'] = method
            print(f'  select method : {method}', end='')

            try:
                result = METHOD[method](batch)
                print(' -- ok')
                return result
            except:
                print(' -- failed')

        print('  all methods failed.')
        return batch

    return mutate


def lidar_mutation_functionV2(queue: FuzzQueue, dataset_name, criteria=None):

    mutator = Mutator(dataset_name, criteria)
    METHOD = mutator.generate_method_list()

    def mutate():
        while True:
            seed = queue.select_next(1)[0]
            parent = seed.parent

            batch = pickle.load(open(seed.fname, 'rb'))

            while isinstance(batch, list):
                batch = batch[0]
            print(
                f'select seed id : {seed.id}, parent {parent.id if parent is not None else None}'
            )

            method_list = np.random.permutation(config.object_level +
                                                config.scene_level)
            for method in method_list:
                if method in config.scene_level and batch[
                        'weather_type'] in config.scene_level:
                    continue

                batch['method'] = method
                print(f'  select method : {method}', end='')

                try:
                    return method, seed, METHOD[method](batch,
                                                        DUMPS=queue.DUMPS)[0]
                except Exception as e:
                    print(f' -- failed')
                    continue

            print('  choose another seed.')

    return mutate


def build_objective_function(args):

    def func(seed, data_batch):
        pd_boxes = data_batch['pred_boxes']
        ground_truth = data_batch['selected_gt_boxes']
        method = data_batch['method']

        pred_labels = data_batch['pred_labels']

        if args.use_distance:
            iou = boxes_bev_centerpoint_distance(ground_truth, pd_boxes)
        else:
            iou = iou3d_nms_utils.boxes_bev_iou_cpu(ground_truth, pd_boxes)

        if args.use_distance:
            if iou.shape[0] != 0:
                fp_list = np.min(iou, axis=0).reshape(-1)
            else:
                fp_list = np.array([1000000] * iou.shape[1], dtype=np.float32)
            fp_list = np.where(fp_list > args.distance_threshold)[0]

            if iou.shape[1] != 0:
                iou = np.min(iou, axis=1).reshape(-1)
            else:
                iou = np.array([1000000] * iou.shape[0], dtype=np.float32)
            fn_list = np.where(iou > args.distance_threshold)[0]
        else:
            if iou.shape[0] != 0:
                fp_list = np.max(iou, axis=0).reshape(-1)
            else:
                fp_list = np.zeros(iou.shape[1], dtype=np.float32)
            fp_list = np.where(fp_list < args.iou_threshold)[0]

            if iou.shape[1] != 0:
                iou = np.max(iou, axis=1).reshape(-1)
            else:
                iou = np.zeros(iou.shape[0], dtype=np.float32)
            fn_list = np.where(iou < args.iou_threshold)[0]

        if args.use_distance:
            tp = np.sum(iou <= args.distance_threshold)
        else:
            tp = np.sum(iou >= args.iou_threshold)

        fp = len(pd_boxes) - tp
        fn = len(ground_truth) - tp
        data_batch['iou_tp'] = tp

        print(f'Predict -- TP : {tp} FP : {fp} FN : {fn}')

        fn_list = np.where(iou < args.iou_threshold)[0]
        # fn_list = data_batch['fn_list'] if 'fn_list' in data_batch else None
        # fn_list = \
        #     np.concatenate([
        #         fn_list,
        #         np.where(iou < 0.5)[0]
        #     ]) if fn_list is not None else\
        #     np.where(iou < 0.5)[0]
        # fn_list = np.unique(fn_list)
        # data_batch['fn_list'] = fn_list

        args.DUMPS['fp'][method][-1] += fp
        args.DUMPS['fn'][method][-1] += fn

        s_gt_boxes = data_batch['selected_gt_boxes']
        s_name = data_batch['selected_name']
        weather_type = data_batch['weather_type']
        sg_type = data_batch['scene_graph_type']
        data_batch['level'] += 1

        scene = seed.root_seed.split('.')[0]
        dict_scene = args.DUMPS['scene'][scene]
        dict_cluster = args.DUMPS['cluster'][sg_type]

        dict_scene['method_times'][method] += 1
        dict_scene['total_times'] += 1

        gt_polygon = get_gt_polygon(s_gt_boxes)
        err_polygon = get_gt_polygon(s_gt_boxes[fn_list])
        if err_polygon is None: err_polygon = Polygon()

        sg_encode = get_scene_graph_encode(args.dataset_name, s_gt_boxes,
                                           s_name, weather_type, sg_type)
        err_sg_encode = get_scene_graph_encode(args.dataset_name,
                                               s_gt_boxes[fn_list],
                                               s_name[fn_list], weather_type,
                                               sg_type)

        args.DUMPS['error_detail'].append({
            'scene': scene,
            'scene_graph_type': sg_type,
            'method': method,
            'weather_type': weather_type,
            'gt_boxes': s_gt_boxes,
            'fn_list': fn_list,
            'fp_list': fp_list,
            'sg_encode': sg_encode,
            'err_sg_encode': err_sg_encode
        })

        ref_gt_polygon = dict_scene['gt_polygon']
        ref_cluster = dict_cluster['cluster']
        ref_err_polygon = dict_scene['error_polygon']
        ref_err_cluster = dict_cluster['error_cluster']

        gt_polygon = shapely.union(ref_gt_polygon, gt_polygon)

        cluster = copy.deepcopy(ref_cluster)
        cluster.add(sg_encode)

        err_polygon = shapely.union(ref_err_polygon, err_polygon)

        err_cluster = copy.deepcopy(ref_err_cluster)
        err_cluster.add(err_sg_encode)

        def ind2cls(labels):
            if args.dataset_name == config.nuscenes:
                CLASS_NAMES = [
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]

            elif args.dataset_name == config.kitti:
                CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

            ret = [CLASS_NAMES[x - 1] for x in labels]
            return np.array(ret)

        fp_polygon = get_gt_polygon(pd_boxes)
        if fp_polygon is None: fp_polygon = Polygon()
        fp_sg_encode = get_scene_graph_encode(args.dataset_name, pd_boxes,
                                              ind2cls(pred_labels),
                                              weather_type, sg_type)

        ref_fp_polygon = dict_scene['fp_polygon']
        ref_fp_cluster = dict_cluster['fp_cluster']

        fp_polygon = shapely.union(ref_fp_polygon, fp_polygon)
        fp_cluster = copy.deepcopy(ref_fp_cluster)
        fp_cluster.add(fp_sg_encode)

        return scene,\
               gt_polygon,\
               sg_encode,\
               err_polygon,\
               err_sg_encode,\
               fp_polygon,\
               fp_sg_encode,\
               cal_single_c1(gt_polygon, dict_scene['road_hull']),\
               cal_single_c2(cluster, sg_type),\
               cal_single_c1(err_polygon, dict_scene['road_hull']),\
               cal_single_c2(err_cluster, sg_type),

    return func


def iterate_function(args):

    def func(queue: FuzzQueue, seed_list, mutated_data_batches,
             objective_function):

        scene_list = []
        gt_polygon_list = []
        sg_encode_list = []
        err_polygon_list = []
        err_sg_encode_list = []
        fp_polygon_list = []
        fp_sg_encode_list = []
        results = []

        DUMPS_utlis.updateIter_DUMPS_errorMetric(args.DUMPS)

        for idx in range(len(mutated_data_batches)):
            res = [0, 0, 0, 0]
            scene, gt_polygon, sg_encode, err_polygon, err_sg_encode, fp_polygon, fp_sg_encode, res[
                0], res[1], res[2], res[3] = objective_function(
                    seed_list[idx], mutated_data_batches[idx])

            scene_list.append(scene)
            gt_polygon_list.append(gt_polygon)
            sg_encode_list.append(sg_encode)
            err_polygon_list.append(err_polygon)
            err_sg_encode_list.append(err_sg_encode)
            fp_polygon_list.append(fp_polygon)
            fp_sg_encode_list.append(fp_sg_encode)
            results.append(res)

        for idx in range(len(mutated_data_batches)):
            scene = scene_list[idx]
            gt_polygon = gt_polygon_list[idx]
            sg_encode = sg_encode_list[idx]
            err_polygon = err_polygon_list[idx]
            err_sg_encode = err_sg_encode_list[idx]
            fp_polygon = fp_polygon_list[idx]
            fp_sg_encode = fp_sg_encode_list[idx]

            dict_scene = args.DUMPS['scene'][scene]
            sg_type = dict_scene['scene_graph_type']
            dict_cluster = args.DUMPS['cluster'][sg_type]

            dict_scene['gt_polygon'] = shapely.union(dict_scene['gt_polygon'],
                                                     gt_polygon)
            dict_cluster['cluster'].add(sg_encode)
            dict_scene['error_polygon'] = shapely.union(
                dict_scene['error_polygon'], err_polygon)
            dict_cluster['error_cluster'].add(err_sg_encode)
            dict_scene['fp_polygon'] = shapely.union(dict_scene['fp_polygon'],
                                                     fp_polygon)
            dict_cluster['fp_cluster'].add(fp_sg_encode)

            if sg_encode not in dict_cluster['count_cluster']:
                dict_cluster['count_cluster'][sg_encode] = 0
            dict_cluster['count_cluster'][sg_encode] += 1

            if err_sg_encode not in dict_cluster['count_error_cluster']:
                dict_cluster['count_error_cluster'][err_sg_encode] = 0
            dict_cluster['count_error_cluster'][err_sg_encode] += 1

            if fp_sg_encode not in dict_cluster['count_fp_cluster']:
                dict_cluster['count_fp_cluster'][fp_sg_encode] = 0
            dict_cluster['count_fp_cluster'][fp_sg_encode] += 1

        if len(mutated_data_batches):
            results = np.array(results, dtype=np.float32)

            def strategy_func(x):
                idx = [np.argmax(results[:, x])]
                return idx

            if args.criteria == config.spc:
                idx = strategy_func(2)
            elif args.criteria == config.sec:
                idx = strategy_func(3)
            elif args.criteria == config.error_spc:
                idx = strategy_func(2)
            elif args.criteria == config.error_sec:
                idx = strategy_func(3)
            elif args.criteria == config.none:
                idx = []
            elif args.criteria == config.ldfuzz:
                idx = strategy_func(0) + strategy_func(1)
            elif args.criteria == config.error_mixed:
                idx = strategy_func(2) + strategy_func(3)
            elif args.criteria == config.lirtest:
                idx = [np.random.choice(len(mutated_data_batches))]
            else:
                raise Exception()

            print(f'Select #{idx}')
            for _idx in idx:

                seed = seed_list[_idx]
                data_batch = mutated_data_batches[_idx]

                method = data_batch['method']

                args.DUMPS['inqueue_method_times'][method] += 1
                queue.save_if_interesting(seed, data_batch, False)

    return func


#############################
# fetch_function
def fetch_function(dataset_name, model, loader, infos_dict):
    '''
    Fetches predictions from a model.
    Args:
        dataset_name (str): The name of the dataset (e.g., 'kitti', 'nuscenes').
        model (object): The pretrained model.
        loader (object): The data loader.
        infos_dict (dict): A dictionary containing dataset information.
    Returns:
        tuple: A tuple containing three lists:
            - pred_boxes (list): A list of predicted bounding boxes.
            - pred_scores (list): A list of predicted scores.
            - pred_labels (list): A list of predicted labels.
    '''

    if dataset_name == config.kitti:
        loader.dataset.kitti_infos = infos_dict
    elif dataset_name == config.nuscenes:
        loader.dataset.infos = infos_dict
    else:
        raise NotImplementedError

    pred_boxes = []
    pred_scores = []
    pred_labels = []
    start_index = 0

    empty_cache()
    with no_grad():
        for batch_dict in loader:
            model_utils.load_data_to_gpu(batch_dict)

            layer_outputs = model(batch_dict)

            layer_outputs = layer_outputs[0]
            batch_size = len(layer_outputs)

            for _b in range(batch_size):
                pred_boxes.append(layer_outputs[_b]
                                  ['pred_boxes'].detach().cpu().numpy()[:, :7])
                pred_scores.append(
                    layer_outputs[_b]['pred_scores'].detach().cpu().numpy())
                pred_labels.append(
                    layer_outputs[_b]['pred_labels'].detach().cpu().numpy())

            # CLASS_NAMES: [
            #     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            #     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
            # ]
            # 6 10
            # CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
            # 2 3
            ######## Filter classes
            for _b in range(start_index, start_index + batch_size):
                if dataset_name == config.kitti:
                    mask = (pred_labels[_b] != 2) * (pred_labels[_b] != 3)
                elif dataset_name == config.nuscenes:
                    mask = (pred_labels[_b] != 6) * (pred_labels[_b] != 10)
                else:
                    raise NotImplementedError

                pred_boxes[_b] = pred_boxes[_b][mask]
                pred_scores[_b] = pred_scores[_b][mask]
                pred_labels[_b] = pred_labels[_b][mask]

            ######## Filter boxes outside range
            ######## Only for nuScenes dataset
            if dataset_name == config.nuscenes:
                point_cloud_range = [-51.2, 0, -5.0, 51.2, 51.2, 3.0]
                for _b in range(start_index, start_index + batch_size):
                    mask = box_utils.mask_boxes_outside_range_numpy(
                        pred_boxes[_b], point_cloud_range)

                    pred_boxes[_b] = pred_boxes[_b][mask]
                    pred_scores[_b] = pred_scores[_b][mask]
                    pred_labels[_b] = pred_labels[_b][mask]

            ######## Filter score
            for _b in range(start_index, start_index + batch_size):
                mask = (pred_scores[_b] > 0.5)

                pred_boxes[_b] = pred_boxes[_b][mask]
                pred_scores[_b] = pred_scores[_b][mask]
                pred_labels[_b] = pred_labels[_b][mask]

            assert len(pred_boxes) == len(pred_scores) and len(
                pred_scores) == len(pred_labels)

            start_index += batch_size

    return pred_boxes, pred_scores, pred_labels


def build_fetch_function(dataset_name, model_name, batch_size):
    '''
    Builds a partial function for fetching predictions based on the provided dataset and model.
    Args:
        dataset_name (str): The name of the dataset.
        model_name (str): The name of the model to use.
        batch_size (int): The batch size to use for fetching predictions.
    Returns:
        callable: A partial function.
    '''

    return partial(
        fetch_function,
        dataset_name,
        *model_utils.load_model(dataset_name, model_name, batch_size),
    )


def build_frd_function(dataset_name):

    return model_utils.load_frd_model(dataset_name).infer
