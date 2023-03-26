# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import datetime
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.fileio import dump
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from .coco_metric import CocoMetric


@METRICS.register_module()
class CocoPartialMetric(CocoMetric):
    """COCO-WholeBody evaluation metric.

    Evaluate AR, AP, and mAP for COCO-WholeBody keypoint detection tasks.
    Support COCO-WholeBody dataset. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`
    """
    default_prefix: Optional[str] = 'coco'
    body_num = 13
    foot_num = 4

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys of the each `gt_dict` in `gt_dicts`:
                    - `img_id`: image id of the data sample
                    - `width`: original image width
                    - `height`: original image height
                    - `raw_ann_info`: the raw annotation information
                Optional keys:
                    - `crowd_index`: measure the crowding level of an image,
                        defined in CrowdPose dataset
                It is worth mentioning that, in order to compute `CocoMetric`,
                there are some required keys in the `raw_ann_info`:
                    - `id`: the id to distinguish different annotations
                    - `image_id`: the image id of this annotation
                    - `category_id`: the category of the instance.
                    - `bbox`: the object bounding box
                    - `keypoints`: the keypoints cooridinates along with their
                        visibilities. Note that it need to be aligned
                        with the official COCO format, e.g., a list with length
                        N * 3, in which N is the number of keypoints. And each
                        triplet represent the [x, y, visible] of the keypoint.
                    - 'keypoints'
                    - `iscrowd`: indicating whether the annotation is a crowd.
                        It is useful when matching the detection results to
                        the ground truth.
                There are some optional keys as well:
                    - `area`: it is necessary when `self.use_area` is `True`
                    - `num_keypoints`: it is necessary when `self.iou_type`
                        is set as `keypoints_crowd`.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        cuts = np.cumsum([
            0, self.body_num, self.foot_num
        ]) * 3

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    body_kpts=np.array(ann['keypoints'])[cuts[0]:cuts[1]].tolist(),
                    foot_kpts=np.array(ann['keypoints'])[cuts[1]:cuts[2]].tolist(),
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    assert 'area' in ann, \
                        '`area` is required when `self.use_area` is `True`'
                    annotation['area'] = ann['area']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmpose CocoMetric.')
        coco_json: dict = dict(
            info=info,
            images=image_infos,
            categories=self.dataset_meta['CLASSES'],
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
        return converted_json_path

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_id = 1
        cat_results = []
        cat_results_body = []
        cat_results_foot = []

        cuts = np.cumsum([
            0, self.body_num, self.foot_num
        ]) * 3

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
            # collect all the person keypoints in current image
            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = [{
                'image_id': img_kpt['img_id'],
                'category_id': img_kpt['category_id'],
                'keypoints': _keypoint.tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, _keypoint in zip(img_kpts, _keypoints)]

            cat_results.extend(result)

            result_body = [{
                'image_id': img_kpt['img_id'],
                'category_id': img_kpt['category_id'],
                'keypoints': _keypoint[cuts[0]:cuts[1]].tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, _keypoint in zip(img_kpts, _keypoints)]
            cat_results_body.extend(result_body)

            result_foot = [{
                'image_id': img_kpt['img_id'],
                'category_id': img_kpt['category_id'],
                'keypoints': _keypoint[cuts[1]:cuts[2]].tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, _keypoint in zip(img_kpts, _keypoints)]
            cat_results_foot.extend(result_foot)

        res_file = f'{outfile_prefix}.keypoints.json'
        dump(cat_results, res_file, sort_keys=True, indent=4)
        res_body_file = f'{outfile_prefix}.body.keypoints.json'
        dump(cat_results_body, res_body_file, sort_keys=True, indent=4)
        res_foot_file = f'{outfile_prefix}.foot.keypoints.json'
        dump(cat_results_foot, res_foot_file, sort_keys=True, indent=4)

    def _do_python_keypoint_eval_mmpose(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']

        cuts = np.cumsum([
            0, self.body_num, self.foot_num
        ])

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints', sigmas, use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        sigmas = self.dataset_meta['sigmas']
        cuts = np.cumsum([
            0, self.body_num, self.foot_num
        ])

        coco_body = self._slice_coco_keypoints(self.coco, cuts[0], cuts[1])
        info_str = self._do_body_part_eval(coco_body, outfile_prefix, sigmas, 'body', cuts[0], cuts[1])
        coco_foot = self._slice_coco_keypoints(self.coco, cuts[1], cuts[2])
        info_str += self._do_body_part_eval(coco_foot, outfile_prefix, sigmas, 'foot', cuts[1], cuts[2])
        info_str += self._do_body_part_eval(self.coco, outfile_prefix, sigmas, '', cuts[0], cuts[2])
        return info_str

    @staticmethod
    def _slice_coco_keypoints(coco, start, end=None):
        _coco = copy.deepcopy(coco)
        for img_id in _coco.getImgIds():
            anns_ids = _coco.getAnnIds(imgIds=img_id, iscrowd=False)
            anns = _coco.loadAnns(anns_ids)
            for ann in anns:
                sliced_keypoints = ann['keypoints'][start * 3:] if end is None else ann['keypoints'][start * 3: end * 3]
                ann['keypoints'] = sliced_keypoints
                ann['num_keypoints'] = np.sum(np.array(sliced_keypoints).reshape(-1, 3)[:, 2] > 0)
        return _coco

    def _do_body_part_eval(self, coco, outfile_prefix, sigmas, prefix, start, end=None):
        if len(prefix) != 0:
            res_file = f'{outfile_prefix}.{prefix}.keypoints.json'
        else:
            res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(coco, coco_det, 'keypoints', sigmas[start:end])
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]
        if len(prefix) != 0:
            stats_names = [prefix+'/'+name for name in stats_names]
        else:
            stats_names = [name for name in stats_names]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str
