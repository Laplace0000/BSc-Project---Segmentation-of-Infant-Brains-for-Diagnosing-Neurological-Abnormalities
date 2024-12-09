import torch
import numpy as np
import VINNA.utils.logging as logging_u
import SimpleITK as sitk
import os
from typing import List

import VINNA.data_processing.utils.data_utils as du

logger = logging_u.get_logger(__name__)


def iou_score(pred_cls, true_cls, nclass=79):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []

    for i in range(1, nclass):
        intersect = ((pred_cls == i).float() + (true_cls == i).float()).eq(2).sum().item()
        union = ((pred_cls == i).float() + (true_cls == i).float()).ge(1).sum().item()
        intersect_.append(intersect)
        union_.append(union)
    intersect, union = np.array(intersect_), np.array(union_)
    return intersect, union


def precision_recall(pred_cls, true_cls, nclass=79):
    """
    Function to calculate recall (TP/(TP + FN) and precision (TP/(TP+FP) per class
    :param pytorch.Tensor pred_cls: network prediction (categorical)
    :param pytorch.Tensor true_cls: ground truth (categorical)
    :param int nclass: number of classes
    :return:
    """
    tpos_fneg = []
    tpos_fpos = []
    tpos = []

    for i in range(1, nclass):
        all_pred = (pred_cls == i).float()
        all_gt = (true_cls == i).float()

        tpos.append((all_pred + all_gt).eq(2).sum().item())
        tpos_fpos.append(all_pred.sum().item())
        tpos_fneg.append(all_gt.sum().item())

    return np.array(tpos), np.array(tpos_fneg), np.array(tpos_fpos)


class EvalMetrics:

    def __init__(self, cfg):
        self.subject_name = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sf = 1.0
        self.lut = du.read_classes_from_lut(cfg.EVALMETRICS.LUT)
        self.num_classes = len(self.lut["LabelName"]) - 1
        self.class_names = self.lut["LabelName"][1:]
        self.labels = self.lut["ID"][1:]
        self.torch_labels = torch.from_numpy(self.labels.values)
        self.names = ["SubjectName", "Average", "Subcortical", "Cortical"]
        mask = np.ones(39, dtype=bool)
        mask[cfg.EVALMETRICS.SUBSTRUCT] = False
        self.range_sub = mask

        # Set up tensors for metrics
        self.union = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        self.intersection = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        self.tpos_fneg = torch.zeros(self.num_classes, device=self.device)
        self.tpos = torch.zeros(self.num_classes, device=self.device)
        self.tpos_fpos = torch.zeros(self.num_classes, device=self.device)
        self.vs = torch.zeros(self.num_classes, device=self.device)
        self.asd = torch.zeros(self.num_classes, device=self.device)

    def volume_similarity(self, binary_pred, binary_gt, idx):
        ruler = sitk.LabelOverlapMeasuresImageFilter()
        try:
            ruler.Execute(binary_pred, binary_gt)
            # Append results
            self.vs[idx] = ruler.GetVolumeSimilarity()

        except RuntimeError:
            # Append results
            self.vs[idx] = np.nan

    """
    def average_surface_distance(self, binary_pred, binary_gt, idx):
        try:
            self.asd[idx] += rv.surface_hausdorff_distance(binary_gt, binary_pred)
        except (ValueError, RuntimeError) as e:
            self.asd[idx] += np.nan
    """

    def iou(self, pred, binary_gt, idx):
        for j in range(self.num_classes):
            binary_pred = (pred == j).float()
            self.intersection[idx, j] += torch.sum(torch.mul(binary_gt, binary_pred))
            self.union[idx, j] += (torch.sum(binary_gt) + torch.sum(pred))

    def prmetrics(self, binary_gt, binary_pred, idx):
        self.tpos[idx] += (binary_gt + binary_pred).eq(2).sum().item()
        self.tpos_fpos[idx] += binary_pred.sum().item()
        self.tpos_fneg[idx] += binary_gt.sum().item()

    def on_epoch_end(self, prediction, groundtruth):
        # Loop over the classes and calculate all metrics
        for i in range(self.num_classes):
            gt_thresholded = (groundtruth == i).float()
            pred_thresholded = (prediction == i).float()

            # Get IOU/DSC
            self.iou(prediction, gt_thresholded, i)

            # Get Precision/Recall
            self.prmetrics(pred_thresholded, gt_thresholded, i)

            # Get VS
            self.volume_similarity(pred_thresholded, gt_thresholded, i)

            # Get ASD
            # self.average_surface_distance(pred_thresholded, gt_thresholded, i)


def torch_dice(pred, labels, num_classes=24):
    # one hot encoding of labels
    empty_lab = torch.zeros(size=(labels.shape + (num_classes,)))
    empty_pred = torch.zeros(size=(pred.shape + (num_classes,)))
    empty_lab.scatter_(-1, labels.unsqueeze(-1).type(torch.int64), 1)
    empty_pred.scatter_(-1, pred.unsqueeze(-1).type(torch.int64), 1)
    intersection = torch.sum(torch.mul(empty_lab, empty_pred), axis=(0, 1, 2))
    union = torch.sum(empty_lab, axis=(0, 1, 2)) + torch.sum(empty_pred, axis=(0, 1, 2))

    dsc = 2 * torch.div(intersection, union)
    print(torch.sum(empty_lab, axis=(0, 1, 2)), torch.sum(empty_pred, axis=(0, 1, 2)))
    return dsc[1:]


class DiceScore:
    """
        Accumulating the component of the dice coefficient i.e. the union and intersection
    Args:
        op (callable): a callable to update accumulator. Method's signature is `(accumulator, output)`.
            For example, to compute arithmetic mean value, `op = lambda a, x: a + x`.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device (str of torch.device, optional): device specification in case of distributed computation usage.
            In most of the cases, it can be defined as "cuda:local_rank" or "cuda"
            if already set `torch.cuda.set_device(local_rank)`. By default, if a distributed process group is
            initialized and available, device is set to `cuda`.
    """

    def __init__(self, num_classes,
                 device=None,
                 output_transform=lambda y_pred, y: (y_pred.data.max(1)[1], y)):
        self._device = device
        self.out_transform = output_transform
        self.n_classes = num_classes
        self.union = torch.zeros(self.n_classes, self.n_classes, device=device)
        self.intersection = torch.zeros(self.n_classes, self.n_classes, device=device)

    def reset(self):
        self.union = torch.zeros(self.n_classes, self.n_classes, device=self._device)
        self.intersection = torch.zeros(self.n_classes, self.n_classes, device=self._device)

    def _check_output_type(self, output):
        if not (isinstance(output, tuple)):
            raise TypeError("Output should a tuple consist of of torch.Tensors, but given {}".format(type(output)))

    def _update_union_intersection_matrix(self, batch_output, labels_batch):
        for i in range(self.n_classes):
            gt = (labels_batch == i).float()
            for j in range(self.n_classes):
                pred = (batch_output == j).float()
                self.intersection[i, j] += torch.sum(torch.mul(gt, pred))
                self.union[i, j] += (torch.sum(gt) + torch.sum(pred))

    def _update_union_intersection(self, batch_output, labels_batch):
        for i in range(self.n_classes):
            gt = (labels_batch == i).float()
            pred = (batch_output == i).float()
            self.intersection[i, i] += torch.sum(torch.mul(gt, pred))
            self.union[i, i] += (torch.sum(gt) + torch.sum(pred))

    def update(self, output, cnf_mat):
        self._check_output_type(output)

        if self._device is not None:
            # Put output to the metric's device
            if isinstance(output, torch.Tensor) and (output.device != self._device):
                output = output.to(self._device)
        y_pred, y = self.out_transform(*output)

        if cnf_mat:
            self._update_union_intersection_matrix(y_pred, y)
        else:
            self._update_union_intersection(y_pred, y)

    def compute_dsc(self):
        dsc_per_class = self._dice_calculation()
        dsc = dsc_per_class.mean()
        return dsc

    def comput_dice_cnf(self):
        dice_cm_mat = self._dice_confusion_matrix()
        return dice_cm_mat

    def _dice_calculation(self):
        intersection = self.intersection.diagonal()
        union = self.union.diagonal()
        dsc = 2 * torch.div(intersection, union)
        return dsc

    def _dice_confusion_matrix(self):
        dice_intersection = self.intersection.cpu().numpy()
        dice_union = self.union.cpu().numpy()
        if not (dice_union > 0).all():
            logger.info("Union of some classes are all zero")
        dice_cnf_matrix = 2 * np.divide(dice_intersection, dice_union)
        return dice_cnf_matrix


def fast_dice(x, y, labels):
    """
    Fast evaluation of Dice scores in torch.
    :param torch.Tensor x: input label map
    :param torch.Tensor y: input label map of the same size as x
    :param torch.Tensor labels: numpy array of labels to evaluate on
    :return torch.Tensor: array with Dice scores (one per label)
    """
    assert x.shape == y.shape, 'both inputs should have same size, but had {} and {}'.format(x.shape, y.shape)
    # cast to torch.Tensor if numpy array
    x, y, labels = tensor_check([x, y, labels])

    # sort labels and create bins for histograms by adding small number to all labels
    # and subtract from first entry [1, 2] --> [0, 0.9, 1.1, 2.1]
    bin_edges = torch.cat(((labels[0]-1).reshape(1), (labels[0] - 0.1).reshape(1), torch.sort(labels)[0] + 0.1))

    # compute Dice and re-arrange scores in initial order
    # compute bi-dimensional histogram of two data samples (x and y), define number of bins as calculated above
    stacked = torch.stack([x.flatten().type_as(bin_edges), y.flatten().type_as(bin_edges)], axis=1)
    hst = torch.histogramdd(stacked, bins=[bin_edges, bin_edges])[0]
    dice_score = 2 * torch.diag(hst) / (torch.sum(hst, axis=0) + torch.sum(hst, axis=1)) #+ 1e-5) -> small number needed for zero division

    return dice_score[1:]


def tensor_check(potential_tensors):
    for i in range(len(potential_tensors)):
        if not isinstance(potential_tensors[i], torch.Tensor):
            potential_tensors[i] = torch.from_numpy(potential_tensors[i])
    return potential_tensors


def surface_hausdorff_distance(gt, pred):
    """
    Compute symmetric surface distances and take the maximum and mean
    """
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(gt)

    statistics_image_filter = sitk.StatisticsImageFilter()

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())

    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(pred)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the prediction surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
    all_surface_distances = seg2ref_distances + ref2seg_distances
    # The Maximum of the symmetric surface distances is the Hausdorff distance between the surfaces (max(d(seg2ref), d(ref2seg))
    return np.nanmean(all_surface_distances)


def append_nan(items):
    for item in items:
        item.append(np.nan)
    return items


def fast_asd_distance(gt, pred, num_classes=24):
    """
    Vectorized implementation of surface hausdorff distance
    :param np.ndarray gt: not one-hot-encoded ground truth
    :param np.ndarray pred: not one-hot-encoded ground truth
    :param int num_classes: number of classes present in the ground truth
    :return np.ndarray result: resulting surface distance
    """
    # distance map = euclidian distance of ground truth, Label contour of ground truth
    # one-hot-encode arrays, excluding first entry (=background)
    reference_distance_map = multiclass_euclidian_distance(gt, axis=0, classes=num_classes)
    segmented_distance_map = multiclass_euclidian_distance(pred, axis=0, classes=num_classes)

    # Use SITK to calculate label outline, not one-hot! --> encode afterwards
    # Label contour can be calculated on full image --> one-hot encoding afterwards
    reference_surface_one_hot = get_one_hot(sitk.GetArrayFromImage(sitk.LabelContour(sitk.GetImageFromArray(gt.astype(int)))), num_classes)
    pred_surface_one_hot = get_one_hot(sitk.GetArrayFromImage(sitk.LabelContour(sitk.GetImageFromArray(pred.astype(int)))), num_classes)

    # sum pixels in surfaces that are 1 per class (excluding background)
    num_reference_surface_pixels = np.sum(reference_surface_one_hot, axis=(1, 2, 3))
    num_pred_surface_pixels = np.sum(pred_surface_one_hot, axis=(1, 2, 3))

    # Multiply the surface segmentations (one-hot-encoded label contours) with the distance maps (Classes x H x W x D).
    # The resulting distance maps contain non-zero values only on the surface
    # (they can also contain zero on the surface); exclude background from one-hot-encoded
    seg2ref_distance_map = reference_distance_map * pred_surface_one_hot[1:, ...]
    ref2seg_distance_map = segmented_distance_map * reference_surface_one_hot[1:, ...]

    # Get all non-zero distances and add zero distances if required (where surface pixels are, but distance 0)
    seg2ref_distance_map = get_distances_from_array(seg2ref_distance_map, num_pred_surface_pixels)
    ref2seg_distance_map = get_distances_from_array(ref2seg_distance_map, num_reference_surface_pixels)

    # sum and done
    return np.nanmean(seg2ref_distance_map + ref2seg_distance_map, axis=1)


def multiclass_euclidian_distance(matrix, axis, classes):
    mo = get_one_hot(matrix, classes)
    return np.stack([sitk.GetArrayFromImage(sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.GetImageFromArray(mo[i])))) for i in range(1, mo.shape[axis])])


def get_one_hot(targets, nb_classes):
    """
    One-hot-encoding of numpy arrays.
    :param np.ndarray targets: target to one-hot-encode
    :param int nb_classes: number of classes to encode into
    :return np.ndarray: one-hot-encoded target
    """
    res = np.eye(nb_classes, dtype=int)[np.array(targets.astype(int)).reshape(-1)]
    return res.reshape([nb_classes] + list(targets.shape))


def get_distances_from_array(distance_map, num_surface_pixels):
    missing_zeroes = num_surface_pixels[1:] - np.count_nonzero(distance_map, axis=(1, 2, 3))
    distance_map[distance_map == 0] = np.nan
    flattened_dm = distance_map.reshape(len(num_surface_pixels)-1, -1)
    for i in range(len(missing_zeroes)):
        nan_index = np.argwhere(np.isnan(flattened_dm[i]))[:missing_zeroes[i]]
        flattened_dm[i][nan_index] = 0
    return flattened_dm


def binary_sitk_metrics(prediction, ground_truth, k):
    # threshold the image and load as SITK image input image
    segmentation = sitk.GetImageFromArray(np.where(prediction == k, 1, 0))
    gold_standard = sitk.GetImageFromArray(np.where(ground_truth == k, 1, 0))

    # Compute overlap
    ruler = sitk.LabelOverlapMeasuresImageFilter()
    try:
        ruler.Execute(segmentation, gold_standard)
        # Append results
        dsc_met = ruler.GetDiceCoefficient()
    except RuntimeError:
        # Append results
        dsc_met = np.nan

    try:
        mean_surf = surface_hausdorff_distance(gold_standard, segmentation)
    except (ValueError, RuntimeError) as e:
        mean_surf = np.nan

    # Append results
    return dsc_met, mean_surf


def compute_metrics(reference_file: str, prediction_file: str, label_list: List[int], split_list: List[int], combine: dict):
    # load images
    seg_ref_header, seg_ref = fio.read_image(reference_file, conform=False, labellist=split_list)
    seg_pred_header, seg_pred = fio.read_image(prediction_file, conform=False, labellist=split_list)

    # Get results
    results = evaluate_metrics(gt=seg_ref, pred=seg_pred, label_list=label_list, combine=combine)
    results["SubjectNames"] = os.path.basename(os.path.dirname(reference_file))
    results["Net_type"] = os.path.basename(prediction_file).split(".")[1]
    results["SF"] = seg_ref_header.header.get_zooms()[0]
    return results


def evaluate_metrics(gt, pred, label_list, combine=None, fs=False):
    """
    Function to evaluate a number of metrics (volume similarity, Jaccard, Dice, Hausdorff and Average Hausdorff)
    for all labels in label_list between ground truth (gt) and prediction (pred)
    :param torch.Tensor/numpy.ndarray gt: gold standard/ground truth image (filename + path)
    :param torch.Tensor/numpy.ndarray pred: network prediction (filename + path)
    :param list(int) label_list: list of labels to calculate the metrics for
    :param dict or None combine: key, value pair of labels to combine (=values) to one label (=key). Metrics
                                 will be calculated for these at the end of the run.
    :return:
    """
    sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1e-04)
    # Prepare lists for storage
    avg_surf_hdist = []
    dsc = []
    # check dsc implementation
    #dsc = fast_dice(pred, gt, label_list)
    # Loop over the labels and ...
    if fs:
        # map labels to hcp, set label list
        hcp_first = {1: 17, 17: [7, 8],  5: 3, 7: 2, 2: 53, 3: 18, 4: 54, 6: 42, 8: 41, 18: [46, 47]}
        label_list = [1, 17, 5, 7, 2, 3, 4, 6, 8, 18]
        combine = False
        for new_lab in label_list:
            mask = np.in1d(pred, hcp_first[new_lab]).reshape(pred.shape)
            pred[mask] = new_lab

    for k in label_list:
        if np.unique(gt[gt == k]).size == 0:
            avg_surf_hdist.append(np.nan)
            dsc.append(np.nan)
        else:
            dsc_m, surf_m = binary_sitk_metrics(pred, gt, k)
            dsc.append(dsc_m)
            avg_surf_hdist.append(surf_m)
    if combine:
        for new_lab, lab in sorted(combine.items()):
            mask = np.in1d(pred, lab).reshape(pred.shape)
            pred[mask] = new_lab
            mask = np.in1d(gt, lab).reshape(gt.shape)
            gt[mask] = new_lab
            dsc_m, surf_m = binary_sitk_metrics(pred, gt, new_lab)
            dsc.append(dsc_m)
            avg_surf_hdist.append(surf_m)
    return {"dice": np.asarray(dsc), "surfaceAverageHausdorff": np.asarray(avg_surf_hdist)}
