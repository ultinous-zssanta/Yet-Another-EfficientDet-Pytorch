#! /usr/bin/env python3

import os
import argparse
import yaml

import numpy as np
import torch

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


DEFAULT_ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
DEFAULT_ANCHOR_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
DEFAULT_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def evaluate(input_list, model, level=0, score_threshold=0.05, iou_threshold=0.5, target_label=None):
  regressBoxes = BBoxTransform()
  clipBoxes = ClipBoxes()

  results = {}
  for img_filename in input_list:
    results[img_filename] = []
    ori_imgs, framed_imgs, framed_metas = preprocess(img_filename, max_size=DEFAULT_INPUT_SIZES[level])
    x = torch.from_numpy(framed_imgs[0])
    x = x.cuda(0)
    x = x.float()

    # NCHW
    x = x.unsqueeze(0).permute(0, 3, 1, 2)

    features, regression, classification, anchors = model(x)

    preds = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        score_threshold, iou_threshold)
    if not preds:
      continue

    preds = invert_affine(framed_metas, preds)[0]

    scores = preds['scores']
    class_ids = preds['class_ids']
    rois = preds['rois']
    if rois.shape[0] > 0:
      # # x1,y1,x2,y2 -> x1,y1,w,h
      # rois[:, 2] -= rois[:, 0]
      # rois[:, 3] -= rois[:, 1]

      bbox_score = scores

      for roi_id in range(rois.shape[0]):
        score = float(bbox_score[roi_id])
        label = int(class_ids[roi_id])
        box = rois[roi_id, :]

        if score < score_threshold:
          break
        results[img_filename].append({
          "score": score,
          "label": label,
          "box": list(box),
        })
  return results


def create_arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_list")
  parser.add_argument("--image_root", type=str, default=None)
  parser.add_argument("--level", help="efficientdet level eg. 0 => efficientdet-d0", default=0, type=int)
  parser.add_argument("-w", "--weights", type=str, default=None, help="/path/to/weights")
  parser.add_argument("--iou_threshold", type=float, default=0.5, help="iou threshold")
  parser.add_argument("--num_classes", type=int, default=90)
  return parser


def _get_input_list(input_list_file, img_root=None):
  output = []
  with open(input_list_file, "r") as fp:
    for l in fp.readlines():
      l_trimmed = l.strip()
      if not l_trimmed:
        continue
      output.append(
        os.path.join(img_root, l_trimmed) if img_root is not None else l_trimmed
      )
  return output


def filter_and_dump_results_to_csv(output_csv, results, keep_categories=None):
  def _convert_to_bbox_format(fname, boxes, scores, delimiter="\t"):
    boxes = np.array(boxes)
    scores = np.array(scores)[:, np.newaxis]
    return delimiter.join([fname] + [("%.3f" % x) for x in np.hstack((boxes, scores)).flatten()])

  def _category_filter(b):
    return (keep_categories is None) or (b["label"] in keep_categories)

  with open(output_csv, "w") as fp:
    for fname, boxes in results.items():
      bbs, scores = zip(*[(b["box"], b["score"]) for b in boxes if _category_filter(b)])
      fp.write(_convert_to_bbox_format(fname, bbs, scores) + "\n")


def main():
  parser = create_arg_parser()
  args = parser.parse_args()

  target_model = "efficientdet-d{}".format(args.level)
  weights_path = "weights/{}.pth".format(target_model) if args.weights is None else args.weights

  model = EfficientDetBackbone(compound_coef=args.level, num_classes=args.num_classes,
                               ratios=DEFAULT_ANCHOR_RATIOS, scales=DEFAULT_ANCHOR_SCALES)
  model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
  model.requires_grad_(False)
  model.eval()

  model.cuda(0)

  inputs = _get_input_list(args.input_list, args.image_root)
  results = evaluate(inputs, model)
  filter_and_dump_results_to_csv("output.csv", results, [0])


if __name__ == "__main__":
  main()
