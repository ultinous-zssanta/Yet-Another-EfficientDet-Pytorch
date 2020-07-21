#! /usr/bin/env python3

import os
import argparse
import yaml
import cv2

import numpy as np
import torch
from tqdm import tqdm

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_images


DEFAULT_ANCHOR_SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
DEFAULT_ANCHOR_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
DEFAULT_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


def my_preprocess(img_obj, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229), convert_to_gray=False):
  if isinstance(img_obj, str):
    return preprocess(img_obj, max_size=max_size, mean=mean, std=std, convert_to_gray=convert_to_gray)
  elif isinstance(img_obj, np.ndarray):
    return ([img_obj], ) + preprocess_images([img_obj], max_size=max_size, mean=mean, std=std, convert_to_gray=convert_to_gray)


def evaluate(input_list, model, level, score_threshold=0.05, iou_threshold=0.5, convert_to_gray=False):
  regressBoxes = BBoxTransform()
  clipBoxes = ClipBoxes()

  results = {}
  for img_obj, img_name in tqdm(input_list):
    results[img_name] = []
    ori_imgs, framed_imgs, framed_metas = my_preprocess(img_obj, max_size=DEFAULT_INPUT_SIZES[level], convert_to_gray=convert_to_gray)
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
        results[img_name].append({
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
  parser.add_argument("--convert_to_gray", default=False, action="store_true")
  parser.add_argument("--video_input", default=False, action="store_true")
  return parser


def _get_input_list(input_list_file, img_root=None):
  output = []
  with open(input_list_file, "r") as fp:
    for l in fp.readlines():
      l_trimmed = l.strip()
      if not l_trimmed:
        continue
      output.append(
        (os.path.join(img_root, l_trimmed) if img_root is not None else l_trimmed, l_trimmed)
      )
  return output


def filter_and_dump_results_to_csv(output_csv, results, keep_categories=None):
  def _convert_to_bbox_format(fname, boxes, scores, delimiter="\t"):
    if boxes:
      boxes = np.array(boxes)
      scores = np.array(scores)[:, np.newaxis]
      return delimiter.join([fname] + [("%.3f" % x) for x in np.hstack((boxes, scores)).flatten()])
    else:
      return fname

  def _category_filter(b):
    return (keep_categories is None) or (b["label"] in keep_categories)

  with open(output_csv, "w") as fp:
    for fname, boxes in results.items():
      packed_results = [(b["box"], b["score"]) for b in boxes if _category_filter(b)]
      if packed_results:
        bbs, scores = zip(*packed_results)
      else:
        bbs, scores = [], []
      fp.write(_convert_to_bbox_format(fname, bbs, scores) + "\n")


def video_frame_generator(video_filename):
  cap = cv2.VideoCapture(video_filename)
  frame_ind = 0
  while cap.isOpened():
    valid, frame = cap.read()
    if valid:
      yield frame, str(frame_ind)
    else:
      break
    frame_ind += 1


def main():
  parser = create_arg_parser()
  args = parser.parse_args()
  print(args)

  target_model = "efficientdet-d{}".format(args.level)
  weights_path = "weights/{}.pth".format(target_model) if args.weights is None else args.weights

  model = EfficientDetBackbone(compound_coef=args.level, num_classes=args.num_classes,
                               ratios=DEFAULT_ANCHOR_RATIOS, scales=DEFAULT_ANCHOR_SCALES)
  model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
  model.requires_grad_(False)
  model.eval()

  model.cuda(0)

  if args.video_input:
    inputs = _get_input_list(args.input_list, args.image_root)
    for video_filename, video_name in inputs:
      results = evaluate(video_frame_generator(video_filename), model, args.level, iou_threshold=args.iou_threshold, convert_to_gray=args.convert_to_gray, score_threshold=0.1)
      fname, _ = os.path.splitext(video_name)
      filter_and_dump_results_to_csv("%s.csv" % fname, results, [0])
  else:
    inputs = _get_input_list(args.input_list, args.image_root)
    results = evaluate(inputs, model, args.level, iou_threshold=args.iou_threshold, convert_to_gray=args.convert_to_gray)
    filter_and_dump_results_to_csv("output.csv", results, [0])


if __name__ == "__main__":
  main()
