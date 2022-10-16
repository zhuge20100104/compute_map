import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

from compute_iou import compute_iou

# check if true positive or false positive
def check_if_true_or_false_positivate(annotations,detections,iou_threshold):
    """
    All detected bounding boxes will be assigned to the GT box that has the largest IOU. 
    Then check whether they are TP or FP by iou_threshold.
    Here, if multiple detected bounding boxes are mapped to the same GT box, then only the highest scored bounding box needs to be assigned to that GT. 
    So note that the detections passed to this function need to be pre-sorted.
    
    Args:
        annotations:      (N,4) ndarray of float
        detections:       (M,5) ndarray of float  xmin,ymin,xmax,ymax,score
        iou_threshuold    default:0.5  float
        
    Returns
        scores：          (M)   list of detection boxs's score
        false_positivate  (M)   list if detection box is FP,value=1.[1,0,0]
        true_positivate:  (M)   list if detection box id TP,value=1.[0,1,1]
    """
    import pdb; pdb.set_trace()
    annotations = np.array(annotations,dtype=np.float64)
    scores,false_positivate,true_positivate = [],[],[]
    detected_annotations = [] # a GT box should be mapped only one predicted box at most
    for query_box in detections:
        scores.append(query_box[4])
        if len(annotations)==0:
            false_positivate.append(1)
            true_positivate.append(0)
            continue
        ious = compute_iou(boxes=annotations,query_boxes=query_box[:4])
        assigned_annotation = np.argmax(ious)   # assigned anno index 
        max_iou = ious[assigned_annotation]
        if max_iou>=iou_threshold and assigned_annotation not in detected_annotations:
            true_positivate.append(1)
            false_positivate.append(0)
            detected_annotations.append(assigned_annotation)
        else:
            false_positivate.append(1)
            true_positivate.append(0)
    return scores,false_positivate,true_positivate

if __name__ == "__main__":
    anno = np.array([(1,1,6,6),(2,2,4,5),(4,9,5,19)],dtype=np.float64)
    detect = np.array([[1,1,5,6,0.5],[2,2,4,5,0.8]],dtype=np.float64)
    scores,false_positivate,true_positivate=check_if_true_or_false_positivate(anno,detect,0.1)
    print(scores)
    print(false_positivate)
    print(true_positivate)