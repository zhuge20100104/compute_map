import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image


# compute all the IOUs between the N annostation boxes and the query_boxes
def compute_iou(boxes, query_boxes):
    """
    Args:
        boxes:       (N,4) ndarray of float
        query_boxes: (4)   ndarray of float  xmin,ymin,xmax,ymax
     Returns
        overlaps:    (N)   ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    ious = np.zeros((N),dtype=np.float64)
    query_box_area=(
        (query_boxes[2]-query_boxes[0])*(query_boxes[3]-query_boxes[1])
    )
    for i in range(N):
        iw=(
#             min(boxes[i][2],query_boxes[2])-max(boxes[i][0],query_boxes[0])
             min(boxes[i,2],query_boxes[2])-max(boxes[i,0],query_boxes[0])
        )
        if iw>0:  
            ih=(
#                 min(boxes[i][3],query_boxes[3])-max(boxes[i][1],query_boxes[1])
                 min(boxes[i,3],query_boxes[3])-max(boxes[i,1],query_boxes[1])
            )
            if ih>0:
                oa=iw*ih
                box_area = np.float64(((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])))
                ua = query_box_area+box_area-oa
                ious[i]=oa/ua
    return ious

if __name__ == "__main__":
    query_box = [0,0,10,10]
    boxes = [(5,5,15,15),(0,0,9,9),(1,1,8,8)]
    query_box = np.array(query_box,dtype=np.float)
    boxes = np.array(boxes,dtype=np.float)
    ious = compute_iou(boxes,query_box)
    print(ious)