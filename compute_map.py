import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

# Average Precision Calculation
def compute_ap(recall,precision):
    """ 
    Compute the average precision, given the recall and precision curves.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    import pdb; pdb.set_trace()
    mrec = np.concatenate(([0.],recall,[1.]))
    mpre = np.concatenate(([0.],precision,[0.]))   # 参考上图的RP曲线，看两端
    
    # compute the precision envelope
    for i in range(mpre.size-1,0,-1):
        mpre[i-1]=np.maximum(mpre[i-1],mpre[i])
    
    i=np.where(mrec[1:]!=mrec[:-1])[0]      # where X axis (recall) changes value

    ap = np.sum((mrec[i+1]-mrec[i])*mpre[i+1])
    return ap

if __name__ == "__main__":
    pre =[1.0,0.5,0.667]
    rec = [0.33,0.33,0.667]
    ap = compute_ap(rec,pre)
    print(ap)