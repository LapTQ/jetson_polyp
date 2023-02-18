import os
import cv2
import numpy as np

from glob import glob
#from tqdm import tqdm


class Metric():
    def __init__(self, name):
        self.name = name
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.list_iou_score = []
        self.list_dice_score = []
        self.list_precision = []
        self.list_recall = []

    def cal(self, predict, mask, ignore=None, smooth=1):
        tp, fp, fn = self.metric(predict, mask, ignore=ignore)
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        iou_score = (tp + smooth) / (tp + fp + fn + smooth)
        dice_score = (2*tp + smooth) / (2*tp + fp + fn + smooth)
        precision = (tp+smooth)/(tp+smooth+fp)
        recall = (tp+smooth)/(tp+smooth+fn)
        self.list_iou_score.append(iou_score)
        self.list_dice_score.append(dice_score)
        self.list_precision.append(precision)
        self.list_recall.append(recall)

    def show(self):
        dice_score = 2 * self.total_tp / \
            (2 * self.total_tp + self.total_fp + self.total_fn)
        iou_score = self.total_tp / \
            (self.total_tp + self.total_fp + self.total_fn)

        print("Evaluate {}".format(self.name))
        print("Dice score micro {}".format(dice_score))
        print("IoU score micro {}".format(iou_score))
        print("Dice score macro {}".format(
            np.array(self.list_dice_score).mean()))
        print("IoU score macro {}".format(np.array(self.list_iou_score).mean()))
        print("Precision score macro {}".format(np.array(self.list_precision).mean()))
        print("Recall score macro {}".format(np.array(self.list_recall).mean()))

    def metric(self, inputs, targets, ignore=None):
        if ignore is None:
            tp = np.sum(inputs * targets)
            fp = np.sum(inputs) - tp
            fn = np.sum(targets) - tp
        else:
            ignore = 1-ignore
            tp = np.sum(inputs * targets * ignore)
            fp = np.sum(inputs * ignore) - tp
            fn = np.sum(targets * ignore) - tp

        return tp, fp, fn


def main(gt,pr):
    GT_DIR = "/home/laptq/test_data/test_label/"# "./dataset/WLIv5_pub_noud/Test/label_images/"
    pr_fps =glob("result/*") # glob("/home/s/thuannh/tensorrt/blazeneo/deploy/result/*")#glob("./result_1/*")
    print(len(pr_fps))
    polyp_metric = Metric("polyp")
    neo_metric = Metric("neo")
    non_metric = Metric("non")

    for i,pr_path in enumerate(pr_fps):
     # try:
        if i %5 ==0 : print(i)
        fn = pr_path.split("/")[-1]
        pr_img = cv2.imread(pr_path)

        gt_path = os.path.join(GT_DIR, fn)
        gt_img = cv2.imread(gt_path)

        pr_img = cv2.resize(
            pr_img, (gt_img.shape[1], gt_img.shape[0]),interpolation= cv2.INTER_NEAREST)

        neo_gt = np.all(gt_img == [0, 0, 255], axis=-1).astype('float')
        non_gt = np.all(gt_img == [0, 255, 0], axis=-1).astype('float')
        polyp_gt =  neo_gt +non_gt #neo_gt +

        neo_pr = np.all(pr_img == [0, 0, 255], axis=-1).astype('float')
        non_pr = np.all(pr_img == [0, 255, 0], axis=-1).astype('float')
        polyp_pr = neo_pr + non_pr #neo_pr + 

        polyp_metric.cal(polyp_pr, polyp_gt)
        neo_metric.cal(neo_pr, neo_gt)
        non_metric.cal(non_pr, non_gt)
    #  except:
     #   continue
        # cv2.imshow("pr_img", pr_img)
        # cv2.imshow("gt_img", gt_img)
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     break

    polyp_metric.show()
    neo_metric.show()
    non_metric.show()
    
main("","")
