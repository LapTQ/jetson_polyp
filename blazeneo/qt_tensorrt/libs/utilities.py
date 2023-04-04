import os
import cv2
import numpy as np

def scan_dir(dir_paths):
    """
    Scan the original image directories, and save the path with key value.
    """
    paths = []
    for root, dirs, files in os.walk(dir_paths, topdown=False):
        for name in files:
            paths.append(os.path.join(root, name))
    
    return paths

def create_dir(dirName):
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")



class Preprocess:
    def __init__(self, width, height, mode):
        self.width = width
        self.height = height
        self.mode = mode 

    def get_input_shape(self):
        x = np.zeros((self.height, self.width, 3)).astype(np.uint8)
        y = self.run(x)
        return y.shape
    
    def run(self, image):
        image = image.copy()
        if self.mode == 0:
            # rgb + hwc
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
        elif self.mode == 1:
            # rgb + chw  
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)
        elif self.mode == 2:
            # rgb + hwc + normalize [-1, 1]
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = image/127.5-1
        elif self.mode == 3:
            # rgb + chw + normalize [-1, 1]
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32)
            image = image/127.5-1
        elif self.mode == 4:
            # bgr + hwc
            image = cv2.resize(image, (self.width, self.height))
            image = image.astype(np.float32)
        elif self.mode == 5:    # anh Thuan
            # resize + BGR + normalize [0, 1] + CHW
            image = cv2.resize(image, (self.width, self.height))
            image = np.float32(image) / 255.
            image = np.moveaxis(image, 2, 0)
        elif self.mode == 6:    # anh Son
            # resize + RGB + normalize [0, 1] + HWC

            def resize(size, image):
                h, w, c = image.shape
                scale_w = size / w
                scale_h = size / h
                scale = min(scale_w, scale_h)
                h = int(h * scale)
                w = int(w * scale)
                padimg = np.zeros((size, size, c), image.dtype)
                padimg[:h, :w] = cv2.resize(image, (w, h))
                return padimg

            # image = cv2.resize(image, (self.width, self.height))
            image = resize(self.height, image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.float32(image) / 255.
            image = np.moveaxis(image, 2, 0)

        return image


class Postprocess:
    def __init__(self, x_start, y_start, x_end, y_end, height, width, n_classes, threshold, iou_threshold, mode, class_labels):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.mode = mode
        self.class_labels = class_labels
    
    def run(self, image, crop_image, output):
        """
        From Mask result, draw contour on image directly
        """
        image = image.copy()
        crop_image = crop_image.copy()

        # segmentation
        if self.mode == 2:
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

            output = np.argmax(output, axis=0)
            h, w = output.shape[:2]
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask[output == 0] = (0, 0, 0)
            mask[output == 1] = (0, 255, 0)
            if self.n_classes == 1:
                mask[output == 2] = (0, 255, 0)
            else:
                mask[output == 2] = (0, 0, 255)
            

            res = mask.copy()
            res[:,:,1] = cv2.erode(res[:,:,1], kernel, iterations=2)
            res[:,:,1] = cv2.dilate(res[:,:,1], kernel, iterations=2)
            res[:,:,2] = cv2.erode(res[:,:,2], kernel, iterations=2)   
            res[:,:,2] = cv2.dilate(res[:,:,2], kernel, iterations=2)   
            p = res.copy()
            res[:,:,1] = cv2.dilate(res[:,:,1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
            res[:,:,2] = cv2.dilate(res[:,:,2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
            bound = res-p
            
            bound = cv2.resize(bound,(self.x_end - self.x_start, self.y_end - self.y_start), interpolation = cv2.INTER_NEAREST)
            crop_image[bound!=0] = bound[bound!=0]

        # detection
        elif self.mode == 3:

            # def _rescale_boxes(out_size, im_shape, boxes):
                
            #     ow, oh = out_size
            #     w, h = im_shape
                
            #     scale_w = ow / w
            #     scale_h = oh / h
            #     new_anns = []
            #     for box in boxes:
            #         xmin = int(box[0] / scale_w)
            #         ymin = int(box[1] / scale_h)
            #         xmax = int(box[2] / scale_w)
            #         ymax = int(box[3] / scale_h)
            #         new_anns.append([xmin, ymin, xmax, ymax])
            #     return np.array(new_anns)

            def _rescale_boxes(size, im_shape, boxes):
                w, h = im_shape
                scale_w = size / w
                scale_h = size / h
                scale = min(scale_w, scale_h)
                new_anns = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = [int(p/scale) for p in box]
                    new_anns.append([xmin, ymin, xmax, ymax])
                return np.array(new_anns)

            def _draw_border(img, box, color):
                xmin, ymin, xmax, ymax = box
                point1, point2 = (xmin, ymin), (xmin, ymax)
                point3, point4 = (xmax, ymin), (xmax, ymax), 
                line_length = min(xmax-xmin, ymax-ymin)//5

                x1, y1 = point1
                x2, y2 = point2
                x3, y3 = point3
                x4, y4 = point4    

                cv2.circle(img, (x1, y1), 3, color, -1)    #-- top_left
                cv2.circle(img, (x2, y2), 3, color, -1)    #-- bottom-left
                cv2.circle(img, (x3, y3), 3, color, -1)    #-- top-right
                cv2.circle(img, (x4, y4), 3, color, -1)    #-- bottom-right

                cv2.line(img, (x1, y1), (x1 , y1 + line_length), color, 2)  #-- top-left
                cv2.line(img, (x1, y1), (x1 + line_length , y1), color, 2)

                cv2.line(img, (x2, y2), (x2 , y2 - line_length), color, 2)  #-- bottom-left
                cv2.line(img, (x2, y2), (x2 + line_length , y2), color, 2)

                cv2.line(img, (x3, y3), (x3 - line_length, y3), color, 2)  #-- top-right
                cv2.line(img, (x3, y3), (x3, y3 + line_length), color, 2)

                cv2.line(img, (x4, y4), (x4 , y4 - line_length), color, 2)  #-- bottom-right
                cv2.line(img, (x4, y4), (x4 - line_length , y4), color, 2)

                return img
            
            def nms(outputs, conf_thr, iou_thr):
                print(conf_thr, iou_thr)
                outputs = np.array([cv2.transpose(outputs[0])]) #1x8400x6
                rows = outputs.shape[1] #8400

                boxes, scores, class_ids = [], [], []

                for i in range(rows):
                    classes_scores = outputs[0][i][4:] #outputs = (1, 8400, 6)
                    (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
                    if maxScore >= conf_thr:
                        # print(outputs[0][i])
                        box = [
                            outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                            # outputs[0][i][2], outputs[0][i][3]] #xywh
                            outputs[0][i][0] + outputs[0][i][2]/2, outputs[0][i][1] + outputs[0][i][3]/2]
                        boxes.append(box)
                        scores.append(maxScore)
                        class_ids.append(maxClassIndex)

                result_boxes = cv2.dnn.NMSBoxes(boxes, scores, conf_thr, iou_thr)

                detections = []
                for i in range(len(result_boxes)):
                    index = result_boxes[i]
                    box = boxes[index]
                    detection = {
                        'class_id': class_ids[index],
                        'score': scores[index],
                        'box': box }
                    detections.append(detection)
                return detections

            
            def parse_detection(detections):

                boxes = np.array([d['box'] for d in detections])
                if len(boxes) == 0:
                    return [], [], []
                ## x1y1wh to xyxy
                scores = [d['score'] for d in detections]
                scale = max(crop_image.shape[:2]) / 640 
                boxes *= scale
                boxes = np.int32(boxes)

                class_names = [self.class_labels[int(d['class_id'])] for d in detections]

                print([(list(b), s, c) for b, s, c in zip(boxes, scores, class_names)])

                # Arange box from top to bottom #
                indices = np.argsort(boxes[:, 3])
                corrected_boxes = [boxes[i] for i in indices] 
                corrected_classes = [class_names[i] for i in indices] 
                corrected_scores = [scores[i] for i in indices]
                #***#
                return corrected_boxes, corrected_scores, corrected_classes
            

            detections = nms(np.expand_dims(output, axis=0), conf_thr=self.threshold, iou_thr=self.iou_threshold)
            
            boxes, scores, class_names = parse_detection(detections)

            for box, score, name in zip(boxes, scores, class_names):
                xmin, ymin, xmax, ymax = box
                label = '{:.4f}'.format(score)
                ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                if self.n_classes == 2:
                    color = (0, 0, 255) if name == 'neo' else (0, 255, 0)
                else:
                    color = (0, 255, 0)
                
                crop_image = _draw_border(crop_image, box, color)
                cv2.rectangle(crop_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (255, 255, 255), -1)
                cv2.putText(crop_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        
        image[self.y_start:self.y_end, self.x_start:self.x_end] =  crop_image


        return image