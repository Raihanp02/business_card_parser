import cv2
import numpy as np
from typing import Union
from PIL import Image

ImageInput = Union[str, Image.Image, np.ndarray]

class PostprocessPaddleOCR:
    def __init__(self):
        pass

    def postprocess(self, result, image):
        result,scalex,scaley = self.plot(result,image)
        sentences = self._sent(result,scalex,scaley)

        return sentences

    def plot(self,result, rotated_image):
        conc = []
        for i in range(len(result[0])):
            conc = conc + result[0][i][0]

        xmax = max(conc, key= lambda x: x[0])[0]
        xmin = min(conc, key= lambda x: x[0])[0]
        ymax = max(conc, key= lambda x: x[1])[1]
        ymin = min(conc, key= lambda x: x[1])[1]

        res = []
        scalex = xmax-xmin
        scaley = ymax-ymin

        for line in result:
            for word in line:
                text = word[1][0]
                bbox = word[0]
                x_min, y_min = int(bbox[0][0]), int(bbox[0][1])
                x_max, y_max = int(bbox[2][0]), int(bbox[2][1])
                res = res + [[[(x_min-xmin), (y_min-ymin), (x_max-xmin), (y_max-ymin)], text]]
        #         cv2.rectangle(rotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #         cv2.putText(rotated_image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.rectangle(rotated_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        res.sort(key=lambda x: x[0][1])

        return res, scalex, scaley

    def _sent(self,res,scalex,scaley):
        i = 0

        sentences = []

        while i < (len(res)):
            j = i
            temp = res.pop(0)
            words = temp[1]
            compareymin = temp[i][1]
            compareymax = temp[i][3]
            comparexmin = temp[i][0]
            comparexmax = temp[i][2]
            while j < (len(res)):
                temp1 = res[j]
                if (((abs(compareymin - temp1[0][3])/scaley)<0.1) or ((abs(compareymax - temp1[0][1])/scaley)<0.1)) and ((abs(comparexmin - temp1[0][0])/scalex)<0.05 or (abs(comparexmax - temp1[0][2])/scalex)<0.05):
                    words = words + " " + (res[j][1])
                    temp = res.pop(j)
                    compareymin = temp[0][1]
                    compareymax = temp[0][3]
                    comparexmin = temp[0][0]
                    comparexmax = temp[0][2]
                else:
                    j=j+1
            sentences.append(words)

        return sentences