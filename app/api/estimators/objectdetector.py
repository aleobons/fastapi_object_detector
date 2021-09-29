import tensorflow as tf
import numpy as np
import cv2


class ObjectDetector:
    OUTPUT_BOXES = 'OUTPUT_BOXES'
    OUTPUT_CROPS = 'OUTPUT_CROPS'
    OUTPUT_VIS_OBJECTS = 'OUTPUT_VIS_OBJECTS'
    INFO_CONFIDENCE_THRESHOLD = 'confidence_threshold'
    INFO_MAX_OBJECTS = 'max_objects'
    INFO_NMS_THRESHOLD = 'non_maximum_suppression_threshold'
    INFO_LABEL_MAP = 'label_map'
    INFO_SHOW_CONFIDENCE = 'show_confidence'

    def __init__(self, model, infos=None):
        self.outputs_functions = {
            self.OUTPUT_BOXES: self._build_output_boxes,
            self.OUTPUT_CROPS: self._build_output_crops,
            self.OUTPUT_VIS_OBJECTS: self._build_output_vis
        }

        self.model = model

        self.label_map = infos.get(self.INFO_LABEL_MAP, None)

        self.confidence_threshold = None
        self.non_maximum_suppression_threshold = None
        self.max_objects = None
        self.show_confidence = None

        self.image_original = None

    def predict(self, image, output):
        # guarda a imagem original
        self.image_original = image

        # guarda as informações do output
        output_function = self.outputs_functions.get(output.get('name', self.OUTPUT_BOXES))
        vars_output = output.get('vars', {})
        self.confidence_threshold = vars_output.get(self.INFO_CONFIDENCE_THRESHOLD, 0.5)
        self.non_maximum_suppression_threshold = vars_output.get(self.INFO_NMS_THRESHOLD, 0.5)
        self.max_objects = vars_output.get(self.INFO_MAX_OBJECTS, 1)
        self.show_confidence = vars_output.get(self.INFO_SHOW_CONFIDENCE, False)

        # Converte o np array em um tensor e adiciona um axis ao tensor pois o modelo espera um batch de imagens
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor)

        # Coleta a quantidade de objetos detectados
        num_detections = int(detections.pop('num_detections'))

        # Filtra as detecções para obter apenas aquilo que é de fato detecções, retirando a dimensão batch
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # As classes detectadas devem ser inteiros
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # Define o que é necessário para construir o output
        key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

        # coletando as informações de interesse
        detections = {key: value for key, value in detections.items() if key in key_of_interest}

        # filtrando as detecções pelo threshold da confiança da predição
        for key in key_of_interest:
            scores = detections['detection_scores']
            current_array = detections[key]
            filtered_current_array = current_array[scores > self.confidence_threshold]
            detections[key] = filtered_current_array

        # creating a zip object that will contain model output info as
        output_info = list(zip(detections['detection_boxes'],
                               detections['detection_scores'],
                               detections['detection_classes']
                               )
                           )
        boxes, scores, classes = self._nms(output_info)

        result_info = list(zip(boxes, scores, classes))
        result_info.sort(key=lambda x: x[1], reverse=True)

        result_info = result_info[:self.max_objects]

        return output_function(result_info)

    def _build_output_crops(self, result_info):
        roi = None
        if len(result_info) > 0:
            boxes = self._calcule_coord(list(result_info[0][0]))

            roi = self.image_original[boxes[0]:boxes[2], boxes[1]:boxes[3]]

            _, roi = cv2.imencode(".png", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        return roi

    def _build_output_vis(self, result_info):
        image_with_objects = self.image_original.copy()

        for info in result_info:
            class_id = info[2]
            label_class = self.label_map.get(class_id, class_id)

            boxes = self._calcule_coord(list(info[0]))
            image_with_objects = cv2.rectangle(image_with_objects, (boxes[1], boxes[0]),
                                               (boxes[3], boxes[2]), (255, 0, 0), 2)

            if self.show_confidence:
                text = f"{label_class}-{info[1] * 100:.2f}%"
            else:
                text = f"{label_class}"

            cv2.putText(image_with_objects, text, (boxes[1], boxes[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        _, image_with_objects = cv2.imencode(".png", cv2.cvtColor(image_with_objects, cv2.COLOR_RGB2BGR))

        return image_with_objects

    def _build_output_boxes(self, result_info):
        output = []
        for info in result_info:
            class_id = f"{info[2]}"
            boxes = self._calcule_coord(list(info[0]))

            object_detected = {
                "detection_box": f"{boxes}",  # formato: [y1, x1, y2, x2]
                "detection_score": f"{info[1]}",
                "detection_class": self.label_map.get(class_id, class_id).get('name', class_id)
            }

            output.append(object_detected)

        return output

    def _calcule_coord(self, boxes_detecteds):
        height, width = self.image_original.shape[:2]

        y1 = int(boxes_detecteds[0] * height)
        x1 = int(boxes_detecteds[1] * width)
        y2 = int(boxes_detecteds[2] * height)
        x2 = int(boxes_detecteds[3] * width)

        return [y1, x1, y2, x2]

    def _nms(self, rects):
        """
        Filter rectangles
        rects is array of oblects ([x1,y1,x2,y2], confidence, class)
        """
        out = []

        remove = [False] * len(rects)

        for i in range(0, len(rects) - 1):
            if remove[i]:
                continue
            inter = [0.0] * len(rects)
            for j in range(i, len(rects)):
                if remove[j]:
                    continue
                inter[j] = self._intersection(rects[i][0], rects[j][0]) / min(self._square(rects[i][0]),
                                                                              self._square(rects[j][0]))

            max_prob = 0.0
            max_idx = 0
            for k in range(i, len(rects)):
                if inter[k] >= self.non_maximum_suppression_threshold:
                    if rects[k][1] > max_prob:
                        max_prob = rects[k][1]
                        max_idx = k

            for k in range(i, len(rects)):
                if (inter[k] >= self.non_maximum_suppression_threshold) & (k != max_idx):
                    remove[k] = True

        for k in range(0, len(rects)):
            if not remove[k]:
                out.append(rects[k])

        boxes = [box[0] for box in out]
        scores = [score[1] for score in out]
        classes = [cls[2] for cls in out]
        return boxes, scores, classes

    def _intersection(self, rect1, rect2):
        """
        Calculates square of intersection of two rectangles
        rect: list with coords of top-right and left-boom corners [x1,y1,x2,y2]
        return: square of intersection
        """
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
        overlapArea = x_overlap * y_overlap
        return overlapArea

    def _square(self, rect):
        """
        Calculates square of rectangle
        """
        return abs(rect[2] - rect[0]) * abs(rect[3] - rect[1])
