import tensorflow as tf
import numpy as np
import cv2
from enum import Enum
import json
import requests
import base64


class ObjectDetector:
    """ Classe de detecção de objetos

    Detecta, trata e retorna os objetos nas imagens em 3 tipos de outputs:
        Coordenadas dos objetos
        Crop do objeto com maior confiança
        Visualização dos objetos na imagem

    Attributes:
        Output(Enum): enum de opções de output
        Infos(Enum): enum de informações importantes
        outputs_functions: mapeamento dos outputs com as funções
        model_url: url do modelo utilizado para inferência
        label_map: mapeamento de rótulos no formato protobuf
        show_confidence: bool indicativo se deve ser exibido a confiança (somente para o output VIS_OBJECTS
        images_original: lista de imagens enviadas para inferência sem alteração

    """

    class Output(Enum):
        OUTPUT_BOXES = 'OUTPUT_BOXES'
        OUTPUT_CROPS = 'OUTPUT_CROPS'
        OUTPUT_VIS_OBJECTS = 'OUTPUT_VIS_OBJECTS'

    class Infos(Enum):
        INFO_CONFIDENCE_THRESHOLD = 'confidence_threshold'
        INFO_MAX_OBJECTS = 'max_objects'
        INFO_NMS_THRESHOLD = 'non_maximum_suppression_threshold'
        INFO_LABEL_MAP = 'label_map'
        INFO_SHOW_CONFIDENCE = 'show_confidence'

    def __init__(self, model_url, infos=None):
        # define o mapeamento de outputs e funções
        self.outputs_functions = {
            self.Output.OUTPUT_BOXES: self._build_output_boxes,
            self.Output.OUTPUT_CROPS: self._build_output_crops,
            self.Output.OUTPUT_VIS_OBJECTS: self._build_output_vis
        }

        # deixa os rótulos disponíveis para toda a classe
        self.label_map = infos.get(self.Infos.INFO_LABEL_MAP, None)

        # inicializa atributos importantes para ser usado em alguns outputs sem precisar passar como parâmetro na função
        self.show_confidence = None
        self.images_original = []

        # armazena a url do modelo
        self.model_url = model_url

    def predict(self, images, output, vars_output):
        """ Predição na imagem conforme o output

        Args:
            images: lista de imagens originais para predição
            output: nome do output que deverá ser retornado
            vars_output: dicionário com as informações do output que será retornado

        Returns:
            O retorno vai variar conforme o output, podendo ser uma imagem ou lista de imagens codificadas ou uma lista
            de coordenadas
        """
        # guarda as imagens originais
        self.images_original = [np.array(image).astype(np.float32) for image in images]

        # define a função que será utilizada para o output passado
        output_function = self.outputs_functions.get(output)

        # coleta outras informações conforme o output passado
        confidence_threshold = vars_output.get(self.Infos.INFO_CONFIDENCE_THRESHOLD, 0.5)
        non_maximum_suppression_threshold = vars_output.get(self.Infos.INFO_NMS_THRESHOLD, 0.5)
        max_objects = vars_output.get(self.Infos.INFO_MAX_OBJECTS, 1)
        self.show_confidence = vars_output.get(self.Infos.INFO_SHOW_CONFIDENCE, False)

        # inicializa a lista de resultados das imagens
        results_info = []

        # percorre cada imagem fazendo a predição individual
        for image in images:
            # Cria o json e o cabeçalho para usar no request
            data = {"instances": [image]}
            headers = {"content-type": "application/json"}

            # inferência
            response = requests.post(self.model_url, data=json.dumps(data), headers=headers)

            # transforma a resposta em json
            detections = response.json()['predictions'][0]

            # Coleta a quantidade de objetos detectados
            num_detections = int(detections.pop('num_detections'))

            # Define o que é necessário para construir o output
            key_of_interest = ['detection_classes', 'detection_boxes', 'detection_scores']

            # coletando as informações de interesse e eliminando os valores que não são detecções, utilizando o número
            # de detecções para fazer o recorte na lista
            detections = {key: np.array(value[0:num_detections]) for key, value in detections.items() if
                          key in key_of_interest}

            # As classes detectadas devem ser inteiros
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            # filtrando as detecções pelo threshold da confiança da predição
            for key in key_of_interest:
                scores = detections['detection_scores']
                current_array = detections[key]
                filtered_current_array = current_array[scores > confidence_threshold]
                detections[key] = filtered_current_array

            # retira as detecções que se sobrescrevem conforme o treshold definido
            boxes, scores, classes = self._nms(detections['detection_boxes'], detections['detection_scores'],
                                               detections['detection_classes'], non_maximum_suppression_threshold)

            # agrupa as informações das detecções e ordena pelos de maior confiança
            result_info = list(zip(boxes, scores, classes))
            result_info.sort(key=lambda x: x[1], reverse=True)

            # restringe as detecções pela quantidade máxima de objetos definida nas configurações
            results_info.append(result_info[:max_objects])

        # chama a função correspondente ao output, passando as detecções
        return output_function(results_info)

    def _build_output_crops(self, results_info):
        """ Recorta o objeto da imagem original

        Args:
            results_info: detecções realizadas com as coordenadas, confianças e rótulos para cada imagem

        Returns:
            crops: lista de recortes dos objetos codificadas em binário das imagens
        """

        # inicializa a lista de recortes
        crops = []

        # percorre a lista de imagens para coletar as coordenadas. É feito uma junção entre as lista de imagens
        # originais e a lista de resultados para garantir a relação correta entre eles
        for (image, result_info) in zip(self.images_original, results_info):

            # inicializa a lista de roi para guardar o recorte de cada objeto
            list_roi = []

            # percorre a lista de objetos detectados da imagem
            for info in result_info:
                # calcula as coordenadas em valores absolutos
                boxes = self._calcule_coord(self.images_original[0], list(info[0]))

                # recorta o objeto da imagem original
                roi = image[boxes[0]:boxes[2], boxes[1]:boxes[3]]

                # codifica o recorte em imagem png. É necessário converter a ordem dos canais de cores devido ao padrão
                # do opencv
                _, roi = cv2.imencode(".png", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

                # converte em binário e armazena na lista de recortes (roi)
                list_roi.append(base64.b64encode(roi))

            # armazena na lista de recortes por imagem
            crops.append(list_roi)

        return crops

    def _build_output_vis(self, results_info):
        """ Inclui a anotação dos objetos na imagem original

        Args:
            results_info: detecções realizadas com as coordenadas, confianças e rótulos

        Returns:
            image_with_objects: imagem com os objetos anotados
        """

        # na visualização somente uma imagem é enviada, portando será sempre o primeiro da lista
        result_info = results_info[0]

        # faz uma cópia da imagem original, evitando alterá-la
        image_with_objects = self.images_original[0].copy()

        # atualiza a imagem com objetos para cada objeto detectado
        for info in result_info:

            # coleta o nome do rótulo utilizando o id detectado no mapeamento de rótulos
            class_id = info[2]
            label_class = self.label_map.get(class_id, class_id)

            # calcula as coordenadas em valores absolutos e desenha um retângulo na imagem com objetos
            boxes = self._calcule_coord(image_with_objects, list(info[0]))
            image_with_objects = cv2.rectangle(image_with_objects, (boxes[1], boxes[0]),
                                               (boxes[3], boxes[2]), (255, 0, 0), 2)

            # define o texto que será incluído na imagem com objetos, incluindo a confiança ou não
            if self.show_confidence:
                text = f"{label_class}-{info[1] * 100:.2f}%"
            else:
                text = f"{label_class}"

            # escreve o texto na imagem com objetos
            cv2.putText(image_with_objects, text, (boxes[1], boxes[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # codifica a imagem png. É necessário converter a ordem dos canais de cores devido ao padrão do opencv
        _, image_with_objects = cv2.imencode(".png", cv2.cvtColor(image_with_objects, cv2.COLOR_RGB2BGR))

        return image_with_objects

    def _build_output_boxes(self, results_info):
        """ Lista as coordenadas dos objetos detectados

        Args:
            results_info: detecções realizadas com as coordenadas, confianças e rótulos

        Returns:
            output: Uma lista de coordenadas com as coordenadas, confianças e nome dos rótulos
        """

        # inicializa a lista de coordenadas de todas as imagens
        outputs = []

        # percorre a lista de imagens para coletar as coordenadas. É feito uma junção entre as lista de imagens
        # originais e a lista de resultados para garantir a relação correta entre eles
        for (image, result_info) in zip(self.images_original, results_info):
            # inicializa a lista de coordenadas da imagem atual
            output_image = []

            # percorre a lista de objetos detectados da imagem
            for info in result_info:
                # coleta o nome do rótulo utilizando o id detectado no mapeamento de rótulos
                class_id = info[2]
                label_class = self.label_map.get(class_id, class_id)

                # constrói o dicionário para ser incluído na lista de coordenadas
                object_detected = {
                    "detection_box": f"{list(info[0])}",  # formato: [ymin, xmin, ymax, xmax]
                    "detection_score": f"{info[1]:.4f}",
                    "detection_class": f"{label_class}"
                }

                # acrescenta o objeto detectado na lista da imagem atual
                output_image.append(object_detected)

            # acrescenta os objetos da imagem na lista de imagens
            outputs.append(output_image)

        return outputs

    def _calcule_coord(self, image, boxes_detecteds):
        """ Converte coordenadas de valores relativos para absolutos

        Args:
            boxes_detecteds: lista de coordenadas com valores relativos

        Returns:
            Uma lista de coordenadas com valores absolutos no formato [ymin, xmin, ymax, xmax]
        """

        # coleta a altura e largura da imagem original
        height, width = image.shape[:2]

        # converte cada coordenada para o seu valor absoluto, garantindo que seja um valor inteiro
        ymin = int(boxes_detecteds[0] * height)
        xmin = int(boxes_detecteds[1] * width)
        ymax = int(boxes_detecteds[2] * height)
        xmax = int(boxes_detecteds[3] * width)

        return [ymin, xmin, ymax, xmax]

    @staticmethod
    def _nms(boxes, scores, classes, iou_threshold=0.5):
        """ Retira as sobreposições de detecções

        Args:
            boxes: lista de coordenadas
            scores: lista de confianças
            classes: lista de rótulos
            iou_threshold: limite para supressão de detecções com sobreposição. Quanto maior, mais sobreposição pode
            ocorrer

        Returns:
            Uma tupla com as coordenadas, confianças e rótulos selecionados após a supressão das sobreposições
        """

        # Utilizo a função do Tensorflow para selecionar os índices que serão mantidos
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_output_size=len(boxes), iou_threshold=iou_threshold)

        # coleto as coordenadas, confianças e rótulos que serão mantidos
        selected_boxes = tf.gather(boxes, selected_indices)
        selected_scores = tf.gather(scores, selected_indices)
        selected_classes = tf.gather(classes, selected_indices)

        # converto em numpy array antes de retornar para evitar erros na construção dos outputs
        return selected_boxes.numpy(), selected_scores.numpy(), selected_classes.numpy()
