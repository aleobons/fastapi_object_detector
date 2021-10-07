import tensorflow as tf
import numpy as np
import cv2
from enum import Enum


class ObjectDetector:
    """ Classe de detecção de objetos

    Detecta, trata e retorna os objetos na imagem em 3 tipos de outputs:
        Coordenadas dos objetos
        Crop do objeto com maior confiança
        Visualização dos objetos na imagem

    Attributes:
        Output(Enum): enum de opções de output
        Infos(Enum): enum de informações importantes
        outputs_functions: mapeamento dos outputs com as funções
        model: modelo utilizado para inferência
        label_map: mapeamento de rótulos no formato protobuf
        show_confidence: bool indicativo se deve ser exibido a confiança (somente para o output VIS_OBJECTS
        image_original: imagem enviada para inferência sem alteração

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

    def __init__(self, model, infos=None):
        # define o mapeamento de outputs e funções
        self.outputs_functions = {
            self.Output.OUTPUT_BOXES: self._build_output_boxes,
            self.Output.OUTPUT_CROPS: self._build_output_crops,
            self.Output.OUTPUT_VIS_OBJECTS: self._build_output_vis
        }

        # deixa o modelo disponível para toda a classe
        self.model = model

        # deixa os rótulos disponíveis para toda a classe
        self.label_map = infos.get(self.Infos.INFO_LABEL_MAP, None)

        # inicializa atributos importantes para ser usado em alguns outputs sem precisar passar como parâmetro na função
        self.show_confidence = None
        self.image_original = None

    def predict(self, image, output):
        """ Predição na imagem conforme o output

        Args:
            image: imagem original para predição
            output: dicionário com as informações do output que será retornado

        Returns:
            O retorno vai variar conforme o output, podendo ser uma imagem codificada ou uma lista de coordenadas
        """
        # guarda a imagem original
        self.image_original = image

        # define a função que será utilizada para o output passado
        output_function = self.outputs_functions.get(output.get('name', self.Output.OUTPUT_BOXES))

        # coleta outras informações conforme o output passado
        vars_output = output.get('vars', {})
        confidence_threshold = vars_output.get(self.Infos.INFO_CONFIDENCE_THRESHOLD, 0.5)
        non_maximum_suppression_threshold = vars_output.get(self.Infos.INFO_NMS_THRESHOLD, 0.5)
        max_objects = vars_output.get(self.Infos.INFO_MAX_OBJECTS, 1)
        self.show_confidence = vars_output.get(self.Infos.INFO_SHOW_CONFIDENCE, False)

        # Converte o np array em um tensor e adiciona um axis ao tensor pois o modelo espera um batch de imagens
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        # inferência
        detections = self.model(input_tensor)

        # Coleta a quantidade de objetos detectados
        num_detections = int(detections.pop('num_detections'))

        # Filtra as detecções para obter apenas aquilo que é de fato detecções, retirando a dimensão batch
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

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
            filtered_current_array = current_array[scores > confidence_threshold]
            detections[key] = filtered_current_array

        # retira as detecções que se sobrescrevem conforme o treshold definido
        boxes, scores, classes = self._nms(detections['detection_boxes'], detections['detection_scores'],
                                           detections['detection_classes'], non_maximum_suppression_threshold)

        # agrupa as informações das detecções e ordena pelos de maior confiança
        result_info = list(zip(boxes, scores, classes))
        result_info.sort(key=lambda x: x[1], reverse=True)

        # restringe as detecções pela quantidade máxima de objetos definida
        result_info = result_info[:max_objects]

        # chama a função correspondente ao output, passando as detecções
        return output_function(result_info)

    def _build_output_crops(self, result_info):
        """ Recorta o objeto da imagem original

        Args:
            result_info: detecções realizadas com as coordenadas, confianças e rótulos

        Returns:
            roi: imagem codificada com o recorte do objeto que está no topo da lista de detecções
        """

        # inicializa o roi pois o modelo pode não ter detectado nenhum objeto
        roi = None

        # verifica se houve algum objeto detectado
        if len(result_info) > 0:
            # calcula as coordenadas em valores absolutos
            boxes = self._calcule_coord(list(result_info[0][0]))

            # recorta o objeto da imagem original
            roi = self.image_original[boxes[0]:boxes[2], boxes[1]:boxes[3]]

            # codifica o recorte em imagem png. É necessário converter a ordem dos canais de cores devido ao padrão do
            # opencv
            _, roi = cv2.imencode(".png", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        return roi

    def _build_output_vis(self, result_info):
        """ Inclui a anotação dos objetos na imagem original

        Args:
            result_info: detecções realizadas com as coordenadas, confianças e rótulos

        Returns:
            image_with_objects: imagem com os objetos anotados
        """

        # faz uma cópia da imagem original, evitando alterá-la
        image_with_objects = self.image_original.copy()

        # print(result_info)

        # atualiza a imagem com objetos para cada objeto detectado
        for info in result_info:

            # coleta o nome do rótulo utilizando o id detectado no mapeamento de rótulos
            class_id = info[2]
            label_class = self.label_map.get(class_id, class_id)

            # calcula as coordenadas em valores absolutos e desenha um retângulo na imagem com objetos
            boxes = self._calcule_coord(list(info[0]))
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

    def _build_output_boxes(self, result_info):
        """ Lista as coordenadas dos objetos detectados

        Args:
            result_info: detecções realizadas com as coordenadas, confianças e rótulos

        Returns:
            output: Uma lista de coordenadas com as coordenadas, confianças e nome dos rótulos
        """

        # inicializa a lista de coordenadas
        output = []

        # acrescenta as coordenadas de cada objeto na lista de coordenadas
        for info in result_info:
            # coleta o nome do rótulo utilizando o id detectado no mapeamento de rótulos
            class_id = info[2]
            label_class = self.label_map.get(class_id, class_id)

            # calcula as coordenadas em valores absolutos
            boxes = self._calcule_coord(list(info[0]))

            # constrói o dicionário para ser incluído na lista de coordenadas
            object_detected = {
                "detection_box": f"{boxes}",  # formato: [ymin, xmin, ymax, xmax]
                "detection_score": f"{info[1]:.4f}",
                "detection_class": f"{label_class}"
            }

            output.append(object_detected)

        return output

    def _calcule_coord(self, boxes_detecteds):
        """ Converte coordenadas de valores relativos para absolutos

        Args:
            boxes_detecteds: lista de coordenadas com valores relativos

        Returns:
            Uma lista de coordenadas com valores absolutos no formato [ymin, xmin, ymax, xmax]
        """

        # coleta a altura e largura da imagem original
        height, width = self.image_original.shape[:2]

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
