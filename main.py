import cv2
import mediapipe as mp
import numpy as np
from pynput import keyboard
import os
import time

# Inicializar o módulo de mãos do mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Capturar vídeo da câmera do DroidCam via USB
cap = cv2.VideoCapture('http://192.168.15.23:4747/video')

# Lista para armazenar os pontos do rastro
trail_points = []

# Variável para armazenar o comando digitado
command = ""

# Variáveis para exibição temporária do reconhecimento
recognized_symbol = ""
recognition_time = 0

# Função para reconhecer símbolos (letras e números)
def recognize_symbol(points):
    if len(points) < 10:
        return None

    # Normalizar os pontos
    x_coords, y_coords = zip(*points)
    left, top = min(x_coords), min(y_coords)
    right, bottom = max(x_coords), max(y_coords)
    width, height = right - left, bottom - top
    normalized_points = [(int((x - left) * 100 / width), int((y - top) * 100 / height)) for x, y in points]

    # Calcular a orientação geral do traço
    start_x, start_y = normalized_points[0]
    end_x, end_y = normalized_points[-1]
    angle = np.arctan2(end_y - start_y, end_x - start_x) * 180 / np.pi

    # Implementar reconhecimento de símbolos
    if width > height * 1.5:
        return 'F' if any(p[1] < top + height * 0.2 for p in normalized_points) else 'L'
    elif abs(width - height) < 20:
        return 'O' if width > 50 else '0'
    elif width < height / 2:
        return 'I' if height > 50 else '1'
    elif width > height * 1.2 and any(p[1] > top + height * 0.8 for p in normalized_points[:10]):
        return '7'
    elif width > height * 1.2 and any(p[1] < top + height * 0.2 for p in normalized_points[-10:]):
        return 'T'
    elif abs(angle) < 30 and width > height:
        return '_'
    elif 60 < abs(angle) < 120 and height > width:
        return '|'
    elif -30 < angle < 30:
        return '-'
    elif 30 < angle < 60:
        return '/'
    elif -60 < angle < -30:
        return '\\'
    elif width > height * 1.2 and any(p[1] < top + height * 0.2 for p in normalized_points) and any(p[1] > top + height * 0.8 for p in normalized_points):
        return 'C'
    elif height > width * 1.2 and any(p[0] < left + width * 0.2 for p in normalized_points) and any(p[0] > left + width * 0.8 for p in normalized_points):
        return '3'
    elif width > height and any(p[1] < top + height * 0.3 for p in normalized_points) and any(p[1] > top + height * 0.7 for p in normalized_points):
        return 'S'
    elif height > width and any(p[0] < left + width * 0.3 for p in normalized_points) and any(p[0] > left + width * 0.7 for p in normalized_points):
        if any(p[1] < top + height * 0.2 for p in normalized_points):
            return 'P'
        else:
            return 'D'
    # ... (manter o restante das condições existentes)

    return None

def on_press(key):
    global command
    try:
        if key.char:
            command += key.char
    except AttributeError:
        if key == keyboard.Key.enter:
            os.system(command)
            command = ""

# Configurar o listener do teclado
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Variável para controlar o estado do desenho
is_drawing = False
last_recognition_time = 0

# Criar uma imagem em branco para o desenho
drawing_canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inicializar o canvas de desenho se ainda não foi criado
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(frame)

    # Converter a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processar a imagem e detectar mãos
    results = hands.process(image)

    # Converter a imagem de volta para BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Desenhar as marcações das mãos na imagem
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Verificar se o dedo indicador está levantado
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y:  # Dedo indicador levantado
                if not is_drawing:
                    is_drawing = True
                    trail_points.clear()
                x, y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                trail_points.append((x, y))
            else:
                if is_drawing:
                    symbol = recognize_symbol(trail_points)
                    if symbol:
                        print(f"Símbolo reconhecido: {symbol}")
                        recognized_symbol = symbol
                        recognition_time = time.time()
                    is_drawing = False
                    last_recognition_time = time.time()

    # Desenhar o rastro no canvas
    for i in range(1, len(trail_points)):
        cv2.line(drawing_canvas, trail_points[i - 1], trail_points[i], (0, 255, 0), 3)

    # Combinar o frame original com o canvas de desenho
    combined_image = cv2.addWeighted(image, 1, drawing_canvas, 0.5, 0)

    # Mostrar o símbolo reconhecido na tela
    if time.time() - recognition_time < 2:  # Mostrar por 2 segundos
        cv2.putText(combined_image, f"Reconhecido: {recognized_symbol}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar a imagem combinada
    cv2.imshow('Hand Tracking', combined_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()
