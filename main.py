# Instalar a biblioteca pynput
# pip install pynput

import cv2
import mediapipe as mp
from pynput import keyboard
import os

# Inicializar o módulo de mãos do mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Capturar vídeo da câmera do DroidCam via USB
# Substitua 'http://<IP_DO_DROIDCAM>:4747/video' pelo endereço IP fornecido pelo DroidCam
cap = cv2.VideoCapture('http://192.168.0.15:4747/video')

# Lista para armazenar os pontos do rastro
trail_points = []

# Variável para armazenar o comando digitado
command = ""

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

            # Verificar se apenas o dedo indicador está levantado
            finger_up = [False] * 5
            for i, lm in enumerate(hand_landmarks.landmark):
                if i in [8, 12, 16, 20]:  # Pontos dos dedos
                    if lm.y < hand_landmarks.landmark[i - 2].y:
                        finger_up[(i - 4) // 4] = True

            if finger_up[1] and not any(finger_up[i] for i in [0, 2, 3, 4]):
                # Apenas o dedo indicador está levantado
                x, y = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                trail_points.append((x, y))
            elif any(finger_up[i] for i in [0, 1, 2, 3, 4]):
                # Dois ou mais dedos estão levantados
                trail_points.clear()

    # Desenhar o rastro
    for i in range(1, len(trail_points)):
        cv2.line(image, trail_points[i - 1], trail_points[i], (0, 255, 0), 3)

    # Mostrar a imagem
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
listener.stop()