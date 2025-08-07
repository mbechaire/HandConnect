import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import tkinter as tk
from tkinter import simpledialog
import glob
import time
from collections import deque
from tradutor import traduzir_bruto_para_portugues
import pyttsx3
import speech_recognition as sr

camera = 2

# Garante que os arquivos .pkl sejam buscados na mesma pasta do script
base_dir = os.path.dirname(os.path.abspath(__file__))
lista_pacotes = glob.glob(os.path.join(base_dir, "pacote_*.pkl"))
if not lista_pacotes:
    print("Nenhum pacote de sinais encontrado. Grave sinais primeiro.")
    exit()

pacote_sinais = {}
for caminho in lista_pacotes:
    with open(caminho, 'rb') as f:
        dados = pickle.load(f)
        pacote_sinais.update(dados)

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

engine = pyttsx3.init()

MICROFONE_INDEX = 1  # Troque para o índice do microfone desejado

gravando_audio = False
audio_buffer = None

# Carregar imagens da logo e do texto HandConnect
logo = cv2.imread('logo_mao.png', cv2.IMREAD_UNCHANGED)
logo_text = cv2.imread('logotext.png', cv2.IMREAD_UNCHANGED)

# Função para sobrepor PNG com alpha
def overlay_png(bg, fg, x, y):
    h, w = fg.shape[:2]
    if fg.shape[2] == 4:
        alpha_fg = fg[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + (1 - alpha_fg) * bg[y:y+h, x:x+w, c])
    else:
        bg[y:y+h, x:x+w] = fg
    return bg


def capturar_landmarks(frame, resultados):
    if resultados.multi_hand_landmarks:
        for mao in resultados.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in mao.landmark]
            return landmarks
    return None

def distancia_euclidiana(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalizar(landmarks):
    arr = np.array(landmarks)
    centroide = arr.mean(axis=0)
    return (arr - centroide).tolist()

def normalizar_trajetoria(seq):
    if not seq:
        return []
    ponto_inicial = np.array(seq[0][0])
    return [[(lm[0]-ponto_inicial[0], lm[1]-ponto_inicial[1], lm[2]-ponto_inicial[2]) for lm in frame] for frame in seq]

def reconhecer_sinal(landmarks, pacote_sinais):
    if landmarks is None:
        return None, None
    landmarks_norm = normalizar(landmarks)
    menor_dist = float('inf')
    nome_encontrado = None
    significado = None
    for nome, dados in pacote_sinais.items():
        if dados.get("tipo") == "estatico" and "amostras" in dados:
            for amostra in dados["amostras"]:
                if len(amostra) != len(landmarks_norm):
                    continue
                dist = np.mean([np.linalg.norm(np.array(landmarks_norm[i]) - np.array(amostra[i])) for i in range(len(amostra))])
                if dist < menor_dist:
                    menor_dist = dist
                    nome_encontrado = nome
                    significado = dados.get("significado", "")
    if menor_dist < sens:
        return nome_encontrado, significado
    else:
        return None, None

def dtw(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.mean([
                np.linalg.norm(np.array(seq1[i-1][k]) - np.array(seq2[j-1][k]))
                for k in range(len(seq1[i-1]))
            ])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )
    return dtw_matrix[n, m] / max(n, m)

root = tk.Tk()
root.withdraw()
sens = simpledialog.askfloat(
    "Sensibilidade",
    "Digite o nível de sensibilidade para reconhecimento (mín: 0.03, máx: 0.15):",
    minvalue=0.03, maxvalue=0.15
)
root.destroy()
if sens is None:
    print("Operação cancelada pelo usuário.")
    exit()

cap = cv2.VideoCapture(camera)
print("Mostre um sinal para a câmera.")

cv2.namedWindow("HandConnect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HandConnect", 1366, 768)

ultimo_sinal = None
frames_mesmo_sinal = 0
FRAMES_PARA_CONFIRMAR = 5
frase_sinais = ""
BUFFER_MOVIMENTO = 20
buffer_frames = deque(maxlen=BUFFER_MOVIMENTO)
frase_traduzida = ""
speech_texto = ""


def resize_to_fit(img, max_width, max_height):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_text_box(img, text, box_x, box_y, box_w, box_h, title, font, font_scale, color, thickness, line_spacing=10):
    # Desenha o título
    (title_w, title_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    cv2.putText(img, title, (box_x + (box_w - title_w)//2, box_y + title_h + 10), font, font_scale, color, thickness)
    # Prepara o texto para caber na caixa
    max_width = box_w - 20  # margem
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (test_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if test_w > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    # Desenha as linhas
    y = box_y + title_h + 30
    for line in lines:
        if y + title_h > box_y + box_h - 10:
            break  # Não desenha fora da caixa
        cv2.putText(img, line, (box_x + 10, y), font, font_scale, color, thickness)
        y += title_h + line_spacing

# Função para mostrar a interface sem distorção (letterbox)
def show_letterbox(window_name, img, target_w=1366, target_h=768):
    screen = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    screen[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    cv2.imshow(window_name, screen)

while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Tenta capturar o próximo frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)
    landmarks = capturar_landmarks(frame, resultados)

    # --- UI personalizada ---
    ui = np.full((768, 1366, 3), (173, 124, 64), dtype=np.uint8)  # BGR para #407cad
    ui[:] = (173, 124, 64)

    # Caixa branca da câmera (central) - mais para baixo
    camera_x1, camera_y1 = 75, 160  # y1 aumentado de 130 para 160
    camera_x2, camera_y2 = 925, 610  # y2 aumentado de 580 para 610
    cv2.rectangle(ui, (camera_x1, camera_y1), (camera_x2, camera_y2), (230, 230, 230), -1)
    # Caixa branca superior direita
    cv2.rectangle(ui, (1050, 60), (1300, 340), (230, 230, 230), -1)
    # Caixa branca inferior direita
    cv2.rectangle(ui, (1050, 400), (1300, 680), (230, 230, 230), -1)

    # Redimensiona e insere o print da tela centralizado
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
        frame_resized = cv2.resize(frame, (camera_x2 - camera_x1, camera_y2 - camera_y1))
        ui[camera_y1:camera_y2, camera_x1:camera_x2] = frame_resized

    # Sobrepõe a logo da mão centralizada acima da área da câmera, um pouco mais para cima
    if logo is not None:
        logo_resized = resize_to_fit(logo, 100, 100)
        h, w = logo_resized.shape[:2]
        x_logo = camera_x1 + ((camera_x2 - camera_x1) - w) // 2
        y_logo = camera_y1 - h - 25  # 25 pixels acima da caixa da câmera
        ui = overlay_png(ui, logo_resized, x_logo, y_logo)
    # Sobrepõe o texto HandConnect centralizado abaixo da área da câmera, maior e um pouco mais para baixo
    if logo_text is not None:
        logo_text_resized = resize_to_fit(logo_text, 500, 100)
        h_text, w_text = logo_text_resized.shape[:2]
        x_text = camera_x1 + ((camera_x2 - camera_x1) - w_text) // 2
        y_text = min(camera_y2 + 30, 768 - h_text)
        if y_text >= 0 and x_text >= 0 and y_text + h_text <= 768 and x_text + w_text <= 1366:
            ui = overlay_png(ui, logo_text_resized, x_text, y_text)

    # No quadrado da tradução, mostra o texto bruto até apertar enter
    texto_para_mostrar = frase_traduzida if frase_traduzida else frase_sinais
    draw_text_box(ui, texto_para_mostrar, 1050, 60, 250, 280, "", cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    # No quadrado da voz, só mostra o texto reconhecido
    draw_text_box(ui, speech_texto, 1050, 400, 250, 280, "", cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    # Substitui cv2.imshow por show_letterbox para manter proporção
    show_letterbox("HandConnect", ui)

    if landmarks:
        buffer_frames.append(landmarks)
        nome, significado = None, None
        for sinal_nome, dados in pacote_sinais.items():
            if dados.get("tipo") == "movimento":
                for amostras_movimento in dados["amostras_movimento"]:
                    if len(buffer_frames) >= 5:
                        if len(buffer_frames) >= len(amostras_movimento):
                            janela = list(buffer_frames)[-len(amostras_movimento):]
                        else:
                            janela = list(buffer_frames)
                        buffer_norm = normalizar_trajetoria(janela)
                        movimento_norm = normalizar_trajetoria(amostras_movimento)
                        if len(buffer_norm) == len(movimento_norm):
                            dist_media = np.mean([
                                np.mean([
                                    np.linalg.norm(np.array(buffer_norm[i][j]) - np.array(movimento_norm[i][j]))
                                    for j in range(len(movimento_norm[i]))
                                ])
                                for i in range(len(movimento_norm))
                            ])
                            if dist_media < sens:
                                nome = sinal_nome
                                significado = dados.get("significado", "")
                                buffer_frames.clear()
                                break
                if nome:
                    break
        if not nome:
            nome, significado = reconhecer_sinal(landmarks, pacote_sinais)
        if nome:
            texto = f"Sinal: {nome} ({significado})"
            cor = (0, 255, 0)
            if nome == ultimo_sinal:
                frames_mesmo_sinal += 1
            else:
                frames_mesmo_sinal = 1
            if frames_mesmo_sinal == FRAMES_PARA_CONFIRMAR and (frase_sinais.strip().split()[-1] if frase_sinais.strip() else None) != nome:
                frase_sinais += f"{nome} "
            ultimo_sinal = nome
        else:
            texto = "Sinal não reconhecido"
            cor = (0, 0, 255)
            ultimo_sinal = None
            frames_mesmo_sinal = 0
    else:
        texto = "Mostre a mao para a camera"
        cor = (255, 255, 0)
        ultimo_sinal = None
        frames_mesmo_sinal = 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == 13 and frase_sinais.strip():
        frase_traduzida = traduzir_bruto_para_portugues(frase_sinais.strip())
        print("Frase traduzida:", frase_traduzida)
        frase_sinais = ""
        engine.stop()
        engine.say(frase_traduzida)
        engine.runAndWait()
    if key == ord('r'):
        frase_sinais = ""
        frase_traduzida = ""
    if key == ord('o') and not gravando_audio:
        gravando_audio = True
        recognizer = sr.Recognizer()
        mic = sr.Microphone(device_index=MICROFONE_INDEX)
        print("Fale algo...")
        with mic as source:
            try:
                audio_buffer = recognizer.listen(source, timeout=60)
            except sr.WaitTimeoutError:
                speech_texto = "[Nenhuma fala detectada]"
                gravando_audio = False
                continue
    if key == ord('p') and gravando_audio:
        gravando_audio = False
        try:
            speech_texto = recognizer.recognize_google(audio_buffer, language="pt-BR")
        except sr.UnknownValueError:
            speech_texto = "[Não entendi]"
        except sr.RequestError:
            speech_texto = "[Erro de conexão]"
    if key == ord('l'):
        speech_texto = ""

cap.release()
cv2.destroyAllWindows()

