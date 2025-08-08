# HandConnect - Protótipo

# Etapas principais:
# 1. Gravação de sinais customizados (etapa atual)
# 2. Reconhecimento de gestos com MediaPipe
# 3. Mapeamento gesto -> texto bruto (pacote de sinais)
# 4. Tradução com API do ChatGPT
# 5. Conversão texto -> fala (TTS)
# 6. Interface gráfica bonita/simples/moderna

import cv2
import mediapipe as mp
import pickle
import os
import shutil
import tkinter as tk
import numpy as np
from tkinter import simpledialog, messagebox
import glob

camera = 0

# Inicialização do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def perguntar_nome_pacote():
    root = tk.Tk()
    root.withdraw()
    nome = simpledialog.askstring("Nome do Pacote", "Digite o nome do novo pacote de sinais:")
    root.destroy()
    if not nome:
        print("Operação cancelada pelo usuário.")
        exit()
    return nome

# Pergunta o nome do pacote ao iniciar
nome_pacote = perguntar_nome_pacote()
CAMINHO_PACOTE = f"pacote_{nome_pacote}.pkl"

# Carregar ou criar pacote
if os.path.exists(CAMINHO_PACOTE):
    with open(CAMINHO_PACOTE, 'rb') as f:
        pacote_sinais = pickle.load(f)
else:
    pacote_sinais = {}


def capturar_landmarks(frame, resultados):
    todas_maos = []
    if resultados.multi_hand_landmarks:
        for idx, mao in enumerate(resultados.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                mao,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            landmarks = [(lm.x, lm.y, lm.z) for lm in mao.landmark]
            todas_maos.append(landmarks)
    return todas_maos


def perguntar_nome_significado():
    root = tk.Tk()
    root.withdraw()
    nome = simpledialog.askstring("Novo Sinal", "Digite o nome do novo sinal:")
    if nome is None:
        return None, None
    significado = simpledialog.askstring("Significado", "Digite o significado do sinal:")
    root.destroy()
    if significado is None:
        return None, None
    return nome, significado


def perguntar_continuar():
    root = tk.Tk()
    root.withdraw()
    resposta = messagebox.askyesno("Continuar", "Deseja capturar outro sinal?")
    root.destroy()
    return resposta


def perguntar_tipo_sinal():
    import tkinter as tk

    tipo = None

    def escolher_estatico():
        nonlocal tipo
        tipo = 'e'
        root.quit()

    def escolher_movimento():
        nonlocal tipo
        tipo = 'm'
        root.quit()

    root = tk.Tk()
    root.title("Tipo de Sinal")
    root.geometry("300x120")
    tk.Label(root, text="Escolha o tipo de sinal:", font=("Arial", 12)).pack(pady=10)
    btn_estatico = tk.Button(root, text="Estático", width=12, command=escolher_estatico)
    btn_estatico.pack(side="left", padx=20, pady=20)
    btn_movimento = tk.Button(root, text="Movimento", width=12, command=escolher_movimento)
    btn_movimento.pack(side="right", padx=20, pady=20)
    root.mainloop()
    root.destroy()
    return tipo


def gravar_novo_sinal():
    while True:
        nome, significado = perguntar_nome_significado()
        if not nome or not significado:
            print("Operação cancelada pelo usuário.")
            break

        if nome in pacote_sinais:
            root = tk.Tk()
            root.withdraw()
            confirmar = messagebox.askyesno("Sobrescrever", f"Sinal '{nome}' já existe. Deseja sobrescrever?")
            root.destroy()
            if not confirmar:
                print("Operação cancelada.")
                continue

        # Escolha do tipo de sinal
        tipo = perguntar_tipo_sinal()
        if tipo is None:
            print("Tipo de sinal inválido ou operação cancelada.")
            continue

        cap = cv2.VideoCapture(camera)

        if tipo.lower() == 'e':
            capturas = []
            print("Mostre o sinal para a câmera. Pressione 'c' para capturar, 'q' para cancelar.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = hands.process(frame_rgb)
                todas_maos = capturar_landmarks(frame, resultados)

                cv2.putText(frame, f"Capturas: {len(capturas)}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Gravando sinal", frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('c'):
                    if todas_maos:
                        if len(capturas) < 10:
                            capturas.append(todas_maos[0])
                            print(f"Captura {len(capturas)} salva.")
                        else:
                            print("Limite de 10 capturas atingido.")
                    else:
                        print("Nenhuma mão detectada. Tente novamente.")
                elif tecla == ord('q'):
                    print("Cancelado.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if len(capturas) == 10:
                    print("Limite de 10 capturas atingido para este sinal.")
                    break

            cap.release()
            cv2.destroyAllWindows()

            if capturas:
                def normalizar(landmarks):
                    arr = np.array(landmarks)
                    centroide = arr.mean(axis=0)
                    return (arr - centroide).tolist()

                capturas_normalizadas = [normalizar(c) for c in capturas]

                if os.path.exists(CAMINHO_PACOTE):
                    shutil.copy(CAMINHO_PACOTE, CAMINHO_PACOTE + ".bak")

                pacote_sinais[nome] = {
                    "amostras": capturas_normalizadas,
                    "significado": significado,
                    "tipo": "estatico"
                }
                with open(CAMINHO_PACOTE, 'wb') as f:
                    pickle.dump(pacote_sinais, f)
                print(f"Sinal estático '{nome}' salvo com sucesso com {len(capturas)} exemplos e significado: '{significado}'.")

        else:  # tipo == 'm'
            movimentos = []
            print("Você pode gravar até 10 movimentos para este sinal.")
            while len(movimentos) < 10:
                print(f"Pressione 'm' para começar a gravar o movimento {len(movimentos)+1}, 's' para parar, ou 'q' para cancelar.")
                movimento = []
                gravando_movimento = False

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resultados = hands.process(frame_rgb)
                    todas_maos = capturar_landmarks(frame, resultados)

                    cv2.putText(frame, f"Movimentos gravados: {len(movimentos)}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow("Gravando sinal", frame)
                    tecla = cv2.waitKey(1) & 0xFF

                    if tecla == ord('m'):
                        gravando_movimento = True
                        movimento = []
                        print("Gravação de movimento iniciada.")
                    elif tecla == ord('s') and gravando_movimento:
                        gravando_movimento = False
                        print("Gravação de movimento finalizada.")
                        break
                    elif tecla == ord('q'):
                        print("Cancelado.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    if gravando_movimento and todas_maos:
                        movimento.append(todas_maos[0])  # Salva a mão do frame atual

                if movimento:
                    def normalizar(landmarks):
                        arr = np.array(landmarks)
                        centroide = arr.mean(axis=0)
                        return (arr - centroide).tolist()
                    movimento_normalizado = [normalizar(c) for c in movimento]
                    movimentos.append(movimento_normalizado)
                    print(f"Movimento {len(movimentos)} salvo com {len(movimento)} frames.")

            cap.release()
            cv2.destroyAllWindows()

            if movimentos:
                if os.path.exists(CAMINHO_PACOTE):
                    shutil.copy(CAMINHO_PACOTE, CAMINHO_PACOTE + ".bak")

                pacote_sinais[nome] = {
                    "amostras_movimento": movimentos,
                    "significado": significado,
                    "tipo": "movimento"
                }
                with open(CAMINHO_PACOTE, 'wb') as f:
                    pickle.dump(pacote_sinais, f)
                print(f"Sinal de movimento '{nome}' salvo com sucesso com {len(movimentos)} exemplos e significado: '{significado}'.")

        if not perguntar_continuar():
            break


if __name__ == "__main__":

    gravar_novo_sinal()
