import openai
import os

OPENAI_API_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def traduzir_bruto_para_portugues(texto_bruto):
    prompt = (
        "Você é um tradutor de Libras para português. "
        "Receberá frases em português direto do reconhecimento de sinais, "
        "com estrutura simples e sem conectivos. Existiram sinais parecidos que o programa tem dificuldade em diferenciar, entao use contexto para saber qual é o certo."
        "Reescreva a frase de forma natural, sem mudar o sentido. o programa nao reconhece acentos, nao os use nas tradações. ex: se a tradução for 'meu nome é' voce fara apenas 'meu nome e' sem acentos.\n\n"
        f"Frase bruta: {texto_bruto}\nFrase natural:"
    )
    resposta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.2,
    )
    return resposta.choices[0].message.content.strip()