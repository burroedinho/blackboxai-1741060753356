from flask import Flask, render_template, request
import speech_recognition as sr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    recognizer = sr.Recognizer()
    if 'file' not in request.files:
        return {'text': "Nenhum arquivo enviado."}
    
    file = request.files['file']
    if file.filename == '':
        return {'text': "Nenhum arquivo selecionado."}
    
    try:
        with sr.AudioFile(file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="pt-BR")
            return {'text': text}
    except sr.UnknownValueError:
        return {'text': "Não entendi o que você disse."}
    except sr.RequestError:
        return {'text': "Erro ao acessar o serviço de reconhecimento de voz."}

if __name__ == '__main__':
    app.run(debug=True)
