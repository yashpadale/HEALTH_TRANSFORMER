from flask import Flask, render_template, request, jsonify
from PIL import Image
from dictionary import dictionary
from generator import query_gen_sentences,train_model
from functions import translate_to_marathi,translate_to_english,detect_language,play_marathi_text,play_english_text


model, d, maxlen = train_model(dictionary_train=dictionary, maxlen=10, per=0.90,
                                   embed_dim=256, num_heads=8, ff_dim=128,
                                   num_blocks=8, dropout_rate=0.1, num_encoders=2, num_decoders=2, batch_size=64,
                                   epochs=5)

app = Flask(__name__)



@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload_text', methods=['POST'])
def upload_text():
    data = request.json
    user_input = data['text']
    output=query_gen_sentences(query=user_input,model=model,dictionary=d,maxlen=maxlen)
    response_data = {'message': f"{output} "}
    print(response_data)
    return jsonify(response_data)

@app.route('/language_translation', methods=['POST'])
def process_text():
    data = request.json
    text_content = data.get('text', '')
    selected_language = data.get('language', '')
    if selected_language=="mr":
        output = translate_to_marathi(text=text_content)
        response_data = {'translated_text': f'{output}'}
        print(f'response_data - {response_data}')
        return jsonify(response_data)
    if selected_language=='en':
        output = translate_to_english(text=text_content)
        response_data = {'translated_text': f'{output}'}
        return jsonify(response_data)



@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image = Image.open(file)
        resized_image = image.resize((224, 224))
        #output=cnn_output(input_image=resized_image)
        return jsonify({'filename': file.filename, 'message': f"You apploaded a photo is it {file.filename}"})

    return jsonify({'error': 'Unknown error'})


app.run()