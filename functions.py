from googletrans import Translator
from gtts import gTTS
import pygame
from io import BytesIO
def translate_to_marathi(text):
    translator = Translator()
    try:
        translation = translator.translate(text, src='en', dest='mr')
        return translation.text
    except Exception as e:
        print("Translation error:", e)
        return None

def translate_to_english(text):
    translator = Translator()
    try:
        translation = translator.translate(text, src='mr', dest='en')
        return translation.text
    except Exception as e:
        print("Translation error:", e)
        return None
from langdetect import detect

def detect_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        print("Error during language detection:", str(e))
        return None


'''
o="how are u"
f=detect_language(text=o)
print(f)
o="माझा पाय खूप वेदना होत आहे"
f=detect_language(text=o)
print(f)
'''




def play_marathi_text(text):
    # Create a gTTS object
    tts = gTTS(text=text, lang='mr')

    # Create an in-memory file object
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Initialize Pygame mixer
    pygame.mixer.init()

    pygame.mixer.music.load(audio_file)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

def play_english_text(text):
    # Create a gTTS object
    tts = gTTS(text=text, lang='en')

    # Create an in-memory file object
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Initialize Pygame mixer
    pygame.mixer.init()

    pygame.mixer.music.load(audio_file)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue
def return_order(query,dictionary):
    d_k=list(dictionary.keys())
    query=query.split(' ')
    order=[]
    for i in range(len(query)):
        for j in range(len(d_k)):
            if d_k[j].lower()==query[i].lower():
                order.append(j)
    return order

def words_to_numbers(sentence:str):
    unique_words=set()
    sentence=sentence.split(' ')
    for i in range(len(sentence)):
        unique_words.add(sentence[i])
    return list(unique_words)

def return_dict(unique_words:list):
    dictionary={}
    for i in range(len(unique_words)):
        dictionary[unique_words[i]]=i
    return dictionary


def create_batches(list1:list):
    return list1

def split_list(lst,per):
    len_75=int(len(lst)*per)
    first_list=lst[:len_75]
    second_list=lst[len_75:]
    return first_list,second_list


def number_to_words(lst, dictionary):
    d_k = list(dictionary.keys())
    d_v = list(dictionary.values())
    o = []
    for i in range(len(lst)):
        for j in range(len(d_v)):
            if d_v[j] == lst[i]:
                o.append(d_k[j])
    return ' '.join(o)

