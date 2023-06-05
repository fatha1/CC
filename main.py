import itertools
import uvicorn, re
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TextClassificationPipeline
from transformers import BertTokenizer, BertForSequenceClassification
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = FastAPI()  # create a new FastAPI app instance
# port = int(os.getenv("PORT"))
port = 8080

# Define a Pydantic model for an item
class Item(BaseModel):
    query:str

model = BertForSequenceClassification.from_pretrained("./model", from_tf=True)                 # sesuaikan dengan nama folder model yang sudah diupload
tokenizer = BertTokenizer.from_pretrained("./tokenizer", local_files_only=True)               # sesuaikan dengan nama folder tokenizer yang sudah diupload
data_rekomendasi = pd.read_csv("./content/Dataset Rekomendasi Hasil Prediksi Merge.csv", sep=';')     # sesuaikan dengan nama file data rekomendasi yang sudah diupload
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k = 24)

factory = StopWordRemoverFactory()
addStopwords = ['saya', 'itu', 'juga']
removeStopwords = ['atau', 'dalam', 'dan', 'dari', 'di', 'pada', 'ke', 'saat','sekitar', 'seperti', 'tidak', 'yang']
stopWords = factory.get_stop_words()+addStopwords

for removeStopword in removeStopwords:
    if removeStopword in stopWords:
        stopWords.remove(removeStopword)
    else:
        continue

dictionaryWord = ArrayDictionary(stopWords)
stopWordRemover = StopWordRemover(dictionaryWord)

def preprocessing_user_input(text):
    global stopWordRemover

    text = re.sub('  +', ' ', text)
    text = re.sub(r'[^\x00-\x7f]','r', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))
    text = text.lower()
    text = stopWordRemover.remove(text)

    return text

def predict(input):
    pred = pipe(preprocessing_user_input(input))
    kelas = pred[0][0]['label'].title()
    prob = round((pred[0][0]['score'])*100, 2)
    return(kelas, prob)

@app.get("/")
def main_page():
    return (
        "Selamat datang di FastAPI untuk klasifikasi teks SymptoMed. Silahkan gunakan metode POST untuk mengirimkan data. "
    )

@app.post("/")
def add_item(item: Item):
    global data_rekomendasi

    if len(item.query) < 25 or len((item.query).split()) < 4:
        hasil = 'Karakter terlalu sedikit'
        probability = ''
        link = ''
        saran = 'Cobalah masukkan gejala yang lebih detail'
    else:
        hasil, probability = predict(item.query)

        if probability < 50:
            hasil = 'Tidak Ada Kecocokan'
            probability = ''
            link = ''
            saran = 'Cobalah masukkan gejala yang lebih spesifik'
        else:
            probability = str(probability) + '%'
            indeks_hasil = data_rekomendasi[data_rekomendasi['Symptom'] == hasil.lower()]
            link = indeks_hasil['Detail'].values[0]
            saran = indeks_hasil['Saran'].values[0]

    return {
        "Kelas": hasil,
        "Proabilitas": probability,
        "link": link,
        "Rekomendasi": saran
    }

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=port, timeout_keep_alive=1200)