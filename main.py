import io # creating CSV files
from concurrent.futures.thread import ThreadPoolExecutor # for doing multiple tasks at the same time
from flask import Flask , request, jsonify, Response
import re # check if domain is IP address
from urllib.parse import urlparse, urlunparse # split URL parts
import torch
import torch.nn as nn
from flask_cors import CORS #allow communication between front-end and back-end
import socket # resolve domain to IP address
import pandas as pd
from feature_extraction_for_new_URLs import getFeature # change URL to feature vector
from lstm_v2_demo import predict_from_sample_row #chanfe features to prediction
from werkzeug.utils import secure_filename
import os

app = Flask(__name__) #create Flask server
app.config['PREFERRED_URL_SCHEME'] = 'https'
CORS(app, resources={ #link front-end and back-end
    r"/detect": {"origins": "*"},
    r"/upload_csv": {"origins": "*"},  
    r"/export_csv": {"origins": "*"}
})

class LSTMClassifier(nn.Module): #define LSTM Structure (neural network)
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #convert characters to vectors
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional) #process sequences of URL
        self.dropout = nn.Dropout(dropout) #Prevent overfitting
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim) #final output layer (predict phishing or not)


    def forward(self, x): # define sequence of operations for LSTM
        embedded = self.embedding(x)
        output, (hidden, _) = self.lstm(embedded)
        hidden = self.dropout(hidden)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(hidden)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_to_idx = torch.load('char_to_idx.pth') 
# parameters for LSTM model
vocab_size = len(char_to_idx) + 1  #unique characters + 1 for unknown character
embedding_dim = 128 #size of each character vector 
hidden_dim = 128 # number of neurons in LSTM
output_dim = 1 # binary classification (phishing or not)
num_layers = 2 # number of LSTM layers
dropout = 0.5 # dropout rate to prevent overfitting

baseModel = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
baseModel.load_state_dict(
        torch.load("phishing_detector_lstm_fold3.pth", map_location=device))

baseModel.to(device) # move model to GPU if available
baseModel.eval() # set model to evaluation mode1    

max_length = 100

@app.route('/')
def home():
    return "Welcome to the Phishing Detection System!"

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/detect', methods=['POST'])
def detect_phishing():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    return predict_from_sample_row(getFeature(url))
    return result

def getResult(url):
    return predict_from_sample_row(getFeature(url))
@app.route('/upload_csv', methods=['POST'])
def upload_csv():

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV'}), 400
    try:
        df = pd.read_csv(file)
        if 'url' not in df.columns:
            return jsonify({'error': 'CSV file must contain "url" column'}), 400
            results = []
        for url in df['url']:
            if pd.notna(url):  
                result = predict_from_sample_row(getFeature(url))
                results.append(result)
        urls = [url for url in df['url'] if pd.notna(url)]

        with ThreadPoolExecutor(max_workers=100) as executor: # create a thread pool to process multiple URLs at the same time
            results = list(executor.map(getResult, urls))

        return jsonify({
            'message': 'File processed successfully',
            'results': results,
            'total': len(results)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


def resolve_domain(url): # resolve domain names to IP addresses for better model performance
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname and re.match(r'^\d+\.\d+\.\d+\.\d+$', hostname):
            return url

        if hostname:
            ip = socket.gethostbyname(hostname)
    
            new_netloc = ip
            if parsed.port:
                new_netloc += f':{parsed.port}'

            parsed = parsed._replace(netloc=new_netloc)
            return urlunparse(parsed)
        return url
    except:
        return url  


def preprocess_url(url, char_to_idx, max_length): # convert URL to characters , convert characters to numbers and pad or truncate to fixed length
    processed_url = resolve_domain(url)
    sequence = [char_to_idx.get(char, 0) for char in processed_url] 
    if len(sequence) < max_length:
        padded = sequence + [0] * (max_length - len(sequence))
    else:
        padded = sequence[:max_length]
    return torch.LongTensor([padded])


def predict_url(url):
    input_tensor = preprocess_url(url, char_to_idx, max_length)
    with torch.no_grad():
        output = baseModel(input_tensor.to('cuda'))
        probability = torch.sigmoid(output).item()
    return {
        'original_url': url,
        'processed_url': resolve_domain(url), 
        'phishing_probability': probability,
        'prediction': 'phishing' if probability > 0.5 else 'legitimate'
    }
...
@app.route('/export_csv', methods=['POST']) # endpoint to export results as CSV file
def export_csv():
    data = request.json
    if not data or 'data' not in data:
        return jsonify({'error': 'Invalid data format'}), 400

    try:
        df = pd.DataFrame(data['data'])
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=export.csv"}
        )
    except Exception as e:
        return jsonify({'error': f'Error exporting CSV: {str(e)}'}), 500

if __name__ == '__main__':
    app.run()

