import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load

FEATURE_COLUMNS = [
    'url_length', 'digit_count', 'special_char_count', 
    'num_dots', 'hyphen_count', 'is_ip', 'num_subdomains',
    'registered', 'domain_age', 'domain_expiry', 'dns_valid',
    
    'num_links', 'num_images', 'num_forms', 'num_stylesheets',
    'num_meta_tags', 'title_length', 'num_iframes',
    'num_obfuscated_scripts',
    
    'protocol_http', 'protocol_https',
    
    'tld_BIZ', 'tld_BR', 'tld_CA', 'tld_COM', 'tld_DE', 'tld_FR',
    'tld_INFO', 'tld_JP', 'tld_LA', 'tld_MX', 'tld_NET', 'tld_ORG',
    'tld_RU', 'tld_TECH', 'tld_TH', 'tld_TR', 'tld_TW', 'tld_UK',
    'tld_ac', 'tld_ae', 'tld_africa', 'tld_ar', 'tld_au', 'tld_biz',
    'tld_ca', 'tld_cloud', 'tld_cn', 'tld_co', 'tld_com', 'tld_cz',
    'tld_de', 'tld_eu', 'tld_fr', 'tld_host', 'tld_id', 'tld_in',
    'tld_info', 'tld_io', 'tld_it', 'tld_jp', 'tld_kr', 'tld_mx',
    'tld_net', 'tld_nl', 'tld_org', 'tld_pk', 'tld_pl', 'tld_rs',
    'tld_ru', 'tld_se', 'tld_tr', 'tld_tw', 'tld_ua', 'tld_uk', 'tld_za'
]
device='cuda' if torch.cuda.is_available() else 'cpu'
class FeatureLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                output_dim=1, num_layers=2):
        super(FeatureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        output, (hidden, cell) = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)

def load_pipeline(filepath, device='cpu'):
    pipeline = load(filepath)
    model = FeatureLSTM(
        input_dim=pipeline['model_meta']['input_dim'],
        hidden_dim=pipeline['model_meta']['hidden_dim'],
        num_layers=pipeline['model_meta']['num_layers']
    ).to(device)
    
    model.load_state_dict(pipeline['model_state'])
    model.eval()
    
    return model, pipeline['preprocessor']

def prepare_sample(sample_row):
    fields = sample_row.strip().split(',')
    
    url = fields[0]
    label = fields[1]
    feature_values = fields[2:]
    
    features = {}
    for col, val in zip(FEATURE_COLUMNS, feature_values):
        if val == '':
            features[col] = np.nan
        elif val.lower() in ['true', 'false']:
            features[col] = 1 if val.lower() == 'true' else 0
        else:
            try:
                features[col] = float(val)
            except:
                features[col] = np.nan
    return pd.DataFrame([features])


def predict_single_sample(model, preprocessor, sample_df):
    X = preprocessor['imputer'].transform(sample_df)
    X = preprocessor['scaler'].transform(X)
    X_tensor = torch.FloatTensor(X).to(device)
    model.eval()
    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.sigmoid(output).item()
    return 1 if prob > 0.5 else 0, prob


def predict_from_sample_row(sample_row, model_path="best_fold4.joblib"):
    model, preprocessor = load_pipeline(model_path, device=device)
    sample_df = prepare_sample(sample_row)
    pred_class, pred_prob = predict_single_sample(model, preprocessor, sample_df)

    return {
        'original_url': sample_row.split(',')[0],
        'probability': pred_prob,
        'prediction': 'phishing' if pred_class == 1 else 'legitimate',
        'class': pred_class
    }

if __name__ == "__main__":
    sample_row = "https://www.baidu.com,nan,21,0,5,2,0,0,1,1,9316,1276,nan,1,11,2,1,1,3,27,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    print(predict_from_sample_row(sample_row))