import os
import pandas as pd
import numpy as np
from feature_engineering_segment import extract_features


def getFeature(new_url):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    final_features_file = os.path.join(data_dir, "final_dataset_with_features_145626-146625.csv")
    try:
        df_final = pd.read_csv(final_features_file)
    except Exception as e:
        print(f"Load {final_features_file} error: {e}")
        return

    new_features = extract_features(new_url)
    final_columns = df_final.columns.tolist()
    final_features = {}

    url_protocol = new_features.get('protocol', '').lower()
    url_tld_set = new_features.get('unique_dns_tld_set', set())
    new_features.pop('protocol', None)
    new_features.pop('unique_dns_tld_set', None)

    for col in final_columns:
        if col == "url":
            final_features[col] = new_url
        elif col.startswith("protocol_"):
            prot = col.split("_", 1)[1].lower()
            final_features[col] = "TRUE" if \
                url_protocol == prot else "FALSE"
        elif col.startswith("tld_"):
            tld = col.split("_", 1)[1]
            final_features[col] = 1 \
                if tld in url_tld_set else 0
        else:
            final_features[col] = new_features.get(col, np.nan)

    final_features.pop("label", None)
    feature_values = []
    for key, value in final_features.items():
        if pd.isna(value):
            feature_values.append("0")
        else:
            feature_values.append(str(value))
    while len(feature_values) < 78:
        feature_values.append("0")
    return ",".join(feature_values)


if __name__ == '__main__':
    new_url = "https://www.baidu.com"
    print(getFeature(new_url))
