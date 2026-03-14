import os
import re
import math
import numpy as np
import pandas as pd
import concurrent.futures
import time
import urllib.parse
import urllib.request
import ipaddress
from datetime import datetime
import whois          # pip install python-whois
import dns.resolver   # pip install dnspython
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import logging
import requests
import warnings
import ast
logging.getLogger("whois.whois").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# A function that extracts domain-based features
def extract_domain_features(host, max_retries=20, retry_delay=1):
    features = {}

    for attempt in range(max_retries):
        # 1. Get WHOIS information for domain-based features
        try:
            # (1) Get the creation date
            w = whois.whois(host)
            creation_date = w.creation_date

            # (2) Judge if the domain is registered (1) or not (0)
            features['registered'] = int(creation_date is not None)

            # (3) Calculate the age of the domain
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if isinstance(creation_date, datetime):
                features['domain_age'] = (datetime.now() -
                                          creation_date).days
            elif isinstance(creation_date, str):
                try:
                    dt = datetime.strptime(creation_date, "%Y-%m-%d")
                    features['domain_age'] = \
                        (datetime.now() - dt).days
                except Exception:
                    features['domain_age'] = np.nan
            else:
                features['domain_age'] = np.nan

            # (4) Calculate how long until it expires
            expiration_date = w.expiration_date
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            if isinstance(expiration_date, datetime):
                features['domain_expiry'] = (expiration_date -
                                             datetime.now()).days
            else:
                features['domain_expiry'] = np.nan

            # (5) Get the DNS server name of the domain
            ns = w.name_servers if w.name_servers else None

            # If ns is empty, fill the features with default value
            if ns is None:
                features['unique_dns_tld_set'] = set()

            # If ns is not empty
            else:
                # Parse strings or other forms of ns into lists
                if isinstance(ns, list):
                    ns_list = ns
                elif isinstance(ns, str):
                    try:
                        ns_list = ast.literal_eval(ns)
                    except Exception:
                        ns_list = [ns]
                else:
                    ns_list = []

                # Split the servers by space and get the token that
                # has the valid domain pattern
                valid_tokens = []
                domain_pattern = re.compile(r'^[a-zA-Z][a-zA-Z0-9\-]*'
                                            r'(?:\.[a-zA-Z0-9\-]+)+$')
                for server in ns_list:
                    if not isinstance(server, str):
                        continue
                    tokens = server.strip().split()
                    for token in tokens:
                        token = token.strip().strip('.')
                        if token.lower().startswith("www"):
                            continue
                        if domain_pattern.match(token):
                            valid_tokens.append(token)

                # Extract TLD from every valid token
                tld_set = set()
                for token in valid_tokens:
                    parts = token.split('.')
                    if parts:
                        tld_set.add(parts[-1])
                features['unique_dns_tld_set'] = tld_set
            break

        # (6) Fill the exception cases with default values
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                features['registered'] = 0
                features['domain_age'] = np.nan
                features['domain_expiry'] = np.nan
                features['unique_dns_tld_set'] = set()

    # 2. Judge if the domain can be resolved by DNS
    try:
        dns.resolver.resolve(host, 'A')
        features['dns_valid'] = 1
    except Exception:
        features['dns_valid'] = 0

    # 3. Return all the domain-based features
    return features


# A function that calculates data entropy
def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    length = len(data)
    for char in set(data):
        p_x = float(data.count(char)) / length
        entropy += - p_x * math.log2(p_x)
    return entropy


# A function that extracts HTML and java-script features
def extract_html_js_features(url, max_retries=3, retry_delay=1):
    features = {}
    for attempt in range(max_retries):
        try:
            # Use requests to get the HTML content of the page
            response = requests.get(url, timeout=(5, 10))
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Link tags (<a> tags)
            links = soup.find_all('a')
            features['num_links'] = len(links)

            # Image tags (<img> tags)
            images = soup.find_all('img')
            features['num_images'] = len(images)

            # Form tags (<form> tags)
            forms = soup.find_all('form')
            features['num_forms'] = len(forms)

            # Stylesheet links (<link rel="stylesheet">)
            stylesheets = soup.find_all('link', rel='stylesheet')
            features['num_stylesheets'] = len(stylesheets)

            # Meta tags (<meta> tags)
            metas = soup.find_all('meta')
            features['num_meta_tags'] = len(metas)

            # Title tag length
            title_tag = soup.find('title')
            features['title_length'] = len(title_tag.text.strip())\
                                       if title_tag else 0

            # Count the number of inline frames (<iframe> tags)
            iframes = soup.find_all('iframe')
            features['num_iframes'] = len(iframes)

            # Check for obfuscated JavaScript by analyzing <script> tags
            scripts = soup.find_all('script')
            obfuscated_script_count = 0
            entropy_threshold = 4.5
            for script in scripts:
                script_content = script.get_text()
                # Only analyze longer scripts
                if len(script_content) > 100:
                    entropy = calculate_entropy(script_content)
                    if entropy > entropy_threshold:
                        obfuscated_script_count += 1
            features['num_obfuscated_scripts'] = obfuscated_script_count
            return features

        # Deal with retry mechanism
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    # Fill the default values if all retries fail
    features = {
        'num_links': np.nan,
        'num_images': np.nan,
        'num_forms': np.nan,
        'num_stylesheets': np.nan,
        'num_meta_tags': np.nan,
        'title_length': np.nan,
        'num_iframes': np.nan,
        'num_obfuscated_scripts': np.nan,
    }
    return features


# A function that combines all features
def extract_features(url):
    # 1. Initialize the features dictionary
    features = {}

    # 2. Extract bar-based features
    # (1) Get url length and protocol in the host
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname if parsed.hostname else ''
    features['protocol'] = parsed.scheme
    features['url_length'] = len(url)

    # (2) Get the number of dots, digits, special char and hyphen
    features['digit_count'] = sum(c.isdigit() for c in url)
    features['special_char_count'] = \
        len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dots'] = host.count('.')
    features['hyphen_count'] = host.count('-')

    # (3) Judge if a host is in the form of ip address
    try:
        ipaddress.ip_address(host)
        features['is_ip'] = 1
        features['num_subdomains'] = 0
    except ValueError:
        features['is_ip'] = 0
        # Number of subdomains excludes main domain & TLD
        domain_parts = host.split('.')
        features['num_subdomains'] = max(len(domain_parts) - 2, 0)

    # 3. Extract domain-based features
    domain_features = extract_domain_features(host)
    features.update(domain_features)

    # 4. Extract HTML and JavaScript-based features (e.g.,
    # presence of inline frames or obfuscated scripts)
    html_js_features = extract_html_js_features(url)
    features.update(html_js_features)

    # 5. Return all the features
    return features


# A function that processes a data segment
def process_segment(df, start, end):
    # 1. Slice the dataset
    original_df = df.iloc[start - 1: end]

    # 2. Extract features from every url
    urls = original_df['url'].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) \
            as executor:
        feature_list = list(executor.map(extract_features, urls))
    features_df = pd.DataFrame(feature_list)

    # 3. Combine the original dataframe with the features
    final_df = pd.concat([original_df.reset_index(drop=True),
                          features_df], axis=1)
    return final_df


# Main function
def main():
    # 1. Read final_dataset.csv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    input_file = os.path.join(data_dir, 'final_dataset.csv')
    df = pd.read_csv(input_file)

    # 2. Define the segments and maximum records
    segment_size = 5000
    start = 1
    max_records = len(df)

    # 3. Process the segments
    while start <= max_records:
        end = min(start + segment_size - 1, max_records)
        success = False
        print(f"Processing segment: {start}-{end}")
        while not success:
            try:
                segment_df = process_segment(df, start, end)
                output_file = os.path.join(data_dir, f'final_dataset'
                              f'_with_features_{start}-{end}.csv')
                segment_df.to_csv(output_file, index=False)
                print(f"Segment {start}-{end} processed successfully."
                      f"\nSaved at {output_file}.")
                success = True
            except Exception as e:
                print(f"Error processing segment {start}-{end}: {e}")
                print("Retrying the current segment...")
                time.sleep(1)
        start = end + 1


if __name__ == '__main__':
    main()
