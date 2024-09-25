import time

from flask import Flask
from catboost import CatBoostClassifier
from flask_cors import CORS
import pandas as pd
import re
from flask import Flask, request, jsonify
import os
from bs4 import BeautifulSoup
import requests
from catboost import CatBoostClassifier
import urllib.parse
import pandas as pd
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import warnings
import pandas as pd
from requests.exceptions import InvalidSchema
import tldextract  # Import tldextract for more accurate domain extraction

from requests.exceptions import InvalidSchema

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# # Load your CatBoost model
# model = CatBoostClassifier()
# model.load_model('pretrained_model.bin')


# Load the whitelist domains from the CSV file
def load_whitelist_domains(csv_path):
    df = pd.read_csv(csv_path)
    unique_domains = df['domain'].drop_duplicates().tolist()
    return set(unique_domains)


def extract_main_domain(url):
    extracted = tldextract.extract(url)
    if extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    return extracted.domain


whitelist_domains = load_whitelist_domains('Unique_Domains_Whitelist.csv')

# Load the pretrained model
# clf = CatBoostClassifier()
# clf.load_model('pretrained_model.bin')



def count_hyperlinks(html_content):
    try:
        html_content = os.path.join(html_content)
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Count the number of hyperlinks (href, link, and src tags)
        total_hyperlinks = len(soup.find_all(['a', 'link', 'img']))

        # Determine if the website is legitimate or phishing based on the hyperlink count
        return 1 if total_hyperlinks == 0 else 0

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Error occurred


def calculate_internal_link_ratio(html_content, base_domain):
    try:
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(html_content, 'r', encoding='utf-8').read()

        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        total_links = 0
        internal_links = 0

        for a_tag in soup.find_all('a', href=True):
            total_links += 1
            href = a_tag['href']
            parsed_href = urllib.parse.urlparse(href)

            if not parsed_href.netloc:  # Relative link
                internal_links += 1
            elif parsed_href.netloc == base_domain:  # Internal link
                internal_links += 1

        if total_links == 0:
            return 1  # Avoid division by zero

        internal_link_ratio = internal_links / total_links

        if internal_link_ratio >= 0.5:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Error occurred


def calculate_external_link_ratio(html_content, base_domain):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        total_links = 0
        external_links = 0

        for a_tag in soup.find_all('a', href=True):
            total_links += 1
            href = a_tag['href']
            parsed_href = urllib.parse.urlparse(href)

            if parsed_href.netloc and parsed_href.netloc != base_domain:  # External link
                external_links += 1

        if total_links == 0:
            return 1  # Avoid division by zero

        external_link_ratio = external_links / total_links

        if external_link_ratio < 0.5:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 0


def has_external_css(html_content, base_domain):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        for link_tag in soup.find_all('link', rel='stylesheet', href=True):
            href = link_tag['href']
            parsed_href = urllib.parse.urlparse(href)

            if parsed_href.netloc and parsed_href.netloc != base_domain:
                return 1  # External CSS found with a foreign domain

        return 0  # No external CSS with a foreign domain found

    except Exception as e:
        print(f"Error: {e}")
        return 0


def is_suspicious_form_action(html_content, base_domain):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        for form_tag in soup.find_all('form', action=True):
            action = form_tag['action']
            parsed_action = urllib.parse.urlparse(action)

            if parsed_action.netloc and parsed_action.netloc != base_domain:
                return 1  # Suspicious form action with external link or PHP file

            # Additional checks for common suspicious values
            if action.lower() in {'null', '#', 'javascript:void()', 'javascript:void(0)'}:
                return 1  # Suspicious form action with null or JavaScript value

        return 0  # No suspicious form action found

    except Exception as e:
        print(f"Error: {e}")
        return 0

def detect_null_links(html_content):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all anchor tags in the HTML content
    anchors = soup.find_all('a')

    if not anchors:  # If no anchors found, return 0 (no null links)
        return 0

    # Count the number of null links (href contains '#', '#content', 'javascript:void(0)')
    null_links = 0
    total_anchors = len(anchors)

    for anchor in anchors:
        href = anchor.get('href', '')
        if href in ['#', '#content', 'javascript:void(0)']:
            null_links += 1

    # Calculate the ratio of null links
    null_link_ratio = null_links / total_anchors

    # If the ratio of null links is greater than 0.34, return 1 (phishing), else return 0 (legitimate)
    return 1 if null_link_ratio > 0.34 else 0

def is_external_favicon(html_content, base_domain):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        for link_tag in soup.find_all('link', rel='icon', href=True):
            href = link_tag['href']
            parsed_href = urllib.parse.urlparse(href)

            if parsed_href.netloc and parsed_href.netloc != base_domain:
                return 1  # External favicon found with a foreign domain

        return 0  # No external favicon with a foreign domain found

    except Exception as e:
        print(f"Error: {e}")
        return 0


def calculate_common_page_detection_ratio(html_content):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        anchor_links = [a_tag['href'] for a_tag in soup.find_all('a', href=True)]
        if not anchor_links:
            return 0  # No anchor links found

        most_common_link = max(set(anchor_links), key=anchor_links.count)
        common_link_frequency = anchor_links.count(most_common_link)

        common_page_detection_ratio = common_link_frequency / len(anchor_links)
        return common_page_detection_ratio

    except Exception as e:
        print(f"Error: {e}")
        return 0


def calculate_common_page_in_footer_ratio(html_content):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract anchor links within the footer section
        footer_anchor_links = [a_tag['href'] for footer_tag in soup.find_all('footer') for a_tag in
                               footer_tag.find_all('a', href=True)]

        if not footer_anchor_links:
            return 0  # No anchor links found in the footer section

        most_common_link = max(set(footer_anchor_links), key=footer_anchor_links.count)
        common_link_frequency = footer_anchor_links.count(most_common_link)

        common_page_in_footer_ratio = common_link_frequency / len(footer_anchor_links)
        return common_page_in_footer_ratio

    except Exception as e:
        print(f"Error: {e}")
        return None


def check_sfh(html_content, base_domain):
    try:
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        # html_content = os.path.join(PATH, html_content)
        # html_content = open(os.path.join(PATH, html_content), 'r', encoding='utf-8').read()
        html_content = open(html_content, 'r', encoding='utf-8').read()
        soup = BeautifulSoup(html_content, 'html.parser')

        for form_tag in soup.find_all('form', action=True):
            sfh = form_tag['action']

            # Check for empty or 'about:blank' SFH
            if sfh.lower() in {'', 'about:blank'}:
                return 0.5  # SFH is empty or 'about:blank'

            # Check if SFH domain is external
            parsed_sfh = urllib.parse.urlparse(sfh)
            if parsed_sfh.netloc and parsed_sfh.netloc != base_domain:
                return 1  # SFH domain is external

        return 0  # Legitimate SFH

    except Exception as e:
        print(f"Error: {e}")
        return 0

# Define the feature extraction functions
def has_www(url):
    pattern = re.compile(r'^www\.')
    match = re.search(pattern, url)
    return 1 if bool(match) else 0


def count_subdomains(url):
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]
    parts = url.split('.')
    part_count = len(parts)
    return 0.0 if part_count == 2 else 0.5 if part_count == 3 else 1.0


def check_ip_in_domain(url):
    domain_match = re.search(r'https?://([^/]+)', url)
    if domain_match:
        domain = domain_match.group(1)
        return 1.0 if re.match(r'^\d+\.\d+\.\d+\.\d+$', domain) or re.match(r'^[0-9a-fA-F:]+$', domain) else 0.0
    return 0.0


def check_at_symbol(url):
    return 1.0 if "@" in url else 0.0


def check_url_length(url):
    url_length = len(url)
    return 0.5 if 75 < url_length < 100 else 1.0 if url_length >= 100 else 0.0


def calculate_url_depth(url):
    path_segments = url.split("/")[3:]
    return len(path_segments)


def check_double_slash(url):
    last_double_slash = url.rfind("//")
    return 1.0 if last_double_slash > 7 else 0.0


def check_http_in_domain(url):
    domain_start = url.find('//') + 2
    domain_end = url.find('/', domain_start)
    if domain_end == -1:
        domain_end = None
    domain = url[domain_start:domain_end]
    return 1.0 if "http" in domain else 0.0


def check_https_protocol(url):
    return 0.0 if url.startswith('https://') else 1.0


def check_url_shortening(url):
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co']
    return 1.0 if any(service in url for service in shortening_services) else 0.0


def check_dash_in_domain(url):
    domain_start = url.find('//') + 2
    domain_end = url.find('/', domain_start)
    if domain_end == -1:
        domain_end = None
    domain = url[domain_start:domain_end]
    return 1.0 if '-' in domain else 0.0


def check_sensitive_words(url):
    phishing_terms = ['login', 'update', 'validate', 'activate', 'secure', 'account', 'verification', 'password',
                      'confirm', 'verification', 'signin', 'recover', 'auth', 'identity', 'info', 'alert', 'warning',
                      'urgent']
    return 1.0 if any(term in url.lower() for term in phishing_terms) else 0.0


def check_trendy_brand_name(url):
    trendy_brands = ['apple', 'google', 'microsoft', 'amazon', 'facebook', 'paypal', 'netflix', 'instagram', 'twitter',
                     'linkedin', 'ebay', 'yahoo', 'dropbox', 'adobe', 'linkedin', 'outlook', 'snapchat', 'uber',
                     'whatsapp']
    return 1.0 if any(brand in url.lower() for brand in trendy_brands) else 0.0


def check_uppercase_letters(url):
    return 1.0 if any(char.isupper() for char in url) else 0.0


def count_dots(url):
    return 1.0 if url.count('.') > 2 else 0.0


@app.route('/check_phishing_catboost', methods=['POST'])
def check_phishing():
    try:
        start = time.time()
        data = request.json
        url = data.get('url')

        # Extract the main domain from the URL
        main_domain = extract_main_domain(url)

        # Check if the main domain is in the whitelist
        if main_domain in whitelist_domains:
            prediction = 1
            confidence = 100
            return jsonify({
                'isPhishing': bool(1 - prediction),
                'confidenceLevel': round(confidence, 2)
            })

        # request web page
        resp = requests.get(url)

        # get the response text. in this case it is HTML
        html = resp.text

        # parse the HTML
        soup = BeautifulSoup(html, "html.parser")

        # Save HTML content to a file
        with open("homepage.html", "w", encoding="utf-8") as file:
            file.write(str(soup))

        data = {'website': ['homepage.html'], 'url': [url]}

        df = pd.DataFrame(data)

        df['UF1'] = df['website'].apply(lambda x: count_hyperlinks(x))

        df['UF1'] = df['website'].apply(lambda x: count_hyperlinks(x))
        df['UF2'] = df.apply(
            lambda row: calculate_internal_link_ratio(row['website'], urllib.parse.urlparse(row['url']).netloc), axis=1)

        df['UF3'] = df.apply(
            lambda row: calculate_external_link_ratio(row['website'], urllib.parse.urlparse(row['url']).netloc), axis=1)
        df['UF4'] = df.apply(lambda row: has_external_css(row['website'], urllib.parse.urlparse(row['url']).netloc),
                             axis=1)
        df['UF5'] = df.apply(
            lambda row: is_suspicious_form_action(row['website'], urllib.parse.urlparse(row['url']).netloc),
            axis=1)
        df['UF6'] = df.apply(lambda row: detect_null_links(row['website']), axis=1)
        df['UF7'] = df.apply(lambda row: is_external_favicon(row['website'], urllib.parse.urlparse(row['url']).netloc), axis=1)
        df['UF8'] = df.apply(lambda row: calculate_common_page_detection_ratio(row['website']), axis=1)
        df['UF9'] = df.apply(lambda row: calculate_common_page_in_footer_ratio(row['website']), axis=1)
        df['UF10'] = df.apply(lambda row: check_sfh(row['website'], urllib.parse.urlparse(row['url']).netloc), axis=1)
        df['UF11'] = df.apply(lambda row: has_www(row['url']), axis=1)
        df['UF12'] = df.apply(lambda row: count_subdomains(row['url']), axis=1)
        df['UF13'] = df.apply(lambda row: check_ip_in_domain(row['url']), axis=1)
        df['UF14'] = df.apply(lambda row: check_at_symbol(row['url']), axis=1)
        df['UF15'] = df.apply(lambda row: check_url_length(row['url']), axis=1)
        df['UF16'] = df.apply(lambda row: calculate_url_depth(row['url']), axis=1)
        df['UF17'] = df.apply(lambda row: check_double_slash(row['url']), axis=1)
        df['UF18'] = df.apply(lambda row: check_http_in_domain(row['url']), axis=1)
        df['UF19'] = df.apply(lambda row: check_https_protocol(row['url']), axis=1)
        df['UF20'] = df.apply(lambda row: check_url_shortening(row['url']), axis=1)
        df['UF21'] = df.apply(lambda row: check_dash_in_domain(row['url']), axis=1)
        df['UF22'] = df.apply(lambda row: check_sensitive_words(row['url']), axis=1)
        df['UF23'] = df.apply(lambda row: check_trendy_brand_name(row['url']), axis=1)
        df['UF24'] = df.apply(lambda row: check_uppercase_letters(row['url']), axis=1)
        df['UF25'] = df.apply(lambda row: count_dots(row['url']), axis=1)

        clf = CatBoostClassifier(allow_const_label=True)
        # clf.load_model('pretrained_model_html.bin')
        clf.load_model('pretrained_model_html_.bin')

        df.drop(columns=['website', 'url'], inplace=True)
        # Predict on the test set
        
        # y_pred = clf.predict_proba(df)[0][0]

        # if y_pred >= 0.5:
        #     isPhishing = False
        # else:
        #     isPhishing = True
        # response_data = {
        #     'isPhishing': isPhishing,
        #     'confidenceLevel': round(y_pred * 100, 2)
        # }
        

        prediction = clf.predict(df)
        prediction_proba = clf.predict_proba(df)

        # In kết quả
        result = "Phishing" if prediction[0] == 1 else "Legitimate"
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        print(f"URL: {url}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence * 100:.2f}%")
        response_data = {
            'isPhishing': result,
            'confidenceLevel': round(confidence * 100, 2)
        }
        print(response_data)
        end = time.time()

        print(end - start)
        return jsonify(response_data)
    except InvalidSchema:
        return jsonify({'isPhishing': 1, 'confidenceLevel': 100})




# Load the pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('best_model_.pt').eval()


# Tokenize the URLs
def tokenize_urls(urls):
    return tokenizer.encode_plus(
        urls,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )


@app.route('/check_phishing_bert', methods=['POST'])
def predict_bert():
    data = request.json
    url = data.get('url')

    # Extract the main domain from the URL
    main_domain = extract_main_domain(url)

    print()
    # Check if the main domain is in the whitelist
    if main_domain in whitelist_domains:
        prediction = 1
        confidence = 100
        return jsonify({
            'isPhishing': bool(1 - prediction),
            'confidenceLevel': round(confidence, 2)
        })
    # Tokenize the input URL
    # tokens = tokenize_urls([url])

    encoding = tokenizer.encode_plus(
        url,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move tensors to the same device as the model
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    # Get the logits (predictions)
    logits = outputs.logits

    # Get the predicted class and confidence score
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence_score = torch.softmax(logits, dim=1).max().item()

    # confidence, predicted_label = torch.max(probs, dim=1)

    # Convert to desired output format
    # isPhishing = bool(1 - predicted_label.item())
    confidenceLevel = round(confidence_score * 100, 2)

    if predicted_class >= 0.5:
        isPhishing = True
    else:
        isPhishing = False

    response_data = {
        'isPhishing': isPhishing,
        'confidenceLevel': confidenceLevel
    }
    print(response_data)
    return response_data


if __name__ == '__main__':
    app.run(debug=True)
