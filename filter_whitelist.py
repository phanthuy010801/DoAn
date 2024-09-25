import pandas as pd

from urllib.parse import urlparse

data = pd.read_csv('new_data_urls.csv')
# Filter URLs with status = 1 to create a whitelist
whitelist_urls = data[data['status'] == 1]


def extract_domain(url):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    if len(domain_parts) > 2:
        return '.'.join(domain_parts[-2:])
    return parsed_url.netloc


# Apply the function to extract the domain and create a new column
whitelist_urls['domain'] = whitelist_urls['url'].apply(extract_domain)

# Drop duplicate domains to get a unique list of domains
unique_domains = whitelist_urls[['domain']].drop_duplicates()

unique_domains.to_csv('Unique_Domains_Whitelist.csv', index=False)
