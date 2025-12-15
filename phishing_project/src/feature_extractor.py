# src/feature_extractor.py
import re
import socket
import ssl
import math
import requests
import tldextract
import whois
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from datetime import datetime
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- helper functions ---
def contains_ip(host):
    try:
        # IPv4 pattern
        return bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', host))
    except:
        return False

def count_subdomains(url):
    ext = tldextract.extract(url)
    sub = ext.subdomain
    if sub == '':
        return 0
    return len(sub.split('.'))

def count_suspicious_tokens(url):
    # 增加了一些常见的钓鱼词汇
    tokens = ['login', 'signin', 'secure', 'update', 'verify', 'account', 'confirm', 
              'banking', 'alert', 'client', 'service', 'pay', 'free', 'bonus']
    return sum(1 for t in tokens if t in url.lower())

def has_at_symbol(url):
    return int('@' in url)

def count_hyphen(url):
    # 只计算 hostname 里的连字符
    parsed = urlparse(url)
    return parsed.netloc.count('-')

def count_digits(url):
    return sum(c.isdigit() for c in url)

def entropy(s):
    if not s:
        return 0.0
    probs = [float(s.count(c)) / len(s) for c in set(s)]
    return - sum(p * math.log(p, 2) for p in probs)

# WHOIS / domain age
def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if cd is None:
            return -1
        if isinstance(cd, str):
            try:
                cd = datetime.strptime(cd.split()[0], "%Y-%m-%d")
            except:
                return -1
        delta = datetime.now() - cd
        return delta.days
    except Exception:
        return -1

# SSL info via socket
def get_ssl_days_left(hostname, port=443, timeout=3):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                notAfter = cert.get('notAfter')
                if notAfter:
                    try:
                        exp = datetime.strptime(notAfter, '%b %d %H:%M:%S %Y %Z')
                        return (exp - datetime.now()).days
                    except:
                        return -999
    except Exception:
        return -999
    return -999

# HTML content analysis
def safe_fetch(url, timeout=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
try:
        # 配置重试策略：总共重试3次，遇到连接错误或500/502等错误时重试
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        resp = session.get(url, headers=headers, timeout=timeout, verify=False)
        if resp is not None and len(resp.content) > 3_000_000:
            return None
        return resp
    except Exception:
        return None

def analyze_html(url):
    res = safe_fetch(url)
    if res is None:
        return {
            'has_login_form': 0,
            'num_inputs': 0,
            'num_iframes': 0,
            'num_scripts': 0,
            'meta_refresh': 0,
            'suspicious_js': 0,
            'ext_favicon': 0
        }
    try:
        soup = BeautifulSoup(res.text, 'lxml')
    except Exception:
        soup = BeautifulSoup(res.text, 'html.parser')

    forms = soup.find_all('form')
    has_login = 0
    num_input = 0
    for f in forms:
        inputs = f.find_all('input')
        num_input += len(inputs)
        if any(i.get('type') and i.get('type').lower() == 'password' for i in inputs):
            has_login = 1

    iframes = soup.find_all('iframe')
    scripts = soup.find_all('script')
    meta = soup.find_all('meta', attrs={'http-equiv': 'refresh'})
    meta_refresh = 1 if meta else 0

    suspicious_js = 0
    for s in scripts:
        text = ""
        try:
            text = s.string or ""
        except:
            text = ""
        if 'document.oncontextmenu' in text or 'eval(' in text:
            suspicious_js = 1
            break
            
    # 检查 favicon 是否引用了其他域名
    ext_favicon = 0
    links = soup.find_all('link', rel=lambda x: x and 'icon' in x.lower())
    domain_info = tldextract.extract(url)
    base_domain = f"{domain_info.domain}.{domain_info.suffix}"
    
    for l in links:
        href = l.get('href', '')
        if href.startswith('http') and base_domain not in href:
            ext_favicon = 1
            break

    return {
        'has_login_form': has_login,
        'num_inputs': num_input,
        'num_iframes': len(iframes),
        'num_scripts': len(scripts),
        'meta_refresh': meta_refresh,
        'suspicious_js': suspicious_js,
        'ext_favicon': ext_favicon
    }

# main feature extraction
def extract_features(url):
    f = {}
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''
    
    domain_info = tldextract.extract(url)
    domain = '.'.join(part for part in [domain_info.domain, domain_info.suffix] if part)

    # 拆分长度特征
    f['hostname_length'] = len(hostname)
    f['path_length'] = len(path) + len(query)
    f['double_slash_in_path'] = path.count('//')
    
    f['has_ip'] = int(contains_ip(hostname))
    f['subdomain_cnt'] = count_subdomains(url)
    f['suspicious_tokens'] = count_suspicious_tokens(url)
    f['has_at'] = has_at_symbol(url)
    f['hyphen_count'] = count_hyphen(url)
    f['digit_count'] = count_digits(url)
    f['entropy'] = entropy(url)
    f['is_https'] = 1 if parsed.scheme == 'https' else 0

    f['domain_age_days'] = get_domain_age_days(domain)
    f['ssl_days_left'] = get_ssl_days_left(hostname) if hostname else -999
    
    html_feats = analyze_html(url)
    f.update(html_feats)
    return f

# 更新后的特征列表
FEATURE_ORDER = [
    'hostname_length', 'path_length', 'double_slash_in_path', 'is_https',
    'has_ip', 'subdomain_cnt', 'suspicious_tokens', 'has_at',
    'hyphen_count', 'digit_count', 'entropy', 'domain_age_days', 'ssl_days_left',
    'has_login_form', 'num_inputs', 'num_iframes', 'num_scripts', 'meta_refresh', 
    'suspicious_js', 'ext_favicon'
]

def features_to_vector(feat_dict):
    return [feat_dict.get(k, 0) for k in FEATURE_ORDER]

if __name__ == "__main__":
    sample = "https://example.com/login"
    print(extract_features(sample))