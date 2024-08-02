import pandas as pd
import aiohttp
import asyncio
import logging
from aiohttp import ClientSession
from tqdm.asyncio import tqdm
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import streamlit as st

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Add file handler for logging
file_handler = logging.FileHandler('redirect_mapper.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
logger.addHandler(file_handler)

async def fetch(session, url, semaphore):
    async with semaphore:
        try:
            async with session.get(url) as response:
                status = response.status
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    # Extract relevant elements
                    h1 = ' '.join([tag.get_text() for tag in soup.find_all('h1')])
                    title = soup.title.get_text() if soup.title else ''
                    return {'url': url, 'status': status, 'h1': h1, 'title': title}
                else:
                    return {'url': url, 'status': status}
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {'url': url, 'status': None}

async def fetch_all(urls, max_simultaneous_requests, requests_per_second):
    semaphore = asyncio.Semaphore(max_simultaneous_requests)
    async with ClientSession() as session:
        tasks = [fetch(session, url, semaphore) for url in urls]
        responses = []
        for i, f in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks))):
            responses.append(await f)
            if (i + 1) % max_simultaneous_requests == 0:
                await asyncio.sleep(1 / requests_per_second)  # Control request rate
        return responses

def preprocess_text(text):
    danish_stopwords = stopwords.words('danish')
    text = text.lower()
    text = re.sub(r'[^a-zæøå\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in danish_stopwords]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def calculate_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

def get_slug(url):
    parsed_url = urlparse(url)
    return parsed_url.path.split('/')[-1]

def get_domain(url):
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

async def main(from_file, to_file):
    tips_from_df = pd.read_csv(from_file, header=None, names=['url']).drop_duplicates()
    tips_tourls_df = pd.read_csv(to_file, header=None, names=['url']).drop_duplicates()

    old_urls = tips_from_df['url'].tolist()
    new_urls = tips_tourls_df['url'].tolist()

    max_simultaneous_requests = 5
    requests_per_second = 5

    logger.info("Starting to fetch old URLs")
    old_responses = await fetch_all(old_urls, max_simultaneous_requests, requests_per_second)
    logger.info("Finished fetching old URLs")

    logger.info("Starting to fetch new URLs")
    new_responses = await fetch_all(new_urls, max_simultaneous_requests, requests_per_second)
    logger.info("Finished fetching new URLs")

    homepage = get_domain(new_urls[0])
    results = []

    for old_resp in tqdm(old_responses, desc="Processing old URLs"):
        old_url, old_status, old_h1, old_title = old_resp['url'], old_resp['status'], old_resp.get('h1', ''), old_resp.get('title', '')
        
        if old_status == 200:
            best_match = None
            best_score = 0

            for new_resp in new_responses:
                new_url, new_status, new_h1, new_title = new_resp['url'], new_resp['status'], new_resp.get('h1', ''), new_resp.get('title', '')
                
                if new_status == 200:
                    # Check slug match
                    if get_slug(old_url) == get_slug(new_url):
                        best_match = new_url
                        best_score = 1.0
                        break
                    
                    # Check heading and title similarity
                    score_h1 = calculate_similarity(old_h1, new_h1)
                    score_title = calculate_similarity(old_title, new_title)
                    score = max(score_h1, score_title)
                    
                    if score > best_score:
                        best_score = score
                        best_match = new_url
            
            if best_match:
                results.append((old_url, old_status, best_match, 200, best_score))
            else:
                results.append((old_url, old_status, homepage, 200, 0))
        else:
            old_slug = get_slug(old_url)
            match_found = False

            for new_resp in new_responses:
                new_url, new_status = new_resp['url'], new_resp['status']
                if new_status == 200 and old_slug == get_slug(new_url):
                    results.append((old_url, old_status, new_url, new_status, 0))
                    match_found = True
                    break

            if not match_found:
                results.append((old_url, old_status, homepage, 200, 0))

    # Remove duplicates, keeping only the best match for each old URL
    results_df = pd.DataFrame(results, columns=['old_url', 'old_status', 'new_url', 'new_status', 'similarity_score'])
    results_df = results_df.sort_values(by='similarity_score', ascending=False).drop_duplicates(subset=['old_url'])
    results_df.to_csv('output.csv', index=False)
    logger.info("Similarity matching completed successfully. Results saved to output.csv")

    return results_df

# Streamlit UI
st.title('URL Redirect Mapper')

from_file = st.file_uploader("Upload the 'from' CSV file", type="csv")
to_file = st.file_uploader("Upload the 'to' CSV file", type="csv")

if from_file and to_file:
    st.write("Processing...")
    results_df = asyncio.run(main(from_file, to_file))
    st.write("Processing complete. Results:")
    st.write(results_df)
    st.download_button(label="Download Results", data=results_df.to_csv(index=False).encode('utf-8'), file_name='output.csv', mime='text/csv')
