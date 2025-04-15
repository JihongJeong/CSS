import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import numpy as np

# NLTK 필요 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_reddit_data(filename):
    """JSON 파일에서 Reddit 데이터 로드"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def normalize_score(scores):
    """Reddit score를 0~1 범위로 정규화"""
    if len(scores) == 0:
        return scores
    
    # 최소값이 음수인 경우를 대비하여 모든 값을 양수로 이동
    min_score = min(scores)
    if min_score < 0:
        scores = [score - min_score for score in scores]
    
    # 최대값으로 나누어 정규화 (0~1 범위)
    max_score = max(scores)
    if max_score == 0:  # 모든 점수가 0인 경우
        return [0.5] * len(scores)  # 중간값 0.5 반환
    
    return [score / max_score for score in scores]

def preprocess_text(text):
    """텍스트 전처리 함수"""
    if not isinstance(text, str):
        return ""
    
    # 소문자 변환 및 특수문자 제거
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # 토큰화
    tokens = word_tokenize(text)
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # 표제어 추출
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def get_sentiment_scores(text):
    """텍스트의 감성 점수 계산"""
    if not text:
        return {'positive': 0, 'negative': 0, 'neutral': 0}
    
    analysis = TextBlob(text)
    
    # 극성 점수를 기반으로 감성 점수 계산
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return {'positive': polarity, 'negative': 0, 'neutral': 0}
    elif polarity < 0:
        return {'positive': 0, 'negative': abs(polarity), 'neutral': 0}
    else:
        return {'positive': 0, 'negative': 0, 'neutral': 1}

def analyze_sentiment(df):
    """데이터프레임의 감성 분석 수행"""
    # 날짜 형식 변환
    df['date'] = pd.to_datetime(df['created_utc']).dt.date
    
    # 필요한 컬럼만 선택
    df = df[['date', 'score', 'num_comments', 'selftext']]
    
    # score 정규화
    df['normalized_score'] = normalize_score(df['score'].tolist())
    
    # 텍스트 전처리
    df['processed_text'] = df['selftext'].apply(preprocess_text)
    
    # 감성 분석 수행
    sentiments = df['processed_text'].apply(get_sentiment_scores)
    
    # 감성 점수를 개별 컬럼으로 분리
    df['positive'] = sentiments.apply(lambda x: x['positive'])
    df['negative'] = sentiments.apply(lambda x: x['negative'])
    df['neutral'] = sentiments.apply(lambda x: x['neutral'])
    
    # 정규화된 score를 가중치로 사용하여 감성 점수 조정
    df['positive'] = df['positive'] * df['normalized_score']
    df['negative'] = df['negative'] * df['normalized_score']
    df['neutral'] = df['neutral'] * df['normalized_score']
    
    # 날짜별 감성 점수 합계 계산
    daily_sentiment = df.groupby('date').agg({
        'positive': 'sum',
        'negative': 'sum',
        'neutral': 'sum',
        'normalized_score': 'mean'  # 일별 평균 정규화 점수도 저장
    }).reset_index()
    
    return daily_sentiment

def plot_sentiment_trends(daily_sentiment):
    """감성 분석 결과 시각화"""
    plt.figure(figsize=(12, 6))
    
    # 선 그래프 그리기
    plt.plot(daily_sentiment['date'], daily_sentiment['positive'], 
             label='Positive', color='green', marker='o')
    plt.plot(daily_sentiment['date'], daily_sentiment['negative'], 
             label='Negative', color='red', marker='o')
    plt.plot(daily_sentiment['date'], daily_sentiment['neutral'], 
             label='Neutral', color='blue', marker='o')
    
    plt.title('Daily Sentiment Trends in Reddit Posts')
    plt.xlabel('Date')
    plt.ylabel('Weighted Sentiment Score')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 그래프 저장
    plt.tight_layout()
    plt.savefig('sentiment_trends.png')
    plt.close()

def main():
    # JSON 파일 로드
    json_filename = 'reddit_python_20250415.json'  # 실제 파일명으로 수정 필요
    df = load_reddit_data(json_filename)
    
    # 감성 분석 수행
    daily_sentiment = analyze_sentiment(df)
    
    # 결과 시각화
    plot_sentiment_trends(daily_sentiment)
    
    # 결과를 CSV로 저장
    daily_sentiment.to_csv('daily_sentiment_scores.csv', index=False)
    print("분석이 완료되었습니다. 결과는 'sentiment_trends.png'와 'daily_sentiment_scores.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main() 