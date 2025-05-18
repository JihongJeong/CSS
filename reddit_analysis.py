import praw
import datetime
import os
import re
import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

FILE_NAME = "reddit_posts_2022-01-01_2025-04-30"

def get_reddit_data(file_path): 
    """
    지정된 경로의 JSON 파일에서 Reddit 게시물 데이터를 불러옵니다.
    JSON 파일은 게시물 딕셔너리의 리스트 형태여야 합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"오류: JSON 파일 '{file_path}'의 최상위 요소가 리스트가 아닙니다. 게시물 딕셔셔너리의 리스트 형태여야 합니다.")
            return pd.DataFrame()

        if not data:
            print(f"경고: JSON 파일 '{file_path}'이 비어있거나 게시물 데이터가 없습니다.")
            return pd.DataFrame()

        print(f"'{file_path}'에서 {len(data)}개의 게시물을 성공적으로 불러왔습니다.")
        
        df = pd.DataFrame(data)

        if 'created_utc' in df.columns and pd.api.types.is_numeric_dtype(df['created_utc']):
            try:
                df['created_utc_str'] = df['created_utc'].apply(
                    lambda ts: datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    if pd.notnull(ts) else None
                )
            except Exception as e:
                print(f"경고: 'created_utc'를 datetime 문자열로 변환 중 오류 발생: {e}. 원본 값을 사용합니다.")
                df['created_utc_str'] = df['created_utc'].astype(str)
        elif 'created_utc' in df.columns:
            df['created_utc_str'] = df['created_utc'].astype(str)
            print("정보: 'created_utc' 필드가 이미 존재하며 숫자형이 아닙니다. 문자열로 사용합니다.")
        else:
            print("경고: 'created_utc' 필드가 DataFrame에 없습니다.")

        return df

    except FileNotFoundError:
        print(f"오류: JSON 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return pd.DataFrame() 
    except json.JSONDecodeError:
        print(f"오류: JSON 파일 '{file_path}'의 형식이 잘못되었습니다. 유효한 JSON 파일인지 확인해주세요.")
        return pd.DataFrame()
    except Exception as e:
        print(f"데이터 로드 중 예기치 않은 오류 발생: {e}")
        return pd.DataFrame()


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text) 

    tokens = word_tokenize(text)

    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            processed_tokens.append(lemmatizer.lemmatize(token))

    return " ".join(processed_tokens) 


analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if not text or pd.isna(text): 
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'label': 'Neutral'}
    
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    if compound_score >= 0.05:
        label = 'Positive'
    elif compound_score <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    
    return {'compound': compound_score, 'pos': vs['pos'], 'neu': vs['neu'], 'neg': vs['neg'], 'label': label}


def perform_lda_topic_modeling(texts_for_lda, num_topics=8, passes=15, random_state=42):
    """
    texts_for_lda: 전처리 및 토큰화된 문서 리스트 ( [[token1, token2], [token3, token4], ...] )
    num_topics: 생성할 토픽의 수 (논문에서는 8개 사용)
    """
    if not texts_for_lda or all(not t for t in texts_for_lda):
        print("LDA를 위한 텍스트 데이터가 비어있습니다.")
        return None, None, None

    dictionary = corpora.Dictionary(texts_for_lda)
    corpus = [dictionary.doc2bow(text) for text in texts_for_lda]

    if not corpus or all(not doc for doc in corpus):
        print("LDA를 위한 코퍼스가 비어있습니다. (모든 문서가 비어있거나 단어가 없음)")
        return None, None, None
        
    print(f"LDA 모델 학습 중... (토픽 수: {num_topics})")
    lda_model = models.LdaMulticore(corpus=corpus, 
                                    id2word=dictionary, 
                                    num_topics=num_topics,
                                    random_state=random_state, 
                                    passes=passes, workers=3
                                    ) 

    print("\nLDA 토픽 결과 (각 토픽별 상위 10개 단어):")
    topics = lda_model.print_topics(num_topics=num_topics, num_words=10)
    for topic_num, topic_words in topics:
        print(f"Topic {topic_num+1}: {topic_words}")

    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts_for_lda, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f'\nCoherence Score (c_v): {coherence_lda:.4f}')
    
    return lda_model, dictionary, corpus

def assign_topic_to_documents(lda_model, corpus, df):
    """각 문서에 주요 토픽을 할당"""
    doc_topics = []
    for i, row in enumerate(corpus):
        if not row: 
            doc_topics.append(None) 
            continue
        topic_probs = lda_model.get_document_topics(row, minimum_probability=0.0)
        dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0][0] if topic_probs else None
        doc_topics.append(dominant_topic)
    df['dominant_topic'] = doc_topics
    return df


if __name__ == "__main__":
    # 1. 데이터 로드
    print("\n--- 1. Data Loading ---")
    collected_posts_df = get_reddit_data(file_path= f"{FILE_NAME}.json")
    collected_posts_df['full_text'] = (collected_posts_df['title'].fillna('') + " " + collected_posts_df['selftext'].fillna('')).str.strip()

    # 2. 데이터 전처리
    print("\n--- 2. Text Preprocessing ---")
    collected_posts_df['processed_text_for_vader'] = collected_posts_df['full_text'].apply(preprocess_text)
    collected_posts_df['tokens_for_lda'] = collected_posts_df['processed_text_for_vader'].apply(lambda x: word_tokenize(x) if pd.notna(x) else [])

    # 3. VADER 감정 분석
    print("\n--- 3. VADER sentiment analysis ---")
    sentiment_results = collected_posts_df['processed_text_for_vader'].apply(get_vader_sentiment)
    sentiment_df = pd.json_normalize(sentiment_results) 
    collected_posts_df = pd.concat([collected_posts_df, sentiment_df], axis=1)

    # print(collected_posts_df[['id', 'title', 'compound', 'label']].head())
    print(collected_posts_df['label'].value_counts(normalize=True) * 100)

    print("\n--- VADER 감성 분석 결과 시계열 시각화 ---")
    if not collected_posts_df.empty and 'created_utc' in collected_posts_df.columns:
        collected_posts_df['datetime_utc'] = pd.to_datetime(collected_posts_df['created_utc'])
        collected_posts_df['date'] = collected_posts_df['datetime_utc'].dt.date

        collected_posts_df['month_year'] = collected_posts_df['datetime_utc'].dt.to_period('M')
        
        if len(collected_posts_df['month_year'].unique()) > 1:
            # 월별 감성 레이블 평균
            monthly_avg_compound = collected_posts_df.groupby('month_year')['compound'].mean()
            
            plt.figure(figsize=(12, 6))
            monthly_avg_compound.plot(kind='line', marker='o', linestyle='-')
            plt.title('Monthly Average Compound Sentiment Score Over Time')
            plt.xlabel('Month-Year')
            plt.ylabel('Average Compound Score')
            plt.grid(True, axis= 'y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            if not os.path.isdir(FILE_NAME):
                os.mkdir(FILE_NAME)
            filename_compound = os.path.join(FILE_NAME, "monthly_avg_compound_score.png")
            plt.savefig(filename_compound, dpi=300, bbox_inches='tight')
            print(f"그래프 저장 완료: {filename_compound}")
            plt.show()

            # 월별 감성 레이블 비율 변화 (선 그래프)
            monthly_label_proportions = collected_posts_df.groupby('month_year')['label'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
            
            for col in ['Positive', 'Neutral', 'Negative']:
                if col not in monthly_label_proportions.columns:
                    monthly_label_proportions[col] = 0
            
            plt.figure(figsize=(14, 7))
            monthly_label_proportions[['Positive', 'Neutral', 'Negative']].plot(kind='line', marker='o', ax=plt.gca()) 
            plt.title('Monthly Sentiment Label Proportions Over Time')
            plt.xlabel('Month-Year')
            plt.ylabel('Proportion of Posts (%)')
            plt.legend(title='Sentiment Label')
            plt.grid(True, axis= 'y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename_compound = os.path.join(FILE_NAME, "monthly_sentiment_proportions_line.png")
            plt.savefig(filename_compound, dpi=300, bbox_inches='tight')
            print(f"그래프 저장 완료: {filename_compound}")
            plt.show()

            # 월별 감성 레이블 비율 변화 (누적 영역 그래프)
            plt.figure(figsize=(14, 7))
            monthly_label_proportions[['Positive', 'Neutral', 'Negative']].plot(kind='bar', stacked=True, alpha=0.7, ax=plt.gca())
            plt.title('Monthly Sentiment Label Proportions Over Time (Stacked Bar)')
            plt.xlabel('Year-Month')
            plt.ylabel('Proportion of Posts (%)')
            plt.legend(title='Sentiment Label')
            plt.grid(False)
            plt.xticks(rotation=45)
            plt.tight_layout()
            filename_compound = os.path.join(FILE_NAME, "monthly_sentiment_proportions_bar.png")
            plt.savefig(filename_compound, dpi=300, bbox_inches='tight')
            print(f"그래프 저장 완료: {filename_compound}")
            plt.show()
        else:
            print("시계열 시각화를 위한 충분한 월별 데이터가 없습니다 (최소 2개월 필요).")
    else:
        print("시계열 시각화를 위한 날짜 정보('created_utc')가 없거나 데이터가 비어있습니다.")

    # 4. LDA topic modeling
    # print("\n--- 4. LDA topic modeling ---")
    # texts_for_lda_input = collected_posts_df['tokens_for_lda'].tolist()
    # texts_for_lda_input_filtered = [text_list for text_list in texts_for_lda_input if text_list]

    # if texts_for_lda_input_filtered:
    #     NUM_TOPICS_FROM_PAPER = 8 
    #     num_topics_to_run = min(NUM_TOPICS_FROM_PAPER, len(texts_for_lda_input_filtered)) if texts_for_lda_input_filtered else 1
    #     lda_model, dictionary, corpus_lda = perform_lda_topic_modeling(texts_for_lda_input_filtered, num_topics=num_topics_to_run)

    #     if lda_model:

    #         print("\nWord Cloud 생성 중...")
    #         cols = 2
    #         rows = (num_topics_to_run + cols -1) // cols
    #         plt.figure(figsize=(15, rows * 7))
            
    #         for topic_id in range(lda_model.num_topics):
    #             topic_terms = lda_model.show_topic(topic_id, topn=50) 
    #             wordcloud_dict = {term: weight for term, weight in topic_terms}
                
    #             if not wordcloud_dict: 
    #                 print(f"Topic {topic_id+1} has no words for wordcloud.")
    #                 continue

    #             wc = WordCloud(background_color='white', width=800, height=400, max_words=50).generate_from_frequencies(wordcloud_dict)
                
    #             plt.subplot(rows, cols, topic_id + 1)
    #             plt.imshow(wc, interpolation='bilinear')
    #             plt.title(f'Topic {topic_id + 1}')
    #             plt.axis('off')
    #         plt.tight_layout()
    #         plt.show()

    #         # (선택) 논문의 Figure 5와 같이 토픽별 감성 분포 분석
    #         # 이를 위해서는 각 문서에 토픽을 할당하고, 해당 토픽별로 감성 레이블의 분포를 계산해야 합니다.
    #         # collected_posts_df = assign_topic_to_documents(lda_model, corpus_lda, collected_posts_df) # 이 부분은 필터링된 corpus_lda와 원본 df 매칭 필요
    #         # print("\n(구현 필요) 토픽별 감성 분포:")
    #         # topic_sentiment_distribution = collected_posts_df.groupby('dominant_topic')['label'].value_counts(normalize=True).unstack(fill_value=0)
    #         # print(topic_sentiment_distribution)
    #         # topic_sentiment_distribution.plot(kind='bar', stacked=True, figsize=(12,7))
    #         # plt.title('Sentiment Distribution per Topic')
    #         # plt.ylabel('Proportion')
    #         # plt.show()
    # else:
    #     print("LDA 토픽 모델링을 위한 충분한 데이터가 없습니다.")

    # 최종 DataFrame 저장 (선택 사항)
    collected_posts_df.to_csv(f"{FILE_NAME}/{FILE_NAME}_analysis_results.csv", index=False, encoding='utf-8-sig')
