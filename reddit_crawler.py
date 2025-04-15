import praw
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import json

# .env 파일에서 환경 변수 로드
load_dotenv()

def initialize_reddit():
    """Reddit API 클라이언트 초기화"""
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )

def crawl_subreddit(subreddit_name, limit=10, sort='hot'):
    """특정 서브레딧의 게시물을 크롤링"""
    reddit = initialize_reddit()
    subreddit = reddit.subreddit(subreddit_name)
    
    posts = []
    
    # sort 방식에 따라 게시물 가져오기
    if sort == 'hot':
        submissions = subreddit.hot(limit=limit)
    elif sort == 'new':
        submissions = subreddit.new(limit=limit)
    elif sort == 'top':
        submissions = subreddit.top(limit=limit)
    
    for submission in submissions:
        post = {
            'title': submission.title,
            'score': submission.score,
            'id': submission.id,
            'url': submission.url,
            'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
            'author': str(submission.author),
            'num_comments': submission.num_comments,
            'permalink': f'https://reddit.com{submission.permalink}',
            'selftext': submission.selftext
        }
        posts.append(post)
    
    return posts

def save_to_json(data, filename):
    """데이터를 JSON 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_to_csv(df, filename):
    """데이터프레임을 CSV 파일로 저장"""
    df.to_csv(filename, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    # 크롤링할 서브레딧 이름
    SUBREDDIT_NAME = 'python'  # 예시로 'python' 서브레딧을 크롤링
    
    # 게시물 크롤링
    posts = crawl_subreddit(
        subreddit_name=SUBREDDIT_NAME,
        limit=100,  # 크롤링할 게시물 수
        sort='hot'  # 'hot', 'new', 'top' 중 선택
    )
    
    # 결과를 JSON 파일로 저장
    json_filename = f'reddit_{SUBREDDIT_NAME}_{datetime.now().strftime("%Y%m%d")}.json'
    save_to_json(posts, json_filename)
    
    # CSV로도 저장하고 싶은 경우 아래 코드 사용
    # posts_df = pd.DataFrame(posts)
    # save_to_csv(posts_df, f'reddit_{SUBREDDIT_NAME}_{datetime.now().strftime("%Y%m%d")}.csv') 