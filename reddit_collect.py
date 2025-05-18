import os, time, json
from tqdm.auto import tqdm
from dotenv import load_dotenv
import praw
import datetime

load_dotenv()
CLIENT_ID = os.getenv("REDDIT_CLIENT_ID") 
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT")
KEYWORDS = ["generative ai", "chatgpt"] 
SUBREDDITS = ["Artificial",
              "MachineLearning",
              "DeepLearning",
              "GenerativeAI",
              "OpenAI",
              "StableDiffusion",
              "LanguageTechnology",
              "ComputerVision",
              "DeepFakes",
              "MLQuestions",
              "Design",
              "photography",
              "Entrepreneur",
              "marketing",
              "education",
              "gamedev",
              "architecture",
              "writing",
              "Finance",
              "healthcare"
              ] 
START_DATE_STR = "2022-01-01"
END_DATE_STR = "2025-04-30"
POST_LIMIT_PER_SUBREDDIT = 1000

start_datetime_utc = datetime.datetime.strptime(START_DATE_STR + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
end_datetime_utc = datetime.datetime.strptime(END_DATE_STR + " 23:59:59", "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
start_timestamp_utc = start_datetime_utc.timestamp()
end_timestamp_utc = end_datetime_utc.timestamp()

def save_to_json(collected_posts):
    filename_json = f"reddit_posts_{START_DATE_STR}_{END_DATE_STR}.json"
    try:
        with open(filename_json, 'w', encoding='utf-8') as jsonfile:
            json.dump(collected_posts, jsonfile, ensure_ascii=False, indent=4)
        print(f"\n결과를 {filename_json} 파일로 저장했습니다.")
    except IOError as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")
    except Exception as e:
        print(f"JSON 파일 저장 중 예기치 않은 오류 발생: {e}")

def get_reddit_post():
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
    )

    # search_query = " OR ".join([f'"{keyword}"' for keyword in KEYWORDS])
    search_query = " OR ".join([f'(title:"{keyword}" OR selftext:"{keyword}")' for keyword in KEYWORDS])

    print(f"다음 조건으로 Reddit 게시물을 검색합니다:")
    print(f"  키워드: {search_query}")
    print(f"  서브레딧: {', '.join(SUBREDDITS)}")
    print(f"  기간: {START_DATE_STR} ~ {END_DATE_STR}")
    print("-" * 30)

    collected_posts = []
    for subreddit_name in SUBREDDITS:
        print(f"\n'{subreddit_name}' 서브레딧에서 검색 중...")
        try:
            subreddit = reddit.subreddit(subreddit_name)
            submissions = subreddit.search(query=search_query, sort='new', time_filter='all', limit=POST_LIMIT_PER_SUBREDDIT) 

            count_in_subreddit = 0
            for submission in tqdm(enumerate(submissions)):
                if start_timestamp_utc <= submission.created_utc <= end_timestamp_utc:
                    post_data = {
                        'id': submission.id,
                        'title': submission.title,
                        'score': submission.score,
                        'author': str(submission.author),
                        'created_utc': datetime.datetime.fromtimestamp(submission.created_utc, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'url': f"https://www.reddit.com{submission.permalink}",
                        'num_comments': submission.num_comments,
                        'subreddit': str(submission.subreddit),
                        'selftext': submission.selftext if submission.is_self else None
                    }
                    collected_posts.append(post_data)
                    count_in_subreddit += 1
                    if count_in_subreddit >= POST_LIMIT_PER_SUBREDDIT:
                        break 
                time.sleep(0.1)

            print(f"  '{subreddit_name}'에서 {count_in_subreddit}개의 게시물을 찾았습니다 (지정 기간 내).")

        except praw.exceptions.PRAWException as e:
            print(f"'{subreddit_name}' 처리 중 오류 발생: {e}")
        except Exception as e:
            print(f"'{subreddit_name}' 처리 중 예기치 않은 오류 발생: {e}")
        
        time.sleep(1)


    print("-" * 30)
    print(f"총 {len(collected_posts)}개의 게시물을 수집했습니다.")
    
    if collected_posts:
#     print("\n수집된 게시물 샘플 (최대 5개):")
#     for i, post in enumerate(collected_posts[:5]):
#         print(f"\n--- 게시물 {i+1} ---")
#         print(f"  제목: {post['title']}")
#         print(f"  작성자: {post['author']}")
#         print(f"  작성일: {post['created_utc']}")
#         print(f"  점수: {post['score']}")
#         print(f"  댓글 수: {post['num_comments']}")
#         print(f"  URL: {post['url']}")
#         print(f"  서브레딧: {post['subreddit']}")
#         # print(f"  본문: {post['selftext'][:200] if post['selftext'] else 'N/A'}...") 
        save_to_json(collected_posts)

if __name__=="__main__":
    get_reddit_post()