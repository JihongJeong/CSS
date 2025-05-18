# Reddit Subreddit-Specific Sentiment and Topic Analysis for Generative AI

This Python script collects posts containing the keywords 'generative ai' or 'chatgpt' from specified Reddit subreddits. It then analyzes user sentiment and identifies key discussion topics for each subreddit. The analysis utilizes VADER for sentiment analysis and LDA (Latent Dirichlet Allocation) for topic modeling.

## Key Features

-   **Targeted Subreddit Data Collection**: Gathers relevant posts from multiple user-specified subreddits.
-   **Keyword-Based Filtering**: Focuses on posts containing 'generative ai' or 'chatgpt'.
-   **Date Range Filtering**: Collects posts only within a specified period.
-   **Text Preprocessing**: Performs lowercasing, noise removal, stop-word removal, and lemmatization on the collected text data.
-   **VADER Sentiment Analysis**: Classifies the sentiment of each post as 'Positive', 'Negative', or 'Neutral'. Visualizes the average sentiment score and sentiment label distribution per subreddit.
-   **LDA Topic Modeling**: Extracts key discussion topics for each subreddit and visualizes the representative words for each topic as word clouds.
-   **API Request Management**: Includes delays between API requests to avoid hitting Reddit API rate limits.

## Prerequisites

1.  **Python**: Python 3.7 or higher must be installed.
2.  **Reddit API Credentials**:
    *   Log in to your Reddit account and create a 'script' type application at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
    *   Prepare your `client_id` (found under "personal use script"), `client_secret` (next to "secret"), and a `user_agent` string (you can define this freely, e.g., `python:myKeywordSubredditScraper:v1.0 (by /u/your_username)`).


## Installation

1.  **Download the Script**: Clone this repository or download the script file directly.
    ```bash
    # git clone <repository_url> # If using a repository
    # cd <repository_name>
    ```

2.  **Install Required Libraries**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

You need to modify the following variables at the top of the script according to your environment and analysis goals:

```python
# --- User Settings ---
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
```

-   `CLIENT_ID`, `CLIENT_SECRET`, `USER_AGENT`: create your own .env file with your Reddit API credentials.
-   `KEYWORDS`: Specify the keywords to search for.
-   `TARGET_SUBREDDITS`: A list of subreddits to analyze.
-   `START_DATE_STR`, `END_DATE_STR`: The period for data collection.
-   `POST_LIMIT_PER_KEYWORD_SUBREDDIT`: The maximum number of posts to collect per subreddit and keyword combination. Adjust this considering API limits and execution time.


## Usage

Once all configurations are set, run the script from your terminal:

```bash
python download_nltk.py 
python reddit_collect.py 
python reddit_analysis.py 
```
each scripts are for:
- downloading nltk resources
- collecting reddit posts
- sentiment analysis and topic modeling with collected posts

## Expected Output

When the script runs, you can expect the following outputs:

1.  **Console Output**:
    *   Progress of data collection and the number of posts collected from each subreddit.
    *   Overall and per-subreddit sentiment analysis results (average compound score, sentiment label distribution).
    *   LDA topic modeling results for each subreddit (top words per topic, Coherence Score).
2.  **Matplotlib Visualizations**:
    *   A bar chart showing the sentiment label distribution per subreddit.
    *   Word cloud images for each topic within each subreddit.
3.  **CSV File (Optional)**:
    *   The DataFrame containing the analysis results will be saved to a CSV file.

## Troubleshooting

-   **PRAW Errors (e.g., `praw.exceptions.ResponseException: received 401 HTTP response`)**: Double-check your `CLIENT_ID`, `CLIENT_SECRET`, and `USER_AGENT` credentials.
-   **NLTK Resource Errors (`LookupError`)**: Re-run the NLTK resource download steps in the 'Prerequisites' section.
-   **API Rate Limit**: Try reducing the `POST_LIMIT_PER_KEYWORD_SUBREDDIT` value or increasing the `time.sleep()` values within the script.
-   **Insufficient Data**: There might not be enough posts matching your criteria (period, keywords, subreddits). Try changing the settings. Topic modeling might not be performed if a subreddit doesn't meet the `MIN_POSTS_FOR_TOPIC_MODELING` threshold.
-   **LDA Model Training Errors**: This can happen if the number of documents is very small, or if all words are filtered out after preprocessing, resulting in an empty corpus. Check `MIN_POSTS_FOR_TOPIC_MODELING` and ensure you have enough data.

## Contributing

Suggestions for improvements or bug fixes are always welcome. Feel free to open an issue or submit a pull request.

