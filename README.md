# Reddit Sentiment Analysis

This project crawls Reddit posts using the Reddit API and performs sentiment analysis on the collected data. It includes both data collection and sentiment analysis components.

## Features

### Data Collection
- Crawls posts from specified subreddits using PRAW (Python Reddit API Wrapper)
- Collects post information including:
  - Title
  - Score (upvotes - downvotes)
  - Post ID
  - URL
  - Creation timestamp
  - Author
  - Number of comments
  - Post content (selftext)
- Saves data in both JSON and CSV formats

### Sentiment Analysis
- Text Preprocessing:
  - Case normalization
  - Special character removal
  - Tokenization
  - Stop word removal
  - Lemmatization
- Sentiment Scoring:
  - Analyzes text sentiment using TextBlob
  - Calculates positive, negative, and neutral scores
  - Normalizes Reddit scores (0-1 range) for weighted sentiment calculation
  - Aggregates sentiment scores by date
- Visualization:
  - Generates time series plots of sentiment trends
  - Shows daily positive, negative, and neutral sentiment scores
  - Includes grid lines for better readability

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up Reddit API credentials:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create Application" or "Create Another App"
   - Select "script" for application type
   - Fill in the name and description
   - Set redirect uri to `http://localhost:8080`
   - Note down the generated client_id and client_secret

3. Configure environment variables:
   - Create a `.env` file in the project root
   - Add the following variables:
```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT="python:my_reddit_crawler:v1.0 (by /u/your_username)"
```

## Usage

### Data Collection
```bash
python reddit_crawler.py
```
This will:
- Crawl posts from the specified subreddit
- Save the results in JSON format
- Default configuration crawls 100 'hot' posts from r/python

### Sentiment Analysis
```bash
python sentiment_analysis.py
```
This will:
- Load the crawled Reddit data
- Perform text preprocessing and sentiment analysis
- Generate visualization of sentiment trends
- Save results in:
  - `sentiment_trends.png`: Time series plot of sentiment scores
  - `daily_sentiment_scores.csv`: Daily aggregated sentiment data

## Technical Details

### Score Normalization
- Reddit scores (upvotes - downvotes) are normalized to 0-1 range
- Normalization process:
  1. Handles negative scores by shifting all scores to positive range
  2. Divides by maximum score to get 0-1 range
  3. Uses 0.5 as default when all scores are 0
- Normalized scores are used as weights for sentiment scores

### Sentiment Calculation
- TextBlob's polarity score determines sentiment direction
- Sentiment categories:
  - Positive: polarity > 0
  - Negative: polarity < 0
  - Neutral: polarity = 0
- Final scores are weighted by normalized Reddit scores

## Output Files

1. Data Collection:
   - `reddit_[subreddit]_[date].json`: Raw crawled data
   - `reddit_[subreddit]_[date].csv`: Optional CSV format data

2. Sentiment Analysis:
   - `sentiment_trends.png`: Visualization of sentiment trends
   - `daily_sentiment_scores.csv`: Daily sentiment scores with columns:
     - date: Analysis date
     - positive: Aggregated positive sentiment score
     - negative: Aggregated negative sentiment score
     - neutral: Aggregated neutral sentiment score
     - normalized_score: Average normalized Reddit score for the day 