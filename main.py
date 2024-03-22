import googleapiclient.discovery
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import os
from dotenv import load_dotenv
from collections import Counter

# Function to extract video ID from URL
def get_video_id(url):
  if "v=" in url:
    return url.split("=")[1]
  else:
    return None

# To create your application's API key:

# Go to the API Console. (https://console.developers.google.com/)
# From the projects list, select a project or create a new one.
# If the APIs & services page isn't already open, open the left side menu and select APIs & services.
# On the left, choose Credentials.
# Click Create credentials and then select API key.

# Enable YouTube Data API v3 : In "Enable APIs & Services" - API that provides access to YouTube data, such as videos, playlists,…
load_dotenv()
DEVELOPER_KEY = os.environ.get('DEVELOPER_KEY')

def start_crapp(video_id):
    
    comments = get_comments(video_id, DEVELOPER_KEY)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(perform_sentiment_analysis(eliminate_stopwords(words_count(comments))))
    with col2:
        generate_wordcloud(eliminate_stopwords(words_count(comments)))



   
def words_count(comments):
     # Find top 3 most frequent words
    
    word_counts = Counter()
    for comment in comments:
            # Split comments into lowercase words
        words = comment.lower().split()
        word_counts.update(words)  # Update counter with each word
    
    return word_counts


def eliminate_stopwords(all_words):

    # nltk.download("stopwords_es")  # Download stopwords if not already available
    # nltk.download('punkt')
    from nltk.corpus import stopwords
    
    stopwords_spanish = set(stopwords.words('spanish'))

    # Get top 10 most frequent words (excluding stopwords)
    filtered_counts = Counter({word: count for word, count in all_words.items() if word not in stopwords_spanish})
    top_10_words = filtered_counts.most_common(20)  # List of tuples (word, count)

    # Create a list of the firts element of each tuple
    first_element_top_10 = []
    for element in top_10_words:
       first_element_top_10.append(element[0])
    
    return first_element_top_10


def get_comments(video_id, developer_key):
    """Retrieves comments from a YouTube video using the YouTube Data API v3.

    Args:
        video_id (str): The ID of the YouTube video.
        developer_key (str): Your YouTube Data API v3 key.

    Returns:
        list: A list of dictionaries containing comment text and other details.
    """
    initial_text = "Creating a service object & Retrieving comments"
    middle_text = "Getting pages (too many comments)"

    
    progress_bar = st.progress(50, text=initial_text)

    # create a service object for interacting with the YouTube Data API v3

    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=developer_key
    )
    
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, textFormat="plainText"
    )
    # retrieve comments for the specified video
    response = request.execute()
    
    progress_bar.progress(80, text=middle_text)
    comments = []
    # list contains dictionaries representing individual comment threads
    for item in response["items"]:
        comment_text = item["snippet"]["topLevelComment"]["snippet"]['textDisplay'] # route inside the response for the actual comment
        # comments.append({"text": comment_text})
        comments.append(comment_text)

    # Check for next page token and retrieve additional comments if needed
        
    next_page_token = response.get("nextPageToken")
    while next_page_token:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, textFormat="plainText", pageToken=next_page_token
        )
        response = request.execute()
        for item in response["items"]:
            comment_text = item["snippet"]["topLevelComment"]["snippet"]['textDisplay']
            # comments.append({"text": comment_text})
            comments.append(comment_text)
        next_page_token = response.get("nextPageToken")


    progress_bar.empty()
    st.success('Data OK --> Go to Results TAB')
    return comments

def perform_sentiment_analysis(comments):
    """Analyzes the sentiment of comments using VADER.

    Args:
        comments (list): A list of dictionaries containing comment text.

    Returns:
        dict: A dictionary containing sentiment scores (positive, negative, neutral, compound)
    """

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0, "compound": 0}

    for comment in comments:

        scores = analyzer.polarity_scores(comment)
        sentiment_scores["positive"]    += scores["pos"]
        sentiment_scores["negative"]    += scores["neg"]
        sentiment_scores["neutral"]     += scores["neu"]
        sentiment_scores["compound"]    += scores["compound"]

    # Calculate average scores
    for key in sentiment_scores:
        sentiment_scores[key] /= len(comments)

    return sentiment_scores

def generate_wordcloud(comments):

    from nltk.corpus import stopwords
    
    # """Creates a word cloud visualization of frequently used words.
    # Args:
    #     comments (list): A list of dictionaries containing comment text.
    # """

    # nltk.download("stopwords_es")  # Download stopwords if not already available
    # nltk.download('punkt')

    stopwords = stopwords.words('spanish')

    # stopwords.add("jajajaja")  # Add additional stopwords as needed

    comment_text = " ".join([comment for comment in comments])
    text = comment_text.lower()  # Convert to lowercase

    # Remove punctuation and non-alphanumeric characters
    text = "".join([char for char in text if char.isalnum() or char.isspace()])

    # Tokenize and filter out stopwords
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords]

    wordcloud = WordCloud(
        width=800, height=600, background_color="white", stopwords=stopwords
    ).generate(" ".join(filtered_tokens))


    st.subheader("Word Cloud")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

# if __name__ == "__main__":
#     comments = get_comments(VIDEO_ID, DEVELOPER_KEY)
#     print(perform_sentiment_analysis(comments))


# Streamlit App
st.header("CrApp Analyzer 1.0")
st.subheader("Developed by: Martin Riveros")

# Input field and description
video_url = st.text_input("Enter YouTube video URL", placeholder="Paste YouTube video URL here")
description = f"""For example:https://www.youtube.com/watch?v=Ds-vROxwmcs"""

st.markdown(description, unsafe_allow_html=True)

# Extract video ID and show thumbnail (if valid URL)
video_id = get_video_id(video_url)
if video_id:
  thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"  # YouTube thumbnail format
  st.image(thumbnail_url, width=200)
else:
  st.warning("Please enter a valid YouTube video URL.")

st.button("Show CrApp", on_click=lambda: start_crapp(video_id))


