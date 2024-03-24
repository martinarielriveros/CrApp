import googleapiclient.discovery
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import os
from dotenv import load_dotenv
from collections import Counter
import re
from unidecode import unidecode

import streamlit as st

# Set the width of the screen
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="CrApp",
    page_icon=":poop:")

# Your Streamlit app code goes here
# Streamlit App

st.header("CrApp 1.0, by Martin Riveros")

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

col1, col2, col3 = st.columns(3)


def start_crapp(video_id, results_number, additional_stopwords):
    
    comments = get_comments(video_id, DEVELOPER_KEY)

    with col2:

        word_count_result, reformat_comments = words_count(comments)
        top_words_without_stopwords = eliminate_stopwords(word_count_result, results_number, additional_stopwords)
        st.header('Sentiment Analysis')
        st.dataframe(data=perform_sentiment_analysis(top_words_without_stopwords, reformat_comments))
        
        st.write(f'Total comments analized: {len(comments)}')
        st.write(f'Total words analized: {len(word_count_result)}')


    with col3:
        generate_wordcloud(top_words_without_stopwords)
   
def words_count(comments):

    
    word_counts = Counter()
    reformat_comments = []
    
    for comment in comments:
        # Convert accented characters to ASCII and lowercase
        comment = unidecode(comment).lower()
        
        # Remove time-like words (e.g., "1:23")
        comment = re.sub(r'\b\d+[:.,;]+\d+\b', '', comment)
        
        # Remove symbols and non-alphanumeric characters
        comment = re.sub(r'[^\w\s]', '', comment)

        # we need to set the new comments because in the later match, word_counts may be different (Eg: "senora" vs "senora" with "enie")
        reformat_comments.append(comment)
        
        # Split the comment into words and remove empty words
        words = [word for word in comment.split() if word]

        word_counts.update(words)  # Update counter with each word
    
    return word_counts, reformat_comments


def eliminate_stopwords(all_words, number_to_select, custom_selected_stopwords=[]):

    # nltk.download("stopwords_es")  # Download stopwords if not already available
    # nltk.download('punkt')
    from nltk.corpus import stopwords
    
    stopwords_spanish = list(stopwords.words('spanish'))
    stopwords_spanish.extend(custom_selected_stopwords)  # Add additional stopwords as needed
    
    # stopwords_english = set(stopwords.words('english'))

    # Get most frequent words (excluding stopwords)
    filtered_counts = Counter({word: count for word, count in all_words.items() if word not in stopwords_spanish})

    top_words_tuple = filtered_counts.most_common(number_to_select)  # List of tuples (word, count)

    # Create a list of the firts element of each tuple. The word we are lookging for is formed: ('the word we are loogin for', numer of appearences)
    first_element_top_words = []
    for element in top_words_tuple:
       first_element_top_words.append(element[0])
    
    return first_element_top_words


def get_comments(video_id, developer_key):
    """
    Retrieves comments from a YouTube video using the YouTube Data API v3.

    Args:
        video_id (str): The ID of the YouTube video.
        developer_key (str): Your YouTube Data API v3 key.

    Returns:
        list: A list of dictionaries containing comment text and other details.
    """
    initial_text = "Creating a service object & Retrieving comments"
    middle_text = "Getting pages (too many comments)"

    with col1:
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
        st.success('Data OK')
    return comments

def perform_sentiment_analysis(top_words_without_stopwords, comments):
    """Analyzes the sentiment of comments using VADER.

    Args:
        top_words_without_stopwords (list): top N words based on repetition, cleaned from stopwords done by library and user
        comments (list): A list of dictionaries containing comment text.

    Returns:
        dict: A dictionary containing sentiment scores (positive, negative, neutral, compound)
    """
    analyzer = SentimentIntensityAnalyzer()
    all_scores = []

    for word in top_words_without_stopwords:
        comments_matched_for_the_word = 0
        sentiment_scores = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for comment in comments:
            if word in comment.split():
                comments_matched_for_the_word += 1

                scores = analyzer.polarity_scores(comment)
                sentiment_scores["Positive"] += scores["pos"]
                sentiment_scores["Negative"] += scores["neg"]
                sentiment_scores["Neutral"] += scores["neu"]

        # Calculate average scores
        for key in sentiment_scores:
            # sentiment_scores[key] /= round(comments_matched_for_the_word,2)
            sentiment_scores[key] = round(sentiment_scores[key]/comments_matched_for_the_word*100,2)

        sentiment_scores["Comments"] = comments_matched_for_the_word
        sentiment_scores["Word"] = word

        all_scores.append(sentiment_scores)

    return all_scores


def generate_wordcloud(comments):

    
    """Creates a word cloud visualization of frequently used words.
    Args:
        comments (list): A list of dictionaries containing comment text.
    """
    from nltk.corpus import stopwords

    # nltk.download("stopwords_es")  # Download stopwords if not already available
    # nltk.download('punkt')

    stopwords = stopwords.words('spanish')

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


    st.header("Word Cloud")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)
    
with col1:

    st.header('Selection Zone')

    # Input fields and description
    video_url = st.text_input("Enter YouTube video URL", placeholder="Paste YouTube video URL here")
    description = f"""Example: https://www.youtube.com/watch?v=Ds-vROxwmcs"""
    st.caption(description, unsafe_allow_html=True)
    
    results_number = st.selectbox("Select a number between 5 and 20", range(4, 21), index=0, format_func=lambda x: 'Select a number' if x == 4 else x)

    # Enable or disable the button based on whether both inputs are provided
    button_disabled = (not video_url) or (not results_number)
    
    cola, colb = st.columns(2)
    with cola:
        # Extract video ID and show thumbnail (if valid URL)
        video_id = get_video_id(video_url)
        if video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"  # YouTube thumbnail format
            st.image(thumbnail_url, width=200)
        else:
            st.warning("Please enter a valid YouTube video URL.")
    with colb:
        # Show the button with dynamic enable/disable based on inputs
        if button_disabled:
            st.button("Show CrApp", disabled=True)
        else:
            word_count_result, reformat_comments = words_count(get_comments(video_id, DEVELOPER_KEY))
            top_words_without_stopwords = eliminate_stopwords(word_count_result, results_number)
            additional_stopwords = st.multiselect('Filter words', word_count_result)
            st.button("Show CrApp", on_click=lambda: start_crapp(video_id, results_number, additional_stopwords))


