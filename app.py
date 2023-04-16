from flask import Flask, render_template, request

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']

        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(review)

        sentiment_score = round((scores['compound'] + 1) / 2 * 5, 1)
        sentiment_emoji = get_emoji(sentiment_score)

        return render_template('index.html', review=review, sentiment_score=sentiment_score,
                               sentiment_emoji=sentiment_emoji)


def get_emoji(score):
    if score <= 1:
        return '😢'
    elif score <= 2:
        return '😔'
    elif score <= 3:
        return '😐'
    elif score <= 4:
        return '😊'
    else:
        return '😃'


if __name__ == '__main__':
    app.run(debug=True)
