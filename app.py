import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


def image():
    st.markdown(
        f"""
         <style>
         .stApp {{
             # background: url("https://images.firstpost.com/wp-content/uploads/2020/06/boat-airdopes-1280.jpg");
             background: url("https://img.republicworld.com/republic-prod/stories/images/1611402057600c0b49271ca.png");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


image()

tf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))
# model = load_model("lstm_model.h5")

model = pickle.load(open('lr_model.pkl', 'rb'))

ps = PorterStemmer()


def text_processing(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    # text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text


st.title('Review Classification')

# preprocessing
title = st.text_input("Enter the review title")
review = st.text_area('Enter the review')

review = title + ' ' + review

if st.button('Predict'):

    transformed_review = text_processing(review)

    # vectorize
    vector_ip = tf.transform([transformed_review])

    # predict
    res = model.predict(vector_ip)[0]

    if res > 0.5:
        st.success('The Review is positive')
    else:
        st.error('The review is negative')
