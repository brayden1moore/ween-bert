# Ween-Bert
- A BERT language model for masked predictions finetuned on Ween lyrics.<br>
- Compete against it to see if you know more Ween than an the AI on the [web app](http://braydenmoore.pythonanywhere.com/).

![image](https://github.com/brayden1moore/ween-bert/assets/98822684/fe62d32f-15d6-4cd7-99c6-8b5724f51345)


## About
- The language model I used is bert-base-uncased from HuggingFace.co, and I scraped the lyrics from songlyrics.com.<br>
- When I tested it, it was about 40-45% accurate in guessing the hidden word in a verse, compared to 30% for the untrained model.<br>
- Because PythonAnywhere doesn't update global variables in Flask apps, Humans vs AI scores persist for all users, since they are stored in a json file in the website backend (lol).
