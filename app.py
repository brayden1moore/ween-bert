from flask import Flask, render_template, request
import os
import random
import json
from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

jsonPath = os.path.abspath(f'{THIS_FOLDER}/generationDict.json')

with open(jsonPath,'r') as f:
    generationDict = dict(json.load(f))


def recall():
    global usedIndices
    lyrIdx = str(round(random.random() * len(generationDict)-1))

    while lyrIdx in usedIndices:
        lyrIdx = str(round(random.random() * len(generationDict)-1))

    maskVal = generationDict[lyrIdx]['maskVal']
    prompt = generationDict[lyrIdx]['prompt'].replace(f' {maskVal}',' [HIDDEN] ').replace('MASK','HIDDEN')
    guessVal = generationDict[lyrIdx]['guessVal']
    unmaskedPrompt = generationDict[lyrIdx]['unmaskedPrompt']
    usedIndices.append(lyrIdx)

    return prompt, maskVal, guessVal, unmaskedPrompt


usedIndices = []

thisPrompt = ''
thisGuess = ''
thisMask = ''
thisAnswer = ''
thisUser = ''

nextPrompt = ''
nextGuess = ''
nextMask = ''
nextAnswer = ''

userScore = 0
bertScore = 0

app = Flask(__name__)
templateDir = os.path.abspath(f'{THIS_FOLDER}/templates')
app.template_folder = templateDir


@app.route('/', methods=['GET'])
def home():
    global thisPrompt, thisGuess, thisMask, thisAnswer, thisUser, nextPrompt, nextGuess, nextMask, nextAnswer, userScore, bertScore
    
    thisPrompt = ''
    thisGuess = ''
    thisMask = ''
    thisAnswer = ''
    thisUser = ''

    nextPrompt = ''
    nextGuess = ''
    nextMask = ''
    nextAnswer = ''

    userScore = 0
    bertScore = 0

    nextPrompt, nextMask, nextGuess, nextAnswer = recall()
    
    return render_template('weenLand.html')


@app.route('/play', methods=['POST','GET'])
def play(): 
    global nextPrompt, nextGuess, nextMask, nextAnswer, thisPrompt, thisGuess, thisMask, thisAnswer, thisUser, userScore, bertScore

    thisPrompt = nextPrompt
    thisGuess = nextGuess
    thisMask = nextMask
    thisAnswer = nextAnswer

    nextPrompt, nextMask, nextGuess, nextAnswer = recall()
    
    return render_template('weenGame.html', prompt=thisPrompt, userScore=userScore, bertScore=bertScore)


@app.route('/result', methods=['POST'])
def result():
    global thisPrompt, thisGuess, thisMask, thisAnswer, thisUser, userScore, bertScore
    
    thisUser = request.form['user']

    userScore += (thisUser.lower().strip() == thisMask.lower().strip())
    bertScore += (thisGuess.lower().strip() == thisMask.lower().strip())

    return render_template('weenResult.html', prompt=thisAnswer, guessVal=thisGuess, maskVal=thisMask, user=thisUser, userScore=userScore, bertScore=bertScore)


if __name__ == '__main__':
    app.run()