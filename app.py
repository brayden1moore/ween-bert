from flask import Flask, render_template, request, session
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

app = Flask(__name__)
app.secret_key = 'pi-33pp-co-sk-32'
templateDir = os.path.abspath(f'{THIS_FOLDER}/templates')
app.template_folder = templateDir


@app.route('/', methods=['GET'])
def home():
    session['sessionId'] = random.random()*100
    currentGeneration = {'prompt':'',
                         'mask':'',
                         'guess':'',
                         'answer':'',
                         'userScore':0,
                         'bertScore':0}

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','w') as f:
        json.dump(currentGeneration, f)
    
    return render_template('weenLand.html')


@app.route('/play', methods=['POST','GET'])
def play(): 

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','r') as f:
        lastGeneration = dict(json.load(f))

    userScore = lastGeneration['userScore']
    bertScore = lastGeneration['bertScore']

    prompt, mask, guess, answer = recall()

    currentGeneration = {'prompt':prompt,
                         'mask':mask,
                         'guess':guess,
                         'answer':answer,
                         'userScore':userScore,
                         'bertScore':bertScore}

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','w') as f:
        json.dump(currentGeneration, f)
    
    return render_template('weenGame.html', prompt=prompt, userScore=userScore, bertScore=bertScore)


@app.route('/result', methods=['POST'])
def result():

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','r') as f:
        currentGeneration = dict(json.load(f))

    prompt = currentGeneration['prompt']
    mask = currentGeneration['mask']
    guess = currentGeneration['guess']
    answer = currentGeneration['answer']
    userScore = currentGeneration['userScore']
    bertScore = currentGeneration['bertScore']

    user = request.form['user']

    userScore += (user.lower().strip() == mask.lower().strip())
    bertScore += (guess.lower().strip() == mask.lower().strip())

    currentGeneration = {'prompt':prompt,
                         'mask':mask,
                         'guess':guess,
                         'answer':answer,
                         'userScore':userScore,
                         'bertScore':bertScore}
    
    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','w') as f:
        json.dump(currentGeneration, f)
        
    return render_template('weenResult.html', prompt=answer, guessVal=guess, maskVal=mask, user=user, userScore=userScore, bertScore=bertScore)


if __name__ == '__main__':
    app.run()