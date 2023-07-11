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
    
    with open(f'{THIS_FOLDER}/totalScores.json','r') as f:
        totalScores = dict(json.load(f))

    aiScore = totalScores['aiScore']
    humanScore = totalScores['humanScore']
    
    return render_template('weenLand.html', humanScore=humanScore, aiScore=aiScore)


@app.route('/play', methods=['POST','GET'])
def play(): 

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','r') as f:
        lastGeneration = dict(json.load(f))

    userScore = lastGeneration['userScore']
    bertScore = lastGeneration['bertScore']
    scoreState = ('tied' if userScore==bertScore else 'winning' if userScore>bertScore else 'losing')

    prompt, mask, guess, answer = recall()

    currentGeneration = {'prompt':prompt,
                         'mask':mask,
                         'guess':guess,
                         'answer':answer,
                         'userScore':userScore,
                         'bertScore':bertScore}

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','w') as f:
        json.dump(currentGeneration, f)
    
    with open(f'{THIS_FOLDER}/totalScores.json','r') as f:
        totalScores = dict(json.load(f))

    aiScore = totalScores['aiScore']
    humanScore = totalScores['humanScore']
    
    return render_template('weenGame.html', prompt=prompt, userScore=userScore, bertScore=bertScore, humanScore=humanScore, scoreState=scoreState, aiScore=aiScore)


@app.route('/result', methods=['POST'])
def result():

    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','r') as f:
        currentGeneration = dict(json.load(f))
    
    with open(f'{THIS_FOLDER}/totalScores.json','r') as f:
        totalScores = dict(json.load(f))

    aiScore = totalScores['aiScore']
    humanScore = totalScores['humanScore']

    prompt = currentGeneration['prompt']
    mask = currentGeneration['mask'].replace('(','').replace(')','').replace("'","").replace('?','').replace(',','').replace('!','')
    guess = currentGeneration['guess']
    answer = currentGeneration['answer']
    userScore = currentGeneration['userScore']
    bertScore = currentGeneration['bertScore']

    user = request.form['user'].replace('(','').replace(')','').replace("'","").replace('?','').replace(',','').replace('!','')

    userCorrect = (user.lower().strip() in mask.lower().strip())
    bertCorrect = (guess.lower().strip() in mask.lower().strip())

    userScore += userCorrect
    bertScore += bertCorrect
    scoreState = ('tied' if userScore==bertScore else 'winning' if userScore>bertScore else 'losing')

    humanScore += userCorrect
    aiScore += bertCorrect

    currentGeneration = {'prompt':prompt,
                         'mask':mask,
                         'guess':guess,
                         'answer':answer,
                         'userScore':userScore,
                         'bertScore':bertScore}
    
    with open(f'{THIS_FOLDER}/cache/currentGeneration{session["sessionId"]}.json','w') as f:
        json.dump(currentGeneration, f)

    totalScores = {'aiScore':aiScore,
                   'humanScore':humanScore}

    with open(f'{THIS_FOLDER}/totalScores.json','w') as f:
        json.dump(totalScores, f)
        
    return render_template('weenResult.html', prompt=answer, guessVal=guess, maskVal=mask, user=user, userScore=userScore, bertScore=bertScore, scoreState=scoreState, humanScore=humanScore, aiScore=aiScore)


if __name__ == '__main__':
    app.run()