from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
#torch.set_num_threads(1)

from flask import Flask, render_template, request
import os

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

modelPath = os.path.abspath(f'{THIS_FOLDER}/weenbert')
lyricsPath = os.path.abspath(f'{THIS_FOLDER}/weenLyrics.txt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(modelPath)
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

with open(lyricsPath,'r') as f:
    text = f.read().split('[split]')

def generate(tokenizer,model):
    lyrIdx = round(random.random() * len(text)-1)
    verse = text[lyrIdx]
    splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace('"',' " ').replace('(',' ( ').replace(')',' ) ').split(' ')
    maskVal = '.'

    while maskVal == ',' or maskVal == '.' or maskVal == '\n' or maskVal == '' or len(maskVal)<4 or len(splitText)<2 or 'We do not have the lyrics' in verse:
        lyrIdx = round(random.random() * len(text)-1)
        verse = text[lyrIdx]
        splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace(' " ','"').replace(' ( ','(').replace(' ) ',')').split(' ')
        maskIdx = round(random.random() * len(splitText)-1)
        maskVal = splitText[maskIdx].replace('"','').replace('.','').replace(',','')

    splitText[maskIdx] = '[MASK]'
    prompt = ' '.join(splitText).replace(' \n ','\n').replace(' , ',',').replace(' . ','.')
    
    encodings = tokenizer(prompt, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    inputIds = encodings['input_ids'].to(device)
    maskIdx = (inputIds == 103).flatten().nonzero().item()
    attentionMask = encodings['attention_mask'].to(device)

    # Finetuned
    model.to(device)
    outputs = model(input_ids=inputIds, attention_mask=attentionMask)
    logits = outputs.logits
    soft = logits.softmax(dim=-1)
    arg = soft.argmax(dim=-1).view(-1)
    guessVal = tokenizer.convert_ids_to_tokens(arg[maskIdx].item())

    return prompt.replace(f'{maskVal}', '[HIDDEN]').replace('MASK','HIDDEN'), guessVal, maskVal

thisPrompt = ''
thisGuess = ''
thisMask = ''
thisUser = ''

prevPrompt = ''
prevGuess = ''
prevMask = ''
prevUser = ''

userScore = 0
bertScore = 0

app = Flask(__name__)
templateDir = os.path.abspath(f'{THIS_FOLDER}/templates')
app.template_folder = templateDir


@app.route('/')
def home():
    global thisPrompt, thisGuess, thisMask, thisUser, prevPrompt, prevGuess, prevMask, prevUser, userScore, bertScore
    thisPrompt = ''
    thisGuess = ''
    thisMask = ''
    thisUser = ''

    prevPrompt = ''
    prevGuess = ''
    prevMask = ''
    prevUser = ''

    userScore = 0
    bertScore = 0

    return render_template('weenLand.html')


@app.route('/game', methods=['POST'])
def game(): 
    global thisPrompt, thisGuess, thisMask, thisUser, prevPrompt, prevGuess, prevMask, prevUser, userScore, bertScore

    thisPrompt, thisGuess, thisMask = generate(tokenizer,model)
    
    return render_template('weenGame.html', prompt=thisPrompt, guessVal=prevGuess, maskVal=prevMask, user=prevUser, userScore=userScore, bertScore=bertScore)


@app.route('/play', methods=['POST'])
def play():
    global prevPrompt, prevGuess, prevMask, prevUser, thisPrompt, thisGuess, thisMask, thisUser, userScore, bertScore
    
    prevPrompt = thisPrompt.replace('[HIDDEN]',thisMask.upper())
    prevGuess = thisGuess
    prevMask = thisMask
    prevUser = thisUser

    thisUser = request.form['user']

    thisPrompt, thisGuess, thisMask = generate(tokenizer,model) 

    userScore += (thisUser.lower().strip() == prevMask.lower().strip())
    bertScore += (prevGuess.lower().strip() == prevMask.lower().strip())

    return render_template('weenGame.html', prompt=thisPrompt,guessVal=prevGuess,maskVal=prevMask,user=thisUser,userScore=userScore,bertScore=bertScore,prevPrompt=prevPrompt)


if __name__ == '__main__':
    app.run()