from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
torch.set_num_threads(1)

from flask import Flask, render_template, request
import os

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

modelPath = os.path.abspath(f'{THIS_FOLDER}/weenbert-2')
lyricsPath = os.path.abspath(f'{THIS_FOLDER}/weenLyrics.txt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(modelPath)
#device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
device = torch.device('cpu')

with open(lyricsPath,'r') as f:
    text = f.read().split('[split]')


def generate():
    lyrIdx = round(random.random() * len(text)-1)
    verse = text[lyrIdx]
    splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace('"',' " ').replace('(',' ( ').replace(')',' ) ').split(' ')[:30]
    maskVal = '.'

    while maskVal == ',' or maskVal == '.' or maskVal == '\n' or maskVal == '' or len(maskVal)<4 or len(splitText)<2 or 'We do not have the lyrics' in verse:
        lyrIdx = round(random.random() * len(text)-1)
        verse = text[lyrIdx]
        splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace(' " ','"').replace(' ( ','(').replace(' ) ',')').split(' ')[:30]
        maskIdx = round(random.random() * len(splitText)-1)
        maskVal = splitText[maskIdx].replace('"','').replace('.','').replace(',','')

    splitText[maskIdx] = '[MASK]'
    prompt = ' '.join(splitText).replace(' \n ','\n').replace(' , ',',').replace(' . ','.')
    return prompt, maskVal

 
def guess(prompt,maskVal):
    encodings = tokenizer(prompt, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    inputIds = encodings['input_ids'].to(device)

    try:
        maskIdx = (inputIds == 103).flatten().nonzero().item()
        attentionMask = encodings['attention_mask'].to(device)

        # Finetuned
        model.to(device)
        outputs = model(input_ids=inputIds, attention_mask=attentionMask)
        logits = outputs.logits
        soft = logits.softmax(dim=-1)
        arg = soft.argmax(dim=-1).view(-1)
        guessVal = tokenizer.convert_ids_to_tokens(arg[maskIdx].item())
    except:
        guessVal = 'I dunno man...'

    unmaskedPrompt = prompt.replace('[MASK]',maskVal.upper()).replace(maskVal,maskVal.upper())
    return guessVal, unmaskedPrompt


thisPrompt = ''
thisGuess = ''
thisMask = ''
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
    global nextPrompt, nextGuess, nextMask, nextAnswer, userScore, bertScore
    
    nextPrompt, nextMask = generate() 
    nextGuess, nextAnswer = guess(nextPrompt,nextMask)

    userScore = 0
    bertScore = 0
    
    return render_template('weenLand.html')


@app.route('/play', methods=['POST','GET'])
def play(): 
    global nextPrompt, nextGuess, nextMask, nextAnswer, thisPrompt, thisGuess, thisMask, thisAnswer, thisUser, userScore, bertScore

    thisPrompt = nextPrompt
    thisGuess = nextGuess
    thisMask = nextMask
    thisAnswer = nextAnswer

    nextPrompt, nextMask = generate() 
    nextGuess, nextAnswer = guess(nextPrompt,nextMask)
    
    return render_template('weenGame.html', prompt=thisPrompt.replace('MASK','HIDDEN').replace(f'{thisMask}','[HIDDEN]'), userScore=userScore, bertScore=bertScore)


@app.route('/result', methods=['POST'])
def result():
    global thisPrompt, thisGuess, thisMask, thisAnswer, thisUser, userScore, bertScore
    
    thisUser = request.form['user']

    userScore += (thisUser.lower().strip() == thisMask.lower().strip())
    bertScore += (thisGuess.lower().strip() == thisMask.lower().strip())

    return render_template('weenResult.html', prompt=thisAnswer, guessVal=thisGuess, maskVal=thisMask, user=thisUser, userScore=userScore, bertScore=bertScore)


if __name__ == '__main__':
    app.run()