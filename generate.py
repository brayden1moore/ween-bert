from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import json

modelPath = 'weenbert'
lyricsPath = 'weenLyrics.txt'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained(modelPath)
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

with open(lyricsPath,'r') as f:
    text = f.read().split('[split]')


def generate():
    lyrIdx = round(random.random() * len(text)-1)
    verse = text[lyrIdx]
    splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace('"',' " ').replace('(',' ( ').replace(')',' ) ').split(' ')[:60]
    maskVal = '.'

    while maskVal == ',' or maskVal == '.' or maskVal == '\n' or maskVal == '' or len(maskVal)<4 or len(splitText)<2 or 'We do not have the lyrics' in verse:
        lyrIdx = round(random.random() * len(text)-1)
        verse = text[lyrIdx]
        splitText = verse.replace('\n',' \n ').replace(',',' , ').replace('.',' . ').replace(' " ','"').replace(' ( ','(').replace(' ) ',')').split(' ')[:60]
        maskIdx = round(random.random() * len(splitText)-1)
        maskVal = splitText[maskIdx].replace('"','').replace('.','').replace(',','')

    splitText[maskIdx] = '[MASK]'
    prompt = ' '.join(splitText).replace(' \n ','\n').replace(' , ',',').replace(' . ','.')
    return prompt, maskVal

 
def guess(prompt,maskVal):
    encodings = tokenizer(prompt, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
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

from tqdm import tqdm

generationDict = {}
for i in tqdm(range(50000)):
    prompt, maskVal = generate()
    guessVal, unmaskedPrompt = guess(prompt,maskVal)

    generationDict[i] = {'prompt':prompt,
                         'maskVal':maskVal,
                         'guessVal':guessVal,
                         'unmaskedPrompt':unmaskedPrompt}
    
with open('generationDict.json','w') as f:
    json.dump(generationDict, f)