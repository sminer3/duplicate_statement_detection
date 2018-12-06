import os
os.chdir("data")
import requests
import json
import pandas as pd
from tqdm import tqdm

train = pd.read_csv("train.csv")

with open('statement_splits.json') as f:
	splits = json.load(f)
statements = list(set(pd.concat((train.statement1, train.statement2))))

test = [s for s in statements if s not in splits.keys()]
print(len(test), 'statements to get splits for')
count_splits = 0
for j, stmt in tqdm(enumerate(test)):
	url = 'https://us-central1-groupsolver-research.cloudfunctions.net/sentenceSplitting/split-sentence'
	req = { "sentence": stmt, "languange":"en"}
	r = requests.post(url, data=req)
	if r.status_code == 200:
		result = json.loads(r.text)['sentenceSplit']
		split_sentences = []
		for i in range(len(result)):
			split_sentences.extend(result[i]['splitStatements'])
		if len(split_sentences) > 1:
			count_splits = count_splits + 1
		splits[stmt] = {'sentences': split_sentences, 'num_splits': len(split_sentences)}
	if j % 100 == 0:
		with open('statement_splits.json', 'w') as f:
			json.dump(splits, f)

with open('statement_splits.json', 'w') as f:	
	json.dump(splits, f)


# test = ["Money, time and space","COST AND SPACE","The package was easy to understand, modern and simple in design.","The gold packaging, name, minimalist design, and longwear are appealing to me.","It is clean, simple and to the point.","interest in the product.  Look glamoruus","There is nothing different, my grocery shopping is the same as a couple of years ago","I think it's very cool and kids will love having there lunch or snack in them."]
# print(len(test), 'statements to get splits for')
# count_splits = 0
# splits_test = {}
# for j, stmt in tqdm(enumerate(test)):
# 	url = 'https://us-central1-groupsolver-research.cloudfunctions.net/sentenceSplitting/split-sentence'
# 	req = { "sentence": stmt, "languange":"en"}
# 	r = requests.post(url, data=req)
# 	if r.status_code == 200:
# 		result = json.loads(r.text)['sentenceSplit']
# 		split_sentences = []
# 		for i in range(len(result)):
# 			split_sentences.extend(result[i]['splitStatements'])
# 		if len(split_sentences) > 1:
# 			count_splits = count_splits + 1
# 		splits_test[stmt] = {'sentences': split_sentences, 'num_splits': len(split_sentences)}
