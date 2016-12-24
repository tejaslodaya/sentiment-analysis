import pytreebank
import re
import gensim
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import csv
import pickle

dataset = pytreebank.import_tree_corpus('trees/train.txt')

examples = []
for i in range(len(dataset)):
	examples.append(dataset[i])

full_data=[]
sent=[]

label_vect=[]

for example in examples:
	for label, sentence in example.to_labeled_lines():
		
	   full_data.append((sentence,label))

	   if(0<=label<=1):
	   	label=0
	   elif label==2:
	   	label=1
	   else:
	   	label=2

	   label_vect.append(label)

	   sentence = re.sub(r'[?|$|.|!|-|,|:|\']',r'',sentence)
	   sent.append((wordnet_lemmatizer.lemmatize(sentence)).split())


print(len(sent))
model = gensim.models.Word2Vec(sent, size=32, window=5, min_count=5, workers=4)

sent_vec=[]
for i in range(len(sent)):
	row=[]
	for j in range(len(sent[i])):
		if sent[i][j] in model:
			row.append(model[sent[i][j]])
	sent_vec.append(row)

y=open("sent.pkl","wb")
pickle.dump(sent_vec,y)
y.close()

z=open("labels.pkl","wb")
pickle.dump(label_vect,z)
z.close()