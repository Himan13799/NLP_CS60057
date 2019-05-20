import sys
import json 
from xpinyin import Pinyin
from pprint import pprint

p=Pinyin()
with open('test.json') as f:
	d = str(f.read())
	
data = json.loads(d)
en=[]

for i in data:
	id_ = i["id"]
	eng = p.get_pinyin(i[" sentence"], tone_marks='numbers')	
	eng = eng.encode('utf-8')
	eng.split(' ')
	eng=eng.replace("-","")
	eng=eng.split()
	en=en+eng

DICTIONARY = "/home/himanshu/Submission_Ass_1/vocab.xml";

NodeCount = 0
WordCount = 0

class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children: 
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

trie = TrieNode()
for word in open(DICTIONARY, "rt").read().split():
    WordCount += 1
    trie.insert( word )

def count(str1,str2):
	set1=set(str1)
	set2=set(str2)
	match=set1 & set2
	return(len(match))

def search( word, maxCost ):

    currentRow = range( len(word) + 1 )

    results = []

    for letter in trie.children:
        searchRecursive( trie.children[letter], letter, word, currentRow, 
            results, maxCost )

    return results

def searchRecursive( node, letter, word, previousRow, results, maxCost ):

    columns = len( word ) + 1
    currentRow = [ previousRow[0] + 1 ]

    for column in xrange( 1, columns ):

        insertCost = currentRow[column - 1] + 1
        deleteCost = previousRow[column] + 1

        if word[column - 1] != letter:
            replaceCost = previousRow[ column - 1 ] + 1
        else:                
            replaceCost = previousRow[ column - 1 ]

        currentRow.append( min( insertCost, deleteCost, replaceCost ) )

   
    if currentRow[-1] <= maxCost and node.word != None:
        results.append( (node.word, currentRow[-1] ) )

    
    if min( currentRow ) <= maxCost:
        for letter in node.children:
            searchRecursive( node.children[letter], letter, word, currentRow, 
                results, maxCost )

f = open('Corrections.txt','w')
          
for word in en:
	if(search(word,0)): 
		continue
	for i in range(1,(len(word)+1)):
		results = search(word,i)
		if results:
			f.write("Wrong word: "+word+"\t\t")
			f.write("Correct word: ")
			for r in results:
				n=0
				if (count(r[0],word)>n):
					correct=r[0]
			f.write(correct+"\t\t")
			f.write("Edit Distance: "+str(r[1]))
			f.write("\n\n")
			print r
			break
			
f.close()
