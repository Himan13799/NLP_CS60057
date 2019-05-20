import collections

f = open('train.txt','r')
train = f.read()
train = train.split('\n')
f.close()
d = collections.defaultdict(dict)
tag_count=0
for line in train:
	try:
		word, tag, chunk_tag = line.split(' ')
		if tag not in d:
			d[tag] = {}
		if chunk_tag in d[tag]:
			d[tag][chunk_tag] += 1
		else:
			d[tag][chunk_tag] = 1
	except:
		continue

f = open('test.txt','r')
test = f.read()
test = test.split('\n')
f.close()

with open('output.txt','w') as out:
	for line in test:
		try:
			word, tag, chunk_tag = line.split(' ')
			predicted_chunk_tag = max(d[tag].items(), key=lambda k: k[1])
			print(predicted_chunk_tag)
			out.write(word + ' ' + tag + ' ' + chunk_tag + ' ' + predicted_chunk_tag[0])
			out.write('\n')
		except:
			continue

out.close()