import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
import numpy as np

# 分词
text = "Sentiment analysis is a challenging subject in machine learning.\
 People express their emotions in language that is often obscured by sarcasm,\
  ambiguity, and plays on words, all of which could be very misleading for \
  both humans and computers.".lower()
text_list = nltk.word_tokenize(text)
# 去掉标点符号
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
text_list = [word for word in text_list if word not in english_punctuations]
# 去掉停用词
stops = set(stopwords.words("english"))
text_list = [word for word in text_list if word not in stops]
print(nltk.pos_tag(text_list))
brown_taged= nltk.corpus.brown.tagged_words()

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
#默认标注
tags = [tag for (word,tag) in brown.tagged_words(categories='news')]
print(nltk.FreqDist(tags).max())


raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.tag(tokens))
print(default_tagger.evaluate(brown_tagged_sents))

#正则表达式标注器
patterns= [(r'.*ing$','VBG'),(r'.*ed$','VBD'),(r'.*es$','VBZ'),(r'.*ould$','MD'),\
           (r'.*\'s$','NN$'),(r'.*s$','NNS'),(r'^-?[0-9]+(.[0-9]+)?$','CD'),(r'.*','NN')]
regexp_tagger = nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
print(regexp_tagger.evaluate(brown_tagged_sents))

#查询标注器：找出100个最频繁的词，存储它们最有可能的标记。然后可以使用这个信息作为
#"查询标注器"（NLTK UnigramTagger）的模型
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = list(fd.keys())[:100]
likely_tags = dict((word,cfd[word].max()) for word in most_freq_words)
# baseline_tagger = nltk.UnigramTagger(model=likely_tags)
#许多词都被分配了None标签，因为它们不在100个最频繁的词中，可以使用backoff参数设置这些词的默认词性
baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
print(baseline_tagger.evaluate(brown_tagged_sents))


unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
print(unigram_tagger.tag(brown_sents[2007]))
unigram_tagger.evaluate((brown_tagged_sents))