from urllib import request
import re
import nltk
from bs4 import BeautifulSoup
#读取python页面内容
response = request.urlopen('http://python.org/')
html = response.read()
print(len(html))
#分词
tokens = [tok for tok in html.split()]
print("Total no of tokens:"+ str(len(tokens)))
print(tokens[0:100])
#类型转换
tokens = str(tokens)
tokens = re.split(r'\W+',tokens)

print(len(tokens))
print(tokens[0:100])

#clean = nltk.clean_html(html)
#python3中使用如下方式获取文本内容
clean = BeautifulSoup(html,"lxml").get_text()
tokens = [tok for tok in clean.split()]
print(tokens[:100])