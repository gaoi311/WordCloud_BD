import os
import re
import numpy as np
import collections  # 词频统计库
import jieba  # 结巴分词
import wordcloud  # 词云展示库
from PIL import Image  # 图像处理库
import matplotlib.pyplot as plt  # 图像展示库

from pyspark import SparkContext, SparkConf

os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['SPARK_HOME'] = '/usr/local/spark'
os.environ['HADOOP_HOME'] = '/usr/local/hadoop'

# Create SparkConf
sparkConf = SparkConf().setAppName('Python Spark WordCount').setMaster('local[2]')
# Create SparkContext
sc = SparkContext(conf=sparkConf)
sc.setLogLevel('WARN')

# 读取HDFS文件
textfile = sc.textFile("/user/gaoi/input/text.txt")


# 产生二维列表
def printCH(line):
    words = line
    pattern = re.compile(u'，|。|、|“|”|！|（|）|\t|\n|\.|-|:|;|\)|\(|\?|"')  # 定义正则表达式匹配模式
    words = re.sub(pattern, '', words)  # 将符合模式的字符去除
    jieba.suggest_freq("社会主义", True)  # 主动添加“社会主义”词汇
    jieba.del_word("会主")  # 去掉“会主”
    jieba.del_word("社会")  # 去掉“社会”
    jieba.del_word("主义")  # 去掉“主义”  因为会被jieba判别为常用词汇，所以需要主动去除
    seg_list = jieba.cut(words, cut_all=False)  # 精准统计模式
    print("/".join(seg_list))


textfile.foreach(printCH)


def splitCH(line):
    seg_list = jieba.cut(line, cut_all=True)
    return "/".join(seg_list)


# 转换成列表
words = textfile.map(splitCH).collect()

remove_words = [u'的', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'中', u'在', u'了', u'通常', u'如果', u'我们', u'需要',
                '']  # 自定义去除词库，用于最后剔除单个无用词，若放在上述正则表达式中，容易将包含这些字的词汇剔除

word = []  # 生成最终一维列表
word_list = []  # 临时存放二维中的每一项

for word_one_level in words:
    word_list.extend(word_one_level.split('/'))  # 将通过 / 连接的词汇串用 / 分割

for word_two_level in word_list:
    if word_two_level not in remove_words:  # 使用自定义去除词库去除无用词
        word.append(word_two_level)

# 词频统计
wordcount = collections.Counter(word)
print(wordcount)

# 词频展示
mask = np.array(Image.open('mask.jpg'))  # 定义词频背景
wc = wordcloud.WordCloud(
    font_path='simhei.ttf',  # 设置字体格式
    mask=mask,  # 设置背景图
    max_words=200,  # 最多显示词数
    max_font_size=100  # 字体最大值
)

wc.generate_from_frequencies(wordcount)  # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask)  # 从背景图建立颜色方案
wc.recolor(color_func=image_colors)  # 将词云颜色设置为背景图方案
plt.imshow(wc)  # 显示词云
plt.axis('off')  # 关闭坐标轴
wc.to_file('China.png')
plt.show()  # 显示图像
