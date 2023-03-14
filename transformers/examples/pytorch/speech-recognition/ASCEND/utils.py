from itertools import chain

import re
import jieba
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.metrics import confusion_matrix
#####
# Common Functions
#####
CHARS_TO_IGNORE = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                   "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")",
                   "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                   "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽",
                   "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "ʻ", "ˆ"]

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

#####
# Metric Helper Functions
#####
def tokenize_for_mer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, jieba.lcut(text)))
    tokens = [[tok] if tok.isascii() else list(tok) for tok in tokens]
    return list(chain(*tokens))

def tokenize_for_cer(text):
    tokens = list(filter(lambda tok: len(tok.strip()) > 0, list(text)))
    return tokens

def plt_confusion_matrix(output,target,acc):
    labels=['silent','english','chinese']
    a=confusion_matrix(target,output)
    sns.set()
    f,ax=plot.subplots(figsize=(10,10))
    sns.heatmap(a,annot=True,ax=ax,xticklabels=labels, yticklabels=labels) #画热力图
    ax.set_title('confusion matrix') #标题
    ax.set_xlabel('predict') #x轴
    ax.set_ylabel('true') #y轴
    plot.savefig(str(acc)+".png")