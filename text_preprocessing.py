import re


def load_text():
    text_data_dir = 'D:/datasets/水浒传.txt'
    with open(text_data_dir, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = raw_text.replace('\n\n', '\n')
    partern = '[.+[。！？；：”]]?'
    sequences = re.findall(partern, text, re.S)
    return sequences
