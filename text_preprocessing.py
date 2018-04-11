import re


def text_to_sentences():
    text_data_dir = 'D:/datasets/水浒传.txt'
    with open(text_data_dir, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = raw_text.replace('\t', '').replace('\n\n', '')
    # partern = '“.+?[。！？’…，]”\n*|[^\n“”‘’…].*?[。！？：…；]\n*'
    partern = '[^\n“”‘’…].*?[。！？：…；]\n*'
    sentences = re.findall(partern, text)
    return sentences


def sentences_into_pairs(sentences, max_len=70):
    print(len(sentences))
    sentences = [s for s in sentences if len(s) < max_len]
    print(len(sentences))
    pairs = []
    for s1, s2 in zip(sentences, sentences[1:]):
        line = s1 + '\t' + s2 + '\n'
        pairs.append(line)
    pretrain_text = 'D:/datasets/Water Margin pretrain.txt'
    with open(pretrain_text, 'w', encoding='utf-8') as f:
        f.writelines(pairs)


if __name__ == "__main__":
    sentences = text_to_sentences()
    sentences_into_pairs(sentences)
