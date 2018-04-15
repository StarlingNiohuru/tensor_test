import re


def text_to_sentences(input_file_dir):
    with open(input_file_dir, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = raw_text.replace('\t', '').replace('\n\n', '')
    # partern = '“.+?[。！？’…，]”\n*|[^\n“”‘’…].*?[。！？：…；]\n*'
    partern = '[^\n“”‘’…].*?[，。！？：…；]\n*'
    sentences = re.findall(partern, text)
    return sentences


def sentences_by_line(sentences, output_file_dir, max_len=20):
    total_count = len(sentences)
    sentences = [s for s in sentences if len(s) < max_len]
    print(len(sentences) / total_count)
    lines = []
    for s in sentences:
        line = s + '\n'
        lines.append(line)
    with open(output_file_dir, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def sentences_into_pairs(sentences, max_len=20):
    total_count = len(sentences)
    sentences = [s for s in sentences if len(s) < max_len]
    print(len(sentences) / total_count)
    pairs = []
    for s1, s2 in zip(sentences, sentences[1:]):
        line = s1 + '\t' + s2 + '\n'
        pairs.append(line)
    pretrain_text = 'D:/deep_learning/datasets/Water Margin pretrain.txt'
    with open(pretrain_text, 'w', encoding='utf-8') as f:
        f.writelines(pairs)


def load_pairs_save_one(input_file_dir, output_file_dir):
    with open(input_file_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.split('\t')[1] for line in lines]
    with open(output_file_dir, 'w', encoding='utf-8') as f:
        f.writelines(lines)


if __name__ == "__main__":
    load_pairs_save_one('D:/deep_learning/datasets/cmn-eng/cmn.txt', 'D://cntest.txt')
