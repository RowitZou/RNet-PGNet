"""
    This file takes a CoQA data file as input and generates the input files for training the pipeline models.
"""


import argparse
import json
import re
import time
import string
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
UNK = 'unknown'


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s


def process(text):
    paragraph = nlp.annotate(text, properties={
                             'annotators': 'tokenize, ssplit',
                             'outputFormat': 'json',
                             'ssplit.newlineIsSentenceBreak': 'two'})

    output = {'word': [],
              # 'lemma': [],
              # 'pos': [],
              # 'ner': [],
              'offsets': []}

    for sent in paragraph['sentences']:
        for token in sent['tokens']:
            output['word'].append(_str(token['word']))
            # output['lemma'].append(_str(token['lemma']))
            # output['pos'].append(token['pos'])
            # output['ner'].append(token['ner'])
            output['offsets'].append((token['characterOffsetBegin'], token['characterOffsetEnd']))
    return output


def get_str(output, lower=False):
    s = ' '.join(output['word'])
    return s.lower() if lower else s


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', '-d', type=str, required=True)
    parser.add_argument('--output_file', '-o', type=str, required=True, help='Json file for the DrQA models')
    args = parser.parse_args()

    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    data = []
    start_time = time.time()
    for i, datum in enumerate(dataset['data']):
        if i % 10 == 0:
            print('processing %d / %d (used_time = %.2fs)...' %
                  (i, len(dataset['data']), time.time() - start_time))
        context_str = datum['story']
        _datum = {'context': context_str,
                  'source': datum['source'],
                  'id': datum['id'],
                  'filename': datum['filename']}
        _datum['annotated_context'] = process(context_str)
        _datum['qas'] = []
        _datum['context'] += UNK
        _datum['annotated_context']['word'].append(UNK)
        _datum['annotated_context']['offsets'].append(
            (len(_datum['context']) - len(UNK), len(_datum['context'])))
        assert len(datum['questions']) == len(datum['answers'])

        for question, answer in zip(datum['questions'], datum['answers']):
            assert question['turn_id'] == answer['turn_id']
            idx = question['turn_id']
            _qas = {'turn_id': idx,
                    'question': question['input_text'],
                    'answer': answer['input_text']}

            _qas['annotated_question'] = process(question['input_text'])
            _qas['annotated_answer'] = process(answer['input_text'])

            _datum['qas'].append(_qas)

        data.append(_datum)

    dataset['data'] = data
    with open(args.output_file, 'w') as output_file:
        json.dump(dataset, output_file, sort_keys=True, indent=4)
