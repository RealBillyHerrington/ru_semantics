import csv
import re
from collections import namedtuple
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pymorphy2
from nltk.corpus import stopwords

Record = namedtuple('Record', ['term', 'tag', 'value', 'pstv', 'neut', 'ngtv', 'dunno', 'distortion'])


class Dataset:
    _source: str
    _data: dict

    def __init__(self, source: str) -> None:
        self._source = source
        self._data = dict()
        with open(self._source, 'r') as file:
            rec: Record
            for rec in map(Record._make, csv.reader(file, delimiter=';')):
                self._data[rec.term] = rec

    def __getitem__(self, item) -> Optional[Record]:
        try:
            return self._data[item]
        except KeyError:
            return None


Word = np.dtype([('term', '<U25'), ('pos', int)])

WordInfo = np.dtype(Word.descr + [('score', float), ('occ', int)])


def get_masked(array) -> (np.ndarray, np.ndarray, np.ndarray):
    positive = np.ma.masked_where(array['score'] < 0.55, array)
    negative = np.ma.masked_where(array['score'] > -0.35, array)
    neutral = np.ma.masked_where((array['score'] <= -0.35) | (array['score'] >= 0.55), array)
    return positive, neutral, negative


class Sample:
    text: str
    dataset: Dataset = None
    words = None
    _unique_with_score = None
    _words_in_dataset = None

    def __init__(self, text: str, dataset: Dataset) -> None:
        self.dataset = dataset
        self.text = re.sub(r'(?<=[a-яА-Я])\n(?=[a-яА-Я])', '', text)
        self.text = re.sub(r'\n+', ' ', self.text)
        lemm = pymorphy2.MorphAnalyzer()
        blacklist = stopwords.words('russian')
        words = list()
        for match in re.finditer(r'[a-яА-Я]+', self.text):
            try:
                normal_form = lemm.parse(match.group())[0].normal_form
            except IndexError:
                normal_form = match.group()
            if normal_form not in blacklist:
                words.append((normal_form, match.start()))
        self.words = np.array(words, dtype=Word)
        print(f'--- Text info ---\n'
              f'Trimmed text length: {len(self.text)}\n'
              f'Words found: {len(self.words)}\n'
              f'Words in dataset: {len(self.words_in_dataset)}\n'
              f'Unique words in dataset: {len(self.unique_with_score)}\n')

    @property
    def words_in_dataset(self):
        if self._words_in_dataset is None:
            tmp = list()
            for word in self.words:
                score = self.dataset[word['term']]
                if score is not None:
                    tmp.append((*word, score.value, 1))
            self._words_in_dataset = np.array(tmp, dtype=WordInfo)
        return self._words_in_dataset

    @property
    def unique_with_score(self):
        if self._unique_with_score is None:
            _, index, count = np.unique(self.words_in_dataset['term'], return_index=True, return_counts=True)
            self._unique_with_score = self.words_in_dataset[index]
            self._unique_with_score['occ'] = count
        return self._unique_with_score

    def _plot_top_words(self, words, top, title):
        # get top X words by occurrences
        words = np.sort(words, order='occ')[-top:]
        words = np.sort(words, order='score')

        plt.rcdefaults()
        fig, ax = plt.subplots()
        fig.dpi = 300

        positive, neutral, negative = get_masked(words)

        y_pos = np.arange(len(words))
        if positive['term'].count():
            ax.barh(y_pos, positive['score'], color='tab:pink', label='Positive')
        if neutral['term'].count():
            ax.barh(y_pos, neutral['score'], color='tab:gray', label='Neutral')
        if negative['term'].count():
            ax.barh(y_pos, negative['score'], color='tab:cyan', label='Negative')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
        ax.xaxis.grid(True, linewidth=0.5, alpha=0.4)
        # set axis info and legend
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words['term'])
        ax.invert_yaxis()
        ax.set_xlabel('Semantic score')
        ax.legend()

        fig.set_size_inches(5, len(words) * 0.16)
        plt.xlim((-1.05, 1.05))
        plt.ylim((0 - len(y_pos) * 0.005, len(y_pos) + len(y_pos) * 0.005))
        plt.title(title)
        plt.show()

    def plot_top_all_words(self, top: int) -> None:
        words = self.unique_with_score
        self._plot_top_words(words, top, f'Top {top} words')

    def plot_top_positive_words(self, top: int) -> None:
        words = self.unique_with_score
        positive = np.nonzero(words['score'] > 0.55)
        self._plot_top_words(words[positive], top, f'Top {top} positive words')

    def plot_top_negative_words(self, top: int) -> None:
        words = self.unique_with_score
        negative = np.nonzero(words['score'] < -0.35)
        self._plot_top_words(words[negative], top, f'Top {top} negative words')

    def plot_words_by_index(self) -> None:
        plt.rcdefaults()
        fig, ax = plt.subplots()
        fig.dpi = 300

        size = len(self.words_in_dataset) if len(self.words_in_dataset) < 500 else 500
        chunks = list()
        for chunk in np.array_split(self.words_in_dataset, size):
            start = chunk[0]['pos']
            end = chunk[-1]['pos'] + len(chunk[-1])
            chunks.append((np.mean(chunk['score']), start, end))
        chunks = np.array(chunks, dtype=np.dtype([('score', float), ('start', int), ('end', int)]))

        # mask data for coloring
        positive, neutral, negative = get_masked(chunks)

        x_pos = np.arange(len(chunks))
        ax.bar(x_pos, positive['score'], color='tab:pink')
        ax.bar(x_pos, neutral['score'], color='tab:gray')
        ax.bar(x_pos, negative['score'], color='tab:cyan')
        # set grid
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
        ax.xaxis.grid(True, linewidth=0.5, alpha=0.4)
        ax.set_ylabel('Semantic score')
        ax.set_xlabel(f'% Location in text')
        xticks = np.arange(0, size, size / 20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.arange(0, 100, 5).astype(int), rotation=65)
        # set axis info and legend
        fig.set_size_inches(10, 5)
        fig.tight_layout()
        plt.ylim((-1.05, 1.05))
        plt.xlim((0 - size * 0.005, size + size * 0.005))
        plt.title('Text semantic score')
        plt.show()

        sorted_chunks = np.sort(chunks, order='score')
        print(f'===\nChunks with lowest scores:\n===')
        for chunk in sorted_chunks[:3]:
            chunk_index = np.argwhere(chunks == chunk)[0][0]
            chunk_text = self.text[chunk['start']:chunk['end']]
            print(f'Score: {chunk["score"]:.2f} at chunk #{chunk_index} (near {((chunk_index / size) * 100):.1f}%)\n'
                  f'Text: {chunk_text}\n'
                  f'---\n')
        print(f'===\nChunks with highest scores:\n===')
        for chunk in sorted_chunks[-3:]:
            chunk_index = np.argwhere(chunks == chunk)[0][0]
            chunk_text = self.text[chunk['start']:chunk['end']]
            print(f'Score: {chunk["score"]:.2f} at chunk #{chunk_index} (near {((chunk_index / size) * 100):.1f}%)\n'
                  f'Text: {chunk_text}\n'
                  f'---\n')


if __name__ == '__main__':
    ds = Dataset('./data/emo_dict.csv')
    file = open('./data/Война и мир (Том 2, часть 2).txt', 'r')
    sample = Sample(file.read(), ds)
    sample.plot_top_negative_words(25)
    sample.plot_top_positive_words(25)
    sample.plot_top_all_words(25)
    sample.plot_words_by_index()
