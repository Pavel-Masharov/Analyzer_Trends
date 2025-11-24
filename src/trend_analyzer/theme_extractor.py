from typing import List, Dict
import numpy as np
from collections import Counter
import re
from loguru import logger


class ThemeExtractor:
    """extraction of topics from text clusters"""

    def __init__(self):
        self.stop_words = self._load_stop_words()


    def _load_stop_words(self) -> set:
        """Russian and English stop words"""

        return {
            'это', 'что', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 'то', 'с', 'ли', 'а', 'во',
            'от', 'со', 'для', 'о', 'же', 'ну', 'вы', 'бы', 'что', 'кто', 'он', 'она',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was',
            'were'
        }


    def extract_theme(self, texts: List[str], embeddings: np.ndarray) -> str:
        """Extracts a meaningful topic from a cluster of texts"""
        if not texts:
            return "An undefined topic"

        key_phrases = self._extract_key_phrases(texts)
        centroid_theme = self._centroid_based_theme(texts, embeddings)

        if key_phrases and centroid_theme:
            theme = f"{centroid_theme} ({', '.join(key_phrases[:2])})"
        elif key_phrases:
            theme = "Trend: " + ", ".join(key_phrases[:3])
        else:
            theme = centroid_theme

        return theme


    def _extract_key_phrases(self, texts: List[str], max_phrases: int = 3) -> List[str]:
        """Extracts keywords from texts"""

        full_text = " ".join(texts).lower()

        words = re.findall(r'\b[a-zа-яё]{4,}\b', full_text)
        words = [word for word in words if word not in self.stop_words]

        bigrams = []
        for text in texts:
            text_words = re.findall(r'\b[a-zа-яё]{3,}\b', text.lower())
            text_words = [w for w in text_words if w not in self.stop_words]
            bigrams.extend([f"{text_words[i]} {text_words[i + 1]}"
                            for i in range(len(text_words) - 1)])

        all_terms = words + bigrams
        term_counts = Counter(all_terms)

        top_terms = [term for term, count in term_counts.most_common(max_phrases * 2)]

        filtered_terms = [term for term in top_terms
                          if not self._is_too_general(term)]

        return filtered_terms[:max_phrases]

    def _is_too_general(self, term: str) -> bool:
        """Checks whether the term is too general"""

        general_terms = {'тренд', 'новый', 'новость', 'новости', 'разработка',
                         'технология', 'система', 'программа', 'версия', 'модель'}
        return term in general_terms

    def _centroid_based_theme(self, texts: List[str], embeddings: np.ndarray) -> str:
        """Determines the topic based on the cluster centroid"""

        if len(embeddings) == 0:
            return "Разное"

        centroid = np.mean(embeddings, axis=0)

        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        representative_text = texts[closest_idx]

        theme = self._simplify_text_for_theme(representative_text)
        return theme

    def _simplify_text_for_theme(self, text: str) -> str:
        """Simplifies the text for use as a topic name"""

        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)

        sentences = re.split(r'[.!?]', text)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 10:
                if len(first_sentence) > 50:
                    first_sentence = first_sentence[:47] + "..."
                return first_sentence

        return "Разное"