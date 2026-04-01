"""
Theme Clustering Preprocessing Module

Minimal preprocessing for messages before embedding.
Matches the preprocessing approach from vector_copy.py.
"""

import os
import re
import unicodedata
from typing import List, Tuple, Dict, Optional

import ftfy
from unidecode import unidecode

from .theme_config import PreprocessingConfig, DEFAULT_CONFIG


# Try to import optional dependencies
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDICNLP_AVAILABLE = True
except ImportError:
    INDICNLP_AVAILABLE = False


class TextPreprocessor:
    """
    Text preprocessor for customer messages.

    Uses minimal preprocessing to preserve semantic meaning for embeddings.
    Matches vector_copy.py approach: no stopword removal, no lemmatization.
    """

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or DEFAULT_CONFIG.preprocessing

        # Initialize language detection model
        self._fasttext_model = None
        if FASTTEXT_AVAILABLE:
            try:
                # Try .bin first, fallback to .ftz
                model_path = os.getenv('FASTTEXT_MODEL_PATH', './lid.176.bin')
                if not os.path.exists(model_path):
                    model_path = os.getenv('FASTTEXT_MODEL_PATH', './lid.176.ftz')
                self._fasttext_model = fasttext.load_model(model_path)
            except Exception as e:
                print(f"[PREPROCESS] Warning: Could not load FastText model: {e}")

        # Initialize Indic normalizers
        self._indic_normalizers = {}
        if INDICNLP_AVAILABLE:
            factory = IndicNormalizerFactory()
            for lang in self.config.indian_languages:
                try:
                    self._indic_normalizers[lang] = factory.get_normalizer(lang)
                except Exception:
                    pass

    def detect_language(self, text: str) -> str:
        """
        Detect language of text using FastText.

        Args:
            text: Input text

        Returns:
            ISO language code (e.g., 'en', 'hi')
        """
        if not self._fasttext_model or not text.strip():
            return self.config.default_language

        try:
            # FastText expects single line
            clean_text = text.replace('\n', ' ').strip()
            predictions = self._fasttext_model.predict(clean_text, k=1)
            lang_code = predictions[0][0].replace('__label__', '')
            return lang_code
        except Exception:
            return self.config.default_language

    def normalize_indic_text(self, text: str, lang: str) -> str:
        """
        Normalize Indic language text using IndicNLP.

        Args:
            text: Input text
            lang: Language code

        Returns:
            Normalized text
        """
        if lang in self._indic_normalizers:
            try:
                return self._indic_normalizers[lang].normalize(text)
            except Exception:
                pass
        return text

    def normalize_hinglish_variants(self, text: str) -> str:
        """
        Normalize common Hinglish spelling variants.

        Args:
            text: Input text with potential Hinglish

        Returns:
            Text with normalized spellings
        """
        variants = {
            'aap': ['aap', 'ap', 'app', 'aapka'],
            'hai': ['hai', 'h', 'he', 'hain'],
            'kya': ['kya', 'kya', 'kyaa', 'kia'],
            'nahi': ['nahi', 'nhi', 'nahin', 'na'],
            'haan': ['haan', 'han', 'haa', 'ha'],
            'bahut': ['bahut', 'bhut', 'boht', 'bahot'],
            'paisa': ['paisa', 'paise', 'paesa', 'pesa'],
            'kitna': ['kitna', 'kitne', 'kitni'],
            'mujhe': ['mujhe', 'muje', 'mjhe', 'mughe'],
            'chahiye': ['chahiye', 'chahie', 'chaiye', 'chiye'],
            'theek': ['theek', 'thik', 'thek'],
            'accha': ['accha', 'acha', 'achha', 'achaa'],
        }

        for normalized, variations in variants.items():
            for variant in variations:
                text = re.sub(
                    r'\b' + re.escape(variant) + r'\b',
                    normalized,
                    text,
                    flags=re.IGNORECASE
                )

        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text.

        Minimal preprocessing approach (from vector_copy.py):
        - Unicode normalization
        - Remove URLs, emails, phone numbers
        - No stopword removal (transformers need full context)
        - No lemmatization
        - Language-specific handling for Indian languages

        Args:
            text: Raw input text

        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""

        # Unicode normalization (NFC - Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        text = ftfy.fix_text(text)

        # Preserve urgency markers for sentiment
        urgency_markers = re.findall(r'[!]{2,}|[?]{2,}|[.]{3,}', text)

        # Basic cleaning
        text = re.sub(r"http\S+", "", text)        # Remove URLs
        text = re.sub(r"\S+@\S+", "", text)        # Remove emails
        text = re.sub(r"\+?\d[\d -]{8,}\d", "", text)  # Remove phone numbers
        text = re.sub(r"[$€₹¥]", "", text)         # Remove currency symbols

        text = text.strip()

        # Language detection
        lang = self.detect_language(text)

        # Language-specific processing
        if lang.startswith("en"):
            # Minimal preprocessing for English
            # No lemmatization, no stopword removal
            processed = text.lower()
            processed = re.sub(r'[^\w\s]', ' ', processed)
            processed = re.sub(r'\s+', ' ', processed).strip()

        elif lang.startswith("hi") or "hi" in lang or any(
            lang.startswith(l) for l in self.config.indian_languages
        ):
            # Indian language processing
            text = self.normalize_indic_text(text.lower(), lang)
            text = self.normalize_hinglish_variants(text)
            processed = re.sub(r'[^\w\s]', ' ', text)
            processed = re.sub(r'\s+', ' ', processed).strip()

        else:
            # Other languages - basic cleaning with unidecode
            processed = unidecode(text.lower())
            processed = re.sub(r'[^\w\s]', ' ', processed)
            processed = re.sub(r'\s+', ' ', processed).strip()

        # Restore urgency markers (limited to 2)
        if urgency_markers:
            processed += " " + " ".join(urgency_markers[:2])

        return processed

    def preprocess_messages(
        self,
        messages: List[str],
        session_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str], Dict[str, List[str]], List[int]]:
        """
        Preprocess a list of messages.

        Applies:
        1. Length filtering (min_message_length)
        2. Text cleaning
        3. Deduplication with session tracking

        Args:
            messages: List of raw message texts
            session_ids: Optional parallel list of session IDs

        Returns:
            Tuple of:
            - cleaned_messages: Deduplicated cleaned messages
            - cleaned_session_ids: Session IDs for cleaned messages
            - message_to_sessions: Dict mapping cleaned_text -> list of all session_ids
            - original_indices: List of original message indices (before filtering/dedup)
        """
        print(f"[PREPROCESS] Starting preprocessing of {len(messages)} messages...")

        # Ensure session_ids parallel to messages
        if session_ids is None:
            session_ids = [f"session_{i}" for i in range(len(messages))]

        # Step 1: Filter by length
        valid_indices = []
        for i, msg in enumerate(messages):
            if msg and len(msg.strip()) >= self.config.min_message_length:
                valid_indices.append(i)

        filtered_count = len(messages) - len(valid_indices)
        print(f"[PREPROCESS] Filtered {filtered_count} short messages (< {self.config.min_message_length} chars)")

        # Step 2: Clean texts and track original indices
        cleaned_texts = []
        cleaned_session_ids_raw = []
        cleaned_original_indices = []

        for i in valid_indices:
            cleaned = self.clean_text(messages[i])
            # Skip if empty or too short after cleaning
            if cleaned and len(cleaned) >= self.config.min_message_length:
                cleaned_texts.append(cleaned)
                cleaned_session_ids_raw.append(session_ids[i])
                cleaned_original_indices.append(i)

        print(f"[PREPROCESS] Cleaned {len(cleaned_texts)} messages")

        # Step 3: Deduplicate while tracking all sessions and original indices
        # Maps cleaned_text -> list of session_ids where it appeared
        message_to_sessions: Dict[str, List[str]] = {}
        seen_texts = set()
        deduplicated_texts = []
        deduplicated_session_ids = []
        deduplicated_original_indices = []

        for text, sid, orig_idx in zip(cleaned_texts, cleaned_session_ids_raw, cleaned_original_indices):
            if text not in message_to_sessions:
                message_to_sessions[text] = []

            message_to_sessions[text].append(sid)

            if text not in seen_texts:
                seen_texts.add(text)
                deduplicated_texts.append(text)
                deduplicated_session_ids.append(sid)
                deduplicated_original_indices.append(orig_idx)

        dup_count = len(cleaned_texts) - len(deduplicated_texts)
        print(f"[PREPROCESS] Removed {dup_count} duplicate messages")
        print(f"[PREPROCESS] Final: {len(deduplicated_texts)} unique messages")

        return deduplicated_texts, deduplicated_session_ids, message_to_sessions, deduplicated_original_indices


def preprocess_messages(
    messages: List[str],
    session_ids: Optional[List[str]] = None,
    config: PreprocessingConfig = None,
) -> Tuple[List[str], List[str], Dict[str, List[str]], List[int]]:
    """
    Convenience function for preprocessing messages.

    Args:
        messages: List of raw message texts
        session_ids: Optional list of session IDs
        config: Optional preprocessing config

    Returns:
        Tuple of (cleaned_messages, session_ids, message_to_sessions, original_indices)
    """
    preprocessor = TextPreprocessor(config)
    return preprocessor.preprocess_messages(messages, session_ids)
