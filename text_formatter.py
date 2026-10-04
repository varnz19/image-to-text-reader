"""
text_formatter.py
=================
Transforms raw, noisy OCR output into clean, structured, grammatically
consistent "Proper Text" without using external cloud AI APIs.

Features:
- De-hyphenation across line wraps (e.g. 'com- \n puter' -> 'computer')
- Intelligent paragraph reconstruction (merging soft line breaks while keeping paragraph/bullet boundaries)
- Punctuation spacing & normalization (e.g. 'hello ,world' -> 'hello, world')
- Fixes common OCR glyph confusions (e.g., 'th1s' -> 'this', '0K' -> 'OK')
- Sentence casing (capitalizing the start of sentences)
- Offline statistical dictionary correction (Peter Norvig algorithm, pure Python)
"""

import re
from collections import Counter
from typing import Optional, List, Dict


# Common English word frequency list for offline statistical correction
# Built-in lightweight dictionary of ~1200 most common English words
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
    "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
    "us", "is", "was", "are", "been", "has", "had", "were", "said", "did", "having",
    "may", "should", "call", "world", "school", "still", "try", "last", "ask", "need",
    "too", "feel", "three", "state", "never", "become", "between", "high", "really",
    "something", "most", "another", "much", "family", "own", "out", "leave", "put",
    "old", "while", "mean", "keep", "student", "why", "let", "great", "same", "big",
    "group", "begin", "seem", "country", "help", "talk", "where", "turn", "problem",
    "every", "start", "hand", "might", "american", "show", "part", "against", "place",
    "such", "again", "few", "case", "week", "company", "system", "each", "right",
    "program", "hear", "so", "question", "during", "work", "play", "government", "run",
    "small", "number", "off", "always", "move", "night", "live", "point", "believe",
    "hold", "today", "bring", "happen", "next", "without", "before", "large", "million",
    "must", "home", "under", "water", "room", "write", "mother", "area", "national",
    "money", "story", "young", "fact", "month", "different", "lot", "right", "study",
    "book", "eye", "job", "word", "though", "business", "issue", "side", "kind", "four",
    "head", "far", "black", "long", "both", "little", "house", "yes", "since", "provide",
    "service", "around", "friend", "important", "father", "sit", "away", "until", "power",
    "hour", "game", "often", "yet", "line", "political", "end", "among", "ever", "stand",
    "bad", "lose", "however", "member", "pay", "law", "meet", "car", "city", "almost",
    "include", "continue", "set", "later", "community", "much", "name", "five", "once",
    "white", "least", "president", "learn", "real", "change", "team", "minute", "best",
    "several", "idea", "kid", "body", "information", "nothing", "ago", "lead", "social",
    "understand", "whether", "watch", "together", "follow", "parent", "only", "stop",
    "face", "anything", "create", "public", "already", "speak", "others", "read", "level",
    "allow", "add", "office", "spend", "door", "health", "person", "art", "sure", "war",
    "history", "party", "within", "grow", "result", "open", "change", "morning", "walk",
    "reason", "low", "win", "research", "girl", "guy", "early", "food", "moment", "himself",
    "air", "teacher", "force", "offer", "enough", "both", "education", "across", "although",
    "remember", "foot", "second", "boy", "maybe", "toward", "able", "age", "policy",
    "everything", "love", "process", "music", "including", "consider", "appear", "actually",
    "buy", "probably", "human", "wait", "serve", "market", "die", "send", "expect", "sense",
    "build", "stay", "fall", "oh", "nation", "plan", "cut", "college", "interest", "death",
    "course", "someone", "experience", "behind", "reach", "local", "kill", "six", "remain",
    "effect", "use", "yeah", "suggest", "class", "control", "raise", "care", "perhaps",
    "little", "late", "hard", "field", "else", "pass", "former", "sell", "major", "sometimes",
    "require", "along", "development", "themselves", "report", "role", "better", "economic",
    "effort", "decide", "rate", "strong", "possible", "heart", "drug", "show", "leader",
    "light", "voice", "wife", "whole", "police", "mind", "finally", "pull", "return",
    "free", "military", "price", "report", "less", "according", "decision", "explain",
    "son", "hope", "even", "develop", "view", "relationship", "carry", "town", "road",
    "drive", "arm", "true", "federal", "break", "better", "difference", "thank", "receive",
    "value", "international", "building", "action", "full", "model", "join", "season",
    "society", "tax", "director", "early", "position", "player", "agree", "especially",
    "record", "pick", "wear", "paper", "special", "space", "ground", "form", "support",
    "event", "official", "whose", "matter", "everyone", "center", "couple", "site", "project",
    "hit", "base", "activity", "star", "table", "need", "court", "produce", "eat", "american",
    "teach", "oil", "half", "situation", "easy", "cost", "industry", "figure", "street",
    "image", "itself", "phone", "either", "data", "cover", "quite", "picture", "clear",
    "practice", "piece", "land", "recent", "describe", "product", "doctor", "wall", "patient",
    "worker", "news", "test", "movie", "certain", "north", "personal", "simply", "third",
    "technology", "catch", "step", "baby", "computer", "type", "attention", "draw", "film",
    "republican", "tree", "source", "red", "nearly", "organization", "choose", "cause",
    "hair", "look", "point", "century", "evidence", "window", "difficult", "listen", "soon",
    "culture", "billion", "chance", "brother", "energy", "period", "course", "summer",
    "less", "realize", "hundred", "available", "plant", "likely", "opportunity", "term",
    "short", "letter", "condition", "choice", "single", "rule", "daughter", "administration",
    "south", "husband", "congress", "floor", "campaign", "material", "population", "well",
    "call", "economy", "medical", "hospital", "church", "close", "thousand", "risk", "current",
    "fire", "future", "wrong", "involve", "defense", "anyone", "increase", "security",
    "bank", "myself", "certainly", "west", "sport", "board", "seek", "per", "subject",
    "officer", "private", "rest", "behavior", "deal", "performance", "fight", "throw",
    "top", "quickly", "past", "goal", "second", "bed", "order", "author", "fill", "represent",
    "focus", "foreign", "drop", "plan", "blood", "upon", "agency", "push", "nature", "color",
    "no", "recently", "store", "reduce", "sound", "note", "fine", "near", "movement", "page",
    "enter", "share", "than", "common", "poor", "other", "natural", "race", "concern",
    "series", "significant", "similar", "hot", "language", "each", "usually", "response",
    "dead", "rise", "animal", "factor", "decade", "article", "shoot", "east", "save", "seven",
    "artist", "away", "scene", "stock", "career", "despite", "central", "eight", "thus",
    "treatment", "beyond", "happy", "exactly", "protect", "approach", "lie", "size", "dog",
    "fund", "serious", "occur", "media", "ready", "sign", "thought", "list", "individual",
    "simple", "quality", "pressure", "accept", "answer", "hard", "resource", "identify",
    "left", "meeting", "determine", "prepare", "disease", "whatever", "success", "argue",
    "cup", "particularly", "amount", "ability", "staff", "recognize", "indicate", "character",
    "growth", "loss", "degree", "wonder", "attack", "herself", "traditional", "equipment",
    "invoice", "receipt", "total", "subtotal", "tax", "date", "quantity", "price", "amount",
    "description", "item", "payment", "balance", "thank", "you", "welcome", "customer",
    "address", "phone", "email", "website", "sale", "order", "card", "cash", "change",
    "document", "machine", "learning", "artificial", "intelligence", "cloud", "zero",
    "execution", "precision", "system", "systems", "fast", "local", "model", "network",
    "neural", "deep", "python", "image", "text", "reader", "scanner", "file", "download"
}

# Frequency counts for words
WORD_COUNTS = Counter({w: 100 for w in COMMON_WORDS})


class ProperTextFormatter:
    """
    Cleans raw OCR output and constructs coherent, properly punctuated,
    and cleanly capitalized paragraphs.
    """

    def __init__(self, enable_spell_check: bool = False):
        self.enable_spell_check = enable_spell_check

    def clean_light(self, text: str) -> str:
        """
        Fidelity-preserving cleanup: never changes words, only whitespace,
        spacing before punctuation and stray quote glyphs.
        """
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" +([,.;:!?])(?=\s|$)", r"\1", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def format_text(self, raw_text: str) -> str:
        """
        Main pipeline to format raw OCR text into 'Proper Text'.
        """
        if not raw_text or not raw_text.strip():
            return ""

        # Step 1: Normalize Unicode and line breaks
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: Remove OCR junk characters (lone symbols on lines, random control chars)
        text = self._remove_ocr_artifacts(text)

        # Step 3: De-hyphenate line breaks (e.g. 'auto- \n matic' -> 'automatic')
        text = self._dehyphenate(text)

        # Step 4: Fix fused / glued words from OCR (e.g. 'documentwith' -> 'document with')
        text = self._split_fused_words(text)

        # Step 5: Fix OCR character confusions (e.g., '1' or '0' inside purely alphabetic words)
        text = self._fix_character_confusions(text)

        # Step 5: Stitch soft line breaks while preserving paragraph & list structures
        text = self._reconstruct_paragraphs(text)

        # Step 6: Fix spacing around punctuation
        text = self._normalize_punctuation(text)

        # Step 7: Sentence capitalization
        text = self._capitalize_sentences(text)

        # Step 8: Optional offline statistical spell check
        if self.enable_spell_check:
            text = self._apply_spell_correction(text)

        return text.strip()

    def _remove_ocr_artifacts(self, text: str) -> str:
        """Removes standalone garbage characters produced by noise or speckles."""
        lines = []
        for line in text.split("\n"):
            # If line is only non-alphanumeric noise of length 1-2 like '|', '~', '`', remove it
            cleaned = line.strip()
            if len(cleaned) <= 2 and not any(c.isalnum() for c in cleaned):
                continue
            # Remove repeated vertical bars or tildes common in OCR border misdetections
            cleaned = re.sub(r'^[|_~`\-+=]{3,}$', '', cleaned)
            lines.append(line)
        return "\n".join(lines)

    def _dehyphenate(self, text: str) -> str:
        """Joins words broken by a hyphen at the end of a line."""
        # e.g. "inform-\nation" -> "information"
        # e.g. "infor- \n mation" -> "information"
        # e.g. "docu - \n ment" -> "document"
        pattern = r'([a-zA-Z]{2,})\s*-\s*\n\s*([a-zA-Z]{2,})'
        return re.sub(pattern, r'\1\2', text)

    def _split_fused_words(self, text: str) -> str:
        """
        Splits words accidentally glued together by OCR (e.g. 'documentwith' -> 'document with',
        'ofan' -> 'of an', 'gearand' -> 'gear and', 'tothe' -> 'to the').
        Uses common English dictionary splitting.
        """
        two_letter_words = {"of", "to", "in", "on", "at", "an", "is", "it", "as", "by", "or", "we", "he", "my", "up", "so", "no"}

        def split_match(match):
            word = match.group(0)
            if len(word) < 4 or not word.isalpha():
                return word
            
            lower = word.lower()
            # If the entire word is already in vocabulary or very common, don't split
            if lower in COMMON_WORDS and len(lower) <= 5:
                # E.g. 'often', 'today', 'plant', 'point' shouldn't be split
                return word

            # Try splitting into two valid words
            for i in range(2, len(lower) - 1):
                left, right = lower[:i], lower[i:]
                valid_left = (left in COMMON_WORDS) or (left in two_letter_words) or (left.endswith('s') and left[:-1] in COMMON_WORDS)
                valid_right = (right in COMMON_WORDS) or (right in two_letter_words) or (right.endswith('s') and right[:-1] in COMMON_WORDS)
                
                if valid_left and valid_right:
                    # Guard against false splits on common valid words
                    if lower in {"into", "upon", "today", "often", "point", "sound", "plant", "water"}:
                        return word
                    # Preserve original capitalization
                    left_orig = word[:i]
                    right_orig = word[i:]
                    return f"{left_orig} {right_orig}"

            return word

        return re.sub(r'\b[a-zA-Z]+\b', split_match, text)

    def _fix_character_confusions(self, text: str) -> str:
        """
        Fixes common OCR glyph mistakes within words:
        e.g., 'm0del' -> 'model', 'w1th' -> 'with', 'th1s' -> 'this', 'f1rst' -> 'first'
        """
        def replace_confusions(match):
            word = match.group(0)
            # If word is an intentional alphanumeric code (like 'COVID19', 'A4', 'H2O', '3D'), keep it
            if len(word) <= 2:
                return word
            
            # Common specific OCR typo dictionary
            common_typos = {
                "th1s": "this", "w1th": "with", "f1rst": "first", "sec0nd": "second",
                "m0del": "model", "rec1pe": "recipe", "rece1pt": "receipt",
                "t0": "to", "y0u": "you", "th3": "the", "c4n": "can"
            }
            if word.lower() in common_typos:
                fixed = common_typos[word.lower()]
                return fixed.capitalize() if word.istitle() else fixed

            # Count letters vs digits
            letters = sum(1 for c in word if c.isalpha())
            digits = sum(1 for c in word if c.isdigit())
            
            # If mostly letters with 1 or 2 isolated digits that look like letters:
            if letters >= 3 and digits <= 2:
                # Replace '0' with 'o' if surrounded by letters
                fixed = re.sub(r'(?<=[a-zA-Z])0(?=[a-zA-Z])', 'o', word)
                # Replace '1' with 'i' if surrounded by consonants (e.g. th1s -> this) or 'l'
                fixed = re.sub(r'(?<=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])1(?=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])', 'i', fixed)
                fixed = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', fixed)
                # Replace '5' with 's' if surrounded by letters
                fixed = re.sub(r'(?<=[a-zA-Z])5(?=[a-zA-Z])', 's', fixed)
                # Replace '4' with 'a' if surrounded by letters (e.g. r4ndom -> random)
                fixed = re.sub(r'(?<=[a-zA-Z])4(?=[a-zA-Z])', 'a', fixed)
                return fixed
            return word

        return re.sub(r'\b[a-zA-Z0-9]+\b', replace_confusions, text)

    def _reconstruct_paragraphs(self, text: str) -> str:
        """
        Distinguishes between hard paragraph breaks (double newlines, bullet points,
        numbered lists, headers) and soft line wraps (a single newline in the middle
        of a sentence) and joins soft wraps into single flowing lines.
        """
        lines = [line.rstrip() for line in text.split("\n")]
        paragraphs: List[str] = []
        current_para: List[str] = []

        # Indicators for list items or distinct blocks
        list_pattern = re.compile(r'^(\d+[\.\)]|[\*\-\•\–\—]|\([a-zA-Z0-9]+\))\s+')

        for line in lines:
            line_stripped = line.strip()

            # Empty line indicates deliberate paragraph break
            if not line_stripped:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                continue

            # Check if this line starts a new list item
            is_list_item = bool(list_pattern.match(line_stripped))

            # Check if this line looks like a header (short, all caps or title case, no ending punctuation)
            is_header = len(line_stripped) < 40 and not line_stripped.endswith(('.', ',', ';', ':')) and (
                line_stripped.isupper() or line_stripped.istitle()
            )

            if is_list_item or is_header:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                current_para.append(line_stripped)
            else:
                if current_para:
                    # Check if the previous line ended with sentence punctuation
                    last_line = current_para[-1]
                    # If previous ended with period or colon, or current begins with large indent
                    if last_line.endswith(('.', '!', '?', ':')) and len(line) - len(line_stripped) >= 4:
                        paragraphs.append(" ".join(current_para))
                        current_para = [line_stripped]
                    else:
                        current_para.append(line_stripped)
                else:
                    current_para.append(line_stripped)

        if current_para:
            paragraphs.append(" ".join(current_para))

        return "\n\n".join(paragraphs)

    def _normalize_punctuation(self, text: str) -> str:
        """
        Cleans up spacing around punctuation:
        - Removes space before: , . ! ? ; : ) ] }
        - Ensures single space after: , . ! ? ; :
        - Fixes stray double quotes and apostrophes (e.g. "don ' t" -> "don't")
        """
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        # Fix spaces before punctuation
        text = re.sub(r'\s+([,.:;!?\)\]\}])', r'\1', text)

        # Fix missing space after punctuation when followed by a letter (e.g. "Hello.World" -> "Hello. World")
        # Be careful not to break decimals like 3.14 or URLs like example.com
        text = re.sub(r'([,;!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)

        # Fix contractions with spaces: "don ' t" -> "don't", "it ' s" -> "it's"
        text = re.sub(r"([a-zA-Z])\s*['’]\s*([a-zA-Z])", r"\1'\2", text)

        # Fix parenthesis spacing: "( word )" -> "(word)"
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # Fix multiple periods (unless it's an ellipsis like ...)
        text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)

        return text

    def _capitalize_sentences(self, text: str) -> str:
        """
        Capitalizes the first character of each sentence and paragraph.
        """
        paragraphs = text.split("\n\n")
        formatted_paras = []

        for para in paragraphs:
            lines = para.split("\n")
            formatted_lines = []
            for line in lines:
                if not line:
                    formatted_lines.append("")
                    continue

                # Capitalize first character of the line if it's an alphabet
                first_char_idx = None
                for idx, ch in enumerate(line):
                    if ch.isalpha():
                        first_char_idx = idx
                        break

                if first_char_idx is not None:
                    line = line[:first_char_idx] + line[first_char_idx].upper() + line[first_char_idx + 1:]

                # Capitalize letters following sentence terminators (. ! ?)
                def cap_after_punct(match):
                    punct = match.group(1)
                    space = match.group(2)
                    letter = match.group(3)
                    return f"{punct}{space}{letter.upper()}"

                line = re.sub(r'([.!?])(\s+)([a-z])', cap_after_punct, line)
                formatted_lines.append(line)

            formatted_paras.append("\n".join(formatted_lines))

        return "\n\n".join(formatted_paras)

    def _apply_spell_correction(self, text: str) -> str:
        """
        Offline Norvig spell-checker for English words.
        Only corrects isolated alphabetic words with low edit distance to common words.
        """
        def correct_word(match):
            word = match.group(0)
            lower = word.lower()

            # Don't touch capitalized acronyms or very short words
            if len(word) <= 2 or len(word) > 16:
                return word

            # Already in dictionary
            if lower in COMMON_WORDS:
                return word

            # If word is a standard plural or inflection of a common word, keep it!
            if lower.endswith('s') and lower[:-1] in COMMON_WORDS:
                return word
            if lower.endswith('es') and lower[:-2] in COMMON_WORDS:
                return word
            if lower.endswith('ed') and (lower[:-2] in COMMON_WORDS or lower[:-1] in COMMON_WORDS):
                return word
            if lower.endswith('ing') and (lower[:-3] in COMMON_WORDS or (lower[:-3] + 'e') in COMMON_WORDS):
                return word

            candidates = self._edits1(lower) & set(COMMON_WORDS)
            if candidates:
                best = max(candidates, key=lambda w: WORD_COUNTS.get(w, 0))
                if word.istitle():
                    return best.capitalize()
                return best

            return word

        return re.sub(r'\b[a-zA-Z]+\b', correct_word, text)

    def _edits1(self, word: str):
        """All edits that are one edit distance away from word."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)


# Quick test
if __name__ == "__main__":
    raw_sample = """
    th1s  is an ex- 
    ample of a docu - 
    ment with bad  spac- 
    ing , broken lines and r4nd0m   
    artifacts .
    
    1. first   bullet point
    2. sec0nd bullet point
    """
    formatter = ProperTextFormatter(enable_spell_check=True)
    clean = formatter.format_text(raw_sample)
    print("--- RAW ---")
    print(raw_sample)
    print("--- PROPER TEXT ---")
    print(clean)
