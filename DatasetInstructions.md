# Dataset Instructions - Deep Past Challenge

## Overview

The biggest challenge in working with Akkadian / Old Assyrian texts is dealing with formatting issues. "Garbage in, garbage out" - the format of text in transliteration poses challenges at each step of the ML workflow, from tokenization to the transformation and embedding process.

This document provides guidance for handling the different formatting challenges in both transliterated and translated texts.

---

## Texts in Transliteration

### Main Formatting Challenges

In addition to the standard transliteration format with hyphenated syllables, additional scribal additions have encumbered the text with:
- **Superscripts** and **subscripts** 
- **Punctuation** meaningful only to specialists in Assyriology

### Capitalization

Capitalization encodes meaning in two different ways:

1. **Lowercase with Capital First Letter** - Indicates personal name or place name (proper noun)
   - Example: `Kanesh` (proper noun)

2. **ALL CAPS** - Indicates Sumerian logogram
   - These were written in place of Akkadian syllabic spelling for scribal simplicity
   - Example: `KÙ.BABBAR` (Sumerian logogram)

### Determinatives

**Determinatives** are used in Akkadian as a type of classifier for nouns and proper nouns. These signs are usually printed in superscript format adjacent to the nouns they classify.

**Format**: To avoid confusion of reading a determinative as part of a word, they are enclosed in curly brackets.

**Examples**:
- `a-lim{ki}` - location classifier
- `A-mur-{d}UTU` - deity classifier

**Important**: Curly brackets are used ONLY for determinatives in transliteration. They are the only use of curly brackets in the transliteration data.

### Broken Text on Tablets

Since these are ancient texts, they include breaks and lacunae (gaps in the text). To standardize these breaks:

1. **Small break** (single sign): `<gap>`
2. **Large break** (more than one sign): `<big_gap>`

---

## Texts in Translation

### Main Challenges

There is currently **no complete or extensive database** for translations of ancient cuneiform documents, especially for Old Assyrian texts. The translations were gathered from books and articles with translations and commentaries:

1. **OCR processing** was used to digitize them
2. **LLM corrections** were applied for accuracy
3. **Formatting issues still remain** - a central component of the challenge

### Formatting Issues

Translations usually retain the same proper noun capitalization. These proper nouns are **generally where most ML tasks underperform**.

**Included Resources**: A lexicon is provided in the dataset with all proper nouns as normalized by specialists for print publications.

---

## Modern Scribal Notations

### Line Numbers

Line numbers are typically numbered 1, 5, 10, 15, etc.

**Broken Lines**: If there are broken lines, line numbers have special trailing marks:
- Single quote `'` - first set of broken lines
- Double quotes `''` - second set of broken lines
  - (These are **NOT** quotation marks, but scribal convention)

### Additional Scribal Notations

| Mark | Meaning |
|------|---------|
| `!` | Exclamation mark - scholar is certain about a difficult reading |
| `?` | Question mark - scholar is uncertain about a difficult reading |
| `/` | Forward slash - signs belonging to a line are found below the line |
| `:` | Colon - Old Assyrian word divider sign |
| `.` | Period - word divider (modern addition) |
| `( )` | Parentheses - comments for breaks and erasures |
| `< >` | Pointy brackets - scribal insertions/corrections |
| `<< >>` | Double pointy brackets - demarcation of errant/erroneous signs |
| `˹ ˺` | Half brackets - partially broken signs |
| `[ ]` | Square brackets - clearly broken signs and lines |
| `{ }` | Curly brackets - determinatives (Akkadian classifiers) |

---

## Formatting Suggestions for Transliterations and Translations

### Remove (Modern Scribal Notations)

Remove these from your preprocessing:

- `!` - certain reading mark
- `?` - questionable reading mark
- `/` - line divider
- `:` OR `.` - word dividers (but keep text as spaces)
- `< >` - scribal insertions (remove brackets but **keep the text** inside)
- `˹ ˺` - partially broken signs (remove markers)
- `[ ]` - square brackets (remove from document-level transliteration, keep text)
  - Example: `[KÙ.BABBAR]` → `KÙ.BABBAR`

### Replace (Breaks, Gaps, Superscripts, Subscripts)

- `[x]` → `<gap>` (small break)
- `…` or `[… …]` → `<big_gap>` (large break)
- `ki` (superscript) → `{ki}` (determinative notation)
- `il5` (subscripted number) → `il5` (normal notation)
  - (Same for any subscripted number)

---

## Character Encoding Normalization

### Accented and Special Characters

| Source | Standard | Unicode |
|--------|----------|---------|
| á | a2 | a₂ |
| à | a3 | a₃ |
| é | e2 | e₂ |
| è | e3 | e₃ |
| í | i2 | i₂ |
| ì | i3 | i₃ |
| ú | u2 | u₂ |
| ù | u3 | u₃ |

### Special Akkadian Characters

| Character | Alt Format 1 | Alt Format 2 | Unicode |
|-----------|------------|------------|---------|
| š | sz | š | U+161 |
| Š | SZ | Š | U+160 |
| Ṣ | s, | ṣ | U+1E63 |
| ṣ | S, | Ṣ | U+1E62 |
| ṭ | t, | ṭ | U+1E6D |
| Ṭ | T, | Ṭ | U+1E6C |
| ḫ | h | h | U+1E2B |
| Ḫ | H | H | U+1E2A |
| ʾ | ' | ʾ | U+02BE |

### Subscript Numbers

| Source | Standard | Unicode |
|--------|----------|---------|
| ₀-₉ | 0-9 | U+2080-U+2089 |
| ₓ | Xx | U+208A |

### Important Note: H Character

**Training data** contains both `Ḫ` and `ḫ` (H with cedilla)  
**Test data** contains only `H` and `h` (regular H)

**Action**: For transliteration text, substitute: `Ḫ ḫ` → `H h`

---

## Akkadian Determinatives in Curly Brackets

Determinatives are semantic classifiers. Complete list of the 17 types used:

| Symbol | Akkadian | English Meaning | Usage |
|--------|----------|-----------------|-------|
| `{d}` | dingir | god, deity | d preceding non-human divine actors |
| `{mul}` | - | stars | MUL preceding astronomical bodies |
| `{ki}` | - | earth | KI following geographical place names |
| `{lu₂}` | LÚ | person | preceding people and professions |
| `{e₂}` | É | building | preceding temples, palaces, institutions |
| `{uru}` | URU | settlement | preceding villages, towns, cities |
| `{kur}` | KUR | land, territory | preceding lands and mountains |
| `{mi}` | munus | feminine | preceding feminine personal names |
| `{m}` | m/1 | masculine | preceding masculine personal names |
| `{geš}` | GIŠ | wood | preceding trees and wooden objects |
| `{tug₂}` | TÚG | textile | preceding woven textiles |
| `{dub}` | DUB | tablet, document | preceding tablets and legal records |
| `{id₂}` | ÍD/A.ENGUR | river | preceding canal/river names or divine river |
| `{mušen}` | MUŠEN | bird | preceding bird names |
| `{na₄}` | na4 | stone | preceding stones |
| `{kuš}` | kuš | skin, hide | preceding animal skins and fleeces |
| `{u₂}` | Ú | plant | preceding plants |

---

## Preprocessing Workflow Recommendation

### For Transliteration (Akkadian):

1. **Normalize Unicode** - Convert special characters to standard form
2. **Remove scribal marks** - Remove `!`, `?`, `/`
3. **Standardize gaps** - Replace `[x]` with `<gap>`, `…` with `<big_gap>`
4. **Normalize determinatives** - Keep in curly brackets or remove based on strategy
5. **Clean whitespace** - Normalize multiple spaces to single space
6. **Remove square brackets** - Convert `[text]` → `text`
7. **Preserve structure** - Keep meaningful punctuation and capitalization

### For Translation (English):

1. **Remove scribal marks** - Remove `!`, `?`
2. **Handle brackets** - Remove `˹ ˺`, convert `[text]` → `text`
3. **Clean parenthetical content** - Keep or remove based on strategy
4. **Normalize whitespace** - Standard spacing
5. **Preserve capitalization** - Keep proper nouns and semantically meaningful caps

---

## Data Quality Notes

- **Proper nouns** are critical for translation quality
- **Lexicon provided** includes normalized proper nouns from specialist publications
- **ML tasks underperform on proper nouns** - this is a known challenge
- **Character encoding varies** - be prepared for multiple Unicode representations
- **Determinatives carry semantic weight** - don't discard carelessly
- **Gaps indicate missing content** - may affect translation quality

---

## Citation

Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A., Ryan Holbrook, and María Cruz. Deep Past Challenge - Translate Akkadian to English. https://kaggle.com/competitions/deep-past-initiative-machine-translation, 2025. Kaggle.

---

**Document Version**: 1.0  
**Created**: February 2, 2026  
**Source**: Kaggle Competition Overview - Deep Past Initiative Machine Translation Challenge  
**Status**: Reference Guide - Use during data preprocessing and cleaning
