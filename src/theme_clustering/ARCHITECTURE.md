# Theme Clustering Pipeline - Architecture & Design

**Pipeline Type:** GPT + Cosine Similarity Theme Extraction
**Approach:** Theme-First Discovery (discover themes first, then assign messages)
**Version:** 1.1
**Date:** February 2026

---

## Recent Updates (v1.1)

### Phase 3.5: Pre-Assignment Merging (NEW)

- **Redundant theme merging now happens BEFORE assignment** (Phase 3.5)
- **Old flow:** Discover → Embed → Assign → Merge → Reassign (inefficient)
- **New flow:** Discover → Embed → **Merge** → Assign (efficient)
- **Benefit:** Avoid wasting computation assigning to themes that will be merged
- Uses GPT-based evaluation with strict criteria to decide which themes merge

### Borderline Reassignment (UPDATED)

- **Only BORDERLINE messages** (similarity ≥ 0.41) are reassigned when misc > 15%
- **MISCELLANEOUS messages** (similarity < 0.41) now **stay in misc** - no longer reassigned
- **Removed "weak misc" reassignment** for cleaner theme quality
- Prevents polluting themes with genuinely weak matches

### Updated Thresholds

- **Primary threshold:** 0.55 → 0.52 (slightly more lenient)
- **Confidence gap:** 0.08 → 0.07 (accept closer matches)
- **Borderline threshold:** 0.45 → 0.41 (lower floor for reassignment)
- **Discovery:** 5-12 themes per batch (stricter), target 18 final themes
- **Key phrases:** 6-10 per theme (more coverage)

---

## Table of Contents

1. [Overview](#overview)
2. [Core Philosophy](#core-philosophy)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
5. [Design Decisions](#design-decisions)
6. [Configuration System](#configuration-system)
7. [Quality Assurance](#quality-assurance)
8. [Code Structure](#code-structure)
9. [Usage Guide](#usage-guide)

---

## Overview

### What This Pipeline Does

The Theme Clustering Pipeline automatically discovers conversation themes from customer messages using a hybrid approach:

1. **GPT-4.1** discovers themes from a diverse sample
2. **OpenAI embeddings** (text-embedding-3-large) enable semantic matching
3. **Cosine similarity** assigns messages to themes
4. **Quality assurance** refines and validates results

### Key Differentiators

Unlike traditional clustering (cluster first, label later), this pipeline:

- **Discovers themes first** with GPT understanding
- **Assigns messages second** via semantic similarity
- **Preserves semantic nuance** through multiple key phrases per theme
- **Adapts to data size** with intelligent sampling
- **Self-validates** with coherence metrics

---

## Core Philosophy

### Theme-First vs Cluster-First

**Traditional Approach (Cluster-First):**

```
Messages → Embeddings → KMeans/HDBSCAN → Clusters → GPT Labels
```

**Problems:**

- Clusters formed without understanding content
- Hard distance boundaries miss semantic overlap
- Labels applied after the fact may not fit well

**Our Approach (Theme-First):**

```
Messages → Sample → GPT Themes → Embed Key Phrases → Assign via Similarity
```

**Benefits:**

- Themes based on semantic understanding
- Multiple key phrases cast wider net
- Soft boundaries via similarity thresholds
- Themes can overlap semantically

---

## Pipeline Architecture

### Mid-Level Flow (Detailed)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Messages + Session IDs                  │
│                    (Customer chat data from JSON)                     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 0: Preprocessing       │
              │  ─────────────────────────    │
              │  1. FastText language detect  │
              │  2. NFKC unicode normalize    │
              │  3. Hinglish normalization    │
              │  4. Exact deduplication       │
              │  → cleaned_messages           │
              │  → message_to_sessions map    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Message Embedding (Cached)   │
              │  ─────────────────────────    │
              │  1. Check cache file exists   │
              │     embeddings/{client}.npy   │
              │  2. If exists: load & return  │
              │  3. If not: call OpenAI API   │
              │     text-embedding-3-large    │
              │  4. Save to cache for reuse   │
              │  → message_embeddings (N×1024)│
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 1: Adaptive Sampling   │
              │  ─────────────────────────    │
              │  1. Calculate sample size:    │
              │     • ≤500: 75% (max 500)     │
              │     • ≤1000: 75% (max 500)    │
              │     • ≤1500: 33% (max 650)    │
              │     • >1500: fixed 800        │
              │  2. Mini-KMeans clustering    │
              │     k = sample_size / 5       │
              │  3. Stratified selection      │
              │     (proportional per cluster)│
              │  → sampled_messages (100-800) │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 2: Theme Discovery     │
              │  ─────────────────────────    │
              │  1. Batch samples (200 each)  │
              │  2. For each batch:           │
              │     • Call GPT-4.1 with       │
              │       discovery prompt        │
              │     • Extract 5-12 themes     │
              │     • Each with 6-10 key      │
              │       phrases in English      │
              │     • Phrases expand abbrevs  │
              │       and cover variations    │
              │  3. If multiple batches:      │
              │     • Consolidate via GPT     │
              │     • Deduplicate themes      │
              │  4. Add seed themes if any    │
              │  → themes (12-18)             │
              │    each with key_phrases[]    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 3: Theme Embedding     │
              │  ─────────────────────────    │
              │  1. Embed all key phrases     │
              │     for all themes (batch API)│
              │  2. For each theme:           │
              │     centroid = mean(phrase_   │
              │                embeddings)    │
              │  → themes with embeddings     │
              │    phrase_embeddings (P×1024) │
              │    theme_embedding (1×1024)   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ PHASE 3.5: Redundant Merging  │
              │  ─────────────────────────    │
              │  **NEW: Merge BEFORE assign** │
              │  1. Compute theme centroid    │
              │     similarity matrix         │
              │  2. Find pairs > 0.65         │
              │     (candidate threshold)     │
              │  3. GPT evaluates each pair   │
              │     with strict criteria      │
              │  4. Merge approved pairs:     │
              │     • Keep theme with lower ID│
              │     • Combine key phrases     │
              │  → merged themes (10-15)      │
              │                               │
              │  **Benefit:** Avoid assigning │
              │  messages to themes that will │
              │  be merged later!             │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 4: Message Assignment  │
              │  ─────────────────────────    │
              │  For each message:            │
              │  1. For each theme:           │
              │     • Compute cosine_sim to   │
              │       ALL key phrases         │
              │     • theme_score = MAX(sims) │
              │  2. Find best & 2nd best      │
              │  3. confidence_gap = best -   │
              │                      second   │
              │  4. Decision:                 │
              │     • best≥0.52 AND gap≥0.07  │
              │       → CONFIDENT             │
              │     • best≥0.41               │
              │       → BORDERLINE            │
              │     • else                    │
              │       → MISCELLANEOUS         │
              │  → assignments[] with status  │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  PHASE 5: Quality Assurance   │
              │  ─────────────────────────    │
              │  Step 1: Dissolve Tiny Themes │
              │    • Count msgs per theme     │
              │    • If < 5: dissolve         │
              │    • Move to miscellaneous    │
              │                               │
              │  Step 2: Borderline Reassign  │
              │    • Check if misc > 15%      │
              │    • If yes: reassign ONLY    │
              │      BORDERLINE msgs (≥0.41)  │
              │      to their best theme      │
              │    • MISC msgs stay misc      │
              │                               │
              │  Step 3: Coherence Analysis   │
              │    For each theme:            │
              │    • CV = std/mean of         │
              │      pairwise similarities    │
              │    • IQR outlier detection    │
              │    • Coherence score          │
              │                               │
              │  Step 4: Silhouette           │
              │    • Global separation metric │
              │    • Per-cluster validation   │
              │                               │
              │  → refined themes & assignments│
              │  → quality_report             │
              │                               │
              │  **Note:** Redundant merging  │
              │  already done in Phase 3.5!   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Build Output & Export        │
              │  ─────────────────────────    │
              │  1. Group by theme_id         │
              │  2. Create ClusterResult      │
              │     for each theme            │
              │  3. Add miscellaneous cluster │
              │  4. Generate exports:         │
              │     • clusters_*.json         │
              │     • quality_report_*.json   │
              │     • themes_*.json           │
              │     • report_*.html           │
              └───────────────┬───────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Complete Results                           │
│  • 10-18 actionable themes with descriptions & key phrases           │
│  • 70-85% message coverage (assigned to themes)                      │
│  • 10-20% miscellaneous (genuine outliers)                           │
│  • Quality metrics: coherence, silhouette, coverage                  │
│  • Interactive HTML report with all messages                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase-by-Phase Breakdown

### Phase 0: Preprocessing (`theme_preprocessing.py`)

**Purpose:** Clean and normalize messages for consistent embedding

**Operations:**

1. **Language Detection** (FastText lid.176.bin)

   - Detect language for each message
   - Filter to primary languages (configurable)
   - Preserve multilingual messages if needed
2. **Unicode Normalization**

   - NFKC normalization for consistent encoding
   - Handle Hinglish variants (transliteration normalization)
3. **Deduplication**

   - Exact duplicate removal
   - Preserves message-to-session mapping

**Design Decision: Minimal Preprocessing**

- **NO stopword removal** - context matters for embeddings
- **NO lemmatization** - preserves natural language structure
- **NO stemming** - maintains semantic richness
- **Rationale:** Modern embeddings (text-embedding-3-large) work best with natural, minimally-processed text

**Code Location:** `ThemePreprocessor` class in `theme_preprocessing.py`

---

### Phase 1: Adaptive Sampling (`theme_sampling.py`)

**Purpose:** Select diverse, representative messages for GPT theme discovery

**Why Sample?**

- GPT has token limits (can't process 10,000 messages)
- Sampling reduces cost and latency
- Representative sample yields same themes as full dataset

**Sampling Strategy: Stratified via Mini-KMeans**

```python
def calculate_sample_size(total: int) -> int:
    if total <= 100:    return total        # Small: use all
    elif total <= 500:  return min(250, 60%)  # Medium: 60%
    elif total <= 1500: return min(300, 20%)  # Large: 20%
    else:               return 350            # Very large: fixed 350
```

**Stratification Process:**

1. Run mini-KMeans on message embeddings
   - k = sample_size // 5 (e.g., 300 samples → 60 clusters)
2. From each mini-cluster, select messages proportionally
3. Ensures diversity across semantic space

**Design Decision: Why Stratified?**

- **Pure random:** Might miss rare but important themes
- **Top-K by frequency:** Biases toward common patterns
- **Stratified:** Guarantees coverage of semantic diversity

**Code Location:** `ThemeSampler` class in `theme_sampling.py`

---

### Phase 2: Theme Discovery (`theme_discovery.py`)

**Purpose:** Use GPT-4.1 to discover themes from sampled messages

**Process:**

1. **Batching** (100 messages per GPT call)

   - Sample split into batches of 100
   - Prevents token overflow
   - Parallel processing potential
2. **GPT Prompt Engineering**

   ```
   System: You are analyzing customer messages...
   User: Here are 100 messages. Discover 10-15 themes.
   Requirements:
   - Actionable themes (not too broad/narrow)
   - 4-6 English key phrases per theme
   - Phrases in customer language style
   ```
3. **Response Parsing**

   - Structured JSON output
   - Validates theme names, descriptions, key phrases
   - Handles GPT formatting inconsistencies
4. **Consolidation** (if multiple batches)

   - If sample > 100: Multiple GPT calls
   - Themes merged and deduplicated by GPT
   - Final count: 15-35 themes

**Design Decision: Multiple Key Phrases**

- Each theme has **4-6 key phrases** instead of just a label
- **Rationale:** Casts wider semantic net
  - "delivery status" AND "track my order" AND "where is my package"
  - Captures linguistic variation
  - Better assignment coverage

**Seed Themes Support:**

- User can provide seed themes to ensure coverage
- Seed themes always included in final set
- GPT discovers additional themes around seeds

**Code Location:** `ThemeDiscoverer` class in `theme_discovery.py`

---

### Phase 3: Embedding (`theme_embedding.py`)

**Purpose:** Embed themes and messages for similarity computation

**Operations:**

1. **Key Phrase Embedding**

   - Embed all key phrases for all themes
   - Model: text-embedding-3-large (1024 dimensions)
   - Batched API calls with retry logic
2. **Theme Centroid Computation**

   ```python
   theme_centroid = mean(phrase_embeddings)
   ```

   - Average of all key phrase embeddings
   - Represents theme's semantic center
3. **Message Embedding** (with caching)

   - Cache file: `embeddings/{client}_{count}.npy`
   - Loads from cache if exists
   - Saves after generation for reuse

**Design Decision: Why Centroid?**

- **Alternative:** Use only first key phrase
- **Our choice:** Average all phrases
- **Rationale:** More stable representation, less sensitive to phrase selection

**Embedding Model Choice:**

- **text-embedding-3-large** (1024-dim)
- **Why not smaller?** Better semantic understanding
- **Why not larger?** 1024 dims sufficient, faster compute

**Code Location:** `ThemeEmbedder` class in `theme_embedding.py`

---

### Phase 3.5: Redundant Theme Merging (`theme_quality.py`)

**Purpose:** Merge redundant themes BEFORE message assignment to avoid wasted computation

**Why Before Assignment?**

- **Old flow (inefficient):** Discover → Embed → Assign → Merge → Reassign
  - Problem: Waste computation assigning messages to themes that get merged later
  - Problem: Have to update all assignments after merging
- **New flow (efficient):** Discover → Embed → **Merge** → Assign
  - Benefit: Only assign to final set of themes
  - Benefit: No assignment updates needed

**Process:**

1. **Candidate Detection**

   ```python
   similarity_matrix = cosine_similarity(theme_centroids)
   candidate_threshold = 0.65  # Lower threshold for GPT evaluation
   ```

   - Compute pairwise centroid similarities
   - Find theme pairs above 0.65 as candidates
   - Note: Lower than final merge threshold to catch more potential merges
2. **GPT-Based Evaluation**

   - For each candidate pair, GPT evaluates with strict criteria:
     - ✓ SAME CUSTOMER GOAL: Exact same customer need
     - ✓ INTERCHANGEABLE: Support agent handles identically
     - ✓ NO INFORMATION LOSS: No actionable distinction
   - GPT returns "merge" or "separate" with reasoning
3. **Union-Find Merging**

   ```python
   # Without assignments yet, merge by theme_id (keep lower ID)
   if root1 < root2:
       merge_map[root2] = root1
   else:
       merge_map[root1] = root2
   ```

   - Build equivalence classes of themes to merge
   - Keep theme with lower ID (arbitrary but deterministic)
   - Combine key phrases from merged themes

**Design Decision: Why GPT for Merge Decisions?**

- **Embedding similarity ≠ merge decision**
- Themes can be semantically related but functionally distinct
- Examples of high similarity but should NOT merge:
  - "Order Tracking" vs "Order Contents"
  - "Product Usage" vs "Product Ingredients"
  - "Cancel Order" vs "Modify Order"
- GPT understands functional equivalence beyond semantic similarity

**Code Location:** `QualityAssessor.detect_redundant_themes()` and `QualityAssessor.merge_themes()` in `theme_quality.py`

---

### Phase 4: Message Assignment (`theme_assignment.py`)

**Purpose:** Assign messages to themes via cosine similarity

**Assignment Algorithm:**

```python
For each message:
    For each theme:
        similarities = cosine_similarity(msg_embedding, all_key_phrase_embeddings)
        theme_score = MAX(similarities)  # Best matching phrase

    best_theme = theme with highest score
    second_best = theme with second highest score

    confidence_gap = best_score - second_best_score

    if best_score >= 0.55 AND confidence_gap >= 0.08:
        ASSIGN to best_theme (CONFIDENT)
    elif best_score >= 0.45:
        Mark as BORDERLINE (may reassign later)
    else:
        MISCELLANEOUS
```

**Key Thresholds:**

- **Primary threshold:** 0.52 (similarity to best theme)
- **Confidence gap:** 0.07 (difference from second-best)
- **Borderline threshold:** 0.41 (minimum consideration)

**Design Decision: Why Max Similarity?**

- Each theme has multiple key phrases
- We take the **maximum** similarity to any phrase
- **Rationale:** If message matches ANY phrase well, it belongs to theme
- **Alternative (mean):** Would be too conservative

**Design Decision: Confidence Gap**

- Not enough to just match best theme
- Must be **clearly better** than second-best
- **Rationale:** Prevents ambiguous assignments
- **Example:**
  - Theme A: 0.56, Theme B: 0.55 → Gap 0.01 → BORDERLINE
  - Theme A: 0.58, Theme B: 0.48 → Gap 0.10 → CONFIDENT

**AssignmentStatus Enum:**

- `CONFIDENT`: High similarity + clear gap
- `BORDERLINE`: Moderate similarity or small gap
- `MISCELLANEOUS`: Low similarity to all themes

**Code Location:** `ThemeAssigner` class in `theme_assignment.py`

---

### Phase 5: Quality Assurance (`theme_quality.py`)

**Purpose:** Post-assignment refinement and quality validation

**IMPORTANT:** Redundant theme merging now happens in **Phase 3.5** (before assignment), not here!

**QA Steps (in order):**

#### Step 1: Dissolve Tiny Themes

```python
min_messages_per_theme = 5  # Configurable
```

- Themes with < 5 messages are dissolved
- Messages moved to miscellaneous
- **Rationale:** Tiny themes likely noise or over-specific
- **Why after assignment?** Need message counts to identify tiny themes

#### Step 2: Reassign Borderline Messages

```python
if miscellaneous_rate > 15%:
    reassign_borderline_messages()  # Only BORDERLINE, not weak misc!
```

- Check if misc rate is high (> 15%)
- Reassign **ONLY BORDERLINE messages** (similarity ≥ 0.41, failed confidence gap)
- **MISCELLANEOUS messages stay misc** (similarity < 0.41, genuinely weak match)
- **Rationale:** Recover messages stuck in limbo without polluting theme quality

**Design Decision: Why 15% Threshold?**

- 10-15% miscellaneous is healthy (genuine outliers)
- > 15% suggests too conservative assignment
  >
- Borderline reassignment recovers these without lowering quality bar

**Design Decision: Why Not Reassign Weak Misc?**

- **Weak misc** = MISCELLANEOUS with similarity 0.30-0.40
- These are genuinely weak matches to all themes
- Forcing them into themes would:
  - ❌ Pollute theme coherence
  - ❌ Add noise to clean clusters
  - ❌ Make themes harder to interpret
- **Only reassign BORDERLINE** (≥ 0.41) = messages that were close but failed confidence gap

#### Step 3: Coherence Analysis

For each theme, compute:

1. **Coefficient of Variation (CV)**

   ```python
   pairwise_similarities = cosine_similarity(cluster_embeddings)
   cv = std(similarities) / mean(similarities)
   ```

   - Lower CV = more coherent
   - Measures consistency of message similarity
2. **IQR-Based Outlier Detection**

   ```python
   distances = 1 - cosine_similarity(messages, centroid)
   Q1, Q3 = percentiles(distances, [25, 75])
   IQR = Q3 - Q1
   outlier_threshold = Q3 + 1.5 * IQR
   outlier_fraction = count(distances > threshold) / total
   ```

   - Identifies messages far from cluster center
   - High outlier fraction = low coherence
3. **Cluster Coherence Score**

   ```python
   coherence = mean(pairwise_similarities) * (1 - std(similarities))
   ```

   - Combines mean similarity and consistency
   - Range: 0.0-1.0 (higher = better)

**Design Decision: Why Multiple Metrics?**

- No single metric captures all quality aspects
- CV: Spread/consistency
- IQR: Outlier presence
- Coherence: Overall quality
- **Reported but not filtered** - user interprets results

#### Step 4: Silhouette Analysis

- Global metric across all clusters
- Validates separation between themes
- Low silhouette OK for theme clustering (soft boundaries)

**Code Location:** `QualityAssessor` class in `theme_quality.py`

**Note:** Phase 5 used to include redundant theme merging, but that now happens in **Phase 3.5** (before assignment) for efficiency.

---

## Design Decisions

### 1. Why Not Traditional Clustering (HDBSCAN/KMeans)?

**Traditional clustering issues:**

- Hard boundaries (message must pick one cluster)
- Sensitive to hyperparameters (epsilon, min_samples)
- Post-hoc labeling often inaccurate
- No semantic understanding during clustering

**Our approach:**

- Soft boundaries via similarity thresholds
- Themes discovered with semantic understanding
- Labels intrinsic to discovery process
- Flexible assignment based on confidence

### 2. Why Multiple Key Phrases?

**Single label approach:**

```
Theme: "Order Tracking"
Problem: Miss variations like "delivery status", "package location"
```

**Multiple key phrases:**

```
Theme: "Order Tracking"
Key phrases:
  - "track my order"
  - "order status"
  - "where is my package"
  - "delivery tracking"
  - "shipment location"
```

**Benefits:**

- Captures linguistic variation
- Better coverage (more messages assigned)
- Reduces false negatives

### 3. Why Adaptive Sampling?

**Fixed sample (e.g., 500):**

- Too small for large datasets (miss rare themes)
- Too large for small datasets (overfitting, waste)

**Adaptive sampling:**

- Small datasets (< 100): Use all
- Medium (100-500): 60% sample
- Large (500-1500): 20% sample
- Very large (> 1500): Fixed 350

**Rationale:**

- Theme discovery saturates around 300-400 samples
- Stratification ensures diversity
- Cost-effective without sacrificing quality

### 4. Why Merge Before Assignment? (Phase 3.5)

**Problem with old flow:**

```
Discover → Embed → Assign → Merge → Reassign
```

- Waste computation assigning messages to themes that get merged
- Have to update all assignments after merging
- Inefficient pipeline

**New flow:**

```
Discover → Embed → Merge → Assign
```

**Benefits:**

- ✅ Only assign to final set of themes
- ✅ No wasted computation
- ✅ No assignment updates needed
- ✅ More efficient pipeline

**Merge Strategy:**

- Without assignments yet, keep theme with lower ID (arbitrary but deterministic)
- For truly redundant themes, which one we keep doesn't matter
- Key phrases from both themes are combined

### 5. Why Borderline Reassignment (Not Weak Misc)?

**User requirement:** "No threshold relaxation based on misc size"

**Problem:** Conservative assignment → high miscellaneous rate

**Solution:**

- Keep strict thresholds during assignment (0.52 primary, 0.07 gap)
- IF misc > 15%: Reassign **ONLY BORDERLINE messages** (≥ 0.41)
- Do **NOT** reassign weak misc (< 0.41)

**Borderline vs Weak Misc:**

```
BORDERLINE (≥0.41): Failed confidence gap, close match → CAN reassign
MISCELLANEOUS (<0.41): Genuinely weak match → STAY in misc
```

**Rationale:**

- Maintains quality bar during assignment
- Recovers messages that were "close" but failed confidence gap
- Doesn't pollute themes with genuinely weak matches
- User control: Can adjust 15% threshold

### 6. Why Embedding Caching?

**Problem:** Re-embedding 7,000 messages costs ~$0.10 and 30 seconds

**Solution:**

- Save embeddings to `.npy` file
- Cache key: `{client}_{message_count}.npy`
- Load if exists, generate if missing

**Rationale:**

- Iterative development (adjust thresholds, rerun)
- Embeddings don't change between runs
- 30-second → instant rerun

### 7. Why Minimal Preprocessing?

**Traditional NLP pipeline:**

```
Tokenize → Lowercase → Remove stopwords → Lemmatize → Embed
```

**Our pipeline:**

```
Detect language → Normalize unicode → Embed
```

**Rationale:**

- Modern embeddings (OpenAI) trained on natural text
- Stopword removal loses context ("not good" → "good")
- Lemmatization loses nuance ("running" vs "ran")
- Less preprocessing = better semantic preservation

---

## Configuration System

### Centralized Configuration (`theme_config.py`)

All thresholds and parameters in one place:

```python
@dataclass
class DiscoveryConfig:
    messages_per_gpt_call: int = 200     # Messages per GPT batch
    min_themes: int = 5                  # Min themes per batch
    max_themes: int = 12                 # Max themes per batch
    target_final_themes: int = 18        # Target after consolidation
    key_phrases_per_theme: int = 10      # Target 6-10 phrases

@dataclass
class AssignmentConfig:
    primary_threshold: float = 0.52      # Minimum similarity for assignment
    confidence_gap: float = 0.07         # Minimum gap vs second-best
    borderline_threshold: float = 0.41   # Minimum for borderline

@dataclass
class QualityConfig:
    min_messages_per_theme: int = 5                    # Dissolve smaller themes
    redundancy_similarity_threshold: float = 0.82      # Merge candidate threshold
    cv_coherence_threshold: float = 0.1                # Coherence CV threshold
    iqr_multiplier: float = 0.5                        # Outlier detection

@dataclass
class ThemeClusteringConfig:
    assignment: AssignmentConfig
    quality: QualityConfig
    discovery: DiscoveryConfig
    sampling: SamplingConfig
    embedding: EmbeddingConfig
    output: OutputConfig
```

**Usage:**

```python
# Use defaults
config = None

# Override specific values
from new_clustering.theme_config import (
    ThemeClusteringConfig, AssignmentConfig
)
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(primary_threshold=0.55),
    discovery=DiscoveryConfig(max_themes=10, target_final_themes=15)
)
```

---

## Quality Assurance

### Metrics Reported

1. **Coverage:** % of messages assigned to themes

   - Target: 70-85%
   - < 70%: Too conservative
   - > 90%: Possibly over-assigning
     >
2. **Miscellaneous Rate:** % in misc bucket

   - Healthy: 10-20%
   - Genuine outliers and noise
3. **Average Coherence:** Mean coherence across themes

   - > 0.4: Excellent
     >
   - 0.3-0.4: Good
   - < 0.3: Review themes
4. **Silhouette Score:** Global cluster separation

   - > 0.3: Well-separated
     >
   - 0.1-0.3: Acceptable for theme clustering
   - < 0.1: Overlapping themes (may be OK)
5. **Borderline Reassigned:** Count of recovered messages

   - Shows effectiveness of reassignment
   - High count indicates initial assignment was conservative

### Quality Report Example

```json
{
  "total_messages": 7075,
  "assigned_messages": 5959,
  "miscellaneous_messages": 1116,
  "coverage_percent": 84.23,
  "num_themes": 23,
  "redundant_themes_merged": 1,
  "tiny_themes_dissolved": 0,
  "borderline_reassigned": 3334,
  "avg_coherence": 0.3662,
  "silhouette_score": 0.0702
}
```

---

## Code Structure

### Module Organization

```
new_clustering/
├── theme_config.py           # Centralized configuration
├── theme_models.py           # Data classes (Theme, MessageAssignment, etc.)
├── theme_prompts.py          # GPT prompt templates
├── theme_preprocessing.py    # Phase 0: Preprocessing
├── theme_sampling.py         # Phase 1: Sampling
├── theme_discovery.py        # Phase 2: Theme discovery
├── theme_embedding.py        # Phase 3: Embedding
├── theme_assignment.py       # Phase 4: Assignment
├── theme_quality.py          # Phase 5: Quality assurance
├── theme_clustering.py       # Main pipeline orchestration
├── theme_html_report.py      # HTML report generation
└── __init__.py               # Package exports
```

### Class Hierarchy

```
ThemeClusteringPipeline (orchestrator)
  ├── ThemePreprocessor
  ├── ThemeSampler
  ├── ThemeDiscoverer
  ├── ThemeEmbedder
  ├── ThemeAssigner
  └── QualityAssessor
```

### Data Flow

```
Input:
  List[str] messages
  List[str] session_ids

→ Preprocessing:
  List[str] cleaned_messages
  Dict[str, List[str]] message_to_sessions

→ Embedding:
  np.ndarray message_embeddings (N x 1024)

→ Sampling:
  List[str] sampled_messages (100-350)

→ Discovery:
  List[Theme] themes (15-35)

→ Embedding:
  themes with phrase_embeddings and theme_embedding

→ Assignment:
  List[MessageAssignment] assignments

→ Quality:
  refined themes + assignments
  QualityReport

Output:
  PipelineOutput
    ├── clusters: List[ClusterResult]
    ├── themes: List[Theme]
    ├── quality_report: QualityReport
    └── miscellaneous: ClusterResult
```

---

## Usage Guide

### Basic Usage

```python
from new_clustering import run_pipeline

# Your data
messages = ["where is my order", "i want to return", ...]
session_ids = ["sess_1", "sess_2", ...]

# Run pipeline
output = run_pipeline(
    messages=messages,
    session_ids=session_ids,
    client_name="my_client",
    save_outputs=True
)

# Access results
for cluster in output.clusters:
    print(f"{cluster.theme.theme_name}: {cluster.message_count} messages")
```

### With Filters

```python
# In main() function in theme_clustering.py

FILTER_PAGES = ["checkout", "product"]     # Only these pages
FILTER_KEYWORD = "order"                   # Pages containing "order"
FILTER_USECASES = ["support", "inquiry"]   # Specific usecases
FILTER_PHRASE = "I recommend"              # Messages before AI says this
REMOVE_BUBBLES = True                      # Remove repeated bubble clicks
MAX_MESSAGES_LOAD = 5000                   # Sampling cap during load
```

### With Seed Themes

```python
# Ensure specific themes are included
SEED_THEMES = [
    "Order Tracking",
    "Product Returns",
    "Payment Issues"
]

output = run_pipeline(
    messages=messages,
    session_ids=session_ids,
    seed_themes=SEED_THEMES,  # Will always be in results
    client_name="my_client"
)
```

### Custom Configuration

```python
from new_clustering.theme_config import (
    ThemeClusteringConfig,
    AssignmentConfig,
    QualityConfig
)

config = ThemeClusteringConfig(
    assignment=AssignmentConfig(
        primary_threshold=0.60,    # More conservative
        confidence_gap=0.10        # Larger gap required
    ),
    quality=QualityConfig(
        min_messages_per_theme=10  # Larger themes only
    )
)

output = run_pipeline(
    messages=messages,
    session_ids=session_ids,
    config=config
)
```

### Generate HTML Report

```python
# Automatic (if save_outputs=True)
output = run_pipeline(..., save_outputs=True)
# HTML saved to: new_clustering/outputs/report_{client}_{timestamp}.html

# Manual from JSON files
from new_clustering.theme_html_report import generate_report_from_files

generate_report_from_files(
    clusters_file="path/to/clusters.json",
    quality_file="path/to/quality_report.json",
    output_file="path/to/report.html",
    client_name="My Client"
)
```

---

## Performance Characteristics

### Time Complexity

| Phase           | Operation                         | Complexity        | Time (7k messages)           |
| --------------- | --------------------------------- | ----------------- | ---------------------------- |
| Preprocessing   | Language detection, normalization | O(N * M)          | ~30 sec                      |
| Embedding       | API calls (batched)               | O(N / batch_size) | ~30 sec (cached: instant)    |
| Sampling        | Mini-KMeans                       | O(N * k * d * i)  | ~5 sec                       |
| Discovery       | GPT calls                         | O(S / 100)        | ~20 sec                      |
| Theme Embedding | API calls                         | O(T * P)          | ~5 sec                       |
| Assignment      | Similarity computation            | O(N * T * P)      | ~10 sec                      |
| Quality         | Various metrics                   | O(N * d)          | ~10 sec                      |
| **Total** |                                   |                   | **~2 min (first run)** |
| **Total** |                                   |                   | **~45 sec (cached)**   |

Where:

- N = number of messages
- M = avg message length
- k = number of mini-clusters
- d = embedding dimensions (1024)
- i = KMeans iterations
- S = sample size
- T = number of themes
- P = phrases per theme

### Cost (OpenAI API)

**Embeddings:**

- text-embedding-3-large: $0.13 per 1M tokens
- ~100 tokens per message average
- 7,000 messages = 700k tokens
- Cost: ~$0.10

**GPT-4.1:**

- ~$3 per 1M input tokens, ~$15 per 1M output
- 350 sample messages = ~35k tokens input
- Response = ~5k tokens output
- Cost: ~$0.18

**Total: ~$0.28 per 7k messages**

- Cached reruns: $0.18 (embedding reused)

---

## Future Enhancements

### Potential Improvements

1. **Hierarchical Themes**

   - Parent-child theme relationships
   - Drill-down capability in UI
2. **Temporal Analysis**

   - Track theme evolution over time
   - Identify trending themes
3. **Multi-Language Support**

   - Better handling of code-switching
   - Language-specific preprocessing
4. **Active Learning**

   - User feedback on assignments
   - Iterative refinement
5. **Theme Merging Suggestions**

   - Prompt user before auto-merge
   - Interactive threshold tuning
6. **Message Influence Scoring**

   - Which messages most define a theme
   - Prototype messages for each theme

---

## Troubleshooting

### Low Coverage (< 70%)

**Causes:**

- Thresholds too strict
- Themes too specific
- Dataset very diverse

**Solutions:**

1. Lower `primary_threshold` (0.52 → 0.48)
2. Lower `confidence_gap` (0.07 → 0.05)
3. Lower `borderline_threshold` (0.41 → 0.38)
4. Check if borderline reassignment triggered
5. Review sample quality and key phrase coverage

### Too Many Themes (> 40)

**Causes:**

- Sample too large
- Dataset very granular

**Solutions:**

1. Reduce `max_themes` in DiscoveryConfig
2. Increase `redundancy_similarity_threshold`
3. Review GPT prompt for specificity

### Poor Coherence (< 0.25)

**Causes:**

- Over-assignment (too lenient)
- Themes not well-defined
- Dataset naturally noisy

**Solutions:**

1. Increase `primary_threshold`
2. Check sample quality (is it representative?)
3. Review theme descriptions
4. Consider more aggressive merging

### High Miscellaneous (> 25%)

**Causes:**

- Thresholds too conservative
- Borderline reassignment not triggering
- Genuinely diverse dataset

**Solutions:**

1. Lower reassignment threshold (15% → 20%)
2. Check borderline count in quality report
3. Review misc messages (are they assignable?)

---

## References

### Key Papers & Concepts

1. **Semantic Similarity:** Cosine similarity in high-dimensional embedding spaces
2. **GPT-based Labeling:** Using LLMs for theme discovery and naming
3. **Stratified Sampling:** Ensuring diverse representation in samples
4. **Union-Find Algorithm:** Efficient equivalence class merging
5. **Silhouette Analysis:** Cluster quality validation

### Related Work

- **vector.py:** Inspiration for filtering and embedding caching
- **uf_option_generator.py:** Merging logic and union-find approach
- **intent_expansion:** Coherence metrics (CV, IQR, silhouette)
- **generate_html_report.py:** HTML report structure and styling

---

## Conclusion

This pipeline represents a **theme-first** approach to conversation clustering that combines:

- **Human-like understanding** (GPT theme discovery)
- **Scalable matching** (embedding similarity)
- **Quality assurance** (coherence, merging, reassignment)

The result is **actionable themes** that accurately represent customer conversation patterns with minimal manual tuning.

---

**Version:** 1.1
**Last Updated:** February 6, 2026
**Maintainer:** Admin Backend Team

**Key Changes in v1.1:**

- Added Phase 3.5: Pre-assignment redundant theme merging
- Removed weak misc reassignment (only BORDERLINE reassignment now)
- Updated thresholds: primary=0.52, gap=0.07, borderline=0.41
- Enhanced key phrase generation (6-10 phrases with abbreviation expansion)
