# Theme Clustering Pipeline - Thresholds & Configuration Guide

This guide explains all configurable thresholds in the theme clustering pipeline and how they affect results.

---

## Quick Reference: Main Thresholds

| Threshold | Default | Impact | To Reduce Misc | To Improve Merging |
|-----------|---------|--------|----------------|-------------------|
| `primary_threshold` | 0.55 | Assignment strictness | ↓ Lower (0.50-0.52) | - |
| `confidence_gap` | 0.08 | Ambiguity tolerance | ↓ Lower (0.05-0.06) | - |
| `borderline_threshold` | 0.50 | Reassignment zone | ↓ Lower (0.45) | - |
| `redundancy_similarity_threshold` | 0.80 | Merge sensitivity | - | ↓ Lower (0.75-0.78) |
| `min_messages_per_theme` | 5 | Tiny theme filter | ↓ Lower (3) | - |
| `key_phrases_per_theme` | 8 | Phrase coverage | ↑ Increase (10-12) | - |
| `target_final_themes` | 20 | Theme count target | - | ↓ Lower (15-18) |
| `max_themes` | 15 | Batch theme limit | ↑ Increase (18-20) | ↓ Lower (12) |

---

## Assignment Thresholds (Phase 4)

### 1. Primary Threshold (`primary_threshold`)

**Location:** `AssignmentConfig.primary_threshold`
**Default:** 0.55
**Range:** 0.40 - 0.65

**What it does:**
- Minimum cosine similarity between a message and a theme for confident assignment
- Messages below this threshold are marked as BORDERLINE or MISCELLANEOUS

**Effect on results:**
- **Higher (0.58-0.65):** More strict assignment → Higher misc rate, cleaner themes
- **Lower (0.48-0.52):** More lenient assignment → Lower misc rate, potentially noisier themes

**When to adjust:**
```python
# High misc rate (>25%)? Lower the threshold
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(primary_threshold=0.50)
)

# Poor theme coherence? Raise the threshold
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(primary_threshold=0.58)
)
```

**Trade-off:**
- ⬇️ Lower threshold = More coverage, less precision
- ⬆️ Higher threshold = Less coverage, more precision

---

### 2. Confidence Gap (`confidence_gap`)

**Location:** `AssignmentConfig.confidence_gap`
**Default:** 0.08
**Range:** 0.04 - 0.12

**What it does:**
- Minimum difference between best and second-best theme similarity
- Prevents assigning messages that could fit multiple themes equally well

**Example:**
```
Message X similarities:
- Theme A: 0.56
- Theme B: 0.55
Gap = 0.01 < 0.08 → BORDERLINE (ambiguous)

Message Y similarities:
- Theme A: 0.58
- Theme B: 0.48
Gap = 0.10 > 0.08 → CONFIDENT (clear winner)
```

**Effect on results:**
- **Higher (0.10-0.12):** Fewer assignments, only very clear matches
- **Lower (0.04-0.06):** More assignments, accepts closer competitions

**When to adjust:**
```python
# Many borderline messages? Lower the gap
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(confidence_gap=0.05)
)

# Ambiguous assignments? Raise the gap
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(confidence_gap=0.10)
)
```

**Trade-off:**
- ⬇️ Lower gap = More assignments, potential ambiguity
- ⬆️ Higher gap = Fewer assignments, clearer boundaries

---

### 3. Borderline Threshold (`borderline_threshold`)

**Location:** `AssignmentConfig.borderline_threshold`
**Default:** 0.50
**Range:** 0.40 - 0.55

**What it does:**
- Messages scoring between `borderline_threshold` and `primary_threshold` are marked BORDERLINE
- BORDERLINE messages get reassigned if misc > 15%

**Effect on results:**
- **Higher (0.52-0.55):** More messages eligible for reassignment
- **Lower (0.42-0.48):** Only very weak matches get reassigned

**When to adjust:**
```python
# Want to recover more borderline messages?
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(borderline_threshold=0.45)
)
```

**Trade-off:**
- ⬇️ Lower threshold = Stricter reassignment criteria
- ⬆️ Higher threshold = More aggressive reassignment

---

## Quality Assurance Thresholds (Phase 5)

### 4. Min Messages Per Theme (`min_messages_per_theme`)

**Location:** `QualityConfig.min_messages_per_theme`
**Default:** 5
**Range:** 3 - 10

**What it does:**
- Themes with fewer than this many messages are dissolved
- Messages moved to miscellaneous bucket

**Effect on results:**
- **Higher (7-10):** Removes more tiny themes, higher misc rate
- **Lower (3-4):** Keeps more themes, potentially including noise

**When to adjust:**
```python
# Want to keep smaller themes?
config = ThemeClusteringConfig(
    quality=QualityConfig(min_messages_per_theme=3)
)

# Want only substantial themes?
config = ThemeClusteringConfig(
    quality=QualityConfig(min_messages_per_theme=8)
)
```

**Trade-off:**
- ⬇️ Lower threshold = More themes (including tiny ones)
- ⬆️ Higher threshold = Fewer themes (only substantial ones)

---

### 5. Redundancy Similarity Threshold (`redundancy_similarity_threshold`)

**Location:** `QualityConfig.redundancy_similarity_threshold`
**Default:** 0.80
**Range:** 0.70 - 0.90

**What it does:**
- Themes with centroid similarity > this threshold are merged
- Uses 85th percentile of all pairwise similarities as adaptive threshold

**Effect on results:**
- **Higher (0.85-0.90):** Less merging, keeps similar themes separate
- **Lower (0.72-0.78):** More aggressive merging, fewer final themes

**When to adjust:**
```python
# Too many redundant themes? Lower threshold
config = ThemeClusteringConfig(
    quality=QualityConfig(redundancy_similarity_threshold=0.75)
)

# Themes being merged incorrectly? Raise threshold
config = ThemeClusteringConfig(
    quality=QualityConfig(redundancy_similarity_threshold=0.85)
)
```

**Trade-off:**
- ⬇️ Lower threshold = Fewer themes, risk of over-merging
- ⬆️ Higher threshold = More themes, risk of redundancy

---

## Discovery Thresholds (Phase 2)

### 6. Key Phrases Per Theme (`key_phrases_per_theme`)

**Location:** `DiscoveryConfig.key_phrases_per_theme`
**Default:** 8
**Range:** 6 - 12

**What it does:**
- Target number of key phrases GPT generates per theme
- More phrases = wider semantic net for matching

**Effect on results:**
- **Higher (10-12):** Better coverage, more variations captured
- **Lower (6-7):** Tighter themes, potentially miss edge cases

**When to adjust:**
```python
# Messages not matching themes well? Increase phrases
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(key_phrases_per_theme=10)
)
```

**Trade-off:**
- ⬆️ More phrases = Better coverage, potential overlap
- ⬇️ Fewer phrases = Tighter themes, potential gaps

---

### 7. Target Final Themes (`target_final_themes`)

**Location:** `DiscoveryConfig.target_final_themes`
**Default:** 20
**Range:** 12 - 30

**What it does:**
- Target number of themes after consolidation
- Pipeline iteratively consolidates until reaching this target

**Effect on results:**
- **Higher (25-30):** More granular themes, less consolidation
- **Lower (12-18):** Broader themes, more consolidation

**When to adjust:**
```python
# Want fewer, broader themes?
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(target_final_themes=15)
)

# Want more granular themes?
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(target_final_themes=25)
)
```

**Trade-off:**
- ⬇️ Lower target = Fewer, broader themes
- ⬆️ Higher target = More, granular themes

---

### 8. Min/Max Themes Per Batch (`min_themes`, `max_themes`)

**Location:** `DiscoveryConfig.min_themes` / `DiscoveryConfig.max_themes`
**Default:** 5 / 15
**Range:** 3-8 / 10-20

**What it does:**
- GPT generates between min and max themes per batch of 200 messages
- Constrains theme explosion during discovery

**Effect on results:**
- **Higher max (18-20):** More themes per batch, more consolidation needed
- **Lower max (10-12):** Fewer themes per batch, more consistent granularity

**When to adjust:**
```python
# Getting too many themes? Lower max_themes
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(max_themes=12)
)

# Missing themes? Raise max_themes
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(max_themes=18)
)
```

**Trade-off:**
- Lower max = More controlled discovery, risk of missing themes
- Higher max = More comprehensive discovery, more consolidation needed

---

### 9. Min Key Phrases (`min_key_phrases`)

**Location:** `DiscoveryConfig.min_key_phrases`
**Default:** 4
**Range:** 3 - 6

**What it does:**
- Minimum acceptable key phrases for a theme to be valid
- Themes with fewer phrases are rejected

**Effect on results:**
- **Higher (5-6):** Only themes with rich phrase sets
- **Lower (3):** Accept themes with minimal phrases

**When to adjust:**
```python
# Want stricter quality control?
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(min_key_phrases=5)
)
```

---

## Sampling Thresholds (Phase 1)

### 10. Sample Size Caps

**Location:** `SamplingConfig.medium_sample_cap`, `large_sample_cap`, `very_large_sample_cap`
**Defaults:** 500 / 650 / 800

**What they do:**
- Maximum messages sampled for theme discovery
- Larger samples = better coverage but higher cost

**When to adjust:**
```python
# Large dataset, want better coverage?
config = ThemeClusteringConfig(
    sampling=SamplingConfig(
        large_sample_cap=750,
        very_large_sample_cap=1000
    )
)
```

---

## Common Configuration Scenarios

### Scenario 1: High Miscellaneous Rate (>25%)

**Problem:** Too many messages not assigned to themes

**Solution:**
```python
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(
        primary_threshold=0.50,        # ↓ Lower threshold
        confidence_gap=0.06,           # ↓ Accept closer matches
        borderline_threshold=0.45      # ↓ More reassignment
    ),
    discovery=DiscoveryConfig(
        key_phrases_per_theme=10,      # ↑ More phrases per theme
        max_themes=18                  # ↑ Discover more themes
    )
)
```

---

### Scenario 2: Too Many Themes (>30)

**Problem:** Theme explosion, many similar themes

**Solution:**
```python
config = ThemeClusteringConfig(
    discovery=DiscoveryConfig(
        max_themes=12,                 # ↓ Fewer themes per batch
        target_final_themes=15,        # ↓ Aggressive consolidation
        always_consolidate=True
    ),
    quality=QualityConfig(
        redundancy_similarity_threshold=0.75,  # ↓ More aggressive merging
        min_messages_per_theme=8       # ↑ Remove tiny themes
    )
)
```

---

### Scenario 3: Poor Theme Coherence

**Problem:** Themes contain unrelated messages

**Solution:**
```python
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(
        primary_threshold=0.58,        # ↑ Stricter assignment
        confidence_gap=0.10            # ↑ Clear winners only
    ),
    quality=QualityConfig(
        min_messages_per_theme=6       # ↑ Remove noise clusters
    )
)
```

---

### Scenario 4: Themes Being Merged Incorrectly

**Problem:** Distinct themes getting consolidated together

**Solution:**
```python
config = ThemeClusteringConfig(
    quality=QualityConfig(
        redundancy_similarity_threshold=0.85,  # ↑ Less aggressive merging
    ),
    discovery=DiscoveryConfig(
        target_final_themes=25,        # ↑ Keep more themes
        max_themes=15                  # Balanced discovery
    )
)
```

---

## Threshold Interactions

**Important:** Thresholds interact with each other:

1. **Assignment Thresholds Work Together:**
   - Lower `primary_threshold` + Lower `confidence_gap` = More assignments
   - Higher `primary_threshold` + Higher `confidence_gap` = Fewer, cleaner assignments

2. **Discovery vs Quality:**
   - High `max_themes` + Low `target_final_themes` = Discover many, consolidate aggressively
   - Low `max_themes` + High `target_final_themes` = Discover few, consolidate less

3. **Phrases vs Assignment:**
   - More `key_phrases_per_theme` + Lower `primary_threshold` = Maximum coverage
   - Fewer `key_phrases_per_theme` + Higher `primary_threshold` = Precision focus

---

## Recommended Starting Points

### Conservative (High Precision)
```python
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(
        primary_threshold=0.58,
        confidence_gap=0.10,
        borderline_threshold=0.52
    ),
    discovery=DiscoveryConfig(
        key_phrases_per_theme=8,
        target_final_themes=18,
        max_themes=12
    ),
    quality=QualityConfig(
        redundancy_similarity_threshold=0.82,
        min_messages_per_theme=6
    )
)
```

### Balanced (Default)
```python
config = ThemeClusteringConfig()  # Use defaults
```

### Aggressive (Maximum Coverage)
```python
config = ThemeClusteringConfig(
    assignment=AssignmentConfig(
        primary_threshold=0.50,
        confidence_gap=0.06,
        borderline_threshold=0.45
    ),
    discovery=DiscoveryConfig(
        key_phrases_per_theme=10,
        target_final_themes=22,
        max_themes=18
    ),
    quality=QualityConfig(
        redundancy_similarity_threshold=0.75,
        min_messages_per_theme=4
    )
)
```

---

## Iterative Tuning Process

1. **Run with defaults** → Analyze coverage & coherence
2. **Identify main issue:**
   - High misc? → Lower assignment thresholds
   - Too many themes? → Lower consolidation targets
   - Poor coherence? → Raise assignment thresholds
3. **Adjust 1-2 thresholds** → Re-run
4. **Repeat** until satisfactory

**Tip:** Change one threshold at a time to understand its effect.

---

## Monitoring Metrics

After each run, check:

- **Coverage:** Target 75-85%
- **Miscellaneous:** Target 10-20%
- **Num Themes:** Target 15-25
- **Avg Coherence:** Target >0.30
- **Silhouette:** Target >0.05

Use these to guide threshold adjustments.
