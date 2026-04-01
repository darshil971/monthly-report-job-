"""
Theme Clustering GPT Prompts

Prompt templates for theme discovery and consolidation using GPT.
"""

from typing import List, Optional, Dict


def get_cross_batch_merge_pairs_prompt(
    themes_batch_1: List[dict],
    themes_batch_2: List[dict],
) -> str:
    """
    Generate a prompt that asks the LLM to output ONLY the pairs of cross-batch duplicates.

    This is deliberately minimal in output format:
    - The LLM outputs only the pairs to merge (not all N themes)
    - If no duplicates → returns []
    - The caller does all the merging programmatically

    This prevents:
    - Token overflow (output is tiny regardless of input size)
    - Silent theme dropping (we keep all themes not named in a pair)
    - Catastrophic over-merging (pairs output is verifiable and conservative)
    """
    import json

    b1_compact = [
        {"name": t.get("theme_name", ""), "description": t.get("description", ""),
         "phrases": t.get("key_phrases", [])[:4]}
        for t in themes_batch_1
    ]
    b2_compact = [
        {"name": t.get("theme_name", ""), "description": t.get("description", ""),
         "phrases": t.get("key_phrases", [])[:4]}
        for t in themes_batch_2
    ]

    b1_json = json.dumps(b1_compact, indent=2, ensure_ascii=False)
    b2_json = json.dumps(b2_compact, indent=2, ensure_ascii=False)

    b1_names = [t.get("theme_name", "") for t in themes_batch_1]
    b2_names = [t.get("theme_name", "") for t in themes_batch_2]

    prompt = f'''You are reviewing two sets of customer message themes discovered from two
separate halves of the same corpus.

YOUR ONLY JOB: identify which themes are TRUE CROSS-BATCH DUPLICATES — the same
customer intent discovered independently in both batches, just with different names.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BATCH 1 THEMES ({len(b1_compact)} themes):
{b1_json}

BATCH 2 THEMES ({len(b2_compact)} themes):
{b2_json}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT: Return ONLY a JSON array of pairs where each pair names one theme from
Batch 1 and one theme from Batch 2 that represent IDENTICAL customer intent.
If no duplicates exist, return an empty array: []

WHAT QUALIFIES AS A DUPLICATE PAIR:
  ✓ Same customer intent, just different vocabulary
    e.g. "COD Payment" ↔ "Cash on Delivery"
    e.g. "Order Tracking" ↔ "Track My Shipment"
    e.g. "Acne Concerns" ↔ "Pimple Treatment"
  ✓ One is clearly a renamed/rephrased version of the other

WHAT DOES NOT QUALIFY — DO NOT INCLUDE:
  ✗ Themes that are related but distinct (e.g. "Delivery ETA" ≠ "Order Cancellation")
  ✗ Themes that overlap partially but cover different sub-intents
  ✗ Any pair you are not fully confident about → leave it out

DEFAULT: When in doubt, DO NOT include the pair. The cost of a missed duplicate
is a slightly redundant theme. The cost of a wrong pair is destroying two distinct
themes. Err on the side of caution.

VALID THEME NAMES FROM BATCH 1: {json.dumps(b1_names)}
VALID THEME NAMES FROM BATCH 2: {json.dumps(b2_names)}

Use the exact theme names as they appear above.

OUTPUT FORMAT (return ONLY this JSON, nothing else):
```json
[
  {{"batch1": "Exact theme name from Batch 1", "batch2": "Exact theme name from Batch 2"}},
  ...
]
```
If no duplicates: return []'''

    return prompt


def get_theme_discovery_prompt(
    messages: List[str],
    seed_themes: Optional[List[str]] = None,
    min_themes: int = 5,
    max_themes: int = 25,
    key_phrases_per_theme: int = 6,
) -> str:
    """
    Generate the prompt for theme discovery from messages.

    Args:
        messages: List of customer message texts
        seed_themes: Optional list of seed theme names to include
        min_themes: Minimum expected themes
        max_themes: Maximum allowed themes
        key_phrases_per_theme: Target number of key phrases per theme

    Returns:
        System prompt string for GPT
    """
    # Format messages for the prompt
    message_list = "\n".join([f"- {msg[:200]}" for msg in messages])

    seed_instruction = ""
    if seed_themes:
        seed_list = ", ".join([f'"{t}"' for t in seed_themes])
        seed_instruction = f"""
MANDATORY SEED THEMES:
You MUST include these themes in your output: {seed_list}
Generate key phrases for each seed theme, then discover additional themes from the messages.
Do NOT merge or consolidate seed themes with other themes.
"""

    prompt = f'''You are a Voice of Customer (VoC) analyst. Your job is to read a batch
of customer messages from an e-commerce chatbot and produce a set of
**themes**, each with **key phrases** that act as the theme's semantic
fingerprint.

Downstream, every message in the corpus will be assigned to a theme by
computing cosine similarity between the message embedding and the
embeddings of each theme's key phrases. Themes that are too broad will
attract unrelated messages; themes that are too narrow will leave most
messages unassigned. Your goal is the sweet spot: each theme captures
one **distinct, business-actionable customer intent or concern**.

─────────────────────────────────────────────────
A. WHY THESE THEMES MATTER (business context)
─────────────────────────────────────────────────

These themes will be used to:

1. **Cohort conversion analytics** — e.g. "users who asked about
   shipping converted at 12 %; users who asked about returns converted
   at 4 %." This only works if "shipping" and "returns" are separate
   themes.
2. **Customer language / SEO insights** — the key phrases tell the
   brand how real customers talk about a topic, informing ad copy,
   product descriptions, and search keywords.
3. **PDP & ad-campaign optimization** — each theme maps to a product
   page section or ad angle the brand can improve.
4. **Trend tracking** — e.g. "pregnancy-safety queries rose from 2 %
   to 9 % in Q2." This only works if "pregnancy safety" is its own
   theme rather than buried inside "product safety."
5. **Chatbot optimization** — identify topics where the bot says
   "I don't have info," surface relevant quick-reply bubbles, and
   detect handoff-worthy intents.
6. **Brand insights** — emergent themes reveal where customer
   interests, expectations, and pain points lie.

Keep these six uses in mind for every decision you make.

CUSTOMER MESSAGES:
{message_list}
{seed_instruction}
─────────────────────────────────────────────────
B. WHEN TO CREATE SEPARATE THEMES vs. MERGE
─────────────────────────────────────────────────

Create a **separate** theme when any of these are true:

- Different customer pain points or needs — a customer worried about
  product safety has a different concern from one asking about dosage.
- Different SEO / ad-campaign keywords — "beard growth serum" and
  "hair fall treatment" would never appear in the same Google ad.
- Different conversion patterns — customers asking about returns
  behave differently from those asking about discounts.
- Different chatbot responses — the bot needs different knowledge to
  answer "how to use" vs. "any side effects."
- Different product features or health conditions — each feature or
  condition is worth tracking on its own.
- High business-value niche — e.g. pregnancy/breastfeeding safety is
  a small but critical segment that brands want to see separately.
- Different purchase-funnel stages — pre-purchase evaluation, checkout
  friction, and post-purchase support are fundamentally different
  moments.

**Merge** into one theme only when:

- Messages express the **exact same** underlying intent, just in
  different words (vocabulary variation, not intent variation).
- Splitting would yield no additional analytical, SEO, or conversion
  insight — the brand would take the same action either way.
- Sub-categories are so tightly coupled that tracking them separately
  fragments the picture without adding value (e.g. "acne marks" +
  "acne inflammation" + "acne treatment" → one "Acne Concerns"
  theme, because the customer problem and product set are the same).

─────────────────────────────────────────────────
C. GRANULARITY CALIBRATION (with examples)
─────────────────────────────────────────────────

Use the examples below to calibrate how specific or broad a theme
should be. The core test is: **would the brand take a different
action for these two groups of messages?** If yes, keep separate.

CORRECT separations (each is its own theme):
  - order tracking & delivery status
  - order cancellation
  - order modification / changes
  - refunds and returns
  - payment issues & methods
  - cash on delivery (COD)
  - discounts and promo codes
  - product availability & purchase channels
  - product comparison & selection
  Reason: each represents a distinct customer action, a distinct
  chatbot flow, and a distinct conversion signal.

CORRECT merges:
  - "acne marks" + "acne inflammation" + "acne treatment"
    → 1 theme: Acne Concerns
  - "dark spots" + "hyperpigmentation" + "melasma"
    → 1 theme: Pigmentation & Dark Spots
  - "customer care number" + "callback request" + "WhatsApp contact"
    → 1 theme: Contact & Callback Requests
  Reason: same underlying customer need, same product set, no
  analytical value in splitting further.

CORRECT separation of seemingly-similar themes:
  - "Product Usage & Application" (how to use, how to apply, kaise
    lagaye) — HOW do I use this?
  - "Dosage & Quantity" (how many capsules, kitni tablet, how much
    oil) — HOW MUCH do I take?
  - "Course Duration & Results Timeline" (kitne din, how many months,
    when will I see results) — HOW LONG until it works?
  Reason: these are three different customer questions. The chatbot
  needs different information to answer each. SEO keywords differ.
  The brand learns different things from each.

  - "Side Effects & Safety" (general safety, any side effects, koi
    problem toh nahi) — is this safe?
  - "Pregnancy & Breastfeeding Safety" (breastfeeding mother, safe
    during pregnancy) — is this safe FOR MY BABY?
  Reason: pregnancy safety is a high-value niche. Brands are often
  surprised by its volume. Always keep it separate.

RED FLAGS that a theme is too broad:
  ✗ Name is vague: "Product Questions," "General Queries."
  ✗ It has 15+ phrases spanning unrelated sub-topics.
  ✗ You cannot describe ONE specific customer concern it represents.
  ✗ The brand cannot take a single targeted action from it.

GREEN FLAGS that a theme is well-formed:
  ✓ Name is specific: "Beard Growth Concerns," "Piles Treatment."
  ✓ 3–10 phrases, all clearly belonging to the same intent.
  ✓ You can state one customer concern it captures.
  ✓ It directly informs one type of business decision.

─────────────────────────────────────────────────
D. KEY PHRASE REQUIREMENTS
─────────────────────────────────────────────────

Each theme needs {key_phrases_per_theme} key phrases (range: 4–12
depending on the topic's breadth). These phrases are the theme's
**semantic anchors** — every message in the corpus will be matched to
themes via embedding similarity to these phrases. The PRIMARY GOAL of
phrases is to MAXIMIZE COVERAGE — every real message that belongs to
this theme should be close to at least one phrase.

1. **Language**: Phrases can be in ANY language that customers use
   (English, Hindi, Hinglish, regional languages). If customers write
   in Hindi/Hinglish, include Hindi/Hinglish phrases alongside English
   ones to maximize embedding matches. The goal is to match how
   customers ACTUALLY write, not to normalize everything to English.

   Example for a beard growth theme:
     "beard growth serum", "dadhi ugane ka tel", "dhari nahi aa rahi",
     "beard nahi aa rahi", "patchy beard solution", "dadhi badhane ka
     upay", "moustache growth oil"

2. **Length**: 2–6 words. Short enough to embed well, long enough to
   carry context.
     Bad:  "track"  (too short, ambiguous)
     Good: "track my order status"
     Good: "mera order kab aayega"

3. **Diversity**: Each phrase should pull in a DIFFERENT subset of
   messages. Cover different facets, synonyms, and phrasings within
   the theme — including across languages.
     Theme: Order Tracking & Delivery
     Bad:  ["track order", "order tracking"] (near-duplicates)
     Good: ["track my order", "order delivery status",
            "where is my package", "when will order arrive",
            "mera order kab aayega", "delivery kitne din me",
            "shipment not received", "order kab tak milega"]

4. **Customer voice**: Phrases should mirror how customers actually
   ask, not how a support agent would categorize it.
     Bad:  "post-purchase delivery inquiry"
     Good: "where is my order"
     Good: "mera order kahan hai"

5. **Abbreviation expansion**: If an abbreviation is common, include
   BOTH the abbreviation form and the expanded form as separate
   phrases.
     "COD payment", "cash on delivery"
     "EMI option", "monthly installment plan"

6. **No specific data**: No order IDs, phone numbers, personal names,
   specific ages, or brand-specific product codes in phrases.

─────────────────────────────────────────────────
E. THEME COUNT & OUTPUT FORMAT
─────────────────────────────────────────────────

Extract {min_themes}-{max_themes} themes from these messages.

Return a JSON array. No text before or after the JSON.

```json
[
  {{
    "theme_name": "Short Specific Name (1–5 words)",
    "description": "One or two polished sentences describing the specific customer concerns, questions, and scenarios this theme covers. Do NOT include importance indicators or business justifications.",
    "key_phrases": [
      "phrase one",
      "phrase two — can be in any language customers use",
      "phrase three",
      "..."
    ],
    "example_messages": [
      "verbatim message from the input that belongs here",
      "another verbatim message from the input"
    ]
  }}
]
```

Field rules:
- theme_name — 1–5 word noun phrase. Specific and descriptive.
- description — a polished, client-facing description of the customer
  concerns and scenarios this theme covers. Must represent all sub-topics
  within the theme. Do NOT include internal importance indicators like
  "Critical for brand trust", "Strong conversion lever", etc.
- key_phrases — {key_phrases_per_theme} phrases (±2), following the
  rules in section D. Include multilingual phrases where customers
  use non-English.
- example_messages — 2–3 messages copied verbatim from the input.

─────────────────────────────────────────────────
F. FINAL CHECKLIST (review before responding)
─────────────────────────────────────────────────

Before you output, walk through these checks:

1. Does every theme represent ONE clear customer intent or concern?
2. Would the brand take a DIFFERENT action for each theme?
3. Are there any themes so broad they would attract unrelated messages?
4. Did I keep high-value niches (e.g. pregnancy safety) separate?
5. Did I avoid merging themes that need different chatbot responses?
6. Are key phrases diverse and multilingual where customers use
   non-English, to maximize message coverage?
7. Did I include {min_themes}–{max_themes} themes?
8. Is the JSON valid (no trailing commas, all strings quoted)?

Return ONLY the JSON array.'''

    return prompt


def get_theme_merge_prompt(
    themes: List[dict],
    candidate_pairs: Optional[List[tuple]] = None,
    seed_themes: Optional[List[str]] = None,
    themes_batch_1: Optional[List[dict]] = None,
    themes_batch_2: Optional[List[dict]] = None,
) -> str:
    """
    Generate prompt for merging/deduplicating themes across two discovery batches.

    Args:
        themes: Combined list of all themes (batch_1 + batch_2)
        candidate_pairs: Optional list of theme pairs to review (unused in holistic mode)
        seed_themes: Seed themes that must be preserved
        themes_batch_1: Themes from first batch (for cross-batch framing)
        themes_batch_2: Themes from second batch (for cross-batch framing)

    Returns:
        System prompt string for GPT
    """
    import json

    seed_instruction = ""
    if seed_themes:
        seed_list = ", ".join([f'"{t}"' for t in seed_themes])
        seed_instruction = f"""
PROTECTED SEED THEMES: {seed_list}
These themes MUST be preserved and CANNOT be merged or consolidated away.
You may add key phrases to them from other similar themes, but the theme itself must remain.
"""

    # Present themes as two labeled batches when batch structure is available —
    # this is the single most important structural signal for the model.
    if themes_batch_1 is not None and themes_batch_2 is not None:
        batch1_json = json.dumps(themes_batch_1, indent=2)
        batch2_json = json.dumps(themes_batch_2, indent=2)
        themes_section = f"""BATCH 1 THEMES — {len(themes_batch_1)} themes (already internally distinct):
```json
{batch1_json}
```

BATCH 2 THEMES — {len(themes_batch_2)} themes (already internally distinct):
```json
{batch2_json}
```"""
    else:
        themes_json = json.dumps(themes, indent=2)
        themes_section = f"""THEMES TO REVIEW:
```json
{themes_json}
```"""

    prompt = f'''You are a Voice of Customer (VoC) analyst. Your task is cross-batch
theme deduplication — identifying themes that were independently discovered in two
separate batches of the same customer message corpus and merging only those
exact duplicates.

These themes are used for conversion analytics, SEO insights, PDP optimization,
trend tracking, chatbot improvement, and brand insights. Every distinct theme
represents a real, recurring customer concern that has analytical value.

═══════════════════════════════════════════════════════════════════════════════
⚠  DEFAULT OPERATING MODE: KEEP SEPARATE
   Merge ONLY when you have unambiguous proof of cross-batch duplication.
   Preservation of granularity is more valuable than a tidier list.
═══════════════════════════════════════════════════════════════════════════════

─────────────────────────────────────────────────
A. THEMES TO REVIEW
─────────────────────────────────────────────────

{themes_section}
{seed_instruction}
─────────────────────────────────────────────────
B. WHAT THIS JOB IS AND IS NOT
─────────────────────────────────────────────────

THIS JOB IS:
  Identifying themes that are duplicates across batches — i.e., Batch 1 discovered
  "Order Tracking" and Batch 2 also discovered "Track My Order" — same concept,
  different name, discovered independently. Merge these into one.

THIS JOB IS NOT:
  • Reorganizing, optimizing, or "cleaning up" the theme list.
  • Merging themes that are merely related or in the same topic area.
  • Reducing the list to a compact set of broad categories.
  • Dropping themes you consider minor or redundant.
  • Merging themes within the same batch (they are already distinct by design).

ACCOUNTING REQUIREMENT — CRITICAL:
  Every single input theme must appear in your output exactly once, either:
  (a) Unchanged — because it has no duplicate in the other batch, OR
  (b) As the survivor of a named cross-batch merge — explicitly state which
      Batch 1 and Batch 2 themes were merged.
  You may NOT silently omit any input theme. If you output fewer themes than
  expected and cannot name which themes were merged into which, you are
  dropping themes — this is a critical error.

─────────────────────────────────────────────────
C. MERGE vs. KEEP-SEPARATE DECISION FRAMEWORK
─────────────────────────────────────────────────

**MERGE two themes into one** only when ALL of the following are true:

1. **Same customer intent** — both themes address the exact same
   underlying customer need, pain point, or action. Not just related
   — identical in purpose.

2. **Same business action** — the brand would take the same marketing,
   product, or support action regardless of which theme the message
   fell into. There is no analytical value in knowing which sub-theme
   a customer belongs to.

3. **Same chatbot response** — the bot would give the same answer to
   messages from both themes. If different information is needed,
   they are different themes.

4. **No loss of analytical granularity** — merging does not destroy
   any insight the brand would want. If the brand would benefit from
   knowing the theme split (even if the topics are related), keep
   them separate.

**KEEP SEPARATE** if ANY of the following are true:

- Different customer pain points or needs, even if superficially
  related (e.g. "acne treatment" ≠ "pigmentation/dark spots" — both
  are skin concerns but different problems, products, and SEO keywords).
- Different SEO / ad-campaign keywords — would you run different
  Google ads for each? If yes, keep separate.
- Different conversion patterns or customer personas.
- Different chatbot knowledge required to answer.
- Different product features or health conditions.
- One is a high-value niche worth tracking on its own (e.g.
  pregnancy safety, breastfeeding concerns).
- Different purchase-funnel stages (pre-purchase vs. checkout vs.
  post-purchase).
- Different post-purchase actions (tracking ≠ cancellation ≠
  modification ≠ returns/refunds).
- Different question types about the same product ("how to use" ≠
  "how much to take" ≠ "how long until results" ≠ "what ingredients").
- They are in the same topic area but represent meaningfully different
  customer intents (e.g. "Product Pricing" ≠ "Discounts & Coupons").

─────────────────────────────────────────────────
D. CALIBRATION EXAMPLES
─────────────────────────────────────────────────

✅ MERGE — True cross-batch duplicates (same intent, different vocabulary):

  Batch 1: "COD Payment"  +  Batch 2: "Cash on Delivery"
  → merge into "Cash on Delivery (COD)"
  Why: identical concept, abbreviation vs. full form. Zero analytical
  difference — the chatbot answer and business action are 100% the same.

  Batch 1: "Customer Care Contact"  +  Batch 2: "Request Callback"
  → merge into "Contact & Callback Requests"
  Why: both express the same intent — the customer wants to speak to a
  human. Same chatbot flow, same business response.

  Batch 1: "Order Delivery Status"  +  Batch 2: "Track My Shipment"
  → merge into "Order Tracking & Delivery"
  Why: same customer action (checking where their order is), same
  chatbot response (provide tracking link), same business metric.

  Batch 1: "Dark Spots Treatment"  +  Batch 2: "Hyperpigmentation"
  → merge into "Pigmentation & Dark Spots"
  Why: same underlying skin concern, same product recommendations,
  same chatbot answer, no business value in splitting.

❌ KEEP SEPARATE — Related but analytically distinct:

  "Order Tracking" vs. "Order Cancellation" vs. "Returns & Refunds"
  Why: different post-purchase customer actions. Cancellation rate is
  a critical KPI on its own. Returns signal product-market fit issues.
  Tracking is a satisfaction signal. Each needs a different chatbot flow.

  "Product Pricing & Affordability" vs. "Discounts & Coupons"
  Why: price sensitivity ("it's too expensive") ≠ deal-seeking
  ("where's my coupon code?"). Different marketing responses, different
  customer segment signals.

  "Side Effects & Safety" vs. "Pregnancy & Breastfeeding Safety"
  Why: pregnancy safety is a high-value niche insight worth tracking
  separately. Different regulatory implications, different customer
  segment, different chatbot response required.

  "Product Usage & Application" vs. "Dosage & Quantity"
  vs. "Results Timeline"
  Why: three distinct customer questions — HOW to use, HOW MUCH to take,
  HOW LONG until results. Each needs different information.

  "Acne Treatment" vs. "Acne Marks & Post-Acne Scars"
  Why: active acne (ongoing treatment) ≠ post-acne marks (pigmentation
  left behind). Different products recommended, different customer stage,
  different SEO keywords ("pimple treatment" vs. "acne scar removal").

  "Beard Growth" vs. "Hair Fall & Hair Care"
  Why: different body area, different products, different customer
  demographic. Despite both being "hair" topics, they are distinct.

─────────────────────────────────────────────────
E. KEY PHRASE CONSOLIDATION RULES
─────────────────────────────────────────────────

When merging two themes:

1. Union the key phrases from both themes.
2. Remove exact or near-exact duplicates.
3. Keep 6–12 of the most diverse, representative phrases.
4. Ensure the final phrase set covers all facets of the merged theme.
5. Phrases can be in any language customers use (English, Hindi,
   Hinglish, etc.) — maximize embedding coverage.
6. Include both abbreviated and expanded forms where applicable.
7. No personal data (names, ages, order IDs) in phrases.

When keeping themes separate:

- Do NOT move phrases between them.
- Ensure each theme's phrases are distinct from the other's.

─────────────────────────────────────────────────
F. OUTPUT FORMAT
─────────────────────────────────────────────────

Return a JSON array with the same structure. No text before or after the JSON.

```json
[
  {{
    "theme_name": "Theme Name",
    "description": "Polished, client-facing description of the customer concerns and scenarios covered. No importance indicators.",
    "key_phrases": ["diverse", "multilingual", "representative", "phrases"],
    "example_messages": ["example 1", "example 2"]
  }},
  ...
]
```

- Themes kept separate appear unchanged.
- Merged themes appear once with a consolidated name, description,
  phrases, and examples.
- Protected/seed themes keep their original name.
- Every input theme must be represented — either standalone or as a
  named merge survivor.

─────────────────────────────────────────────────
G. PRE-OUTPUT CHECKLIST — REQUIRED BEFORE RESPONDING
─────────────────────────────────────────────────

Work through this mentally before generating the JSON:

1. For each merge decision: can I name exactly which Batch 1 theme and
   which Batch 2 theme were merged? If I can't name both, undo the merge.

2. Did I merge themes from within the same batch with each other?
   → If yes, undo. Within-batch themes are already distinct by design.

3. Did I silently drop any input theme (it appears in neither the output
   as a standalone theme NOR as a named participant in a merge)?
   → If yes, add it back.

4. Did I merge any themes with DIFFERENT customer pain points?
   → If yes, undo. Keep separate.

5. Did I merge themes that need DIFFERENT chatbot responses?
   → If yes, undo.

6. Is every remaining theme specific enough that I can state ONE
   precise customer concern it captures in a single sentence?

7. Do any theme names contain vague language like "General Queries",
   "Various Product Questions", or use "and" to link two different
   customer intents? → If yes, the theme is over-merged. Split it.

8. Did I preserve all protected/seed themes?

9. Is the JSON valid?

When in doubt about any merge, DEFAULT TO KEEPING SEPARATE.
Over-merging destroys granularity that cannot be recovered downstream.

Return ONLY the JSON array.'''

    return prompt


def get_phrase_refinement_prompt(
    theme_name: str,
    description: str,
    initial_phrases: List[str],
    sample_messages: List[str],
) -> str:
    """
    Generate prompt for refining and validating key phrases (2nd GPT call).

    This prompt analyzes initial phrases and:
    - Expands abbreviations for better similarity matching
    - Adds linguistic variations
    - Ensures phrases cover the full semantic range
    - Generates 6-10 high-quality phrases

    Args:
        theme_name: Name of the theme
        description: Theme description
        initial_phrases: Initial key phrases from discovery
        sample_messages: Sample messages from this theme

    Returns:
        System prompt string for GPT
    """
    import json

    phrases_json = json.dumps(initial_phrases, indent=2)
    messages_list = "\n".join([f"- {msg[:150]}" for msg in sample_messages[:20]])

    prompt = f'''You are refining key phrases for a customer message theme to maximize similarity matching.

THEME: "{theme_name}"
DESCRIPTION: {description}

INITIAL KEY PHRASES:
{phrases_json}

SAMPLE MESSAGES from this theme:
{messages_list}

YOUR TASK: Generate 6-10 REFINED key phrases optimized for semantic similarity matching.

CRITICAL REQUIREMENTS:

1. **EXPAND ABBREVIATIONS**: Always include BOTH the abbreviation AND full form
   - "BOGO" → include BOTH "BOGO offer" AND "buy one get one"
   - "COD" → include BOTH "COD payment" AND "cash on delivery"
   - "EMI" → include BOTH "EMI option" AND "monthly installment"

2. **SEMANTIC COVERAGE**: Phrases should capture how customers ACTUALLY phrase their messages
   - Look at the sample messages above
   - Extract the exact patterns customers use
   - Include common misspellings that customers use (fix them but keep the concept)

3. **LENGTH**: 2-5 words per phrase (flexible to capture meaning)
   - Too short loses context: "track" → BAD
   - Just right: "track my order" → GOOD
   - Full variations: "order delivery status" → GOOD

4. **DIVERSITY**: Each phrase should match DIFFERENT messages
   - Don't just paraphrase the same concept
   - Cover different aspects of the theme
   - Include action-oriented phrases ("check order", "request refund")
   - Include question-style phrases ("where is order", "how to return")

5. **SIMILARITY MATCHING**: Phrases will be embedded and matched to customer messages
   - Use natural language that customers actually write
   - Include verb forms customers use ("tracking", "track", "tracked")
   - Include common synonyms ("delivery", "shipping", "shipment")

EXAMPLES:

Theme: "Order Tracking"
POOR phrases: ["track order", "order status"] (only 2 phrases, limited coverage)
GOOD phrases: [
  "track my order",
  "order tracking status",
  "where is my package",
  "delivery status update",
  "shipment tracking",
  "when will order arrive",
  "order not received",
  "check delivery status"
]

Theme: "BOGO Offers"
POOR phrases: ["BOGO", "buy one get one"] (missing variations)
GOOD phrases: [
  "BOGO offer",
  "buy one get one",
  "BOGO deal available",
  "buy one get one free",
  "BOGO discount",
  "second item free",
  "two for one offer",
  "BOGO promotion"
]

OUTPUT FORMAT: Return ONLY a JSON object:
```json
{{
  "refined_phrases": [
    "phrase 1",
    "phrase 2",
    "phrase 3",
    "phrase 4",
    "phrase 5",
    "phrase 6",
    "phrase 7",
    "phrase 8"
  ],
  "changes_made": "Brief explanation of refinements"
}}
```

CRITICAL:
- Generate 6-10 phrases (aim for 8)
- Expand ALL abbreviations to full forms
- Include phrases that match how customers ACTUALLY write
- Return ONLY valid JSON'''

    return prompt


def get_cluster_merge_decision_prompt(
    theme_pairs: List[dict],
) -> str:
    """
    Generate prompt for GPT to decide which similar themes should be merged.

    Uses semantic understanding to distinguish between:
    - Themes that should merge (same user intent/goal)
    - Themes that should stay separate (related but functionally distinct)

    Args:
        theme_pairs: List of theme pair dicts with structure:
            {
                "theme_a": {"name": str, "description": str, "key_phrases": List[str]},
                "theme_b": {"name": str, "description": str, "key_phrases": List[str]},
                "similarity": float
            }

    Returns:
        System prompt string for GPT
    """
    import json

    # Format pairs for prompt
    pairs_text = []
    for i, pair in enumerate(theme_pairs, 1):
        theme_a = pair['theme_a']
        theme_b = pair['theme_b']
        sim = pair.get('similarity', 0.0)

        pairs_text.append(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAIR {i} (Embedding Similarity: {sim:.3f})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Theme A: "{theme_a['name']}"
Description: {theme_a['description']}
Key Phrases: {', '.join(theme_a['key_phrases'][:6])}

Theme B: "{theme_b['name']}"
Description: {theme_b['description']}
Key Phrases: {', '.join(theme_b['key_phrases'][:6])}
""")

    pairs_formatted = "\n".join(pairs_text)

    prompt = f'''You are a VoC analytics expert deciding which similar themes should merge vs stay separate.

⚠️  CRITICAL: High embedding similarity does NOT mean themes should merge!
    Embeddings detect semantic relatedness, but we need BUSINESS EQUIVALENCE.

BUSINESS CONTEXT: These themes drive:
- Conversion analytics (different themes = different customer segments with different conversion rates)
- SEO & ad targeting (different themes = different keywords and campaigns)
- PDP optimization (need to know WHICH features customers care about, not just "features")
- Bot training (different themes need different responses)
- Trend insights for clients (granular themes show emerging opportunities)

THEME PAIRS TO EVALUATE:
{pairs_formatted}

═══════════════════════════════════════════════════════════════════════════════
                         DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

🔴 MERGE ONLY IF ALL FOUR CONDITIONS ARE TRUE:
   ✓ SAME CUSTOMER INTENT: Both themes address the EXACT same customer need/pain point
   ✓ SAME CHATBOT RESPONSE: An agent's answer to Theme A would fully satisfy Theme B
   ✓ SAME BUSINESS ACTION: Both would trigger identical marketing/product decisions
   ✓ NO ANALYTICAL VALUE: Keeping them separate provides NO actionable insights

🟢 KEEP SEPARATE IF ANY OF THESE ARE TRUE:
   ✗ DIFFERENT CUSTOMER PAIN POINTS: Battery anxiety ≠ GPS tracking needs
   ✗ DIFFERENT SEO/AD VALUE: "GPS smartwatch" ≠ "long battery smartwatch" campaigns
   ✗ DIFFERENT CONVERSION SEGMENTS: Fitness users ≠ business users ≠ health trackers
   ✗ DIFFERENT PRODUCT FEATURES: Each feature worth tracking separately for PDP optimization
   ✗ DIFFERENT PURCHASE STAGES: Pre-order ≠ tracking ≠ returns ≠ reviews
   ✗ DIFFERENT CHATBOT NEEDS: Require different information/responses
   ✗ DIFFERENT TREND VALUE: Client wants to track separately ("pregnancy queries up 40%")
   ✗ DIFFERENT QUESTION TYPES: What/how/when/where/why on same topic

DECISION CHECKLIST - Ask yourself for each pair:
1. "If I merged these, would the client lose valuable segmentation insights?" → If YES, keep separate
2. "Do these represent different SEO keywords or ad campaigns?" → If YES, keep separate
3. "Would splitting help identify WHICH specific feature/concern drives conversions?" → If YES, keep separate
4. "Are these just vocabulary differences for the EXACT same need?" → If YES, merge
5. "Would a chatbot need different data/responses for these?" → If YES, keep separate

DEFAULT TO SEPARATE when uncertain. Over-merging destroys valuable granularity.

═══════════════════════════════════════════════════════════════════════════════
                              EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

✅ MERGE - These are TRUE DUPLICATES:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "Acne Treatment Products"                                          │
│ Phrases: "product for acne", "acne serum", "acne treatment"                 │
│ Theme B: "Acne Marks Removal"                                               │
│ Phrases: "remove acne marks", "clear acne spots", "acne scars"              │
│ → MERGE into "Acne Treatment & Concerns"                                    │
│ Reason: Same underlying customer problem (acne), same products recommended, │
│         same customer segment, no analytical value in splitting.            │
│         Sub-categories (marks vs active acne) don't warrant separate themes.│
└─────────────────────────────────────────────────────────────────────────────┘

✅ MERGE - These are TRUE DUPLICATES:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "BOGO Deals"                                                       │
│ Phrases: "BOGO offer", "BOGO discount", "BOGO deal"                         │
│ Theme B: "Buy One Get One Promotions"                                       │
│ Phrases: "buy one get one", "buy one get one free"                          │
│ → MERGE into "BOGO & Buy One Get One Offers"                                │
│ Reason: Identical concept, just abbreviation vs full form. No analytical    │
│         value in keeping separate.                                          │
└─────────────────────────────────────────────────────────────────────────────┘

❌ KEEP SEPARATE - Different Product Features:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "GPS & Step Tracking"                                              │
│ Phrases: "does it have GPS", "track steps", "built-in GPS"                  │
│ Theme B: "Battery & Charging"                                               │
│ Phrases: "battery life", "how long to charge", "battery backup"             │
│ → KEEP SEPARATE                                                             │
│ Reason: Different customer pain points (fitness tracking vs battery anxiety)│
│         Different SEO ("GPS smartwatch" vs "long battery smartwatch")       │
│         Different conversion segments (athletes vs business travelers)      │
│         Client needs to know WHICH feature drives conversions               │
└─────────────────────────────────────────────────────────────────────────────┘

❌ KEEP SEPARATE - Different Skin Concerns:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "Acne Treatment"                                                   │
│ Phrases: "acne serum", "clear acne", "pimple treatment"                     │
│ Theme B: "Pigmentation & Dark Spots"                                        │
│ Phrases: "dark spots", "hyperpigmentation", "reduce pigmentation"           │
│ → KEEP SEPARATE                                                             │
│ Reason: Different skin concerns, different products, different ingredients, │
│         different customer segments. Valuable to track separately.          │
└─────────────────────────────────────────────────────────────────────────────┘

❌ KEEP SEPARATE - Different Safety Concerns:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "Pregnancy Safety"                                                 │
│ Phrases: "safe during pregnancy", "can pregnant women use"                  │
│ Theme B: "General Product Safety"                                           │
│ Phrases: "any side effects", "is it safe", "product safety"                 │
│ → KEEP SEPARATE                                                             │
│ Reason: Pregnancy safety is high-value niche insight worth tracking         │
│         separately. Different customer segment, different marketing value.  │
│         Client wants to know "pregnancy queries increased 40% in Q2"        │
└─────────────────────────────────────────────────────────────────────────────┘

❌ KEEP SEPARATE - Different Order Actions:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "Order Tracking"                                                   │
│ Phrases: "track my order", "where is my order", "delivery status"           │
│ Theme B: "Order Cancellation"                                               │
│ Phrases: "cancel order", "cancel my order", "order cancellation"            │
│ → KEEP SEPARATE                                                             │
│ Reason: Different customer actions, different conversion implications,      │
│         different chatbot flows. Cancellation rate is critical metric.      │
└─────────────────────────────────────────────────────────────────────────────┘

❌ KEEP SEPARATE - Different Question Types:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Theme A: "Product Usage Instructions"                                       │
│ Phrases: "how to use", "application method", "how to apply"                 │
│ Theme B: "Product Ingredients"                                              │
│ Phrases: "what ingredients", "contains niacinamide", "ingredient list"      │
│ → KEEP SEPARATE                                                             │
│ Reason: Different questions (HOW vs WHAT), different chatbot responses,     │
│         different customer journey stage, different SEO value.              │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

For each pair in the input, make a decision and provide clear business reasoning.

OUTPUT FORMAT: Return a JSON array:
```json
[
  {{
    "pair_number": 1,
    "decision": "separate",
    "reason": "Different question types: A asks WHERE order is, B asks WHAT order contains"
  }},
  {{
    "pair_number": 2,
    "decision": "merge",
    "reason": "True duplicates: Both about skin lightening with different vocabulary"
  }}
]
```

Return ONLY the JSON array, no other text.'''

    return prompt


def get_theme_naming_prompt(
    messages_sample: List[str],
    current_name: str,
    key_phrases: List[str],
) -> str:
    """
    Generate prompt for improving a theme name based on its messages.

    Args:
        messages_sample: Sample messages from the theme
        current_name: Current theme name
        key_phrases: Key phrases for the theme

    Returns:
        System prompt string for GPT
    """
    message_list = "\n".join([f"- {msg[:150]}" for msg in messages_sample[:15]])
    phrase_list = ", ".join(key_phrases)

    prompt = f'''Create a concise, SPECIFIC theme name (1-5 words) for this cluster of customer messages.

CURRENT NAME: "{current_name}"
KEY PHRASES: {phrase_list}

SAMPLE MESSAGES:
{message_list}

CRITICAL NAMING PRINCIPLES:

🎯 GOOD THEME NAMES are:
   ✓ SPECIFIC - "GPS & Step Tracking" not "Product Features"
   ✓ ACTIONABLE - Client knows what to optimize
   ✓ CUSTOMER-FOCUSED - Describes customer need/pain point
   ✓ SEO-RELEVANT - Could be a search keyword
   ✓ ANALYTICAL - Enables cohort segmentation

❌ BAD THEME NAMES are:
   ✗ Vague: "Product Questions", "General Inquiries", "Customer Concerns"
   ✗ Too generic: "Features", "Issues", "Help"
   ✗ Sentences: "Questions about when products will arrive"
   ✗ Internal jargon: "Tier 1 Support", "Pre-sales"

REQUIREMENTS:
- Name must be 1-5 words (flexible if needed for clarity)
- Must be a NOUN PHRASE describing the specific concern/need
- Should differentiate this theme from related themes
- Should be specific enough for SEO/ad targeting
- Avoid generic terms unless combined with specifics (e.g., "Order Tracking" is OK, "Orders" is not)

EXCELLENT EXAMPLES:
✅ "GPS & Step Tracking" (not just "Fitness Features")
✅ "Battery & Charging Concerns" (not just "Battery")
✅ "Pregnancy Safety Questions" (not just "Product Safety")
✅ "Dark Spots & Hyperpigmentation" (not just "Skin Concerns")
✅ "Order Tracking & Delivery Status" (not just "Orders")
✅ "Acne Treatment & Concerns" (not just "Acne")

BAD EXAMPLES:
❌ "Various Product Questions" (too vague)
❌ "Customer Support Issues" (too generic)
❌ "Questions about shipping times" (sentence, not noun phrase)
❌ "Features and Specifications" (covers too much)

Return ONLY the theme name, nothing else.'''

    return prompt

def get_misc_validation_prompt(
    messages_with_matches: List[dict],
) -> str:
    """
    Generate prompt for Stage 2 misc validation - GPT validates message-to-theme matches.

    Args:
        messages_with_matches: List of dicts with:
            - idx: message index
            - message: message text
            - theme_name: candidate theme name
            - theme_description: theme description
            - best_phrase: best matching phrase
            - similarity: cosine similarity score

    Returns:
        System prompt for validation
    """
    import json

    matches_text = []
    for m in messages_with_matches:
        matches_text.append(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDEX {m['idx']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Message: "{m['message']}"

Candidate Theme: "{m['theme_name']}"
Description: {m['theme_description']}
Best Matching Phrase: "{m['best_phrase']}"
Similarity Score: {m['similarity']:.3f}
""")

    formatted_matches = "\n".join(matches_text)

    prompt = f'''You are validating message-to-theme assignments in a VoC clustering system.

Each message below was not confidently assigned to any theme, but has a "closest match" with similarity score between 0.45-0.52. Your task is to decide if the match is semantically valid.

MESSAGES TO VALIDATE:
{formatted_matches}

DECISION CRITERIA:

✅ VALID - Assign "VALID" if:
   - The message is clearly about the same topic as the theme
   - The customer's intent aligns with the theme's purpose
   - The message would logically belong in this theme cluster
   - The similarity score makes sense given the semantic overlap

❌ INVALID - Assign "INVALID" if:
   - The message is about a different topic entirely
   - There's just superficial word overlap but different meaning
   - The customer's intent doesn't match the theme
   - This looks like a false positive from embedding similarity

OUTPUT FORMAT: Return ONLY a JSON array:
```json
[
  {{
    "idx": 0,
    "decision": "VALID",
    "reason": "Message asks about beard growth oil, theme is beard growth products"
  }},
  {{
    "idx": 1,
    "decision": "INVALID",
    "reason": "Message is about hair care, not related to battery charging theme"
  }}
]
```

CRITICAL:
- Return decision for EVERY index from 0 to {len(messages_with_matches) - 1}
- Use "VALID" or "INVALID" exactly (case-sensitive)
- Provide brief reason explaining your decision
- Return ONLY the JSON array, no other text'''

    return prompt


def get_second_pass_discovery_prompt(
    messages: List[str],
    min_themes: int = 3,
    max_themes: int = 10,
    key_phrases_per_theme: int = 10,
) -> str:
    """
    Generate the prompt for BLIND second-pass theme discovery on miscellaneous messages.

    IMPORTANT: This does NOT pass existing themes as seeds. The second pass runs blind
    discovery to find micro-themes in miscellaneous messages, then deduplication happens
    afterward to check against first-pass themes.

    This uses the SAME strict phrase quality requirements as the first pass to ensure
    high-quality key phrases.

    Args:
        messages: List of miscellaneous message texts (sampled)
        min_themes: Minimum new micro-themes to discover
        max_themes: Maximum new micro-themes to discover
        key_phrases_per_theme: Target number of key phrases per theme

    Returns:
        System prompt string for GPT
    """
    message_list = "\n".join([f"- {msg[:200]}" for msg in messages])

    prompt = f'''You are a Voice of Customer (VoC) analyst performing a **second pass**
on messages that were not assigned to any theme during the first
clustering pass. These messages were left over because they did not
match the themes discovered earlier.

Your job is to find **new micro-themes** — coherent clusters of intent
that the first pass missed. Apply the same analytical rigor as the
first pass: each theme must capture one **distinct, business-actionable
customer intent or concern**.

─────────────────────────────────────────────────
A. WHY THESE THEMES MATTER (business context)
─────────────────────────────────────────────────

These themes will be used to:

1. **Cohort conversion analytics** — e.g. "users who asked about
   shipping converted at 12 %; users who asked about returns converted
   at 4 %."
2. **Customer language / SEO insights** — the key phrases tell the
   brand how real customers talk about a topic.
3. **PDP & ad-campaign optimization** — each theme maps to a product
   page section or ad angle the brand can improve.
4. **Trend tracking** — e.g. "age suitability queries rose from 2 %
   to 8 % in Q2."
5. **Chatbot optimization** — identify knowledge gaps, handoff-worthy
   intents, and quick-reply bubble opportunities.
6. **Brand insights** — emergent themes reveal customer interests,
   expectations, and pain points.

MISCELLANEOUS MESSAGES:
{message_list}

─────────────────────────────────────────────────
B. WHEN TO CREATE SEPARATE THEMES vs. MERGE
─────────────────────────────────────────────────

Create a **separate** theme when any of these are true:

- Different customer pain points or needs.
- Different SEO / ad-campaign keywords.
- Different conversion patterns or customer personas.
- Different chatbot responses needed.
- Different product features or health conditions.
- High business-value niche worth tracking on its own.
- Different purchase-funnel stages.

**Merge** into one theme only when:

- Messages express the **exact same** underlying intent, just in
  different words.
- Splitting would yield no additional analytical, SEO, or conversion
  insight.
- Sub-categories are so tightly coupled that splitting fragments the
  picture without adding value.

─────────────────────────────────────────────────
C. GRANULARITY CALIBRATION
─────────────────────────────────────────────────

These are MICRO-THEMES from miscellaneous messages — they may be more
specific than first-pass themes since they represent niche concerns
the first pass missed.

CORRECT separations (each is its own theme):
  - "Product Usage & Application" (how to use, how to apply)
  - "Dosage & Quantity" (how many capsules, kitni tablet)
  - "Course Duration & Results Timeline" (kitne din, how long)
  Reason: three different customer questions needing different
  chatbot knowledge and revealing different brand insights.

  - "Side Effects & Safety" vs. "Pregnancy & Breastfeeding Safety"
  Reason: pregnancy safety is high-value niche. Always separate.

CORRECT merges:
  - "Customer care number" + "callback request" + "WhatsApp contact"
    → 1 theme: Contact & Callback Requests
  Reason: same underlying intent — talk to a human.

RED FLAGS that a theme is too broad:
  ✗ Vague name: "Product Questions," "General Queries."
  ✗ 15+ phrases spanning unrelated sub-topics.
  ✗ Cannot describe ONE specific customer concern.

GREEN FLAGS that a theme is well-formed:
  ✓ Specific name: "Age Suitability Questions," "Drug Interactions."
  ✓ 3–10 phrases clearly belonging to the same intent.
  ✓ Directly informs one type of business decision.

─────────────────────────────────────────────────
D. KEY PHRASE REQUIREMENTS
─────────────────────────────────────────────────

Each theme needs {key_phrases_per_theme} key phrases (range: 4–12).
These are the theme's **semantic anchors** — the PRIMARY GOAL is to
MAXIMIZE COVERAGE so every real message belonging to this theme is
close to at least one phrase.

1. **Language**: Phrases can be in ANY language customers use (English,
   Hindi, Hinglish, regional languages). If customers write in
   Hindi/Hinglish, include Hindi/Hinglish phrases alongside English
   ones to maximize embedding matches.

2. **Length**: 2–6 words.
     Bad:  "age"  (too short, ambiguous)
     Good: "age limit for product"
     Good: "kitni umar me use kare"

3. **Diversity**: Each phrase should pull in a DIFFERENT subset of
   messages. Cover different facets, synonyms, and phrasings —
   including across languages.

4. **Customer voice**: Mirror how customers actually ask.
     Bad:  "age eligibility inquiry"
     Good: "minimum age to use"
     Good: "meri umar me use kar sakte hain"

5. **Abbreviation expansion**: Include BOTH abbreviated and expanded
   forms as separate phrases where applicable.

6. **No specific data**: No order IDs, phone numbers, personal names,
   specific ages, or brand-specific product codes.

─────────────────────────────────────────────────
E. THEME COUNT & OUTPUT FORMAT
─────────────────────────────────────────────────

Extract {min_themes}-{max_themes} micro-themes from these messages.
Only create a theme if you see at least 5-8 messages that clearly
cluster around it. It is perfectly fine to produce fewer than
{min_themes} themes if the remaining messages are too noisy.
Quality over quantity.

Return a JSON array. No text before or after the JSON.

```json
[
  {{
    "theme_name": "Short Specific Name (1–5 words)",
    "description": "Polished, client-facing description of the customer concerns and scenarios covered. No importance indicators.",
    "key_phrases": [
      "phrase one — can be in any language customers use",
      "phrase two",
      "..."
    ],
    "example_messages": [
      "verbatim message from the input",
      "another verbatim message"
    ]
  }}
]
```

─────────────────────────────────────────────────
F. FINAL CHECKLIST
─────────────────────────────────────────────────

1. Does every theme represent ONE clear customer intent or concern?
2. Would the brand take a DIFFERENT action for each theme?
3. Are there any themes so broad they would attract unrelated messages?
4. Did I keep high-value niches separate?
5. Are key phrases diverse and multilingual where customers use
   non-English, to maximize message coverage?
6. Is the JSON valid?

Return ONLY the JSON array.'''

    return prompt


def get_usecase_tagging_prompt(
    clusters: List[Dict[str, str]],
    usecases: List[str],
) -> str:
    """
    Generate prompt for GPT to tag each cluster with all relevant usecases.

    This is used in the usecase-aware boosting (Phase 3.6 / Phase 6): GPT tags
    each cluster with usecases so we can boost message-to-cluster scores when
    the message's usecase matches a cluster's tagged usecases.

    Args:
        clusters: List of dicts with "cluster_name" and "description" keys
        usecases: List of unique usecase strings discovered from chat data

    Returns:
        System prompt string for GPT
    """
    import json

    clusters_json = json.dumps(clusters, indent=2, ensure_ascii=False)
    usecases_list = "\n".join([f"  - {uc}" for uc in usecases])

    prompt = f'''You are a Voice of Customer (VoC) analyst. You will be given:
1. A list of **theme clusters** (each with a name and description)
2. A list of **usecases** (customer intent labels from the chat data)

Your task: for each cluster, tag **every usecase whose customers would
naturally ask questions that belong in that cluster**. This is a
many-to-many annotation — most clusters will have 1–3 matching usecases,
and some broad clusters may have 4–6.

─────────────────────────────────────────────────
A. CLUSTERS
─────────────────────────────────────────────────

{clusters_json}

─────────────────────────────────────────────────
B. AVAILABLE USECASES
─────────────────────────────────────────────────

{usecases_list}

─────────────────────────────────────────────────
C. MENTAL MODEL — HOW USECASES RELATE TO CLUSTERS
─────────────────────────────────────────────────

**Usecase** = the customer's goal or motivation (WHY they opened the chat).
**Cluster** = the topic of their specific question (WHAT they asked about).

These are DIFFERENT axes, so the relationship is many-to-many:

• One usecase can span many clusters:
  A customer who is "Unsure About Effectiveness" might ask about
  whether the product grows beard (→ Beard Growth Effectiveness),
  how long it takes to work (→ Results Timeline), or what side
  effects it has (→ Side Effects & Safety). All three clusters
  should be tagged with "Unsure About Effectiveness".

• One cluster can attract many usecases:
  "Side Effects & Safety" attracts:
    - customers doubting the product (Unsure About Effectiveness)
    - customers wanting factual information (Product Info)
    - customers who experienced a problem (Customer Problem)
    - customers comparing products (Differentiation)
    - customers leaving feedback about reactions (Feedback)

The KEY TEST for tagging: "Would a customer with this usecase label
naturally ask something that belongs in this cluster?"
If YES → tag it. If NO or UNSURE → do not tag it.

─────────────────────────────────────────────────
D. TAGGING RULES
─────────────────────────────────────────────────

1. **Tag ALL usecases that genuinely apply** — do not limit to 1 or 2.
   Narrow, action-specific clusters (e.g. "Order Placement",
   "Discount & Coupons") typically map to 1–2 usecases.
   Broad, skeptic-type clusters (e.g. "Side Effects & Safety",
   "Trust & Authenticity", "Ingredients & Proof") may map to 4–6
   because they attract customers with diverse motivations.

2. **Exact match only** — use the EXACT usecase string from the list
   above. Do NOT invent or modify usecase strings.

3. **Relevance is required** — only tag a usecase if the key test
   above passes. Do not tag loosely related usecases just to
   increase coverage.

4. **Dead-end usecases tag to NO cluster** — some usecases represent
   customers who ended or abandoned the conversation without a real
   topical query (e.g. a usecase labelled something like "no more
   queries", "session ended", "goodbye"). If a usecase has no clear
   connection to any cluster topic, it should appear in no cluster's
   list. Do NOT force a match.

5. **Some clusters may have NO matching usecase** — clusters covering
   off-topic content, pure language preference, or miscellaneous
   messages may not align with any customer usecase. Return an empty
   list for those.

─────────────────────────────────────────────────
E. CALIBRATION EXAMPLES
─────────────────────────────────────────────────

NARROW CLUSTER — maps to 1 usecase:
  Cluster: "Usage & Application" (how to apply the product)
  → Tag: ["Product Info"]
  Why: only customers seeking product information ask HOW to use it.

ACTION CLUSTER — maps to 2–3 usecases:
  Cluster: "Delivery & Order Tracking" (order status, ETA, cancellation)
  → Tag: ["Delivery Time", "Order Status", "Cancellation"]
  Why: all three usecases produce questions that land in this cluster.

SKEPTIC CLUSTER — maps to 4–5 usecases:
  Cluster: "Side Effects & Safety" (is it safe, any reactions, harmful?)
  → Tag: ["Unsure About Effectiveness", "Product Info",
          "Customer Problem", "Differentiation", "Feedback"]
  Why: doubtful customers, info-seekers, people who already had a
  reaction, comparison shoppers, and reviewers all ask safety
  questions — five distinct motivations, one cluster.

VALUE-QUERY CLUSTER — usecase reveals pricing intent:
  Cluster: "Dosage & Quantity" (how many capsules, how much product)
  → Tag: ["Product Info", "Pricing"]
  Why: "how many tablets in one bottle for ₹499?" is a dosage
  question framed as a value-for-money (Pricing) query.

─────────────────────────────────────────────────
F. OUTPUT FORMAT
─────────────────────────────────────────────────

Return a JSON array. No text before or after the JSON.

```json
[
  {{
    "cluster_name": "Exact cluster name from input",
    "usecases": ["usecase_1", "usecase_2", "usecase_3"]
  }},
  {{
    "cluster_name": "Narrow cluster",
    "usecases": ["usecase_1"]
  }},
  {{
    "cluster_name": "Cluster with no matching usecase",
    "usecases": []
  }}
]
```

Return ONLY the JSON array.'''

    return prompt


def get_category_tagging_prompt(
    clusters: List[Dict[str, str]],
) -> str:
    """
    Generate prompt for GPT to tag each cluster as pre-sales, post-sales, or miscellaneous.

    Args:
        clusters: List of dicts with "cluster_name" and "description" keys

    Returns:
        System prompt string for GPT
    """
    import json

    clusters_json = json.dumps(clusters, indent=2, ensure_ascii=False)

    prompt = f'''You are a Voice of Customer analyst. Tag each cluster below as EITHER
"pre-sales" OR "post-sales" based on where it falls in the customer purchase journey.

IMPORTANT: You must ONLY use "pre-sales" or "post-sales". No other values are valid.

CLUSTERS:
{clusters_json}

CATEGORIES:

**pre-sales** — The customer has NOT yet completed a purchase. They are evaluating,
researching, deciding, or attempting to place an order. This includes:
  - Product information: usage, dosage, ingredients, side effects, safety
  - Product concerns: effectiveness, results timeline, suitability
  - Health/skin/hair conditions the product addresses
  - Pricing, discounts, coupons, offers, combos, gift with purchase, free samples,
    freebies, BOGO deals, buy-X-get-Y promotions, promotional bundles
  - Product comparison and selection
  - Brand authenticity and trust
  - Contact/callback requests (customer reaching out before buying)
  - Available payment methods (COD, EMI, card options) — customer is deciding HOW to pay
  - Checkout and ordering issues — customer is trying to place an order but it has NOT been placed yet
  - Any question a customer asks BEFORE or WHILE trying to complete a purchase

**post-sales** — The customer's order has already been placed and confirmed.
This includes:
  - Order tracking, delivery status, shipping delays
  - Order cancellation, modification, address changes (after order is placed)
  - Returns, refunds, exchanges
  - Post-delivery issues (damaged product, wrong item received, etc.)
  - Any question that arises AFTER the order is confirmed in the system

RULES:
1. Every cluster gets exactly ONE of: "pre-sales" or "post-sales".
2. The key question is: "Has the customer's order already been placed and confirmed?"
   If YES → post-sales. If NO (still deciding, still trying to buy) → pre-sales.
3. Payment methods / payment options → ALWAYS pre-sales. The customer is asking about
   how to pay, which means they have NOT paid yet.
4. Checkout issues / ordering problems → ALWAYS pre-sales. If the order had been
   placed, they would be asking about tracking, not how to order.
5. Gifts / freebies / free samples / BOGO deals / promotional offers / gift-with-purchase
   → ALWAYS pre-sales. These are purchase incentives used to persuade a customer to buy;
   they are not post-purchase concerns.
6. Contact/callback/escalation requests → pre-sales (customer is not yet a buyer).
7. Never output "miscellaneous" — that value is reserved for the Miscellaneous
   bucket only and is NOT part of this prompt's output.

OUTPUT FORMAT: Return ONLY a JSON array, no other text:
```json
[
  {{"cluster_name": "Exact cluster name", "category": "pre-sales"}},
  {{"cluster_name": "Another cluster", "category": "post-sales"}}
]
```

Return ONLY the JSON array.'''

    return prompt