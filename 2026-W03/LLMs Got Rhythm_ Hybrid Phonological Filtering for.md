# LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation

**Authors**: Stergios Chatzikyriakidis

**Published**: 2026-01-14 17:05:17

**PDF URL**: [https://arxiv.org/pdf/2601.09631v1](https://arxiv.org/pdf/2601.09631v1)

## Abstract
Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research.

## Full Text


<!-- PDF content starts -->

LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry
Rhyme Detection and Generation
Stergios Chatzikyriakidis
Department of Philology
University of Crete
stergios.chatzikyriakidis@uoc.gr
Abstract
Large Language Models (LLMs), despite their
remarkable capabilities across NLP tasks, strug-
gle with phonologically-grounded phenomena
like rhyme detection and generation. This
is even more evident in lower-resource lan-
guages such as Modern Greek. In this pa-
per, we present a hybrid system that combines
LLMs with deterministic phonological algo-
rithms to achieve accurate rhyme identifica-
tion/analysis and generation. Our approach im-
plements a comprehensive taxonomy of Greek
rhyme types, including Pure, Rich, Imperfect,
Mosaic, and Identical Pre-rhyme V owel (IDV)
patterns, and employs an agentic generation
pipeline with phonological verification. We
evaluate multiple prompting strategies (zero-
shot, few-shot, Chain-of-Thought, and RAG-
augmented) across several LLMs including
Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and
open-weight models like Llama 3.1 8B and
70B and Mistral Large. Results reveal a signifi-
cant "Reasoning Gap": while native-like mod-
els (Claude 3.7) perform intuitively (40% ac-
curacy in identification), reasoning-heavy mod-
els (Claude 4.5) achieve state-of-the-art perfor-
mance (54%) only when prompted with Chain-
of-Thought. Most critically, pure LLM gen-
eration fails catastrophically (under 4% valid
poems), while our hybrid verification loop
restores performance to 73.1%. We release
our system and a crucial, rigorously cleaned
corpus of 40,000+ rhymes, derived from the
AnemoskalaandInterwar Poetrycorpora, to
support future research.1
1 Introduction
Rhyme is a fundamental feature of verse across cul-
tures. In Modern Greek, rhyme ( ομοιοκαταληξία )
has been systematically employed from the Cretan
1The material needed to run the experiments
and verify the results of this paper can be found
here: https://osf.io/d7hr6/overview?view_only=
6e780f09bfd444ccb60ec3732b0d2a9bRenaissance (14th–17th centuries) all the way to
contemporary poetry and popular music, including
hip-hop and rap (Topintzi et al., 2019).
LLMs, unsurprisingly given their prevalent text-
based training, exhibit notable weaknesses in
phonologically-aware reasoning. LLMs process
text at the token level, which poorly aligns with
phonological units like syllables, stress patterns,
and rhyme domains.
We tackle the issue of rhyme identification and
generation using a hybrid Neural-Symbolic archi-
tecture that combines the generative and reasoning
capabilities of LLMs with deterministic phonologi-
cal algorithms. Our contributions are:
1.A dataset of 40k rhymes aggregated from ex-
isting Modern Greek corpora, based on the
phonological taxonomy found in (Topintzi
et al., 2019).
2.A hybrid detection system combining LLM
prompting strategies with rule-based phono-
logical verification, achieving 100% verifica-
tion accuracy compared to 54% (best case) for
pure LLM approaches.
3.An agentic generation pipeline with a
Generate-Verify-Refine loop that raises the
validity of generated poems from nearly 0%
to 73.1%.
4.A multi-model evaluation across proprietary
(Claude, GPT-4o) and open-weight models,
with analysis of prompting strategies includ-
ing RAG augmentation.
2 Background
2.1 Rhyme in Modern Greek Poetry
Rhyme in Modern Greek poetry dates to the me-
dieval period, appearing in works like those of Ste-
fanos Sachlikis (14th c.), and flourishing in CretanarXiv:2601.09631v1  [cs.CL]  14 Jan 2026

Renaissance masterpieces such as Kornaros’sEro-
tokritos. The phenomenon spans the Heptanesian
Romanticism of Solomos, the Athenian school, Par-
nassian and Symbolist movements, and continues,
albeit more selectively, in modern and contempo-
rary poetry (Κοκόλης, 1993).
Greek rhyme exhibits several distinctive charac-
teristics:
Stress-based ClassificationGreek is a stress-
accent language where rhyme domains are defined
relative to the stressed syllable. More particularly,
Greek is subject to the three-syllable rule according
to which accent must fall in one of the last three
syllables of the word (broadly defined, includes
also cases of what we call phonological word, e.g.
a word plus a weak object pronoun). We follow
Topintzi et al. (2019) and use the following cate-
gories:
•Masculine (M): Here, stress is on the final
syllable (oxytone), e.g.,καρδιά/φωτιά
•Feminine-2 (F2): In this category, stress is
on the penultimate syllable (paroxytone), e.g.,
θάλασσα/τάλασσα
•Feminine-3 (F3): Here, stress is on the an-
tepenultimate syllable (proparoxytone), e.g.,
στόματα/σώματα
Greek poetry further employs several other
rhyme types that are independent of stress posi-
tion:
Rich Rhyme (RICH): In this type, the onset
consonant(s) of the stressed syllable must match.
This is further distinguished into Total Rich (TR)
rhyme where complete onset matching is at play,
or Partial Rich (PR) with partial matching:
•TR-S (singleton): καλά /μαλά [ka-’la]/[ma-’la]
•PR-C1 (first consonant): στόματα /σώματα
[’sto-ma-ta]/[’so-ma-ta]
Identical Pre-rhyme Vowel (IDV): In this cate-
gory, the vowel that precedes the stressed syllable
must match:
•ξανθή/γραφή: pre-stress vowel [a] matches
Mosaic (MOS): In this category, the rhyme do-
main spans across more than one word:
•όνομά της /ο μπάτης : [’o-no-’ma tis]/[o ’ba-
tis]Imperfect (IMP): In this category, [artial] pho-
netic matching with systematic variation:
• IMP-V: V owel differs (χάνετε/γίνετε)
•IMP-C: Consonant differs ( ξαφνίζει /τεχνίτη )
• IMP-0F: Final consonant-zero alternation
2.2 Computational Approaches to Rhyme
Early approaches on rhyme identification used
classic unsupervised machine learning (Reddy
and Knight, 2011) or probabilistic models using
phoneme frequencies for rhyme detection in rap
music (Hirjee and Brown, 2010). More recent su-
pervised methods based on neural networks achieve
higher accuracies. For example, Haider and Kuhn
(2018) achieve 97% accuracy via a single Siamese
Recurrent Network model trained in German, En-
glish, and French, using no explicit phonetic fea-
tures.
In poetry generation, we find early statistical
machine translation for Classical Chinese quatrain
generation by He et al. (2012), where they in effect
treat each line as a kind of translation of the previ-
ous line. The Hafez system (Ghazvininejad et al.,
2016) is a hybrid system that puts together finite-
state acceptors that encode metrical and rhyme con-
straints with RNNs for English sonnet generation.
Lau et al. (2018) developed Deep-speare, a joint
neural model for Shakespearean sonnets whose
outputs proved largely indistinguishable from hu-
man verse in crowd evaluations. Moving on to the
LLM era, we find byte-level transformers such as
ByGPT5 (Belouadi and Eger, 2023), as well as
synthetic-data approaches like GPoeT (Popescu-
Belis et al., 2023).
For Greek specifically, the only computational
work is the one by Topintzi et al. (2019), which
resulted in the Greek Rhyme database (GrRh). The
authors use rule-based algorithms that cover mul-
tiple rhyme types, though with acknowledged lim-
itations in precision. This is a pioneering paper
that combines solid theoretical linguistics knowl-
edge with computational implementation and its
theoretical core is one of the inspirations of this
paper.
3 Dataset
We constructed our dataset by aggregating and
standardizing two primary high-quality digital re-
sources for Modern Greek poetry. The first source
is the Anemoskala archive from the Centre for

Poet Rhyme Pairs
Palamas 20,620
Tellos Agras 6,119
Valaoritis 4,202
Solomos 2,518
Karyotakis 1,692
Cavafy 1,585
Fotos Giofyllis 1,565
Kostas Ouranis 792
Napoleon Lapathiotis 495
Romos Filiras 484
Mitsos Papanikolaou 404
Kalvos 100
Total 40,576
Table 1: Corpus composition by poet.
the Greek Language (KEG - Κέντρο Ελληνικής
Γλώσσας ), which provides extensive digitized col-
lections of major poets. The second source is the
Interwar Poetry Dataset, an open-access dataset
created by Dr Natsina and Professor Chatzikyri-
akidis with the help of undergraduate students at
the Philology Department, University of Crete2
This corpus comprises over 600 poems by interwar
Greek poets. We merged these collections, normal-
ized the JSON format, and applied our phonolog-
ical cleaning pipeline to ensure consistency. The
corpus statistics are shown in??.
4 System Architecture
We implement a hybrid, neural-symbolic architec-
ture thatr combines LLM capabilities with deter-
ministic symbolic phonological rules (Figure 1)
4.1 Phonological Engine
The basis of the phonological symbolic proces-
sor/verifier is based on (Topintzi et al., 2019). Its
purpose is exactly to handle Greek-specific rhyme
analysis:
•Syllabification.A number of Greek syllab-
ification rules are implemented. The algo-
rithm is designed to identify syllable bound-
aries based on Greek-specific phonotactic con-
straints.
•Stress Detection. We use the accent marks
found in Greek orthography ( ά,έ,ή,ί,ό,ύ,ώ)
in orcer to identify stress. In case of words
with clitics (notably weak object pronouns,
2Dataset available at: https://github.com/
StergiosChatzikyriakidis/Modern_Greek_
Literature/tree/v1.e.g., κάλεσέ με ), we have a mechanism to
handle stress domain extension.
•Rhyme Domain Extraction. The domain
of the rhyme extends from the stressed vowel
until the end of the phonological phrase. In the
case of Mosaic rhymes, it can span multiple
orthographic words.
•Phonetic Transcription. We convert Greek
orthography to a phonetic representation.
4.2 Rhyme Classification Module
We develop a classification module based on the
phonological module. It compares rhyme do-
main pairs and assigns labels. More specifically it
checks:
1.Position Classification: Here it determines
where the stress falls and classifies as
M/F2/F3.
2.Perfect Match Check: In this part, the post-
stress material is checked to see if it is an exact
match.
3.Feature Detection: This checks for RICH
rhyme (onset matching), IDV rhyme (pre-
stress vowel) and/or Mosaic (MOS) rhyme
(word boundary crossing).
4.Imperfection Analysis: In case the rhyme
is not perfect, classify accordingly (IMP-V ,
IMP-C, IMP-0F, IMP-0M).
The output result is a compound label such as
F2-TR-S-IDV (Feminine-2, Total Rich Singleton,
Identical pre-rhyme V owel).
4.3 LLM Integration
We integrate a number of LLM providers through
a unified API layer, including Anthropic (Claude
Sonnet 3.7/4.5), OpenAI (GPT-4o), Gemini and
Open Models (Llama 3.1 8b and 70b and Mistral
Large). We use both open and closed models and
models of varying sizes and reported capabilities.
We implement five prompting strategies for
rhyme identification:
•Zero-Shot Structured: Provides the rhyme
taxonomy and requests analysis
•Zero-Shot CoT: Requests explicit reasoning
through detection steps
•RAG Augmentation: Retrieves relevant ex-
amples from our corpus

Phonological
Engine
Symbolic Rules
(Syllabification/Stress)Input Pair
LLM Analysis
Correct?
Validated LabelA. Identification
Prompt
LLM
Valid?
PoemFeedback GenB. Generation
Ground TruthDraft Verify
Y esNo
Figure 1: Hybrid system architecture.Left: Identification combines LLM predictions with Engine-generated ground
truth for validation.Right: Generation uses the Engine to verify and refine LLM outputs.
Algorithm 1Generate-Verify-Refine Loop
1:Input:Theme, rhyme_type, features, num_lines
2:Output:Phonologically valid poem
3:attempts←0
4:whileattempts <15do
5:poem←LLM.generate(prompt)
6:errors←verify_rhymes(poem)
7:iferrors=∅then
8:returnpoem
9:else
10:feedback←format_errors(errors)
11:prompt←update_prompt(feedback)
12:attempts←attempts+ 1
13:end if
14:end while
15:returnpoemwith warning
4.4 Agentic Generation Pipeline
For the task of rhyme generation, we implement an
agentic loop with phonological verification based
on our phonological verifier:
The verifier’s task is to check each rhyme pair
using the phonological rules and provide feedback
in case the rhyme does not comply with the rules.
5 Experimental Setup
We use a test set of 40 poems for rhyme identifica-
tion. This produces a total of 160 test cases in total
(2 strategies × 2 RAG configs × 40 poems). The
set has a balanced distribution across rhyme types
(13 Masculine, 16 Feminine-2, 11 Feminine-3), as
well as comprehensive coverage of rhyme features
including more rare types of ryme (21 PURE, 10
RICH, 10 IDV , 6 IMPERFECT, 5 MOSAIC). We
went for a balance that was big enough to makemeaningful claims, while, at the same time, main-
taining evaluation feasibility across 8 models and
4 configurations (1,280 total API calls).
For generation, we evaluate generation quality
on 26 test cases with specified rhyme constraints
(e.g., “Write a 4-line poem with F3-RICH rhyme
on theme: love”). Each test runs twice: once with
our verification loop (Generate-Verify-Refine) and
once without (pure LLM generation).
6 Results
6.1 Rhyme Identification Results
In Table 2, we see the results across all prompting
configurations.
Table 3 breaks down performance by rhyme type,
revealing systematic biases.
Table 4 shows feature detection accuracy for
each individual feature type.
A number of interesting findings are borne out
from our experiments. First, Claude 4.5 exhibits
large variation in performance (26.9% to 53.8%)
between non-CoT and CoT modes. This might
indicate that reasoning-heavy models may require
explicit prompting strategies.
Second, we find that the best performance from
proprietary models (Claude 4.5 using CoT RAG:
53.8%) is better than the best open model per-
formance (Mistral Large using Chain-of-Thought:
26.9%) by roughly 27 points. This is a substan-
tial difference that points to a large difference in
capability for low-resource phonological tasks.
Furthermore, we see that the size of the model

Model Structured Structured+RAG CoT CoT+RAG
Proprietary Models
Claude 4.5 53.8% 26.9% 46.2%53.8%
Claude 3.7 38.5%42.3%42.3% 30.8%
GPT-4o 7.7% 7.7%50.0%26.9%
Gemini 2.0 23.1%42.3%19.2% 11.5%
Open-Weight Models
Mistral Large 11.5% 15.4%26.9%11.5%
Llama 3.1 70B23.1%19.2% 11.5% 15.4%
Llama 3.3 70B23.1%23.1% 7.7% 7.7%
Llama 3.1 8B 7.7% 15.4% 7.7% 3.8%
Table 2: Rhyme identification accuracy (%) across configurations. Bold means best configuration per model.
Model M (n=32) F2 (n=40) F3 (n=32)
Claude 4.565.6%47.5% 21.9%
Claude 3.7 37.5%52.5%21.9%
GPT-4o 31.2% 25.0% 12.5%
Gemini 2.0 43.8% 20.0% 9.4%
Mistral Large 25.0% 17.5% 6.2%
Llama 3.1 70B 12.5% 32.5% 3.1%
Llama 3.3 70B 25.0% 12.5% 9.4%
Llama 3.1 8B 6.2% 12.5% 6.2%
Table 3: Identification accuracy by rhyme type. All
models struggle most with F3 (proparoxytone) rhymes.
plays a role. Llama 3.1 70B is significantly better
from Llama 3.1 8B (average accuracy 16.2% vs
8.8% across configurations), demonstrating that
the rhyme detection task in Modern Greek benefits
from larger models.
In terms of diffuclty with respect to stress, F3
rhymes seem to be the hardest, as all models
achieve lowest accuracy on F3 (proparoxytone)
rhymes, with most below 22%. In terms opf
prompting strategies, Chain-of-Thought is helps
GPT-4o substantiatlly, increasing its accuracy from
7.1% (Structured) to 50.0% (CoT). The same is
true for Claude 4.5 achieves peak overall accuracy
(53.8%) with CoT RAG prompting. Open-weight
models generally fail to leverage CoT effectively.
Mosaic rhymes seem to be the most difficult to
detect. No model achieved an exact feature match
for MOSAIC rhymes (0% accuracy). However,
Claude 4.5 using CoT+RAG successfully identi-
fied the MOSAIC feature in qualitative analysis,
marking it as the only model capable of this com-
plex phonological parsing.
6.2 Qualitative Analysis
To better the model’s capabilities and failure modes,
we analyzed 9 representative test cases, checking
the performance for all models and configurations(full outputs are provided in Appendix A):
1.Baseline Success (M-PURE): απαιτώ /λαμ-
πρό. This was correctly identified exactly by
Claude 4.5 (Structured+RAG). GPT-4o (CoT)
got the rhyme type right, but missed the exact
features. Llama 3.1 8B (all configurations)
failed this case across the board.
2.Baseline Success (F2-PURE): κρίνοι /κρίνει .
This ia a perfect homophone rhyme. It was
correctly identified by Llama 3.1 70B (Struc-
tured+RAG). Claude 4.5 (Structured) flagged
this as ’COPY’, conflating distinct-lemma ho-
mophony, with identical rhyme (repetition).
3.Structural Failure (F3): παράπονο /
άπονο . No model achieved a perfect match.
Claude 3.7 (Structured+RAG) and GPT-4o
(CoT+RAG) correctly identified the rhyme
type (F3). Claude 4.5 failed to identify the
rhyme type in all configurations.
4.Mosaic Failure: λυγμέ /για με . Cross-
boundary rhyme. Claude 3.7 (Struc-
tured+RAG) was the only configuration to get
a perfect match. Claude 4.5 (CoT+RAG) suc-
cessfully identified the MOSAIC feature, but
added extraneous tags. Open models consis-
tently failed.
5.Feature Hallucination (Rich): αφρός /
εμπρός . Correctly identified by Claude
4.5 (CoT+RAG) and Gemini 2.0 (Struc-
tured+RAG). Claude 4.5 (Structured+RAG)
hallucinated a RICH tag in zero-shot. How-
ever, the CoT configuration corrected it by
analyzing the phonetics (/fr/ vs /pr/).
6.Archaic Language Failure (curse of the
dative: Ελληνίς /ουρανοίς . This includes

Model PURE (n=68) RICH (n=24) MOSAIC (n=4) IDV (n=24) IMP (n=8)
Mistral Large 4.4%20.8%0.0% 12.5% 12.5%
Llama 3.1 70B 5.9% 16.7% 0.0% 12.5%37.5%
Claude 3.7 8.8% 12.5% 0.0% 0.0% 25.0%
Claude 4.5 5.9% 8.3% 0.0% 4.2% 25.0%
Llama 3.3 70B 2.9% 8.3% 0.0% 0.0%37.5%
Gemini 2.0 5.9% 0.0% 0.0% 0.0%37.5%
GPT-4o 5.9% 0.0% 0.0% 0.0% 25.0%
Table 4: Feature detection accuracy by individual feature type.MOSAIC rhymes are never detectedby any model.
RICH and IDV detection remains below 21%.
Model No Verify With Verify Improvement
Claude 4.5 0.0% 34.6% +34.6%
Claude 3.7 3.8%73.1%+69.3%
GPT-4o 0.0% 42.3% +42.3%
Table 5: Generation validity (% of perfectly valid po-
ems) with and without phonological verification loop.
Katharevousa forms, in specific a nominative
inίςthat rhymes with a plural dative in οίς
(both morphological forms largely absent in
SMG). Surprisingly, Llama 3.3 (Structured)
was the only model to perfectly identify this
rhyme. The rest struggled with the archaic
spelling variances.
7.Imperfect Detection: γρήγορο /είσοδο . Im-
perfect F3 rhyme. No model perfectly cap-
tured the full feature set. Claude 4.5 (CoT)
and Claude 3.7 (Structured) correctly identi-
fied the rhyme type (F3) but missed the spe-
cific imperfection details.
8.Proper Noun Distraction: νιότα /Ευρώτα .
Llama 3.1 70B (Structured) outputted a for-
mat error (’STEP’). Claude 4.5, GPT-4o, and
Gemini 2.0 generally handled the entity cor-
rectly in CoT modes, identifying the rhyme
type (F2), besides the challenging synizisis of
the first word.
9.Visual vs Phonetic: ορθός /φως. Only
Claude 4.5 (CoT+RAG) got the rhyme cor-
rect, a fact that might point to more robust
grapheme-to-phoneme mapping capabilities
despite the visual mismatch.
6.3 Rhyme Generation Results
Table 6 shows generation validity broken down by
feature complexity.
The verification loop greatly enhances rhyme
validity across all models. Claude 3.7 achieves theFeature Type nClaude 3.7 Claude 4.5 GPT-4o
BASIC (no features) 6100.0%83.3% 90.9%
IMPERFECT 2 50.0% 50.0% 33.3%
IDV+PURE 2 50.0% 0.0% 66.7%
IDV+RICH 2 50.0% 0.0% 0.0%
IDV+IMPERFECT 2100.0%0.0%100.0%
IDV+MOSAIC+PURE 2 0.0% 0.0% 0.0%
Table 6: Generation validity (%) with verification,
by feature complexity. BASIC rhymes (no features)
achieve highest success. Complex multi-feature com-
binations (e.g., IDV+MOSAIC+PURE) fail even with
verification.
highest verified generation rate (73.1% valid po-
ems), demonstrating that the hybrid approach suc-
cessfully compensates for LLM phonological weak-
nesses. Critically, pure LLM generation fails catas-
trophically (0-4% validity), confirming that deter-
ministic verification is essential for constrained po-
etry generation, at least for the models used.
6.4 Qualitative Analysis of Generation
Three main types of generation errors (full output
traces are provided in Appendix B) are generally
corrected by the loop:
1.Correcting the stress pattern(F3 vs F2):
When asked for F3 (proparoxytone) rhymes,
models are often incorrectly defaulting to F2
(paroxytone).Example: GPT-4o initially gen-
erated αναβιώνει /απλώνει (F2). The verifier
returned "Stress mismatch: Expected F3,
found F2" . The model corrected this to a
valid F3 rhyme in the subsequent iteration.
2.Feature Precision (Rich vs Pure): LLMs
often treat rhyme types loosely. When M-
PURE was requested, Claude 3.7 generated
σκληρό /θησαυρό (which is RICH, sharing
the /r/ onset). After the intervention of the
loop, the model successfully refined the output
to a strict PURE rhyme, demonstrating the

system’s ability to enforce precise stylistic
constraints.
3.Non-rhymes: In some cases, Claude 4.5 pro-
posed non-rhyming outputs, e.g. pairs like
αγαπώ /ξέρω . The verification loop noted
these invalid pairs ( "No rhyme" ), and forced
the model to regenerate valid phonological
matches.
7 Discussion
7.1 Why LLMs Struggle with Rhyme
One issue is potentially tied to the nature of LLM
tokenizers (e.g., BPE, WordPiece), i.e. that fact that
they segment text based on statistical co-occurrence
and not linguistically motivated units. In Greek,
this often results in words being split mid-syllable
or mid-grapheme-cluster, entirely obscuring the
phonological structure required for rhyme. For ex-
ample, a word like παράπονο (pa-’ra-po-no) might
be tokenized as [ πα,ρά,πο,νο] in an ideal case,
but often appears as [ παρ,άπ,ονο] depending on
the vocabulary, stripping the model of the ability
to map the stress position (F3) relative to the final
syllable.
Unlike orthographical systems where text maps
close to 1:1 to sound, Greek’s retention of historical
orthography, i.e. a system of orthography that has
been retained for millennia and, thus, has not fol-
lowed the changes in the language, features many-
to-one mappings (e.g., the sound /i/ can be spelled
asι, η, v, ϵι, oι, vι ). LLMs trained primarily on text
often rely on visual similarity ("eye rhyme") rather
than phonetic identity. This explains why models
fail on κρίνοι /κρίνει (visual mismatch but phonetic
match) while hallucinating rhymes for ορθός /φως
(visual mismatch and phonetic mismatch).
Finally, note that Greek rhyme is strictly defined
by stress position (M, F2, F3). Since LLMs lack
an explicit prosodic module, they are prone to fail
when attempting to distinguish minimal pairs that
differ only in stress (e.g.,νόμοςvsνομός).
7.2 Creativity vs. Hallucination
Our analysis of "hallucinated" words reveals a nu-
anced trade-off between semantic grounding and
poetic creativity. We observe two distinct cate-
gories of invented vocabulary.
Some of the hallucinations can be very well-
taken as poetic neologism. For example, Claude
4.5 gives us αιωνημένα (eternal-ized) and θάρραμα
(courage-thing), which not only valid phonotacticderivational rules but also are meaningful enough
neologisms (for eample αιωνημένα can be an ad-
jective that means something related to enternity,
while θάρραμα can be seen as a collage of θάρρος
(courage) and χάραμα (dawn). A striking exam-
ple is πυραφί (fire-colored), which appears to be
constructed by analogy to χρυσαφί (gold-colored),
demonstrating a deep (if unauthorized) grasp of
Greek morphology.Similarly, αγέρη appears as a
valid homophone of αγέρι (breeze), suggesting a
spelling variation rather than a failure. Nonsense
Failures, on the other hand, are phonotactically cor-
rect words but are difficult to coerce into a mean-
ingful word. Examples include τσίγκλο andσκάμ-
πουνε , which lack semantic transparency. It is quite
positive that no phonotactic violations of Greek
happen in this cases (modulo the total English non-
sense generated by some Llama models).
We find a clear, we would dare to call it a
“personality” difference in models: GPT-4o is the
"safest" model, producing almost zero nonsense
but also fewer neologisms. In contrast, Claude 4.5
is the boldest, with a high rate of invention, the
majority of which are plausible neologisms rather
than nonsense. This suggests that what is often pe-
nalized as "hallucination" in factual tasks can serve
as a proxy for creativity in poetic tasks.
Crucially, this higher level of invention coincides
with stronger performance in the verification loop.
It appears that the strict constraints imposed by
the verifier push capable models to neologize in
order to conform to the poetic form, inventing new
words when standard vocabulary fails to meet the
phonological requirements.
7.3 Benefits of Hybrid Architecture
The hybrid approach succeeds by decoupling rea-
soning from phonology. The deterministic engine
provides the ground truth that the neural model
optimizes towards. This "Sandwich" architecture
(Prompt→LLM→Verify→Refine) allows us
to leverage the semantic richness and creativity of
large language models while imposing the strict
phonological constraints required by the poetic
form. By offloading the "scoring" of a rhyme to a
precise symbolic module, we free the LLM to focus
on lexical selection and thematic coherence, effec-
tively bypassing its inherent blindness to sub-token
phonological structures.

8 Future Work
The natural next step for such a system is to go
beyond rhyme and implemenent metrical verifica-
tion. Greek poetry relies heavily on stress-timed
constituent meters (e.g., iambic 15-syllable verse)
and systems that could handle both rhyme and met-
rical structure would be a very interesting research
direction to take.
Additionally, we plan to explore Reinforcement
Learning from Phonological Feedback (RLPF). In-
stead of a simple rejection sampling loop at in-
ference time, the signals from our deterministic
verifier could be used as a reward function to fine-
tune a smaller model (e.g., Llama 8B), potentially
internalizing phonological constraints directly into
the model’s weights.
Finally, we would hope to extend this hybrid
neuro-symbolic approach beyond Greek to other
low-resource languages, the goal being to create
a generalized "Universal Rhyme Engine" that re-
quires only a language-specific phonological rule
set.
9 Conclusion
In this work, we presented a hybrid neuro-symbolic
system designed to bridge the gap between Large
Language Models and strict phonological con-
straints in Modern Greek poetry. Our experiments
reveal that while LLMs possess latent creative ca-
pabilities, they fundamentally struggle with the
precise phonological computations required for rig-
orous rhyme detection and generation, particularly
in lower-resource languages. By integrating a de-
terministic phonological engine with an agentic
generation loop, we demonstrated a dramatic im-
provement in generation validity, raising success
rates from a baseline of under 4% to 73.1%. Fur-
thermore, our identification benchmarks exposed
a significant "Reasoning Gap," where only the
most advanced reasoning models (using Chain-of-
Thought) could compete with symbolic verification.
We hope that our released codebase and the curated
corpus of 40,000+ Modern Greek rhymes will serve
as foundational resources for future research into
phonologically-aware NLP, suggesting that hybrid
architectures remain essential for mastering the
structural nuances of poetic form.
Limitations
The size of our filtered corpus, while ensuring high
phonological quality, remains significantly smallerthan equivalent datasets available for high-resource
languages like English. Second, the proposed hy-
brid architecture imposes a computational over-
head; the iterative nature of the Generate-Verify-
Refine loop inevitably increases the total genera-
tion time compared to standard single-pass LLM
inference.
Acknowledgments
We gratefully acknowledge the Centre for the
Greek Language ( Κέντρο Ελληνικής Γλώσσας )
for providing access to the Anemoskala corpus and
granting permission to derive our rhyming dataset
from their digital resources. The original corpus
is available through the Portal for the Greek Lan-
guage (www.greek-language.gr).
References
Jonas Belouadi and Steffen Eger. 2023. ByGPT5:
End-to-end style-conditioned poetry generation with
token-free language models. InProceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
7364–7381, Toronto, Canada. Association for Com-
putational Linguistics.
Marjan Ghazvininejad, Yejin Choi, and Kevin Knight.
2016. Generating topical poetry. InProceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing, pages 1183–1191.
Thomas Haider and Jonas Kuhn. 2018. Supervised
rhyme detection with siamese recurrent networks. In
Proceedings of the Workshop on Stylistic Variation,
pages 81–86.
Jing He, Ming Zhou, and Long Jiang. 2012. Generating
chinese classical poems with statistical machine trans-
lation models. InProceedings of the Twenty-Sixth
AAAI Conference on Artificial Intelligence, pages
1650–1656.
Hussein Hirjee and Daniel Brown. 2010. Automatic
detection of rhyme in rap music. InProceedings of
the 11th International Society for Music Information
Retrieval Conference, pages 395–400.
Jey Han Lau, Trevor Cohn, Timothy Baldwin, Julian
Brooke, and Adam Hammond. 2018. Deep-speare:
A joint neural model of poetic language, meter and
rhyme. InProceedings of the 56th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1948–1958, Melbourne,
Australia. Association for Computational Linguistics.
Andrei Popescu-Belis, Àlex R. Atrio, Bastien Bernath,
Etienne Boisson, Teo Ferrari, Xavier Theimer-
Lienhard, and Giorgos Vernikos. 2023. GPoeT: a

language model trained for rhyme generation on syn-
thetic data. InProceedings of the 7th Joint SIGHUM
Workshop on Computational Linguistics for Cultural
Heritage, Social Sciences, Humanities and Litera-
ture, pages 10–20, Dubrovnik, Croatia. Association
for Computational Linguistics.
Sravana Reddy and Kevin Knight. 2011. Unsupervised
discovery of rhyme schemes. InProceedings of the
49th Annual Meeting of the Association for Compu-
tational Linguistics: Human Language Technologies,
pages 77–82.
Nina Topintzi, Konstantinos Avdelidis, and Thomai
Valkanou. 2019. Quantifying greek rhyme. InSe-
lected Papers from the 23rd International Sympo-
sium on Theoretical and Applied Linguistics, pages
429–447. School of English, Aristotle University of
Thessaloniki.
Ξενοφών Κοκόλης . 1993. Η ομοιοκαταληξία :Τύποι
και λειτουργικές διαστάσεις.Στιγμή, Athens.

A Appendix: Detailed Qualitative Results
This appendix presents the full raw outputs for the 9 representative test cases discussed in the Qualitative
Analysis. For each case, we show the predicted Rhyme Type and Features for all 8 models across 4
configurations (Structured vs CoT, No RAG vs RAG).
Legend:✓= Correct Type & Features,☞= Correct Type but Wrong Features,✗= Incorrect Type.
A.1 1. Baseline Success (M) (Keyword: ’απαιτώ’)
Poem:΄Ετσι από σένα περιμένω κι απαιτώ. /της Τραγωδίας τον Λόγο τον λαμπρό—
True Label: M [’PURE’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞M ✓M [PURE] ✗S ☞M
Claude 3.7 ✗MISS ✗MISS ☞M ✗I
GPT-4o ✗MISS ☞M [IMPERFECT] ☞M ✗S
Gemini 2.0 ☞M [IMPERFECT] ☞M [IMPERFECT] ✗MISS ✗MISS
Llama 70B ☞M [IDV , RICH] ✗F2 [PURE] ✗F2 ✗+
Llama 3.3 ✗S ☞M [IMPERFECT] ✗MISS ✗OF
Mistral ☞M [IMPERFECT] ✗STRESS ✗S ✗IN
Llama 8B ✗S ✗MISS ✗WORKERJOE ✗ANN
A.2 2. Baseline Success (F2) (Keyword: ’κρίνοι’)
Poem: Από ρουμπίνια ρόδα ,από μαργαριτάρια κρίνοι , /από αμεθύστους μενεξέδες .Ως αυτός τα κρίνει ,
True Label: F2 [’RICH’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞F2 [COPY , RICH] ✗BETWEEN ✗S ☞F2
Claude 3.7 ☞F2 [COPY] ☞F2 [COPY] ✗M ✗TR [CC]
GPT-4o ✗S ✗S ☞F2 ✗MASCULINE
Gemini 2.0 ✗MISS ✗MISS ✗S ✗MISS
Llama 70B ☞F2 [2W, C2, IDV , RICH] ✓F2 [RICH] ✗S ✗TR
Llama 3.3 ✗MISS ☞F2 [PURE] ✗M ✗MISS
Mistral ✗STRESS ☞F2 [CC, RICH] ✗S ✗2
Llama 8B ✗S ✗RECEIVE ✗CLIENT ✗M
A.3 3. Structural Failure (F3) (Keyword: ’παράπονο’)
Poem:αρχίζει το μωρό ένα παράπονο, /που ήρθε σ’έναν κόσμο τόσον άπονο.
True Label: F3 [’PURE’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ✗MISS ✗IN ✗S ✗NOT
Claude 3.7 ✗MISS ☞F3 [MOSAIC] ☞F3 ☞F3 [IMPERFECT]
GPT-4o ✗MISS ✗MISS ✗MATCH ☞F3 [IMPERFECT, RICH]
Gemini 2.0 ✗MISS ☞F3 [IDV , RICH] ✗MISS ✗F2
Llama 70B ✗MISS ✗F2 [PURE] ✗F2 ✗M
Llama 3.3 ✗M [C1, RICH] ✗MISS ✗S ✗MISS
Mistral ✗STRESS ✗F2 [IDV , PURE] ✗S ✗MISS
Llama 8B ✗K ✗F2 [IDV , RICH] ✗S ✗ZIGSPACE
A.4 4. Mosaic Failure (Keyword: ’λυγμέ’)
Poem:του αθηναίικου εσύ χινόπωρου λυγμέ, /ψιχάλα κυνηγάρα,που έβρεχες για με;
True Label: M [’MOSAIC’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞M [COPY] ☞M [COPY] ☞M [COPY] ☞M [MOSAIC, PURE]
Claude 3.7 ☞M [IMPERFECT] ✓M [MOSAIC] ✗S ☞M [PURE]
GPT-4o ✗S ✗MISS ☞M [COPY] ☞M
Gemini 2.0 ✗MISS ☞M [IMPERFECT] ✗MISS ☞M
Llama 70B ✗S ✗MISS ✗S ✗NO
Llama 3.3 ✗GIVEN ☞M [IMPERFECT] ✗S ✗MISS
Mistral ✗STRESS ✗STRESS ☞M [IMPERFECT, RICH] ☞M [IMPERFECT]
Llama 8B ☞M ✗MISS ✗SATURDAY ✗S

A.5 5. Feature Hallucination (Keyword: ’αφρός’)
Poem: Γιασεμιά ,και κοράκια .Και των άσπρων ο αφρός /και του μαύρου η φοβέρα πάντα εντός μου κι
εμπρός.
True Label: M [’PURE’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞M [IMPERFECT] ☞M [C1, RICH] ☞M ✓M [PURE]
Claude 3.7 ✗S ☞M [RICH] ☞M ✗IN
GPT-4o ✗S ✗IN ☞M ✗IN
Gemini 2.0 ☞M [C2, RICH] ✓M [PURE] ☞M ✗S
Llama 70B ☞M [IMPERFECT] ✗MISS ✗[IA] ✗F2
Llama 3.3 ✗S ✗OF ✗IN ☞M
Mistral ✗STRESS ✗MISS ☞M [F2, IMPERFECT] ☞M [IMPERFECT]
Llama 8B ✗MISS ✗F2 [MOSAIC] ✗STRING ✗COMPLETE
A.6 6. Archaic Failure (Keyword: ’Ελληνίς’)
Poem:Την εγέννησεν είς δήμος,μία πόλις Ελληνίς, /αλλ’ευθύς εκείνη έπτη,κι έστησεν εν ουρανοίς
True Label: M [’RICH’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ✗MISS ✗S ☞M ☞M [PURE]
Claude 3.7 ✗F2 [RICH] ☞M [PURE] ✗S ☞M
GPT-4o ✗OF ✗OF ☞M [IDENTICAL] ✗S
Gemini 2.0 ☞M [IMPERFECT] ✗MISS ☞M ✗MISS
Llama 70B ✗ING ✗MISS ☞M ✗NO
Llama 3.3 ✓M [RICH] ✗F2 [PURE] ✗BOTH ✗OF
Mistral ✗MISS ✗F2 [C1, IMPERFECT, RICH] ✗S ✗MISS
Llama 8B ✗IRSECURITY ✗DA V ✗CHOOSING ✗MISS
A.7 7. Imperfect Detection (Keyword: ’γρήγορο’)
Poem:αόριστη,με διάβα γρήγορο, /Στου καφενείου την είσοδο
True Label: F3 [’IMP’, ’C’, ’IMPERFECT’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ✗S ✗S ☞F3 [IMPERFECT] ✗IMPERFECT
Claude 3.7 ☞F3 [C1, IMPERFECT, RICH] ✗S ✗F2 ✗I
GPT-4o ☞F3 [IMPERFECT] ✗F2 [IMPERFECT] ✗F2 [F2] ✗S
Gemini 2.0 ✗MISS ☞F3 [IMPERFECT] ✗MISS ✗F2
Llama 70B ✗MISS ✗F2 [IMPERFECT] ✗M [IMPERFECT] ✗S
Llama 3.3 ✗MISS ✗M [IMPERFECT] ✗MISS ✗MISS
Mistral ✗STRESS ✗STRESS ✗F2 [IMPERFECT] ✗STRESS
Llama 8B ✗F2 [2W, IDV] ✗MISS ✗D ✗S
A.8 8. Proper Noun Failure (Keyword: ’νιότα’)
Poem:Πολεμιστή,τα γαληνά σου νιότα/΄Οπου έπεσες,κλωνάρια κι απ’του Ευρώτα
True Label: F2 [’PURE’]
Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞F2 ✗STEP ☞F2 ☞F2
Claude 3.7 ☞F2 [IMPERFECT] ✗S ✗F3 ✗I
GPT-4o ✗S ✗IN ☞F2 ✗M [IDV , PURE]
Gemini 2.0 ✗MISS ✗MISS ☞F2 ☞F2
Llama 70B ✗MISS ✗MISS ☞F2 ☞F2
Llama 3.3 ✗MISS ✗THE ✗SINCE ✗BUT
Mistral ✗STRESS ✗MISS ✗S ✗3
Llama 8B ✗S ✗F3 [2W, CC, IDV , RICH] ✗THE ✗F3
A.9 9. Visual vs Phonetic (Keyword: ’ορθός’)
Poem:άξαφνα το παράθυρο και στάθηκα ορθός, /τις μυρωδιές,τα χρώματα και το ιλαρό το φως.
True Label: M [’MOSAIC’, ’IDV’]

Model Struct (No RAG) Struct (RAG) CoT (No RAG) CoT (RAG)
Claude 4.5 ☞M [COPY] ☞M [IMPERFECT] ☞M [IDV , IMPERFECT] ☞M [PURE]
Claude 3.7 ✗MISS ☞M [IMPERFECT] ✗I ✗IN
GPT-4o ✗S ✗S ✗MISS ☞M [PURE]
Gemini 2.0 ✗MISS ☞M [IMPERFECT] ✗MISS ✗MISS
Llama 70B ✗S ✗MISS ✗S ☞M [IMPERFECT]
Llama 3.3 ☞M [IMPERFECT] ☞M [IMPERFECT] ✗MISS ✗IN
Mistral ✗MISS ☞M [PURE] ✗S ✗ONSET
Llama 8B ✗MISS ✗THE ✗BETWEEN ✗S

B Appendix: Detailed Generation Traces
We present full interaction traces for the three classes of generation errors discussed in the Qualitative
Analysis. These examples demonstrate how the hybrid verification loop intercepts compliant but incorrect
outputs (Stress, Feature Precision) and hallucinations, guiding the model to phonologically valid results.
B.1 1. Stress Pattern Correction (GPT-4o)
Test Case: Theme "Sky" (ουρανός), Constraint:F3 (Proparoxytone) Rhyme.
The model initially defaults to the more common F2 pattern.
Initial Draft (Pure LLM):
Στο χέρι σου το μυστικό αναβιώνει,
και μέσα στα σκιερά πελάγη απλώνει.
Verifier Feedback:Stress mismatch: Expected F3, found F2 for ’αναβιώνει’/’απλώνει’.
Refined Output (Hybrid):
Ω θεέ μου,στον ουρανό να ψιθυρίσουμε,
τα όνειρά μας να πετάξουν να αγγίξουμε.
Analysis: The corrected rhymes ( ψιθυρίσουμε /αγγίξουμε ) correctly carry stress on the antepenultimate
syllable.
B.2 2. Feature Precision (Claude 3.7)
Test Case: Theme "Light" (φως), Constraint:M-PURE Rhyme.
The model generates a valid rhyme that violates the "Pure" constraint (by being "Rich").
Initial Draft (Pure LLM):
Θα λάμπει πάντα φωτεινό στον κόσμο το σκληρό,
γλυκαίνει κάθε σκοτεινή γωνιά σαν θησαυρό.
Verifier Feedback:PURE requested but ’σκληρό’/’θησαυρό’ is RICH (Onset: /r/).
Refined Output (Hybrid):
Το φως σαν κύμα της χαράς ξυπνά
τα σκοτεινά μου όνειρα φωτίζει καθαρά
Analysis: The refined pair ( ξυπνά /καθαρά ) has distinct onsets (/p/ vs /th/), satisfying the PURE constraint.
B.3 3. Hallucination Correction (Claude 4.5)
Test Case: Theme "Love" (αγάπη), Constraint:M Rhyme.
The model hallucinates rhymes with no phonetic overlap.
Initial Draft (Pure LLM):
Η καρδιά μου χτυπά δυνατά για σένα που αγαπώ,
κάθε στιγμή που περνά χωρίς εσένα είναι πικρή,το ξέρω.
Verifier Feedback:No rhyme: ’αγαπώ’ / ’ξέρω’.
Refined Output (Hybrid):
Μέσα στην καρδιά μου κρύβεται αγαπώ,
ένα όνειρο γλυκό που με κρατώ,
Analysis: The model successfully regenerates a valid rhyme pair (αγαπώ/κρατώ).
C Prompt Templates
See source code for full templates.