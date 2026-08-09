# An Approach for Embedding-Guided Function Reuse Detection in Embedded C Software

**Authors**: A A Talha Talukder, Omar Alam, Akramul Azim

**Published**: 2026-08-04 18:46:17

**PDF URL**: [https://arxiv.org/pdf/2608.04137v1](https://arxiv.org/pdf/2608.04137v1)

## Abstract
Reusing embedded software functions across products is economically valuable but technically difficult: the same functionality implemented for two different microcontroller platforms can be entirely incompatible at the hardware level, even when the functions score above 0.90 cosine similarity and both pass SonarQube quality checks. Static analysis tools were designed to measure code quality, not hardware-domain compatibility, and have no model of peripheral interfaces, hardware abstraction layer (HAL) dependencies, or register-map constraints. This paper presents a domain-aware retrieval-augmented generation (RAG) pipeline for embedded C software reuse detection that addresses the hardware-compatibility gap directly. The pipeline enriches each function by extracting its existing inline comments, call-graph context, and a project README before embedding it with eight backbone models (MiniLM, MPNet, BGE, E5, GraphCodeBERT, OpenAI text-embedding-3-small, LLaMA 3 8B, StarCoder2 3B) acting as feature extractors. Four hardware-compatibility validators---covering peripheral token overlap, parameter count parity, call-graph dependency overlap, and structural branching pattern---filter candidates directly in the retrieval stack. Evaluated on six public embedded C software projects (184 functions, 4,815 above-plateau pairs), the pipeline reveals that SonarQube produces a 93.6% false-positive rate as a reuse filter, with 83.5% of failures caused by hardware-environment mismatches that static analysis cannot detect. Manual verification of 40 rejected pairs confirms 97.5% validator accuracy, and a diagnostic rule-injection variant identifies the dominant failure categories (McNemar chi-squared~=~294.0, p~$<$~0.001).

## Full Text


<!-- PDF content starts -->

An Approach for Embedding-Guided Function Reuse
Detection in Embedded C Software
A A Talha Talukder∗, Omar Alam∗, Akramul Azim†
∗Trent University, Peterborough, Ontario, Canada
Email:{talha, omaralam}@trentu.ca
†Department of Electrical, Computer and Software Engineering, Ontario Tech University, Ontario, Canada
Email: akramul.azim@ontariotechu.ca
Abstract—Reusing embedded software functions across prod-
ucts is economically valuable but technically difficult: the same
functionality implemented for two different microcontroller
platforms can be entirely incompatible at the hardware level, even
when the functions score above 0.90 cosine similarity and both
pass SonarQube quality checks. Static analysis tools were designed
to measure code quality, not hardware-domain compatibility, and
have no model of peripheral interfaces, hardware abstraction
layer (HAL) dependencies, or register-map constraints. This paper
presents a domain-aware retrieval-augmented generation (RAG)
pipeline for embedded C software reuse detection that addresses
the hardware-compatibility gap directly. The pipeline enriches
each function by extracting its existing inline comments, call-graph
context, and a project README before embedding it with eight
backbone models (MiniLM, MPNet, BGE, E5, GraphCodeBERT,
OpenAI text-embedding-3-small, LLaMA 3 8B, StarCoder2
3B) acting as feature extractors. Four hardware-compatibility
validators—covering peripheral token overlap, parameter count
parity, call-graph dependency overlap, and structural branching
pattern—filter candidates directly in the retrieval stack. Evaluated
on six public embedded C software projects (184 functions,
4,815 above-plateau pairs), the pipeline reveals that SonarQube
produces a 93.6% false-positive rate as a reuse filter, with 83.5% of
failures caused by hardware-environment mismatches that static
analysis cannot detect. Manual verification of 40 rejected pairs
confirms 97.5% validator accuracy, and a diagnostic rule-injection
variant identifies the dominant failure categories (McNemar chi-
squared = 294.0, p<0.001).
Index Terms—embedded software, code reuse, SonarQube false
positives, hardware abstraction layer, code embeddings, retrieval-
augmented generation, domain-aware validation, threshold cali-
bration, diagnostic framework
I. INTRODUCTION
Embedded software reuse has long been recognised as a
strategy to reduce development cost and accelerate product
cycles in the embedded systems industry [1], [2]. When a com-
pany produces multiple products on the same microcontroller
family, e.g., a microwave oven controller and an electric water
heater, both running on Texas Instruments Tiva C, it is natural
to ask whether software functions developed for one product
can be safely reused in another. However, designing software
to facilitate modular reuse remains a challenging software
engineering problem [3]. Reusing a well-tested sensor driver or
peripheral-initialisation routine would reduce duplicated effort
and inherit bug fixes already validated in the source project.
However, embedded software reuse is qualitatively harder
than reuse in general-purpose software [1], [4]. Unlike a sortingalgorithm or a string formatter, an embedded function’s correct-
ness is tightly coupled to the specific hardware it controls: its tar-
get microcontroller, its peripheral register map, and the vendor
hardware abstraction layer (HAL) it depends on. A function that
drives GPIO Port F through TivaWare’s GPIO_writePort()
cannot be substituted for a function that drives GPIO Port C
through ARM’s HAL_GPIO_WritePin() , even when both
implement identical logical behaviour. This coupling creates
three systematic obstacles [4]–[6].First, tight hardware cou-
pling:a software function’s correctness depends on the exact
peripheral it controls, not merely its logic.Second, HAL frag-
mentation:vendor layers—TivaWare, ARM CMSIS, ARM LL,
ESP-IDF, Zephyr—provide overlapping but incompatible APIs,
so functions with identical intent are not interchangeable across
projects.Third, documentation scarcity:embedded codebases
carry terse inline comments and product-level READMEs,
depriving code-embedding models of the natural-language
signals on which they were trained [7], [8].
The state of the art offers two kinds of tools for identifying
reuse candidates: static analysis tools such as SonarQube [9],
[10], and embedding-based code-similarity search [11], [12].
SonarQube is widely used as a quality gate in industrial
development pipelines [10]: functions that pass its checks are
considered clean enough to consider for reuse. Embedding-
based search scores function pairs by cosine similarity over
learned code representations, surfacing structurally and se-
mantically similar candidates. Both tools are effective in their
intended domains—but neither was designed to reason about
hardware-domain compatibility. SonarQube checks cyclomatic
complexity, null-pointer risks, and coding standards [9]; it has
no model of which peripherals a function touches, which HAL
it depends on, or what register space it occupies. Embedding
models assign high similarity to functions that share logical
structure and natural-language comments, regardless of whether
they target the same or different hardware interfaces [13].
As a consequence, two functions that are entirely hardware-
incompatible can simultaneously pass SonarQube and score
above 0.90 cosine similarity—and no existing tool will flag
this mismatch before the ported code fails at hardware test.
Despite this known limitation, no prior work has empirically
measured how severely SonarQube’s quality-only model fails
as a reuse-compatibility filter for embedded C software. We
address this gap with a domain-aware retrieval-augmented
arXiv:2608.04137v1  [cs.SE]  4 Aug 2026

generation (RAG) pipeline for embedded C reuse detection that
integrates hardware-compatibility validation directly into the
retrieval stack, rather than checking compatibility post-hoc. The
pipeline enriches each software function with inline comments,
call-graph context, and a project README into a single docu-
ment DOC( f), which is then embedded using eight backbone
models spanning compact encoders [14], [15], retrieval-tuned bi-
encoders [16], [17], a code transformer [12], and large language
models [18], [19]. Four hardware-compatibility validators—
Venv(peripheral token overlap), Vsig(parameter count parity),
Vcall(call-graph dependency overlap), and Vstruct (structural
branching pattern)—filter candidates directly in the retrieval
stack. An unsupervised plateau analysis calibrates a per-model
cosine threshold without requiring labelled data. A basic RAG
run is performed first; the validator failures from that run are
clustered to derive a rule library L, which a dynamic-rule RAG
variant then injects to expose which failure categories drive
retrieval quality.
We evaluate the pipeline on six public embedded C software
repositories across three domains (184 functions, 4,815 above-
plateau pairs). The central empirical finding is that SonarQube
produces a93.6% false-positive ratewhen used as a reuse
filter: of 1,494 pairs it approves as quality-clean, 1,399 are
rejected by our domain validators, with 83.5% of failures caused
by hardware-environment mismatches that static analysis cannot
detect. Manual verification of 40 randomly selected rejected
pairs confirms 97.5% validator accuracy.
The main contributions of this paper are as follows:
•We propose a domain-aware RAG pipeline for embedded
C software reuse detection that integrates four hardware-
compatibility validators directly into the retrieval stack,
enabling hardware-aware candidate filtering without labelled
training data (Section IV).
•We introduce four embedded-domain validators— Venv,Vsig,
Vcall, and Vstruct —that together capture peripheral identity,
interface shape, dependency burden, and structural branching
pattern, providing explainable rejection signals for each
incompatible pair (Section IV-D).
•We provide the first empirical quantification of SonarQube’s
false-positive rate as a reuse-compatibility filter for embedded
C software: 93.6%, with 83.5% of failures attributable to
hardware-environment mismatches that static analysis is
structurally unable to detect (Section III).
•We evaluate the pipeline across eight embedding backbones
spanning four architectural families, demonstrating that
the 93.6% false-positive finding holds across all models
(per-model rates 89.8%–97.3%), and identify BGE-small,
LLaMA 3 8B, and MPNet as the top-performing backbones
for this task (Section VI-B).
•We develop a dynamic-rule RAG diagnostic variant that
exposes hardware-token and call-graph mismatch as the
dominant retrieval failure categories, confirmed by a sta-
tistically significant McNemar test ( χ2= 294.0 ,p <0.001 ,
Section VI-D).
The remainder of this paper is organised as follows.Section II surveys related work. Section III presents the
empirical motivation. Section IV describes the methodology
and algorithms. Section V reports the experimental setup.
Section VI answers each research question. Section VII
discusses implications and threats. Section VIII concludes.
II. RELATEDWORK
In contrast to existing approaches that primarily rely on
code similarity or static analysis, we propose a domain-aware
RAG pipeline for embedded C software reuse detection that
integrates hardware-compatibility validators for explainable
candidate filtering. Our approach captures hardware-driven
incompatibilities, enabling more reliable reuse detection without
requiring labelled training data. Below, we discuss some of
the related work to our approach.
Embedded software reuse.Maruf et al. [1] extract reusable
functions via static call-graph analysis; Talukder et al. [20]
use LLMs for feature extraction; FeaMod [2] targets modu-
larity. None quantify static-tool false-positive rates or propose
hardware-compatibility validators. AutoFirm [21] finds that
67.3% of IoT vendors fail to update reused libraries, confirming
that compatibility checking is largely absent from current
practice.
Static analysis limitations.Lenarduzzi et al. [9], [22] show
SonarQube rules have statistically significant but small effects
on fault-proneness across 33 Apache projects. Sadowski et
al. [10] establish 10% as the industrial adoption floor above
which FP rates collapse developer trust. Charoenwet et al. [23]
report ≥76% of SAST warnings in C/C++ vulnerability
detection are irrelevant—the closest published analog to our
93.6% finding. Cui et al. [24] catalogue SonarQube FP root
causes but do not include hardware-environment mismatch.
Johnson et al. [25] identify false positives as the dominant
barrier to ASAT adoption.
Code quality vs. reusability.Papamichail et al. [26] show that
standard static metrics (complexity, coupling, cohesion) do not
predict reuse rates; Mehboob et al. [27] confirm they measure
quality proxies rather than portability. In embedded software the
disconnect is sharper: a well-structured HAL function couples
tightly to a specific peripheral and is thereforelessportable
than a structurally messier hardware-free helper.
Code clone and similarity detection.SourcererCC [28]
demonstrates that token-bag Jaccard scales to 250 million
lines for Types 1–3 clones, underpinning our VenvandVstruct
validators. The contribution is not the Jaccard mechanic but the
vocabulary: hardware-identifying tokens ( Πhw) turn a generic
similarity measure into a domain-aware compatibility check.
CCGraph [29] and FA-AST [30] apply graph-neural networks
to structural similarity, motivating our lightweight branching-
pattern fingerprint.
Pre-trained models for code.CodeBERT [11] and Graph-
CodeBERT [12] advance code search on mainstream languages;
BGE [16] and E5 [17] optimise retrieval via contrastive pre-
training. Muennighoff et al. [13] show similarity distributions
vary by up to 0.3 cosine units across backbones, motivating

TABLE I
CLASSIFICATION OF4,815ABOVE-PLATEAU PAIRS BYSONARQUBE AND
VALIDATOR OUTCOME.
Group Sonar Validators Count %
A Pass Pass 95 1.97
B Pass Fail 1,399 29.06
C Fail Pass 33 0.69
D Fail Fail 3,288 68.29
Total 4,815 100.0
our per-model plateau calibration rather than a fixed global
threshold.
RAG for code.CoCoMIC [31] reports +33.94% exact match
from cross-file call-graph context, directly motivating our
DOC( f) construction. DocPrompting [32] and ProConSuL [8]
show that retrieved documentation improves code generation
and summarisation, motivating our README enrichment.
FirmUp [33] establishes that call-graph context is necessary
for accurate firmware function matching. Asteria-Pro [34]
achieves 91.65% precision on vulnerable IoT function de-
tection by combining deep-learning similarity with explicit
domain knowledge—the closest architectural analog to our
four-validator stack.
CheckList [35] and HANS [36] establish behavioral rule-
injection as a first-class research contribution for exposing
systematic model failures. Errudite [37] formalises error
analysis as a primary output. Our dynamic-rule RAG variant
follows this tradition: its value lies in diagnosing failure
categories, not in improving headline metrics.
III. MOTIVATION: SONARQUBE AS AREUSEFILTER
SonarQube checks cyclomatic complexity, null-pointer
risks, and coding standards—none of which capture
whether two functions touch compatible hardware periph-
erals. For example, when a function in one project uses
GPIO_writePort() and a function in another project uses
HAL_GPIO_WritePin() , both can pass SonarQube cleanly
even though they target entirely different register maps and
HAL layers. To make this gap concrete, we evaluated all 4,815
above-plateau function pairs from six embedded C repositories
using both SonarQube and our domain validators, classifying
each pair into one of four groups (Table I).
Table I classifies all 4,815 pairs into four groups based
on two independent judgments: SonarQube’s quality verdict
(pass/fail) and our validators’ compatibility verdict (pass/fail).
Group A pairs are approved by both — these are the genuine
reuse candidates. Group D pairs are rejected by both —
SonarQube and our validators agree they are unsuitable.
Groups B and C are the disagreements. Group C (33 pairs,
0.69%) represents cases where SonarQube flags code quality
issues but our validators confirm hardware compatibility — a
minor discrepancy. Group B (1,399 pairs, 29.06%) is the critical
case: SonarQube approves these pairs as quality-clean, yet our
hardware-compatibility validators reject every one of them.
These are the false positives — pairs that would be mistakenly
considered for reuse based on SonarQube alone.Group B is theTABLE II
VALIDATOR FAILURE REASONS FORGROUPB (1,399PAIRS).V env IS
INVOLVED IN83.5%OF ALL FAILURES.
Failure Reason Count %
Venv + Vcall 527 37.7
Venv + Vsig + Vcall 243 17.4
Venv only 228 16.3
Venv + Vsig 170 12.2
Vcall only 133 9.5
Vsig + Vcall 50 3.6
Vsig only 48 3.4
TABLE III
SONARQUBE STATUS OF THE128VALIDATOR-APPROVED PAIRS.
Sonar status Count %
Both functions Sonar-clean 0 0.0
One function Sonar-flagged 107 83.6
Both functions Sonar-flagged 21 16.4
false-positive problem.SonarQube approved |A|+|B|= 1,494
pairs as quality-clean. Our validators reject 1,399 of those. The
Sonar false-positive rate is:
Sonar FP rate=|B|
|A|+|B|=1,399
1,494= 93.6%(1)
This rate is 9.4× the 10% industrial adoption floor identified
by Sadowski et al. [10] and 17 percentage points above the
≥76% irrelevance rate for C/C++ vulnerability detection [23].
Table II breaks down why validators rejected Group B
pairs. Venv (hardware-token mismatch) is involved in
527+243+228+170 = 1,168 pairs, or83.5%of Group B fail-
ures. Among these,78.4%of Group B pairs have Venv-score=
0—zero hardware-token overlap—yet SonarQube passed all of
them. This confirms that static quality analysis cannot substitute
for domain compatibility checking in embedded software [5],
[6], [33].
Table III reveals a counterintuitive result: of the 128 validator-
approved pairs, none are entirely free of SonarQube warnings.
This suggests that code quality and hardware-compatibility
are not merely uncorrelated in this domain — they point in
opposite directions. A well-structured, warning-free HAL driver
is typically one that has been carefully optimised for a specific
peripheral interface, making it more hardware-specific and
therefore less portable to another project. Conversely, functions
with some code-quality issues (e.g., unused variables or non-
standard constructs) may be more generic in their hardware
dependencies, making them more portable [26], [27].
Manual verification.To confirm that validator rejections
reflect genuine hardware incompatibilities rather than valida-
tor errors, we manually inspected a random sample of 40
Group B pairs with ( Venv) equals zero — meaning the two
functions share no hardware-identifying tokens whatsoever
(Table IV).39 of 40 pairs (97.5%) were confirmed genuinely
hardware-incompatible.The remaining pair ( init_LCD vs.
LCD_INIT , pair 30) uses different GPIO port assignments on
different Tiva C boards; it is borderline—not directly portable

TABLE IV
REPRESENTATIVE MANUAL-VERIFICATION EXAMPLES FROMGROUPB
(VENV-SCORE= 0, SONAR-PASS, VALIDATOR-FAIL).
# Source Target Sim. Incompatibility
32state_A moistureSensor0.962 Microwave idle vs.
HV AC I2C sensor.
37ADC_Read LED_BUZZER0.960 ADC input vs.
GPIO output.
10LCD_write Generic_delay0.931 LCD GPIO vs. Sy-
sTick delay.
30init_LCD LCD_INIT0.713 Same LCD, differ-
ent GPIO port (bor-
derline).
Summary:39 incompatible, 1 borderline, 0 reusable.
Fig. 1. Seven-stage embedded C reuse detection pipeline. Stages: (1) six repos,
(2) regex extraction (184 fns), (3) RAG enrichment DOC( f), (4) eight embed-
dings, (5) plateau threshold τ∗, (6) four validators ( Venv,Vsig,Vcall,Vstruct ),
(7) Sonar diagnostic. Amber dashed arc: dynamic rule injection from Group B.
without pin-remapping. No pair was found to be directly
reusable without hardware modification. Critically, 12 of the
40 pairs had embedding similarity ≥0.90 yet were confirmed
incompatible, including state_A vs.moistureSensor
(sim = 0.962) and ADC_Read vs.LED_BUZZER (sim = 0.960).
These cases show that high semantic similarity isnot sufficient
for embedded reuse.
IV. METHODOLOGY
Overview.Fig. 1 shows the seven-stage pipeline and Fig. 2
shows the detailed component architecture. The pipeline takes
software repositories and an embedding model as input and
returns a validated set of reuse-compatible function pairs
together with a SonarQube diagnostic breakdown.
A. Dataset and Function Extraction
We evaluated six public GitHub repositories spanning three
embedded domains, each with two independent implementa-
tions (Table V). The domain pairing allows measurement of
both same-domain reuse (e.g., microwave →microwave alt)
Fig. 2. Full model architecture across four stages: corpus preparation with
DOC( f) construction, eight-backbone embedding and plateau calibration
(Alg. 2), four domain validators (Eqs. 2–5), and SonarQube diagnostic (Groups
A–D; 93.6% FP rate). Amber arc: Group B rule injection (Alg. 3).
TABLE V
FIRMWARE REPOSITORIES AND EXTRACTED FUNCTION COUNTS.
Project Domain Functions
microwave Microwave oven controller 41
microwave alt Microwave (alt. impl.) 35
water heater Electric water heater 32
waterheater alt Water heater (alt. impl.) 17
hvac HV AC controller 10
hvac alt HV AC (alt. impl.) 49
Total 184
and cross-domain reuse (e.g., microwave →hvac). A regex-
based extractor isolates each top-level C function definition
(return type, name, argument list, body, line range), yielding 184
functions in total. Each extracted function retains its original
source-level formatting and structure.
B. Retrieval-Augmented Function Documents
Embedding only the raw function body discards two cate-
gories of information critical for embedded reuse decisions.
First,hardware intentis frequently expressed only in inline
comments (e.g., “writes 0x01 to GPIO Port F to enable the
buzzer”)—without the comment, an embedding model sees
an assignment to an opaque register name. Second,context
captured in the call graph determines whether the dependency
burden of porting a function is low or high.
For each software function fwe construct an enriched
document DOC( f) by concatenating four text components
separated by a delimiter token: (i) the raw source code of
f; (ii) all inline and block comments within f’s line range;

TABLE VI
EMBEDDING BACKBONES EVALUATED.
Key Model Training
minilm MiniLM-L3 Self-attn distillation [14]
mpnet MPNet Masked+permuted
LM [15]
bge BGE-small-v1.5 Contrastive retrieval [16]
e5 E5-small Weakly supervised [17]
gcbert GraphCodeBERT Code+data-flow [12]
gpt text-emb-3-small OpenAI API [38]
llama LLaMA 3 8B LLM, mean-pooled [18]
starcoder StarCoder2 3B Code LLM [19]
(iii) a textual call-graph summary—the names of functions
fcalls and the names of functions that call f; and (iv) a
short excerpt from the project README describing the
system context (e.g., “Microwave oven controller running on
TM4C123GH6PM with 4 ×4 keypad, HD44780 LCD, and
piezo buzzer”). We pass DOC( f) to the embedding model. The
design is motivated by CoCoMIC [31] ( +33.94% exact match
from call-graph context), DocPrompting [32] (documentation
improves retrieval), and ProConSuL [8] (project-level context
aids LLM code summarisation).
C. Embedding Models
Table VI lists the eight backbones evaluated. All eight act
as feature extractors; they produce cosine similarity scores
over DOC( f) pairs but do not generate or modify code. The
multi-backbone design is essential because Muennighoff et
al. [13] show that similarity score distributions vary by up to
0.3 cosine units across models, so any single-model result risks
being model-specific. For large language models (LLaMA 3,
StarCoder2) we use mean-pooled last-layer representations; for
BERT-family models we use [CLS] pooling; for BGE/E5 we
use the model-default pooler.
D. Domain-Specific Validators
The four validators defined here identify Group B pairs
— approved by SonarQube but hardware-incompatible —
by checking four orthogonal aspects of compatibility that
SonarQube cannot model, encoding the hardware-domain
knowledge that pure semantic similarity lacks. A candidate pair
(fs, ft)is accepted only when it passesall foursimultaneously;
failure on any one is a diagnostic signal for why that pair
cannot be reused. The validators target four orthogonal aspects
of compatibility:
•Venv— Hardware/HAL footprint (Section IV-D1)
•Vsig— Parameter count parity (Section IV-D2)
•Vcall— Call-footprint similarity (Section IV-D3)
•Vstruct — Structural branching pattern (Section IV-D4)
1)Venv— Hardware/HAL Footprint:Intuition.Two em-
bedded functions are hardware-compatible only if they touch
similar peripherals — an ADC reader and a GPIO writer share
nothing at the hardware level, regardless of how similar their
branching patterns look. We capture this via Jaccard overlap
of each function’s hardware-identifying tokens.Definition.
LetH(f) be the set of hardware-identifying tokens in f’sbody, comments, and call-graph neighbourhood — drawn
from the pattern set Πhw:GPIO_ *,ADC_ *,UART_ *,SPI_ *,
PWM_ *,TIMER_ *,HAL_ *,LL_*, vendor register names
(e.g., PORTF ,GPIO_PORTC_DATA_R ), and HAL function
families (e.g., HAL_GPIO_WritePin ,gpio_set_level ).
H(f)answers:which hardware peripherals doesftouch?
Venv(fs, ft) =

1, H(f s) =∅ ∧H(f t) =∅,
0,exactly one ofH(f s),
H(ft)is∅,
|H(f s)∩H(f t)|
|H(f s)∪H(f t)|,otherwise.
(2)
The equation covers three cases. When neither function
touches hardware, both are pure logic helpers that move
freely between projects (score 1, accepted). When exactly one
does, substitution would silently drop hardware interactions
the destination may lack (score 0, always rejected). When
both do, the Jaccard score measures the overlap of their
hardware footprints, and the pair is accepted once at least
one token is shared (score >0), indicating at least partial
peripheral compatibility.Example. state_A hasH=∅ ;
moistureSensor hasH={ADC_Read ,ADC0_BASE ,
I2C_master_init. . .} . Exactly one is empty ⇒Case 2 ⇒
Venv= 0⇒rejected(sim = 0.962).
2)Vsig— Parameter Count Parity:Intuition.Drop-in reuse
requires both functions to accept the same number of arguments
— a mismatched count breaks every call site without rewriting
callers, a heavier change than copying the function body. This
validator checks argumentcountonly; type-level signature
matching is left as future work.Definition.Let #args(f) be
the number of formal parameters.
Vsig(fs, ft) = 1−|#args(f s)−#args(f t)|
max(#args(f s),#args(f t),1)(3)
The formula normalises the argument-count difference by the
maximum of the two counts, producing a score in [0,1] . Ac-
ceptance requires Vsig= 1, i.e., argument counts exactly equal
— the only condition under which drop-in reuse is possible at
every call site without modifying callers. Any score below 1
means the counts differ, and copying the function would break
call sites passing the wrong argument count. The max(. . . ,1)
guard prevents division by zero for no-argument functions,
which score Vsig= 1 and pass through to the remaining
validators.Example. BUTTON_READ(port, pin) has 2
args; ReadPin(pin_id) has 1. |2−1|/max(2,1,1) =
0.5⇒V sig= 0.5⇒rejected.
3)Vcall— Call-Footprint Similarity:Intuition.A function’s
non-HAL dependencies form part of its portability contract:
porting fsintoft’s project requires porting every non-HAL
helper fscalls that isn’t already present at the destination. The
call-graph overlap measures how much of this burdenf sand
ftalready share; HAL calls are excluded from C(f) since
they’re already captured by Venv, avoiding double-counting the

hardware-mismatch signal.Definition.Let C(f) be the set of
non-HAL functions called byf.
Vcall(fs, ft) =(
1C(f s)=∅ ∧C(f t)=∅
|C(f s)∩C(f t)|
|C(f s)∪C(f t)|otherwise(4)
A pair isacceptedwhen Vcall>0.Case 1: both are leaf func-
tions (call no user-defined helpers) and carry no transitive depen-
dency burden, so their portability contract is satisfied (score =
1), and they proceed to Vstruct for structural screening.Case 2:
Jaccard similarity over the non-HAL call sets — if one function
calls non-HAL helpers and the other calls none, the score is 0
and the pair is rejected, since porting the non-leaf function
would require transplanting all its dependencies into a destina-
tion that cannot satisfy them.Examples. C(Buzzer_ON) =
C(ArrayLED_ON) ={delay_ms} :Vcall= 1⇒ accepted.
C(state_A) ={setOutput} ,C(moistureSensor) =
{ADC_Read,ADC_Configure}:V call= 0⇒rejected.
4)Vstruct — Structural Branching Pattern:Intuition.Two
functions with very different branching structures — a tight
register-read helper versus a multi-state event handler — are
unlikely to be interchangeable even if they share hardware
context and call dependencies, which we capture via a count-
based comparison of branching keywords.Definition.Let
K(f) be themultisetof branching keywords in f’s body,
drawn from {if,else ,for,while ,do,switch ,case ,
return ,break ,continue} , recording counts so that three
ifs contributeif×3, distinguishing it from a singleif.
Vstruct(fs, ft) =(
1K(f s)=∅ ∧K(f t)=∅
|K(f s)∩K(f t)|
|K(f s)∪K(f t)|otherwise
(5)
Intersection uses element-wise minimum, union element-
wise maximum. A pair isacceptedwhen Vstruct ≥0
(always true) — the multiset representation lets three
ifs register as more committed to conditional dispatch
than one, a distinction the intersection count captures di-
rectly.Example. K(state_A) ={if× 2,return× 1};
K(moistureSensor) ={while× 1,return× 1}. Inter-
section = 1, union = 4,Vstruct = 0.25≥0 (diagnostic role,
not hard rejection).
E. Plateau-Based Threshold Calibration
Why per-model calibration is needed.Cosine similarity
is not comparable across embedding spaces. BGE-small’s
contrastive pretraining produces a well-separated distribution
where 0.71 already indicates a strong match; E5-small packs
most distinct pairs near 0.85, requiring ∼0.92 for equivalent
retrieval quality [13], [17]. A single global threshold would
flood tightly-packed models with false candidates or starve
well-separated models of true matches.
Definition.For each model we sweep N= 200 cosine
thresholds τuniformly from the observed τmintoτmax. At
eachτwe count all pairs satisfying both the similarity cut and
all four validator conditions:cnt(τ) =(
(fs, ft)sim(f s, ft)≥τ, V env>0, V sig= 1,
Vcall>0, V struct≥0)
(6)
Asτincreases from τmintoτmax, the count cnt(τ) passes
through three phases. First, it rises: low-similarity noise pairs
are filtered out as the threshold climbs. Second, it plateaus:
the similarity threshold is now strict enough that only the four
validators determine which pairs survive, not the cosine cutoff.
Third, it falls: the threshold has become so strict that even
genuinely hardware-compatible pairs are excluded. We select
the highest τat which the count is still at its maximum—
raising the cosine bar as far as possible without discarding any
validator-approved pair. This unsupervised procedure requires
no labelled ground truth and produces a threshold that is
calibrated to each model’s own similarity distribution. The
plateau threshold is the highest τstill achieving the maximum
count:
τ∗
model = max
τ∈T: cnt(τ) = max
τ′cnt(τ′)	
where Tis the 200-point sweep grid. If the count is strictly
decreasing (no plateau), the algorithm falls back to τ∗=τmax,
the most conservative cutoff (Algorithm 2, line 3).
Example.For BGE-small: τmin= 0.18 ,τmax= 0.94 .cnt(τ)
rises from 2 to a plateau of 23 at τ≈0.55 , holds through
τ= 0.70 , then drops at τ= 0.706 . Hence τ∗
bge= 0.706 .
MPNet’s plateau ends at 0.528; E5-small’s at 0.915. This 1.73×
spread confirms that a fixed global threshold is fundamentally
inappropriate.
F . RAG Variants and Sequential Rule Derivation
We compare two retrieval configurations on 8,136 percentile-
threshold candidates.
basic rag:code + call context + README; no rule injection.
This variant is always run first.
dynamic rule rag:derived from the output ofbasic rag. The
derivation follows three steps: (i) run basic rag and collect
all Group B pairs (Sonar-pass, Validator-fail); (ii) cluster the
validator failure reasons from those pairs and set rule weights
proportional to observed failure frequencies; (iii) build rule
library Land re-run the pipeline with rule injection. The four
rules in Land their weights—ordered by Group B failure
frequency—are: ENV MISMATCH( w= 0.18 , involved in
83.5% of Group B); CALL MISMATCH( w= 0.14 , 68.1%);
SIG MISMATCH( w= 0.08 , 36.5%); STRUCT MISMATCH
(w= 0.00 , absent from Group B). A pair’s adjusted score
under dynamic rule rag is:
score(f s, ft) = sim(f s, ft)−X
ρ∈Lwρ·1[ρmatches(f s, ft)]
(7)
Following CheckList [35] and HANS [36], this variant is
adiagnostic instrument—it exposes which failure categories
drive retrieval quality rather than claiming to improve overall
performance.

G. Algorithms
The pipeline described in Sections IV-A –IV-F involves
several interacting steps whose ordering is critical, particularly
the dependency of the dynamic-rule variant on a prior basic
RAG run. We formalise these steps as three algorithms to make
the procedure unambiguous and reproducible.
Algorithm 1 presents the complete end-to-end procedure. Its
most important design decision is the sequential ordering of
Stages 5a and 5b: the dynamic-rule variant cannot be executed
in isolation because its rule library Lis derived from Group B
pairs produced by an initial basic RAG run. Without this
explicit sequencing, the rule weights would have no empirical
grounding in the specific dataset being evaluated.
Algorithm 2 formalises the per-model plateau threshold
calibration introduced in Section IV-E . This algorithm is needed
because cosine similarity scores are not comparable across
embedding models: a threshold of 0.70 is conservative for
BGE-small but permissive for E5-small. The plateau sweep
identifies the highest threshold that still retains all validator-
approved pairs, without requiring any labelled data.
Algorithm 3 formalises the dynamic-rule re-scoring step.
It applies the penalty weights derived in Stage 5a to each
candidate pair and records which rules fired, producing the
diagnostic log that directly answers RQ4 about dominant failure
categories.Algorithm 1End-to-End Embedded C Reuse Detection
Input:ReposR; embedding modelM; RAG variant∈ {basic,dynamic}
Output:Validated pairsP∗; groups A–D; Sonar FP rate
Stage 1 — Extract functions and build documents
1:F ← ∅
2:foreach repor∈ Rdo
3:F r←EXTRACTFUNCTIONS(r)▷regex: name, body, line range
4:CPG r←BUILDCALLGRAPH(r)▷non-HAL calls only
5:README r←EXTRACTREADME(r)
6:foreachf∈F rdo
7:DOC(f)←BUILDDOC(f,CPG r,README r)▷code + cmt
+ CPG + README
8:F ← F ∪ {f}
9:end for
10:end for
Stage 2 — Embed and compute pairwise similarity
11:E← M({DOC(f) :f∈ F})▷one vector per function
12:Sim←cosine similarity over all ordered pairs(f s, ft),fs̸=ft▷
directed pairs
Stage 3 — Per-model threshold calibration
13:τ∗←PLATEAUTHRESHOLD(Sim,F)▷ Algorithm 2; no labels needed
Stage 4 — Filter high-similarity candidates
14:C ← {(f s, ft) : Sim(f s, ft)≥τ∗}▷∼33k pairs→ ∼600
Stage 5a — Derive rule library (dynamic only)
15:ifvariant=dynamicthen
16: Run Stage 6 once with variant=basic to obtain Group B pairs▷
Sonar-pass, Validator-fail
17:L ←DERIVERULES(Group B failure frequencies)▷weights from
Group B failure freq.
18:end if
Stage 5b — Dynamic rule scoring (optional)
19:ifvariant=dynamicthen
20:C ←DYNAMICRULESCORE(C,L, τ∗)▷Algorithm 3; re-score
with penalty weights
21:end if
Stage 6 — Domain validation
22:P∗← ∅
23:foreach(f s, ft)∈ Cdo
24:ifV env>0andV sig=1andV call>0andV struct≥0then
25:P∗← P∗∪ {(f s, ft)}▷failure on any one⇒rejected and
logged
26:end if
27:end for
Stage 7 — SonarQube diagnostic
28: Classify each pair into groups A–D (Table I)
29:returnP∗, group counts,|B|/(|A|+|B|)▷Sonar FP rate, Eq. (1)
Algorithm 2Plateau Threshold Selection
Input:Similarity matrixSim; function setF; sweep stepsN= 200
Output: Per-model threshold τ∗▷highest τpreserving max validated pairs;
fallbackτ max if no plateau
1:τ min←min Sim;τ max←max Sim
2:T←LINSPACE(τ min, τmax, N)▷200 evenly-spaced thresholds
3:best← −1;τ∗←τmax ▷default: most conservative cutoff
4:foreachτ∈Tdo▷sweep low to high
5:cnt← |{(f s, ft) : Sim(f s, ft)≥τ∧V env>0∧V sig=1∧
Vcall>0∧V struct≥0}|▷Eq. (6)
6:ifcnt≥bestthen▷≥not>: last tie wins = highestτ
7:best←cnt;τ∗←τ
8:end if
9:end for
10:returnτ∗▷ τmax if count strictly decreasing (no plateau)

Algorithm 3Dynamic-Rule RAG Scoring (diagnostic)
Input: Candidate set C; rule library L(derived from Group B failure
frequencies in Algorithm 1 Stage 5a); thresholdτ∗
Output:Re-scored candidate setC′; rule-firing frequency log
1:C′← ∅
2:foreach(f s, ft)∈ Cdo
3:p←0;fired← ∅▷reset penalty accumulator and fired-rule set
4:foreach ruleρ∈ Ldo▷weights from Group B failure freq.
5:ifρ.patternmatches(f s, ft)then
6:p+=ρ.w;fired←fired∪ {ρ.id}
7:end if
8:end for
9:s←Sim(f s, ft)−p ▷Eq. (7); max total penalty= 0.40
10: Annotate(f s, ft)withfired▷diagnostic: explains score reduction
11:C′← C′∪ {(f s, ft, s,fired)}
12:end for
13:freq←FREQCOUNTS
(fs,ft)fired
▷primary diagnostic output:
which rules dominate
14: Logfreq
15:returnC′filtered tos≥τ∗
V. EXPERIMENTS
We conducted experiments on six publicly available em-
bedded C software repositories to evaluate the pipeline’s
effectiveness and robustness, and to answer five research
questions. All experiments were implemented in Python
and executed in Google Colab using an A100 GPU where
available and CPU otherwise. All results are reproducible
using fixed random seeds; repository snapshots are fixed at
specific commit SHAs. Embeddings were computed using
HuggingFace Transformers [39] for all models except OpenAI
text-embedding-3-small, which was accessed via the OpenAI
/embeddings endpoint [38]. Static analysis was performed
using SonarCloud CLI scanner v8.0.1 over all six repositories.
A function is gradedfailif any Critical or Blocker issue overlaps
its line range; a pair is gradedpassonly if both endpoint
functions pass, giving SonarQube the benefit of the doubt.
The five research questions motivating our evaluation are as
follows.
RQ1: What is SonarQube’s false-positive rate when used as a
reuse-compatibility filter for embedded C software, and
what are the dominant root causes of its failures?
RQ2: Does the 93.6% false-positive finding hold across
different embedding model architectures, or is it an artefact
of a specific backbone?
RQ3: How much do per-model plateau thresholds vary across
the eight backbones, and what does this imply for
approaches that apply a single fixed similarity threshold?
RQ4: Does the dynamic-rule RAG variant produce a statisti-
cally significant difference in retrieval behaviour compared
to the basic RAG variant?
RQ5: How does reuse detection performance differ between
same-domain and cross-domain firmware function pairs?
The metrics used to answer each question are: (i) Sonar FP
rate=|B|/(|A|+|B|) ; (ii) Sonar precision =|A|/(|A|+|B|) ;
(iii) validated-pair count per model; (iv) plateau threshold τ∗;
and (v) rule-firing frequency (dynamic variant only).TABLE VII
PER-MODEL RESULTS.τ∗:PLATEAU THRESHOLD(ALG. 2). VAL. PAIRS:
VALIDATOR-SURVIVING PAIRS. TOP-3BOLD.
Modelτ∗Val. Pairs Sonar FP
MiniLM-L3 0.561 16 95.7%
MPNet0.5282089.9%
BGE-small0.7062389.8%
E5-small 0.915 10 93.4%
GraphCodeBERT 0.886 8 97.3%
OpenAI text-emb-3-small 0.574 12 97.1%
LLaMA 3 8B0.6542192.0%
StarCoder2 3B 0.644 18 92.7%
Aggregate (all 8) — 12893.6%
VI. RESULTS
A. RQ1: SonarQube False-Positive Rate
RQ1’s quantitative results are established in Table I, Table II,
and Table III (Section III); we summarise the key findings here.
Of 1,494 Sonar-approved pairs, 1,399 (93.6%) are rejected
by our domain validators (Eq. 1)— 9.4× the 10% industrial
floor [10] and 17 pp above Charoenwet et al.’s 76% analog
for C/C++ vulnerability detection [23]. The dominant root
cause is hardware-environment mismatch: Venvis involved in
83.5% of Group B rejections, and 78.4% of Group B pairs
have Venv-score= 0 , meaning the two functions share no
hardware-identifying tokens whatsoever despite both passing
SonarQube—confirming SonarQube’s code-quality model has
no representation of peripheral compatibility.
B. RQ2: Per-Model Comparison
Table VII summarises per-model results. Per-model FP rates
range from 89.8% (BGE-small) to 97.3% (GraphCodeBERT),
confirming the finding is not an artefact of any single em-
bedding choice. Validated-pair counts range from 8 to 23.
Thetop-3 modelsby validated-pair yield are:BGE-small
(23 pairs),LLaMA 3 8B(21), andMPNet(20). These three
span three distinct architectural families—contrastive retrieval,
autoregressive LLM, and masked language model—suggesting
that pretraining objective matters more than parameter count.
BGE-small (33M parameters) outperforms StarCoder2-3B
and matches LLaMA-3-8B because its contrastive pretraining
directly optimises the similarity-retrieval task. GraphCodeBERT
performs worst (8 pairs, 97.3% FP) despite code-specific
pretraining, because its data-flow training objective assigns
high cosine similarity even to hardware-incompatible pairs
with similar branching patterns. BGE-small offers the best
speed-quality trade-off for limited-compute settings; LLaMA 3
suits richer natural-language understanding of comments and
READMEs.
C. RQ3: Threshold Variability
Plateau thresholds span 0.528 (MPNet) to 0.915 (E5-
small), a 1.73× spread. GraphCodeBERT and OpenAI pro-
duce compressed-high distributions where many hardware-
incompatible pairs score above 0.8, so their high τ∗values adapt
appropriately, while BGE-small and E5-small produce well-
separated distributions with sharp peaks for genuine matches.

TABLE VIII
BASICRAGVS. DYNAMIC-RULERAGON8,136PAIRS.
Metric basic dynamic∆
Pairs evaluated 8,136 8,136 0
Validator-pass total 193 187−6
Sonar-pass total 3,521 3,227−294
Both-pass total 136 131−5
Sonar precision (%) 3.86 4.06+0.20pp
Sonar FP rate (%) 96.14 95.94−0.20pp
A global threshold of 0.706 (BGE-small’s τ∗) would discard
all StarCoder2 validated matches; the same threshold applied
to GraphCodeBERT would flood the pipeline with hundreds
of spurious candidates—confirming the necessity of per-model
calibration.
D. RQ4: Basic RAG vs. Dynamic-Rule RAG
Table VIII compares both variants on 8,136 percentile-
threshold pairs. Dynamic-rule RAG reduces the Sonar-pass
count by 294 ( −8.3% ) and improves Sonar precision from
3.86% to 4.06% (+0.20 pp), at the cost of 6 validator-approved
pairs and 5 both-pass pairs. A McNemar test on the Sonar-pass
outcome (b= 294,c= 0) yields:
χ2=(b−c)2
b+c=2942
294= 294.0, p <0.001
confirming the reduction is statistically significant [40] (Mc-
Nemar is appropriate since the same 8,136 pairs are evaluated
by both variants). The rule-firing frequency log (Algorithm 3)
shows ENV MISMATCHand CALL MISMATCHdominate—
consistent with the CheckList [35]/HANS [36] diagnostic
tradition—directing future work toward targeted token-level
HAL rules rather than broad category penalties.
E. RQ5: Same-Domain vs. Cross-Domain Reuse
We partitioned the 1,989 unique function pairs into same-
domain pairs (e.g., microwave →microwave alt) and cross-
domain pairs (e.g., microwave →hvac). Same-domain pairs
consistently yield higher validated-pair rates across all eight
models. Of the 128 total validated pairs, the overwhelming
majority are same-domain: two microwave firmwares written
by different developers share GPIO peripherals, LCD drivers,
keypad scanners, and state-machine patterns. Cross-domain
validated-pair counts are near zero across all models. The
dominant rejection signal is Venv: a microwave’s GPIO LED-
array driver and an HV AC’s I2C humidity-sensor reader share
no hardware tokens. This is both a finding—practitioners
should focus reuse effort on same-domain candidates—and
a sanity check confirming our validators discriminate correctly
by domain.
VII. DISCUSSION
A. Why SonarQube Fails as a Reuse Filter
SonarQube measures code quality: cyclomatic complexity,
null-pointer risks, coding standards. It has no representation
of hardware-token compatibility. When a function in one
project uses GPIO_writePort() through a custom HALand a function in another project uses write_port()
through a completely different register map, SonarQube sees
two syntactically clean functions and approves both. The
Venv-score= 0 result for 78.4% of Group B pairs quantifies this
precisely: zero hardware-token overlap, yet SonarQube passed
every one of them. This is consistent with prior evidence that
peripheral identity is the primary contract between firmware
and hardware—a contract invisible to syntax-level analysers [4]–
[6].
B. Dynamic-Rule RAG as a Diagnostic Instrument
The dynamic-rule RAG variant does not outperform basic
RAG on validator-pass counts; we argue this is the correct
result to report, not a failure. In the tradition of CheckList [35],
HANS [36], and Errudite [37], a diagnostic instrument’s value
lies in exposing systematic failure modes, not optimising
headline metrics. The fired-rule frequency log from Algorithm 3
shows that ENV MISMATCHand CALL MISMATCHdominate,
directing future work toward targeted HAL-token rules rather
than broad category penalties. The 6 validator-approved pairs
lost to dynamic RAG quantify the cost of the current rule
conservatism and provide a concrete refinement target.
C. Recommendations for Practitioners
Based on our findings, we offer the following recommenda-
tions for embedded software engineers and tool designers.
(1) Do not rely on SonarQube alone to identify reuse
candidates.Our results show that 93.6% of the function pairs
SonarQube approves as quality-clean are rejected by hardware-
compatibility validation—fewer than 1 in 15 Sonar-approved
pairs is genuinely reusable. SonarQube’s quality grade reflects
individual code style and correctness properties; it has no
model of cross-project hardware compatibility. Engineers who
use SonarQube as a reuse gate will invest significant effort
investigating pairs that are fundamentally non-portable.
(2) Adopt a domain-aware pipeline that makes hardware
compatibility an explicit first-class criterion.Our four-
validator pipeline achieves 97.5% accuracy on manually verified
ground-truth checks, and each rejection is explainable: the
engineer is told which specific aspect of compatibility failed
(hardware token overlap, parameter count, call dependencies,
or structural pattern) rather than receiving a binary reject signal.
This explainability is important for engineering adoption, as it
allows the engineer to assess whether the incompatibility can
be resolved through targeted adaptation.
(3) Use the dynamic-rule variant to audit and encode
project-specific compatibility constraints.The rule library
L(Eq. 7) can be extended with project-specific rules—for
example, marking all functions that reference a particular
vendor HAL family as non-portable to a different target
platform. Running the diagnostic variant on a new firmware
corpus immediately surfaces which failure categories dominate
for that specific codebase, allowing targeted engineering effort
rather than broad manual review.

D. Threats to Validity
We discuss threats to validity following the classification of
W¨ohlin et al. [41] into construct, internal, and external validity.
1) Construct Validity:Our primary metric—the Sonar false-
positive rate—is the fraction of Sonar-approved pairs our
validators reject, which assumes the validators correctly identify
hardware-incompatible pairs. Manual verification of 40 ran-
domly selected Group B pairs (97.5% confirmed incompatible)
supports this, though coverage is limited to 2.9% of the
1,399 Group B pairs. A further threat is our requirement
that all four validators pass simultaneously to approve a
pair: different threshold choices for VenvorVcall(currently
>0) would change pair counts, and while plateau calibration
removes subjectivity from the similarity threshold, the validator
thresholds themselves were set by domain reasoning.
2) Internal Validity:The primary internal threat is circularity
in the dynamic-rule variant: rule library Lis derived from
Group B failures on the same dataset used to evaluate it,
so we frame the diagnostic as an instrument rather than a
held-out, generalised result. A second threat is our regex-
based function extractor, which may miss functions defined
via function pointers, macros, or variadic arguments; an AST-
based extractor [42] would be more robust, though we manually
verified coverage across all six repositories.
3) External Validity:Our evaluation covers six repositories
spanning three embedded domains on two microcontroller
families. Industrial codebases may be larger, use more uniform
HAL abstractions, or enforce stricter coding standards, chang-
ing the SonarQube outcome distribution; the rule weights in
Lwere derived from only three domains and may need re-
derivation for others (e.g., automotive, medical devices) with
different peripheral families. Expanding to additional domains
and industrial codebases is a primary direction for future work.
VIII. CONCLUSION
We presented an end-to-end pipeline for detecting reusable
functions in embedded C software. Embedded software reuse
has long been recognised as an effective means of reducing
development cost and accelerating product cycles, yet identify-
ing reusable embedded code remains a challenging software
engineering problem. An additional benefit of automatically
identifying reusable functions is that developers can make
informed reuse decisions based on software evolution. For ex-
ample, they may prefer newer or more mature implementations
depending on the reuse context, since the age of reused code
can be associated with software quality [43].
Our central empirical finding is thatSonarQube produces
a 93.6% false-positive ratewhen used as a reuse filter, with
83.5% of failures caused by hardware-environment mismatches
it cannot detect. This rate— 9.4× the industrial adoption floor—
is consistent with but substantially extends prior evidence that
static quality tools are unreliable proxies for domain-specific
engineering outcomes [9], [10], [23], [26].
Our domain-aware pipeline combines enriched DOC( f) doc-
ument construction, multi-model similarity scoring across eight
backbones, four hardware-compatibility validators (Eqs. 2–5),and plateau-based per-model threshold calibration (Eq. 6). The
dynamic-rule diagnostic (Eq. 7) reveals that ENV MISMATCH
and CALL MISMATCHare the dominant retrieval failure modes
(χ2= 294.0 ,p <0.001 ). Manual verification of 40 randomly
selected rejected pairs confirms 97.5% validator accuracy.
Future work will refine compatibility rules at the HAL-token
level, expand the evaluation to additional embedded domains
and industrial codebases, integrate compilation- and test-based
validation, and investigate per-unique-pair aggregation across
models to further improve reuse detection.
REFERENCES
[1]Md. Al Maruf, Akramul Azim, and Omar Alam, “Facilitating Reuse of
Functions in Embedded Software,”,IEEE Access, vol. 10, pp. 88595–
88605, 2022.
[2]Md. Al Maruf, Akramul Azim, Nitin Auluck, et al., “FeaMod: Enhancing
Modularity, Adaptability and Code Reuse in Embedded Software
Development,”, in2024 IEEE International Conference on Information
Reuse and Integration (IRI), 2024.
[3]Nishanth Thimmegowda, Omar Alam, Matthias Sch ¨ottle, et al., “Concern-
Driven Software Development with jUCMNav and TouchRAM,”, in
Proceedings of the Demonstrations Track of the ACM/IEEE 17th
International Conference on Model Driven Engineering Languages and
Systems (MoDELS 2014), Valencia, Spain, October 1st and 2nd, 2014,
2014.
[4]Abraham A. Clements, Eric Gustafson, Tobias Bhatt, et al., “HALucinator:
Firmware Re-hosting Through Abstraction Layer Emulation,”, inUSENIX
Security Symposium, 2020.
[5]Bo Feng, Alejandro Mera, and Long Lu, “P2IM: Scalable and Hardware-
Independent Firmware Testing via Automatic Peripheral Interface Mod-
eling,”, inUSENIX Security, 2020.
[6]Alejandro Mera, Bo Feng, Long Lu, et al., “From Library Portability
to Para-rehosting: Natively Executing Microcontroller Software on
Commodity Hardware,”, inUSENIX Security, 2021.
[7]Aakash Bansal, Zachary Eberhart, Zachary Karas, et al., “Function Call
Graph Context Encoding for Neural Source Code Summarization,”,IEEE
Transactions on Software Engineering, vol. 49, no. 9, pp. 4268–4281,
2023.
[8]Vadim Lomshakov, Andrey Podivilov, Sergey Savin, et al., “ProConSuL:
Project Context for Code Summarization with LLMs,”, inEMNLP 2024
Industry Track, 2024.
[9]Valentina Lenarduzzi, Nyyti Saarim ¨aki, and Davide Taibi, “Some
SonarQube Issues Have a Significant but Small Effect on Faults and
Changes,”,Journal of Systems and Software, vol. 170, pp. 110750, 2020.
[10] Caitlin Sadowski, Edward Aftandilian, Alex Eagle, et al., “Lessons from
Building Static Analysis Tools at Google,”,Communications of the ACM,
vol. 61, no. 4, pp. 58–66, 2018.
[11] Zhangyin Feng, Daya Guo, Duyu Tang, et al., “CodeBERT: A Pre-
Trained Model for Programming and Natural Languages,”, inFindings
of EMNLP, 2020.
[12] Daya Guo, Shuo Ren, Shuai Lu, et al., “GraphCodeBERT: Pre-training
Code Representations with Data Flow,”, inInternational Conference on
Learning Representations (ICLR), 2021.
[13] Niklas Muennighoff and others, “MTEB: Massive Text Embedding
Benchmark,”, inEACL, 2023.
[14] Wenhui Wang, Furu Wei, Li Dong, et al., “MiniLM: Deep Self-
Attention Distillation for Task-Agnostic Compression of Pre-Trained
Transformers,”, inNeurIPS, 2020.
[15] Kaitao Song, Xu Tan, Tao Qin, et al., “MPNet: Masked and Permuted
Pre-training for Language Understanding,”, inNeurIPS, 2020.
[16] Jianlv Chen, Shitao Xiao, Peitian Zhang, et al., “BGE M3-Embedding:
Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings
Through Self-Knowledge Distillation,”, inFindings of the Annual Meeting
of the Association for Computational Linguistics (ACL), 2024.
[17] Liang Wang, Nan Yang, Xiaolong Huang, et al., “Text Embed-
dings by Weakly-Supervised Contrastive Pre-training,”,arXiv preprint
arXiv:2212.03533, 2022.
[18] A. Grattafiori and others, “The Llama 3 Herd of Models,”,arXiv preprint
arXiv:2407.21783, 2024.

[19] Anton Lozhkov and others, “StarCoder 2 and The Stack v2: The Next
Generation,”,arXiv preprint arXiv:2402.19173, 2024.
[20] A. A. Talha Talukder, Omar Alam, and Akramul Azim, “Leveraging
LLMs for Automatic Feature Extraction in Embedded Systems to Support
Software Reuse,”, in2025 IEEE International Conference on Information
Reuse and Integration for Data Science (IRI), 2025.
[21] Jiaxu Chen, Junqing Zhao, Jiahao Wei, et al., “AutoFirm: Automatically
Identifying Reused Libraries inside IoT Firmware at Large-Scale,”, in
arXiv preprint arXiv:2406.12947, 2024.
[22] Valentina Lenarduzzi, Alberto Sillitti, and Davide Taibi, “Are SonarQube
Rules Inducing Bugs?,”, inSANER 2020, 2020.
[23] Wachiraphan Charoenwet, Patanamon Thongtanunam, Van-Thuan Pham,
et al., “An Empirical Study of Static Analysis Tools for Secure Code
Review,”, inICSME 2024, 2024.
[24] Yihao Cui, Xiaoyuan Xie, Songqiang Su, et al., “An Empirical Study
of False Negatives and Positives of Static Code Analyzers from the
Perspective of Historical Issues,”,ACM Transactions on Software
Engineering and Methodology, 2024.
[25] Brittany Johnson, Yoonki Song, Emerson Murphy-Hill, et al., “Why
Don’t Software Developers Use Static Analysis Tools to Find Bugs?,”,
inICSE, 2013.
[26] Michail Papamichail, Themistoklis Diamantopoulos, and Andreas Syme-
onidis, “Measuring the Reusability of Software Components Using Static
Analysis Metrics and Reuse Rate Information,”,Journal of Systems and
Software, vol. 158, 2019.
[27] Khurram Mehboob, Rafaqat Ali Khan, Ghulam Rasool, et al.,
“Reusability-Affecting Factors and Software Metrics for Reusability:
A Systematic Literature Review,”,Software: Practice and Experience,
vol. 51, no. 6, 2021.
[28] Hitesh Sajnani, Vaibhav Saini, Jeffrey Svajlenko, et al., “SourcererCC:
Scaling Code Clone Detection to Big Code,”, inICSE, 2016.
[29] Yue Zou, Bihuan Ban, Yinxing Xue, et al., “CCGraph: a PDG-based
code clone detector with approximate graph matching,”, inProceedings
of ASE, 2020.
[30] Wenhan Wang, Ge Li, Bo Ma, et al., “Detecting Code Clones with Graph
Neural Network and Flow-Augmented AST,”, inSANER, pp. 261–271,
2020.
[31] Yangruibo Ding and others, “CoCoMIC: Code Completion By Jointly
Modeling In-file and Cross-file Context,”, inLREC-COLING, 2024.
[32] Shuyan Zhou and others, “DocPrompting: Generating Code by Retrieving
the Docs,”, inICLR, 2023.
[33] Yaniv David, Nimrod Partush, and Eran Yahav, “FirmUp: Precise Static
Detection of Common Vulnerabilities in Firmware,”, inACM SIGPLAN
ASPLOS, 2018.
[34] Tao Yang and others, “Asteria-Pro: Enhancing Deep Learning-Based
Binary Code Similarity Detection by Incorporating Domain Knowledge,”,
ACM Transactions on Software Engineering and Methodology, 2023.
[35] Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, et al., “Beyond
Accuracy: Behavioral Testing of NLP Models with CheckList,”, inACL,
2020.
[36] R. Thomas McCoy, Ellie Pavlick, and Tal Linzen, “Right for the
Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language
Inference,”, inACL, 2019.
[37] Tongshuang Wu, Marco Tulio Ribeiro, Jeffrey Heer, et al., “Errudite:
Scalable, Reproducible, and Testable Error Analysis,”, inACL, 2019.
[38] OpenAI, “text-embedding-3-small Model Documentation,”, 2024.
[39] Thomas Wolf and others, “Transformers: State-of-the-Art Natural Lan-
guage Processing,”, inEMNLP: System Demonstrations, 2020.
[40] Quinn McNemar, “Note on the sampling error of the difference between
correlated proportions or percentages,”,Psychometrika, vol. 12, no. 2,
pp. 153–157, 1947.
[41] Claes Wohlin, Per Runeson, Martin H ¨ost, et al., “Experimentation in
Software Engineering,”, Springer, 2012.
[42] Shuo Lu and others, “CodeXGLUE: A Machine Learning Benchmark
Dataset for Code Understanding and Generation,”,arXiv preprint
arXiv:2102.04664, 2021.
[43] Omar Alam, Bram Adams, and Ahmed E. Hassan, “Measuring the
progress of projects using the time dependence of code changes,”, in
25th IEEE International Conference on Software Maintenance (ICSM
2009), September 20-26, 2009, Edmonton, Alberta, Canada, pp. 329–338,
2009.