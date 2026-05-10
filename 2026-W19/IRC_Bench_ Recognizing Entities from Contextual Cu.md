# IRC-Bench: Recognizing Entities from Contextual Cues in First-Person Reminiscences

**Authors**: Yehudit Aperstein, Eden Moran, Alexander Apartsin

**Published**: 2026-05-07 12:39:49

**PDF URL**: [https://arxiv.org/pdf/2605.06142v1](https://arxiv.org/pdf/2605.06142v1)

## Abstract
When people recount personal memories, they often refer to people, places, and events indirectly, relying on contextual cues rather than explicit names. Such implicit references are central to reminiscence narratives: first-person accounts of lived experience used in therapeutic, archival, and social settings. They pose a difficult computational problem because the intended entity must be inferred from dispersed narrative evidence rather than from a local mention. We introduce IRC-Bench, the Implicit Reminiscence Context Benchmark, for evaluating implicit entity recognition in reminiscence transcripts. The benchmark targets non-locality: entity-identifying cues are distributed across multiple, non-contiguous clauses, unlike named entity recognition, entity linking, or coreference resolution. IRC-Bench comprises 25,136 samples constructed from 12,337 Wiki-data-linked entities across 1,994 transcripts spanning 11 thematic domains. Each sample pairs an Entity-Grounded Narrative, in which the target entity is explicitly mentioned, with an Entity-Elided Narrative, in which direct mentions are removed. We evaluate 19 configurations across LLM generation, dense retrieval, RAG, and fine-tuning. QLoRA-adapted Llama 3.1 8B performs best in the open-world setting (38.94% exact match; 51.59% Jaccard), while fine-tuned DPR leads closed-world retrieval (35.38% Hit@1; 71.49% Hit@10). We release IRC-Bench with data, code, and evaluation tools.

## Full Text


<!-- PDF content starts -->

 IRC-Bench: Recognizing Entities from Contextual Cues  
in First -Person Reminiscences  
Yehudit Aperstei n1, * Eden Moran1 and Alexander Apartsin 2 
1 Intelligent Systems, Afeka Academic College of Engineering, Tel Aviv, Israel ; apersteiny@afeka.ac.il , eden. moran@s.afeka.ac.il  
2 School of Computer Science, Faculty of Sciences, Holon Institute of Technology, Holon, Israel ; alexanderap@hit.ac.il  
* Correspondence: apersteiny@afeka.ac.il  
 
Abstract  
When people recount personal memories, they often refer to people, places, and events indirectly, relying on con-
textual cues rather than explicit names. Such implicit references are central to reminiscence narratives: first -person 
accounts of lived experie nce used in therapeutic, archival, and social settings. They pose a difficult computational 
problem because the intended entity must be inferred from dispersed narrative evidence rather than from a local 
mention.  We introduce IRC -Bench, the Implicit Remini scence Context Benchmark, for evaluating implicit entity 
recognition in reminiscence transcripts. The benchmark targets non -locality: entity -identifying cues are distributed 
across multiple, non -contiguous clauses, unlike named entity recognition, entity l inking, or coreference resolution. 
IRC-Bench comprises 25,136 samples constructed from 12,337 Wikidata -linked entities across 1,994 transcripts 
spanning 11 thematic domains. Each sample pairs an Entity -Grounded Narrative, in which the target entity is 
expl icitly mentioned, with an Entity -Elided Narrative, in which direct mentions are removed.  We evaluate 19 con-
figurations across LLM generation, dense retrieval, RAG, and fine -tuning. QLoRA -adapted Llama 3.1 8B performs 
best in the open -world setting (38.94% exact match; 51.59% Jaccard), while fine -tuned DPR leads closed -world 
retrieval (35.38% H it@1; 71.49% Hit@10). We release IRC -Bench with data, code, and evaluation tools.  
Keywords: implicit entity recognition ; reminiscence narratives ; coreference resolution ; non-locality ; large lan-
guage models ; dense passage retrieval ; QLoRA ; benchmark ; entity linking ; Wikidata  
 
1. Introduction  
Reminiscence, the act of recalling and sharing personal memories, plays a central role in human social life. In 
clinical settings, reminiscence therapy has been shown to reduce depression and improve well -being in older 
adults [1,2], while in archival cont exts, recorded reminiscences preserve cultural and historical knowledge that 
would otherwise be lost [3]. A defining characteristic of reminiscence narratives is that speakers assume shared 
context with their audience: they reference people, places, and ev ents through contextual cues rather than explicit 
naming, trusting the listener to fill in the gaps. This implicit referencing is natural in conversation but creates a 
fundamental challenge for automated systems that seek to index, search, or analyze these  narratives . Table 1 pre-
sents a Japanese American reminiscence example used throughout the paper to illustrate the task. It contrasts an 
entity -grounded narrative (EGN), in which the target entity is still explicitly named, with an entity -elided narrative 
(EEN), in which explicit names are removed and only contextual cues remain.  
Table 1 : Example of implicit entity recognition. The EGN contains explicit mentions of the target entity, whereas the EEN 
removes those mentions while preserving contextual cues that allow recovery of the gold entity . 
ENTITY -
GROUNDED 
NARRATIVE 
(EGN)  "The  attack on Pearl Harbor  was the event that changed everything for Japanese 
Americans like me. After December 7, 1941, suspicion and hatred grew, and we were 
treated as enemy aliens despite being American citizens. It was because of  Pearl Har-
bor that the government issued Executive Order 9066 and started the forced relocation."  

 
  
ENTITY -ELIDED 
NARRATIVE 
(EEN)  "The  surprise attack on a naval base in Hawaii  was the event that changed every-
thing for Japanese Americans like me. After December 7, 1941, suspicion and hatred 
grew, and we were treated as enemy aliens despite being American citizens. It was be-
cause of  that attack  that the government issued an order and started the forced reloca-
tion."  
Gold entity:  Attack on Pearl Harbor (Q52418) | Type : Event | Cues : December 
7, 1941; naval base in Hawaii; Executive Order 9066; forced relocation  
 
A human reader readily identifies the Attack on Pearl Harbor from the constellation of cues: the date, the Hawaiian 
naval base, the executive order, the internment of Japanese Americans. No single phrase names the entity; instead, 
recognition depends on in tegrating cultural, temporal, and historical knowledge distributed across the entire pas-
sage. This pattern of implicit entity reference  is pervasive in reminiscence narratives, where speakers routinely allude 
to well -known people, places, and events withou t naming them, relying on shared background knowledge with 
their listener.  
This phenomenon falls between existing NLP tasks without being addressed by any of them. Named Entity Recog-
nition (NER) identifies explicitly mentioned entity spans in text [4,5]. Entity Linking (EL) resolves those spans to 
knowledge base entries [6,7]. Co reference resolution connects multiple references to the same entity but requires 
at least one explicit mention as an antecedent [8]. In implicit entity references, the entity is never named anywhere 
in the text; there is no span to extract, no mention to link, no antecedent to resolve. The task can be viewed as a 
form of zero -mention coreference: resolving a reference to an entity that has no surface realization in the text, only 
a distributed constellation of contextual cues.  
While implicit entity recognition was first explored in short social -media text [9,10], we extend it to a fundamen-
tally different setting: long -form reminiscence narratives where entity cues are non -local, distributed across mul-
tiple clauses. We release IR C-Bench (Implicit Reminiscence Context Benchmark), a large -scale evaluation resource 
constructed from real reminiscence transcripts. This task addresses practical needs across multiple domains. Ar-
chives of personal reminiscences, including oral history col lections containing millions of hours of recorded testi-
mony, remain largely inaccessible to structured search because the entities discussed are rarely stated by name 
[11]. In healthcare, reminiscence therapy is a widely used intervention for older adults with dementia and depres-
sion [1,2,12]; automated systems that support these therapeutic conversations must identify the people and events 
being discussed even when the speaker does not name them. Social robotics and conversational AI for elderly 
companions hip similarly require understanding implicit references to engage meaningfully with users' personal 
histories [13,14]. More broadly, information retrieval over personal narratives requires understanding not just 
what is said, but what is meant.  
Our contributions are as follows:  
• Task extension and non -locality.  Building on Hosseini's [9] formulation of implicit entity recognition in 
tweets, we extend the task to long -form reminiscence narratives and formalize the non-locality property  of 
implicit references in this domain: entity cues are distributed across multiple non -contiguous text spans, 
requiring holistic integration rather than local pattern matching. This structural property, absent in short -

 
 text settings, fundamentally distinguishes reminiscence -based implicit entity recognition from prior for-
mulations.  
• IRC-Bench.  We release a benchmark of 25,136 implicit entity recognition samples derived from 12,337 
unique Wikidata -linked entities sourced from 1,994 reminiscence transcripts across 11 thematic domains, 
with entity -level train/dev/test splits ensuring zero entity overlap between partitions. Each sample in-
cludes both an EGN and an EEN, along with entity metadata (Wikidata identifier (QID), aliases, Wikipedia 
descript ion).  
• Comprehensive evaluation. We systematically compare 19 experimental configurations spanning open -
world LLM inference (zero -shot, few -shot, chain -of-thought, QLoRA fine -tuning), closed -world dense re-
trieval (off -the-shelf and DPR fine -tuned), and hybrid RAG, revealing that fine -tuning doubles perfor-
mance in both paradigms, chain -of-thought reasoning degrades performance on this task, and model scale 
is the dominant factor in open -world accurac y. 
2. Related Work  
2.1. Named Entity Recognition  
Named Entity Recognition identifies and classifies explicit entity mentions in text. Classical approaches relied on 
handcrafted features and conditional random fields [4], while modern systems employ deep learning architectures 
including bidirectional long short -term memory with conditional random fields (BiLSTM -CRF) [11], transformer -
based sequence  labeling [12], and large language model prompting [13,14]. Recent benchmarks such as CoNLL -
2003 [15] and MultiCoNER [16] have driven progress across entity types and languages. The W2NER framework 
[17] unified flat, nested, and discontinuous NER as word -word relation classification, and UniversalNER [18] 
demonstrated targeted distillation from LLMs for open -domain entity extraction. Despite these advances, all NER 
formulations assume the target entity appears as an explicit surface form in the input text, an assumption that does 
not hold for implicit references.  
2.2 Entity Linking  
Entity Linking resolves textual mentions to entries in a knowledge base. Neural approaches include local attention 
models [6], bi -encoder architectures such as BLINK [19], autoregressive generation via GENRE [20], and efficient 
zero -shot systems like ReFin ED [21]. Botha et al. [22] extended entity linking to over 100 languages. These systems 
take an identified mention span as input and rank candidate entities; they cannot operate when no mention span 
exists. Implicit entity recognition requires generating e ntity candidates from distributed contextual cues rather 
than resolving a given span.  
2.3 Reminiscence Analysis and NLP  
Reminiscence, the structured recall of autobiographical memories, has been studied extensively in psychology and 
gerontology. Butler [23] first proposed life review as a therapeutic process, and subsequent work established rem-
iniscence therapy as an eviden ce-based intervention for depression and cognitive decline in older adults [1,2]. 
Webster [24] developed the Reminiscence Functions Scale, identifying eight distinct functions of autobiographical 
memory sharing. Computational approaches to reminiscence hav e focused primarily on two areas: reminiscence 
therapy systems and oral history processing. Therapy -oriented systems use conversational agents or social robots 
to elicit and respond to personal memories [13,14,25], while oral history processing addresses t ranscription, topic 
segmentation, and search [11,26]. However, none of these systems address the fundamental challenge of identify-
ing the entities that speakers reference implicitly. Our work bridges this gap by extending implicit entity 

 
 recognition, previously studied only in short social -media text [9,10], to the reminiscence domain and providing 
the first benchmark derived from real reminiscence narratives.  
2.4 Implicit and Zero -Mention Entities  
Limited prior work has addressed entities that are referenced but not named. Hosseini [9] introduced implicit 
entity recognition in tweets, constructing a dataset of 3,119 tweets with implicit entity mentions. Hosseini and 
Bagheri [10] developed learning -to-rank methods for this Twitter dataset. Perera et al. [27] explored implicit entity 
recognition in clinical documents. The coreference resolution community has studied "zero anaphora" and bridg-
ing references [28,29], where an entity is referenced indirect ly through related concepts.  
Our work differs from these efforts in five fundamental ways. First, domain and text structure. Tweets are short 
(under 280 characters), formulaic, and heavily context -dependent on trending topics; clinical notes follow rigid 
templates. Reminiscence narrat ives are extended first -person accounts (typically 50 to 200 words per sample) with 
rich, diffuse contextual cues spanning dates, locations, personal relationships, sensory details, and historical 
events. Second, non -locality. In tweets, the implicit entit y is typically inferable from a single cue or hashtag context. 
In reminiscence narratives, we formalize and empirically demonstrate the non -locality property: recognition re-
quires integrating multiple non -contiguous cues distributed across the entire passa ge. No prior work has identified 
or characterized this structural property. Third, scale and diversity. IRC -Bench contains 25,136 samples spanning 
12,337 unique Wikidata -linked entities across 11 thematic domains, compared to 3,119 tweet samples in Hossein i 
[9] covering primarily entertainment and sports entities. Fourth, entity -level evaluation. We introduce entity -level 
train/test splitting with zero entity overlap, ensuring that models must generalize to entirely unseen entities rather 
than memorizing en tity-specific patterns. Prior benchmarks used random sample -level splits where the same en-
tity could appear in both training and test data. Fifth, comprehensive method comparison. We systematically eval-
uate 1 9 configurations spanning four paradigms (genera tive LLM, dense retrieval, RAG, fine -tuning), whereas 
prior work evaluated at most two to three approaches on a single paradigm.  
2.5 Oral History NLP  
Computational analysis of oral histories has received growing attention. Technology -assisted reminiscence sys-
tems have been developed for dementia care [2,3], and AI -driven conversational agents have been explored as 
companions for elderly users [30,31]. D igital storytelling platforms combining AI with augmented reality enable 
communities to preserve personal narratives [32]. However, these systems primarily facilitate memory recall and 
do not attempt to recover the implicit entities that speakers reference  without naming.  
2.6 Knowledge -Grounded Question Answering  
The closest existing task to implicit entity recognition is knowledge -grounded question answering, where a system 
must reason over both a text passage and an external knowledge base to produce an answer [33,34]. RAG ap-
proaches retrieve relevant knowledge base passages and condition generation on them [35,36]. While implicit en-
tity recognition shares the requirement for external knowledge, it differs in that the "question" is an entire narrative 
rather than a targeted query, and the answer is always a single  entity rather than a free -form text span. Further-
more, implicit entity recognition exhibits the non -locality property: the relevant cues are distributed throughout 
the passage rather than concentrated near a question token. This structural difference, as we show empirically, 
causes standard RAG pipelines to underperform direct LLM inference.  
 
 

 
 3. Dataset Construction  
3.1. Overview  
IRC-Bench is constructed through a five -stage automated pipeline that transforms oral history transcripts into 
implicit entity recognition samples. Each sample consists of a first -person narrative that references a named entity 
through contextual cues alon e, without ever naming it. The pipeline leverages GPT -4.1-mini for entity extraction, 
summary generation, and implicit rewriting, followed by automated validation, producing 25,136 benchmark sam-
ples spanning 12,337 unique entities.  
3.2 Source Collections  
The raw data comprises 1,994 cleaned oral history transcripts drawn from 11 thematic collections. These collections 
provide broad topical diversity, covering military conflicts, social movements, immigration, public health crises, 
labor history, and academ ic life. The first -person narrative style of oral histories naturally provides rich contextual 
cues (dates, locations, relationships, roles, events) that make implicit entity references solvable for knowledgeable 
readers, while remaining challenging for au tomated systems. Table 2 summarizes the source collections and tran-
script counts used to construct the corpus.  
Table 2. Source collections for IRC -Bench. Transcript counts reflect cleaned JSON files after the processing pipeline.  
Collec-
tion Tran-
scripts  Sources  Description  
Veterans  517 Library of Congress VHP, 
Nevada WWII, Niles Li-
brary, Wisconsin Veterans 
Museum  Military service narra-
tives  
Immi-
gration  402 University of Minnesota, 
Densho Digital Archive  Immigration and assimi-
lation experiences  
Regional  314 University of Nevada Reno, 
Kentucky Oral History 
Commission  Regional and community 
histories  
Depres-
sion Era  213 Federal Writers' Project (Li-
brary of Congress)  Great Depression oral 
histories  
Japanese 
Ameri-
can 156 Densho Digital Archive  Japanese American in-
ternment and post -war 
Aca-
demic  153 Columbia University Oral 
History, Smithsonian Ar-
chives of American Art  Academic and university 
histories  
Septem-
ber 11  72 National Park Service 9/11 
Memorial  9/11 experiences and af-
termath  
Civil 
Rights  68 Civil Rights History Project 
(Library of Congress)  Civil rights movement 
narratives  
COVID -
19 42 Various oral history projects  Pandemic experiences  
Labor  30 Labor Archives and Re-
search Center  Labor movement histo-
ries 
Refugee  27 Voices of Conscience, UN-
HCR collections  Refugee experiences  
Total  1,994  11 thematic domains, 25+ institutional archives  
 

 
  
3.3 Pipeline Stages  
The benchmark construction proceeds in five stages , illustrated in Figure 1.  
Stage 1: Transcript Cleaning . Raw oral history transcripts are cleaned and converted to structured JSON format, 
preserving the first -person narrative voice while removing interviewer questions and metadata artifacts.  
Stage 2: Named Entity Recognition. GPT -4.1-mini performs NER on each transcript, identifying named entities of seven 
types: Place, Organization, Person, Event, Work, Military Unit, and Other. Each extracted entity is linked to Wikidata and 
Wikipedia where possible. This stage produces 31,284 entity mentions across 1,752 transcript files (87.9% coverage ). 
Stage 3: Explicit Summary Generation . For each (transcript, entity) pair, GPT -4.1-mini generates a first -person narra-
tive summary focused on that entity, preserving the contextual cues surrounding the entity's mention. This pro-
duces 25,161 explicit summaries from 1,601 transcript files (80.3 % coverage).  
Stage 4: Implicit Rewriting . Each explicit summary is rewritten by GPT -4.1-mini to remove all direct mentions of the 
entity name while preserving all contextual cues. The entity reference is replaced with generic descriptions (e.g., 
"Attack on Pearl Harbor" becomes "the attack on th e naval base in Hawaii"). This produces the final 25,136 implicit 
rewrites (80.2% coverage), which form the benchmark's implicit_text field. The slight reduction from Stage 3 re-
flects a small number of cases where the entity coul d not be satisfactorily anonymized.  
Stage 5: Automated Validation.  The final implicit rewrites undergo automated validation to detect residual entity -
name leakage, assess cue sufficiency, and filter narratives for naturalness before inclusion in IRC -Bench.  
 
Figure 1. IRC-Bench construction pipeline. Raw oral history transcripts undergo cleaning, named entity recognition with Wik-
idata linking, entity -grounded narrative generation, and entity elision to produce implicit entity recognition evaluation sam-
ples.  The final validation stage checks for leakage, cue sufficiency, and narrative naturalness.  
3.4 Entity Knowledge Base  
The entity knowledge base contains 12,337 unique entities with the following metadata coverage: 84.6% have as-
sociated Wikipedia pages, 70.9% have LLM -generated descriptions, and 51.2% have alternative names sourced 
from Wikidata. Entity representations in the knowledge base serve multiple roles: as retrieval targets for closed -


 
 world experiments, as alias sources for evaluation matching, and as description inputs for embedding -based ap-
proaches.  
3.5 Entity -Level Train/Dev/Test Splitting  
To ensure rigorous evaluation, the dataset is split at the entity level rather than the sample level. All samples for a 
given entity appear in exactly one partition, preventing information leakage where a model might learn entity -
specific patterns from tra ining examples and exploit them at test time. The split uses a 70/10/20 ratio (seed=42 ). 
Table 3 reports the resulting sample and entity counts for each partition.  
Table 3. IRC-Bench partition statistics. Entity -level splits ensure zero overlap between train, dev, and test entities . 
Partition  Samples  Entities  
Train  17,971  8,635  
Dev 2,532  1,234  
Test 4,633  2,468  
Total  25,136  12,337  
 
3.6 Entity Type Distribution  
The dataset exhibits a natural long -tail distribution over entity types. Places dominate (47.3%), reflecting oral his-
tories' emphasis on geographic locations. Organizations (21.3%) and Persons (13.7%) follow, while specialized 
types such as Events, Works, and Military Units are less frequent but still well represented. Table 4 reports the full 
distribution. Figure 2 visualizes the entity -type distribution alongside the train/dev/test dataset split . 
Table 4. Distribution of IRC -Bench samples by entity type . 
Entity Type  Samples  % of Total  Unique Entities  
Place  11,893  47.3%  4,821  
Organization  5,366  21.3%  2,894  
Person  3,450  13.7%  2,207  
Event  2,162  8.6%  1,102  
Work  1,195  4.8%  743 
Military Unit  537 2.1%  312 
Other  533 2.1%  258 
Total  25,136  100%  12,337  
 
 
(a) (b) 
 
Figure 2. IRC-Bench dataset composition. (a) Distribution of samples across entity types. (b) Distribution of samples across 
train/dev/test splits . 


 
 3.7 Example Samples  
Table 5 presents representative IRC -Bench examples across Person, Event, and Organization entities. The examples 
demonstrate that the EENs remove explicit entity names while preserving distributed contextual cues that make 
the target entity recoverable.  
Table 5. Representative IRC -Bench EGN/EEN examples across entity types. The table illustrates how explicit target -entity men-
tions in the entity -grounded narrative (EGN) are replaced in the entity -elided narrative (EEN), while contextual cues are pre-
served to suppor t recovery of the gold entity.  
Entity 
Type  Example  
Person  
 EGN: "Rosa Parks  was arrested on December 5, 1955, in Montgomery, Ala-
bama, for refusing to give up her bus seat, an act that sparked the Montgomery 
bus boycott.  E. D. Nixon  called me late that night to inform me of her arrest and 
to urge action."  
EEN: "A woman  was arrested on  December 5, 1955 , in Montgomery, Al-
abama , for refusing to give up her  bus seat , an act that sparked the Montgom-
ery bus boycott.  A local leader  called me late that night to inform me of her 
arrest and to urge action."  
Gold:  Rosa Parks (Q41921)  |  Cues:  December 5 1955, Montgomery Al-
abama, bus seat refusal, bus boycott  
Event  
 EGN  
"I headed the relief committee during the disastrous  Berkeley Fire of 1923 , 
helping to coordinate aid and recovery efforts for the community. This was a 
challenging time for Berkeley, California, and I took an active role in organizing 
support to help residents rebuild."  
EEN  
"I headed the relief committee during  the disastrous fire of 1923 in a Cali-
fornia city , helping to coordinate aid and recovery efforts for the community. 
This was a challenging time for the city, and I took an active role in organizing 
support to help residents rebuild."  
Gold:  Berkeley Fire of 1923 (Q4561337)  
Organi-
zation  
 EGN  
"After leaving the Navy in 1966, I worked in the warehouse at  Montgomery 
Ward  in Redwood City . It was a non -union job and pretty low -key, just me 
and an older lady doing pricing and warehouse work."  
EEN  
"After leaving  the Navy  in 1966 , I worked in the warehouse at  a national de-
partment store  in Redwood City . It was a  non-union  job and pretty low -
key, just me and an older lady doing pricing and warehouse work."  
Gold:  Montgomery Ward (Q3046) |  Cues:  Navy 1966, warehouse, na-
tional department store, Redwood City, non -union  
 

 
 3.8 Benchmark Calibration  
To validate the quality and difficulty calibration of IRC -Bench, we conducted an automated quality assessment on 
500 randomly sampled test instances using GPT -4o as an evaluator. Each sample was assessed along two dimen-
sions: narrative naturalness (1 to 5 scale) and cue -based recoverability (whether a knowledgeable human could 
identify the entity from the EEN alone, given the entity identity for reference).  
The EEN narratives achieve a mean naturalness score of 4.87 out of 5, with 87% of samples rated at the maxi-
mum score, confirming that the entity elision process produces fluent, natural -sounding first -person text. For re-
coverability, 42.0% of samples were judged as recoverable (5.8% "yes," 36.2% "probably"), 7.2% as "possible with 
expertise," and 50.8% as "unlikely" or "no." This distribution indicates well -calibrated difficulty: the benchmark is 
challenging enough to be non -trivial (half the samples resist  even informed human judgment) yet solvable enough 
to reward strong models. Notably, the 42% recoverability rate closely matches the performance of the best systems 
(open -world configuration 10 (O10; QLoRA) at 41.4% alias match; closed -world configuration 5 (C5; DPR) at 42.8% 
alias Hit@1), suggesting that top models are ap proaching the practical ceiling imposed by the available contextual 
cues. Cue sufficiency was rated at a mean of 3.0 out of 5, confirming moderate overall difficulty with substantial 
variance across samples. Figure 3 summarizes these validation judgments, showing high naturalness and a delib-
erately challenging recoverability profile.  
 
(a) (b) 
Figure 3. IRC-Bench quality validation (n=500, GPT -4o judge). (a) Distribution of naturalness scores (mean 4.87/5). (b) Recov-
erability judgments showing well -calibrated difficulty . 
4. Methodology  
4.1. Task Formulation  
Implicit entity recognition, the task of identifying entities that are contextually referenced but never explicitly 
named, was first studied by Hosseini [9] in the context of tweets. We adopt the same core objective and extend it 
to long -form reminiscence narratives: given a first -person narrative text  𝑡 that implicitly references a named en-
tity 𝑒 without ever mentioning  𝑒 by name, the task is to identify  𝑒. The text  𝑡 contains contextual cues (dates, 
locations, events, people, roles, descriptions) that jointly constrain the identity of  𝑒, but the model must synthesize 
these cues and draw on world knowledge to produce the correct entity name.  
We evaluate implicit entity recognition under two formulations:  
Open -world formulation : The model generates the entity name as free -form text, without access to a candidate set. 
This tests the model's ability to recall entities from its parametric knowledge. The open -world setting is more real-
istic, as it does not assume a closed inventory o f possible entities.  
Closed -world formulation : The model ranks all 12,337 entities in the knowledge base by relevance to the query text, 
selecting the highest -ranked candidate. This tests the model's ability to match implicit descriptions to entity 


 
 representations via embedding similarity. The closed -world setting provides Hit@K metrics and is analogous to 
entity linking with a fixed knowledge base.  
4.2 The Non -Locality Property  
We define a key structural property that distinguishes implicit entity recognition from span -based entity tasks. 
Let 𝐶(𝑇,𝑒∗)={𝑐1,𝑐2,…,𝑐𝑛} denote the set of textual cues in  𝑇 that collectively identify  𝑒∗. In standard NER and 
EL, the entity is localized: there exists a contiguous span  𝑚 that is sufficient to identify  𝑒∗. In implicit entity recog-
nition, the entity is  non-local: 
∄𝑚⊂𝑇 s.t. 𝑚 is contiguous ∧𝑚⇒𝑒∗ 
but 𝐶(𝑇,𝑒∗)⇒𝑒∗,𝑐𝑖 non-contiguous  
 
That is, no single contiguous substring of  𝑇 is sufficient to identify  𝑒∗, but the set of distributed cues collec-
tively determines it. This non -locality has direct implications for method design: approaches that rely on local span 
matching (NER, EL) or single -vector passage encoding (dense retrieval) are structurally disadvanta ged relative to 
approaches that can integrate information across the full text (LLMs with sufficient context windows).  
We empirically validate non -locality by comparing GPT -4o zero -shot accuracy on full implicit texts versus in-
dividual sentences in isolation (n=200). Full -text accuracy reaches 33.5%, while single -sentence accuracy drops to 
12.9%, a gap of 20.6 percentage p oints. This confirms that entity recognition requires integrating cues distributed 
across the entire passage; no single sentence carries sufficient information in the majority of cases.  
 
4.3 Open -World Methods  
4.3.1. LLM Generative Approach  
We evaluate LLMs in a generative setting where each model receives the implicit text and must produce the 
entity name. All direct -prompting models use temperature 0.0 (greedy decoding) and a maximum of 100 output 
tokens. We test zero -shot (ZS) and few -shot (FS, 5 fixed demonstrations) prompting strategies. Few -shot exemplars 
are selected to cover diverse entity types and are held constant across all test samples. Complete prompt templates 
appear in Appendix A.  
4.3.2 Models  
We evaluate four LLM families in the open -world setting: GPT -4o [37] and GPT -4.1-mini (via OpenAI Batch 
API), and Llama 3.1 8B Instruct [38,39] via OpenRouter API. For GPT -4o, GPT -4.1-mini, and Llama 3.1 8B, we 
additionally evaluate chain -of-thought (CoT) prompting, which instructs the model to reason step -by-step before 
producing the final answer. CoT experiments use temperature 0.7 and a maximum of 300 output tokens to accom-
modate the reasoning trace.  
4.3.3 QLoRA Fine -tuning (O10)  
We fine -tune Llama 3.1 8B Instruct using QLoRA [40,41] for implicit entity recognition. The model is trained 
to generate the entity name given the implicit text, using the standard causal language modeling objective. The 
entity -level splitting guarantees zero overlap between training and test entities,  so the fine -tuned model cannot 
memorize entity -specific patterns; it must learn to generalize the implicit -to-entity mapping to entirely unseen 
entities. Key training parameters include NormalFloat 4 -bit (NF4) quantization, LoRA rank 16, alpha 32, learnin g 
rate 2e -4, and 2 epochs of training on the full train split (17,971 samples). Full hyperparameters are reported in 
Appendix B.  
 

 
  
4.4 Closed -World Methods  
In the closed -world setting, we encode both the implicit query text and all 12,337 entity representations into a 
shared embedding space, then rank entities by cosine similarity. We explore three entity representation strategies:  
Name  (the entity name alone),  Description  (the entity name concatenated with its LLM -generated description), 
and Wiki  (the first sentence from the entity's Wikipedia article). For entities lacking a description or Wikipedia text, 
we fall back to the next available representation.  
4.4.1 BGE -base Baseline (closed -world configurations C1, C2, C3)  
We use BAAI/bge -base -en-v1.5 [42] as our baseline embedding model. This 110M -parameter model produces 768 -
dimensional embeddings and ranks among the top general -purpose bi -encoders on the Massive Text Embedding 
Benchmark (MTEB). Embeddings are L2 -normalize d before computing cosine similarity.  
4.4.2 DPR Fine -tuning (closed -world configurations C4, C5, C6)  
𝐵𝐵−1We fine -tune BGE -base using a DPR approach [36] with Multiple Negatives Ranking Loss (MNRL). Each 
training pair consists of an implicit text (query) and its gold entity representation (positive passage). MNRL uses 
in-batch negatives: for a batch of   query -positive pairs, each positive for one query serves as a negative for all 
other queries, providing   negatives per sample without explicit hard negative mining. We train for 3 epochs with 
batch size 48 and learning rate 2e -5. Three separate models are traine d, one for each entity representation strategy.  
 
4.5 RAG Baseline (RAG configuration 1; RAG1)  
We implement a RAG baseline that combines embedding retrieval with LLM reranking. The pipeline operates in 
two stages. First, BGE -base with entity descriptions (closed -world configuration 2 (C2)) retrieves the top -5 candi-
date entities for each implicit query. Second, GPT -4.1-mini receives the implicit text along with the 5 candidates 
(with their descriptions) and selects the most likely entity or suggests a better one. This approach tests whether an 
LLM can effectively rerank retrieved candidates to improve over pure embedding retrieval . 
5. Evaluation Protocol  
5.1. Matching Hierarchy  
Entity names can be expressed in multiple valid forms (e.g., "United States Marine Corps" vs. "USMC" vs. "Ma-
rines"). To account for this variation, we employ a four -tier matching hierarchy, applied in order of decreasing 
strictness:  
Tier 1 (Exact match) : The prediction and gold entity are identical after lowercasing and whitespace trimming.  
Tier 2 (Alias match):  The prediction matches one of the gold entity's known aliases from Wikidata . For example, predicting 
"NYC" for gold entity "New York City" is an alias match.  
Tier 3 (Containment match) : The prediction is a substring of the gold entity, or vice versa. For example, predicting 
"Pearl Harbor" for "Attack on Pearl Harbor" qualifies as a containment match.  
Tier 4 (Jaccard match) : The token -level Jaccard similarity between the prediction and gold entity is at least 0.5. 
This captures partial overlaps where the prediction includes most of the relevant tokens.  
A prediction is considered correct at a given tier if it matches at that tier or any stricter tier. When reporting 
alias -aware accuracy (the primary metric for open -world experiments), we count any prediction that achieves Tier 
1 or Tier 2 as correct.  
 

 
 5.2 Metrics  
Open -world experiments  report exact match (Tier 1), alias match (Tiers 1+2), containment match (Tiers 1+2+3), and 
Jaccard match (all four tiers).  Closed -world experiments  report Hit@K (K = 1, 3, 5, 10), Mean Reciprocal Rank 
(MRR), and alias -aware Hit@1 (where a hit counts if any alias of the gold entity appears in the top -K). 
5.3 Statistical Significance  
To assess whether performance differences between methods are statistically significant, we use McNemar's test 
(with continuity correction) on the paired per -sample outcomes from each pair of compared systems. Additionally, 
we compute bootstrap confidence intervals (1,000 resamples, seed=42) at the 95% level.  
6. Results and Analysis  
6.1 Open -World Performance  
Table 6 presents the open -world results across all experimental configurations. The QLoRA -adapted Llama 3.1 8B 
(O10) achieves the highest exact match accuracy at 38.94%, substantially outperforming all other open -world meth-
ods. Among non -fine-tuned models, GPT -4o with few -shot prompting (O2) is the strongest at 31.62% exact match, 
rising to 41.10% under the full four -tier Jaccard evaluation. Figure 4 places these open -world results alongside 
closed -world retrieval performance for comparison.  
Model scale is the dominant factor for zero -shot performance: moving from Llama 3.1 8B (13.92%) to GPT -4.1-
mini (25.71%) to GPT -4o (27.02%) yields consistent gains. Few -shot prompting consistently improves performance 
across all model sizes (p < 0.001 by M cNemar's test). The improvement ranges from +2.95 percentage points for 
GPT -4.1-mini to +4.60 points for GPT -4o. The few -shot examples appear to calibrate the model's output format and 
entity granularity, reducing cases where models produce entity types in stead of specific entity names.  
Table 6. Open -world results on the IRC -Bench test set (n=4,633). Exact = exact string match; Alias = alias -aware match; Contain 
= containment match; Jaccard = Jaccard match (>=0.5). Best result in each column is highlighted . 
ID Model  Mode  Exact , % Alias , % Contain , % Jaccard , % 
O1 GPT -4o Zero -
shot  27.02  33.30  33.30  35.05  
O2 GPT -4o Few -
shot  31.62  38.94  38.94  41.10  
O3 GPT -4.1-
mini  Zero -
shot  25.71  27.09  33.50  35.94  
O4 GPT -4.1-
mini  Few -
shot  28.66  36.89  36.89  39.48  
O5 Llama 3.1 8B  Zero -
shot  13.92  14.81  19.47  20.18  
O6 Llama 3.1 8B  Few -
shot  17.83  18.80  24.61  25.66  
O10 Llama 3.1 8B 
(QLoRA)  Fine -
tuned  38.94  41.42  47.90  51.59  
O11/b  GPT -4.1-
mini CoT  t=0.7 / 
t=0.0  18.93 / 
19.44  20.27 / 
20.76  26.48 / 
26.87  27.69 / 28.10  
O12/b  GPT -4o CoT  t=0.7 / 
t=0.0  22.51 / 
25.57  23.89 / 
33.54  30.91 / 
37.21  32.33 / 38.92  

 
 ID Model  Mode  Exact , % Alias , % Contain , % Jaccard , % 
O13 Llama 3.1 8B 
CoT t=0.7  6.22 6.69 11.72  12.24  
RAG1  BGE + GPT -
4.1-mini  RAG  19.71  20.53  28.75  29.55  
 
 
 
 
Figure 4. Comparison of open -world and closed -world methods on the IRC -Bench test set. Open -world methods are measured 
by exact match and alias -aware accuracy; closed -world methods by Hit@1 and Hit@10.  
The most striking open -world result is the effect of QLoRA fine -tuning. Open -world configuration 10 (O10; QLoRA 
Llama 3.1 8B) achieves 38.94% exact match, nearly tripling the base model's zero -shot performance (13.92%) and 
exceeding GPT -4o few -shot (31.62% ) by 7.32 percentage points. At the Jaccard level, O10 reaches 51.59%, meaning 
more than half of all test predictions are at least partially correct. This is particularly notable given the entity -level 
split: O10 has never seen any of the 2,468 test entities during training, demonstrating genuine generalization of the  
implicit -to-entity mapping.  


 
 The failure of chain -of-thought prompting  is equally striking. CoT reduces GPT -4o accuracy from 33.30% (zero -
shot alias) to 23.89%, and GPT -4.1-mini from 25.71% (zero -shot exact) to 18.93%. CoT also degrades Llama 3.1 8B 
from 13.92% (zero -shot exact) to 6.22%. We analyze the reasons for this failu re in Section 7.  
The hybrid RAG approach (19.71% exact match) underperforms even GPT -4.1-mini zero -shot (25.71%). When 
the gold entity does not appear among the top -5 candidates (which occurs in roughly 67% of cases with BGE -base, 
given C2's Hit@5 of 33.41%), the LLM reranker cannot recover it.  
6.2 Closed -World Performance  
Table 7 shows the closed -world retrieval results. Fine -tuned DPR with description representations (closed -world 
configuration 5; C5) achieves the best performance: 35.38% Hit@1, 71.49% Hit@10, and 0.4751 MRR. With alias -
aware evaluation, C5 reaches 42.80% Hit@1 and 74.47% Hit@10. Figure 5 shows how retrieval performance changes 
across Hit@K thresholds.  
Table 7. Closed -world retrieval results on the IRC -Bench test set. The candidate set contains all 12,337 entities. Best result in 
each column is highlighted. Alias columns report alias -aware metrics.  
ID Retriever  Entity 
Repr.  Hit@1, %  Hit@3, %  Hit@5, %  Hit@10, %  MRR  Alias 
H@1, %  
C1 BGE (off -the-
shelf)  Name  16.51  26.38  30.97  36.76  0.2362  22.08  
C2 BGE (off -the-
shelf)  Descrip-
tion 16.64  27.78  33.41  40.60  0.2480  21.78  
C3 BGE (off -the-
shelf)  Wiki  14.38  25.10  29.92  37.32  0.2211  19.32  
C4 DPR (fine -tuned)  Name  30.00  46.36  53.66  63.31  0.4131  37.10  
C5 DPR (fine -tuned)  Descrip-
tion 35.38  53.51  61.82  71.49  0.4751  42.80  
C6 DPR (fine -tuned)  Wiki  27.95  44.98  51.82  59.55  0.3851  34.38  
 
 
 
Figure 5. Hit@K  curves for closed -world retrieval methods. Fine -tuned DPR with description representations (C5) substantially 
outperforms all baseline configurations across all K values.  


 
 The comparison between off -the-shelf BGE and fine -tuned DPR reveals the magnitude of domain adaptation ben-
efits. DPR fine -tuning more than doubles Hit@1 for all entity representation types: Name (16.51% to 30.00%, +13.49 
pp), Description (16.64% to 35.38%,  +18.74 pp), and Wiki (14.38% to 27.95%, +13.57 pp). The largest absolute gain 
occurs for descriptions, indicating that fine -tuning is especially effective at learning to align the narrative cue struc-
ture with the rich attribute content in entity descripti ons. Figure 6 visualizes the effect of DPR fine -tuning across 
entity representation strategies.  
Across both retrieval architectures, entity description representations consistently outperform name -only and 
Wikipedia representations. Descriptions provide a concise, attribute -rich summary that aligns well with the con-
textual cues present in elided narr atives. Wikipedia lead sentences, despite containing more information, introduce 
noise from tangential content.  
 
(a) (b) 
Figure 6. Effect of DPR fine -tuning on retrieval performance. Fine -tuning more than doubles Hit@1 across all entity represen-
tation strategies, with the largest absolute gain for descriptions (+18.74 pp).  
6.3 Cross -Paradigm Comparison  
Table 8 ranks the top -performing systems across both paradigms under an alias -level comparison  metric.  
Table 8. Cross -paradigm ranking by alias -level score. Open -world methods use the Alias, % score (Tiers 1+2); closed -world 
methods use alias -aware Hit@1.  
Rank  System  Paradigm  Alias -level score, %  
1 C5 (DPR + Description)  Closed  42.80  
2 O10 (QLoRA Llama 8B)  Open  41.42  
3 O2 (GPT -4o FS)  Open  38.94  
4 C4 (DPR + Name)  Closed  37.10  
5 O4 (GPT -4.1-mini FS)  Open  36.89  
6 C6 (DPR + Wiki)  Closed  34.38  
7 O1 (GPT -4o ZS)  Open  33.30  
8 O3 (GPT -4.1-mini ZS)  Open  27.09  
 
Under this stricter alias -level comparison, the fine -tuned DPR retriever with descrip tion representations (C5) ranks 
first at 42.80% alias -aware Hit@1, followed closely by the fine -tuned QLoRA model (O10) at 41.42% open -world 
alias accuracy. GPT -4o few -shot (O2) ranks third at 38.94%, and DPR with name representations (C4) ranks fourth 


 
 at 37.10%. This comparison shows that the 110M -parameter fine -tuned retriever is competitive with, and slightly 
ahead of, the best open -world generative configuration under an alias -level criterion.  
 
6.4 Per -Entity -Type Analysis  
Performance varies substantially by entity type. Table 9 reports the alias -aware Hit@1 (all tiers) for selected meth-
ods. Figure 7 visualizes these per -type differences as a heatmap.  
Table 9. Hit@1 (%) by entity type (alias -aware, all tiers). n denotes the number of test samples of each type.  
 
 
 
 
Figure 7. Heatmap of performance (alias -aware Hit@1) by entity type and method. Person entities are consistently the hardest 
across all methods; Events are notably strong for both open -world and closed -world approaches . 
Entity 
Type  n O1 
(GPT -4o 
ZS) O2 
(GPT -4o 
FS) O5 
(Llama 8B 
ZS) C1 
(BGE 
Name)  C2 
(BGE 
Desc)  
Place  2,076  38.15  43.88  18.16  14.88  15.99  
Organiza-
tion 1,152  38.28  45.31  27.34  29.17  27.17  
Person  698 23.82  24.07  14.90  18.34  18.62  
Event  273 34.43  50.18  27.11  48.35  47.25  
Work  215 32.09  39.53  14.42  39.07  36.74  
Military 
Unit  121 26.45  37.19  10.74  23.97  31.40  
Other  98 30.61  36.73  21.43  33.67  25.51  

 
 Persons are the hardest type for open -world methods.  GPT -4o FS achieves only 24.07% on Person entities, com-
pared to 45.31% on Organizations and 43.88% on Places. Person entities often have less distinctive contextual cues 
and are more likely to be obscure individuals not well represented in model training d ata. 
Events are notably strong for closed -world methods.  BGE achieves 48.35% Hit@1 on Events, higher than any 
other type, suggesting that event descriptions provide distinctive semantic signatures that align well with implicit 
event narratives.  
Few -shot examples disproportionately help Events.  GPT -4o jumps from 34.43% (ZS) to 50.18% (FS) on Events 
(+15.75 pp), the largest per -type improvement, likely because the few -shot examples include two Event instances 
(Attack on Pearl Harbor).  
6.5 Error Analysis  
We performed automated error classification on 200 randomly sampled incorrect predictions from each of O1 
through O6, using GPT -4.1-mini to categorize errors. Table 10 reports the distribution.  Representative correct and 
incorrect prediction examples are provided in Appendix C.  
Table 10. Error type distribution (%) over 200 randomly sampled incorrect predictions per model. Categories are mutually 
exclusive . 
 
 
 
 
 
 
The dominant error mode across all models is same -type, unrelated  (42% to 52%), where the model predicts an entity 
of the correct type but one that is semantically unrelated to the gold entity (e.g., predicting "Jack Johnson" when 
the gold is "Lou Ambers," both boxers). The second most common error is wrong type  (22.5% to 35.0%), where the 
model predicts an entity of an entirely different category. Same -type, related  errors (13.5% to 25.5%) represent near -
misses where the prediction is semantically close to the gold (e.g., predicting "Okinawa" for "Iwo Jima"). Halluci-
nations and empty responses are rare (<2.5%), indicating that models reliably produce plausible entity names even 
when incorrect.  
Llama 3.1 8B (O5, O6) shows a higher proportion of same -type, unrelated errors (52.0% and 46.0%) and a lower 
proportion of same -type, related errors (13.5% and 17.0%) compared to GPT models (O1, O2: 24.5% and 25.5%). 
This suggests that smaller models have weaker ability to narrow down candidates within a type using fine -grained 
contextual cues.  
6.6 Key Findings Summary  
Table 11 consolidates the main empirical findings and links each conclusion to the corresponding quantitative 
comparison.  
 Error Type  O1 O2 O3 O4 O5 O6 
Same -type, unrelated  43.0 42.0 43.5 45.0 52.0 46.0 
Wrong type  28.5 27.5 29.5 22.5 31.0 35.0 
Same -type, related  24.5 25.5 22.5 24.0 13.5 17.0 
Partial match  3.5 4.0 3.0 6.0 2.5 1.5 
Empty/hallucination  0.5 1.0 1.5 2.5 1.0 0.0 

 
 Table 11. Main empirical findings and supporting comparisons . 
# Claim  Experimental Results  
1 Fine -tuning is the most 
impactful intervention  QLoRA fine -tuning of Llama 3.1 8B raises exact match from 
13.92% (O5, zero -shot) to 38.94% (O10), a 2.80x improvement. 
DPR fine -tuning of BGE raises Hit@1 from 16.64% (C2) to 35.38% 
(C5), a 2.13x improvement. Both gains are achieved despite zero 
entity ove rlap between training and test sets.  
2 QLoRA fine -tuning yields 
the overall best perfor-
mance  O10 achieves 38.94% exact match (51.59% Jaccard), surpassing 
GPT -4o few -shot (31.62% exact, 41.10% Jaccard) by 7.32 pp on 
exact match and 10.49 pp on Jaccard. This result uses only 6.5M 
trainable parameters on top of an 8B -parameter base.  
3 Chain -of-thought does 
not improve perfor-
mance  CoT reduces GPT -4.1-mini from 25.71% (zero -shot exact) 
to 18.93% and Llama 3.1 8B from 13.92% to 6.22%. For 
GPT -4o, lowering temperature recovers parity with zero -
shot but does not exceed it.  
4 Few-shot prompting 
consistently helps  Adding five demonstrations improves GPT -4o from 
27.02% to 31.62% (+4.60 pp), GPT -4.1-mini from 25.71% to 
28.66% (+2.95 pp), and Llama 3.1 8B from 13.92% to 17.83% 
(+3.91 pp).  
5 Entity descriptions are 
the best retrieval repre-
sentation  C5 (DPR + description) outperforms C4 (DPR + name) by 
5.38 pp on Hit@1 (35.38% vs. 30.00%) and C6 (DPR + wiki) 
by 7.43 pp (35.38% vs. 27.95%). The same pattern appears 
with off -the-shelf BGE.  
6 RAG underperforms 
direct LLM inference  RAG1 reaches 19.71% exact match, which is 5.99 pp below 
GPT -4.1-mini zero -shot (25.71%) and 8.95 pp below GPT -
4.1-mini few -shot (28.66%). The retrieval bottleneck limits 
the reranking stage.  
7 Model scale matters in 
the zero -shot regime  GPT -4o zero -shot reaches 27.02% exact match, outper-
forming Llama 3.1 8B zero -shot at 13.92% by 13.10 pp 
(McNemar chi -squared = 432.28, p < 0.001).  
8 Retriever Hit@10 reveals 
strong latent signal  C5 places the gold entity in the top 10 for 71.49% of queries, in-
dicating that DPR shortlists contain useful candidates and can 
support stronger future reranking approaches.  
 
6.7 Statistical Significance  
All key comparisons are statistically significant at p < 0.001 (McNemar's test with continuity correction). Table 12 
reports the detailed results.  
Table 12. Statistical significance tests (McNemar's test with continuity correction). All p -values < 0.001. A -only and B -only 
report the number of discordant pairs . 
Comparison  Acc A 
(%) Acc B 
(%) McNemar 
χ² A-only  B-only  
O1 vs O2 (ZS vs FS, 
GPT -4o) 35.06  41.11  149.69  120 400 
O3 vs O4 (ZS vs FS, 
mini)  35.16  38.72  36.18  150 275 

 
 Comparison  Acc A 
(%) Acc B 
(%) McNemar 
χ² A-only  B-only  
O1 vs O5 (GPT -4o 
vs Llama 8B)  35.06  20.19  432.28  892 203 
O1 vs C2 (Open vs 
Closed)  35.06  22.58  181.93  1,204  626 
 
The discordant pair counts are informative: for the GPT -4o vs. Llama 8B comparison, 892 samples are solved only 
by GPT -4o while only 203 are solved only by Llama 8B, demonstrating a strong directional advantage. For the ZS 
vs. FS comparison on GPT -4o (O1 v s. O2), 400 samples are gained while only 120 are lost, confirming that few -shot 
examples provide a net benefit with limited trade -offs. The 95% bootstrap confidence intervals confirm non -over-
lapping ranges for all reported comparisons.  
7. Discussion  
The most counterintuitive finding is the failure of chain -of-thought prompting. CoT improves performance 
on mathematical reasoning and multi -hop QA by decomposing complex reasoning into intermediate steps [43], 
yet it degrades implicit entity recognition f or every model tested. The explanation lies in the gestalt nature of the 
task: identifying an implicit entity requires simultaneously attending to a constellation of distributed cues and 
matching this constellation against parametric knowledge. When forced  to reason step by step, models fixate on 
individual cues in isolation, arriving at locally plausible but globally incorrect entities. Temperature control exper-
iments (O11b, O12b) confirm this is structural for smaller models (GPT -4.1-mini: +0.5pp at t=0.0 , still 6.3pp below 
ZS) while for GPT -4o, controlling temperature recovers the gap to ZS parity (33.5% vs 33.3%) without exceeding 
it. CoT therefore neither helps nor hurts large models once temperature is controlled, but genuinely harms smaller 
models tha t lack capacity to maintain holistic context while verbalizing reasoning.  
The RAG pipeline underperforms direct generation (19.71% vs 28.66% for GPT -4.1-mini FS) because dense 
retrievers encode the EEN as a single vector, losing fine -grained cue information through the bottleneck. Retrieved 
candidates are topically related but o ften incorrect, and when presented as context they can override the model's 
own correct intuition. With BGE -base, the gold entity appears in the top -5 only 33.41% of the time (C2 Hit@5), 
severely limiting the reranker.  
The success of QLoRA (O10: 38.94% exact, up from 13.92% base) despite zero entity overlap warrants expla-
nation. Fine -tuning teaches three transferable skills: the task format (extracting a single canonical name, avoiding 
verbose hedging), cue integration p atterns (which temporal, spatial, and relational cue combinations are diagnos-
tic), and entity type priors (calibrating expectations to reduce wrong -type errors). The model learns "how to solve 
implicit entity puzzles" rather than memorizing specific answer s. Comparing the best open -world (O10: 38.94%) 
and closed -world (C5: 35.38% Hit@1, 42.80% alias) results, both paradigms reach roughly comparable alias -level 
performance, though the closed -world Hit@10 of 71.49% suggests that combining fine -tuned retrieval shortlists 
with fine -tuned LLM reranking is a promising future direction. Figure 8 illustrates this scale -performance relation-
ship and shows how QLoRA departs from the zero -shot scale trend.  
Entity difficulty correlates with cue specificity and knowledge -base neighborhood density. Events achieve 
50.18% (GPT -4o FS) due to unique date and participant combinations, while Persons reach only 24.07% because 
generic biographical attributes (occupation, era, region) are shared by many candidates. This pattern is consistent 
across all methods.  

 
 Several limitations should be acknowledged. The benchmark covers English -language oral histories focused 
primarily on American experiences. The LLM -generated entity elision, though validated (naturalness 4.87/5, 42% 
recoverable by informed assessment), may  not perfectly replicate naturally occurring implicit references. The alias -
aware evaluation still penalizes semantically correct predictions using unregistered surface forms. Temperature 
controls were not run for Llama 3.1 8B CoT. Finally, QLoRA training used max_seq=192 tokens, which truncates 
approximately 6% of test prompts.  
 
 
Figure 8. Relationship between model scale and open -world accuracy. Larger models achieve substantially higher accuracy, 
with the relationship appearing roughly log -linear in model parameter count. QLoRA fine -tuning (O10) breaks this trend, en-
abling an 8B model to o utperform much larger models.  
8. Conclusion  
We have extended implicit entity recognition, previously studied in short social -media text [9,10], to the 
domain of long -form reminiscence narratives, formalizing the non -locality property that distinguishes this setting 
and empirically validating it thro ugh a sentence -level ablation showing a 20.6pp accuracy gap between full -text 
and single -sentence inference. We release IRC -Bench, a benchmark of 25,136 samples spanning 12,337 Wikidata -
linked entities from 1,994 oral history transcripts across 11 thematic  domains. Our systematic evaluation across 19 
experimental configurations reveals eight key findings.  
First, fine -tuning is the single most impactful intervention. QLoRA -adapted Llama 3.1 8B achieves 38.94% 
exact match (51.59% Jaccard), nearly tripling the base model's zero -shot performance and surpassing GPT -4o few -
shot by 7.32 percentage points, despite the entity -level split ensuring zero overlap with training entities. In the 
closed -world setting, DPR fine -tuning of BGE -base more than doubles Hit@1 from 16.64% to 35.38%, with the gold 
entity appearing in the top -10 for 71.49% of queries.  
Second, chain -of-thought prompting degrades smaller models (by 4.51 to 7.70 pp), while temperature control 
experiments reveal that for GPT -4o, the observed CoT penalty is largely attributable to the higher sampling tem-
perature rather than the reasoning str ucture itself. In all cases, CoT fails to exceed zero -shot performance, confirm-
ing that implicit entity recognition requires holistic pattern matching rather than sequential reasoning. Third, re-
trieval -augmented generation underperforms direct LLM inferenc e due to the non -locality of implicit cues. Fourth, 


 
 model scale is the dominant factor in zero -shot open -world accuracy, with performance spanning from 13.92% 
(Llama 3.1 8B) to 27.02% (GPT -4o) in exact match. Fifth, entity descriptions are consistently the best representation 
for dense retrieval, outperform ing both entity names and Wikipedia lead sentences.  
Future work should explore several promising directions: multi -modal implicit entity recognition incorpo-
rating audio features from the original recordings, cross -lingual benchmarks constructed from oral history ar-
chives in other languages, active learning approaches that combine fine -tuned DPR shortlists with fine -tuned LLM 
reranking (leveraging C5's 71.49% Hit@10), and the development of specialized architectures that explicitly model 
the non -locality property of implicit entity cues through structured att ention over distributed text spans.  
Author Contributions:  Conceptualization, Y.A.  and A.A.; methodology, Y.A.  and A.A.; software, E.M.; validation, Y.A., E.M., 
and A.A.; formal analysis, Y.A., E .M., and A.A.; data curation, E.M. and A.A.; writing —original draft preparation, Y.A.  and 
A.A.; writing —review and editing, Y.A.  and A.A.; visualization, Y.A., E.M., and A.A.; supervision, Y.A.  and A.A.; project ad-
ministration, Y.A. All authors have read and agreed to the published version of the manuscript  
Funding:  This research received no external funding  
Data Availability Statement:  The IRC -Bench dataset, source code, experimental scripts, prompts, evaluation tools, and result 
files supporting the findings of this study are publicly available in the project repository at https://github.com/ApartsinPr o-
jects/ImplicitEntities. The benchm ark annotations are released under the license specified in the repository. Source oral -history 
transcripts are publicly available from their respective archives, with provenance information provided in the repository . 
Conflicts of Interest:  The authors declare no conflicts of interest.  
Abbreviations  
The following abbreviations are used in this manuscript:  
BAAI  Beijing Academy of Artificial Intelligence  
BGE  BAAI General Embedding  
BiLSTM -CRF  Bidirectional long short -term memory with conditional random fields  
BLINK  Bi-encoder Linker  
C1-C6 Closed -world experimental configurations 1 -6 
CoT Chain -of-thought  
DPR  Dense passage retrieval  
EEN  Entity -elided narrative  
EGN  Entity -grounded narrative  
EL Entity linking  
FS Few -shot 
GENRE  Generative Entity Retrieval  
IRC-Bench  Implicit Reminiscence Context Benchmark  
LoRA  Low -Rank Adaptation  
MNRL  Multiple Negatives Ranking Loss  
MRR  Mean Reciprocal Rank  
MTEB  Massive Text Embedding Benchmark  
NER  Named Entity Recognition  
NF4  NormalFloat 4 -bit 
O1-O13 Open -world experimental configurations 1 -13 
QID  Wikidata identifier  
QLoRA  Quantized Low -Rank Adaptation  
RAG  Retrieval -augmented generation  
W2NER  Word -word relation classification for named entity recognition  
ZS Zero -shot 
 
 

 
 Appendix  A: Prompt Templates  
Appendix A.1 : Zero -Shot Prompt  
System message:  
You are an entity recognition expert. Given a text that implicitly references a named 
entity without mentioning it, identify what entity is being referenced.  
User message:  
What named entity is implicitly referenced in this text? The entity is never men-
tioned by name.  
Text: "{text}"  
Think about the contextual cues (dates, places, events, people, roles) and identify 
the specific named entity being referenced.  
Answer with ONLY the entity name (canonical Wikipedia name), nothing else.  
Parameters: temperature=0.0, max_tokens=100  
 
Appendix A. 2: Few-Shot Prompt  
System message:  
You are an entity recognition expert. Given a text that implicitly references a 
named entity without mentioning it, identify what entity is being referenced.  
User message:  
What named entity is implicitly referenced in this text? The entity is never men-
tioned by name.  
Examples: Text: "I remember that Sunday morning in December '41. We were lis-
tening to the radio when the news broke about the attack on the naval base in Ha-
waii. That's when everything changed." Entity: Attack on Pearl Harbor  
Text: "I enlisted right out of high school and went to boot camp in San Diego. As 
an aircraft mechanic, I was sent to the Pacific." Entity: United States Marine Corps  
Text: "After the surrender, we flew into the main islands. I landed in the bay and 
spent six months there for the occupation. The capital was flattened by the B -29s." 
Entity: Tokyo  
Text: "In late 1941, I was set to ship out from San Francisco. A friend ran up saying 
they're bombing the base in Hawaii." Entity: Attack on Pearl Harbor Text: "Grow-
ing up in that bustling metropolis with towering skyscrapers, I was immersed in a 
vibrant c ulture." Entity: New York City  
Now identify the entity in this text:  
Text: "{text}"  
Answer with ONLY the entity name (canonical Wikipedia name), nothing else.  
Parameters: temperature=0.0, max_tokens=100  

 
 Appendix A. 3: Chain -of-Thought Prompt  
System message:  
You are an entity recognition expert. Think step by step.  
User message:  
What named entity is implicitly referenced in this text? The entity is never men-
tioned by name.  
Text: "{text}"  
Think step by step:  
1. What contextual cues are present? (dates, places, events, people, roles)  
2. What type of entity do these cues suggest? (Person, Place, Organization, Event)  
3. What specific named entity matches ALL these cues?  
Reasoning: [your step -by-step analysis]  
Entity: [canonical Wikipedia name]  
Parameters: temperature=0. 7, max_tokens= 300 
Appendix A. 4: RAG Prompt  
User message  (no system message) : 
This text implicitly references a named entity without naming it. Based on the con-
textual cues, which candidate is most likely?  
Text: "{text}"  
Candidates:  
1. {candidate_1} - {description_1}  
2. {candidate_2} - {description_2}  
3. {candidate_3} - {description_3}  
4. {candidate_4} - {description_4}  
5. {candidate_5} - {description_5}  
If none match well, suggest a better entity.  
Answer: [number]. [entity name]  
Parameters: temperature=0. 7, max_tokens= 50 
 
Appendix A.5: QLoRA Fine -tuning Prompt (O10)  
<|begin_of_text|>  
<|start_header_id|>system<|end_header_id|>  
You identify implicitly referenced entities.<|eot_id|> 
<|start_header_id|>user<|end_header_id|>  
What entity is implicitly referenced? Answer with only the entity name.  
Text: {implicit_text}<|eot_id|> <|start_header_id|>assistant<|end_header_id|>  
{entity}<|eot_id|>  
 

 
 Parameters: greedy decoding, max_new_tokens=30  
 
Appendix  B: Training Hyperparameters  
Table B.1. DPR (Dense Passage Retrieval) Fine -tuning  
Base model  BAAI/bge -base -en-v1.5 
Model parameters  ~110M  
Embedding dimension  768 
Training examples  17,971  
Epochs  3 
Batch size  48 
Learning rate  2e-5 
Warmup steps  100 
Loss function  MultipleNegativesRankingLoss (MNRL)  
Optimizer  AdamW  
Mixed precision  FP16 (AMP)  
Negatives  In-batch (47 negatives per sample)  
Random seed  42 
 
Table B.2. QLoRA (O10) Fine -tuning  
Base model  meta -llama/Llama -3.1-8B-Instruct  
Model parameters  ~8B (base); ~6.5M trainable (LoRA)  
Quantization  4-bit NormalFloat (NF4)  
Compute dtype  bfloat16  
LoRA rank (r)  16 
LoRA alpha (α)  32 
LoRA dropout  0.05 
Target modules  q_proj, v_proj, k_proj, o_proj  
Training examples  17,971  
Epochs  2 
Per-device batch size  48 
Gradient accumulation  1 (effective batch: 48)  
Learning rate  2e-4 
Max sequence length  192 tokens  
Warmup steps  50 
Precision  bfloat16  
Validation samples  500 (subset of dev)  
Framework  TRL SFTTrainer + PEFT  
 
 
 
 
 

 
 Appendix  C: Example Predictions  
We present five correct and five incorrect predictions from GPT -4o few -shot (O2) and selected O10 predictions, to 
illustrate the task characteristics and failure modes.  
Appendix C.1: Correct Predictions (O2: GPT -4o Few -shot)  
 
Correct Example 1  
"I studied at a major public university in Northern California, where I was part of the De-
sign Department. During my time there, I combined academic courses with art classes, fo-
cusing on three -dimensional design..."  
Gold:  University of California, Berkeley  |  Prediction:  University of California, 
Berkeley   ✔ 
 
Correct Example 2  
"I was born in the capital city of Germany in the early 1930s. It was a turbulent time as 
the political climate was rapidly changing. My family decided to leave that city in 1938 to 
escape the dangers posed by the Nazi regime. That move shaped much of my e arly life and 
future."  
Gold:  Berlin  |  Prediction:  Berlin   ✔ 
 
Correct Example 3  
"When I first came to America, I worked in a Pacific island territory for eight months on a 
sugar plantation. I was only 15 years old and worked under a Chinese boss for $18 a 
month..."  
Gold:  Hawaii  |  Prediction:  Hawaii    ✔ 
 
Correct Example 4  
"My grandfather was a teenager during the major 1950s political upheaval in our Carib-
bean homeland and once found a journal from someone fighting with the revolutionary 
leader. That period was filled with fear for my family and community. The uprising brou ght 
about communism, which had some positive effects like high literacy rates, but also caused 
extreme poverty and suffering. The memories of that era shape how older immigrants from 
that island view politics in the United States today."  
Gold:  Cuban Revolution   |  Prediction:  Cuban Revolution   ✔ 
 

 
 Correct Example 5  
"He and I were close when he was Senate majority leader, and he was very cordial to me 
when I first came to the Senate. He gave me important committee assignments, including 
chairing the Calendar Committee and seats on the Agricultural and Finance Committe es. 
He was probably the most able majority leader in history, knowing the Senate's personali-
ties and how to motivate them. As President, he overcommitted on social programs, which 
I believe contributed to the huge deficits we face today."  
Gold:  Lyndon B. Johnson   |  Prediction:  Lyndon B. Johnson   ✔ 
 
These correct examples demonstrate cases where geographic cues ("capital city of Germany," "Pacific island terri-
tory"), temporal markers ("1950s," "Senate majority leader"), and contextual details (the Nazi regime, sugar planta-
tions, Caribbean communism) a re sufficiently distinctive for the model to identify entities across Place, Event, and 
Person types.  
 
Appendix C.2: Incorrect Predictions (O2: GPT -4o Few -shot)  
 
Incorrect Example 1: Wrong type  
"My great -grandfather left Lithuania in the early 1900s to escape oppression and seek a 
better life in America. He arrived before World War I and worked hard to establish himself, 
eventually sending for his family..."  
Gold : Solomon Goodman  |  Prediction : Lithuanian Jews   ✘ 
The model predicted a group/category rather than the specific individual being de-
scribed.  
 
 
Incorrect Example 2: Wrong type  
"I once kept a newspaper clipping of a write -up about a historic estate by a famous 19th -
century author, but unfortunately, I have misplaced it. The author's writing gave me some 
insight into the estate..."  
Gold : Harriet Beecher Stowe  |  Prediction:  Monticello   ✘ 
The model focused on the "historic estate" cue rather than the "famous 19th -century 
author" cue.  
 
 
 

 
 Incorrect Example 3: Wrong type  
"The priest who taught me algebra and later became the bishop of a diocese in eastern Wash-
ington questioned my presence in his advanced algebra class because I lacked the necessary 
background..."  
Gold : Bishop of Spokane  |  Prediction : West Point   ✘ 
The model produced a completely unrelated entity, likely confusing the religious 
context.  
 
Incorrect Example 4: Same -type, related  
"Remote healthcare delivery became a critical part of how we provided care during the viral 
outbreak. Initially, we relied on phone calls, but within weeks, our organization quickly 
implemented video..."  
Gold : Telehealth  |  Prediction : COVID -19 pandemic   ✘ 
The model identified the correct general domain but predicted the contextual event 
rather than the practice being described.  
 
Incorrect Example 5: Same -type, near miss  
"He was my Ph.D. advisor at the California university starting in 1956. He was a brilliant 
economist who later won the Nobel Prize, and studying under him greatly influenced my 
academic development. His mentorship helped shape my approach to economics and game 
theory."  
Gold : Kenneth Arrow  |  Prediction : John Forbes Nash Jr.   ✘ 
The model predicted a Nobel laureate economist associated with game theory, but 
confused the advisor (Arrow, at Stanford) with another famous figure in the same 
field.  
 
These errors illustrate the principal challenges of implicit entity recognition: distinguishing the referenced entity 
from related contextual entities (Examples 4, 5), resolving references to obscure individuals (Examples 1, 3), and 
focusing on the correct  cue among multiple competing signals (Example 2). Example 5 is particularly instructive: 
both Kenneth Arrow and John Nash are Nobel laureate economists linked to game theory, but the cues (Ph.D. advi-
sor, California, 1956) point specifically to Arrow at St anford.  
 
 
 
 

 
 References  
1. Boyd, Doug, ' Achieving the Promise of Oral History in a Digital Age', in Donald A. Ritchie (ed.), The Oxford Handbook of 
Oral History, Oxford Handbooks (2010; online edn, Oxford Academic, 18 Sept. 2012), https://doi.org/10.1093/ox-
fordhb/9780195339550.013. 0021, accessed 27 Apr. 2026. .  
2. Lazar, A., Demiris, G., and Thompson, H. (2016). Evaluation of a multifunctional technology system in a memory care unit: 
Opportunities for innovation in dementia care. Informatics for Health and Social Care, 41(4):373 -389. 
3. Subramaniam, P. and Woods, B. (2012). The impact of individual reminiscence therapy for people with dementia. Expert Re-
view of Neurotherapeutics, 12(5):545 -555. 
4. Nadeau, D. and Sekine, S. (2007). A survey of named entity recognition and classification. Lingvisticae Investigationes, 30(1 ):3-
26. 
5. Li, J., Sun, A., Han, J., & Li, C. (2020). A survey on deep learning for named entity recognition. IEEE transactions on knowledge 
and data engineering, 34(1), 50 -70.. 
6. Ganea, O. E., & Hofmann, T. (2017, September). Deep joint entity disambiguation with local neural attention. In Proceedings o f 
the 2017 conference on empirical methods in natural language processing (pp. 2619 -2629).  
7. Kolitsas, N., Ganea, O. E., & Hofmann, T. (2018, October). End -to-end neural entity linking. In Proceedings of the 22nd confer-
ence on computational natural language learning (pp. 519 -529)..  
8. Lee, K., He, L., Lewis, M., & Zettlemoyer, L. (2017, September). End -to-end neural coreference resolution. In Proceedings of the 
2017 conference on empirical methods in natural language processing (pp. 188 -197).  
9. Hosseini, H. (2022). Implicit entity recognition and linking in tweets. PhD thesis, Toronto Metropolitan University.  
10. Hosseini, H. and Bagheri, E. (2021). Learning to rank implicit entities on Twitter. Information Processing & Management, 
58(3):102503.  
11. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016, June). Neural architectures for named entity 
recognition. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguis-
tics: hu man language technologies (pp. 260 -270).  
12. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre -training of deep bidirectional transformers for language 
understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational lin-
guistics: human language technologies, volume 1 (long and short papers) (pp. 4171 -4186). .  
13. Xie, T., Li, Q., Zhang, J., Zhang, Y., Liu, Z., & Wang, H. (2023, December). Empirical study of zero -shot NER with ChatGPT. In 
Proceedings of the 2023 conference on empirical methods in natural language processing (pp. 7935 -7956).  
14. Ashok, D., & Lipton, Z. C. (2023). Promptner: Prompting for named entity recognition. arXiv preprint arXiv:2305.15444.  
15. Sang, E. T. K., & De Meulder, F. (2003). Introduction to the CoNLL -2003 shared task: Language -independent named entity 
recognition. In Proceedings of the seventh conference on Natural language learning at HLT -NAACL 2003 (pp. 142 -147).  
16. Malmasi, S., Fang, A., Fetahu, B., Kar, S., & Rokhlenko, O. (2022, October). MultiCoNER: A large -scale multilingual dataset for 
complex named entity recognition. In Proceedings of the 29th international conference on computational linguistics (pp. 3798 -
3809). 
17. Li, J., Fei, H., Liu, J., Wu, S., Zhang, M., Teng, C., ... & Li, F. (2022, June). Unified named entity recognition as word -word relation 
classification. In proceedings of the AAAI conference on artificial intelligence (Vol. 36, No. 10, pp. 10965 -10973).  
18. Zhou, W., Zhang, S., Gu, Y., Chen, M., & Poon, H. (2023). Universalner: Targeted distillation from large language models for 
open named entity recognition. arXiv preprint arXiv:2308.03279.  
19. Wu, L., Petroni, F., Josifoski, M., Riedel, S., & Zettlemoyer, L. (2020, November). Scalable zero -shot entity linking with dense 
entity retrieval. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP) (pp. 
6397 -6407).  
20. De Cao, N., Izacard, G., Riedel, S., & Petroni, F. (2020, September). Autoregressive Entity Retrieval. In ICLR 2021 -9th Interna-
tional Conference on Learning Representations (Vol. 2021). ICLR.  
21. Ayoola, T., Tyagi, S., Fisher, J., Christodoulopoulos, C., & Pierleoni, A. (2022, July). Refined: An efficient zero -shot -capable 
approach to end -to-end entity linking. In Proceedings of the 2022 Conference of the North American Chapter of the Association 
for Computational Linguistics: Human Language Technologies: Industry Track (pp. 209 -220).  
22. Botha, J. A., Shan, Z., & Gillick, D. (2020, November). Entity linking in 100 languages. In Proceedings of the 2020 Conferenc e 
on Empirical Methods in Natural Language Processing (EMNLP) (pp. 7833 -7845).  
23. Butler, R. N. (1963). The life review: An interpretation of reminiscence in the aged. Psychiatry, 26(1), 65 -76. 

 
 24. Webster, J. D. (1993). Construction and validation of the Reminiscence Functions Scale. Journal of Gerontology, 48(5), P256 -
P262.  
25. Nikitina, S., Callaioli, S., and Baez, M. (2018). Smart conversational agents for reminiscence. Proceedings of the 1st Intern ational 
Workshop on Software Engineering for Cognitive Services, 52 -57. 
26. Pessanha, F., & Salah, A. A. (2021). A computational look at oral history archives. ACM Journal on Computing and Cultural 
Heritage (JOCCH), 15(1), 1 -16. 
27. Perera, N., Dehmer, M., & Emmert -Streib, F. (2020). Named entity recognition and relation detection for biomedical information 
extraction. Frontiers in cell and developmental biology, 8, 673.  
28. Hou, Y. (2020, July). Bridging anaphora resolution as question answering. In Proceedings of the 58th Annual Meeting of the 
Association for Computational Linguistics (pp. 1428 -1438)..  
29. Poesio, M., Stuckardt, R., and Versley , Y. (2016). Anaphora Resolution: Algorithms, Resources, and Applications. Springer.  
30. Treder, M. S., Lee, S., & Tsvetanov, K. A. (2024). Introduction to Large Language Models (LLMs) for dementia care and research. 
Frontiers in dementia, 3, 1385303.  
31. Broadbent, E., Stafford, R., & MacDonald, B. (2009). Acceptance of healthcare robots for the older population: Review and fut ure 
directions. International journal of social robotics, 1(4), 319 -330. 
32. De Jager, A., Fogarty, A., Tewson, A., Lenette, C., & Boydell, K. (2017). Digital storytelling in research: A systematic revi ew. 
The Qualitative Report..  
33. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diver se, 
explainable multi -hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language 
proces sing (pp. 2369 -2380).  
34. Petroni, F., Rocktäschel, T., Riedel, S., Lewis, P., Bakhtin, A., Wu, Y., & Miller, A. (2019, November). Language models as 
knowledge bases?. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th 
internationa l joint conference on natural language processing (EMNLP -IJCNLP) (pp. 2463 -2473).  
35. Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler et al. "Re-
trieval -augmented generation for knowledge -intensive nlp tasks." Advances in neural information processing systems 33 
(2020): 9459 -9474. 
36. Karpukhin, Vladimir, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen -tau Yih. 
"Dense passage retrieval for open -domain question answering." In Proceedings of the 2020 conference on empirical methods in 
natural language processing (EMNLP), pp. 6769 -6781. 2020.  
37. Hurst, Aaron, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, A. J. Ostrow et al. "Gpt -4o system 
card." arXiv preprint arXiv:2410.21276 (2024)..  
38. Touvron, Hugo, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov et al. "Llama 
2: Open foundation and fine -tuned chat models." arXiv preprint arXiv:2307.09288 (2023).  
39. Grattafiori, Aaron, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al -Dahle, Aiesha Letman 
et al. "The llama 3 herd of models." arXiv preprint arXiv:2407.21783 (2024).  
40. Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen -Zhu, Yuanzhi Li, Shean Wang, Liang Wang, and Weizhu Chen. "Lora: 
Low -rank adaptation of large language models." Iclr 1, no. 2 (2022): 3.  
41. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). Qlora: Efficient finetuning of quantized llms. Advances in  
neural information processing systems, 36, 10088 -10115.  
42. Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D., & Nie, J. Y. (2024, July). C -pack: Packed resources for general chinese 
embeddings. In Proceedings of the 47th international ACM SIGIR conference on research and development in information 
retrieval (pp. 641 -649).  
43. Wei, Jason, Xuezhi  Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. "Chain -of-thought 
prompting elicits reasoning in large language models." Advances in neural information processing systems 35 (2022): 24824 -
24837.  