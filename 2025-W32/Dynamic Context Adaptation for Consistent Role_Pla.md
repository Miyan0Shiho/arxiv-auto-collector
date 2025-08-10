# Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations

**Authors**: Jeiyoon Park, Yongshin Han, Minseop Kim, Kisu Yang

**Published**: 2025-08-04 03:27:05

**PDF URL**: [http://arxiv.org/pdf/2508.02016v1](http://arxiv.org/pdf/2508.02016v1)

## Abstract
We propose AMADEUS, which is composed of Adaptive Context-aware Text Splitter
(ACTS), Guided Selection (GS), and Attribute Extractor (AE). ACTS finds an
optimal chunk length and hierarchical contexts for each character. AE
identifies a character's general attributes from the chunks retrieved by GS and
uses these attributes as a final context to maintain robust persona consistency
even when answering out of knowledge questions. To facilitate the development
and evaluation of RAG-based RPAs, we construct CharacterRAG, a role-playing
dataset that consists of persona documents for 15 distinct fictional characters
totaling 976K written characters, and 450 question and answer pairs. We find
that our framework effectively models not only the knowledge possessed by
characters, but also various attributes such as personality.

## Full Text


<!-- PDF content starts -->

Dynamic Context Adaptation for Consistent Role-Playing Agents with
Retrieval-Augmented Generations
Jeiyoon Park1, Yongshin Han1, Minseop Kim1, Kisu Yang2
1SOOP,2Korea University
{naruto, winterfeb, usio }@sooplive.com, willow4@korea.ac.kr
Abstract
The burgeoning prominence of role-playing agents (RPAs)
with large language models (LLMs) is driven by their capac-
ity to model complex social interactions and human-like be-
haviors in controlled, reproducible environments. However,
collecting utterances from a specific character or individ-
ual is costly and presents substantial challenges for contin-
uously updating character attributes and model parameters,
especially for daily streamers or VTubers whose persona in-
formation frequently changes. Although retrieval-augmented
generation (RAG) can alleviate this problem, there are three
underlying limitations. (i) Previous RAG methods truncate
each character’s persona paragraph to a fixed length, regard-
less of hierarchical contexts and varying lengths across char-
acters, which results in responses with lower persona con-
sistency. (ii) If a persona does not contain knowledge rele-
vant to a given query, RAG-based RPAs are prone to hal-
lucination, making it challenging to generate accurate re-
sponses. (iii) Existing role-playing datasets are composed of
dialogues, and there is no dataset designed for efficiently con-
structing and evaluating RPAs based on RAG. To address the
aforementioned challenges, we propose A MADEUS , which is
composed of Adaptive Context-aware Text Splitter (ACTS),
Guided Selection (GS), and Attribute Extractor (AE). ACTS
finds an optimal chunk length and hierarchical contexts for
each character. AE identifies a character’s general attributes
from the chunks retrieved by GS and uses these attributes
as a final context to maintain robust persona consistency
even when answering out-of-knowledge questions. To facil-
itate the development and evaluation of RAG-based RPAs,
we construct CharacterRAG, a role-playing dataset that con-
sists of persona documents for 15 distinct fictional characters
totaling 976K written characters, and 450 question–answer
pairs. We find that our framework effectively models not only
the knowledge possessed by characters, but also various at-
tributes such as personality.
Introduction
Large language models with long-context capabilities are
engineered to manage lengthy input sequences, allowing
them to interpret and utilize extended contextual informa-
tion. (OpenAI 2025; Qwen et al. 2025; Gemini Team 2025,
2024). Although LLMs exhibit enhanced abilities in under-
standing extended contexts, they still face significant chal-
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.lenges when handling tasks involving genuinely long con-
texts (Li et al. 2024a). Furthermore, utilizing all relevant in-
formation from long-context models to answer each query
can be computationally expensive (Li et al. 2024b).
Retrieval-augmented generation (RAG) cost-efficiently
mitigates factual inaccuracies and hallucinations in respond-
ing to knowledge-intensive queries by integrating external
retrieval mechanisms that provide accurate and up-to-date
supporting information (Gao et al. 2023; Huang et al. 2025).
However, despite these advantages of RAG, there has been
little research on RAG-based role-playing agents (RPAs).
Moreover, existing role-playing datasets are composed ex-
clusively of dialogues involving characters that are difficult
to collect, and there is no dataset designed for building and
evaluating RAG-based RPAs.
In this paper, we examine the challenges inherent in
RAG-based role-playing and propose approaches to miti-
gate these challenges. In real-world applications, users and
RPAs frequently engage in conversations on topics that ex-
tend beyond the knowledge defined in the character’s per-
sona. However, we observe that the existing RAG method
tends to excessively utilize chunks that are less relevant to
the question when the question is not explicitly answered by
the available knowledge (Figure 1). To address this issue and
maintain persona consistency, we first introduce an Adaptive
Context-aware Text Splitter (ACTS), which segments each
character’s persona using optimal chunk length and overlap,
thereby preserving context across chunks while simultane-
ously incorporating hierarchical context information within
each chunk. Then, the Guided Selection (GS) retrieves ap-
propriate chunks in order to infer information relevant to
the question from the character’s persona such as previous
actions and behaviors. Finally, the Attribute Extractor (AE)
identifies the general attributes of the character from the re-
trieved chunks, encouraging the RPA to respond in a manner
consistent with that character. Note that if GS fails to find a
chunk corresponding to the question, AE extracts the char-
acter’s attributes using the chunks with the highest similarity
scores, which enables consistent responses.
We conduct extensive experiments to examine the factors
influencing the performance of RAG-based role-playing,
utilizing various settings with 15 fictional character RPAs,
the manually constructed CharacterRAG dataset, and mul-
tiple psychological questionnaire datasets. Results demon-arXiv:2508.02016v1  [cs.AI]  4 Aug 2025

Figure 1: Chunk Duplication Frequencies. We compare
the distribution of chunk duplication frequencies and chunk
usage rates between Naive RAG and our method when ques-
tions involving knowledge not present in the persona doc-
ument were given. We observe that when each of the 60
MBTI questions is asked to 15 fictional characters, the aver-
age chunk usage rate increases from 34.93% to43.84%, and
the distribution becomes more uniform.
strate that our framework reveals new possibilities for RAG-
based RPAs. We summarize our contributions as follows:
• We propose AMADEUS, a RAG-based RPA framework
that not only elicits information related to a character, but
also maintains persona consistency even when respond-
ing to queries beyond its explicit knowledge.
• We manually construct CharacterRAG, a role-playing
dataset for implementing and evaluating RAG-based
RPAs comprising persona documents for 15 distinct fic-
tional characters totaling 976K written characters, and
450 question–answer pairs.
• We systematically investigate and uncover key consider-
ations for building RAG-based RPAs through extensive
experiments performed in a range of settings.Related Work
Role-Playing Agents. Early persona-grounded dialogue
work such as P ERSONA -CHAT demonstrated that condition-
ing dialogue models on explicit character descriptions im-
proves conversational coherence and engagement (Zhang
et al. 2018). With the advent of LLMs, researchers have pur-
sued finer-grained persona consistency . Ji et al. (2025) pro-
pose a Persona-aware Contrastive Learning framework that
automatically aligns generations with the target character.
CHARACTER GPT (Park, Park, and Lim 2025) proposes a
framework that implements four fictional characters through
eight attributes. DITTO (Lu et al. 2024) leverages charac-
ter information to prompt instruction-following LLMs to to
engage in role-playing dialogues, treating them as a form
of reading comprehension. C HARACTER GLM (Zhou et al.
2024) introduces an open LLM family that aligns social
characters with social traits
Complementary benchmarks soon followed, along with
various evaluation methods. RPE VAL (Boudouri et al.
2025) evaluates emotional understanding, decision-making,
moral alignment, and in-character fidelity in single-turn
settings, while T IMECHARA (Ahn et al. 2024) probes
point-in-time role-playing and exposes temporal halluci-
nations in state-of-the-art LLMs. I NCHARACTER (Wang
et al. 2024b) adopts an interview-driven methodology to
precisely assess RPA personalities by eliciting their under-
lying mindsets and behaviors. Wang et al. (2025) provides
a dynamic, multi-agent virtual environment specifically de-
signed to draw out nuanced, human-like behaviors from
large language models during role-playing assessments.
However, current RPAs still truncate persona text, halluci-
nate beyond retrieved knowledge, and lack a benchmark for
incremental updates—gaps our framework fills. In this pa-
per, we propose a novel RAG-based role-playing framework
that facilitates the addition and deletion of information.
Retrieval-Augmented Generation (RAG). RAG cou-
ples non-parametric memory with an LLM to miti-
gate hallucination and stale knowledge (Lewis et al.
2020). Fixed-length chunking can either omit critical con-
text or waste prompt budget. Recent analyses quantify
this trade-off. Bhat et al. (2025) systematically evaluate
fixed-size chunking across multiple datasets and embedding
models, showing that optimal sizes vary widely with do-
main and answer locality (Bhat et al. 2025). To move be-
yond one-size-fits-all heuristics, Zhong et al. (2025) intro-
duce M IX-OF-GRANULARITY , a router that dynamically
selects chunk granularity per query and boosts medical QA
accuracy (Zhong et al. 2025). Evaluation toolkits are like-
wise evolving. RAGE VAL generates scenario-specific doc-
uments, QA pairs, and proposes Completeness, Halluci-
nation, and Irrelevance metrics for rigorous RAG assess-
ment (Zhu et al. 2024). Nevertheless, no existing bench-
mark explicitly targets role-playing under RAG, and prior
work still assumes personas are short, knowledge-dense
snippets. We mitigate this discrepancy with a novel text
splitter that tailors chunk lengths and hierarchical context
to each character. We also introduce C HARACTER RAG, the
first dataset designed for building and evaluating RAG-based
role-playing agents across 15 fictional personas.

# Yui's Characteristics, Clumsy Girl (Dojikko):
-Yui usually sleeps in late. Occasionally, when 
she decides she has to do something, she wakes 
up ridiculously early. As she goes through high 
school, she gradually wakes up late less often.
-Yui is lazy, and her hobby is lazing around at 
home. After joining the light music club, she 
became absorbed in the guitar, but before joining, 
her daily routine was just lying around. During 
holidays, she sometimes buries herself in the 
kotatsu and doesn’t come out.
-Yui joined the light music club without knowing 
how to play any instruments at all. She didn’t 
even have a proper instrument, so at Ritsu’s
suggestion, the club members worked part -time 
together to save money, but it wasn’t enough. 
Later, thanks to Mugi’s help, she was able to buy 
a guitar at a much cheaper price. Although she got 
help from her friends to buy the guitar, from the 
store’s perspective, they still caused some trouble.
Adaptive Context -aware Text Splitter (ACTS)
Persona Documents
Belief and Value
Psychological Traits
## Yui Hirasawa's Belief and Value:
Yui Hirasawa values the small joys and comfort in 
everyday life. She cherishes the time spent with 
friends and is open to both receiving and giving help. 
She isn't afraid to take on challenges or learn new 
things, and she maintains a positive attitude even 
when she fails. She naturally accepts her own laziness 
and carelessness, and she prioritizes enjoyment and 
present satisfaction over perfection.## Attributes:## Knowledge:
##  Yui Hirasawa’s Psychological Traits:
Yui Hirasawa has an innocent and clumsy (dojikko) 
personality, and she tends to be laid -back and lazy. 
She enjoys spending time alone, such as lazing 
around at home, but when she's with friends, she is 
cheerful, sociable, and also likes group activities. She 
expresses her emotions honestly, tends to 
procrastinate with bothersome tasks, and immerses 
herself in things she loves. She also has a passionate 
side that allows her to quickly dive into new hobbies 
(such as playing the guitar).
Persona Documents
+Optimal Chunk Length
+Hierarchical Context
Do you prefer hobbies or activities that you can do alone rather than group activities?
1.
2.
3.0.712
0.681
0.651
4. 0.648
…Guided Selection (GS)
Attribute Extractor (AE)RAG (+ ACTS)
# Context:
Yui’s Response
## Role:
Refer to the given Knowledge and Attributes 
of Yui Hirasawa and answer the user's 
questions as if Yui Hirasawa .
Hmm~ I really love just lazing around in the kotatsu by 
myself or practicing guitar in my room! Playing alone is 
fun and comfortable, so I like it a lot. Hehe…
But, eating cake with Ho -kago Tea Time friends, 
practicing together, and making lots of noise is super, 
super fun too!
It’s hard to say exactly which one I like better, but I think 
both my alone time and time with my friends are precious 
and make me happy!
Hmm... Since I enjoy both, maybe I can practice guitar by 
myself today, and then go hang out with Ricchan and 
Mio-chan tomorrow~ That sounds good, don’t you think?
Figure 2: An overview of our framework. The Adaptive Context-aware Text Splitter (ACTS) divides a persona into meaningful
chunks, selecting the optimal chunk length and overlap for each character. Each chunk is further enhanced with hierarchical
context, aiming to minimize information loss about the character. To answer questions beyond the character’s pre-existing
knowledge, the Guided Selection (GS) module identifies chunks related to the relevant knowledge, while the Attribute Extractor
(AE) extracts general attributes of the character from these chunks. Finally, our framework generates vivid responses by treating
the chunks obtained from the RAG as knowledge and the outputs from GS and AE as the character’s attributes.
Task Formulation
Given user query u, RAG-based RPAs can be formulated as:
R=f(u,Dp) (1)
, where Dpis a character’s persona, and fis a RPA. A text
splitter gdivides the persona into nchunks as follows:
g(Dp) ={c1, c2, ..., c n} (2)
Each chunk cicontains the character’s attributes such as re-
lationships andbackstory . The objective of fis to vividly
embody the character and generate response Rtouwhile
maintaining persona consistency. However, previous RAG
methods (Guu et al. 2020; Guo et al. 2024; Yang et al. 2024;
Shukla et al. 2025; Wang, Leung, and Shen 2025) truncate
each character’s persona paragraph to a fixed length, regard-
less of the varying lengths across characters, which results in
hallucinations or responses with lower persona consistency.
In addition, if the persona does not contain knowledge rel-
evant to the given query, RAG-based role-playing methods
are prone to hallucination, making it challenging to generate
accurate responses.
Method
As shown in Figure 2, our framework consists of (i) Adap-
tive Context-aware Text Splitter (ACTS), (ii) Guided Selec-tion (GS), and (iii) Attribute Extractor (AE), in order to build
a realistic role-playing agent based on RAG.
Adaptive Context-aware Text Splitter
Given the character’s persona Dp, ACTS aims to preserve
the context between chunks, as well as the information
about corresponding subsections within the persona for each
chunk, that is, the hierarchical context. To this end, ACTS
first finds the maximum length of the paragraphs that consti-
tute the persona as follows:
lmax=φ(p1, p2, ..., p l) (3)
, where φdenotes a length-calculating function. Then,
ACTS sets the overlap length of the text splitter to half of
lmax(i.e.,lo=lmax/2). Note that the reason for setting the
overlap length sufficiently large is to minimize information
loss, as the context between pieces of information contained
in each chunk is indispensable in RAG-based role-playing.
Finally, Dpis segmented using lmaxandloand hierarchical
context Hiis appended to each chunk in order to preserve
information such as descriptions of the corresponding char-
acters and situational context at each point in the narrative:
ACTS (Dp,H, lmax, lo) ={ˆc1,ˆc2, ...,ˆcm} (4)
ˆci=ci+Hi (5)

Figure 3: Statistics of CharacterRAG Dataset. CharacterRAG consists of persona documents for 15 distinct fictional charac-
ters totaling 976K written characters, and 450 question–answer pairs.
Algorithm 1: Guided Selection (GS)
Input : User query u; Set of knowledge chunks C; Maximum
number of search iterations N; Slot size K
Output : Selected chunk set S
1:Initialize slot S=∅
2:Sort chunks Cin descending order according to similar-
ity to u, obtaining Csorted
3:Set iteration counter t←0
4:foreach chunk cinCsorteddo
5: ift≥Nor|S| ≥Kthen
6: break
7: end if
8: With an LLM, determine if chunk ccontains infor-
mation from which the character’s attributes can be
inferred regarding u
9: ifthe LLM returns True then
10: Addcto slot S
11: end if
12: t←t+ 1
13:end for
14:if|S|= 0then
15: S←Top-K+ 1chunks from Csorted (highest similar-
ity to u)
16:end if
17:return S
Guided Selection
While RAG has demonstrated significant potential in im-
proving factual correctness of LLMs, existing RAG methods
tend to generate uninformative responses (e.g., I’m sorry, but
I don’t have specific information. ) for questions outside their
knowledge base (Guo et al. 2024; Shukla et al. 2025; Wang,
Leung, and Shen 2025), or excessively and repetitively use
irrelevant chunks that are not pertinent to the given query.
In this paper, GS focuses on selecting appropriate chunksto generate natural and persona-consistent responses. GS is
composed of three stages. First, we iterate over the chunks,
which are sorted in descending order of similarity to the user
query u, and employ an LLM to determine whether it is pos-
sible to infer the corresponding character’s attributes from
each chunk for the u. Second, the chunks selected in the pre-
vious step are added to the slot, and the iteration terminates
when the slot is full. Finally, if the slot remains empty af-
ter the maximum number of search iterations, the Kchunks
with the highest similarity to the query is returned.
GS is effective in identifying chunks containing informa-
tion that can be inferred from a character’s actions, such as
beliefs or personality traits, which are not explicitly stated
in the knowledge base and therefore are difficult to re-
trieve through direct search. For example, if there is no ex-
plicit knowledge corresponding to the query ”My living and
work spaces are clean and organized” but the information
”Megumin is diligent. ” is available, RPA can use this infor-
mation to infer the characteristics of Megumin and generate
an appropriate response. GS is summarized in Algorithm 1.
Attribute Extractor
Inspired by the observations that incorporating character’s
attributes can lead to more realistic responses (Park, Park,
and Lim 2025; Chen et al. 2025), AE considers two at-
tributes: Belief and Value , and Psychological Traits . Beliefs
and values are fundamental principles and ideological ori-
entations that inform and influence a character’s viewpoints
and choices. On the other hand, psychological traits refer to
characteristics related to personality, emotional states, per-
sonal interests, and cognitive tendencies. AE extracts at-
tributes of a character from the chunks generated as a result
of GS, and dynamically exploits them as context.
Experiments
Setup
Dataset. We construct the CharacterRAG dataset, which
consists of 15 fictional characters, to leverage a RAG-based

BGE-M3 Qwen3 mE5 large-instruct PµPσ2PµPσ2PµPσ2
Vanilla 6.4325 0.1026 8.3306 0.1557 12.3136 0.0071
ATS 6.7007 0.0884 8.4718 0.1281 12.2336 0.0070
ACTS (Ours) 6.8575 0.0784 8.6226 0.1179 12.3240 0.0063
Table 1: Distribution of Similarity Scores. Vanilla refers
to the method of splitting chunks with a fixed length and
overlap, ATS refers to dividing a persona for each character
with optimal segment length and overlap, without hierarchi-
cal context.
role-playing framework. CharacterRAG is a high-quality,
role-playing dataset in which all external information about
works featuring characters that could affect persona consis-
tency has been manually removed, and each persona doc-
ument has been directly reconstructed from the perspec-
tive of each character by human annotators1. For instance,
any information speculated from the perspective of editors
rather than the characters themselves, as well as informa-
tion such as character popularity polls that may disrupt role-
playing, is excluded. CharacterRAG consists of 15 distinct
fictional characters, 976K written char- acters, and 450 ques-
tion–answer pairs. Six attributes2define each character’s
persona and the corresponding QA pairs (Chen et al. 2025):
•Activity : A documented history comprising prior activi-
ties, behaviors, and interactions, encompassing elements
such as backstory andschedules .
•Belief and Value : The foundational tenets, dispositions,
and ideological orientations that inform and guide a char-
acter’s viewpoints and decision-making processes (e.g.,
beliefs andattitudes ).
•Demographic Information : Information that can identify
a character, including but not limited to their name, age,
educational background, professional history, and geo-
graphic location.
•Psychological Traits : Attributes associated with person-
ality traits, emotional states, preferences, and patterns of
cognitive behavior.
•Skill and Expertise : The extent of understanding, skillful-
ness, and competence regarding particular fields or tech-
nologies.
•Social Relationships : The characteristics and processes
of social interactions, encompassing individuals’ roles,
relational ties, and patterns of communication.
Each section of the character’s persona contains subsec-
tions, preserving the hierarchical context of each chunk
(e.g., ”Frieren’s activities in the story, the journey to find
Aureole, the First-Class Mage Exam” ). Furthermore, Each
QA pair consists of a question and corresponding answer
derived from the character’s knowledge, pertaining to one
1CharacterRAG dataset is sourced from Namuwiki and is based
on Korean data: https://namu.wiki/
2Unlike the other attributes, Belief and Value and Psychological
Traits directly impact the behavior of a character; therefore, AE
utilizes information from these two attributes.µ σ Mdn Cronbach’s alpha
BFI 3.890 0.965 4.100 0.845
MBTI 3.870 0.911 3.933 0.830
Table 2: Human Evaluation. We conduct human evalua-
tion using a 5-point Likert scale to verify whether inferring
character attributes with the Attribute Extractor (AE) from
chunks extracted by Guided Selection (GS) is reasonable.
Note that Srepresents the results of all human evaluators, µ
isE(E(S)),σisE(σ(S)), and Mdn isE(Mdn (S)).
Chunk Usage Rate GS Failure
CharacterRAG 67.56% 25.11%
MBTI 43.84% 15.33%
Table 3: Average Chunk Usage Rate and GS Failure. We
present 30 CharacterRAG questions and 60 MBTI questions
to 15 characters, and measure the average chunk usage rate
and GS failure.
of the six attributes manually constructed for each character.
Detailed statistics of CharacterRAG are shown in Figure 3.
Baselines. We evaluate our method against three off-the-
shelf RAG baselines: Naive RAG (Gao et al. 2024), CRAG
(Yan et al. 2024), and LightRAG (Guo et al. 2024). We also
conduct extensive experiments on three different LLMs and
three different embedding models: GPT-4.1 (OpenAI 2025),
Gemma3-27B (Team et al. 2025), Qwen3-32B (Yang et al.
2025), BGE-M3 (Chen et al. 2024), Qwen3-0.6B (Zhang
et al. 2025), and mE5 large-instruct (Wang et al. 2024a).
Settings. We implement Guided Selection (GS) and At-
tribute Extractor (AE) using GPT-4.1 ( ”gpt-4.1-2025-04-
14”). We apply the ACTS to the conventional Naive RAG.
The maximum number of search iterations Nis 30, and
the slot size Kis set to 2. Note that the speed at which
GS identifies and retrieves relevant chunks for answering
is much faster and more cost-efficient than that of previ-
ous graph-based or web search-based RAG methods (Shukla
et al. 2025; Guo et al. 2024; Yan et al. 2024).
Evaluation Protocols
Tasks. We use 450 QA pairs from the CharacterRAG dataset
to verify whether the RPA sufficiently leverages each char-
acter’s knowledge. We also use 60 MBTI3questions and
120 BFI (Barrick and Mount 1991) questions to investigate
whether each character can appropriately respond to ques-
tions for which they do not have explicit prior knowledge.
Since it is not possible to construct QA pairs for questions
outside the scope of a character’s knowledge, we instead
conduct an interview-based assessment (Wang et al. 2024b)
for each character and compare the results to psychological
test outcomes for the character, as determined by thousands
of actual participants’ votes4.
Metrics. We design three metrics to extensively evaluate
the role-playing capabilities of RAG-based RPAs: (i) ACC
3https://www.16personalities.com/
4https://www.personality-database.com/

CRAG LightRAG AMADEUS (Ours) GT
Anya Forger INTJ (-3) INFJ (-2) ENFP (0) ENFP
Chika Fujiwara ENFP (0) INTP (-2) ENFP (0) ENFP
Edward Elric ISTJ (-3) INTP (-1) INFP (-2) ENTP
Frieren INFP (-1) INTP (0) INTP (0) INTP
Hitori Gotoh ISTJ (-2) ENFP (-1) INFP (0) INFP
Light Yagami INTJ (-1) INTJ (-1) INTJ (-1) ENTJ
M¯ao M ¯ao INTJ (-1) INTJ (-1) ISTP (-1) INTP
Megumin INFP (0) INFP (0) ISFP (-1) INFP
Mikoto Misaka ISFP (-4) ENTJ (0) INFJ (-2) ENTJ
Nina Iseri ISFP (0) ENFJ (-3) INFP (-1) ISFP
Saitama ISTP (0) INTP (-1) ISTP (0) ISTP
Son Goku ENFJ (-2) INTP (-3) ESFP (0) ESFP
Tanjiro Kamado ENFJ (0) INTP (-3) ENFJ (0) ENFJ
Tobio Kageyama ENTJ(-2) INFJ (-1) ISTJ (0) ISTJ
Yui Hirasawa ENFP (0) INTP (-2) ESFP (-1) ENFP
Accuracy 68.33% 65.00% 85.00% -
Avg F1-Score 0.6448 0.5344 0.8244 -
Table 4: Predicted MBTI Types per Character. The num-
ber in parentheses indicates the number of times the ground-
truth (GT) type of each character was not correctly identified
(using the GPT-4.1 setting).
measures whether the character’s response contains the cor-
rect answer or not. (ii) ACC Lis a score assigned by the
LLM, ranging from 1 to 10, that evaluates how well the char-
acter’s response reflects the correct answer. (iii) Hallucina-
tion Score (HS) evaluates the degree of hallucination in the
model’s response given a query, the relevant chunks, and the
ground-truth answer, on a scale from 1 to 10. Specifically,
HS is assigned close to 1 when the response faithfully re-
flects only the facts contained in the chunks or answer with-
out distortion or addition, indicating minimal hallucination.
Experimental Results
Optimal persona segmentation and hierarchical contex-
tualization are highly effective. We analyze the distribu-
tions of similarity scores to examine whether splitting the
text into optimally sized chunks for each character’s persona
and incorporating hierarchical context is effective. In Table
1, we provide each character with 30 questions, resulting in a
total of 450 questions, related to their respective knowledge
from the CharacterRAG dataset, and measure the similarity
between each question and the chunks retrieved by the RAG
model under the three different embedding settings: BGE-
M3, Qwen3-0.6B, and mE5 large-instruct . Results demonstrate
that compared to splitting text into fixed-length chunks,
the Adaptive Text Splitter (ATS), which segments text with
an optimal persona length and overlap for each character,
achieves a higher average score and lower variance. This in-
dicates that each chunk generated using ATS contains richer
semantic information for the same query. Building on this,
Adaptive Context-aware Text Splitter (ACTS), which con-
siders hierarchical context in addition to ATS, consistently
achieves better performance across all three embedding set-
tings. This results show that optimal chunk length, appro-
priate overlap, and consideration of hierarchical context all
play essential roles in effective text chunking.
Extracting a character’s attributes from selected text
chunks is reliable. We investigate the reasonableness of in-
ferring character attributes with the Attribute Extractor (AE)
from chunks extracted via Guided Selection (GS) by con-CRAG LightRAG AMADEUS (Ours) GT
Anya Forger SLOAI (-2) RCUEN (-3) SLUEI (-2) SCUAI
Chika Fujiwara SCUAI (0) RLUEN (-4) SCOAI (-1) SCUAI
Edward Elric SLOEI (-1) RCUAN (-4) SLOEI (-1) SLUEI
Frieren RCUAI (-1) SLUEN (-3) RCUAI (-1) RCUEI
Hitori Gotoh RLUAI (0) RCUEN (-3) RLUAI (0) RLUAI
Light Yagami SCOEI (-1) RCUAN (-3) SCOEI (-1) RCOEI
M¯ao M ¯ao RCOAN (-2) SLUAN (-5) RCOEI (0) RCOEI
Megumin SCUAI (-2) RLUEN (-2) SLOEI (-1) SLUEI
Mikoto Misaka SLOAI (-3) RCUEN (-2) SLOAI (-3) RCOEI
Nina Iseri RLUAI (-1) RCUEN (-2) SLUEI (-1) RLUEI
Saitama RCOAN (-1) SCOAI (-3) RCUAN (0) RCUAN
Son Goku SCUAI (-1) RLUEI (-4) SCOAI (-2) SCUAN
Tanjiro Kamado SCOAI (0) RCUAN (-3) SCOAI (0) SCOAI
Tobio Kageyama SLOAN (-2) RCUAI (-4) RCOEN (-1) RLOEN
Yui Hirasawa SLUAI (-1) RCOEN (-4) SCUAI (0) SCUAI
Accuracy 76.00% 34.67% 81.33% -
Avg F1-Score 0.7313 0.2774 0.7986 -
Table 5: Predicted Big 5 SLOAN Types per Character.
The number in parentheses indicates the number of times the
ground-truth (GT) type of each character was not correctly
identified (using the GPT-4.1 setting).
ducting human evaluation using a 5-point Likert scale. To
this end, we invite 11 human evaluators and each evaluator
is asked to score 60 randomly selected samples. Each sample
consists of pairs of chunks selected from GS and attributes
extracted through AE, for 30 BFI questions and 30 MBTI
questions that are not included in the character’s knowledge.
In Table 2, we find that the means µis close to 4, with
small standard deviations σ. It demonstrates that the outputs
of GS and AE are reliable and trustworthy, even from a hu-
man evaluative perspective. We also measure Cronbach’s al-
pha (Cronbach 1951) to evaluate internal consistency among
the human evaluators. We find that the Cronbach’s alpha val-
ues are 0.845 and 0.830, both exceeding the commonly ac-
cepted threshold of 0.7 for acceptable reliability. Since val-
ues above 0.8 are generally interpreted as indicating a high
level of internal consistency, experimental results in Table 2
can be considered highly trustworthy.
Selecting appropriate chunks increases the utilization
of a character’s persona. Table 3 shows the usage rate of
persona chunks and the failure rate of GS when each each
of the 15 characters is presented with questions that are both
within and outside their knowledge. Note that, despite the
limited number of queries in CharacterRAG, the Chunk Us-
age Rate still reaches 67.56%. The findings demonstrate that
GS serves as an effective tool for enabling the use of char-
acter personas. At the same time, it can be observed that the
450 questions in CharacterRAG are designed to capture the
core information of each character.
Meanwhile, as shown in Table 3 and Figure 1, GS sub-
stantially increases the chunk utilization rate for the 60
MBTI questions, from 34.93% to 43.84%. The proportion
of cases where the GS fails to find an appropriate chunk,
i.e., GS Failure, is 25.11% and 15.33%, respectively, which
demonstrates that although GS effectively utilizes the char-
acter’s persona, there is still room for further improvement.
Nevertheless, despite this shortcoming, our framework of-
fers markedly enhanced speed, cost-effectiveness, and stable
performance compared to earlier RAG approaches utilizing
graph structures or web search mechanisms. (Shukla et al.
2025; Guo et al. 2024; Yan et al. 2024).

Method TaskGPT-4.1 Gemma3-27B Qwen3-32B
ACC ↑ ACCL↑HS↓ ACC ↑ ACCL↑HS↓ ACC ↑ ACCL↑HS↓
w/o RAG CharacterRAG 49.56% 6.79 - 27.56% 5.33 - 18.89% 4.35 -
Naive RAG CharacterRAG 91.33% 9.23 3.13 86.44% 8.85 3.27 78.44% 8.49 5.05
LightRAG CharacterRAG 48.00% 6.06 - 69.56% 8.17 - 68.67% 8.20 -
CRAG CharacterRAG 70.00% 8.26 3.21 57.78% 7.57 4.09 28.67% 5.24 8.68
AMADEUS (Ours) CharacterRAG 92.67% 9.26 2.89 88.00% 8.92 3.26 78.89% 8.63 4.66
Naive RAG MBTI - - 2.69 - - 2.53 - - 2.33
CRAG MBTI - - 2.38 - - 2.91 - - 1.80
AMADEUS (Ours) MBTI - - 2.05 - - 2.02 - - 2.04
Naive RAG BFI - - 2.74 - - 2.52 - - 2.42
CRAG BFI - - 2.26 - - 2.75 - - 1.96
AMADEUS (Ours) BFI - - 1.94 - - 1.99 - - 2.03
Table 6: Evaluation of role-playing capabilities on CharacterRAG, MBTI, and BFI tasks. Higher values of ACC (%) and
ACC L(1-10) correspond to better performance, whereas lower values of HS (1-10) are preferable.
Graph-based RAG and web search-based RAG are
unsuitable for role-playing. One of the major challenges
in retrieval-based role-playing is that, when a RPA receives
questions involving knowledge outside a character’s per-
sona, it tends to either overuse irrelevant chunks (Figure
1) or generate uninformative responses (Guo et al. 2024;
Shukla et al. 2025; Wang, Leung, and Shen 2025).
To investigate whether RAG-based RPAs can handle this
problem, we conduct extensive experiments in which we ask
15 characters 60 MBTI questions and 120 BFI questions
each, and and evaluate their ability to accurately infer the
characters’ personality types. Table 4 and Table 5 show pre-
dicted MBTI types and Big 5 SLOAN types per character.
Our framework maintains persona consistency even when
answering questions that are not explicitly specified in each
character’s persona in both MBTI and BFI settings. Note
that the performance gap of CRAG is significant between
the two settings. We assume that questions requiring ana-
logical reasoning are difficult to solve even with web search
and that the search results may contain non-negligible noise.
On the other hand, LightRAG exhibits the lowest perfor-
mance, which shows that graph-based RAG methods are
not well suited for RPA applications due to the high cost
of graph construction, the difficulty in adding or removing
new knowledge, and challenges in maintaining persona con-
sistency. While we did not perform a direct comparative ex-
periment, we observed that GraphRAG (Shukla et al. 2025)
suffers from similar problems. Specifically, applying graph-
based RAG to RPA is challenging in cases where the persona
frequently changes, such as with daily streamers or VTubers.
CharacterRAG dataset serves as a valuable resource
for the construction and evaluation of RAG-based RPAs.
To investigate the factors influencing the performance of
role-playing, we conduct a comprehensive interview-based
assessments on the generalization capabilities of models
with various LLMs and RAG techniques. Table 6 presents
how the ability to accurately answer questions related to
the character’s knowledge, which is a core aspect of role-
playing, varies across the applied methodologies.
We first examine to what extent each LLM possesses
background knowledge about the 15 characters in a setting
without RAG, and results show that none of the three LLMs
are capable of effective role-playing without access to ex-ternal knowledge. Moreover, we observe that LightRAG, a
graph-based RAG, is ill-suited for the storage and retrieval
of character knowledge, as it often suffers from issues such
as entity ambiguity and uninformative responses. In a sim-
ilar vein, CRAG exhibits challenges in maintaining role-
playing fidelity, which can be attributed to the tendency of
web search-based RAG methods to utilize retrieved content
that may undermine the consistency of a character’s persona.
Indeed, despite leveraging web information, CRAG is able
to correctly answer only 6 out of the 30 CharacterRAG ques-
tions pertaining to Nina Iseri . In addition, to analyze how a
thinking mode of LLMs influences their role-playing capa-
bilities, we employ Qwen 3-32B. Results demonstrate that
the thinking mode fails to yield any substantial positive ef-
fect on enhancing role-playing performance.
Note that our framework achieves the best performance
across all three LLMs. We also find that the Hallucination
Score (HS) is the lowest in CharacterRAG setting. These re-
sults highlight the importance of preserving the context of
split-character personas and effectively leveraging appropri-
ate character attributes in RAG-based RPAs. Furthermore,
such elements are especially pronounced in dialogue situa-
tions that transcend the scope of the character’s knowledge
(Table 4 and Table 5). We believe that our findings demon-
strate new possibilities for RAG-based RPAs.
Conclusion
In this work, we addresse critical limitations in building
retrieval-augmented, RPAs with LLMs. By introducing a
novel framework consisting of an Adaptive Context-aware
Text Splitter (ACTS), Guided Selection (GS), and Attribute
Extractor (AE), our approach enables robust and consis-
tent simulation of character personas, even when confronted
with queries that extend beyond explicit persona knowledge.
Through the development of the CharacterRAG dataset, we
provide a valuable resource for reproducible evaluation and
benchmarking of RAG-based RPAs. Our experimental re-
sults demonstrate that the proposed method not only en-
hances the character’s knowledge representation, but also
faithfully models nuanced traits such as personality. We are
enthusiastic about the future prospects of RAG-driven role-
playing agents, along with the creation of stronger character
personas and improved RAG architectures.

References
Ahn, J.; Lee, T.; Lim, J.; Kim, J.-H.; Yun, S.; Lee, H.; and
Kim, G. 2024. TimeChara: Evaluating Point-in-Time Char-
acter Hallucination of Role-Playing Large Language Mod-
els. In Findings of the Association for Computational Lin-
guistics ACL 2024 , 3291–3325.
Barrick, M. R.; and Mount, M. K. 1991. The big five per-
sonality dimensions and job performance: a meta-analysis.
Personnel psychology , 44(1): 1–26.
Bhat, S. R.; Rudat, M.; Spiekermann, J.; and Flores-Herr,
N. 2025. Rethinking Chunk Size For Long-Document
Retrieval: A Multi-Dataset Analysis. arXiv preprint
arXiv:2505.21700 .
Boudouri, Y . E.; Nuninger, W.; Alvarez, J.; and Peter, Y .
2025. Role-Playing Evaluation for Large Language Mod-
els.arXiv preprint arXiv:2505.13157 .
Chen, C.; Yao, B.; Zou, R.; Hua, W.; Lyu, W.; Ye, Y .; Li, T.
J.-J.; and Wang, D. 2025. Towards a Design Guideline for
RPA Evaluation: A Survey of Large Language Model-Based
Role-Playing Agents. arXiv:2502.13012.
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and
Liu, Z. 2024. BGE M3-Embedding: Multi-Lingual, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. arXiv:2402.03216.
Cronbach, L. J. 1951. Coefficient alpha and the internal
structure of tests. psychometrika , 16(3): 297–334.
Gao, T.; Yen, H.; Yu, J.; and Chen, D. 2023. Enabling
Large Language Models to Generate Text with Citations.
In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings of
the 2023 Conference on Empirical Methods in Natural Lan-
guage Processing , 6465–6488. Singapore: Association for
Computational Linguistics.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai,
Y .; Sun, J.; Wang, M.; and Wang, H. 2024. Retrieval-
Augmented Generation for Large Language Models: A Sur-
vey. arXiv:2312.10997.
Gemini Team, G. 2024. Gemini 1.5: Unlocking multi-
modal understanding across millions of tokens of context.
arXiv:2403.05530.
Gemini Team, G. 2025. Gemini 2.5: Pushing the Fron-
tier with Advanced Reasoning, Multimodality, Long
Context, and Next Generation Agentic Capabilities.
arXiv:2507.06261.
Guo, Z.; Xia, L.; Yu, Y .; Ao, T.; and Huang, C. 2024. Ligh-
tRAG: Simple and Fast Retrieval-Augmented Generation.
Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M.-W.
2020. REALM: retrieval-augmented language model pre-
training. In Proceedings of the 37th International Confer-
ence on Machine Learning , ICML’20. JMLR.org.
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2025. A
Survey on Hallucination in Large Language Models: Prin-
ciples, Taxonomy, Challenges, and Open Questions. ACM
Transactions on Information Systems , 43(2): 1–55.Ji, K.; Lian, Y .; Li, L.; Gao, J.; Li, W.; and Dai, B. 2025. En-
hancing Persona Consistency for LLMs’ Role-Playing us-
ing Persona-Aware Contrastive Learning. arXiv preprint
arXiv:2503.17662 .
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural infor-
mation processing systems , 33: 9459–9474.
Li, T.; Zhang, G.; Do, Q. D.; Yue, X.; and Chen, W. 2024a.
Long-context LLMs Struggle with Long In-context Learn-
ing. arXiv:2404.02060.
Li, Z.; Li, C.; Zhang, M.; Mei, Q.; and Bendersky, M. 2024b.
Retrieval Augmented Generation or Long-Context LLMs?
A Comprehensive Study and Hybrid Approach. In Der-
noncourt, F.; Preot ¸iuc-Pietro, D.; and Shimorina, A., eds.,
Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing: Industry Track , 881–893.
Miami, Florida, US: Association for Computational Lin-
guistics.
Lu, K.; Yu, B.; Zhou, C.; and Zhou, J. 2024. Large Language
Models are Superpositions of All Characters: Attaining Ar-
bitrary Role-play via Self-Alignment. In Ku, L.-W.; Martins,
A.; and Srikumar, V ., eds., Proceedings of the 62nd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , 7828–7840. Bangkok, Thailand:
Association for Computational Linguistics.
OpenAI. 2025. Introducing GPT-4.1 in the API.
Park, J.; Park, C.; and Lim, H. 2025. CharacterGPT: A Per-
sona Reconstruction Framework for Role-Playing Agents.
In Chen, W.; Yang, Y .; Kachuee, M.; and Fu, X.-Y ., eds.,
Proceedings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 3: Indus-
try Track) , 287–303. Albuquerque, New Mexico: Associ-
ation for Computational Linguistics. ISBN 979-8-89176-
194-0.
Qwen; :; Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang,
J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.;
Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue,
M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Tang, T.;
Xia, T.; Ren, X.; Ren, X.; Fan, Y .; Su, Y .; Zhang, Y .; Wan,
Y .; Liu, Y .; Cui, Z.; Zhang, Z.; and Qiu, Z. 2025. Qwen2.5
Technical Report. arXiv:2412.15115.
Shukla, N. K.; Prabhakar, P.; Thangaraj, S.; Singh, S.;
Sun, W.; Venkatesan, C. P.; and Krishnamurthy, V . 2025.
GraphRAG Analysis for Financial Narrative Summariza-
tion and A Framework for Optimizing Domain Adaptation.
In Chen, C.-C.; Moreno-Sandoval, A.; Huang, J.; Xie, Q.;
Ananiadou, S.; and Chen, H.-H., eds., Proceedings of the
Joint Workshop of the 9th Financial Technology and Natural
Language Processing (FinNLP), the 6th Financial Narrative
Processing (FNP), and the 1st Workshop on Large Language
Models for Finance and Legal (LLMFinLegal) , 23–34. Abu
Dhabi, UAE: Association for Computational Linguistics.

Team, G.; Kamath, A.; Ferret, J.; Pathak, S.; Vieillard, N.;
Merhej, R.; Perrin, S.; Matejovicova, T.; Ram ´e, A.; Rivi `ere,
M.; Rouillard, L.; Mesnard, T.; Cideron, G.; bastien Grill,
J.; Ramos, S.; Yvinec, E.; Casbon, M.; Pot, E.; Penchev, I.;
Liu, G.; Visin, F.; Kenealy, K.; Beyer, L.; Zhai, X.; Tsit-
sulin, A.; Busa-Fekete, R.; Feng, A.; Sachdeva, N.; Cole-
man, B.; Gao, Y .; Mustafa, B.; Barr, I.; Parisotto, E.; Tian,
D.; Eyal, M.; Cherry, C.; Peter, J.-T.; Sinopalnikov, D.; Bhu-
patiraju, S.; Agarwal, R.; Kazemi, M.; Malkin, D.; Kumar,
R.; Vilar, D.; Brusilovsky, I.; Luo, J.; Steiner, A.; Friesen,
A.; Sharma, A.; Sharma, A.; Gilady, A. M.; Goedeckemeyer,
A.; Saade, A.; Feng, A.; Kolesnikov, A.; Bendebury, A.;
Abdagic, A.; Vadi, A.; Gy ¨orgy, A.; Pinto, A. S.; Das, A.;
Bapna, A.; Miech, A.; Yang, A.; Paterson, A.; Shenoy, A.;
Chakrabarti, A.; Piot, B.; Wu, B.; Shahriari, B.; Petrini, B.;
Chen, C.; Lan, C. L.; Choquette-Choo, C. A.; Carey, C.;
Brick, C.; Deutsch, D.; Eisenbud, D.; Cattle, D.; Cheng,
D.; Paparas, D.; Sreepathihalli, D. S.; Reid, D.; Tran, D.;
Zelle, D.; Noland, E.; Huizenga, E.; Kharitonov, E.; Liu,
F.; Amirkhanyan, G.; Cameron, G.; Hashemi, H.; Klimczak-
Pluci ´nska, H.; Singh, H.; Mehta, H.; Lehri, H. T.; Hazimeh,
H.; Ballantyne, I.; Szpektor, I.; Nardini, I.; Pouget-Abadie,
J.; Chan, J.; Stanton, J.; Wieting, J.; Lai, J.; Orbay, J.; Fer-
nandez, J.; Newlan, J.; yeong Ji, J.; Singh, J.; Black, K.;
Yu, K.; Hui, K.; V odrahalli, K.; Greff, K.; Qiu, L.; Valen-
tine, M.; Coelho, M.; Ritter, M.; Hoffman, M.; Watson, M.;
Chaturvedi, M.; Moynihan, M.; Ma, M.; Babar, N.; Noy, N.;
Byrd, N.; Roy, N.; Momchev, N.; Chauhan, N.; Sachdeva,
N.; Bunyan, O.; Botarda, P.; Caron, P.; Rubenstein, P. K.;
Culliton, P.; Schmid, P.; Sessa, P. G.; Xu, P.; Stanczyk,
P.; Tafti, P.; Shivanna, R.; Wu, R.; Pan, R.; Rokni, R.;
Willoughby, R.; Vallu, R.; Mullins, R.; Jerome, S.; Smoot,
S.; Girgin, S.; Iqbal, S.; Reddy, S.; Sheth, S.; P ˜oder, S.; Bhat-
nagar, S.; Panyam, S. R.; Eiger, S.; Zhang, S.; Liu, T.; Ya-
covone, T.; Liechty, T.; Kalra, U.; Evci, U.; Misra, V .; Rose-
berry, V .; Feinberg, V .; Kolesnikov, V .; Han, W.; Kwon, W.;
Chen, X.; Chow, Y .; Zhu, Y .; Wei, Z.; Egyed, Z.; Cotruta,
V .; Giang, M.; Kirk, P.; Rao, A.; Black, K.; Babar, N.; Lo,
J.; Moreira, E.; Martins, L. G.; Sanseviero, O.; Gonzalez,
L.; Gleicher, Z.; Warkentin, T.; Mirrokni, V .; Senter, E.;
Collins, E.; Barral, J.; Ghahramani, Z.; Hadsell, R.; Matias,
Y .; Sculley, D.; Petrov, S.; Fiedel, N.; Shazeer, N.; Vinyals,
O.; Dean, J.; Hassabis, D.; Kavukcuoglu, K.; Farabet, C.;
Buchatskaya, E.; Alayrac, J.-B.; Anil, R.; Dmitry; Lepikhin;
Borgeaud, S.; Bachem, O.; Joulin, A.; Andreev, A.; Hardin,
C.; Dadashi, R.; and Hussenot, L. 2025. Gemma 3 Technical
Report. arXiv:2503.19786.
Wang, L.; Lian, J.; Huang, Y .; Dai, Y .; Li, H.; Chen, X.;
Xie, X.; and Wen, J.-R. 2025. CharacterBox: Evaluating the
Role-Playing Capabilities of LLMs in Text-Based Virtual
Worlds. In Chiruzzo, L.; Ritter, A.; and Wang, L., eds., Pro-
ceedings of the 2025 Conference of the Nations of the Amer-
icas Chapter of the Association for Computational Linguis-
tics: Human Language Technologies (Volume 1: Long Pa-
pers) , 6372–6391. Albuquerque, New Mexico: Association
for Computational Linguistics. ISBN 979-8-89176-189-6.
Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.;
and Wei, F. 2024a. Multilingual E5 Text Embeddings: ATechnical Report. arXiv:2402.05672.
Wang, X.; Xiao, Y .; Huang, J.-t.; Yuan, S.; Xu, R.; Guo, H.;
Tu, Q.; Fei, Y .; Leng, Z.; Wang, W.; Chen, J.; Li, C.; and
Xiao, Y . 2024b. InCharacter: Evaluating Personality Fidelity
in Role-Playing Agents through Psychological Interviews.
In Ku, L.-W.; Martins, A.; and Srikumar, V ., eds., Proceed-
ings of the 62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers) , 1840–1873.
Bangkok, Thailand: Association for Computational Linguis-
tics.
Wang, Y .; Leung, J.; and Shen, Z. 2025. RoleRAG: En-
hancing LLM Role-Playing via Graph Guided Retrieval.
arXiv:2505.18541.
Yan, S.-Q.; Gu, J.-C.; Zhu, Y .; and Ling, Z.-H. 2024. Correc-
tive Retrieval Augmented Generation. arXiv:2401.15884.
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.;
Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou,
F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.;
Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou,
J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.;
Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men,
R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren,
X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y .; Su, Y .; Zhang,
Y .; Zhang, Y .; Wan, Y .; Liu, Y .; Wang, Z.; Cui, Z.; Zhang,
Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 Technical Report.
arXiv:2505.09388.
Yang, X.; Sun, K.; Xin, H.; Sun, Y .; Bhalla, N.; Chen, X.;
Choudhary, S.; Gui, R. D.; Jiang, Z. W.; Jiang, Z.; Kong, L.;
Moran, B.; Wang, J.; Xu, Y . E.; Yan, A.; Yang, C.; Yuan, E.;
Zha, H.; Tang, N.; Chen, L.; Scheffer, N.; Liu, Y .; Shah, N.;
Wanga, R.; Kumar, A.; Yih, W.-t.; and Dong, X. L. 2024.
CRAG - Comprehensive RAG Benchmark. In Globerson,
A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tom-
czak, J.; and Zhang, C., eds., Advances in Neural Informa-
tion Processing Systems , volume 37, 10470–10490. Curran
Associates, Inc.
Zhang, S.; Dinan, E.; Urbanek, J.; Szlam, A.; Kiela, D.; and
Weston, J. 2018. Personalizing Dialogue Agents: I have a
dog, do you have pets too? In Proceedings of the 56th An-
nual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers) , 2204–2213.
Zhang, Y .; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J.
2025. Qwen3 Embedding: Advancing Text Embedding and
Reranking Through Foundation Models. arXiv:2506.05176.
Zhong, Z.; Liu, H.; Cui, X.; Zhang, X.; and Qin, Z. 2025.
Mix-of-Granularity: Optimize the Chunking Granularity for
Retrieval-Augmented Generation. In Proceedings of the
31st International Conference on Computational Linguis-
tics, 5756–5774.
Zhou, J.; Chen, Z.; Wan, D.; Wen, B.; Song, Y .; Yu, J.;
Huang, Y .; Ke, P.; Bi, G.; Peng, L.; Yang, J.; Xiao, X.;
Sabour, S.; Zhang, X.; Hou, W.; Zhang, Y .; Dong, Y .; Wang,
H.; Tang, J.; and Huang, M. 2024. CharacterGLM: Cus-
tomizing Social Characters with Large Language Models.
In Dernoncourt, F.; Preot ¸iuc-Pietro, D.; and Shimorina, A.,
eds., Proceedings of the 2024 Conference on Empirical

Methods in Natural Language Processing: Industry Track ,
1457–1476. Miami, Florida, US: Association for Computa-
tional Linguistics.
Zhu, K.; Luo, Y .; Xu, D.; Wang, R.; Yu, S.; Wang, S.; Yan,
Y .; Liu, Z.; Han, X.; Liu, Z.; et al. 2024. RAGEval: Sce-
nario Specific RAG Evaluation Dataset Generation Frame-
work. CoRR .