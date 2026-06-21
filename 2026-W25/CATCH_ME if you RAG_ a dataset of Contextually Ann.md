# CATCH-ME if you RAG: a dataset of Contextually Annotated multi-Turn Counterspeech against Hate and Misinformation Exchanges

**Authors**: Helena Bonaldi, Genoveffa Martone, Marco Guerini

**Published**: 2026-06-18 15:32:14

**PDF URL**: [https://arxiv.org/pdf/2606.20369v1](https://arxiv.org/pdf/2606.20369v1)

## Abstract
Online hate speech and misinformation frequently overlap, yet NLP research has mainly treated them in isolation. While LLMs represent a scalable solution for assisting humans in the generation of counterspeech for both threats, zero-shot models frequently generate repetitive and vague responses, underscoring the need for high-quality examples to steer model generation. However, existing counterspeech datasets against the overlap of hate and misinformation are scarce and limited to single-turn English dialogues, while real-life interactions span across multiple turns and languages. To bridge this gap, we introduce the first large-scale, expert-curated, multilingual dataset of dialogues tackling the intersection of hate and misinformation. To ensure factual grounding, the dialogues are also anchored in verified external knowledge (i.e., fact-checking articles and NGO reports) and include document- and chunk-level span annotations, making it directly applicable for RAG systems. Covering five languages and targeting hate directed at seven marginalized groups, this novel resource enables the training and evaluation of more persuasive, factually grounded counterspeech models.

## Full Text


<!-- PDF content starts -->

CATCH-ME if you RAG:
a dataset of Contextually Annotated multi-Turn Counterspeech
against Hate and Misinformation Exchanges
Helena Bonaldi1Genoveffa Martone1,2Marco Guerini1
1Fondazione Bruno Kessler, Italy,2Università Cattolica del Sacro Cuore, Italy,
{hbonaldi, gmartone, guerini}@fbk.eu
Abstract
Online hate speech and misinformation fre-
quently overlap, yet NLP research has mainly
treated them in isolation. While LLMs repre-
sent a scalable solution for assisting humans
in the generation of counterspeech for both
threats, zero-shot models frequently generate
repetitive and vague responses, underscoring
the need for high-quality examples to steer
model generation. However, existing counter-
speech datasets against the overlap of hate and
misinformation are scarce and limited to single-
turn English dialogues, while real-life interac-
tions span across multiple turns and languages.
To bridge this gap, we introduce the first large-
scale, expert-curated, multilingual dataset of
dialogues tackling the intersection of hate and
misinformation. To ensure factual grounding,
the dialogues are also anchored in verified ex-
ternal knowledge (i.e., fact-checking articles
and NGO reports) and include document- and
chunk-level span annotations, making it di-
rectly applicable for RAG systems. Covering
five languages and targeting hate directed at
seven marginalized groups, this novel resource
enables the training and evaluation of more
persuasive, factually grounded counterspeech
models.
Warning: This work contains unobfuscated ex-
amples that some readers may find offensive.
1 Introduction
While online hate and misinformation are tradi-
tionally treated as separate NLP problems (Chung
et al., 2021b; Zeng and Gao, 2024), they frequently
interact in real-world settings (Cazzamatta, 2025).
Hate speech co-occurs more with misinformation
than with factual content (Hameleers et al., 2022),
and misinformation often exacerbates hate speech,
amplifying its reach and potential for harm (War-
dle, 2024). In this context, counterspeech (CS) has
emerged as an alternative to combat these phenom-
ena without resorting to censorship (Myers West,
Fact-checking article: [...] The video is miscaptioned. It does not show the
Lord Mayor of Liverpool, Mary Rasmussen. [...] Check Your Fact could not
verify the man’s identity via reverse image search. It is clear he is not the
current Lord Mayor of Liverpool, however. The current Lord Mayor is Mary
Rasmussen, according to Liverpool Town Hall’s website. Rasmussen’s
appearance differs greatly from the man in the video as she is a woman
with blonde, shoulder-length hair. [...]
NGO report: [...] These people, and their interpretation of Islam, is rightly
called “extremist;” they are a minority within Islam and the vast majority of
Muslims reject their violence and consider their interpretation a distortion
of the Muslim faith. [...] In the Quran, Muslims are reminded that they are
not the only ones who worship God. The Holy Book of Islam sees
Christians and Jews as members of the family of Abraham (one of the
prophets) and Muslims respect these religion s because they believe that
they all worship the same God. [...]  
Muslims are taking over! Did you see that video of the Lord Mayor
of Liverpool reciting Shahadah? That's absurd.
Let's not spread alarmism. I also saw the video you're referring
to, but it does not feature the Mayor of Liverp ool, since it shows
a man, and the current Mayor is Mary Rasmussen, a woman.
This case should teach you how easily false information can
spread and how important it is to check for reliable facts.
Yeah, even if that video was wrong, it doesn't change the fact that
Muslims are trying to force their violent religion on us.  
That's not true either. The narratives about Muslims wanting to
convert everyone to Islam just stem from stereotypes. In reality,
the Quran reminds believers that they worship the same God as
Christians and Jews, and promotes respect towards them, not
conversion.
So you're saying that Muslims aren't all terrorists? You're delusional.  
It's not delusion, it's the truth. The vast majority of Muslims are
peaceful individuals who actively condemn extremism and
violence, emphasizing that radical interpretations of Islam do
not represent their beliefs or values. It's not fair to label an
entire community as terrorists based on the actions of a few.  
EXTERNAL KNOWLEDGE
DIALOGUEFigure 1: An example of a collected dialogue, where
each CS is supported by the external knowledge.
2018). While CS against hate uses empathy and
cogent, fact-based arguments (Schieb and Preuss,
2016), CS against misinformation adopts a non-
partisan, non-emotive style (Guo et al., 2022). Con-
versely, CS tackling their intersection must jointly
correct misinformation with facts and use empa-
thy to challenge negative stereotypes (Martone
et al., 2026). Although NGOs and fact-checkers
manually produce CS (Chung et al., 2019; Winter-
sieck, 2017), this process requires high cognitive
effort and psychological resilience, creating a bot-
tleneck for moderators (Mun et al., 2024). Con-
sequently, interest has surged in leveraging LargearXiv:2606.20369v1  [cs.CL]  18 Jun 2026

Language Models (LLMs) to assist them in writing
CS (Bonaldi et al., 2024a). However, state-of-the-
art LLMs in zero-shot settings often produce repeti-
tive, vague responses that denounce rather than con-
structively engage with harmful content (Mun et al.,
2023). This underscores the need for high-quality
examples to steer generation. Yet, existing CS cor-
pora isolate these issues, focusing either on toxic
content or objective veracity (Bonaldi et al., 2024a;
Guo et al., 2022). The only dataset addressing
both is restricted to single-turn English dialogues
(Martone et al., 2026). To bridge this gap, we
collaborate with 23 experts in CS writing and em-
ploy four human-machine collaboration strategies
to createCATCH-ME: a dataset ofContextually
Annotated multi-TurnCounterspeech againstHate
andMisinformationExchanges. CATCH-ME in-
cludes fictitious multi-turn interactions between a
person spreading hate and misinformation, and a
counterspeaker addressing both. Responses are
grounded in external verified knowledge, includ-
ing fact-checking articles and NGO reports anno-
tated at both chunk and document level, making
the dataset well suited for Retrieval-Augmented
Generation (RAG) research. An example of a col-
lected dialogue with annotated external knowledge
is shown in Figure 1. The dataset covers five lan-
guages (English, Italian, Maltese, Polish, and Span-
ish) and targets hostility toward seven marginalized
groups: Muslims, Jewish people, people of color,
women, LGBTQIA+ individuals, migrants, and
people with disabilities. We describe our collection
and annotation methodologies across the four col-
laboration strategies, and use our dataset to provide
a first benchmark on retrieval and generation tasks
for fact-based CS. The final dataset contains 2015
dialogues and 12,298 turns in total, 6,149 of which
are CS turns grounded in external knowledge.1
2 Related Work
CS data collectionThe techniques employed to
collect CS resources includecrawlingthem from
online platforms (Mathew et al., 2019) or fact-
checking websites (Russo et al., 2023b);crowd-
sourcingthem from non-expert annotators (Fur-
man et al., 2023; He et al., 2023) ornichesourcing
them from domain experts, such as NGO opera-
tors (Chung et al., 2021a) and professional fact-
checkers (Russo et al., 2025b). CS can also be
1The dataset is available at https://github.com/
LanD-FBK/counterspeech_against_hate_and_misinfo.obtained withfully automatedmethods (Vallecillo-
Rodríguez et al., 2023; Stammbach and Ash, 2020):
their use, however, remains limited due to concerns
about the factual accuracy and potential harmful-
ness of their outputs. Finally,human-in-the-loop
(HITL) approaches combine automated generation
with human post-editing to balance scalability and
quality (Tekiro ˘glu et al., 2020; Fanton et al., 2021;
Russo et al., 2023a). In this context, Martone et al.
(2026) introduces the only HITL dataset address-
ing the intersection of hate and misinformation,
which is limited to single-turn interactions. The
only available conversational dataset is DIALOCO-
NAN (Bonaldi et al., 2022), which focuses only on
hate speech, while similar multi-turn corpora for
countering misinformation, either in isolation or
intersecting with hate, are not available.
CS generation against hate and misinformation
For what regards CS generation against hate, re-
search has shown that language models often pro-
duce generic and repetitive responses, which often
fail to address hateful claims effectively (Tekiro ˘glu
et al., 2020; Mun et al., 2023). To overcome these
limitations, existing studies have focused on im-
proving specific aspects of the generated responses,
such as their personalization (de los Riscos and
D’Haro, 2021; Do ˘ganç and Markov, 2023), argu-
mentative quality (Furman et al., 2023; Bonaldi
et al., 2024b), and factual grounding (Russo, 2025).
In the misinformation domain, CS studies have
mainly focused on generating readable, plausible,
and faithful responses, evaluated with the involve-
ment of human experts (Guo et al., 2022; Russo
et al., 2023a; He et al., 2023). While earlier ap-
proaches relied on attention (Popat et al., 2018)
and rule-based methods (Gad-Elrab et al., 2019),
more recent systems adopt summarization (Russo
et al., 2023b), prompting (Russo et al., 2025b), and
RAG techniques (Russo et al., 2025a).
3 Data collection methodology
We collaborated with 23 CS experts (11 fact-
checkers and 12 NGO operators) over a period
of 18 months2to collect fictitious multi-turn dia-
logues addressing the co-occurrence of hate and
misinformation through a human-machine collabo-
ration setup. In the following sections, we describe
the collection process for the external knowledge
2The entire data collection process included also knowl-
edge search, selection, annotation and translation.

and the dialogues, and the human-machine collab-
oration strategies used to create them.
3.1 External knowledge
Each dialogue is grounded in one or two exter-
nal knowledge documents, always including at
least one fact-checking article. Below we de-
scribe the collection of fact-checking articles and
NGO reports, and the matching process for two-
document dialogues. Our data collection is fo-
cused on sources that refer to one of the following
marginalized groups: Muslims, the LGBTQIA+
community, migrants, women, people with disabil-
ities, people of color and Jewish people.
Fact-checking articlesWe use the English fact-
checking articles from Martone et al. (2026), and
collect new articles by translating their proposed
set of keywords into Polish, Spanish, and Italian
using deepl .3Then, the translated keywords were
reviewed by the experts, who made additions, dele-
tions and substitutions in their respective languages
to ensure linguistic coverage of the target minori-
ties and contextual accuracy of the provided terms,
resulting in a list of more than 221 keywords span-
ning the 5 languages (see Table 11 in Appendix A.1
for the full list). Finally, the validated keywords
were used to scrape Google Fact Check Tools Ex-
plorer4with newspaper4k5. The resulting 2,682
articles underwent a manual filtering aimed at keep-
ing only content (i) written by signatories of the
International Fact-Checking Network code of prin-
ciples6, and (ii) that could be used to fuel discrimi-
nation against the target groups (see the selection
criteria in Appendix A.1). Because no dedicated
fact-checking sources exist for Maltese, English ar-
ticles were used as external knowledge to generate
Maltese CS, a decision justified by the bilingualism
in Malta. Overall, we collected 516 fact-checking
articles across five languages.
NGO reportsWe asked the NGO operators to
help us enrich the set of myth–anti-stereotype pairs
created by Martone et al. (2026), by manually
searching for additional NGO reports in Polish,
Spanish, and Italian (see Appendix A.1 for the list
of domains used). Because the number of non-
English reports was limited, we further expanded
3https://pypi.org/project/deepl/
4https://toolbox.google.com/factcheck/
explorer/search/list:recent;hl=
5https://pypi.org/project/newspaper4k/
6https://ifcncodeofprinciples.poynter.org/the dataset by translating and manually checking
from the English corpus specific pairs that were
untied to a defined country/regional setting. Also
in this case, English documents were used as exter-
nal knowledge for Maltese dialogues. Overall, we
obtained 345 stereotype and anti-stereotype pairs.
External knowledge matchingAs a final step,
we constructed NGO-based external knowledge for
the two-document dialogue setting. Unlike fact-
checking sources, which consist of coherent ar-
ticles, NGO materials are structured as isolated
myth–anti-stereotype pairs. To align these for-
mats, we computed the semantic similarity7be-
tween each fact-checking claim and all stereotypes
targeting the same group. The resulting matches
were manually reviewed, and only meaningful as-
sociations were retained. For each fact-checking
article, the validated myth–anti-stereotype pairs
were then merged into a single document (we refer
to it asNGO report). This process yielded 313
aligned fact-checking and NGO report pairs.
3.2 Dialogue collection
The collection was performed on the First-AID an-
notation platform (Menini et al., 2025), designed
for annotating knowledge-based dialogues. An-
notators were tasked to collect multi-turn ficti-
tious dialogues simulating a person spreading hate
and misinformation and an operator providing CS
grounded on the provided document(s) to counter
it. Below, we describe (i) the expert annotators,
(ii) the guidelines employed to collect the dialogue
turns and (iii) the external knowledge annotations.
Expert reviewersThe annotation team consisted
of 23 CS experts (11 fact-checkers and 12 NGO
operators) from Poland, Italy, Malta, and Spain.
To protect annotator well-being, we adapted the
guidelines of Vidgen and Derczynski (2020): the
annotation process was distributed over approxi-
mately 18 months to avoid excessive workload, and
biweekly meetings were held to discuss concerns
or difficulties. No major issues emerged, likely due
to the annotators’ extensive prior experience with
harmful content.
Guidelines: Dialogue turnsAnnotators were
instructed to ensure naturalness and avoid repeti-
tion, producing 4–8 turn dialogues starting with a
hate-and-misinformation message and ending with
a CS response. CS writing followed Martone et al.
7all-mpnet-base-v2(Sentence-Transformers, 2021)

Docs MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
14.958 4.923-0.0354.106 4.045-0.061 2.004 2.139 0.135 0.306 0.292-0.014
2 4.601 4.7630.1623.774 3.8690.095 2.071 2.209 0.138 0.346 0.319-0.027
Table 1: Syntactic metrics results according to number of documents used as reference by the dialogues.
(2026) guidelines: (i) avoid abusive language and
focus on the message rather than the author, (ii)
maintain a respectful and empathetic tone, (iii)
counter claims using verified facts and statistics
with source grounding, and (iv) provide context
while discouraging overgeneralization.
Guidelines: External knowledgeAnnotators
were provided with external knowledge to sup-
port drafting CS: all dialogues always included
a fact-checking article, plus an NGO report for the
two-document condition. In the latter setting, an-
notators had to reference both documents at least
once across the conversation. They also linked the
specific document spans supporting each turn. This
ground textwas required to contain all information
relevant to the response, and was omitted for turns
that didn’t require grounding, (e.g. clarification
questions "Where did you get this information?").
Docs HTER Time Ground RR or RRed RR∆
10.361 151.532.06.411 4.689-1.722
2 0.368 166.445.28.962 6.095-2.867
Table 2: Results of annotation effort and RR metrics
according to number of documents per dialogue.
3.3 Human-machine collaboration strategies
We employ the three annotation strategies pre-
sented by Menini et al. (2025, pre-compiled, in-
teractive, and manual) to reduce the burden on an-
notators. On top of these, we also automatically
translate post-edited English dialogues to obtain a
multilingual parallel portion of the corpus. More
details on the models employed to assist human
annotations are in Appendix A.2.
Pre-compiledDialogues are first generated using
GPT-4o mini (OpenAI, 2024), and the ground text
is automatically retrieved by chunking the source
documents via SaT (Frohmann et al., 2024) and
selecting the highest-similarity chunk with BM25
(Robertson et al., 1995). Annotators then review
the dialogue and retrieved spans, making necessary
adjustments according to the guidelines in §3.2.InteractiveWith this strategy, a model dynam-
ically generates multiple turn’s alternatives based
on the dialogue history. Annotators then select the
best option, modify it if necessary, or write a new
response from scratch. Preliminary experiments
showed that zero-shot GPT-4o mini was unsuitable,
as its safety guardrails frequently triggered refusals
when generating hater turns sequentially, while this
issue did not occur when generating full dialogues
at once in the pre-compiled strategy. To bypass
this, we fine-tuned Llama 3.1 8B (Grattafiori et al.,
2024) using data collected with the other strate-
gies. For grounding retrieval, source documents
were chunked with LlamaIndex8, and we experi-
mented with both a zero-shot BGE-M3 retriever
(Chen et al., 2024a) and a fine-tuned BGE-v2-M3
reranker (Li et al., 2023; Chen et al., 2024b).
ManualThe dialogue is entirely written by the
annotator, who also selects the portion of the doc-
ument(s) to be used as ground text for CS turns.
While this strategy is the most demanding, it serves
as a comparison to evaluate the quality of the dia-
logues obtained with the other approaches.
TranslationPart of the post-edited English di-
alogues is automatically translated using Seam-
lessM4T Large (Barrault et al., 2023). Annotators
then review these translations for correctness and
fluency without altering the core content. To reduce
their workload, the source documents and ground-
ing text remain untouched in English. Additionally,
annotators are encouraged to provide context for el-
ements unfamiliar to their national setting, such as
foreign institutions or events (e.g., the UK’s “PIP”
disability benefit).
4 Annotation and Data description
We first outline our evaluation metrics, followed by
annotation process statistics calculated at dialogue-
level according to the number of referenced docu-
ments, dialogue language, and annotation strategy.
To prevent bias from unequal subset sizes, all re-
sults are reported as macro-averages, calculated
8https://www.llamaindex.ai/

first within each language-strategy-number of doc-
uments combination and then averaged across the
dimensions of interest. Metrics are divided into two
main categories: annotation effort and syntactic
metrics.9We report monolingual data first, where
both dialogue and article are in the same language,
to consistently measure post-editing effort across
the manual, interactive, and pre-compiled strate-
gies. Translation-based data is analyzed separately
because translation is applied after human post-
editing, introducing an additional transformation
that modifies the dialogue surface without chang-
ing its content. This affects both metrics that mea-
sure annotation effort, and all metrics calculated
on the “original” version of the dialogue, since
they depend on the specific post-edited dialogue
that was picked to be translated. Therefore, transla-
tion results are reported independently and are not
aggregated with monolingual conditions.
4.1 Evaluation metrics
Human-targeted Translation Edit Rate (HTER)
measures post-editing effort in terms of “insertion,
deletion, and substitution of single words as well
as shifts of word sequences” (Snover et al., 2006),
where higher values equate to greater interventions.
The 0.4 threshold is used to identify heavily post-
edited dialogues (Turchi et al., 2013).
Annotation timeis automatically computed by the
First-AID platform in terms of seconds needed to
edit the entire dialogue: we normalize it by the
number of turns of each dialogue.
Repetition Rate (RR)quantifies the lexical diver-
sity of a text in terms of non-singleton n-grams
(Bertoldi et al., 2013). We used a sliding window
of 1000 terms over 5 random corpus shuffles.
Syntactic complexity: we use the spacy syntactic
dependency parser to compute three measures of
turn-level syntactic complexity: (i) theMaximum
Syntactic Depth(MSD) and (ii)Average Syntactic
Depth(ASD) of each sentence’s dependency tree,
and (iii) theNumber of Sentences(NST).10
Ground length: for each dialogue, we concate-
nate all the annotated ground text and compute the
average words-level length.
9ordenotes metrics computed on the original dialogue
before post-editing, and edon the post-edited dialogue. Bold
shows the best value, underlined the second best: highest for
ground and syntactic metrics, lowest for all others.
10https://spacy.io/usage/linguistic-features#
dependency-parse . These metrics are not computed for
Maltese, as no model is available for this language.Lang. HTER Time Ground RR or RRed RR∆
EN 0.508 286.118 21.280 6.156 3.418 -2.738
ES0.110 96.35250.7649.622 8.269 -1.353
IT 0.20567.57838.938 9.243 6.980 -2.263
MT 0.507 139.948 25.5005.2153.699 -1.516
PL 0.562 221.050 45.106 6.7923.355 -3.437
Table 3: Results of annotation effort and RR metrics
according to dialogue language.
4.2 Monolingual data
Number of documentsTable 2 shows that single-
document dialogues require less annotation effort,
yielding lower HTER and shorter average anno-
tation times. While their RR is also lower, post-
editing successfully reduces repetitiveness across
both configurations. As expected, single-document
dialogues have shorter average grounding lengths
due to more limited reference knowledge. Regard-
ing syntactic metrics (Table 1), single-document
dialogues exhibit higher MSD and ASD, despite
containing fewer sentences on average, alongside a
lower proportion of complex words.
LanguageTables 3 and 4 present the annotation
effort and syntactic metrics by language. Spanish
and Italian require the least effort, yielding the low-
est average annotation times and the only HTER
below 0.4. The higher HTER in other languages
stems from distinct factors. On the one hand Polish
and Maltese are low-resourced languages, which
could explain worse quality generations used as
a starting point. Meanwhile, English and Polish
show significantly higher annotation times (>200
seconds per turn) paired with the largest reductions
in RR, suggesting significant effort to fix repetitive
generations. Syntactic metrics offer further insight
for Polish: it consistently registers the highest or
second-highest increase in syntactic complexity
with post-editing, showing that Polish annotators
frequently split and articulated sentences with re-
spect to the original generation.
Annotation strategyTables 5 and 6 break down
annotation effort and syntactic metrics by strategy.
As shown in Table 5, the pre-compiled strategy
requires the least effort, yielding the lowest anno-
tation times and the only HTER below 0.4, while
producing the longest ground text. Conversely, the
manual strategy demands the most time but results
in the lowest RR ed. This suggests that while syn-
thetic baselines drastically reduce annotation time,
they partially bias annotators toward specific word
choices. Consequently, machine-assisted strategies

Lang. MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
EN6.156 5.548-0.6085.010 4.443-0.5672.248 2.218 -0.030 0.178 0.140 -0.038
ES 4.531 4.592 0.061 3.769 3.8640.0951.982 1.966 -0.016 0.311 0.303 -0.008
IT 4.532 4.629 0.097 3.816 3.866 0.050 1.969 2.063 0.094 0.362 0.357 -0.005
MT - - - - - - - - - 0.292 0.270 -0.022
PL 4.676 4.8220.1463.781 3.839 0.058 2.0382.423 0.385 0.403 0.406 0.003
Table 4: Syntactic metrics results according to dialogue language.
Strat. HTER Time Ground RR or RRed RR∆
Interactive 0.419 165.286 31.433 10.296 7.917-2.379
Manual - 235.52 28.068 -2.536-
Pre-compiled0.309 96.491 48.372 4.793 4.302 -0.491
Table 5: Results for annotation effort metrics according
to the annotation strategy employed.
increase dialogue repetitiveness and fail to match
the lexical diversity of text written by humans en-
tirely from scratch. The interactive strategy ex-
hibits the highest RR ed, yet annotators invest the
most effort here into reducing repetition, as shown
by the high RR ∆. The RR orindicates that the
model generates more varied content when tasked
with producing the entire dialogue at once rather
than interactively. This occurs in the pre-compiled
strategy, where the model can plan its generation in
advance, rather than generating turn-by-turn, where
it exhibits a greedier behavior and falls more easily
into repetitive phrasing. Regarding syntactic met-
rics (Table 6), manual dialogues have the highest
MSD and ASD alongside the lowest proportion
of complex words, indicating a complex sentence
structure paired with common vocabulary. In con-
trast, the pre-compiled strategy, across both gener-
ated and post-edited versions, features the lowest
MSD and ASD but the highest NST, reflecting a
simpler syntax where content is split across more
sentences. Results grouped by number of docu-
ments and strategy are coherent with those shown
by strategy (Tables 15 and 16 in Appendix A.4).
4.3 Translated data
Tables 7 and 8 report the metrics for translated
dialogues.11Across all languages, HTER ex-
ceeds 1.0, indicating low-quality automatic transla-
tions that required substantial post-editing. Span-
ish and Italian demanded the least effort (lowest
HTER and annotation time), yet their RR increased
post-annotation. Their syntactic metrics remained
largely stable (deltas near 0.0), suggesting the pro-
cess did not alter the dialogues’ core structure. This
11Ground length is excluded as it was not modified during
annotation.indicates that because the initial Spanish and Italian
translations were relatively fluent, annotators made
shallow, local edits rather than deep restructurings.
This minimal patching preserved or amplified the
repetitive phrasing already present in the machine
translation. Conversely, for lower-resourced lan-
guages like Polish and Maltese, annotators were
forced to reformulate text aggressively. This deeper
intervention resulted in higher HTER, longer an-
notation times, larger syntactic deltas (MSD, ASD,
and NST for Polish), and a corresponding reduction
in repetition. These patterns are consistent when
grouping results by the number of documents and
language (see Tables 17 and 18 in Appendix A.5).
4.4 Final dataset description
The final dataset contains 2,015 dialogues (1,565
single-document, 450 two-document) and 12,298
turns. The distribution across languages is uni-
form, with each accounting for 14–16% of single-
document and 2.5–5% of two-document dialogues.
Conversely, the distribution of targeted groups is
skewed due to the varying real-world availabil-
ity of fact-checking articles addressing specific
marginalised groups. Dialogues predominantly
focus on Migrants (30%), Muslims (19%), and
Women (17%); despite targeted scraping efforts, di-
alogues concerning Jewish people and individuals
with disabilities are a minority (2–7%): rather than
artificially balancing the corpus with suboptimal
examples, we prioritize cases for which experts
could rely on verifiable evidence. The distribution
of dialogues according to target and language is
available in Tables 12 13, and 14 in Appendix A.3.
5 Experiments
In the following experiments, we test how our
dataset can be used to evaluate retrieval and gen-
eration in fact-based multilingual CS generation
settings when faced with overlapping hate and mis-
information. To achieve this, we conduct two main
experiments: a retrieval and a generation task.

Strat. MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
Interactive5.001 4.793 -0.2084.165 4.015 -0.150 1.986 2.007 0.0210.339 0.329 -0.010
Manual -5.148- -4.174- - 2.237 - - 0.251 -
Pre-compiled 4.609 4.7310.122 3.762 3.8060.044 2.079 2.271 0.192 0.309 0.308-0.001
Table 6: Results for syntactic metrics according to the annotation strategy employed.
Lang. HTER Time RR or RRed RR∆
ES1.014 85.350 3.141 4.210 1.069
IT 1.03930.7952.704 3.460 0.756
MT 1.422 179.9952.156 2.148 -0.008
PL 1.322 133.775 2.1721.974 -0.198
Table 7: Annotation effort results for translated dia-
logues according to the language.
5.1 Retrieval
We evaluate three embedders on a zero-shot chunk
retrieval task for hate and misinformation counter-
ing, across three query configurations and two sub-
tasks: monolingual and cross-lingual retrieval.12
RetrieverWe compare BM25 as a sparse base-
line against two dense retrievers representing
distinct structural paradigms: BGE-M3, a dedi-
cated encoder-only model engineered for multi-
functional retrieval, and Qwen3-Embedding-4B
(Zhang et al., 2025), a modern LLM-based em-
bedder. Despite both models offer multilingual
support, Qwen3-Embedding is the only offering
native support for the Maltese language.
QueryWe test three different query configura-
tions: (i)Original query( Q): each hate speech
turn is used as is to retrieve the relevant document
chunks needed to formulate the CS; (ii)Dialogue
context query( QDC): the query is formed by con-
catenating the original query with all preceding
conversation turns (Wu et al., 2022); (iii)Rewrit-
ten query( QR): for all hate speech turns that are
not the first, the query is rewritten by condition-
ing it on the preceding dialogue context using the
prompt template from Ye et al. (2023).
TaskRetrieval performance is measured in two
settings: (i)monolingual: queries and reference
documents share the same language; (ii)cross-
lingual: the reference document is in English,
while queries are in a different language. In both
configurations, the search space contains all chunks
from all source documents.
12More details on the preprocessing, prompts, and hyperpa-
rameters are provided in Appendix A.6.ResultsTable 9 presents the zero-shot monolin-
gual retrieval results13: cross-lingual results are
consistent and reported in Table 19 (Appendix
A.6). Two distinct trends emerge. First, dense
models substantially outperform the lexical base-
line: Qwen3-Embedding consistently achieves the
highest performance, followed by BGE-M3 and
BM25, across all settings. Second, query formu-
lation heavily impacts success. Across all models,
providing the full dialogue context ( QDC) yields
the best performance, followed closely by the LLM-
rewritten query ( QR), while the standalone query
(Q) performs worst. This underscores the neces-
sity of conversational context to resolve implicit
or vague references typical in dialogue-based hate
speech and misinformation. A key divergence oc-
curs in the cross-lingual scenario: while Qwen3
maintains its preference for QDC, both BM25 and
BGE-M3 perform better with QR. This variation
suggests that explicit query rewriting may filter out
conversational noise for BM25 and BGE-M3 dur-
ing cross-lingual transfer, whereas larger models
like Qwen3 can align dense, multilingual represen-
tations directly from raw conversational context.
5.2 Generation
We employ Qwen3 8B (Team, 2025) to test three
zero-shot configurations for CS generation, with
different input: (i) CS base: only the harmful state-
ment containing hate and misinformation (and dia-
logue history, if present), (ii) CS gold: CS baseplus
the gold knowledge, and (iii) CS retr: CS baseplus
the top 5 chunks retrieved by Qwen3-Embedding,
i.e. the best retriever emerged from §5.1 (See Ap-
pendix A.7 for more details).
Evaluation metricsWe evaluate the generated
CS using four metrics:BERTScoreto capture
semantic similarity with the human-curated CS
(Zhang et al., 2019)14;NLI Entailmentto as-
sesse factual alignment by computing the probabil-
ity that the output is entailed by the source text
13Bold text indicates the best performance within each
model group. * denotes the overall best for each metric.
14We report F1scores using xlm-roberta-large (Con-
neau et al., 2020)

Lang. MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
ES 4.766 4.82 0.054 3.9743.948-0.026 2.044 2.112 0.068 0.309 0.308 -0.001
IT4.847 4.828 -0.019 3.913 3.836 -0.0772.172 2.2240.052 0.361 0.360 -0.001
MT - - - - - - - - - 0.252 0.254 0.002
PL 4.754.866 0.116 4.084 3.928-0.1561.874 2.2180.344 0.422 0.417 -0.005
Table 8: Syntactic metrics results for translated dialogues according to the language
Model Query Hit@10 MAP@10 Recall@10
BM25Q 0.279 0.135 0.235
QDC 0.452 0.218 0.397
QR 0.425 0.209 0.369
BGE-M3Q 0.384 0.192 0.341
QDC 0.5500.2720.500
QR 0.5400.2730.491
Qwen3Q 0.464 0.238 0.418
QDC *0.595 *0.306 *0.548
QR 0.579 0.305 0.531
Table 9: Monolingual zero-shot chunk retrieval perfor-
mance across varying query configurations.
Metric Setting Value
BERTScoreCSbase 0.884
CSgold 0.895
CSretr 0.888
Faithfulness goldCSbase 2.702
CSgold 4.116
CSretr 3.138
Faithfulness retr CSretr 4.496
NLI Entailment goldCSbase 0.033
CSgold 0.246
CSretr 0.086
NLI Entailment retr CSretr 0.228
RelevanceCSbase 4.711
CSgold 4.538
CSretr 4.514
Table 10: CS generation quality across metrics.
viaxlm-roberta-large-xnli (Davison, 2020);
Faithfulnessto evaluate adherence to the context
without hallucinations;Relevanceto measure how
appropriately the CS addresses the previous turn.
We use GPT-4.1 mini (OpenAI, 2025) to score
Faithfulness and Relevance on a 1–5 Likert scale,
adapting the definitions from Es et al. (2024). For
CSretr, NLI and Faithfulness are computed against
both gold and retrieved knowledge.
ResultsAs shown in Table 10, CSgoldserves as
the optimal generation configuration across all met-
rics except relevance, followed closely by CSretr,
while the ungrounded baseline (CS base) consis-
tently performs worst. CS baseyields the lowest se-mantic similarity to human-curated CS. Providing
retrieved chunks elevates the BERTScore to 0.888 ,
closely matching CSgold(0.895 ) and demonstrating
that automated retrieval successfully improves the
generation’s alignment to expert-curated CS. For
NLI Entailment and Faithfulness, CS baseexhibits
a substantial misalignment with the verified facts,
yielding the lowest scores ( 0.033 and2.702 ). On
the other hand, as regards CS retr, when it is evalu-
ated against the retrieved context, its performance
approaches or even surpasses CS goldon NLI Entail-
ment and Faithfulness, respectively. However, eval-
uating those same generations against the hidden
gold knowledge causes a performance drop. While
this remains a major improvement over CS base,
it underscores that generator quality relies heav-
ily on retriever accuracy. Finally, CS baseachieves
the highest relevance score ( 4.711 ), outperforming
CSretr(4.514 ) and CSgold(4.538 ). This result high-
lights an inherent trade-off in evidence-grounded
generation: forcing a model to integrate external
evidence slightly restrains its ability to adhere to
the user query. However, this is an acceptable trade-
off when the objective is to counter hate speech and
misinformation with verifiable facts. The same dy-
namics and trends are observed in the cross-lingual
configuration (see Table 22 in Appendix A.7).
6 Conclusion
While prior NLP resources treat hate and misinfor-
mation in isolation, we introduce CATCH-ME: the
first multilingual, multi-turn counterspeech dataset
designed to tackle intertwined toxic narratives and
false claims simultaneously. Developed with 23
domain experts across four human-machine collab-
oration strategies, our corpus provides high-quality
refutations in five languages, fully grounded in fact-
checking articles and NGO reports. Beyond intro-
ducing this resource, we analyze the human-in-the-
loop annotation process and establish robust strong
retrieval and generation benchmarks baselines in
RAG settings. Ultimately, this dataset provides a
rigorous framework for training fact-based counter-
speech models to foster safer online discourse.

Acknowledgements
This work was partially supported by the European
Union’s CERV fund under grant agreement No.
101143249 (HATEDEMICS). We are grateful to
the following NGOs, fact-checking organizations
and all annotators for their help: ALDA (Asso-
ciation Europeenne Pour La Democratie Locale),
FUNDEA (Fundacion Euroarabe De Altos Estu-
dios), MALDITA (Fundacion Maldita.Es Contra
la Desinformacion: Periodismo educacion Investi-
gacion y Datos En Nuevos Formatos), CENTRA
(Fundacion Pública Andaluza Centro De Estudios
Andaluces M.P.), CESIE ETS (CESIE ETS), TFCF
(The Fact-Checking Factory S.r.l.), SOS MALTA
(Solidarity And Overseas Service Malta), VSA
(Victim Support Agency), CEO (Fundacja Cen-
trum Edukacji Obywatelskiej), DEMAGOG (Sto-
warzyszenie Demagog), NASK (Naukowa I Aka-
demicka Siec Komputerowa - Panstwowy Instytut
Badawczy).
Limitations
Our dataset is multilingual by design, but not in-
tended to exhaustively cover the linguistic diversity
of online hate and misinformation. We include
both higher-resource languages, i.e., English, Ital-
ian, and Spanish, and lower-resource languages,
i.e., Polish and Maltese, as a feasible compromise
between breadth, expert availability, and the need
for high-quality grounding and post-editing. Future
work can build on this resource by extending the
same methodology to additional languages, espe-
cially underrepresented linguistic communities.
Similarly, the dataset covers multiple targets of
hate, but their distribution is not uniform. This
reflects the availability of reliable external knowl-
edge: some groups are more frequently represented
in fact-checking articles and anti-stereotype re-
sources than others. Rather than artificially bal-
ancing the corpus with suboptimal examples, we
prioritize cases for which experts could rely on ver-
ifiable evidence. Future expansions can address
this imbalance through targeted collection efforts
and collaborations with organizations specializing
in less represented communities.
The hateful and misinformed turns in our dia-
logues are fictitious. This choice allows annota-
tors to focus on the main objective of the resource:
writing grounded counterspeech that jointly ad-
dresses false claims and harmful stereotypes. Con-
sequently, the dataset should not be interpreted asa comprehensive model of how hate speech and
misinformation appear in naturally occurring on-
line interactions, where they may be more implicit,
ambiguous, or context-dependent. Future research
can use our corpus as a controlled starting point
and evaluate transfer to naturally occurring conver-
sations.
Finally, our retrieval and generation experiments
are intended as initial benchmarks rather than an
exhaustive evaluation of all possible modeling
choices. Since the main contribution of this work is
the collection of a large expert-curated, knowledge-
grounded dialogue dataset, we test a limited set of
retrieval and generation models and rely on auto-
matic evaluation metrics. These experiments es-
tablish a first reference point for the task enabled
by our dataset. Future work can extend this bench-
mark with additional model families, fine-tuning
strategies, and human evaluation of factuality, per-
suasiveness, safety, and conversational appropriate-
ness.
Ethical consideration
This work addresses harmful online content and
therefore requires explicit safeguards for both the
people involved in data creation and the potential
downstream use of the resource.
Annotator well-being.The dataset was cre-
ated with domain experts who had prior experi-
ence with counterspeech, fact-checking, and anti-
discrimination work. To reduce the burden of re-
peated exposure to offensive and misleading con-
tent, the annotation campaign was distributed over
an extended period and included regular meetings
where annotators could discuss difficulties, raise
concerns, and receive support. The annotation
guidelines also required counterspeech to avoid
abusive language, address the content rather than
the author, and maintain a respectful and empa-
thetic tone. Finally, annotators were compensated
fairly by their respective institutions in compliance
with applicable national laws.
Privacy and data provenance.The dialogues in
the dataset are fictitious and were produced through
expert writing, post-editing, and human-machine
collaboration rather than by scraping conversations
between real users. This choice avoids the col-
lection of personal user interactions and reduces
privacy risks associated with sensitive online dis-
cussions. At the same time, the counterspeech

turns are grounded in external knowledge from fact-
checking articles and NGO-derived anti-stereotype
material, so that responses are based on verifiable
evidence rather than personal information. To com-
ply with licensing requirements, we will not redis-
tribute the original external knowledge text, but we
will make available the code to replicate our data
collection and to obtain the span-level annotations
from the web pages links.
Harmful content and misuse.Because the
dataset targets the intersection of hate speech and
misinformation, it necessarily contains offensive
and misleading statements. We keep such state-
ments explicit and stereotypical to support the con-
trolled study of grounded counterspeech genera-
tion, not to model or amplify realistic hateful be-
havior. Moreover, dialogues are structured so that
the final turn is always a counterspeech response,
reducing the risk of presenting hateful or mislead-
ing content as the conversational endpoint. The
dataset is intended for research on safer counter-
speech generation, retrieval, and evaluation, and
should not be used to train systems that generate,
rank, or amplify hateful or misleading content.
Model use.The generation experiments in this
paper are intended as benchmarks for the task en-
abled by the dataset, not as deployable moderation
or intervention systems. Automatically generated
counterspeech can be factually incomplete, contex-
tually inappropriate, or ineffective in sensitive real-
world settings. For this reason, we view the models
evaluated here as tools for research and for assisting
expert data creation, rather than as replacements for
trained moderators, fact-checkers, or civil-society
practitioners. Any deployment-oriented use should
include human oversight, additional safety evalua-
tion, and context-specific validation.
References
Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli,
David Dale, Ning Dong, Paul-Ambroise Duquenne,
Hady Elsahar, Hongyu Gong, Kevin Heffernan, John
Hoffman, and 1 others. 2023. Seamlessm4t: Mas-
sively multilingual & multimodal machine transla-
tion.arXiv preprint arXiv:2308.11596.
Nicola Bertoldi, Mauro Cettolo, and Marcello Federico.
2013. Cache-based online adaptation for machine
translation enhanced computer assisted translation.
InMT-Summit, pages 35–42.
Helena Bonaldi, Yi-Ling Chung, Gavin Abercrombie,
and Marco Guerini. 2024a. NLP for counterspeechagainst hate: A survey and how-to guide. InFind-
ings of the Association for Computational Linguis-
tics: NAACL 2024, pages 3480–3499, Mexico City,
Mexico. Association for Computational Linguistics.
Helena Bonaldi, Greta Damo, Nicolás Benjamín
Ocampo, Elena Cabrio, Serena Villata, and Marco
Guerini. 2024b. Is safer better? the impact of
guardrails on the argumentative strength of LLMs
in hate speech countering.Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 3446–3463.
Helena Bonaldi, Sara Dellantonio, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2022. Human-
machine collaboration approaches to build a dialogue
dataset for hate speech countering. InProceedings
of the 2022 Conference on Empirical Methods in
Natural Language Processing, pages 8031–8049.
Regina Cazzamatta. 2025. Global misinformation
trends: Commonalities and differences in topics,
sources of falsehoods, and deception strategies across
eight countries.new media & society, 27(11):6334–
6358.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024a. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024b. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Yi-Ling Chung, Elizaveta Kuzmenko, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2019. CONAN -
COunter NArratives through nichesourcing: a mul-
tilingual dataset of responses to fight online hate
speech. InProceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics,
pages 2819–2829, Florence, Italy. Association for
Computational Linguistics.
Yi-Ling Chung, Serra Sinem Tekiro ˘glu, and Marco
Guerini. 2021a. Towards knowledge-grounded
counter narrative generation for hate speech.Find-
ings of the Association for Computational Linguistics:
ACL-IJCNLP 2021, pages 899–914.
Yi-Ling Chung, Serra Sinem Tekiro ˘glu, Sara Tonelli,
and Marco Guerini. 2021b. Empowering NGOs in
countering online hate messages.Online Social Net-
works and Media, 24:100150.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal,
Vishrav Chaudhary, Guillaume Wenzek, Francisco
Guzmán, Edouard Grave, Myle Ott, Luke Zettle-
moyer, and Veselin Stoyanov. 2020. Unsupervised
cross-lingual representation learning at scale. InPro-
ceedings of the 58th annual meeting of the associa-
tion for computational linguistics, pages 8440–8451.

Joe Davison. 2020. joeddav/xlm-roberta-large-
xnli. https://huggingface.co/joeddav/
xlm-roberta-large-xnli.
Agustín Manuel de los Riscos and Luis Fernando
D’Haro. 2021. Toxicbot: A conversational agent
to fight online hate speech.Conversational dialogue
systems for the next decade, pages 15–30.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. Qlora: Efficient finetuning
of quantized llms.Advances in neural information
processing systems, 36:10088–10115.
Mekselina Do ˘ganç and Ilia Markov. 2023. From generic
to personalized: Investigating strategies for generat-
ing targeted counter narratives against hate speech. In
Proceedings of the 1st Workshop on CounterSpeech
for Online Abuse (CS4OA), pages 1–12.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th conference of the european chapter of
the association for computational linguistics: system
demonstrations, pages 150–158.
Margherita Fanton, Helena Bonaldi, Serra Sinem
Tekiro ˘glu, and Marco Guerini. 2021. Human-in-the-
loop for data collection: a multi-target counter narra-
tive dataset to fight online hate speech. InProceed-
ings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing
(Volume 1: Long Papers), pages 3226–3240.
Simone Fontana. 2022. Dieci falsi miti da sfatare sulla
comunità lgbt+. Accessed: 2026-05-20.
Markus Frohmann, Igor Sterner, Ivan Vuli ´c, Benjamin
Minixhofer, and Markus Schedl. 2024. Segment any
text: A universal approach for robust, efficient and
adaptable sentence segmentation. InProceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing, pages 11908–11941.
Damián Furman, Pablo Torres, José Rodríguez, Diego
Letzen, Maria Martinez, and Laura Alemany. 2023.
High-quality argumentative information in low re-
sources approaches improve counter-narrative gener-
ation. InFindings of the Association for Computa-
tional Linguistics: EMNLP 2023, pages 2942–2956,
Singapore. Association for Computational Linguis-
tics.
Mohamed H Gad-Elrab, Daria Stepanova, Jacopo Ur-
bani, and Gerhard Weikum. 2019. Exfakt: A frame-
work for explaining facts over knowledge graphs and
text. InProceedings of the twelfth ACM international
conference on web search and data mining, pages
87–95.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.Zhijiang Guo, Michael Schlichtkrull, and Andreas Vla-
chos. 2022. A survey on automated fact-checking.
Transactions of the Association for Computational
Linguistics, 10:178–206.
Michael Hameleers, Toni Van der Meer, and Rens
Vliegenthart. 2022. Civilized truths, hateful lies? in-
civility and hate speech in false information–evidence
from fact-checked statements in the us.Information,
Communication & Society, 25(11):1596–1613.
Bing He, Mustaque Ahamad, and Srijan Kumar.
2023. Reinforcement learning-based counter-
misinformation response generation: a case study
of covid-19 vaccine misinformation. InProceedings
of the ACM Web Conference 2023, pages 2698–2709.
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao.
2023. Making large language models a better founda-
tion for dense retrieval.Preprint, arXiv:2312.15503.
Genoveffa Martone, Helena Bonaldi, and Marco
Guerini. 2026. Assisted counterspeech writing at
the crossroads of hate speech and misinformation.
Preprint, arXiv:2605.22435.
Binny Mathew, Punyajoy Saha, Hardik Tharad, Subham
Rajgaria, Prajwal Singhania, Suman Kalyan Maity,
Pawan Goyal, and Animesh Mukherjee. 2019. Thou
shalt not hate: Countering online hate speech. In
Proceedings of the International AAAI Conference
on Web and Social Media, volume 13, pages 369–
380.
Stefano Menini, Daniel Russo, Alessio Palmero Apro-
sio, and Marco Guerini. 2025. First-aid: the first
annotation interface for grounded dialogues. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 3: Sys-
tem Demonstrations), pages 563–571.
Jimin Mun, Emily Allaway, Akhila Yerukola, Laura
Vianna, Sarah-Jane Leslie, and Maarten Sap. 2023.
Beyond denouncing hate: Strategies for countering
implied biases and stereotypes in language. InPro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing.
Jimin Mun, Cathy Buerger, Jenny T Liang, Joshua Gar-
land, and Maarten Sap. 2024. Counterspeakers’ per-
spectives: Unveiling barriers and ai needs in the fight
against online hate. InProceedings of the 2024 CHI
Conference on Human Factors in Computing Systems,
pages 1–22.
Sarah Myers West. 2018. Censored, suspended, shadow-
banned: User interpretations of content moderation
on social media platforms.New Media & Society,
20(11):4366–4383.
OpenAI. 2024. GPT-4o mini: advancing cost-efficient
intelligence. https://openai.com/index/
gpt-4o-mini-advancing-cost-efficient-intelligence/ .
OpenAI blog post about the GPT-4o mini model
release.

OpenAI. 2025. Gpt-4.1 mini. https://platform.
openai.com/docs/models/gpt-4.1-mini . Ac-
cessed: 2026-05-25.
Kashyap Popat, Subhabrata Mukherjee, Andrew Yates,
and Gerhard Weikum. 2018. DeClarE: Debunking
fake news and false claims using evidence-aware
deep learning.Proceedings of the 2018 Conference
on Empirical Methods in Natural Language Process-
ing, pages 22–32.
The Associated Press. 2022. Video of children reciting
quran at qatar stadium is from 2021. Accessed: 2026-
05-20.
Stephen E Robertson, Steve Walker, Susan Jones,
Micheline M Hancock-Beaulieu, Mike Gatford, and
1 others. 1995.Okapi at TREC-3. British Library
Research and Development Department.
Daniel Russo. 2025. Trenteam at multilingual counter-
speech generation: Multilingual passage re-ranking
approaches for knowledge-driven counterspeech gen-
eration against hate. InProceedings of the First
Workshop on Multilingual Counterspeech Genera-
tion, pages 77–91.
Daniel Russo, Shane Kaszefski-Yaschuk, Jacopo Sta-
iano, and Marco Guerini. 2023a. Countering misin-
formation via emotional response generation.Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pages 11476–
11492.
Daniel Russo, Stefano Menini, Jacopo Staiano, and
Marco Guerini. 2025a. Face the facts! evaluating
rag-based pipelines for professional fact-checking.
InProceedings of the 18th International Natural Lan-
guage Generation Conference, pages 846–865.
Daniel Russo, Fariba Sadeghi, Stefano Menini, and
Marco Guerini. 2025b. Euroverdict: A multilingual
dataset for verdict generation against misinformation.
InFindings of the Association for Computational
Linguistics: ACL 2025, pages 16617–16634.
Daniel Russo, Serra Sinem Tekiro ˘glu, and Marco
Guerini. 2023b. Benchmarking the Generation of
Fact Checking Explanations.Transactions of the
Association for Computational Linguistics, 11:1250–
1264.
Carla Schieb and Mike Preuss. 2016. Governing hate
speech by means of counterspeech on facebook. In
66th ica annual conference, at fukuoka, japan, pages
1–23.
Sentence-Transformers. 2021. all-mpnet-
base-v2. https://huggingface.co/
sentence-transformers/all-mpnet-base-v2.
Matthew Snover, Bonnie Dorr, Richard Schwartz, Lin-
nea Micciulla, and John Makhoul. 2006. A study of
translation edit rate with targeted human annotation.
InProceedings of association for machine translation
in the Americas, volume 200, 6. Cambridge, MA.Dominik Stammbach and Elliott Ash. 2020. e-fever: Ex-
planations and summaries for automated fact check-
ing.Proceedings of the 2020 Truth and Trust Online
(TTO 2020), pages 32–43.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Serra Sinem Tekiro ˘glu, Yi-Ling Chung, and Marco
Guerini. 2020. Generating counter narratives against
online hate speech: Data and strategies.Proceedings
of the 58th Annual Meeting of the Association for
Computational Linguistics, pages 1177–1190.
Marco Turchi, Matteo Negri, and Marcello Federico.
2013. Coping with the subjectivity of human judge-
ments in mt quality estimation. InProceedings of the
Eighth Workshop on Statistical Machine Translation,
pages 240–251.
Maria Estrella Vallecillo-Rodríguez, Arturo Montejo-
Raéz, and Maria Teresa Martín-Valdivia. 2023. Auto-
matic counter-narrative generation for hate speech in
spanish.Procesamiento del lenguaje natural, 71:227–
245.
Bertie Vidgen and Leon Derczynski. 2020. Direc-
tions in abusive language training data, a system-
atic review: Garbage in, garbage out.Plos one,
15(12):e0243300.
Claire Wardle. 2024. A conceptual analysis of the over-
laps and differences between hate speech, misinfor-
mation and disinformation.Department of Peace
Operations (DPO). Office of the Special Adviser on
the Prevention of Genocide (OSAPG). United Na-
tions.
Amanda L Wintersieck. 2017. Debating the truth:
The impact of fact-checking during electoral debates.
American politics research, 45(2):304–331.
Zeqiu Wu, Yi Luan, Hannah Rashkin, David Reit-
ter, Hannaneh Hajishirzi, Mari Ostendorf, and Gau-
rav Singh Tomar. 2022. Conqrr: Conversational
query rewriting for retrieval with reinforcement learn-
ing. InProceedings of the 2022 Conference on Em-
pirical Methods in Natural Language Processing,
pages 10000–10014.
Fanghua Ye, Meng Fang, Shenghui Li, and Emine Yil-
maz. 2023. Enhancing conversational search: Large
language model-aided informative query rewriting.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 5985–6006.
Fengzhu Zeng and Wei Gao. 2024. JustiLM: Few-shot
justification generation for explainable fact-checking
of real-world claims.Transactions of the Association
for Computational Linguistics, 12:334–354.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Wein-
berger, and Yoav Artzi. 2019. Bertscore: Evalu-
ating text generation with BERT.arXiv preprint
arXiv:1904.09675.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.

A Appendix
A.1 Document collection
Fact-checking articlesTo retrieve multilingual
fact-checking articles from Google Fact Check
Tools Explorer, the English keyword dictionary
produced by Martone et al. (2026) was automat-
ically translated into Polish, Spanish, and Italian
using deepl .15The translated keywords were then
submitted to our NGO collaborators, who worked
on the lists in their respective native languages.
They contributed by fixing potentially wrong trans-
lations, removing words that were not used in their
linguistic context, and adding relevant synonyms.
The complete keyword lists obtained from this pro-
cess are available in Table 11. Additionally, Fig-
ure 2 shows the guidelines used to select the fact-
checking articles to keep as external knowledge,
based on their ability to create or fuel hate against
the considered target groups. Below we provide an
example by The Associated Press (2022):
Claim: A video shows the 2022 FIFA World Cup in
Qatar opening with children reciting the Quran.
Fact-checking: False. The video was filmed on Oct.
22, 2021, and shows an inauguration ceremony for the
Al Thumama Stadium, a World Cup venue in Doha.
THE FACTS: The World Cup began Sunday in Qatar,
the first Arab or Muslim nation to host the competition.
Social media users shared a year-old video from the
inauguration of the stadium, falsely claiming it showed
the tournament’s opening ceremony[...].
NGO reportsThe URL domains from
which we extracted the NGO reports in Ital-
ian, Spanish, and Polish are the following:
https://biblioteka.ceo.org.pl , https:
//humanityinaction.org/country/poland/ ,
https://fundacionadecco.org , https:
//holocaustcentrenorth.org.uk ,https://
maldita.es ,https://migrantesenigualdad.
es, https://oko.press , https://
porcausa.org , https://uchodzcy.info ,
https://uprzedzuprzedzenia.org ,https://
www.facta.news ,https://www.huffpost.com ,
https://www.poradnikzdrowie.pl . Below is
an example in Italian by Fontana (2022) in Facta:
15https://pypi.org/project/deepl/Myth: Le persone trans sono uomini vestiti da donne
(o viceversa).
Anti-stereotype: Gli uomini vestiti da donne e le
donne vestite da uomini sono chiamati “crossdresser”
e il loro comportamento non ha necessariamente a che
fare con l’identità di genere percepita o l’orientamento
sessuale. [...]
A.2 Human-Machine Collaboration Details
Pre-compiled StrategyFor generation we use
gpt-4o-mini-2024-07-18 (Temp: 0.7, Max To-
kens: 500). Documents are split with SaT
(sat-1l-sm , universal dependencies format). We
use the following prompt:
Given the following article, generate a
dialogue in <LANGUAGE> between a person
spreading hate against <TARGET> and an NGO
operator who provides polite and informed
counterstatements based on the article. The
hater does not give up easily on their
opinions. The dialogue must include at most
<TURNS_NUMBER> exchanges. Return the
response in JSON format where each turn is
clearly marked by the speaker. Use the
following format: { "dialogue": [ {"speaker":
"<SPEAKER_1>", "text": "[Dialogue]"}, {"
speaker": "<SPEAKER_2>", "text": "[Dialogue
]"}, ... ] } Ensure the structure remains
consistent throughout.
Interactive Strategy
•Generators:We run preliminary experiments
with gpt-4o-mini (same hyperparameters
as pre-compiled), but we encounter issues
as the model refuses to generate the hate-
ful/misinformed turns. Therefore, we fine-
tune Llama-3.1-8B via QLoRA (Dettmers
et al., 2023) (4-bit, LoRA r= 32 ,α= 64 ,
drop=0), on an initial corpus of 1108 dia-
logues collected at the moment with the Man-
ual, Pre-compiled and Translation strategies.
Such data is split in the training and validation
set with a 80-20 proportion (886 dialogues for
training and 222 for validation). The dataset
is split by ensuring that there is not a dialogue
on the same article which is present both in
the training and validation set. Moreover, we
stratify the data according to the targeted mi-
nority in the dialogue and the language, so
that there is a similar distribution in the two
subsets according to these variables. Train-
ing runs for 5 epochs, max sequence length
3,000, using AdamW (LR: 5×10−5, cosine
decay, 3% warmup, weight decay 0.01) with

Target Keywords
English Italian Spanish Polish
Muslims muslim, islam, terrorist,
jihadi, jihad, ragheadter-
ror, arab, koran, quran,
sharia, towel head, rag
headmusulmano, islam, ter-
rorist, jihadista, jihad,
arabo, giornale, corano,
shariamusulmán, islam, ter-
rorista, yihadista, yihad,
árabe, corán, shariaszmatogłowy, muzuł-
manin, islam, terrorysta,
d˙zihadysta, d ˙zihad,
Arab, szariat
LGBTQIA+ gay, homosexual, homo-
sexuality, lgbt, lgbt+,
lgbti, lgbtq+, lgbtq,
faggot, gender, lesbian,
trans, transgender, trans-
sexual, queer, sexual,
sex, heterosexual, dyke,
gay pridetrans, frocio, lgbtq,
lgbtq+, genere, lesbica,
transgender, lgbt+,
queer, sessuale, sesso,
lgbti, eterosessuale,
transessuale, lgbt, gay
pride, omosessuale, gay,
omosessualitàtrans, maricón, lgbtq,
lgbtq+, género, lesbiana,
transgénero, lgbt+,
queer, sexual, sexo, lgbti,
heterosexual, bollera,
transexual, lgbt, orgullo
gay, homosexual, gay,
homosexualidadtrans, pedał, ciota,
lgbtq, lgbtq+, płe ´c,
lesbijka, transpł-
ciowy, lgbt+, queer,
seksualny, płe ´c biolog-
iczna, heteroseksualny,
lesba, transseksualista,
transseksualny, marsz
równo ´sci, homoseksual-
ista, homoseksualny, gej,
gejowski, homoseksual-
no´s´c
Migrants migrant, immigrant,
refugee, immigration,
foreigner, migration,
foreign, rapefugees,
invasion, invade,
refugeesnotwelcomerifugiato, i rifugiati
non sono i benvenuti,
invadere, invasione,
rapefugees, estero,
migrazione, straniero,
immigrazione, immi-
grato, migranterefugiado, invadir, in-
vasión, violadores, ex-
tranjero, migración, in-
migración, inmigrante,
migranteuchod´ zca, uchod´ zcy
nie s ˛ a mile widziani,
naje˙zd˙za´c, inwazja,
rapefugees, obcy, mi-
gracja, cudzoziemiec,
imigracja, imigrant,
migrant, refugeesnotwel-
come
Women woman, feminism, femi-
nist, gender, female, ha-
rassment, feminazi, shit-
hole, cunt, blameoneno-
tall, notallmen, victim-
card, sexual assault, vic-
tim cardviolenza sessuale, carta
della vittima, molestie,
colpa di non tutti, fica,
cesso, feminazi, fem-
minile, genere, femmin-
ista, femminismo, donna,
scheda-vittima, non tutti
gli uominiagresión sexual, acoso,
feminazi, mujer, género,
feminista, feminismonapa ´s´c na tle seksu-
alnym, karta ofiary,
n˛ ekanie, notallmen,
blameonenotall, pizda,
zadupie, feminazistka,
kobieta, płe ´c, feministka,
feminizm, kurwa, femi-
nistyczny
People with
disabilities disabled, disability,
autistic, blind, deaf,
retard, downies, downy,
paralympics, wheelchairdisabile, disabilità,
autistico, cieco, sordo,
ritardato, down, par-
alimpiadi, sedia a rotellediscapacitado, discapaci-
dad, autista, ciego,
sordo, retrasado, down,
paralímpicos, silla de
ruedasniepełnosprawny,
niepełnosprawno ´s´c,
autystyczny, niewidomy,
głuchy, niedorozwini˛ ety
Jews jew, jewish, holocaust,
judaism, nazi, nazism,
genocideebreo, ebraico, olo-
causto, nazism, nazi,
genocidio, ebraismojudío, holocausto,
nazismo, nazi, geno-
cidio, judaísmo˙Zyd, ˙zydowski,
holokaust, nazizm,
nazista, nazistowski,
ludobójstwo, judaizm
Table 11: Multilingual keywords used to query Google Fact Check Tools for retrieving fact-checking articles related
to the groups of our interest.

These guidelines are meant to assist you in the task of selecting claims based on their ability to generate hate,
discrimination and negative feelings towards specific minority groups. Guidelines for article selection.
1.    Focus on Groups over Individuals
Avoid claims that focus on a specific individual (e.g., a specific celebrity,
politician, or public figure). Instead, make sure that the focus is on a generic
minority group identified by characteristics such as sex, religion, ethnicity, or
social status
Claim:
“Michelle Obama is a
transsexual.”
How to select a claim that is valuable for us?
2.  The claim may generate or reinforce harmful stereotypes, incite violence, or promote hatred
toward a minority group, even if not openly hateful in both the following cases:
3. Different Hateful Emotions
Negative sentiment toward a minority can be expressed even through
positive emotions caused by the troubles of a targeted group.
Claim:
“Drag shows are sex crimes punishable by death in Florida.”
Hate speech:
They deserve to die for their disgusting behavior.
Contains misinformation and stereotypes:
Claim: 
“The true author of Anne Frank’s diary
was an American man.”
Hate speech:
The whole Holocaust thing is a hoax!
Does not contain misinformation, but content
can be twisted or generalized:
Claim:
“Net migration is the highest it’s ever been.”
Hate speech:
Migrants are invading us!
What else to keep in mind?
4. Ensure Articles
Contradict Potential Hate
Check that the article
challenges potential hate or
offers a counter-narrative.
Discard articles that only
amplify hateful rhetoric
without addressing it.
Uncertain Classifications
If you are unable to confidently classify a claim according
to the above criteria, mark it with a “\”. We will take care
of reviewing them to make the final classification.
Mark Duplicates
If an article is valid but has duplicates (i.e., two or more
claims with identical articles), mark it with a “D”. If two
claims are similar but come from different articles, they may
still provide different perspectives on the same issues.
Maintain Objectivity and Neutrality
Evaluate claims in the most objective
way possible. Identify potential risks
without letting personal biases affect
the assessment. 
Correct Wrong Targets
If a claim’s target is wrongly attributed,
update it based on the claim’s content. This
is common for targets labeled as “OTHER”.
How to proceed?Figure 2: Guidelines for selecting articles with potential to fuel hate and discrimination.

an effective batch size of 18 and early stop-
ping (patience=10, eval every 10 steps). The
prompts employed for training are the same
used to collect dialogues with the interactive
strategy, and is shown below.
•Retrieval:We test both BAAI/bge-m3 and a
fine-tunedBAAI/bge-reranker-v2-m3.
Also for preparing the training data for the
reranker we employed the portion of dialogues
collected at the moment with the Manual, Pre-
compiled and Translation strategies. To pre-
pare the data, each article document was seg-
mented into sentences using the spaCy En-
glish sentencizer. For every dialogue, hater’s
turns were treated as queries, while the subse-
quent counterspeech turns provided grounded
evidence spans annotated in the data. These
evidence spans were also sentence-segmented
and used as positive passages. Negative pas-
sages were obtained by extracting all remain-
ing article sentences that did not overlap with
the normalized positive spans, where nor-
malization included lowercasing, whitespace
stripping, and Unicode normalization to re-
move accent variations. Each resulting train-
ing instance therefore consisted of a query,
a set of positive evidence sentences, and a
set of negative sentences sampled from the
same source article. Instances without nega-
tive examples were removed, thus obtaining
2659 training examples. The reranker was
fine-tuned for 3 epochs using contrastive learn-
ing (no distillation, max len 512, FP16, ef-
fective batch size 8, negative group size 8).
Optimized with AdamW (LR: 6×10−5, 10%
warmup, weight decay 0.01, gradient check-
pointing).
Prompt for Interactive Strategy
• First Turn:
Given the provided claim and fact-checking
article, pretend to be a hateful person and
provide a hypothetical statement in <
LANGUAGE> spreading harmful or
discriminatory rhetoric against <TARGET>
based on the claim (contained in <claim>
tags) and the article (contained in <article
> tags). The information needed to answer
your message must be present exclusively in
the article.
<article><ARTICLE></article>
<claim><CLAIM></claim>
• Hater:TARGET Tot %
DISABLED 141 7.00
JEWS 44 2.18
LGBTQIA+ 264 13.10
MIGRANTS 615 30.52
MUSLIMS 376 18.66
POC 225 11.17
WOMEN 336 16.67
multitarget 11 0.55
Table 12: Distribution of dialogues according to target.
Docs Lang. Mono. Trans. Tot. % Mono. % Trans. % Tot.
1EN 336 0 336 16.67 0.00 16.67
ES 110 194 304 5.46 9.63 15.09
IT 108 194 302 5.36 9.63 14.99
MT 37 256 293 1.84 12.70 14.54
PL 137 193 330 6.80 9.58 16.38
2EN 50 0 50 2.48 0.00 2.48
ES 50 50 100 2.48 2.48 4.96
IT 50 50 100 2.48 2.48 4.96
MT 50 50 100 2.48 2.48 4.96
PL 50 50 100 2.48 2.48 4.96
Table 13: Language distribution across one and two-
document dialogues.
Given the dialogue history and the latest
turn which references context in the
dialogue history, formulate a reply to the
last turn in <LANGUAGE> pretending to be a
person who is spreading hate against <TARGET
>. Do not repeat prior turns. The
information needed to answer your message
must be present exclusively in the provided
article (contained in <article> tags).
<article><ARTICLE></article>
• Operator:
You are an operator who provides polite and
informed counterstatements in <LANGUAGE> to
hateful and misinformed messages. Your task
is to provide a relevant answer to the hater
using the provided evidence and past
dialogue history. The evidence is contained
in <context> tags. Answer by referring
exclusively to the content of the document.
<context><GROUND_TEXT></context>
Translation Strategy: We use
facebook/hf-seamless-m4t-large
(max_length=512,do_sample=False).
A.3 More details on dataset description
Table 12 reports the distribution of dialogues ac-
cording to target, Table 13 according to language
and Table 14 according to both target and language.
“Multitarget” refers to dialogues that can be referred
to more than one target at once.

TARGET language Tot. %
DISABLEDEN 26 1.29
ES 21 1.04
IT 27 1.34
MT 34 1.69
PL 33 1.64
JEWSEN 10 0.50
ES 5 0.25
IT 5 0.25
MT 17 0.84
PL 7 0.35
LGBT+EN 51 2.53
ES 66 3.28
IT 50 2.48
MT 44 2.18
PL 53 2.63
MIGRANTSEN 102 5.06
ES 141 7.00
IT 133 6.60
MT 95 4.71
PL 144 7.15
MUSLIMSEN 81 4.02
ES 65 3.23
IT 69 3.42
MT 84 4.17
PL 77 3.82
POCEN 46 2.28
ES 44 2.18
IT 43 2.13
MT 51 2.53
PL 41 2.03
ROMANI IT 3 0.15
WOMENEN 67 3.33
ES 60 2.98
IT 70 3.47
MT 66 3.28
PL 73 3.62
multitargetEN 3 0.15
ES 2 0.10
IT 2 0.10
MT 2 0.10
PL 2 0.10
Table 14: Distribution of dialogues according to target
and language: “multitarget” refers to dialogues that can
be referred to more than one target at once.
Docs Strat. HTER Time Ground RR or RRed RR∆
1Interactive 0.408 169.544 29.532 8.775 7.676 -1.099
Manual - 179.446 27.558 -2.396-
Pre-compiled 0.314 105.374 38.9264.0473.994 -0.053
2Interactive 0.434 159.962 33.81 12.198 8.219-3.979
Manual - 515.89 30.62 - 3.237 -
Pre-compiled0.302 85.388 60.180 5.725 4.686 -1.039
Table 15: Annotation effort results according to number
of documents and strategy.A.4 More analyses: monolingual data
Table 15 shows the annotation effort results ob-
tained by jointly considering the number of doc-
uments and strategy. The pre-compiled strategy
is the one with the lowest HTER, longest ground,
and shortest annotation time in both one- and two-
documents strategies. Manually writing dialogues
requires the highest annotation time in both doc-
ument configurations, but it also guarantees the
lowest RR ed: in settings where LLM outputs can-
not be modified, manual annotation remains the
best source of high-quality data. consistently with
the results grouped by strategy only, the interactive
strategy produces the most repetitive generations
in both document settings, as shown by the RR or,
while also obtaining the greatest reduction after
post-editing. Syntactic metrics results are coherent
with those shown by strategy only.
A.5 More analyses: translated data
Table 17 and 18 report, respectively, the annotation
effort and syntactic metrics results for translated
dialogues according to the number of documents
and language. Results are consistent with those
obtained when grouping by language only, as dis-
cussed in §4.3.
A.6 Retrieval experiment details
We chunk documents with Llamaindex
(chunk_size=256, chunk_overlap=64). To
obtain positive chunks for testing, we automat-
ically align the human-annotatedground text
spans to the obtained document chunks. For
each query (i.e. hate speech turn), the search
space is restricted to chunks belonging only to the
annotated target documents via a document-to-
chunk index mapping. Each ground text is then
compared against all candidate chunks using a
normalized longest-common-substring overlap
score computed with SequenceMatcher . The
2 chunks with highest overlap over a 0.5 fixed
threshold are labeled as positive examples. This
produces query-level sets of positive chunk indices
aligned directly to the pre-computed chunk corpus.
After matching, we remove any duplicates to
ensure that each chunk is mapped at most once to
each query. Retrieval experiments took roughly 1
hour to run on a NVIDIA A40, Ampere GPU.
We use the sentence-transformers16library
for both BGE-M3 and Qwen3-Embedding. For
16https://www.sbert.net/

Docs Strat. MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
1Interactive5.2404.874 -0.3664.3734.089 -0.284 2.024 2.006 -0.018 0.323 0.314 -0.009
Manual - 5.033 - - 4.132 - - 2.194 - - 0.275 -
Pre-compiled 4.676 4.860.1843.838 3.9130.0751.984 2.2170.2330.288 0.286 -0.002
2Interactive 4.681 4.685 0.004 3.888 3.917 0.029 1.936 2.01 0.0740.358 0.348-0.01
Manual -5.609- -4.339- -2.405- - 0.132 -
Pre-compiled 4.52 4.558 0.038 3.66 3.664 0.0042.206 2.343 0.137 0.334 0.3360.002
Table 16: Syntactic metrics results according to number of documents and strategy.
Docs Lang. HTER Time Ground RR or RRed RR∆
1ES1.010 41.15 17.43 3.115 4.058 0.943
IT 1.07733.37 17.54 2.564 3.076 0.512
MT 1.404 153.7318.04 2.162 2.135 -0.027
PL 1.287 94.49 17.54 2.262 2.165-0.097
2ES 1.019 129.55 30.88 3.166 4.361 1.195
IT1.001 28.2230.62 2.844 3.843 0.999
MT 1.439 206.26 30.62 2.149 2.160 0.011
PL 1.356 173.0631.35 2.081 1.783 -0.298
Table 17: Annotation effort results for translated dia-
logues according to the number of documents and lan-
guage.
what regards the latter, we also make sure to
append the<|endoftext|>token to each instruc-
tion, as suggested by the authors: Instruction
Query<|endoftext|> . For Qwen3, we employ the
following instruction: Given a query, retrieve
relevant passages that refute the query.
For both monolingual and cross-lingual setup,
we employ as queries all hater’s turns followed by
a counterspeech grounded on external knowledge
(we discard counterspeech examples not grounded
on any external knowledge such as clarifying ques-
tions). In this way, we obtain a set of 2710 queries
for the cross-lingual setup and 2409 queries for the
monolingual setup.
For the rewritten query configuration, query is
rewritten with GPT-5.4, max_token = 1024, n = 1,
stop = None, temperature = 0, using the following
prompt from Ye et al. (2023):
• System prompt:
"Given a query and its context, rewrite
the query in {language} and
decontextualize it by addressing
coreference and omission issues. The
resulting query should retain its
original meaning and be as informative as
possible, and should not duplicate any
previous query in the context."
• User prompt:
"Context: {context}\nQuery: {question}\
nRewrite: "Retrieval Cross-lingual resultsTable 19
presents the cross-lingual results of the retrieval
experiment. We include BM25 in the cross-lingual
evaluation purely as a lower-bound baseline to
quantify the lexical overlap (e.g., via shared
entities, proper nouns, and loanwords) between the
non-English queries and English documents. As
expected, BM25 experiences a near-total collapse,
confirming that surface-level keyword matching
is fundamentally inadequate for this task and
validating the necessity of dense cross-lingual
embedding spaces.
As for the monolingual setting, Qwen3 performs
best across all query configurations, followed by
BGE-M3. Interestingly, in the cross-lingual config-
uration both dense embedding models outperform
their monolingual counterparts. We hypothesize
that, because these embedders are predominantly
optimized on high-quality English corpora, map-
ping a non-English query directly into this highly
refined English space yields cleaner alignment and
stronger retrieval signals than navigating the nois-
ier, lower-resource local language document spaces
inherent to the monolingual task. Finally, differ-
ently from the monolingual setting, where QDCis
almost always the best performing query formula-
tion, in the cross-lingual scenario for both BM25
and BGE-M3Q RoutperformsQ DC.
A.7 Generation experiment details
For both the monolingual and cross-lingual set-
ting we employ a subset of 400 dialogues (100
dialogues per language). As for the retrieval
experiment, we employ as queries all hater’s
turns followed by a counterspeech grounded on
external knowledge (we discard counterspeech
examples not grounded on any external knowl-
edge such as clarifying questions). Table 20
and 21 shows the distribution of queries per lan-
guage and per target, respectively. We use the
following hyperparameters for the Qwen3 8B
model: max_new_tokens=100, temperature=0.7,
top_p=0.8, top_k=20, min_p = 0. Generations took

Docs Lang. MSD or MSD ed MSD ∆ASD or ASD ed ASD ∆NST or NST ed NST ∆CW or CW ed CW∆
1ES 4.622 4.630 0.008 3.966 3.911 -0.055 1.881 1.949 0.068 0.309 0.307 -0.002
IT4.658 4.655 -0.003 3.874 3.804 -0.07 1.979 2.049 0.07 0.357 0.356 -0.001
MT - - - - - - - - - 0.252 0.2550.003
PL 4.6214.7190.0984.0913.897 -0.194 1.726 2.0810.355 0.415 0.409 -0.006
2ES 4.911 5.010 0.099 3.9833.985 0.002 2.207 2.276 0.069 0.309 0.308 -0.001
IT5.0365.000 -0.036 3.952 3.868 -0.0842.365 2.3980.033 0.365 0.364 -0.001
MT - - - - - - - - - 0.253 0.252 -0.001
PL 4.8785.013 0.135 4.077 3.959 -0.118 2.023 2.355 0.3320.428 0.425-0.003
Table 18: Syntactic metrics results for translated dialogues according to the number of documents and language.
Model Query Hit@10 MAP@10 Recall@10
BM25Q 0.034 0.015 0.029
QDC 0.058 0.023 0.050
QR 0.063 0.029 0.054
BGE-M3Q 0.432 0.225 0.389
QDC 0.649 0.337 0.601
QR 0.660 0.361 0.611
Qwen3Q 0.457 0.242 0.411
QDC *0.709 *0.401 *0.663
QR 0.696 0.386 0.649
Table 19: Cross-lingual zero-shot chunk retrieval per-
formance (Non-English Queries to English Knowledge
Base). Bold text indicates the best performance within
each model group. * denotes the overall global maxi-
mum for each metric.
roughly 1 hour to run on a NVIDIA A40, Ampere
GPU. Below we show the prompts we employed
(for each prompt mentioning the dialogue history,
this mention was not included for the generation of
the first counterspeech where there was no dialogue
history beyond the given HS):
•System Prompt
Given the hateful claim provided in HS, and
the dialogue history, generate a two-
sentence-long CS in {language} following the
guidelines provided in GUIDELINES.
GUIDELINES: Avoid abusive language and
divisive labels (e.g., racist, fascist).
Challenge the claim, not the person who
wrote it. Counter misinformation with
accurate facts, evidence, impartiality, and
reliable sources. Provide context for the
misinformed hateful claim. Express support
for those under attack and respond with
kindness and respect. Empathize with
underlying fears or anxieties that caused
the expression of hate. Challenge negative
stereotypes using facts and providing
context.
• User prompt without knowledge (CS base)
Dialogue History: {dialogue_history}
Current HS: {hateful_message}Language # Q Mono. # Q Cross.
English 243 -
Italian 235 230
Polish 343 243
Spanish 343 240
Maltese - 237
Table 20: Distribution of language per queries in mono-
lingual and cross-lingual setting for the generation ex-
periment.
Task: Based on the current HS and dialogue
history, generate the two-sentence
counterstatement (CS) in {language}
•User prompt with knowledge: knowledge
used is gold for CS goldand the 5 top retrieved
chunks for CS retr
Dialogue History: {dialogue_history}
GROUND: {knowledge}
Current HS: {hateful_message}
Task: Based on the current HS, dialogue
history, and GROUND, generate the two-
sentence counterstatement (CS) in {language}.
The GROUND context consists of several
distinct, isolated text chunks. You must
necessarily use the facts contained in the
GROUND chunks to contrast misinformation and
stereotypes. Answer by referring
exclusively to GROUND chunks, don't cite the
sources in brackets.
Evaluation metrics detailsForBERTScore
we report F1scores using xlm-roberta-large ,
forNLI Entailmentwe calculate the entail-
ment of the CS by the preceding HS with
xlm-roberta-large-xnli (Davison, 2020). We
employ gpt-4.1-mini asLLM-as-a-judgeto
measure Faithfulness and Answer relevance, with
max_output_tokens=20 and temperature=0. We
employ an adapted version of the definitions given
by Es et al. (2024) in the prompts to calculate these

Target # Q Mono. # Q Cross.
DISABLED 88 147
JEWS 49 45
LGBT+ 243 165
MIGRANTS 244 137
MUSLIMS 163 148
POC 63 149
ROMANI 7 -
WOMEN 214 139
Multitarget 8 20
Table 21: Distribution of target per queries in monolin-
gual and cross-lingual setting for the generation experi-
ment.
metrics, but we add a score rubric to more clearly
direct model scoring. In particular, forFaithful-
nesswe employ the following prompts:
• System Prompt:
Faithfulness measures the information
consistency of the answer against the given
context. Any claims that are made in the
answer that cannot be deduced from context
should be penalized. Given an answer and
context, assign a score for faithfulness in
the range 1-5.
### Score Rubrics [Faithfulness]
Score 1: The answer contains major
contradictions, fabricated information, or
unsupported claims. Most of the content
cannot be verified from the context.
Score 2: The answer includes several
unsupported or inaccurate claims. Important
details are invented, exaggerated, or
inconsistent with the context.
Score 3: The answer is mostly grounded in
the context, but includes some unsupported
assumptions, minor hallucinations, or
overgeneralizations.
Score 4: The answer is highly consistent
with the context and contains only small
ambiguities or negligible unsupported
additions that do not materially affect
accuracy.
Score 5: Every claim in the answer is
directly supported by or can be clearly
inferred from the context. The answer
contains no hallucinations, contradictions,
or misleading interpretations.
Return ONLY a single digit (1-5). No
reasoning. No extra text.
• User Prompt:
Context: {ground}
Answer: {response}
Score:
ForAnswer relevancewe employ these
prompts:• System Prompt:
Answer Relevancy measures the degree to
which a counterstatement (CS) directly
addresses and is appropriate for a given
harmful statement (HS). It penalizes the
presences of redundant information or
incomplete CS given an HS. Given an HS and
CS, assign a score for answer relevancy in
the range 1-5.
### Score Rubrics [Answer Relevancy]
Score 1: The CS fails to address the HS or
is mostly irrelevant. It may omit the main
point entirely or provide unrelated
information.
Score 2: The CS partially addresses the HS
but includes substantial irrelevant,
redundant, or distracting content. Important
aspects of the HS are left unanswered.
Score 3: The CS addresses the main point of
the HS but may be incomplete, somewhat
unfocused, or contain unnecessary
information that reduces clarity.
Score 4: The CS directly addresses the HS
and is mostly complete. Minor redundancy or
small omissions may be present, but the
response remains focused and appropriate.
Score 5: The CS fully, directly, and
efficiently addresses the HS. All relevant
aspects are covered with no unnecessary,
redundant, or off-topic information.
Return ONLY a single digit (1-5). No
reasoning. No extra text.
• User Prompt:
HS: {hs}
CS: {response}
Score:
Generation Cross-lingual resultsTable 22
presents the cross-lingual results for the genera-
tion experiment.

Metric Setting Value
BERTscoreCSgold 0.882
CSbase 0.870
CSretr 0.874
Faithfulness goldCSgold 3.486
CSbase 2.183
CSretr 2.507
Faithfulness retr CSretr 3.858
NLI Entailment goldCSgold 0.202
CSbase 0.045
CSretr 0.077
NLI Entailment retr CSretr 0.212
RelevanceCSgold 3.855
CSbase 3.917
CSretr 3.706
Table 22: CS generation quality across metrics (cross-
lingual). Bold values represent the maximum score
achieved within each evaluation category.