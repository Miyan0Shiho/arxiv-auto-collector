# Detecting Manipulated Contents Using Knowledge-Grounded Inference

**Authors**: Mark Huasong Meng, Ruizhe Wang, Meng Xu, Chuan Yan, Guangdong Bai

**Published**: 2025-04-29 20:33:54

**PDF URL**: [http://arxiv.org/pdf/2504.21165v1](http://arxiv.org/pdf/2504.21165v1)

## Abstract
The detection of manipulated content, a prevalent form of fake news, has been
widely studied in recent years. While existing solutions have been proven
effective in fact-checking and analyzing fake news based on historical events,
the reliance on either intrinsic knowledge obtained during training or manually
curated context hinders them from tackling zero-day manipulated content, which
can only be recognized with real-time contextual information. In this work, we
propose Manicod, a tool designed for detecting zero-day manipulated content.
Manicod first sources contextual information about the input claim from
mainstream search engines, and subsequently vectorizes the context for the
large language model (LLM) through retrieval-augmented generation (RAG). The
LLM-based inference can produce a "truthful" or "manipulated" decision and
offer a textual explanation for the decision. To validate the effectiveness of
Manicod, we also propose a dataset comprising 4270 pieces of manipulated fake
news derived from 2500 recent real-world news headlines. Manicod achieves an
overall F1 score of 0.856 on this dataset and outperforms existing methods by
up to 1.9x in F1 score on their benchmarks on fact-checking and claim
verification.

## Full Text


<!-- PDF content starts -->

Detecting Manipulated Contents Using Knowledge-Grounded Inference
Mark Huasong Meng
Technical University of MunichRuizhe Wang, Meng Xu
University of WaterlooChuan Yan, Guangdong Bai
The University of Queensland
Abstract
The detection of manipulated content, a preva-
lent form of fake news, has been widely studied
in recent years. While existing solutions have
been proven effective in fact-checking and ana-
lyzing fake news based on historical events, the
reliance on either intrinsic knowledge obtained
during training or manually curated context hin-
ders them from tackling zero-day manipulated
content which can only be recognized with real-
time contextual information.
In this work, we propose MANICOD , a tool
designed for detecting zero-day manipulated
content. MANICOD first sources contextual
information about the input claim from main-
stream search engines, and subsequently vec-
torizes the context for the large language model
(LLM) through retrieval-augmented generation
(RAG). The LLM-based inference can produce
a “truthful” or “manipulated” decision and offer
a textual explanation for the decision. To val-
idate the effectiveness of MANICOD , we also
propose a dataset comprising 4,270 pieces of
manipulated fake news derived from 2,500 re-
cent real-world news headlines. MANICOD
achieves an overall F1 score of 0.856 on this
dataset and outperforms existing methods by
up to 1.9x in F1 score on their benchmarks on
fact-checking and claim verification.
Warning: This paper contains manipulated con-
tent that may be offensive, harmful, or biased.
1 Introduction
The widespread mis- and disinformation (colloqui-
ally known as “fake news” (Canadian Centre for
Cyber Security, 2024)) has become a long-lasting
cyber threat in the Internet era (World Risk Poll,
2020; Zurko, 2022). In particular, disinformation
caused by manipulated content has been ranked
among the biggest sources of cybercrime and the
most potent cybersecurity threats (Mazurczyk et al.,
2024). Manipulated content, as the name suggests,
is typically produced based on true information
At least 150,000people have been killed duringtheBangladesh protests.Theinputstatementclaimsthat“Atleast150,000peoplehavebeenkilled”duringtheBangladeshprotests.However,basedontheprovidedcontext,itisstatedthat“atleast150peoplehavebeenkilled”inthenationwideclashesbetweenpoliceanduniversitystudents.Thereisasignificantdiscrepancyinthenumberoffatalitiesreportedintheinputstatement(150,000)comparedtotheactualnumbermentionedinthecontext(150).Thissuggeststhattheinputstatementcontainsanerrororexaggerationregardingthenumberofdeaths.Decision:False(manipulated)
AsecretmeetingtodiscussthefuturegovernanceofGazafollowingthedeadlywarwasheldinAbuDhabilastweekbytopofficialsfromtheU.S.,IranandtheUnitedArabEmirates.Theinputstatementclaimsthatasecretmeeting...However,basedonmyknowledgeofthecontextprovidedearlier,Ifoundseveralcontradictionsandfactualmistakes.Firstly,thereisnomentionofanyjointmeetingordiscussionbetweentheUS,Israel,andIranregarding...Infact,thereportssuggestthattheUS,Israel,andtheUAEareadvancinganalternativepost-wargovernancevision…thereisnomentionofanysecretmeetinginvolvingIran…Basedonmyanalysis,Ifoundcontradictionsbetweentheinputstatementandthecontextprovidedearlier.Decision:False(manipulated)
Figure 1: Two examples on how MANICOD detects
and analyzes simulated zero-day manipulated content
as of 2024 July 29, with manipulated content in the
user queries underlined in red, and key explanations
underlined in black.
(e.g., events that happened in the real world) but
contains subtle fabrications for malicious purposes.
Without proper detection and filtering techniques,
manipulated content could rapidly disseminate prej-
udice and hatred through the Internet, severely
threading the solidarity and harmony of society
and impeding the spread of truth. For instance, the
news on “egg shortage caused by avian flu and in-
flation” was manipulated to be caused by “RNA
technology” in order to discredit the research and
practices (Reuters Fact Check, 2023).
Fortunately, plenty of efforts have been put into
the research on effective manipulated content de-
tection in the past decade, which helps suppress the
spreading of disinformation (The Poynter Institute,
2024; The Annenberg Public Policy Center, 2024;
Google, 2024; Snopes Media Group, 2024). Early
works (Nie et al., 2019; Ma et al., 2019; Popat
et al., 2018; Wu et al., 2021) mainly resort to train-
ing machine learning (ML) classifiers of various
architectures to determine whether a user’s claim is
truthful. In addition to simply producing a binary
label, more recent works (Lee et al., 2021; Wang
and Shu, 2023; Wang et al., 2024) further leverage
emerging large language models (LLMs) to per-
form a more comprehensive inference and provide
1arXiv:2504.21165v1  [cs.CL]  29 Apr 2025

a human-readable justification for the classification,
demonstrating promising performance in validat-
ing historical events and traditional fact-checking
tasks.
Nonetheless, existing solutions mainly rely on
the knowledge learned during the training phase
to support subsequent automatic reasoning and
decision-making (Wang et al., 2024; Lee et al.,
2021; Wang and Shu, 2023). While some works
can take additional context to determine if a claim
is false (Shu et al., 2019; Lu and Te Li, 2020),
they require users to manually source, filter, and
summarize such additional information. The ab-
sence of automated real-time knowledge sourcing
about ongoing or recently happened events around
the world makes existing solutions incapable of
handling zero-day disinformation caused by manip-
ulated content, leaving push-button manipulated
content detection still an open question in the re-
search community.
One possible reason why real-time contextual
information is not emphasized in prior works is
that these solutions do not explicitly distinguish
manipulated content, which is still based on true
information, versus entirely false and baseless fab-
ricated content (which can still be determined to be
false based on logic, common sense, and facts in
the knowledge base). Therefore, a generic detector
that handles both manipulated content and fabri-
cated content may not assume the availability of
contextual information. However, this also means
that they will have to forgo the chance of using
relevant contextual information to decide whether
a claim is truthful.
In this paper, we emphasize that contextual in-
formation is crucial for a push-button solution to
detect and explain zero-day manipulated content,
i.e., fake news created based on events not avail-
able during the training of the detector. In fact,
real-time information is sometimes even necessary
in deciding the veracity of texts about ongoing or
recently happened events, as many events simply
cannot be inferred or predicted based on widely ac-
cepted ground truth and consensus (i.e., the black
swan theory (Taleb, 2010)). This is an inherent
disadvantage to existing works that classify a claim
merely based on pre-trained outdated data, as they
are incapable of precisely recognizing a piece of
zero-day information, regardless of how large an
ocean of knowledge is used in the model training
phase. Biden’s withdrawal from the 2024 presi-
dential election is a representative example of anunexpected event that cannot be verified or rea-
soned through without access to real-time informa-
tion (Post, 2024).
However, even with real-time context informa-
tion, detecting manipulated content autonomously
is still challenging. Manipulated content is often
formulated with minor or sometimes even incon-
spicuous alterations based on real stories, such as
the substitution of persons, time, and venues, or
exaggeration of key numbers – without structural
changes in the story. The minor distortion makes
fake news not only easy to mislead people but also
difficult to capture by conventional ML models that
are widely used in early works (Ma et al., 2019;
Lu and Te Li, 2020; Nie et al., 2019; Shu et al.,
2019). However, contextual information relevant
to this story can be rich. In a dataset of 4,270 ma-
nipulated news headlines, each headline contains
126.7 characters on average, with 28.0 characters
representing modifiable context, while only 8.4
characters are modified from the original news to
mimic malicious manipulation in the real world
(see §4 for dataset details). This makes locating
the minor distortion based on a large volume of
contextual information akin to finding a needle in
a haystack.
To address these challenges, we propose MANI-
COD, amani pulated content detector that can also
provide a textual explanation of why it makes the
decision. More specifically, MANICOD automat-
ically retrieves real-time contextual information
from the Internet, based on which it analyzes the
veracity of users’ queries accordingly. In addi-
tion to merely producing a true-or-false label or a
confidence score, MANICOD also aims to provide
explanations for the decision in natural language,
including a pinpoint on the part of the user’s input
where it suspects might be manipulated if applica-
ble.
We design MANICOD as a two-phase process,
namely online knowledge retrieval andknowledge-
grounded inference by LLMs . Specifically, MAN-
ICOD first searches the input statement via main-
stream search engines on the Internet to source the
most relevant information. The search results of the
user’s query are treated as contextual knowledge to
support the subsequent reasoning. Next, we lever-
age LLMs’ capabilities in natural language com-
prehension and generation to infer the veracity of
the user’s query and to provide an explanation. By
augmenting the LLM’s intrinsic knowledge base
with contextual knowledge, the LLM would be ca-
2

pable of reasoning about any piece of information
from the query, even about a recent event. The
veracity inference, at its core, is to ask the LLM to
look for contradictions, altered text, and/or factual
errors in the claim. Figure 1 depicts two running
examples of disinformation pieces and the output
of M ANICOD .
Considering mainstream LLMs all enforce con-
straints on the maximum token numbers in each
inference session, we resort to retrieval-augmented
generation (RAG) techniques to vectorize the re-
trieved knowledge and embed them into the session
context of LLMs. Thus, our approach can take ad-
vantage of the powerful capabilities of LLMs in
natural language comprehension and generation to
analyze users’ input and explain the decisions made.
At the same time, the adoption of RAG regulates
LLMs to focus on the provided context during the
inference, thereby minimizing the effects caused by
hallucination or inconsistent training data (Béchard
and Ayala, 2024; Shuster et al., 2021).
While useful, existing datasets are limited to
trivial facts or are sourced from social-media sce-
narios (Google, 2021; Thorne et al., 2018) which
might not be the best benchmarks to evaluate
MANICOD as we target real-world news headlines.
Therefore, besides comparing with these datasets,
we also propose a dataset containing manipulated
content based on recent news headlines. Our eval-
uation shows that MANICOD significantly outper-
forms existing approaches in diverse fact-checking
and fake news detection tasks. The experimental
results demonstrate the outstanding performance
ofMANICOD in detecting zero-day manipulated
content about recently happened events around the
world, with an overall F1 score of 0.856.
Summary . Our work makes the following contri-
butions:
•An explainable detector for manipulated con-
tent. We propose MANICOD , an open-sourced
end-to-end autonomous manipulated content de-
tector capable of sourcing real-time information
from the Internet and recognizing manipulated
contents fabricated based on recent real-world
events. Our approach leverages LLMs for in-
ference and explanation generation without ded-
icated training or fine-tuning. To the best of
our knowledge, we are the first of this kind to
address the growing concern of zero-day manip-
ulated content specifically.
•A dataset of manipulated news headlines . Weobserve two common types of content manipu-
lation and follow them to produce a dataset of
“fake news” by simulating the malicious manipu-
lation based on real-world events. Our dataset is
based on 2,500 real-world news headlines in En-
glish and contains 4,270 pieces of manipulated
news headlines, spanning 20 days in 2024. The
goal of this dataset is also to encourage more
research for a safer Internet environment.
Both MANICOD and the dataset will be open-
sourced upon the acceptance of this paper1. An
ethics statement of this work can be found in §8.
2 Background
2.1 Disinformation and Misinformation
In this paper, we take the view that disinforma-
tion describes content that is intentionally false
and designed to cause harm while misinforma-
tion refers to false content created without mali-
cious intents (Canadian Centre for Cyber Security,
2024; Zurko, 2022). According to the taxonomy
in (Wardle, 2020), mis- and disinformation can be
classified into the following types: (1) Fabricated
content: new content that is 100% false. (2) Ma-
nipulated content : distorted genuine information.
(3) Imposter content: impersonation of genuine
sources. (4) False context: factually accurate con-
tent presented together with false and seditious con-
textual information (e.g., a false text posted with
genuine pictures on social networks) (5) Mislead-
ing content: misleading use of genuine information
(e.g., partial quotes or cropped pictures) (6) False
connections: When headlines, visuals, or captions
do not support the content (e.g., “clickbait” con-
tent) (7) Satire and parody: information clearly
marked as false but presented in a way as if it were
the truth.
As highlighted in the list, this paper focuses pri-
marily on manipulated content and our tool MAN-
ICOD does not aim to detect other types of mis-
and disinformation. We also do not attempt to fur-
ther classify whether the manipulated content is
misinformation or disinformation, as such classifi-
cation would require a tool to infer the motivation
behind the production and dissemination of false
information, which is a different research area.
We also assume the mis- and disinformation cir-
culating on the Internet is unlikely completely fab-
ricated without being partially backed by some
1Available online at https://github.com/cyberooo/manicod for
anonymous review.
3

truth (Kaliyar et al., 2021; Gravanis et al., 2019).
Instead, we focus on disinformation in the form of
a text that contains minor or indistinguishable per-
turbations, with all remaining parts being truth to
circumvent mainstream fact-checkers (Mazurczyk
et al., 2024).
2.2 Problem Definition
As discussed in §2.1, we focus on detecting false in-
formation that is derived from genuine information
by adding minor or even indistinguishable pertur-
bations, rather than completely ungrounded fabri-
cated content. More specifically, while genuine
information can be manipulated in arbitrary ways,
we consider the two most common types of trans-
formations in practice are sentiment reversal and
context alteration (Alibaši ´c and Rose, 2019).
Given a truth about a real event as the input, writ-
ten as a pair of text and veracity ⟨x,true⟩,sentiment
reversal transforms it into a piece of disinforma-
tionxnby negating the sentiment, while context
alteration aims to identify the key components that
constitute the context of the input statement (e.g.,
appointment and name of persons, date and time,
geography, numbers, and quantity) and replace at
least one of them either manually or through sophis-
ticated AI techniques, resulting in a piece of disin-
formation xc. The two examples illustrated in Fig-
ure 1 are produced by context alteration, among
which the first one has the country name been tam-
pered with and the second one has the population
of death cases exaggerated by 100 times.
We regard manipulated content detection as an
automatic procedure that takes a piece of text from
the user as the input, written as x, and then outputs
a label y∈ {false,true}, which corresponds to a
“manipulated” or “truthful” decision. Suppose there
is a dataset containing a set of true statements X, a
group of manipulated content created by reversing
the sentiments of true statements written as Xn, and
a group of manipulated content created by altering
context Xc, an ideal manipulated content detector
fshould be able to precisely predict the veracity of
these statements. We define this process as follows:
f:x→y|((∀x∈X, y =true)
∨(∀x∈ {Xn, Xc}, y=false ))(1)
In addition to predicting the veracity, we note that
an auditable manipulated content detector should
ideally be capable of explaining or justifying the
predicted veracity according to the knowledge ei-
ther learned during the training process or citedfrom reliable sources on the Internet. In this work,
we assume a satisfactory explanation should at least
identify the words or phrases that constitute the
manipulation and consequently result in the false
information. Especially for the cases produced by
context alteration, we expect the detector could pro-
duce an explanation containing the key item that
has been altered, even if that key item does not
appear in the user input (e.g., “Israel” in the first
running example in Figure 1).
2.3 Related Work
Textual Veracity Assessment by Conventional
ML Models . The detection of manipulated content
has been studied in the research community for
decades. Early approaches mostly leverage conven-
tional ML classification to predict if a statement is
true or false either through the lens of claim verifi-
cation , which describes the tasks to determine the
veracity of a textual claim through a list of relevant
comments or evidence.
For example, a piece of false information is
widely regarded to have obvious differences with
truth in terms of salient phrases (Popat et al., 2018;
Wu et al., 2021) or news attributes (Yang et al.,
2019). In addition to merely making predictions,
the research community has also put persistent ef-
fort into producing explanations. Shu et al. (Shu
et al., 2019) adopts a hierarchy neural network
model to evaluate the veracity of a claim and mean-
while nominate the top- kcheck-worthy sentences
from the accompanying comments for explanation
purposes, thereby achieving explainable fake news
detection. Ma et al. (Ma et al., 2019) leverage hi-
erarchical attention networks to propose sentence-
level embedding for claim verification, and accord-
ingly highlight the embedding suspicious of false-
ness for explanation purposes. Lu and Li (Lu and
Te Li, 2020) adopt a graph neural network that en-
codes Twitter posts and comments to determine
whether the posts on social networks are fake news.
Nie et al. (Nie et al., 2019) investigate the usage of
neural semantic matching networks in fact extrac-
tion and veracity verification.
In summary, prior works that resort to conven-
tional ML classifiers have a common limitation as
they can only evaluate the veracity of a textual in-
put along with curated supporting documents, and
their explainability is often restricted to nominating
some of the most relevant documents or sentences
that still require users to manually connect the dots,
which is far from a practical detector for zero-day
4

West Nile virus found in mosquitoes in York Region.West Nile virus foundinmosquitoes...
qhttps://www.cdc.gov› west-nile-virus › data-mapsData and Maps for West Nile | West Nile Virus | CDCqhttps://www.nyc.gov› site › doh › about › press › ...West Nile Virus Detected in Record Number of Mosquitoes Two …qhttps://www.usatoday.com› story › news › health › ...USA Today:The 25-year fight to defeat West Nile virus, one convoy at a time.……
…
…
Toanalyzetheinputstatement,Iwill......Iconcludethatthestatementreflectsthetruthratherthanbeingapieceoffakenewsthatcouldmisleadreaders.Decision:TrueRetrieval
"<<SYS>>YouareanAIassistantforchecking*manipulatedcontent*,i.e.,fakenewsfabricatedbasedonatruenewsstory.Youneedtoanalyze...Pleasestrictlyfollowthisruleinyouroutput.<</SYS>>[INST]Theinputstatementis:{statement}[/INST]"
…SearchResultsArticlesinChunksVectorized ChunksVectorDatabasePromptTemplatePhase1Online Knowledge RetrievalPhase2Knowledge-Grounded InferenceFigure 2: An overview of our disinformation detection framework M ANICOD
manipulated content.
Disinformation Detection with LLMs . After the
debut of the BERT model family, the research com-
munity started exploring the adoption of LLMs in
fact-checking and disinformation detection in gen-
eral. Lee et al. (Lee et al., 2021) train a few BERT-
based LLMs for fact-checking, which are evaluated
on two new COVID-related datasets, demonstrat-
ing the powerful capabilities of LLMs in claim
reasoning for the first time. Wang et al. (Wang and
Shu, 2023) leverage the chain-of-thought mecha-
nism of LLM for fact verification. Specifically,
it allows users to provide relevant documents as
the context through the conversation with LLMs,
and then resorts to the backend LLMs to reason
the veracity based on the given context. Wang et
al. (Wang et al., 2024) propose a defense-based ex-
plainable fake news detection solution, which uses
an evidence-extraction module to split the relevant
knowledge, referred to as “wisdom of crowds” in
the literature, into two competing parties and lever-
ages LLM to reason separately for the veracity of
the input claim. By doing so, it significantly re-
duces the reliance on the quality of knowledge and
minimizes the impact of occasional inaccurate or
biased information from the knowledge.
However, the effectiveness of existing LLM-
based approaches is limited to scenarios where
ground truth data is available, either from the in-
trinsic knowledge base of the LLMs or supplied
by users manually. Furthermore, considering all
mainstream LLMs can only take a limited num-
ber of tokens to build the context, it is techni-
cally hard to supply a large amount of knowledge
through in-context learning (Wang and Shu, 2023).
Niu et al. (Niu et al., 2024) address this challenge
through LLM based retrieval-augmented detection
but rely on fine-tuning open-sourced models or
closed-sourced ones like GPT, hindering reproduc-
tion without powerful hardware infrastructure or
financial budget. To the best of our knowledge,our approach is the first of its kind to achieve auto-
mated large-scale context sourcing and embedding
based on a pre-trained open-sourced LLM, making
it a fully autonomous detector for zero-day manip-
ulated content.
Fairness Evaluations . Combating disinformation
is a systematic work and existing solutions are also
evaluated by other research fronts (e.g., usable se-
curity). We discuss these works in Appendix A.
3 Our Approach
A summary of the proposed MANICOD is shown
in Figure 2. MANICOD adopts a two-phase work-
flow, namely the online knowledge retrieval phase
and the LLM-based knowledge-grounded inference
phase. We detail these two phases in the rest of this
section.
Online Knowledge Retrieval . In the first phase,
our approach takes users’ claims as the input and
searches them via mainstream search engines (e.g.,
Google) to source relevant information. Although
MANICOD can detect manipulated content gener-
ated based on historical events, our approach aims
to also detect zero-day manipulated content on re-
cent even ongoing events, for which analyzing it
merely based on the knowledge of LLM or relying
on an additional training process with an off-the-
shelf dataset is infeasible. For that reason, this step
aims to source a significant number of relevant doc-
uments autonomously and construct a context out
of the retrieved information. The context contains
necessary knowledge about the input claim to fa-
cilitate later inference and justification. While we
try to crawl from credible sources only (e.g., Alexa
top 1 million), we remark that documents released
by even reputable sources, may still turn out to be
false or controversial on a later date. For example,
The New York Times published 278 corrections for
its online news in September 2024 (Times, 2024).
To simplify our study, we assume the knowledge
retrieved from the Internet is true.
5

Implementation-wise, we resort to SerpAPI (Ser-
pApi, 2024) to search the users’ claims on Google
and collect the top- ksearch results from the search
list, i.e., the URLs, based on the relevance ranking
performed by Google. We then adopt Beautiful-
Soup (Richardson, 2024) to crawl the text from the
URLs. The crawled text of each webpage will be
saved locally as a document. In case some websites
have applied anti-crawling technologies, which we
did experience with a few notable global news
websites, e.g., Reuters, we skipped that URL and
sequentially moved to the next one to ensure we
source the kmost relevant webpages to construct
the context.
After the documents are saved, we prepare for
feeding them to the next phase: LLM-based in-
ference. Intuitively, we aim to pass all collected
documents as a whole, together with the user’s
input claim, to the conversation with LLM to con-
struct the context. This process is referred to as
in-context learning and has been adopted in (Wang
and Shu, 2023). However, MANICOD differs from
prior work as it is designed to run autonomously
with self-crawled raw textual content instead of
well-organized off-the-shelf processed documents.
It may contain too much irrelevant information,
such as navigating web components and advertise-
ments, mixed into the actual contents that MANI-
COD looked for. This irrelevant content will over-
whelm the LLM to exceed the strict token limita-
tion and threat the scalability of our approach if a
larger number of relevant web pages is required.
To tackle this challenge, we adopt the RAG tech-
nique to encode the relevant information into a
vector database. Specifically, we resort to an open-
sourced toolkit named LangChain (LangChain-
AI, 2024), with nomic-embed-text (Nomic, 2024)
model to vectorize the plain text of the documents,
and ChromaDB (Chroma, 2024) as the choice of
vector database. Our approach is implemented in
Python and the RAG is realized based on the pop-
ular open-sourced and LLM-hosting framework
Ollama (Ollama, 2024) At the end of this phase,
the top- kavailable documents are vectorized into a
temporary database and augmented into the LLM’s
query session of M ANICOD .
Knowledge-Grounded Inference . With the RAG
set up, the second phase of MANICOD aims to de-
termine the veracity of the input claim by analyzing
the augmented context sourced from the Internet
and explaining the decision in natural language. We
note that MANICOD should work with any LLMssince the veracity inference is mainly based on the
external context rather than the intrinsic knowledge.
Still, we have conducted a small scale experiment
to compare the performance of our approach with
different mainstream state-of-the-art LLMs. We
find that the Meta’s Llama 3.1 8b Instruct performs
the best among three other models, and as a result,
we adopt Llama 3.1 in this phase. We present more
details about the model selection in Appendix B to
save space.
Intuitively, we shall only need to draft a prompt
template containing an instruction specifying the
requirement causally (i.e., determining whether the
given claim is false) followed by the user’s input.
However, during our study, we find that requiring
LLMs to produce a reasonable decision and expla-
nation turns out to be a non-trivial process. During
our pilot experiment, we find that the adopted LLM
tends to produce biased decisions that favors the
user. For example, the LLM often determines the
claim as “false information” if we asked “whether
the claim is a piece of false information?” in the
instruction. Conversely, the LLM tends to answer
“truth” if we ask “whether the claim is a truth?”
We found this phenomenon becomes more severe
when we are testing with truth claims, from which
the LLM might struggle to identify the trace of any
factual mistakes and eventually chose to favor the
user’s will (Naveed et al., 2024).
To tackle this issue, we deliberately designed our
prompt template to minimize LLM’s bias in always
producing decisions that might favor the user. We
instruct the LLM to reason the input statement first
(e.g., what a piece of manipulated content looks
like, and in which case a true or false decision
should be produced) and make the binary predic-
tion in the end, to minimize the occurrence that the
adopted LLM makes the prediction merely based
on guessing the user’s intention. The creation of
prompt template used for MANICOD is a continu-
ous tuning process until we find an optimal balance
of true and false predictions, i.e., we repeatedly
instruct LLMs with “hard” cases that tend to result
in different predictions until we find a prompt that
is stable and yields a reasonable performance. The
final prompt can be found in Figure 3.
4 A Dataset of Manipulated Content
AsMANICOD is a detector specialized for manipu-
lated content, its effectiveness is best evaluated by
a dataset that simulates how real-world news head-
6

<<SYS>> You are an AI assistant for checking *manipulated content*, i.e., fake news fabricated based on a true news story. You need to analyze the input statement, which is a news headline, and determine if the headline is true or contains manipulated content based on your knowledge and context.Remember, you *don't need to prove the statement is true*. The statement may not be *complete* in all contexts of an event. So, do not determine the statement as a piece of manipulated content if it *does not mention something*. Instead, you should figure out if there are any contradictions, suspicious alterations, or factual errors in the statement to make it manipulated content.You need to give justification to explain your answer especially when you think it is a piece of manipulated content. At the end of your response, you need to provide a decision that is limited to two options only: 'True' and 'False', i.e., the statement is 'True' for truth, or 'False' for manipulated content.Specifically, you should respond 'False' if you can find *any* contradiction that opposes the statement.You should respond 'False' if you find any factual mistake that is unlikely true based on your knowledge.You should respond 'False' if you find any inconsistent context in the input statement compared with your knowledge, especially the context information such as number, quantity, person and location.For example, response 'False' if you know the statement is partially true but contains wrong key context, such as the event happened in a different place or on a different date, the action was done by a different person, or the amount is in a different value.You should respond 'True' if you believe that statement reflects the truth rather than being fake news that could mislead readers.You should always respond 'True' if you cannot find evidence or clues to support the statement as manipulated content.Remember, *do not* determine a statement as manipulated content just because you are not confident or because you don't know much about the statement.You cannot respond 'False' just because the statement does not contain all key context or information as what is in your knowledge.Your response should always begin with your analysis and justification, and then provide your decision in the end.Your output *must* end with your decision in a *single word*, i.e., 'True' or 'False'. Do not add anything once you provide the finaldecision! Please strictly follow this rule in your output. <</SYS>>[INST] The input statement is: {statement} [/INST]Figure 3: The prompt template used in MANICOD with
task description, key inference rules, and output instruc-
tions highlighted in red, blue, and green, respectively.
ElonMuskdeniesreported$45millionamonthpledgetoTrump.ElonMuskconfirmsreported$45millionamonthpledgetoTrump.ElonMuskdeniesreported$45millionamonthpledgetoBiden.
ManualReview&AlterationRawRSSentries
RSS
titlesummaryStatementFilteringContextExtraction
original:negation:context:id:475073f7b3376a8a8f51da…source:GoogleNews(USsite)date:24Jul2024
Original(X)Negation(Xn)
ProcessednewsLongitudinalCollectionAutomaticProcessingSentimentReversal
Context(Xc)ElonMuskdeniesreported$45millionamonthpledgetoTrump.ElonMuskconfirmsreported$45millionamonthpledgetoTrump.(ElonMusk),($45million),(month),(Trump).Dataset
Figure 4: An overview of our dataset creation
lines will be manipulated in reality. However, to the
best of our knowledge, most existing open-access
datasets are either designed for checking trivial
facts (Google, 2021) that can be determined using
LLM intrinsic knowledge or limited in the context
of social network posts (Thorne et al., 2018). There
does not exist an off-the-shelf dataset of manipu-
lated real-world events that suits the objective of
MANICOD to reflect everyday zero-day fake news.
We thus propose a dataset for manipulated content
for evaluation purposes and also to benefit future
works of similar purposes.
Our dataset is created following the pipeline
in Figure 4. The first step is to collect news head-
lines longitudinally. Specifically, we searched no-
table English news websites with free RSS sub-
scriptions and identified four sources: GoogleNews, BBC, New York Times, and Fox News, from
which all news entries are crawled. For each entry
in the RSS feed, we concatenate the news title and
summary (if available) as the news headline. We
also collect the date of the news being released and
the region or countries of the news site into the
dataset to help customize the online searching (i.e.,
Phase 1 in §3). Our crawling started on 24 July
2024 and lasted for 20 days.
Next, we resort to the same LLM used in our
knowledge-grounded inference (see Phase 2 in §3)
to process the collected RSS entries with three steps
automatically: (1) We instructed the LLM to fil-
ter out news headlines that are not informative or
self-contained to be a claim or a statement that is ap-
plicable for veracity assessment, such as a question
sentence “ Are we in a summer COVID wave? ”2or
an incomplete statement “ Here are the Daily Lotto
numbers .”3(2) We then asked the LLM to produce
a negation of the news headline by identifying and
reversing the sentiment in the text. The negations
of the news headlines will constitute the negation
set (Xnin §2.2). (3) We also leverage LLMs to
extract the key contexts from the news headline,
including persons’ names and titles, geographical
terms such as country, state, and city, quantity, and
units. These extracted contexts will be used as
ingredients to be added back to the original news
headlines to form a separate set of manipulated con-
tent by context alteration (i.e., Xc). We present the
detailed prompts involved in automatic processing
in C.1.
Finally, we manually review and adjust the con-
textual information to ensure the authenticity of the
dataset. Specifically, three team members with pre-
vious experience in combatting manipulated con-
tent or reading news daily manually reviewed all
the generated news headlines and replaced the orig-
inal context with the altered ones to produce sim-
ulated fake news containing manipulated content.
For the generated negations, our manual review fil-
tered out inapplicable news or improper negations
generated by LLMs. A typical example inapplica-
ble negation is provided in Appendix C.2.
We remark that manual effort is necessary for
context alteration to maintain a high quality dataset
because we need to ensure the fake news with al-
tered context contradicts the ground truth. For
example, given the original news headline “ At
2A news headline from BBC Health site on 31 Jul 2024.
3A news headline from Google News South Africa on 28 Jul
2024.
7

Canada (230)
9.2%Singapore (214)
8.6%Australia (205)
8.2%South Africa (176)
7.0%New Zealand(147)
5.9%
UK (533) 21.3%
US & Worldwide (995)39.8%Google News
(1444)
17.8%BBC (443)
13.1%New York
Times (328)
11.4%
Fox News (285)57.8%
Figure 5: Distribution of the 2,500 news collected by
regions (Left) and by news providers (Right)
least 150 people have been killed in Bangladesh
protests ”4, the production of fake news by altering
the casualty number must be a number greater than
150, otherwise, the altered version should still be
considered as truth. Our manual effort also ensures
the logical correctness of the manipulated content
and prevents the altered content from becoming
another truth. For example, given a news headline
“Ukraine wins its first medal in Paris Olympic ”5,
replacing the word “Ukraine” with a few widely
discussed countries like USA or China would not
make it manipulated content. Instead, a good can-
didate to replace the country name in the given
context would be a country without any medal won
from the Olympics (e.g., Mexico), or a country that
has its first medal won only after the date of the
news (e.g., Singapore, with its first medal won on
a later date on 9 Aug).
As a result, we collect 2,500 valid news head-
lines. The distribution of the news collected is
presented in Figure 5. The original version of
these news headlines will constitute the truth set
(X) in our dataset. For the manipulated content
set, we managed to produce 2,243 and 2,027 fake
news containing manipulated content by sentiment
reversal ( Xn) and context alteration ( Xc), respec-
tively. We will present our evaluation of this dataset
in §5.1. Due to the space limitation, we provide
more details of the context alteration and discuss
potential bias in Appendix C.3.
5 Evaluation
In this section, we report the performance of
MANICOD . To explore the effectiveness of the
knowledge-grounded inference, we try to answer
the following research questions (RQs):
•RQ1: How does MANICOD perform on our pro-
posed manipulated contents dataset?
4A news headline from NYT World news on 26 Jul 2024.
5A news headline from Google News Australia on 30 Jul 2024.•RQ2: How does augmenting real-time knowl-
edge help in identifying manipulated contents?
•RQ3: How robust is MANICOD in detecting
other misinformation, as presented in §2.1?
Experiment Settings . We set the temperature of
the adopted LLM to 0.1for all the testing involved
in this work to maintain the consistency of LLM
outputs and reproducibility. However, we remark
that the decision produced by the adopted LLM,
i.e., the claim is truth or a piece of disinformation,
may still change occasionally. For that reason, we
repeated the testing of each claim three times and
recorded the majority results.
For the scale of online knowledge retrieval, i.e.,
the selection of k, we performed small-scale pilot
tests and experimentally found that k= 3 best
balances veracity prediction performance and exe-
cution time. We implement the RAG with Recur-
siveCharacterTextSplitter (LangChain, 2024). We
follow the default settings of the official tutorial
and set the chunk size and overlap to 100 and 20,
respectively. The number of retrieved chunks6is
set to 5.
During the pilot testing, we also observed that
the LLM produced non-conclusive results (e.g., “I
don’t know”, at approximately 5% occurrences).
We adopt a conservative stance and treat all non-
conclusive results as wrong, i.e., non-conclusive
results over truth are treated as disinformation
and non-conclusive results over disinformation are
treated as truth.
Lastly, we stipulate that a successful detection
needs to precisely identify the manipulated content
to make it explainable. Therefore, we only accept a
correct prediction of disinformation caused by con-
text alteration (i.e., claim from Xc) ifMANICOD
simultaneously mentioned the original context and
its replacement in its output.
5.1 RQ1: Veracity Prediction Performance
After evaluating the proposed manipulated con-
tent dataset, our approach achieved promising re-
sults, with an overall 79.5% precision and 92.6%
recall. The F1 score of our evaluation is 0.856.
6We note that the number of retrieved chunks is also referred
totop-kin the LangChain documentation, which indicates the
number of chunks to be retrieved from the vectorized database
and consequently appended to the context of user input. It
is different from the kthat we have mentioned earlier, i.e.,
number of documents to be retrieved from the online search
results.
8

We present the confusion matrix of the evaluation
in Appendix F.
Being specific to each type of claim, our ap-
proach performs the best on verifying manipulated
content by reversed sentiment, i.e., Xn, evidenced
by the accuracy of 93.9%, followed by manipula-
tion by context alteration ( Xc) and original head-
lines ( X), with accuracy at 91.0% and 65.7%, re-
spectively. Although MANICOD achieves promis-
ing overall results, we find the veracity prediction
of the original news headlines worse than the re-
maining two types of claims. We suspect such
unexpected results may be because the news’ or
the LLM’s intrinsic veracity is not completely true.
The online searching performed at a later moment
than the news being drafted may further amplify the
gap between the news and ground truth. For exam-
ple, a piece of news that contains “ Prime Minister
Keir Starmer ”7is considered as False as “Rishi
Sunak has been serving as the Prime Minister of
the United Kingdom since October 2022, and Keir
Starmer is actually the leader of the Labour Party
in the UK”, even the search results also highlights
that the current UK Prime Minister is Keir Starmer.
Finally, we also record the time lapsed during the
evaluation. We present more details in Appendix D
to save space.
5.2 RQ2: Ablation Study
In addition to evaluating MANICOD on our manipu-
lated content dataset, we also conducted an ablation
study on the effect of the knowledge-grounded in-
ference based on online retrieval. Specifically, we
compare our approach with the two popular LLMs:
the LLM adopted in MANICOD named Llama3.1
(8b), and GPT-4o-mini, known as the most pow-
erful proprietary LLM at the moment of this pa-
per being drafted. Both LLMs are tested with the
same dataset but without augmenting the context
retrieved from the Internet. Given the latest data
trained for the two LLMs are far earlier than the
news included in our dataset at this moment,8we
can assume both LLMs can only predict the claims
jointly based on the outdated knowledge and the
language logic in the text.
The results of our ablation study are presented
in Table 1. We can observe both LLMs without
the context constructed from online knowledge per-
form poorly on all three categories of claims. This
7A news headline collected from Fox News on 5 Aug 2024.
8The knowledge cutoff dates of the Llama 3.1 (8b) and GPT-
4o-mini models are Dec. 2023 and Oct. 2023, respectively.Table 1: Comparison of our approach with LLMs with-
out online knowledge retrieval
Accuracy
Original ( X) Negation ( Xn) Context altered ( Xc)
Llama 3.1 (direct) 43.4% 58.8% 69.4%
GPT-4o (direct) 57.9% 56.5% 66.6%
Ours (w. Llama 3.1) 65.7% 93.9% 91.0%
Table 2: Veracity prediction results on binary-class fact-
checking tasks, with our approach compared with the
state-of-the-art approaches
COVID (Google, 2021) FEVER (Thorne et al., 2018)
Accuracy F1 score Accuracy F1 score
Traditional approaches
XLNet ft(Lee et al., 2021) 0.632 0.520 0.492 0.484
Perplexity-based LLMs
BERT PPL(Lee et al., 2021) 0.625 0.611 0.574 0.569
GPT2 PPL (15b)(Lee et al., 2021) 0.783 0.776 0.736 0.717
Ours 0.873 0.911 0.807 0.788
implies the effectiveness of our two-phase design
for real-time manipulated content detection.
5.3 RQ3: Comparison to Other Methods
Our proposed dataset contains only slightly manip-
ulated content based on real-world events. That
makes the performance of MANICOD in handling
other types of disinformation unanswered. For that
reason, we benchmarked our approach with exist-
ing datasets for claim verification and fact-checking
and compared the performance of MANICOD with
previous research.
Our approach is firstly benchmarked on two
datasets designed for binary label fact-checking
tasks, namely COVID-Scientific (Google, 2021)
and FEVER (Thorne et al., 2018). The former col-
lects COVID-19-related myths and truths labeled
by reliable sources including WHO and CDC, and
the latter is designed for a fact extraction and verifi-
cation task containing a large number of erroneous
information by altering sentences on Wikipedia.
We consider claims from both datasets to be suffi-
ciently verified by online information, and there-
fore, we do not consider the evidence provided in
the datasets. We retrieve the existing literature and
compare our approach with the best performing ex-
isting approach (Lee et al., 2021). The comparison
with previous work that benchmarked these two
datasets are presented in Table 2.
In addition to binary labeled tasks, two more
datasets designed for claim verification are consid-
ered in our comparison, namely RAWFC (Yang
et al., 2022) and LIAR-RAW (Wang, 2017). The
RAWFC dataset is a collection of claims sourced
from Snopes (Snopes Media Group, 2024) labeled
either true, false, or half-true. The LIAR dataset
contains over 12,000 fine-grained claims collected
9

Table 3: Veracity prediction results on multi-class claim
verification tasks, with our approach compared with the
state-of-the-art in different types and the best results of
existing work are shown in bold font
LIAR-RAW (Wang, 2017) RAWFC (Yang et al., 2022)
Precision Recall F1 score Precision Recall F1 score
Traditional approaches
dEFEND (Shu et al., 2019) 0.231 0.186 0.175 0.449 0.433 0.441
SBERT-FC (Kotonya and Toni, 2020) 0.241 0.221 0.222 0.511 0.459 0.455
GenFE-MT (Atanasova et al., 2020) 0.186 0.199 0.152 0.456 0.453 0.451
CofCED (Yang et al., 2022) 0.295 0.296 0.289 0.530 0.510 0.511
LLM-based approaches
LLaMA2 claim(Ouyang et al., 2022) 0.171 0.174 0.151 0.373 0.380 0.368
ChatGPT claim(Ouyang et al., 2022) 0.254 0.273 0.251 0.477 0.486 0.444
FactLLaMA know(Cheung and Lam, 2023) 0.325 0.321 0.304 0.561 0.555 0.557
Defense-based approaches with LLM Reasoning
L-Defense LLaMA2 (Wang and Shu, 2023) 0.316 0.317 0.314 0.610 0.600 0.601
L-Defense ChatGPT (Wang and Shu, 2023) 0.306 0.322 0.305 0.617 0.610 0.612
Ours 0.920 0.910 0.915 0.821 0.932 0.873
from PolitiFact (The Poynter Institute, 2024), la-
beled by six options based on the degree of false-
ness, i.e., true, mostly true, half-true, barely true,
false, or pants-on-fire. As MANICOD is designed
to only produce binary decisions, our comparison
does not cover claims with ambiguous or controver-
sial veracity. Specifically, our comparison excludes
claims with the two neutral labels, i.e., half-true
and barely true, from the two datasets. Unlike
the comparison with COVID and FEVER datasets,
we mock the online retrieval process by directly
augmenting the evidence provided by RAWFC
and LIAR-RAW datasets to realize the knowledge-
grounded inference. Still, we review the relevant
literature (Shu et al., 2019; Kotonya and Toni, 2020;
Atanasova et al., 2020; Yang et al., 2022; Ouyang
et al., 2022; Cheung and Lam, 2023; Wang and
Shu, 2023) and compare MANICOD with the repre-
sentative and state-of-the-art existing approaches
in different types. We show our results in Table 3.
As the results show, our approach outperforms
existing approaches on the four datasets all the
time. Compared with the best performing exist-
ing approaches on each task, MANICOD can bring
an improvement of the F1 score up to 17.3% for
binary-label fact-checking tasks (i.e., 0.911 versus
0.776), and an up to 1.9x improvement of the F1
score for multi-label claim verification tasks (0.915
versus 0.314).
We remark that our approach is primarily de-
signed to detect manipulated contents based on
widely accepted ground truth and consensus on the
Internet despite the benchmarking showing impres-
sive results on datasets of fabricated fake news or
rumors such as Snopes9. This, again, demonstrates
9The official FAQ of Snopes (Snopes, 2025) writes that the
verification is jointly realized by “attempting to contact the
source of the claim for elaboration and supporting informa-
tion” and “attempting to contact individuals and organizations
who would be knowledgeable about it.” Its data may containthe effectiveness of knowledge-grounded inference
and reveals a promising direction in relevant re-
search. We also note that our benchmarking does
not claim an exhaustive comparison of existing
approaches and datasets but focusing on represen-
tative ones that have been widely referred to in
prior research.
6 Conclusions
We propose MANICOD , an LLM-based manipu-
lated content detector. Our approach can automati-
cally search and retrieve knowledge from the Inter-
net and, therefore, can analyze the veracity of zero-
day manipulated content about recent real-world
events. MANICOD also leverages the latest open-
sourced LLM to perform a knowledge-grounded
inference and to provide explanations in natural
language to justify its decisions. We also create
a manipulated content dataset dedicated to evalu-
ating MANICOD as well as future research efforts
on combating manipulated content. Our evaluation
shows that MANICOD can effectively detect zero-
day manipulated content evidenced by an overall F1
score of 0.856. A comparison with prior work also
demonstrates MANICOD ’s superior performance in
diverse fact-checking and claim-verification tasks.
Our work advances online safety research and we
hope MANICOD can inspire future efforts on lever-
aging powerful AI technologies in pursuit of social
goodness.
7 Limitations
We find a few limitations that may threaten the
effectiveness of our approach. We categorize the
identified limitations into internal and external ones
and discuss them in this section.
Internal Limitations . First, the online knowledge
retrieval of MANICOD directly searches the user’s
claim through a search engine. Its performance can
be downgraded when dealing with lengthy claims
that exceed the word count limit or contain com-
plicated contexts that the search engine may not
find the best-suited information straightforwardly.
To tackle this challenge, distilling critical contexts
from the original claim as the search keywords may
be a promising direction for future improvement.
Another potential mitigation is decomposing the
original long claim into several atomic statements,
followed by separate online searching and knowl-
edge augmentation. This has been shown effective
fabricated contents that can only be clarified by those involved.
10

in veracity detection in (Niu et al., 2024). We aim
to explore this in future work.
Second, while the manipulated news in the pro-
posed dataset is carefully crafted to be factually
false, it may not fully reflect the complexity of real-
world disinformation. In practice, fakesters care-
fully blend partial truths with misleading contexts,
making detecting them more challenging. These
real-world manipulations may also follow differ-
ent distribution patterns, which our datasets may
not adequately represent. Moreover, fake news in
real-world scenarios may be especially designed
to evade detection by emerging AI-based detectors
and may have the potential to deceive the LLM into
making incorrect decisions.
Third, we find that the performance of our ap-
proach heavily relies on the choice of LLMs and a
high-quality prompt template that fits the adopted
LLM. The tuning of an effective prompt template
is found to be a non-trivial and time-consuming
task. For example, we experimentally found that
the adopted Llama 3.1 model is more sensitive to
terms like “rumors” and “fake news” rather than
“manipulated content” and “disinformation.” For
this reason, we have to explicitly ask the LLM to de-
termine if the input claim is “fake news” although
we are not meant to restrict our scope to this. More-
over, we find a fine-grained prompt designed for
the adopted LLM does not imply its effectiveness
for other LLMs, which may threaten the extensi-
bility of our approach. How to effectively instruct
LLM to perform complex tasks canonically and
generically still awaits our further exploration.
Last, we observe that many false predictions are
caused by the mistakes made by LLMs during in-
ference. For example, although we adopted one of
the most powerful LLMs, it may still face limita-
tions in certain tasks, including text comprehension,
value comparison, and chronological order reason-
ing. Similarly, it was trained with potential bias,
incorrect, or outdated data that may be intrinsically
prone to generating incorrect results. We shall re-
sort to the future improvement of LLMs with more
powerful reasoning capabilities to mitigate this con-
cern. We provide a case study for more details of
this discussion in Appendix E.
External Limitations . The veracity of news head-
lines from our proposed dataset inspires us to
discuss the potential external limitation that may
threaten MANICOD ’s validity. First, we learn that
our collected news, even sourced from reputable
websites, may still turn out to be mis- or disinfor-mation, resulting in potential false negative cases
(i.e., the original news is assumed to be true but is
proven to be false according to online knowledge)
in our evaluation. Besides that, news about the
same event but mentioned on different web pages
may conflict with each other, leading to failed rea-
soning. A representative example could be the
identity of the shooter in the accident of Donald
Trump in 2024, for which a piece of fake news
saying the shooter is Asian has been widely spread-
ing on the Internet (April Xu, 2024). While our
research does not focus on how news should be pro-
duced, we can maintain a pool of reputable media
to allow online searching only from their domains
to maximize the reliability of the online knowledge
retrieved. Although no news website is entirely
free of inaccuracies.
Another limitation that affects MANICOD ’s per-
formance is the veracity dynamics of news. During
the evaluation, we noticed that knowledge on the
Internet may keep changing over time (e.g., the
number of causalities may increase, and celebrities’
claims may suddenly change). Although our online
search simulates searching on the day the news is
released, the web pages with the same URLs may
keep updating, so the ground truth may not always
be stable. Biden’s decision to quit the 2024 elec-
tion would be a notable example to highlight this
limitation, as he kept claiming he would continue
running for the next president until his announce-
ment on 22 July. To mitigate the impact of such
limitation, we shall keep updating the datasets with
the latest news and evaluate our approach only with
the latest data to avoid the unprecedented change
of veracity.
8 Ethics Considerations
Our paper focuses on the detection of manipu-
lated real-world fake news with the aim of en-
hancing the robustness of current large language
models (LLMs) in identifying zero-day disinfor-
mation—fake news that has not been previously
encountered. We aim to develop an accessible solu-
tion that enables ordinary individuals to easily ver-
ify the authenticity of the content they encounter.
However, we recognize the challenge that fake
content creators may adapt and devise countermea-
sures. Still, compromising our system is non-trivial
and would require them to infiltrate major news
sources.
11

References
Haris Alibaši ´c and Jonathan Rose. 2019. Fake news
in context: Truth and untruths. Public Integrity ,
21(5):463–468.
April Xu. 2024. New york post falsely claims chinese
man shot trump chinese communities outraged.
https://www .msn.com/en-us/news/politics/
new-york-post-falsely-claims-chinese-man-
shot-trump-chinese-communities-outraged/
ar-BB1q5Ea9 .
Pepa Atanasova, Jakob Grue Simonsen, Christina Li-
oma, and Isabelle Augenstein. 2020. Generating fact
checking explanations. In Proceedings of the 58th
Annual Meeting of the Association for Computational
Linguistics , pages 7352–7364. Association for Com-
putational Linguistics.
Patrice Béchard and Orlando Marquez Ayala. 2024.
Reducing hallucination in structured outputs
via retrieval-augmented generation. Preprint ,
arXiv:2404.08189.
Canadian Centre for Cyber Security. 2024.
How to Identify Misinformation, Disinfor-
mation, and Malinformation (ITSAP.00.300).
https://www .cyber .gc.ca/sites/default/
files/misinformation-mesinformation-
itsap .00.300-en .pdf. Online; accessed 10 October
2024.
Canyu Chen and Kai Shu. 2024. Can llm-generated
misinformation be detected? In The Twelfth Interna-
tional Conference on Learning Representations .
Tsun-Hin Cheung and Kin-Man Lam. 2023. Factllama:
Optimizing instruction-following language models
with external knowledge for automated fact-checking.
In2023 Asia Pacific Signal and Information Pro-
cessing Association Annual Summit and Conference
(APSIPA ASC) , pages 846–853.
Chroma. 2024. Chroma - the open-source em-
bedding database. https://github .com/chroma-
core/chroma . Online; accessed 04 September 2024.
Google. 2021. Fact-checking - covid19-scientific.
https://github .com/google/BIG-bench/
tree/main/bigbench/benchmark_tasks/
fact_checker/covid19_scientific . Online;
accessed 04 September 2024.
Google. 2024. Fact Check Tools - Google Search.
https://toolbox .google .com/factcheck/
explorer . Online; accessed 07 October 2024.
Georgios Gravanis, Athena Vakali, Konstantinos Dia-
mantaras, and Panagiotis Karadais. 2019. Behind the
cues: A benchmarking study for fake news detection.
Expert Systems with Applications , 128:201–213.
Rohit Kumar Kaliyar, Anurag Goswami, and Pratik
Narang. 2021. Echofaked: improving fake news
detection in social media with an efficient deep neu-
ral network. Neural Computing and Applications ,
33(14):8597–8613.Neema Kotonya and Francesca Toni. 2020. Explainable
automated fact-checking for public health claims. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) ,
pages 7740–7754. Association for Computational
Linguistics.
LangChain. 2024. Recursively
split by character. https://
python .langchain .com/v0 .1/docs/modules/
data_connection/document_transformers/
recursive_text_splitter . Online; accessed 04
September 2024.
LangChain-AI. 2024. Langchain. https://
github .com/langchain-ai/langchain . Online;
accessed 04 September 2024.
Nayeon Lee, Yejin Bang, Andrea Madotto, Madian
Khabsa, and Pascale Fung. 2021. Towards few-
shot fact-checking via perplexity. arXiv preprint
arXiv:2103.09535 .
Yi Ju Lu and Cheng Te Li. 2020. Gcan: Graph-aware
co-attention networks for explainable fake news de-
tection on social media. In 58th Annual Meeting of
the Association for Computational Linguistics, ACL
2020 , pages 505–514. Association for Computational
Linguistics (ACL).
Jing Ma, Wei Gao, Shafiq Joty, and Kam-Fai Wong.
2019. Sentence-level evidence embedding for claim
verification with hierarchical attention networks. As-
sociation for Computational Linguistics.
Wojciech Mazurczyk, Dongwon Lee, and Andreas
Vlachos. 2024. Disinformation 2.0 in the age of
ai: A cybersecurity perspective. Commun. ACM ,
67(3):36–39.
Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad
Saqib, Saeed Anwar, Muhammad Usman, Naveed
Akhtar, Nick Barnes, and Ajmal Mian. 2024. A
comprehensive overview of large language models.
Preprint , arXiv:2307.06435.
Yixin Nie, Haonan Chen, and Mohit Bansal. 2019.
Combining fact extraction and verification with neu-
ral semantic matching networks. In Proceedings of
the AAAI conference on artificial intelligence , vol-
ume 33, pages 6859–6866.
Cheng Niu, Yang Guan, Yuanhao Wu, Juno Zhu, Jun-
tong Song, Randy Zhong, Kaihua Zhu, Siliang Xu,
Shizhe Diao, and Tong Zhang. 2024. VeraCT scan:
Retrieval-augmented fake news detection with justifi-
able reasoning. In Proceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 3: System Demonstrations) , pages
266–277, Bangkok, Thailand. Association for Com-
putational Linguistics.
Nomic. 2024. Introducing Nomic Embed: A Truly
Open Embedding Model. https://www .nomic .ai/
blog/posts/nomic-embed-text-v1 . Online; ac-
cessed 04 September 2024.
12

Ollama. 2024. Ollama. https://github .com/
ollama/ollama . Online; accessed 04 September
2024.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller,
Maddie Simens, Amanda Askell, Peter Welinder,
Paul F Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions with
human feedback. In Advances in Neural Information
Processing Systems , volume 35, pages 27730–27744.
Curran Associates, Inc.
Kashyap Popat, Subhabrata Mukherjee, Andrew Yates,
and Gerhard Weikum. 2018. Declare: Debunking
fake news and false claims using evidence-aware
deep learning. arXiv preprint arXiv:1809.06416 .
The Washington Post. 2024. Biden makes stunning
decision to pull out of 2024 race.
Reuters Fact Check. 2023. No evidence ‘rna technol-
ogy’ in chicken feed behind infertility or u.s. egg
shortage. https://www .reuters .com/article/
fact-check/no-evidence-rna-technology-in-
chicken-feed-behind-infertility-or-us-
egg-shor-idUSL1N34N1RJ/ .
Leonard Richardson. 2024. Beautiful Soup
Documentation. https://beautiful-soup-
4.readthedocs .io/. Online; accessed 04 Septem-
ber 2024.
SerpApi. 2024. SerpApi: Google Search API. https:
//serpapi .com/ . Online; accessed 04 September
2024.
Kai Shu, Limeng Cui, Suhang Wang, Dongwon Lee,
and Huan Liu. 2019. defend: Explainable fake news
detection. In Proceedings of the 25th ACM SIGKDD
international conference on knowledge discovery &
data mining , pages 395–405.
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela,
and Jason Weston. 2021. Retrieval augmentation
reduces hallucination in conversation. Preprint ,
arXiv:2104.07567.
Snopes. 2025. Snopes.com | frequently asked questions.
https://www .snopes .com/faqs/ . Online; accessed
07 October 2024.
Snopes Media Group. 2024. Snopes.com | the defini-
tive face-checking site. https://www .snopes .com.
Online; accessed 07 October 2024.
N.N. Taleb. 2010. The Black Swan: Second Edition:
The Impact of the Highly Improbable Fragility" . In-
certo. Random House Publishing Group.
The Annenberg Public Policy Center. 2024.
FackCheck.org - A Project of The Annenberg
Public Policy Center of the University of Penn-
sylvania. https://www .factcheck .org. Online;
accessed 07 October 2024.The Poynter Institute. 2024. Fact-checks | Politi-
Fact. https://www .politifact .com/factchecks/
list/ . Online; accessed 07 October 2024.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. In Proceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers) , pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
The New York Times. 2024. Corrections.
Bo Wang, Jing Ma, Hongzhan Lin, Zhiwei Yang,
Ruichao Yang, Yuan Tian, and Yi Chang. 2024. Ex-
plainable fake news detection with large language
model via defense among competing wisdom. In
Proceedings of the ACM on Web Conference 2024 ,
pages 2452–2463.
Haoran Wang and Kai Shu. 2023. Explainable
claim verification via knowledge-grounded reason-
ing with large language models. arXiv preprint
arXiv:2310.05253 .
William Yang Wang. 2017. “liar, liar pants on fire”:
A new benchmark dataset for fake news detection.
InProceedings of the 55th Annual Meeting of the
Association for Computational Linguistics (Volume 2:
Short Papers) , pages 422–426, Vancouver, Canada.
Association for Computational Linguistics.
Claire Wardle. 2020. Understanding Information Disor-
der: Essential Guides . First Draft.
World Risk Poll. 2020. ‘fake news’ is the num-
ber one worry for internet users worldwide.
https://wrp .lrfoundation .org.uk/news/
fake-news-is-the-number-one-worry-for-
internet-users-worldwide . Online; accessed 07
September 2024.
Lianwei Wu, Yuan Rao, Ling Sun, and Wangbo He.
2021. Evidence inference networks for interpretable
claim verification. In Proceedings of the AAAI con-
ference on artificial intelligence , volume 35, pages
14058–14066.
Fan Yang, Shiva K Pentyala, Sina Mohseni, Mengnan
Du, Hao Yuan, Rhema Linder, Eric D Ragan, Shui-
wang Ji, and Xia Hu. 2019. Xfake: Explainable fake
news detector with visualizations. In The World Wide
Web Conference , pages 3600–3604.
Zhiwei Yang, Jing Ma, Hechang Chen, Hongzhan Lin,
Ziyang Luo, and Yi Chang. 2022. A coarse-to-fine
cascaded evidence-distillation neural network for ex-
plainable fake news detection. In Proceedings of the
29th International Conference on Computational Lin-
guistics , pages 2608–2621, Gyeongju, Republic of
Korea. International Committee on Computational
Linguistics.
13

Mary Ellen Zurko. 2022. Disinformation and reflec-
tions from usable security. IEEE Security & Privacy ,
20(3):4–7.
A Extended Related Work on Fairness of
AI-based Fake News Detectors
Combating disinformation has been a widely recog-
nized challenge in today’s society. Agents of disin-
formation have learned that using genuine content
and reframing it in misleading but indistinguishable
ways is less likely to be detected by existing AI sys-
tems (Wardle, 2020). Another recent research by
Chen and Shu (Chen and Shu, 2024) also stress
that the AI-generated misinformation, compared
with human-written pieces, can be harder to detect
by human and even AI-based classifiers. Besides
that, the harm of disinformation is often amplified
during dissemination, where people may share it
on their social networks without realizing it is false
and even believing that it is helpful.
The study on disinformation involves not only
the computer science community but also politi-
cal, journalism, and socio-psychology researchers.
Zurko (Zurko, 2022) reviews the evolution of dis-
information and misinformation since the 1970s
and outlooks the technological development for
combatting disinformation from the usable security
perspective. Mazurczyk et al. (Mazurczyk et al.,
2024) share their opinions on leveraging AI tech-
niques in building a holistic solution to counter
disinformation.
B Selection of the Backend LLM
We performed a round of preliminary studies
on a smaller scale to compare the performance
of different LLMs (with similar sizes). Specifi-
cally, we evaluated the performance of four LLMs,
namely llama3.1:8b-instruct ,llama3:8b-instruct ,
gemma:7b , and mistral:7b , using the LangChain
RAG technique on the data collected on Aug 11.
The comparison of overall performance is shown
in Table 4.
Table 4: Comparison of different choices of the the
backend LLM (the best results are shown in bold font)
Models
(w. Knowledge
Cutoff)llama3.1:8b-
instruct
(Dec 2023)llama3:8b-
instruct
(Mar 2023)gemma:7b
(Not specified, not
later than Feb 2024)mistral:7b
(Not specified, not
later than Sep 2024)
Precision 0.846 0.776 0.698 0.775
Recall 0.941 0.930 0.842 0.980
Accuracy 0.891 0.846 0.764 0.865
F1 score 0.852 0.779 0.696 0.797
We observed that the four evaluated LLMs per-
formed similarly, and we chose the Llama 3.1
<<SYS>> I will give you a statement, you need to provide me with the negation of the statement. For example, when given 'Tom said it is a big success.', you should output 'Tom said it is a big failure.' Here is another example, when given 'The weather today is good.', you should output 'The weather today is bad.' Remember, try to make minimal modifications to the original statement. Try to use antonyms rather than merely inserting a 'not'. Only output the negative version of the statement. Output 'Not applicable' if there does not exist a negation. Please strictly follow this rule in your output. <</SYS>>[INST] The input statement is:{statement} [/INST]
IranianSupremeLeaderAliKhameneisaiditisIran’sdutyto“takerevenge”aftertheassassinationofHamasleaderIsmailHaniyehonWednesdayinTehran.(FoxWorldNews)IranianSupremeLeaderAliKhameneisaiditisnotIran'sdutytotakerevengeaftertheassassinationofHamasleaderIsmailHaniyehonWednesdayinTehran.
NewreportbytheLungCancerEducationandAdvocacyforPatientshighlightsroleofpatientadvocacyinprovidingbettercareforpatients.(GoogleNewsSingapore)NewreportbytheLungCancerEducationandAdvocacyforPatientshighlightslackofroleofpatientadvocacyinprovidingbettercareforpatients
Project2025toissueblisteringresponsetoHarris,othercriticsviadozensofindependentfactchecks:TheHeritageFoundation'sProject2025planstounveilaseriesoffactchecksaimingtobluntcontinuedcriticismofitsnearly-1,000pageframeworkforaconservativeadministration.(FoxPoliticsNews)Project2025tofailinrespondingtoHarris,othercriticsvialackofcrediblefactchecks:TheHeritageFoundation'sProject2025planstounveilaseriesofunconvincingfactchecksthatwillnotadequatelyaddresscontinuedcriticismofitsnearly-1,000pageframeworkforaconservativeadministration.(llama3.1:8b-instruct) in the paper for its best over-
all performance. Besides this, the results also re-
veal that the detection of Manicod mainly relies
on the natural language comprehension capabilities
of LLMs rather than their intrinsic knowledge of
recent events.
We have also attempted to perform the
knowledge-grounded inference on larger models,
e.g., 70b version of Llama3, and observed very
limited improvement with significant computation
resource usage and time sacrifice. Therefore, we
adopt Llama 3.1 (8b) to balance the performance
and cost of time and hardware resources.
C Additional Details of our Dataset
C.1 Prompts involved in Automatic
Processing
The prompt template used for generating the nega-
tion is shown below:
We then depict the negation generation with
three real-world news headlines crawled on 1 Au-
gust 2024 as follows:
The prompt template used for automatically iden-
tifying the key context is shown below:
We then demonstrate the process of context al-
teration with three real-world news headlines col-
lected on 25 July 2024, during which the LLM is
used to automatically extract the key context and
manual alteration is then involved to replace the
14

<<SYS>> I will give you a statement, you need to extract if there is any context about quantity, number, percentage, year, time, date, person name who is famous or a celebrity (e.g., Obama, Taylor Swift, etc.), country, city, or any other geographical concept from it. You also need to use parentheses '()' to enclose the extracted text. For example, when given 'Tom said he went to Los Angeles on last Saturday.', you should output '(Tom) said he went to (Los Angeles) on last (Saturday).'Another example, when given 'Families face £1,045 bill for summer holiday clubs: The cost of holiday provision has risen by 6% across Great Britain', you should output 'Families face £(1,045) bill for summer holiday clubs: The cost of holiday provision has risen by (6%) across (Great Britain)'. When given 'Barack Obama is the 44th president of the United States from 2009 to 2017.', you should output '(Barack Obama) is the (44th) president of (the United States) from (2009) to (2017).'Remember, try to make minimal modifications to the original statement. You should post a conservative stance, do not change if you are not sure. Do not mark a title (e.g., director, secretary), appointment name (e.g., Prime Minister), or entity name (e.g., government, university, etc.). Only output the statement with your parentheses. Output 'Not applicable' if you cannot find anything. <</SYS>>[INST] The input statement is:{statement} [/INST]
Satellitecapturesfirst-of-a-kindcloudimage:AjointEuropean-Japanesemissioncapturesaspaceviewoftheinternalstructureofacloud.(BBCScienceandEnvironmentNews)Satellitecapturesfirst-of-a-kindcloudimage:Ajoint(European)-(Japanese)missioncapturesaspaceviewoftheinternalstructureofacloud.(Japanese)->(Russian)Satellitecapturesfirst-of-a-kindcloudimage:AjointEuropean-Russianmissioncapturesaspaceviewoftheinternalstructureofacloud.
NepalPlaneCrashKills18PeopleAfterTakeoff:ThepilotoftheSauryaAirlinesflight,whichwasdepartingfromKathmandu,wastheonlysurvivor,officialssaid.(NewYorkTimesWorldNews)(Nepal)PlaneCrashKills(18)PeopleAfterTakeoff:Thepilotofthe(SauryaAirlines)flight,whichwasdepartingfrom(Kathmandu),wastheonlysurvivor,officialssaid.(18)->(180)NepalPlaneCrashKills180PeopleAfterTakeoff:ThepilotoftheSauryaAirlinesflight,whichwasdepartingfromKathmandu,wastheonlysurvivor,officialssaid.
AfterHarrisDeclines,SenatorWillPresideatNetanyahu'sSpeechtoCongress:AnaidetoVicePresidentKamalaHarrissaidshehadaschedulingconflictbutwouldmeetwiththeIsraeliprimeministerthisweek.(NewYorkTimesWorldNews)After(Harris)Declines,(Senator)(WillPreside)at(Netanyahu’s)Speechto(Congress):Anaideto(VicePresident)(KamalaHarris)saidshehadaschedulingconflictbutwouldmeetwiththe(Israeli)(primeminister)thisweek.(Congress)->(WhiteHouse)AfterHarrisDeclines,SenatorWillPresideatNetanyahu'sSpeechtoWhiteHouse:AnaidetoVicePresidentKamalaHarrissaidshehadaschedulingconflictbutwouldmeetwiththeIsraeliprimeministerthisweek.
original context with manipulated ones.
C.2 Improper Negation Generation: An
Example
We remark that the adoption of LLMs during the
dataset creation is only for generating negations
and identifying key context. Inappropriate or illog-
ical negations are excluded during manual review.
Below is a typical example:
The original news headline satisfies our require-
ment as a claim with its veracity verifiable. How-
ever, the negation is not a reasonably manipulated
real-world claim. So, that negation is not included
in our dataset, although it can be correctly identi-
fied by M ANICOD .
OriginalHeadline: High Energy singer Evelyn Thomas dies, aged 70. Source: BBC Entertainment&ArtsDate: 24 Jul 2024Negation:High Energy singer Evelyn Thomas doesnot die at age 70.C.3 Context Alteration and Potential Bias
Manual Alteration and Rules . Three members
of our team independently performed the alteration
by referring to the manipulated content observed
from Google fact-check tools (Google, 2021) and
accordingly mimicking the generation of manip-
ulated content in the real world. Each member
performs iteration for 1/3 samples and reviews the
remaining 2/3 of samples to reach an agreement.
The modification is based on a set of rules de-
fined by us. For example, for the numbers, we con-
sider many manipulations are made to exaggerate
facts. So, we increase the numbers’ magnitude in
manipulated samples. For entity names like coun-
tries and celebrities, our manipulations are based
on personal inception about certain entities, e.g.,
swapping country names, swapping the name of
the current US president with the historical one,
etc.
Potential Bias . The context alteration, although
is performed manually, can minimize the potential
bias of the adopted LLMs in detecting the manipu-
lated contents, given a reasonable assumption that
LLMs may have an aptitude for identifying con-
tents generated by themselves. Nonetheless, we
remark that the subjective bias of our members
may unavoidably be involved, which may indeed
reflect the process of how manipulated contents
were generated. Our work focuses on the detection
of manipulated content. The bias and social aspects
involved in generating manipulated content are not
in our scope.
D Time Performance
Our experiments are partially performed on an
Nvidia H100 GPU (80GB VRAM) and an Nvidia
A100 GPU (40GB VRAM). As part of our evalua-
tion, we record the time lapsed during the context
construction (i.e., document chunking and vectoriz-
ing during the RAG) and the knowledge-grounded
inference. Given the top three search results to
be crawled for retrieval, the median value of time
spent on context construction is 5.774 seconds on
the tested H100 GPU server, with the 25% and 75%
percentiles of 5.774 seconds and 27.0 seconds, re-
spectively. The vectorization of some retrieved con-
tents takes a longer time mainly because the con-
tents on those webpages are too complicated and
contain too many irrelevant contents like advertise-
ments. We did not observe an obvious difference
in the time performance of context construction be-
15

Claim:AnewToryleaderwillbeannouncedby2Octoberunderplanssetoutbythe1922committee.Dataset: Context alteration (November -> October)     Source: BBC PoliticsDate: 24 Jul 2024Inference:ThestatementsaysthatanewToryleaderwillbeannouncedby2October,whichalignswiththeinformationprovidedinthecontextthat“TheConservativePartywillappointanewleaderonNovember2toreplaceRishiSunak.”However,thisdateislaterthantheonementionedinthestatement.Decision: True (Truth)tween the tested H100 GPU and the A100 GPU. In
contrast to the context construction, we observed a
much more uniform distribution of the time elapsed
for the LLM’s knowledge-grounded inference. The
median of inference time recorded is 2.657 seconds
on the H100 server and 3.58 seconds on the A100
server.
E Case Study of Inference Anomaly
Failure in Reasoning Chronological Order &
Confusion of Premise and Hypothesis . We find
a case failing due to this reason in our evaluation.
The claim and the inference output are presented
as follows.
Given the original news headline states that “ the
leadership of the Tory will be announced by 2
November ,” we altered the due date to one month
earlier, making the claim a piece of disinformation
because the deadline provided in the claim contra-
dicts the ground truth. However, by interpreting
the inference output, we find that the adopted LLM
failed to reason the chronological order of these
two dates, especially the logical relationship when
these two dates are used as deadlines, resulting in
a false prediction. Besides that, from the last sen-
tence of the inference, we observed that the LLM
became confused about the premise and hypothesis.
It mistakenly treated the input claim as the premise
(i.e., the ground truth), and attempted to determine
the veracity of the retrieved online knowledge (i.e.,
the context supplied to the LLM).
Failure in Comprehending Synonyms . Although
LLMs are known to be powerful in natural lan-
guage comprehension, we still find some cases
failed because of mistakes in understanding syn-
onyms. We present an example below.
From the inference output, we find the LLM
correctly reasoned the claim against the retrieved
knowledge. However, it failed to understand the
meaning of the term “ending the re-election”, and
mistakenly differed it from “quitting the race” for
the presidential election although these two terms
express the exactly same meaning. As a result, the
Claim:InararetelevisedaddressfromtheOvalOffice,PresidentJoeBidensaysheendedhisre-electioncampaigntounitehispartyandcountry.Dataset:Original(Truthful)/Source:BBCWorld/Date:25Jul2024Inference:Basedonthecontextprovided,Icananalyzetheinputstatementasfollows:Theinputstatementclaimsthat...However,accordingtothecontext,itisactuallyPresidentBiden'sfirstspeechafterquittingtherace,notendinghisre-electioncampaign.ThissuggeststhatthestatementcontainsafactualmistakeregardingthereasonforPresidentBiden'sspeech.Additionally,thecontextmentionsthatPresidentBidenendedhisrunforasecondterm,whichimpliesthathewasindeedrunningforre-electionbeforedecidingtoquit.Decision:False(Manipulated)
Claim: COVID-19 can survive temperatures higher than 25 Celsius.Dataset: COVID-ScientificInference:Toanalyzetheinputstatement,Iwillcompareitwiththecontextprovided.TheinputstatementclaimsthatCOVID-19cansurvivetemperatureshigherthan25°C.However,accordingtothecontext,SARS-CoV-2(thenewcoronavirusthatcausesCOVID-19)issensitivetohightemperaturesandcanbekilledwhenexposedtotemperaturesofbetween50and55°C(122to131°F)for20minutes.Additionally,anotherstudymentionedinthecontextfoundthattheviruswaskilledafter5minutesat70°C(158°F),andamathematicalmodelestimatedthattheviruswouldbekilledafteranaverageof2.5minutesat70°C(158°F).ThesefindingssuggestthatCOVID-19isnotcapableofsurvivingtemperatureshigherthan25°C,butratherissusceptibletoheatandcanbekilledattemperaturesabove50°C.Therefore,Iconcludethattheinputstatementisapieceofmisinformationbecauseitcontradictstheevidenceprovidedinthecontext.ThecorrecttemperaturerangeforkillingCOVID-19ismuchhigherthan25°C.Decision: FalseLLM determined it as false information.
Failure in Reasoning in Numbers . LLMs some-
times may not be able to compare the values of
multiple numbers. Below we show an example
from the COVID-Scientific dataset observed dur-
ing the baseline benchmarking.
In the inference, we found that M ANICOD man-
aged to find suitable knowledge from the Internet
but failed to compare the values of 25 °Cand 50 °C.
Based on the knowledge in the context, the lowest
temperature that may kill the virus is 50 °C, which
makes the claim indisputably true. However, the
adopted LLM did not manage to compare these
two values and mistakenly determined the claim as
a false information.
F Confusion Matrix of the Evaluation
(RQ1)
We count the occurrence of true positive, true nega-
tive, false positive, and false negative cases based
on the evaluation of our proposed dataset contain-
ing 6,770 claims, i.e., 2,500 recent real-world news
headlines and derived 4,270 manipulated fake news.
The confusion matrix is shown in the table below:
Positive (fake) Negative (truth)
True 3956 1483
False 1017 314
Using those results, we calculate the precision,
recall, and F1 score of MANICOD and present them
in §5.1.
16