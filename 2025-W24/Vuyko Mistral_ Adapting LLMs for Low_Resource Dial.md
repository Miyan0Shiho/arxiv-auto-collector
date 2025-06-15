# Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation

**Authors**: Roman Kyslyi, Yuliia Maksymiuk, Ihor Pysmennyi

**Published**: 2025-06-09 10:30:35

**PDF URL**: [http://arxiv.org/pdf/2506.07617v1](http://arxiv.org/pdf/2506.07617v1)

## Abstract
In this paper we introduce the first effort to adapt large language models
(LLMs) to the Ukrainian dialect (in our case Hutsul), a low-resource and
morphologically complex dialect spoken in the Carpathian Highlands. We created
a parallel corpus of 9852 dialect-to-standard Ukrainian sentence pairs and a
dictionary of 7320 dialectal word mappings. We also addressed data shortage by
proposing an advanced Retrieval-Augmented Generation (RAG) pipeline to generate
synthetic parallel translation pairs, expanding the corpus with 52142 examples.
We have fine-tuned multiple open-source LLMs using LoRA and evaluated them on a
standard-to-dialect translation task, also comparing with few-shot GPT-4o
translation. In the absence of human annotators, we adopt a multi-metric
evaluation strategy combining BLEU, chrF++, TER, and LLM-based judgment
(GPT-4o). The results show that even small(7B) finetuned models outperform
zero-shot baselines such as GPT-4o across both automatic and LLM-evaluated
metrics. All data, models, and code are publicly released at:
https://github.com/woters/vuyko-hutsul

## Full Text


<!-- PDF content starts -->

arXiv:2506.07617v1  [cs.CL]  9 Jun 2025Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation
Roman Kyslyi1, Yuliia Maksymiuk2, Ihor Pysmennyi1
1National Technical University of Ukraine "Igor Sikorsky Kyiv Polytechnic Institute"
2Ukrainian Catholic University
Correspondence: kyslyi.roman@lll.kpi.ua, yuliia.maksymyuk@ucu.edu.ua, pysmennyi.ihor@lll.kpi.ua
Abstract
Inthispaperweintroducethefirstefforttoadapt
largelanguagemodels(LLMs)totheUkrainian
dialect(inourcaseHutsul), alow-resourceand
morphologically complex dialect spoken in the
Carpathian Highlands. We created a parallel
corpus of 9852 dialect-to-standard Ukrainian
sentence pairs and a dictionary of 7320 dialec-
tal word mappings. We also addressed data
shortage by proposing an advanced Retrieval-
Augmented Generation (RAG) pipeline to gen-
erate synthetic parallel translation pairs, ex-
pandingthecorpuswith52142examples. We
have fine-tuned multiple open-source LLMs
using LoRA and evaluated them on a standard-
to-dialect translation task, also comparing
with few-shot GPT-4o translation. In the ab-
sence of human annotators, we adopt a multi-
metric evaluation strategy combining BLEU,
chrF++, TER, and LLM-based judgment (GPT-
4o). The results show that even small(7B)
finetuned models outperform zero-shot base-
lines such as GPT-4o across both automatic
and LLM-evaluated metrics. All data, mod-
els, and code are publicly released at: https:
//github.com/woters/vuyko-hutsul .
1 Introduction
Despiterecentadvancesinlargelanguagemodels
(LLMs), most research and applications remain
centered on high-resource languages and their stan-
dard variants (Li et al., 2024). This imbalance
has significant consequences for linguistic diver-
sity, particularly for underrepresented dialects that
lack sufficient textual resources and standardized
orthographies(Zhongetal.,2024). Despitebeing
an integral part of the linguistic identity of many
communities,dialectsareoftenexcludedfromNLP
toolsandresearch,limitingtheiraccessibilityand
riskingfurthermarginalizationandextinction(Syed
et al., 2023).
Language technologies and especially LLMs are
playing a growing role in the preservation of en-dangeredandunderrepresentedlanguages. While
muchattentionhasfocusedonmajorindigenouslan-
guages(e.g.,M ¯aori,Quechua,Inuktitut)(Trudgill,
2003; Cooper et al., 2024), dialects of national lan-
guages are often overlooked despite facing similar
pressures of attrition and assimilation. Dialectal
variants, particularly in post-Soviet contexts, often
carrysuppressedculturalidentitiesthatarenotre-
flected in the standard language. These dialects are
not only linguistically rich but also culturally vital
and deserve computational attention.
Ukrainian, a language low in resources accord-
ingtoglobalstandardsitself(Kiulianetal.,2024),
exhibitsrichinternalvariation,withdialectssuch
as Hutsul, Boyko and Lemko 1preserving unique
phonetic, lexical and grammatical characteristics.
Among these, the Hutsul dialect, spoken in the
CarpathianMountains,isoneofthemostlinguis-
tically distinct and has the most written sources.
From the culture standpoint, Hutsul dialect has a
greatsignificanceasitencapsulatestraditions, folk-
lore,andauniqueworldview,playingacentralrole
in community identity.
However, the lack of digitized corpora, dictio-
naries, and processing tools makes it practically
invisible to modern LLMs.
Here are some of the linguistic Characteristics
of Hutsul dialect:
•Phonetics : vowel transformations, such as
changing vowels "є"instead of "а", "я"(ya)
(example: "як"→"єк", "ягода" →"єгода"
(“yak”→“yek”, “yahoda” →“yehoda”)).
•Morphology : unique case endings ( -єдь, -
ci) (’-yed’, ’-si’) and preserved dual forms
двi яблуцii (“two apples”, with dual form
“yablutsi” instead of plural “yabluka”).
•Lexicon: Romanian, Polish and German
borrowings such as "бринза" (cheese) and
1https://en.wikipedia.org/wiki/Hutsuls

"шпацiрувати" (go for a walk). 2
Figure1: MapofUkrainiandialects. TheHutsuldialect
islocatedinthesouthwesternCarpathianregion. Source:
Wikipedia
In this work, we present an effort to adapt LLMs
to the Hutsul dialect of Ukrainian, addressing both
data shortage and modeling challenges. Our contri-
butions are:
•A new parallel corpus of original Hutsul-
Ukrainian(9852sentencepairs),dictionaryof
7320dialectalwordmappingsandalsosynthet-
icallyextendedcorpus(52142sentencepairs),
using an advanced RAG approach (detailed
described below).
•Fine-tuning of several open-source LLMs for
Ukrainian to Hutsul dialect translation task.
We frame our task as standard-to-dialect transla-
tion,inwhichmodelhastotakestandardUkrainian
as input and produce grammatically correct (or
as close as possible) Hutsul dialect. Our models
showthatitisfeasibletoaddresssuchtranslation
withlimitedparalleldataandtargetedaugmentation
strategies.
To our knowledge, this is the first work that tries
to adapt LLMs to a Ukrainian dialect and among
thefewgloballyaddressingdialect-to-standardgen-
eration using synthetic augmentation.
2 Related Work
2.1 Dialectal NLP and Language Variation
In recent years we can see growing interest in di-
alectmodeling, particularlyforArabic(Zampieri
etal.,2017),German(Hollensteinetal.,2020),and
Romance languages (Ramponi and Plank, 2021).
2https://en.wikipedia.org/wiki/Eastern_
Romance_influence_on_Slavic_languagesThese efforts mainly focus on classification, gen-
eration,andtranslationbetweendialectsandtheir
standard variants. However, most research remains
concentrated on high-resource languages and di-
alects with pre-existing NLP resources. At the
same time, within Ukrainian language, dialectal
NLP remains underexplored. The VarDial work-
shop series(Zampieri et al., 2024) has supported
work for different Slavic languages on related tasks
suchascross-dialectmachinetranslationandmor-
phologicalmodeling(Bloklandetal.,2024;Kinn
and Åfarli, 2024). For example, Kinn and Åfarli
(2024) explore MT between Bokmål and Nynorsk,
while Blokland et al. (2024) tackle dialectal vari-
ation in North Sámi. The SIGMORPHON 2023
shared task (Kirov et al., 2023) highlighted the
importance of lexicon-based inflection modeling
for low-resource morphological variants.
2.2 Dialect-to-Standard Normalization
The task of normalizing dialectal language to its
standardformhasbeenexploredusingvariousalign-
menttechniques. Scherrer(2023)evaluatedcharac-
ter alignment methods for sentence-level standard-
izationofdialecttranscriptionsacrossFinnish,Nor-
wegian, and Swiss German. The study compared
approaches from dialectometry, speech process-
ing, and machine translation, finding that trained
alignmentmethodsofferedonlysmallbenefitsover
simpleLevenshteindistance. Thissuggeststhatsim-
pleyetrobuststatisticalmethodsmaystillprovide
strongbaselinesinresource-constraineddialectal
settings. Moreover,thestudyunderlinestheneed
fortailoredpreprocessingandalignmenttoolswhen
working with highly variable and phonetically rich
dialect data.
2.3 LLMs and Dialect Adaptation
Several recent studies investigate adapting LLMs
to dialectal data. Held and Klakow (2024) pro-
posetask-agnosticadaptersfordialectadaptation,
whileLiuetal.(2024)introducedynamicadapter
aggregationbasedonlinguisticdistance. Tokenizer
retrofittingformorphologicallyrichdialectsisex-
ploredbyCs’akietal.(2023). Theseworksdemon-
strate that both architectural and data-centric in-
terventions are necessary for effective adaptation.
However, these approaches are primarily evaluated
on English dialects (e.g., African American En-
glish,IndianEnglish)usingcuratedcorporasuch
as Multi-VALUE (Lin et al., 2021), and rely on an-
notateddialect-to-standardpairs,whicharerarely

available for under-resourced dialects.
2.4 Low-Resource and Synthetic Data
Techniques
Our work also benefits from previous research in
low-resource translation and text generation with
synthetic data. Gudibande et al. (2023) and Garcia
et al. (2024) propose retrieval-based or prompt-
based augmentation techniques to bootstrap perfor-
manceinlimited-datasettings. Atthesametimewe
propose our own approach for generating synthetic
data using advanced RAG techniques.
3 Dataset Creation
3.1 Parallel Corpus Collection
We constructed the first parallel corpus for the Hut-
sul dialect and standard Ukrainian by combining
multiple sources and annotation strategies. The
dataset includes 9852 sentence pairs, manually
alignedatthesentencelevel. SourcetextsinHutsul
werecollectedfrompubliclyavailablebooks,ethno-
graphic transcripts, folklore websites, and dialect
blogs. A significant portion of the dataset is based
on the novel "Дiдо Иванчiк" (Dido Yvanchik) by
Petro Shekeryk-Donykiv 3, a foundational literary
workwritteninauthenticHutsul. Weareespecially
gratefultothepublishinghouse Дискурс andtrans-
latorIван Андрусяк ,whokindlyapprovedtheuse
of their modern standard Ukrainian translation for
academic purposes.
StandardUkrainianreferencesinthedatasetwere
either manually translated or sourced from bilin-
gual editions where available. To ensure linguistic
diversity, we tried to included examples from both
everydayconversationandstylizednarrativetexts
(e.g.,folktales,songs,etc.),butduetodatashortage
some topics remain uncovered.
3.2 Lexical Resource
We compiled a Hutsul -to-Ukrainian dictionary that
now contains about 7 300 word pairs. The work
started from the vocabulary that appears in the
book "Дiдо Иванчiк" (Dido Yvanchik), but we
soon enlarged it with data taken from websites that
explain Hutsul dialect words. Among the most
useful web sources were:
•"Dictionary of Hutsul Words" 4.
3https://pl.wikipedia.org/wiki/Petro_
Szekeryk-Donykiw
4https://karnauhova.at.ua/publ/1-1-0-3•"Hutsul Hovir" 5.
•"Dictionary of Ukrainian Dialects of the
Carpathian Region" 6.
•"ExplanatoryDictionaryofHutsulDialects"
by Petro Havuka 7.
•"Hutsul dictionary". 8
All these pages were automatically scraped. The
rawtextcontainedalotofnoise: strangecharacters,
extra commentary, uneven tabulation, and inconsis-
tentseparatorsbetweentheHutsulentryandandits
Ukrainian translation. We wrote simple cleaning
scripts,convertedeverythingtoasingleCSVfile,
and then manually checked the list to remove the
lasterrors. Thefinalresultisacleanlexiconwith7
320Hutsul–Ukrainian pairs. Each entry includes
standard and dialectal word forms.
Despite this effort, the lexicon remains biased
towardthevocabularyfoundinliteratureandfolk-
loric domains. Due to theshortage ofHutsul texts
ontopicslikenews,science,orpolitics,ourdataset
lacks sufficient lexical diversity in those domains.
3.3 Synthetic Data via Advanced RAG
To overcome shortage of written sources in Hutsul
dialect,wedevelopedanadvancedRAGpipelineto
generate additional Hutsul-standard sentence pairs.
The foundation of this pipeline was the dialectal
novel "Дiдо Иванчiк" (Dido Yvanchik), which
served as both the primary corpus for retrieval and
the source of linguistic examples. We used GPT-
4o to build a RAG module capable of retrieving
semantically related Hutsul sentences. For each
generation step, a prompt was created containing
linguistic transformation rules representative of
Hutsul phonological and lexical variation.
TheconstructionoftheRAGpipelineinvolved
several steps:
1.Grammar Rule Extraction: Using "Дiдо
Иванчiк" (Dido Yvanchik) as input, we
prompted GPT-4o to extract and structure
grammatical transformation rules character-
istic of the Hutsul dialect. These included
phonological shifts, morphological alterna-
tions,andsyntacticreordering. Weaugmented
these rules with material from Wikipedia and
5https://rakhiv-mr.gov.ua/hutsulskyj-hovir/
6https://evrika.if.ua/88/
7https://evrika.if.ua/1565/
8http://www.webteka.com/hutsul-language/

ModelsofWordFormationinHutsulDialects
Greshchuk(2016)tocreateacomprehensive
prompt template (see Figure 2).
2.Indexing via RAG: We indexed the "Дiдо
Иванчiк" (Dido Yvanchik) corpus into a re-
trieval system to serve as a reference base
for generating dialectal outputs using text-
embedding-3-large 9.
3.Candidate Sentence Selection: Standard
Ukrainian sentences were sampled from
the UberText corpus (Chaplynskyi (2023)).
For each such sentence, we used the RAG
module to retrieve the top-3 semantically
similar Hutsul-like sentences from "Дiдо
Иванчiк" (Dido Yvanchik).
4.Prompt Construction: Theretrieved exam-
ples were inserted into the prompt template
along with the standard Ukrainian sentence as
the source for translation.
5.Dialect Generation: GPT-4o wasinstructed
to produce a Hutsul translation of the input
sentence using the provided grammar rules
and examples as context (see Figure 3).
Below is a main part or our rule-based
prompt (Full prompt can be found here: https:
//github.com/woters/vuyko-hutsul/blob/
main/prompts/hutsul_rules_prompt.txt ):
Here are Grammatical Rules for
Converting Ukrainian Text into the
Hutsul Dialect:
1. Vowel Shifts:
-“як” →“єк”(“yak” →“yek”)
-“яблуко” →“єблуко” (“yabluko” →
“yebluko”)
-“йдеш” →“єдеш”(“yidesh” →“yedesh”)
2. Consonant Transformations:
-“дiвка” →“ґiвка”(“divka” →“givka”)
-“чого” →“чьо”(“choho” →“cho”)
-“ти” →“ци”(“ty” →“tsy”)
3. Word Order and Syntax:
-“Я тебе люблю” →“Люблю я тебе” (“I
love you” →“Love I you”)
-“Вiн смiється” →“Вiн смiєтси” (“He is
laughing” →“He laugh-reflexive”)
-“Ти знаєш?” →“Ци ти знаєш?” (“Do you
know?” →“Do you know?” with dialectal
marker “tsy”)
Apply only contextually appropriate
transformations.
9https://platform.openai.com/docs/models/
text-embedding-3-largeThisprocesshavecreatedsomedataalignment
challenges in the generated dataset. To address
thesechallengesandalsotocleangenerateddataset
we have developed a hybrid alignment strategy.
First we leveraged the expected textual similarity
between a language and its dialect using difflib’s
SequenceMatcher 10. This approach directly com-
pares character sequences, effectively identifying
pairs even with minor dialectal variations. Pairs
falling below a similarity threshold of 0.45 was
removed from the dataset. To measure quality
of remained sentence pairs we have used several
statistical metrics as described by Scherrer(2023):
•U-src–proportionofunalignedsourcechar-
acters,
•U-tgt– proportion of unaligned target charac-
ters,
•X- proportion of crossing alignment pairs
(swaps)
Thesemetricswerecalculatedoversymmetrized
alignmentpairsobtainedwithfastalign(Dyeretal.,
2013). We have compared alignment metrics
acrossthreedatasets: theoriginalmanuallyanno-
tated dataset (mainly from "Дiдо Иванчiк" (Dido
Yvanchik)),rawsyntheticallygenerateddataset,and
the filtered synthetic dataset.
Beforefiltering,thesyntheticdataalreadyexhib-
ited lower proportions of unaligned source and tar-
getwords(U-src=0.139,U-tgt=0.136)comparedto
theoriginaldata(U-src=0.260,U-tgt=0.265). How-
ever, it presented a higher proportion of crossing
alignments(X=0.033vs. 0.022original),indicating
increased structural variability.
To improvethe qualityof ourgenerated dataset,
we applied alignment-based filtering - for each
sentencepair,wehaveusedpreviouslycalculated
statistics(U-src, U-tgt and X) and we empirically
definedathresholdsforthem: 𝑈-src<0.1,𝑈-tgt<
0.1, and 𝑋 < 0.2.
Any sentence pair that exceeded one or more of
these thresholds was excluded from the final data
set. Thisprocedureremovedinconsistentexamples,
reducingthenumberofreorderings,andimproving
alignment. As the result we got a better quality
syntheticdatasetwithbetterstructuralalignment,as
demonstratedbythecomparativemetricsinTable1.
10https://docs.python.org/3/library/difflib.
html

Metric Original Synthetic Synthetic
Dataset (Raw) (Filtered)
U-src 0.260 0.139 0.005
U-tgt 0.265 0.136 0.005
X 0.022 0.033 0.019
Table1: Alignmentqualitymetricscomparisonbetween
the original dataset, raw synthetic dataset, and synthetic
dataset after alignment-based filtering.
Although we acknowledge that the obtained syn-
theticdatahassomevariationandlackofcertainlex-
icalphrasespresentinauthenticdialectalspeech,its
inclusionisjustifiedbyshortageofHutsultextualre-
sources. Thisfilteringstepeffectivelyimprovedthe
consistency and reliability of the synthetic dataset
and added additional 52142 phrase pairs to our
training dataset.
Figure 2: Overview of the rules generation pipeline
basedon "Дiдо Иванчiк" (DidoYvanchik),Wikipedia,
and Greshchuk (2016).
Although this approach enabled us to signifi-
cantlyenlargethedataset,italsointroducedcertain
limitations. Specifically, the synthetic data reflects
thelexicalandtopicalrangeofthesourcecorpus,
whichlacksmoderndomainssuchasaviation,tech-
nology, news and politics.
As a result, lexical coverage in these areas re-
mains quite sparse or absent (even after generation,
words still remain the same as they are in standard
Ukrainian). Toavoidintroducinghallucinatedvo-
cabulary, we deliberately excluded modern news
andweb-basedcorporafromthegenerationprocess.
3.4 Data Splits and Availability
Thefinalcorpuswassplitinto80%training,10%
validation, and 10% test sets. Test and validation
setscontainonlyhuman-annotatedsentencepairs
from"Дiдо Иванчiк" (Dido Yvanchik).
Figure 3: Overview of the synthetic data generation
pipeline: A RAG system using "Дiдо Иванчiк" (Dido
Yvanchik)and UberCorpus retrievesand prompts GPT-
4o to generate high-quality Hutsul-Ukrainian pairs.
4 Fine-Tuning
To adapt large language models (LLMs) to the
Ukrainian-to-Hutsul translation task, we used
parameter-efficient fine-tuning using LoRA (Hu
et al., 2021).
Wefine-tunedtwostate-of-the-artopen-source
models in the 7B–13B parameter range (as we
consideredourtrainingresourcesandthatthemodel
should be not too big to be able to run locally):
•Mistral-7B-Instruct v0.3 11– Chosen for its
performance-to-size ratio. It outperforms
somelargermodelsonmanybenchmarks,sup-
ports multilingual instructions, and includes
explicit support for Ukrainian (AI, 2023).
•LLaMA-3.1 8B Instruct 12– The instruction-
tuned version of LLaMA 3.1 8B. This model
hasastrongmultilingualsupportandimproved
instruction-followingability,makingitagood
candidate for low-resource translation (Tou-
vron et al., 2024).
Models were selected based on the following
criteria:
•Tokenizer support – Both models use tokeniz-
erswithfallbackstrategiesforrareorout-of-
vocabulary tokens, enabling good handling of
Cyrillic-based dialects.
•Multilingualcapabilities–Mistral-7B-Instruct
v0.3 explicitly lists Ukrainian among sup-
11https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.3
12https://huggingface.co/meta-llama/Llama-3.
1-8B-Instruct

ported languages. LLaMA-3.1 8B Instruct
has shown strong generalization capabilities 13.
•Open licensing and reproducibility – Both
models are publicly available under open-
source licenses.
•Feasibility on a single GPU – Using LoRA or
QLoRA,allselectedmodelscanbefine-tuned
within the a single not very big GPU.
We also considered other multilingual models
suchasBLOOMZ(7.1B) 14andNLLB-200(3.3B) 15,
which offer extensive language coverage. However,
thesemodelseitherunderperformedongenerallan-
guagemodelingtasksorlackedstronggeneration
qualitycomparedtoselectedmodels. Recentbench-
marks demonstratethat Mistral-7B-Instruct-v0.3 16
matches or surpasses larger models in translation
tasks,particularlyinlow-resourceandinstruction-
tuned settings (Wu et al., 2023).
4.1 Fine-Tuning Setup
Each model was trained for 3 epochs using LoRa
on two dataset variants (complete setup can be
foundintheGuthub 17): (1)amanuallycreatedHut-
sul–Ukrainian parallel corpus, and (2) an extended
versionthatincludedcombinedmanualandfiltered
synthetic data.
5 Evaluation
5.1 Metrics
Evaluatingdialectalmachinetranslationisnotasim-
pletask,asstandardreference-basedmetricsmay
penalizecorrectlexicalvariation. Toinsuretrans-
lation quality we calculated the following widely
used metrics:
•BLEU(Papineni et al., 2002) - a precision-
based metric measuring n-gram overlap be-
tweenhypothesisandreference. Whilewidely
used, it may penalize valid lexical and syntac-
tic variations common in dialects.
•chrF++(Popović,2015)computescharacter
n-gram F-scores and has been shown to out-
performBLEUonmorphologicallyrichand
13https://huggingface.co/blog/akjindal53244/
llama31-storm8b
14https://huggingface.co/bigscience/bloomz-7b1
15https://huggingface.co/facebook/nllb-200-3.
3B
16https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.3
17https://github.com/woters/vuyko-hutsulnon-standard languages. It is more robust
to minor spelling or inflectional differences,
making it particularly suitable for dialectal
text.
•TER(Translation Edit Rate) (Snover et al.,
2006) quantifies the number of edits required
toconvertthesystemoutputintothereference.
Itcapturesstructuraldivergenceandpenalizes
reordering errors.
Eachmetricemphasizesdifferentaspectsoftrans-
lation quality:
•BLEUreflects n-gram precision,
•chrF++capturesmorphologicalsimilarityand
recall,
•TERpenalizes structural mismatches
We apply these metrics to test set of manually
translatedUkrainian–Hutsulsentencepairs (1900
pairs). Ratherthanaggregatingthemintoasingle
score,weinterpretthemjointlytounderstanddiffer-
ent behavioral aspects of each model. For instance,
ahighchrF++scorealongsidealowBLEUscore
may indicate valid variation in surface realization.
Asmentionedbefore,whilethesemetricsprovide
a useful baseline, they often struggle to evaluate
dialectal outputs. So, following the framework
of Aepli et al. (2023), we incorporate LLMs as
evaluators.
We prompt GPT-4o model to rate model outputs
along three axes:
•Fluency: grammaticality and naturalness in
the Hutsul dialect.
•Adequacy : preservation of the source sen-
tence’s meaning.
•Dialectal Quality : consistency with known
lexical, phonological, and morphosyntactic
properties of Hutsul.
Eachevaluationisperformedinazero-shotset-
ting. Wethoughtaboutincludingsomegrammatical
rulesintotheprompt,buttoavoidcreationofpoten-
tial bias through this rules decided to use zero-shot
instead.
LLM receives the source, model output, and
a reference translation and returns scores from 1
(poor) to 5 (excellent). The prompt is structured as
follows:

You are a linguistic expert evaluating
machine-translated dialectal text. Rate
the translation on the following dimen-
sions:
1. Fluency (1–5): Is the output gram-
maticallycorrectandnaturalinthetarget
dialect?
2. Adequacy (1–5): Does the output pre-
servethemeaningoftheoriginalsource?
3. Dialectal Quality (1–5): Does the
output reflect the expected phonological,
lexical, and grammatical properties of
the Hutsul dialect?
Return your answer in this exact JSON
format:
{"fluency": x,"adequacy": y,"dialect":
z }
Do not explain your ratings.
Source (Standard Ukrainian): <source
sentence>
ModelOutput(Hutsul): <modelpredic-
tion>
Reference (Hutsul): <reference sen-
tence>
As we didn’t have an opportunity to perform a
human evaluation for our translation, and consider-
ingthatstandardreference-basedmetricsmaynot
be a good fit for the dialect translation(Aepli et al.,
2023), we have used the LLM-based adequacy and
dialect scores as our primary evaluation metrics.
Thesearebetteralignedwithhumanintuitionand
more tolerant of different variations than BLEU or
TER.
Automatic metrics are used in a supporting role
to identify trends such as reordering or character-
level similarity. We report all metrics side-by-side.
This multi-metric approach enables a more holistic
interpretationofmodelbehavior,especiallyinthe
absence of human raters.
5.2 Baselines
We compared our fine-tuned models against the
GPT-4o baseline. Queried via the OpenAI API,
promptedtotranslatestandardUkrainianintothe
Hutsul dialect. To ensure consistency and lexi-
cal coverage, we used the same RAG context and
dictionary entries as in our synthetic generation
pipeline.We did not include non fine-tuned Mistral or
LLaMA models as baselines, since their perfor-
mance in dialect generation tasks was much worse.
Duetotheirsmallsize,theirinstructtuningisinsuf-
ficientforzero-shotgenerationinunderrepresented
languages or dialects.
5.3 Results
Asmentionedearlier,weevaluateourmodelsusing
both automatic metrics and LLM-based judgments.
Table 2 presents the BLEU, chrF++, TER scores
and GPT-4o as an LLM-based judge, rating each
output on a 1–5 scale for fluency, adequacy, and
dialectalqualityscorescomputedonaheld-outtest
set of 1900 sentences.
From the results we can see that all fine-tuned
modelsoutperformtheGPT-4obaselineforevery
metric. Mistralfine-tunedoncombinedmanually
collectedandsyntheticdataperformsbestoverall,
with the highest BLEU (74.35), chrF++ (81.89),
and dialect rating (3.60). While adequacy scores
remain stable across all models ( ≈4.7), dialectal
accuracyvariesmoresubstantiallyandprovesmost
sensitive to the source of training data. Also we
can see that both, LLaMA and Mistral trained on
combined synthetic and manually annotated data
showstrongscoresonautomaticmetricsbutslightly
underperform on dialectal quality, highlighting the
limitations of our method of generating synthetic
data.
5.4 Qualitative Examples
Below we show an example depicting LLM-
calculatedscoresoverrealdataalongwithrespec-
tiveBLEU,chrF++,andTERmetrics. Thisdemon-
stratesthatevensmallfine-tunedmodelsareslightly
better at preserving dialect-specific meaning and
lexiconthanzero-shotcommercialmodels,butstill
far from perfect.
Reference (Hutsul): "Прошумавси у вечєр, єк
зробивси в дiда в обох хатах гармiдер."
(Eng: He came to his senses in the evening,
after a mess started in both of the grandfather’s
houses.)
GPT-4o: "Отєтавсi аж увечєру, єк зчинивсi
в ґiда в обидвох осєдочьках гармидер."
(Eng: He snapped out of it only in the evening,
after a chaos started in both of the old man’s
huts.)

Model BLEU chrF++ TER Fluency Adequacy Dialect
GPT-4o 56.64 65.90 34.34 3.76 4.30 3.22
LLaMA (manual annotated + synthetic) 69.02 74.92 22.90 4.11 4.72 3.33
LLaMA (manual annotated only) 59.98 72.61 28.62 4.13 4.72 3.38
Mistral (manual annotated only) 62.36 75.65 28.62 4.14 4.74 3.35
Mistral (manual annotated + synthetic) 74.35 81.89 22.90 4.18 4.72 3.60
Table2: AutomaticandLLM-basedevaluationresults. BLEU,chrF++,andTERarecomputedwith sacreBLEU .
Fluency, adequacy, and dialect quality are rated by GPT-4o (1–5 scale).
BLEU:7.77,chrF++: 32.8,TER:72.73,Flu-
ency: 4, Adequacy: 4, Dialect: 3
Mistral (manual + synthetic): "Отямивси аж
увечерi, коли зчинився в дiда в обох
хатах гармидер." (Eng: He came to his
senses in the evening, when a chaos broke out
in both grandfather’s houses. )
BLEU: 31.46, chrF++: 46.62, TER: 54.55,
Fluency: 4, Adequacy: 5, Dialect: 3
Mistral (manual only): "Отємивси аж
увечєрi, єк зчєнивси в дiда в обох хатах
гармидер." (Eng: He came to his senses
only in the evening, after stirring up a mess in
both of the grandfather’s houses. )
BLEU: 34.39, chrF++: 56.92, TER: 45.45,
Fluency: 4, Adequacy: 5, Dialect: 4
LLAMA (manual + synthetic): "Отямивси аж
у вечєрє, коли зчинивси в дiда в обох
хатях гармидер." (Eng: He came to his
senses in the evening, when a mess started in
both grandfather’s houses. )
BLEU: 20.45, chrF++: 46.42, TER: 63.64,
Fluency: 4, Adequacy: 4, Dialect: 3
LLAMA (manual only): "Отємивси аж
увечерi, єк зчинивси у дiда в обох хатах
гармидер." (Eng: He came to his senses
only in the evening, when a mess in the
grandfather’s both houses started. )
BLEU: 24.71, chrF++: 49.59, TER: 54.55,
Fluency: 4, Adequacy: 5, Dialect: 3
Limitations
Our work makes first step in Ukrainian dialect
adaptation for LLMs, a lot of limitations remain
open.
An important limitation is that although we in-
troduced a synthetic data generation pipeline to
mitigatelimiteddataavailabilityproblem,synthetictranslationsmaylacknativefluencyorhavestylis-
tic inconsistencies, especially for underrepresented
topics. This is particularly can be seen in domains
not covered by the original corpus, such as politics,
technology, etc. where Hutsul lexicon is either
verylimitedorabsent. Despitefilteringlow-quality
generations, automatic evaluation metrics still may
overestimate linguistic validity.
In addition, evaluation remains challenging. Au-
tomatic metrics such as BLEU and chrF++ often
penalize valid dialectal variation (Garcia et al.,
2024;HeldandKlakow,2024). Tobettercapture
stylistic and synthetic diversity, we use GPT-4o
asanLLM-basedjudgefollowingrecentworkon
LLM-basedevaluationframeworks(Wang,2023;
Liu,2023). However,wenotethatGPT-4oisnotex-
plicitlyfine-tunedfordialectalassessment,andits
preferences may still align with standard Ukrainian
and human evaluation would provide much more
reliable assessments.
Alsoweneedtomentionthatourcurrentmethods
aretailoredtoHutsul,arelativelywell-documented
dialectwithintheUkrainianlanguage. Extension
to other dialects or usage of the same approach
for other low-resource languages will require adap-
tation of both the data pipeline and prompting
strategies.
Acknowledgments
We express our sincere gratitude to Iван
Андрусяк (IvanAndrusiak)forprovidingaccessto
hisUkrainiantranslationof "Дiдо Иванчiк" (Dido
Yvanchik), which served as a cornerstone of our
dataset. We also thank the publishing house
"Дискурс" (Dyskurs) and its director Василь
Карп’юк (Vasyl Karpiuk) for their kind permis-
sion to use the text and for their continued support
of linguistic and cultural preservation initiatives.
Their generosity made this research possible.

References
NoëmiAepli,SarahEbling,andRicoSennrich.2023. A
benchmarkforevaluatingmachinetranslationmetrics
on dialects without standard orthography. arXiv
preprint arXiv:2311.16865 .
Mistral AI. 2023. Introducing mistral 7b and mixtral.
https://mistral.ai/news/mistral-7b/ .
Rogier Blokland, Trond Trosterud, and Jack Rueter.
2024. Morphologicalvariantsinnorths’amidialects.
InProceedings of the Tenth Workshop on NLP for
Similar Languages, Varieties and Dialects (VarDial) .
DmytroChaplynskyi.2023. IntroducingUberText2.0:
A corpus of modern Ukrainian at scale. In Pro-
ceedings of the Second Ukrainian Natural Language
Processing Workshop , pages 1–10, Dubrovnik, Croa-
tia. Association for Computational Linguistics.
NedCooper,CourtneyHeldreth,andBenHutchinson.
2024. “it’showyoudothingsthatmatters”: Attending
to process to better serve indigenous communities
with language technologies. In Proceedings of the
18th Conference of the European Chapter of the
Association for Computational Linguistics (EACL
2024), pages 204–211.
G’aborCs’akiand1others.2023. Tokenizerretrofitting
for morphologically rich languages. In Findings of
the Association for Computational Linguistics: ACL
2023.
ChrisDyer,VictorChahuneau,andNoahA.Smith.2013.
A simple, fast, and effective reparameterization of
IBMmodel2. In Proceedings of the 2013 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies ,pages644–648,Atlanta,Georgia.As-
sociation for Computational Linguistics.
Xavier Garcia and 1 others. 2024. Don’t hallucinate,
retrieve! a survey on retrieval-augmented text gener-
ation. arXiv preprint arXiv:2402.07836 .
Vasyl Greshchuk. 2016. Models of word formation in
hutsul dialects (based on the dictionary “hutsul di-
alectalvocabularyinukrainianbelletristiclanguage”).
Gramatychni Studii , 6:272–286.
Aditya Gudibande and 1 others. 2023. Synthetic
data scaling for low-resource nlp. arXiv preprint
arXiv:2305.11864 .
Wolfgang Held and Dietrich Klakow. 2024. Tada: Task-
agnosticdialectadaptersformultilingualtransformers.
InProceedings of the First Workshop on Domain
Adaptation for NLP .
Nora Hollenstein, C’edrick Fairon, and Julie Snyers.
2020. German’smanyvoices: Acorpusofregional
variation in german. In Proceedings of the 12th
Language Resources and Evaluation Conference .Edward J Hu and 1 others. 2021. Lora: Low-rank
adaptationoflargelanguagemodels. arXiv preprint
arXiv:2106.09685 .
Torodd Kinn and Tor A. Åfarli. 2024. Exploring par-
allel machine translation for norwegian nynorsk and
bokmål. In Proceedings of the Tenth Workshop on
NLP for Similar Languages, Varieties and Dialects
(VarDial) .
ChristoKirov,RyanCotterell,and1others.2023. Sig-
morphon2023sharedtask: Morphologicalinflection
incontext. In Proceedings of the 20th SIGMORPHON
Workshop .
Artur Kiulian, Anton Polishko, Mykola Khandoga,
OrynaChubych,JackConnor,RaghavRavishankar,
andAdarshArunkumarShirawalmath.2024. From
bytestoborsch: Fine-tuninggemmaandmistralfor
theukrainianlanguagerepresentation. arXiv preprint
arXiv:2404.09138 .
Zihao Li, Yucheng Shi, Zirui Liu, Fan Yang, Ning-
haoLiu,andMengnanDu.2024. Languageranker:
A metric for quantifying llm performance across
high and low-resource languages. arXiv preprint
arXiv:2404.11553 .
ZiLin,JoelTetreault,and1others.2021. Multi-value:
Amultilingual,multi-dialect,andmulti-taskbench-
mark for language understanding. In Proceedings of
EMNLP 2021 .
Xiang Liu and 1 others. 2024. Dada: Dynamic adapter
aggregationfordialectaladaptation. arXiv preprint
arXiv:2409.11404 .
Ziweietal.Liu.2023. Gpt-4asanautomaticgrader: An
evaluation of zero-shot and few-shot prompting for
textscoringtasks. arXiv preprint arXiv:2304.02329 .
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. In Proceedings of the
40th Annual Meeting of the Association for Computa-
tional Linguistics , pages 311–318.
Maja Popović. 2015. chrf: character n-gram f-score for
automatic mt evaluation. In Proceedings of the Tenth
Workshop on Statistical Machine Translation ,pages
392–395.
Alan Ramponi and Barbara Plank. 2021. Neural multi-
dialect language models for zero-shot cross-dialect
transfer. In Proceedings of the 16th Conference of
the European Chapter of the Association for Compu-
tational Linguistics: Main Volume .
Yves Scherrer. 2023. Character alignment for dialect
standardization: A comparative evaluation. In Pro-
ceedings of the 1st Workshop on NLP for Less-
Resourced Languages .
MatthewSnover,BonnieDorr,RichardSchwartz,Lin-
nea Micciulla, andJohn Makhoul. 2006. A study of
translationeditratewithtargetedhumanannotation.

InProceedings of the 7th Conference of the Associa-
tion for Machine Translation in the Americas ,pages
223–231.
ShahbazSyed,AhmadDawarHakimi,KhalidAl-Khatib,
and Martin Potthast. 2023. Quantifying the dialect
gapanditscorrelatesacrosslanguages. In Findings
of the Association for Computational Linguistics:
EMNLP 2023 , pages 5196–5210.
HugoTouvron,ThibautLavril,AlpYurtsever,and1oth-
ers. 2024. Llama 3: Open foundation and instruction
models. arXiv preprint arXiv:2404.14219 .
Peter Trudgill. 2003. Dialect contact and new-dialect
formation: The inevitability of colonial englishes. In
Proceedings of the 14th International Congress of
Phonetic Sciences , pages 2193–2196.
Yizhong et al. Wang. 2023. Llm-eval: Unified, auto-
maticandrobustevaluationoflargelanguagemodels
with gpt-4. arXiv preprint arXiv:2305.03045 .
ShĳieWu,YuxuanLi,ChengLi,HaoZhu,and1others.
2023. Benchmarkingpubliclargelanguagemodels
in low-resource settings. In Proceedings of the 2023
EMNLP.
MarcosZampieri,TommiJauhiainen,NikolaLjubešić,
NoëmiAepli,SimonClematide,andJörgTiedemann.
2024. Overviewofthevardialevaluationcampaign
2024. In Proceedings of the Tenth Workshop on
NLP for Similar Languages, Varieties and Dialects
(VarDial) .
Marcos Zampieri, Shervin Malmasi, Preslav Nakov,
Ahmed Ali, and Stephan Vogel. 2017. Arabic dialect
identification for the dsl 2017 shared task. In Pro-
ceedings of the Fourth Workshop on NLP for Similar
Languages, Varieties and Dialects (VarDial) .
TianyuZhong,ZiqiYang,ZhenLiu,RuiZhang,Yiheng
Liu, Hanqi Sun, Yujia Pan, Yiming Li, and Yifan
Zhou. 2024. Opportunities and challenges of large
language models for low-resource languages in hu-
manities research. arXiv preprint arXiv:2412.04497 .