# Evaluating RAG for French immigration law: a benchmark and baseline study

**Authors**: Annia Abtout, Julien Delaunay, Monika Ewa Rakoczy

**Published**: 2026-07-27 13:55:35

**PDF URL**: [https://arxiv.org/pdf/2607.24449v1](https://arxiv.org/pdf/2607.24449v1)

## Abstract
International recruitment in France requires navigating a layered legal framework absent from existing legal AI benchmarks. We present a publicly available benchmark and first comparative evaluation for this domain, covering permit-type recommendation, required-document retrieval, and legal citation coverage. Comparing a parametric LLM baseline against dense retrieval augmentation at two model scales (Qwen3.5-9B and -27B) on 52 annotated synthetic profiles, we find that retrieval improves administrative guidance at both scales, most notably permit-type accuracy. Our results confirm that retrieval grounding is important for more reliable administrative guidance in this domain, and motivate further investigation of hybrid retrieval strategies.

## Full Text


<!-- PDF content starts -->

Evaluating RAG for French immigration law: a
benchmark and baseline study
Annia Abtout⋆[0009−0006−2377−1926], Julien Delaunay∗[0009−0001−9247−5745], and
Monika Ewa Rakoczy[0000−0001−5249−5509]
Talan, Paris, France
first_name.name@talan.com
Abstract.International recruitment in France requires navigating a
layered legal framework absent from existing legal AI benchmarks. We
present a publicly available benchmark and first comparative evalua-
tion for this domain, covering permit-type recommendation, required-
document retrieval, and legal citation coverage. Comparing a parametric
LLM baseline against dense retrieval augmentation at two model scales
(Qwen3.5-9B and -27B) on 52 annotated synthetic profiles, we find that
retrieval improves administrative guidance at both scales, most notably
permit-type accuracy. Our results confirm that retrieval grounding is
important for more reliable administrative guidance in this domain, and
motivate further investigation of hybrid retrieval strategies.
Keywords:RAG·Human resources·Immigration law.
1 Introduction
International recruitment requires navigating administrative procedures that de-
termine whether and under which conditions a foreign national may legally work
in a given country. In France, Human Resource (HR) practitioners must iden-
tify the appropriate residence permit, determine the supporting documents re-
quired by the administration, and ensure that these recommendations are con-
sistent with the applicable regulatory framework. This process draws on multiple
sources,includingimmigrationlaw,labourregulations,bilateralagreements,and
administrative guidance.
Recent advances in Large Language Models (LLMs) have stimulated grow-
ing interest in legal and regulatory AI applications, supported by benchmarks
such as LegalBench [6], LawBench [5], and LexEval [12]. However, to our knowl-
edge, there is no publicly available benchmarks targeting administrative guid-
ance for international recruitment in the French immigration context. Moreover,
the task differs from many existing legal benchmarks as a system must produce
a structured recommendation combining an administrative decision, the asso-
ciated procedural requirements, and sufficient legal grounding to support the
recommendation.
⋆These authors contributed equally.
arXiv:2607.24449v1  [cs.IR]  27 Jul 2026

2 A. Abtout et al.
In this work, we introduce a publicly available benchmark for French immi-
gration administrative procedures in the context of international recruitment.
The benchmark contains 52 annotated synthetic applicant profiles and supports
evaluation along three dimensions: residence permit type, required documents
and legal basis. Using this benchmark, we conduct a first comparison, at two
model scales, between a parametric LLM baseline and a Retrieval-Augmented
Generation (RAG) [10] approach. Code, benchmark, and annotation guidelines
are available through this repository.1
2 Related work
Legal AI and LLMs Limitations.LLMs have demonstrated impressive capabil-
ities in various areas but systematic studies reveal that GPT-4 [1] hallucinates
in at least 58% of cases when queried about U.S. case law [4], generating what
researchers term “large legal fictions”. These models exhibit sycophancy bias,
accepting erroneous legal premises rather than correcting them [8,16]. This un-
reliability has motivated the development of more constrained architectures.
Legal Benchmarks and RAG Validation.Recent benchmarks [6,12,5,14] demon-
strate that RAG outperforms both generic and fine-tuned LLMs in legal reason-
ing, with LexEval showing that specialized models excel at memorization but fail
at logical inference. However, retrieval quality, particularly chunking and context
management,remainstheprimaryperformancebottleneck.Existingbenchmarks
predominantly cover common law systems, leaving civil law jurisdictions under-
explored. Adaptive systems like HyPA-RAG [7] and LexRAG [11] address legal
complexity by dynamically adjusting retrieval strategies, though at substantial
computational cost. To the best of our knowledge, the intersection of legal rea-
soning and administrative guidance in the French immigration context remains
unexplored, despite its practical importance for international recruitment.
3 Benchmark construction
3.1 Legal framework
The recruitment of foreign workers in France operates under a hierarchical legal
framework combining immigration law, labour regulations, bilateral agreements,
and administrative procedures. The Code on the Entry and Residence of Foreign
NationalsandtheRightofAsylum(CESEDA)definesthemainresidencepermit
categories and eligibility conditions. For nationals of countries covered by bilat-
eralagreements,theseagreementstakeprecedenceoverCESEDAprovisions.The
most prominent example is the Franco–Algerian Agreement, which establishes
a distinct legal regime for Algerian nationals. Labour market access conditions
are further governed by the Employment Act (Book V, articles L5221–L5224
1https://github.com/jdelaunay/foreign_workers_recruitment_dataset

Evaluating RAG for French immigration law 3
and associated regulatory provisions), while administrative procedures are doc-
umented through the Service-Public2portal and, in some cases, supplemented
by prefecture-specific requirements. As a consequence, administrative guidance
requires reasoning across multiple legal sources whose applicability depends on
applicant characteristics. This distinguishes the task from standard legal ques-
tion answering and motivates the construction of a dedicated benchmark.
3.2 Task description
While traditional legal benchmarks focus on question answering or legal reason-
ing in isolation, immigration guidance requires the simultaneous production of
several complementary outputs. Given a structured applicant profile, the system
must identify the appropriate residence permit category, determine the support-
ing administrative documentation, and provide the legal references justifying
the recommendation. Accordingly, each benchmark instance contains three tar-
get components: a permit recommendation composed of four attributes (permit
category, statutory mention, renewal status, and visa requirement), the set of
documents required to support the application and a set of legal references corre-
sponding to the applicable legal basis. This multi-output formulation reflects the
operational needs of HR practitioners, who must identify the correct permit and
understand the associated administrative procedure and its legal justification.
3.3 Dataset construction
The benchmark combines synthetic applicant profiles and annotations of the
corresponding residence permit application process. Applicant profiles were gen-
erated using a rule-based simulation process calibrated on publicly available
demographic statistics from the French National Institute of Statistics and Eco-
nomic Studies (INSEE). We adopted synthetic profiles because real immigration
cases are difficult to share due to privacy and confidentiality constraints. Also,
real-world datasets often under-represent rare permit categories and legal situa-
tions. Finally, synthetic generation allows controlled coverage of the main legal
regimes encountered in international recruitment. The resulting profiles cover
different nationalities, residence situations, employment conditions, and permit
categories. Detailed generation parameters are provided in Supp. A.1. To fa-
cilitate annotation, we employed a RAG-augmented Qwen3.5-27B [15] over a
curated legal corpus (Supp. A.2) to generate candidate recommendations, which
annotators then validated or corrected using the INCEpTION annotation tool
[9]. The annotation campaign involved six annotators with varying levels of fa-
miliarity with French immigration procedures and an HR specialist in interna-
tional recruitment acting as domain expert. Annotators independently produced
permit recommendations, required documents lists, and legal references follow-
ing a shared annotation guideline. To ensure consistency across annotations,
annotators were instructed to rely on a predefined set of authoritative legal
2https://www.service-public.gouv.fr

4 A. Abtout et al.
sources covering immigration law, labour regulations, bilateral agreements, and
administrative guidance. Five calibration profiles were first annotated by all par-
ticipants and the HR expert to refine the protocol. The remaining profiles were
double-annotatedanddisagreementswereadjudicatedbyathirdannotator.This
process yielded 52 curated benchmark instances. To assess benchmark reliability,
wemeasuredinter-annotatoragreementonalltargetdimensions.Agreementwas
moderate for permit and documents and higher for legal basis annotations, with
most disagreements arising from optional residence permits for EU citizens and
the distinction between permit renewal and permit upgrade. Detailed agreement
analyses are provided in Supp. A.4.
4 Experiments
4.1 Evaluation settings
We evaluate the effect of retrieval on recommendation quality across two model
scales, Qwen3.5-9B and Qwen3.5-27B [15]. For each model we compareLLM-
onlywhichreceivestheapplicantprofileandastructuredprompt,withoutaccess
to any external knowledge and serves as a parametric baseline withDense RAG
which augments the same model with retrieved legal context drawn from a vec-
torized legal corpus through MediaTech3. The corpus combines resources from
all the relevant legal sources (see Supp. A.2). Documents are segmented into
semantically coherent chunks corresponding to articles, sections, or procedural
contentblocksandarerepresentedusingpre-computedBGE-M3embeddings[2].
Given an applicant profile, dense vector search retrieves the top-kmost relevant
chunks, which are then injected into the LLM prompt. All four conditions share
the same prompt template (see Supp. C.1) and legal corpus, so that observed
differences are attributable to the two controlled factors : the presence or ab-
sence of retrieval and model scale. Experiments are conducted on the annotated
subset of our benchmark, which covers the main nationality groups and permit
categories introduced in Supp. A.1 and A.5.
4.2 Evaluation metrics
The benchmark evaluates three aspects of french administrative guidance so we
define one task-specific metric for each output dimension.
Permit-type.Eachrecommendationisdecomposedintofourcomponents:is_renewal,
permit_type(16-categoryclosedtaxonomy,Table2inSupp.A.3),mention(e.g.
skilled worker) andvisa_required. Each component is evaluated as a binary
match against the gold standard. Full permit match requires all four compo-
nents to be simultaneously correct. We additionally reportmention*as mention
accuracy conditioned on correctpermit_type, to isolate intra-category errors.
Normalisation and matching rules are described in Supp. B.
3MediaTech is an open-source platform providing pre-processed and embedded public
datasets for AI applications : https://beta.gouv.fr/startups/media-tech.html

Evaluating RAG for French immigration law 5
Table 1: Component-level permit accuracy for Qwen-27B and Qwen-9B.∆is the
absolute difference between Dense RAG and LLM-alone.mention∗is conditional
onpermit_typebeing correct.
Qwen-27B Qwen-9B
Component LLM-alone Dense RAG∆LLM-alone Dense RAG∆
is_renewal88.5% 82.7%−5.8 pts 82.7% 82.7%±0.0 pts
permit_type50.0% 75.0%+25.0 pts 30.8% 57.7%+26.9 pts
mention∗84.6% 94.9%+10.3 pts 81.2% 90.0%+8.8 pts
visa_required53.8% 55.8%+1.9 pts 53.8% 53.8%±0.0 pts
full_permit_match25.0% 34.6%+9.6 pts 9.6% 25.0%+15.4 pts
Documents.Document requirements are evaluated only on full permit match
profiles, which isolate downstream document prediction from upstream permit-
classification errors. Matching combines rule-based normalisation of frequent
administrative variants with embedding-based semantic alignment. We report
precision, recall and F1 score. See Supp. B.2 for matching procedure.
Legal citation.Legal grounding is evaluated through a recall-oriented coverage
metric. Since the gold standard contains a minimal set of supporting references,
precision would be difficult to interpret. Predicted citations are first mapped
to canonical representations (CESEDA, Service Public or bilateral agreements),
after which coverage is computed as the proportion of gold references recovered
by the system. This metric is also restricted to full permit match profiles.
5 Results
Permit type recommendation accuracyTable 1 reports component-level
permit accuracy across the 52 benchmark profiles. Dense RAG improves per-
mit recommendation at both scales:permit_typeaccuracy rises from 50.0% to
75.0%forQwen-27Bandfrom30.8%to57.7%forQwen-9B,andfull_permit_match
from 25.0% to 34.6% and from 9.6% to 25.0% respectively. The relative gain is
at least as large for the smaller model, whose weaker parametric baseline points
to a stronger reliance on retrieval. Retrieval seems to narrow but not close the
gap with the larger model. Conditional mention accuracy (mention∗) is high and
improves at both scales which indicates that, once the permit family is correct,
the appropriate mention is generally recovered.
Erroranalysis(Table4,Supp.C.2)indicatesthatretrievalprimarilyimproves
identification of the applicable legal regime: the LLM-alone baselines frequently
confuse permit families governed by different frameworks and Dense RAG sub-
stantially reduces these errors. The same error families appear at both scales,
with the 9B model defaulting more often to the most common categories for
workers (employee, Talent).

6 A. Abtout et al.
By contrast, retrieval has little effect onvisa_required, which is essentially
unchanged across both conditions and both scales. Indeed, most residual errors
correspond to omitted visas. Since visa requirements were not explicitly empha-
sised in the prompt, these should not be read solely as retrieval failures.
Document requirement matchingTable 5 reports document-set matching
under the strict full permit match filter. Dense RAG substantially improves
document prediction at both scales, raising F1 from 23.9% to 45.3% for Qwen-
27B and from 23.5% to 36.4% for Qwen-9B with consistent gains in precision
and recall. The gain appears smaller for the 9B model, which appears to exploit
retrieved context less effectively. Because evaluation is restricted to profiles with
correct permit recommendations (a small subset), these scores should be read
qualitatively.
Error analysis (Supp. C.3) suggests that retrieval primarily reduces halluci-
nated administrative requirements while improving recovery of permit-specific
supporting documents. However, several procedural requirements remain fre-
quently omitted in both conditions, indicating that retrieval alone is insufficient
to guarantee exhaustive procedural guidance.
Legal citation qualityDense RAG improves legal grounding for Qwen-27B by
raisingcitationcoveragefrom7.7%to46.6%(Table7,Supp.C.4),withreferences
recovered from all source types. For Qwen-9B the corresponding improvement
is much smaller (10.0% to 13.1%), with almost no Service Public or bilateral
references recovered. We stress, however, that the 9B citation figures rest on
a very small conditional subset (n= 5andn= 13profiles) so this contrast
should be read as a tentative indication that citation grounding may be more
scale-dependent than permit classification, not as an established effect. Coverage
remains incomplete even for Qwen-27B, and all scores should be treated as lower
bounds, since the gold standard contains only a minimal reference set.
6 Discussion and conclusion
The benchmark campaign exposed recurring ambiguities, including visa require-
mentsadministrativevocabulary,andlegalcitationexhaustivity.Onefielddefini-
tionwasrevisedmid-campaign(degreeobtainedbecamerecognisedinFrance)re-
quiring retrospective adjudication. Such issues are inherent to first-round bench-
markconstructioninadomainwherelegalpracticeitselfadmitsvariability.Even
with structured applicant profiles and a simplified representation focused on per-
miteligibility,documentsandlegalreferences,recommendationsremainsensitive
to profile attributes. The challenges observed during annotation and evaluation
therefore reflect the complexity of the underlying administrative framework.
The results suggest that this complexity remains challenging for current AI
systems. Retrieval grounding improves permit classification at both model scales
indicating that retrieval partly compensates for limited parametric knowledge.

Evaluating RAG for French immigration law 7
Nonetheless, errors persist despite a constrained corpus and the benefit of re-
trieval is less uniform on the downstream dimensions: gains on document and
citation recovery are smaller for the 9B model, suggesting that exploiting re-
trieved context for detailed legal grounding is more scale-dependent than permit
classification. As our evaluation still covers a single model family, broader cross-
familyvalidationremainsfuturework.Ourfindingsareotherwiseconsistentwith
prior work on legal and administrative reasoning.
Several limitations should also be acknowledged. The benchmark contains
only 52 annotated profiles and the documents and citation evaluations are con-
ditioned on correct permit classification, yielding small subsets so cross-scale dif-
ferences on these two dimensions should be read as indicative only. Legal citation
coverage should be interpreted as a conservative lower bound, since annotators
provided only a minimal set of references. Also, using the same Qwen model
family for generation and evaluation might have introduced a self-preference
bias [13]. Future work should broaden the range of model families and scales
and investigate richer evaluation protocols for legal grounding. Hence, this work
informs two next steps. On the benchmark side, a second annotation campaign
targeting at least 200 profiles is planned, with a revised protocol addressing
identified ambiguities. On the system side, persistent under-retrieval points to
hybrid retrieval (HyPA-RAG [7]), following the state of the art.
Acknowledgments.We express our gratitude to Khaouthar MOUSSAOUI, Elodie
WIKSZAK, Thierry SCHULLER, Pauline LALLEMENT and Tsamba MBEMBO for
their invaluable assistance in the annotation process. We would also like to thank
Laurent CERVONI, head of the team, for his support and guidance throughout this
project. We are also grateful to Florence REAL for identifying the initial use case
within the HR department that gave rise to this project.
Disclosure of Interests.The authors have no competing interests to declare that
are relevant to the content of this article.
References
1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida,
D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv
preprint arXiv:2303.08774 (2023)
2. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: M3-embedding: Multi-
linguality, multi-functionality, multi-granularity text embeddings through self-
knowledge distillation. In: Ku, L.W., Martins, A., Srikumar, V. (eds.) Findings
of the Association for Computational Linguistics: ACL 2024. pp. 2318–2335. As-
sociation for Computational Linguistics, Bangkok, Thailand (Aug 2024).https:
//doi.org/10.18653/v1/2024.findings-acl.137,https://aclanthology.org/
2024.findings-acl.137/
3. Cohen, J.: A coefficient of agreement for nominal scales. Educational and psycho-
logical measurement20(1), 37–46 (1960)
4. Dahl, M., Magesh, V., Suzgun, M., Ho, D.E.: Large legal fictions: Profiling legal
hallucinations in large language models. Journal of Legal Analysis16(1), 64–93
(2024)

8 A. Abtout et al.
5. Fei, Z., Shen, X., Zhu, D., Zhou, F., Han, Z., Huang, A., Zhang, S., Chen, K., Yin,
Z., Shen, Z., et al.: Lawbench: Benchmarking legal knowledge of large language
models. In: Proceedings of the 2024 conference on empirical methods in natural
language processing. pp. 7933–7962 (2024)
6. Guha, N., Nyarko, J., Ho, D., Ré, C., Chilton, A., Chohlas-Wood, A., Peters, A.,
Waldon, B., Rockmore, D., Zambrano, D., et al.: Legalbench: A collaboratively
built benchmark for measuring legal reasoning in large language models. Advances
in neural information processing systems36, 44123–44279 (2023)
7. Kalra, R., Wu, Z., Gulley, A., Hilliard, A., Guan, X., Koshiyama, A., Treleaven,
P.C.: Hypa-rag: A hybrid parameter adaptive retrieval-augmented generation sys-
tem for ai legal and policy applications. In: Proceedings of the 1st workshop on
customizable nlp: Progress and challenges in customizing nlp for a domain, appli-
cation, group, or individual (customnlp4u). pp. 237–256 (2024)
8. Keisha, F., Singh, P., Fernandes, D., Manivannan, A., Wicaksono, I., Ahmad, F.,
Rim, W.B., et al.: All for law and law for all: Adaptive rag pipeline for legal
research. arXiv preprint arXiv:2508.13107 (2025)
9. Klie, J.C., Bugert, M., Boullosa, B., Eckart de Castilho, R., Gurevych, I.: The
INCEpTION platform: Machine-assisted and knowledge-oriented interactive an-
notation. In: Proceedings of the 27th International Conference on Computa-
tional Linguistics: System Demonstrations. pp. 5–9. Santa Fe, New Mexico (2018),
https://www.aclweb.org/anthology/C18-2002
10. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
11. Li, H., Chen, Y., YiRan, H., Ai, Q., Chen, J., Yang, X., Yang, J., Wu, Y., Liu, Z.,
Liu, Y.: Lexrag: Benchmarking retrieval-augmented generation in multi-turn legal
consultation conversation. In: Proceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval. pp. 3606–3615
(2025)
12. Li, H., Chen, Y., Ai, Q., Wu, Y., Zhang, R., Liu, Y.: Lexeval: A comprehensive
chinese legal benchmark for evaluating large language models. Advances in Neural
Information Processing Systems37, 25061–25094 (2024)
13. Panickssery, A., Bowman, S.R., Feng, S.: Llm evaluators recognize and favor their
own generations. Advances in Neural Information Processing Systems37, 68772–
68802 (2024)
14. Pipitone, N., Alami, G.H.: Legalbench-rag: A benchmark for retrieval-augmented
generation in the legal domain. arXiv preprint arXiv:2408.10343 (2024)
15. Qwen Team: Qwen3.5: Towards native multimodal agents (February 2026),https:
//qwen.ai/blog?id=qwen3.5
16. Savelka, J., Ashley, K.D., Gray, M.A., Westermann, H., Xu, H.: Explaining
legal concepts with augmented large language models (gpt-4). arXiv preprint
arXiv:2306.09525 (2023)
17. Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S., Dur-
mus,E.,Hatfield-Dodds,Z.,Johnston,S.,Kravec,S.,etal.:Towardsunderstanding
sycophancy in language models. In: International Conference on Learning Repre-
sentations. vol. 2024, pp. 110–144 (2024)

Evaluating RAG for French immigration law 9
Supplementary material
A Benchmark building
A.1 Profiles generation
The objective of profile generation was not to reproduce the exact distribution
of immigration cases observed in practice, but rather to obtain a benchmark
coveringtheprincipallegalregimesencounteredduringinternationalrecruitment
while preserving demographic plausibility.
To create a realistic evaluation dataset, we developed a demographic simula-
tion system that generates synthetic applicant profiles reflecting France’s immi-
grantpopulationdistributionbasedontheFrenchNationalInstituteofStatistics
and Economic Studies (INSEE)4empirical data. Nationality distribution follows
observed immigration patterns in 20245through weighted sampling across four
regions: Africa (42.2%, led by Algeria 12.4%, Morocco 11.7%), Europe (26.5%,
led by Portugal 7.3%), Asia (14.3%, led by Turkey 3.4%), and the Americas
(4.0%).Two-thirdsofprofilesindicatecurrentFrenchresidence.Geographicloca-
tions are sampled from GeoNames6with population-based weights. Educational
attainment skews toward higher qualifications (Master’s 25%, Bachelor’s 20%),
notably to enable talent card eligibility for some profiles. Professional status
heavily favors permanent employment (75%). Proposed salaries follow a right-
skewed distribution (Skew-normal,α=3) centered at€35,000 with a€21,876.40
minimum threshold, to allow high salaries which can allow specific residence per-
mits like the talent card. Current residence permits are assigned based on time in
France: recent arrivals (0–1 years) predominantly hold temporary residence per-
mit (55%), mid-term residents (1–6 years) hold multi-year residence permit, and
long-term residents (7+ years) primarily hold 10-year residence permits (70%).
Algerian nationals receive certificates of residence under the Franco-Algerian
Agreement. Ninety percent of permits approach expiration (1–4 months remain-
ing), simulating typical renewal scenarios.
A.2 Legal Corpus Construction
Because retrieval quality is inherently bounded by corpus quality, particular
attention was devoted to source selection and legal validity filtering.
The legal corpus underlying our framework aggregates four heterogeneous
normative sources, each subject to source-specific preprocessing and filtering
prior to indexing. This filtering stage is a deliberate architectural choice: index-
ing abrogated or thematically irrelevant provisions would expose the retrieval
pipeline to legally incorrect recommendations regardless of retrieval quality, di-
rectly undermining the traceability guarantees.
4The INSEE collects, produces, analyses and disseminates information on the French
economy and society
5https://www.insee.fr/fr/statistiques/3633212
6https://www.geonames.org

10 A. Abtout et al.
TheCESEDAcorpus,builtfromtheFrenchconsolidatedlegislationdataset7,
retains only articles whose status is markedin force, excluding abrogated provi-
sions that remain present in the raw dataset. The same validity filter is applied
to the Employment Act, constructed from the same corpus, which is further
restricted to articles targeting Book V, Title II, Chapter II, Sections 1–4, the
statutory provisions governing labor market access conditions for foreign na-
tionals (articles L5221–L5224 and their regulatory counterparts). This targeted
restriction reduces index noise from unrelated labor law provisions while pre-
serving full coverage of the relevant perimeter.
Thebilateralagreementscorpusisconstructedbymergingtwosourcedatasets
(legi_decretandlegi_decret_loi) from the French consolidated legislation
dataset,filtered toin-forcetextswhose titlessimultaneouslyreferencetheFrench
Republic, a legal instrument type (agreement,convention,joint statement), and
a thematic scope related to persons, residence, or employment. This compound
filter operationalizes the legal hierarchy described in Section 3.1, ensuring that
only instruments with direct bearing on immigration and labor rights are in-
dexed.
Finally, the Service-Public corpus8is indexed without thematic filtering, as
itsproceduralscope(applicationworkflows,requireddocumentation,andadmin-
istrative deadlines) is inherently bounded to the immigration domain by source
selection.
A.3 Permit Type Taxonomy
Table 2 lists the permit categories used in the closed taxonomy forpermit_type
evaluation. Categories marked†are those for which the distinction between
temporary and multi-year variants carries distinct documentary requirements
and is therefore evaluated independently.
A.4 Inter-annotator agreement
The objective of the agreement analysis is not to establish a definitive legal
ground truth, but to assess whether independent annotators converge towards
similar administrative recommendations under a shared protocol.
Inter-annotator agreement (IAA) is reported across the three output fields
using Cohen’s kappa (κ) [3] for permit type, and F1 score for required documents
andlegalbasis.Figure1aillustratestheinherentambiguityofpermitannotation.
Two systematic errors emerged: assigning a residence permit to EU/EEA/Swiss
citizens despite it being optional—a guideline deviation whose impact is con-
firmed by the improved agreement scores in Figure 1b when both valid choices
are treated as equivalent—and recommending permit renewals where an upgrade
was available (e.g., a temporary rather than a multi-year residence permit), as
can be seen on Figure 2. Given the exploratory nature of this work and the
7https://huggingface.co/datasets/AgentPublic/legi
8https://huggingface.co/datasets/AgentPublic/service-public

Evaluating RAG for French immigration law 11
Table 2: Permit type taxonomy. Temporary = 1 year.†marks categories where
thetemporary/multi-yeardistinctioncarriesdistinctdocumentaryrequirements.
Label Description
NO_PERMITNo permit required
EU_CITIZENEU/EEA/Swiss free movement
RC_LONG_STAY_EULong-stay EU resident card
RESIDENT_CARD10-year resident card
RC_PERMANENTPermanent resident card
ARC_1YAlgerian resident certificate (1 year)
ARC_10YAlgerian resident certificate (10 years)
TALENTTalent Card
T_EMPLOYEE†Temporary work permit
MY_EMPLOYEE†Multi-year work permit
T_PFL†Temporary private/family life
MY_PFL†Multi-year private/family life
ENTREPRENEURSelf-employed / liberal profession
MY_ENTREPRENEUR†Multi-year entrepreneur
TWTemporary worker
TRAINEETrainee
limited number of annotations per pair,κshould be interpreted as indicative
only.
As an exploratory contribution, the annotations covered permit type, re-
quired documents, and legal basis simultaneously. Figure 3 reveals mixed re-
sults: while some documents are common across permit types (e.g., passport,
passport photo), systematic errors arose around visa-required documents pre-
ceding the residence permit application, such as tax stamp of€99 or proof of
qualification—sources not included in our knowledge base, requiring annotators
to consult France-Visa independently. These discordances are also visible in Ta-
ble3.Also,mostcommonerrorsconcernedcommondocuments,liketheprevious
residence permit in case of renewal, medical certificate issued by the OFII or the
signed copy of the pledge to uphold the principles of the French Republic. Using
the recommandation of the LLM, while gaining time especially for the legal ba-
sis, also introduced a bias of sycophancy [17], which permitted some document
not required to be selected by some annotators (such as employment contract or
offer of employment and proof of income for instance), or incorrect fees for tax
stamps.
As shown in Figure 4, legal basis scores are encouraging, though ambiguities
remain in the annotation guidelines themselves: it is unclear whether annotators
should report the minimal legal basis strictly related to the permit type, the
basis necessary to identify required documents, or an exhaustive enumeration.
This warrants clarification in future work.

12 A. Abtout et al.
Table 3: Most frequent documents involved in annotation discordances (all per-
mit types)
Document Count
Medical certificate issued by OFII 15
Valid residence permit 13
Signed commitment to respect the principles of the French Republic 13
Copy of highest diploma obtained 12
Proof of qualification 11
Long-stay visa serving as a residence permit 11
Employment contract or job offer 10
Tax stamp of€99 10
Employment certificate 10
Proof of address (less than 6 months) 10
Proof of resources 8
Curriculum Vitae (CV) 8
Work permit issued by employer 7
Valid travel document 6
Birth certificate 6
Proof of compliance with regulatory practice conditions 4
Statutory declaration of non-polygamy 4
Tax stamp of€225 4
Evidence supporting continuation of employment contract 4
Visa regularisation fee (€300) 4
A.5 Dataset statistics
This section explores the statistics related to our 52 curated annotations.
Figure 5 reflects the data generation choices: the temporary work permit
is the most frequent, consistent with the deliberate oversampling of first-time
applications. The one-year Algerian residence certificate ranks second, owing to
the strong representation of Algerian nationals and its eligibility for both first-
time and renewal applications. Profiles requiring no permit follow, reflecting the
prevalence of EU nationals, with the talent card also notably represented.
Figure6reflectspermit-typediversity:whilemostprofilessharecommondoc-
uments (passport, identity photo, signed republican commitment, OFII medical
certificate), certain permit types impose specific requirements: the Algerian res-
idence certificate, for instance, requires proof of address less than three months
old rather than six, along with a host certificate for the visa. Visa requirements
further contribute to document variability across profiles. This diversity reper-
cutates on the legal basis, as can be seen in Figure 7.

Evaluating RAG for French immigration law 13
B Normalisation and matching hypotheses for the
evaluation protocol
B.1 Permit matching details
Priortomatching,generatedpermitsarenormalisedthroughcasefolding,accent
removal, removal of parenthetical content, and suppression of legal reference
strings. When a prediction contains multiple alternatives separated by “or”, only
the first is retained, reflecting the expectation that the system should provide a
single primary recommendation.
ForEU/EEA/Swissnationals,predictionsmappedtoUE_CITIZENaretreated
as equivalent toNO_PERMIT. Both recommendations correctly reflect the free-
movement regime and the absence of a legal obligation to obtain a residence
permit. The gold standard nevertheless favoursNO_PERMIT, as this formulation
provides the most direct and operational guidance for HR practitioners and
applicants.
B.2 Document matching details
Canonical layer.The rule-based canonical layer resolves frequent surface-form
variants prior to embedding-based matching. Covered categories include identity
photos (e.g.3 photos d’identité,e-photo avec code), proofs of address, passports
and travel documents, OFII medical certificates, work authorisations, and mar-
riage certificates with or without transcription clauses.
Threshold calibration.The cosine similarity threshold for BGE-M3 embedding
matching was calibrated over0.65,0.70,0.75,0.80on the full evaluation subset.
Lower thresholds increased recall but frequently matched semantically related
yet administratively distinct documents. We therefore retained the most conser-
vative setting (0.80), prioritising precision over recall to avoid conflating docu-
ments with similar wording but different administrative purposes. This choice is
consistent with the evaluation objective, where false matches would artificially
inflate document-level performance.
Administrative consistency filters.Additional post-matching filters invalidate
semantically matched document pairs when critical administrative constraints
disagree. Two filters are applied. First, euro-denominated amounts must match
exactly,preventingconfusionsbetweenfee-relateddocumentsofdifferentdenom-
inations (e.g. a€99 visa tax versus a€225 residence permit fee). Second, proofs
of address containing explicit validity requirements must agree on the duration
constraint (e.g. less than 3 months versus less than 6 months). These constraints
are frequently overlooked by embedding similarity and are therefore enforced
explicitly during matching.

14 A. Abtout et al.
C Results
C.1 Generation prompt
The prompt comprises four functional blocks: (i) asystem roleand source hier-
archy, (ii)critical rulesencoding known legal special cases, (iii) an orderedtask
listscaffolding the reasoning, and (iv) a strictoutput specification.
Three design choices are central. First, legal sources are ranked by authority
(bilateral agreements, then the CESEDA, the Employment Act, and the admin-
istrative portal Service-Public), so the model resolves conflicts deterministically.
Second, hard-coded rules capture cases where naive retrieval is known to fail:
free-movement rights for EU/EEA/Swiss nationals, the precedence of the 1968
Franco-Algerian Agreement over the CESEDA for Algerian nationals, and the
effect of shortage-occupation status on eligibility. Third, the model must abstain
whenever no authoritative citation supports a recommendation, discouraging
unsupported or hallucinated legal claims.
Accompanying quality controls constrain the citation format (one article per
line with its collection and canonical reference, e.g.L.123-45for the CESEDA
and Employment Act, or theF-prefixed identifier for Service-Public) and require
the response to specify exact tax amounts, payment procedures, and the recency
requirement for proof of address. The full template appears in Figure 8.
C.2 Permit matching
Table 4 summarizes the main permit recommendation errors identified through
manual inspection of model outputs. The most frequent failure modes involve
confusionbetweenlegalregimes,permitfamilies,andpermitdurations,aswellas
errors on administrative attributes such as renewal status and visa prerequisites.
We do, however, observe some differences between Qwen-27B and Qwen-9B.
The 9B model produces more vague or generic recommendations, and tends to
fall back on the most common permit categories for foreign workers in France
(employeeand Talent), even when a more specific category would be correct
or when no permit is required, as for EU nationals. Overall, though, the same
families of errors appear across both models.
C.3 Document matching results
Table 5 reports document-set matching performance under the strict full permit
match filter. As discussed in the main paper, document evaluation is conditioned
on correct permit recommendation and should therefore be interpreted as a
downstream assessment of document retrieval quality rather than an end-to-end
system score.
To better characterise residual failures, Table 6 reports the document cate-
gories most frequently omitted by each system. The results show that retrieval
substantiallyimprovesrecoveryofseveralpermit-specificrequirements,butsome
procedural documents remain frequently missing even when the correct permit
category has been identified.

Evaluating RAG for French immigration law 15
Table 4: Qualitative error categories observed during permit recommendation
evaluation.
Error category Description
Legal regime confusion Confusion between permits governed by different legal frame-
works, most notably CESEDA permits and residence certificates
issued under the Franco-Algerian agreement.
Permit family confusion Confusion between Talent permits and standard salaried per-
mits, or between employment-based and family-based permits.
Duration confusion Confusion between temporary (carte de séjour temporaire) and
multi-year (carte de séjour pluriannuelle) permits despite cor-
rect identification of the underlying permit family.
Renewal omission Correct permit category predicted but renewal status omitted
or incorrectly inferred.
Visa prerequisite omission Failure to identify required visa prerequisites despite otherwise
correct permit recommendations.
Mention-level error Incorrect statutory mention despite correct permit family. Such
errors were comparatively rare once the permit type was cor-
rectly identified.
C.4 Legal basis matching results
Table 7 reports citation coverage under the strictfull_permit_matchfilter.
Results are broken down by legal source type (CESEDA, Service Public, and bi-
lateral agreements). Coverage scores should be interpreted as conservative lower
bounds because the gold standard contains only a minimal set of supporting
references.

16 A. Abtout et al.
Table 5: Document-set matching performance under the strict full permit match
filter at cosine similarity threshold 0.80. The strict subset contains 9 evaluated
profiles for Qwen-27B in the LLM-alone setting, 12 for Qwen-27B with Dense
RAG, 5 for Qwen-9B in the LLM-alone setting, and 8 for Qwen-9B with Dense
RAG.
Model Condition Precision Recall F1
Qwen 27B LLM-alone 28.3% 21.0% 23.9%
Dense RAG 55.3% 39.5% 45.3%
∆+27.0 pts +18.5 pts +21.4 pts
Qwen 9B LLM-alone 26.2% 21.8% 23.5%
Dense RAG 44.3% 31.5% 36.4%
∆+18.2 pts +9.7 pts +12.9 pts
Table 6: Most frequently missing document categories under the strict
full_permit_matchsubset at cosine threshold 0.80. Each cell reports the num-
ber of evaluated profiles for which at least one gold-standard document in the
category was not recovered.
Document LLM-alone Dense RAG
Qwen-27B(n= 9LLM-alone;n= 12Dense RAG)
Signed commitment to respect the principles of the French Republic 9/9 11/12
Administrative fee receipt / tax stamp 9/9 11/12
Proof of address (less than 6 months) 8/9 4/12
Degree copy / diploma evidence 6/9 9/12
Qwen-9B(n= 5LLM-alone;n= 8Dense RAG)
Signed commitment to respect the principles of the French Republic 5/5 8/8
Administrative fee receipt / tax stamp 4/5 7/8
Degree copy / diploma evidence 4/5 5/8
Birth certificate / full birth record 3/5 3/8

Evaluating RAG for French immigration law 17
(a) All errors considered
(b) "No permit required" and "Residence permit for EU-
/EEA/Swiss citizens" considered equals
Fig.1: Pairwiseκon permit-type annotations (training profiles excluded)

18 A. Abtout et al.
Fig.2: Confusion matrix for the permit annotation (counts). ‘Renewal’ and visas
mentions are ommited. Calibration profiles excluded.

Evaluating RAG for French immigration law 19
Fig.3: Pairwise F1 on required documents (calibration profiles excluded, and
only profiles for which the annotators agreed on the permit-type
Fig.4: Pairwise F1 on legal basis (calibration profiles excluded, and only profiles
for which the annotators agreed on the permit-type

20 A. Abtout et al.
Fig.5: Distribution of recommended permit types (regardless of renewal status)
Fig.6: Distribution of recommended documents per permit type (regardless of
renewal status)

Evaluating RAG for French immigration law 21
Fig.7: Legal basis distribution per permit type (regardless of renewal status)
Table 7: Citation coverage under the strict full permit match filter, broken down
by legal source family. Reported scores correspond to gold-coverage recall and
should be interpreted as conservative lower bounds because the gold standard
contains only a minimal set of supporting references. The strict subset contains
13 evaluated profiles for Qwen-27B in the LLM-alone setting, 18 for Qwen-27B
with Dense RAG, 5 for Qwen-9B in the LLM-alone setting, and 13 for Qwen-9B
with Dense RAG.
Metric LLM-alone Dense RAG
Qwen 27B
Overall 7.7% 46.6%
Service-Public 0/14 (0.0%) 11/20 (55.0%)
CESEDA 2/19 (10.5%) 8/28 (28.6%)
Bilateral 0/2 (0.0%) 4/4 (100.0%)
Qwen 9B
Overall 10.0% 13.1%
Service-Public 0/5 (0.0%) 1/14 (7.1%)
CESEDA 1/6 (16.7%) 5/22 (22.7%)
Bilateral 0/1 (0.0%) 0/2 (0.0%)

22 A. Abtout et al.
You are a French immigration law consultant. I am an HR manager at a firm
in {cityOfJob}, {postalCodeOfJob}, France.
From the applicant’s profile, recommend the appropriate residence permit
type to find relevant legal texts and official eligibility guidelines
. Cite specific articles. If no authoritative citation exists, state
"legal uncertainty" instead of recommending.
Legal sources (priority order):
1. Bilateral agreements
2. CESEDA
3. Code du Travail
4. Service-Public
Critical rules:
- EU/EEA/Swiss citizens: Check free movement rights first
- Algerians: Apply Franco-Algerian Agreement (1968) for all substantive
rules; CESEDA only for procedural gaps. NOT eligible for CESEDA
permit types.
- Metier en tension: Verify shortage occupation status -- significantly
affects eligibility
Tasks:
1. Check EU/EEA/Swiss status -> bilateral agreement France-{nationality
} -> CESEDA eligibility articles
2. Verify metier en tension status
3. Find salary thresholds (Code du Travail / CESEDA carte Talent)
4. Determine qualifying permit type(s) and required documents
(Service-Public / CESEDA)
5. Recommend the most appropriate permit with legal justification and
document list
Reason internally. Output only the final result as valid JSON in French,
matching the provided output format.
{
"general_recommendation": "<type of permit recommended>",
"justification": "<reasoning for why this permit type fits the
applicant’s profile; no legal citations here>",
"required_documents": ["<doc1>", "<doc2>", "<doc3>"],
"legal_citations": ["<citation1>", "<citation2>", "<citation3>"]
}
Quality controls:
- Cite the exact article number and collection (CESEDA, Code du Travail
, etc.): format FXXXX for Service-Public, and L./R./D.123-45 for Code
du Travail and CESEDA. Only one article per line.
- Never recommend a permit without a supporting legal citation (article
and collection).
- Always give exact tax amounts and payment procedures for the
recommended permit, citing the relevant legal articles.
- For proof of address, always state the recency requirement.
Fig.8:Role-conditioneduserprompt.Tokensinbracesaresubstitutedatruntime
from the applicant profile.