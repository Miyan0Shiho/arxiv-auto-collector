# Curated retrieval versus open web search in public AI information services: a coverage-trust trade-off

**Authors**: Hafsteinn Einarsson, Hafsteinn Birgir Einarsson, Jón Gunnar Ólafsson, Jón Gunnar Þorsteinsson

**Published**: 2026-07-06 15:32:46

**PDF URL**: [https://arxiv.org/pdf/2607.05217v2](https://arxiv.org/pdf/2607.05217v2)

## Abstract
Public institutions increasingly use large language models (LLMs) to answer citizens' questions, often pairing a curated knowledge base with live web search, yet whether the sources behind these answers can be trusted has received little empirical scrutiny. We report a pre-launch expert evaluation of Evrópuvefur, an independent, government-funded service run by the University of Iceland that answers questions about the European Union, conducted as Iceland prepared for its referendum of 29 August 2026 on whether to resume EU accession talks. Five domain experts produced 551 evaluations of 449 AI-generated answers, scoring each against a seven-criterion quality rubric and, separately, flagging individual cited sources. We compared two retrieval paths: a curated local corpus (RAG) and open web search. In more than a third of the reviewed web-search answers (35%, 65 of 187), at least one cited source was flagged, almost always as untrustworthy or irrelevant; curated sources were flagged far less often and only for being out of date. Web search answered more questions, but at the cost of source quality; the curated corpus was trustworthy yet limited in coverage, and the model declined to respond when it fell short. The citation mix also passed over strong sources: across all 287 web-search answers, the system never cited RÚV, the public broadcaster and the country's most widely used news source. A companion prompt ablation shows how weak prompt-level steering is: a trusted-domain list in the system prompt raised the share of citations to listed domains only from 12% to 21%. Fluency and topical fit did not predict source trustworthiness. We argue that source trustworthiness is a measurable yet largely invisible dimension of information quality in public AI services, and we discuss transparency-oriented responses and their trade-offs.

## Full Text


<!-- PDF content starts -->

CuratedretrievalversusopenwebsearchinpublicAI
information services: a coverage–trust trade-off
Hafsteinn Einarsson‗1, Hafsteinn Birgir Einarsson2, Jón Gunnar Ólafsson2, and Jón
Gunnar Þorsteinsson3
1Faculty of Industrial Engineering, Mechanical Engineering and Computer Science, University of Iceland,
Reykjavík, Iceland
2Faculty of Political Science, University of Iceland, Reykjavík, Iceland
3The Icelandic Web of Science, University of Iceland, Reykjavík, Iceland
Preprint, July 2026
Abstract
Public institutions increasingly use large language models (LLMs) to answer citizens’
questions,oftenpairingacuratedknowledgebasewithlivewebsearch,yetwhetherthe
sources behind these answers can be trusted has received little empirical scrutiny. We
reportapre-launchexpertevaluationofEvrópuvefur,anindependent,government-funded
servicerunbytheUniversityofIcelandthatanswersquestionsabouttheEuropeanUnion,
conductedasIcelandpreparedforitsreferendumof29August2026onwhethertoresume
EUaccessiontalks. Fivedomainexpertsproduced551evaluationsof449AI-generated
answers, scoring each against a seven-criterion quality rubric and, separately, flagging
individual cited sources. We comparedtwo retrieval paths: a curated local corpus (RAG)
and open web search. In more than a third of the reviewed web-search answers (35%,
65 of 187), at least one cited source was flagged, almost always as untrustworthy or
irrelevant; curated sources were flagged far less often and only for being out of date. Web
search answered more questions, but at the cost of source quality; the curated corpus
wastrustworthyyetlimitedincoverage,andthemodeldeclinedtorespondwhenitfell
short. The citation mix also passed over strong sources: across all 287 web-search answers,
thesystemnevercitedRÚV,thepublicbroadcasterandthecountry’smostwidelyused
newssource. A companionprompt ablationshowshow weakprompt-level steeringis: a
trusted-domainlistinthesystempromptraisedtheshareofcitationstolisteddomains
only from 12% to 21%. Fluency and topical fit did not predict source trustworthiness.
Wearguethatsourcetrustworthinessisameasurableyetlargelyinvisibledimensionof
information quality in public AI services, and we discuss transparency-oriented responses
and their trade-offs.
Keywords:artificialintelligence·largelanguagemodels·retrieval-augmentedgeneration
·information quality·source trustworthiness·public information·expert evaluation
‗Corresponding author:hafsteinne@hi.is
1
arXiv:2607.05217v2  [cs.CY]  7 Jul 2026

Curated retrieval versus open web search2
1 Introduction
Publicbodieshavebegunusinglargelanguagemodels(LLMs)toanswercitizens’questions
directly,andgenerativeAIisalreadyinwidespread,ifuneven,useacrossgovernment
(Brightetal.,2025;OECD,2024). Iftheaimistogroundanswersinrealmaterialrather
thanthemodel’sownparametricmemory,atleasttwoapproachesareavailable: retrieval-
augmented generation (RAG) over a curated knowledge base the institution controls
(Lewis et al., 2020), and live web search over the open internet. We study these two paths
separately. The appeal of either is plain. One system can answer far more questions than
a hand-written FAQ, in the citizen’s own language, at any hour.
SomerisksofputtingLLMsinfrontofcitizensarebynowfamiliar. Apublic-sector
chatbotcangiveconfidentlywrongadvice;NewYorkCity’sbusiness-helpbotwascaught
telling firms to break the law (Offenhartz, 2024), and early systems hallucinated and
mishandled citations. Those failures have eased as models matured, and they are not our
subject. A subtler risk persists even when the prose is fluent and the citations resolve: the
sourcesthesystemleansonmaynotdeservethetrustreadersplaceinthem. Readerswere
neverreliablejudgesofonlinesourcecredibility,longbeforeLLMsenteredthepicture
(Metzger,2007),andanLLMremoveseventherankedlistoflinksthatasearchengine
onceleftthemtoinspect. Recentauditsmaketheconcernconcrete: acrosseightAIanswer
engines,citationstonewssourceswerewronginmosttests(JaźwińskaandChandrasekar,
2025). Citationbehaviourhasimprovedasmodelshavematured,anditremainsimperfect
even in frontier systems built for the task (Onweller et al., 2026); through 2024, reliability
did not climb steadily as models scaled (L. Zhou et al., 2024), even if the newest systems
appeartohavenarrowedsomeofthesegaps. Forpublicinformationthismattersmore
than for casual search. Citizens often have little choice but to turn to a public service, and
whether they trust the information it gives rests on their trust in the institution behind it
(BélangerandCarter,2008);oncequestioned,thattrustcanerodeintoaself-reinforcing
spiral of distrust (Ólafsson and Jóhannsdóttir, 2024). Someone consulting a public service
about a contested political question is not after a plausible paragraph; they are relying on
the institution to stand behind the facts and the sources.
Source trustworthiness is, in principle, a dimension of information quality, alongside
accuracy,currency(howuptodateasourceis),andrelevance(howwellitfitsthequestion
asked)(WangandStrong,1996). Itisalsoamongtheeasiesttomanipulate: ajournalist
recently seeded a fabricated claim on a personal page and watched mainstream AI
assistantsrepeatitasfactwithinaday(Germain,2026),areminderthatasystemdrawing
ontheopenwebcanabsorbandamplifymisinformation.1YetpublicAIservicesrarely
measure trustworthiness. They log queries and answers; they seldom record whether the
citedsourceswouldsurviveexpertscrutiny. Thesegapstendtobelargerwherethestakes
arehigher: inlanguageswithfewhigh-qualityresources,andonpoliticallychargedtopics
whereunreliablematerialismoreabundant. WhetheritisevenanAIprovider’splace
to decide which sources count as trustworthy is itself contested, since source scrutiny
shades into editorial control and what makes a source trustworthy is unsettled; a lighter
1Weusemisinformationforthebroaderphenomenonoffalseormisleadinginformation,irrespectiveof
intent, rather than the narrowerdisinformation, which implies deliberate deception (Wardle and Derakhshan,
2017).

Curated retrieval versus open web search3
alternative would let users constrain which domains a query may draw on. We return to
these questions in the discussion.
We study this gap, between benchmark evidence and the sources a public service
would actually cite, in a concrete setting. Evrópuvefurinn (“The Icelandic Web on
European Affairs”) is an independent, government-funded service, run by the Icelandic
WebofScienceattheUniversityofIceland,thatanswersquestions,inIcelandic,aboutthe
European Union (EU) and Iceland’s relationship with it. It has received funding from
the state but operates independently, an arrangement that makes its credibility central
to its mandate. The service had lain largely dormant since Iceland’s last accession bid
stalledin2013;asthecountrypreparedforareferendumscheduledfor29August2026
on whether to resume EU accession talks, it was being reactivated for a renewed wave
of public questions. Ahead of that relaunch we ran a pre-deployment comparison of
two ways the system could answer: grounded in a curated corpus of vetted Evrópuvefur
articles(RAG),oronopenwebsearch. Thesystemgeneratedanswersundercontrolled
conditions,whichexpertsthenreviewed,weighingacontrolledsourceofevidenceagainst
an uncontrolled one before either reached the public.
Overroughlyeightweeks,fromearlyMaytothestartofJuly2026,fivedomainexperts
reviewed the answers. The questions were not real user queries but a fixed set of 287
questions, generated to span the live debate (Section 4); each was answered twice, once in
eachmode,sothetwopathscouldbecomparedonidenticalinputs. Reviewersscored
eachansweronaseven-criterionqualityrubricand,separately,flaggedindividualcited
sources they judged unfit. This produced 551 scored evaluations and 128 source flags.
Threefindingsmotivatethepaper. First,sourceproblemswerecommon,andwhere
the system reached the open web the dominant complaint was trust: experts judged most
flaggedwebsourcesuntrustworthyratherthanmerelyoff-topic. Second,thetworetrieval
paths embodieda coverage–trust trade-off. Websearch answered thequestion farmore
often,becausetheopenwebnearlyalwayshassomethingonpoint,butitfrequentlyrested
onsourcesexpertswouldnotendorse. Thecuratedcorpuswastrustedbyconstruction
yet often lacked coverage, and when it did the model said so rather than inventing an
answer. Third, source trustworthiness was largely decoupled from answer quality: once
wesetasidewhetheranansweraddressedthequestionatall,mostofthegapbetween
the modes closed, and within the web path an answer’s fluency and topical fit carried no
signal about whether its sources were sound.
Our aim is to make this risk visible and measurable, not to fix it, on the premise
that an institution cannot govern or disclose what it does not first measure. We treat
sourcetrustworthinessasaninformation-qualitydimensionthatpublicAIservicescan
and should assess, we show one practical way to assess it, and we weigh responses
ranging from provenance disclosure to expert source labelling against their costs. We
are deliberately cautious about remedies that restrict which sources a system may use,
because curation by a public body carries its own risks to open access to information.
Concretely,weask how trustworthythe sources citedby the web-search path are, as
judgedbydomainexperts;howthetworetrievalpathscompareonanswerqualityand
on the reasons reviewers flagged their sources, including what the reviewers’ written
comments reveal; whether steering the model toward trusted domains in its prompt
changeswhatitcites, whichwetestwithacontrolledpromptablation; andwetakeup, in

Curated retrieval versus open web search4
the discussion rather than the results, what these patterns imply for the governance and
transparency of public AI information services.
Thepapermakesthreecontributions. Itprovidesexpert-evaluatedevidenceonthe
trustworthiness of the sources a public AI information service would cite, in a low-
resource language and a high-stakes civic setting. It introduces a simple, reusable review
instrument that separates answer quality from per-source trustworthiness. And it frames
sourcetrustworthinessasameasurableinformation-qualitydimensionforpublicAI,with
a transparency-oriented discussion of what institutions might do about it.
2 Background
2.1 AI and large language models in the public sector
Public administrations have moved quickly to put LLMs in front of citizens. A 2024
surveyofUKpublic-sectorprofessionalsfoundgenerativeAIalreadyinwidespread,if
disorganised,use,withlittlegoverningguidance(Brightetal.,2025),andcross-government
reviews report rapid, uneven adoption alongside shared concerns about accuracy and
oversight (OECD, 2024). Independent audits bear the concern out: testing leading
LLMs on tens of thousands of public-service questions, the Open Data Institute found
inconsistent and sometimes inaccurate answers, and a recurring failure to signal what the
model did not know (Majithia et al., 2026). Conversational interfaces are a common entry
point, and a growing body of e-government research examines government chatbots:
which social characteristics citizens prefer (Ju et al., 2023), whether a chatbot should
present a civil-servant identity (Li and Wang, 2024), how chatbots reshape public-service
provisionandpublicvalue(LarsenandFølstad,2024),andhowcitizensandfront-line
officers weigh their design (Hemesath and Tepe, 2024). Because capabilities shift quickly,
evidenceonanyspecificmodeldatesfast, soweweightrecentauditsandtieourresults
tothesystemsandperiodwestudied. Themovefromretrievaltogenerationalsochanges
the risk profile: a search engine returns ranked links a user can inspect; an LLM returns a
single composed answer whose provenance can be harder to see.
2.2 Information quality and trust in public information
Whetherpublicinformationcanbetrustedis,atroot,aquestionofinformationquality.
Established frameworks treat quality as multi-dimensional, with accuracy, completeness,
currency, relevance, and believability among the dimensions information consumers
actuallycareabout(WangandStrong,1996). Sourcetrustworthinessisonesuchdimension:
ananswercanbeaccurateinitswordingyetrestonasourceareaderwouldnotaccept.
Ine-government,acitizen’suseofaservicedependsontrustintheinstitutionbehindit
(Bélanger and Carter, 2008), so a service that cites questionable material risks more than a
single bad answer: it risks the institutional standing on which its usefulness depends.
What counts as a trustworthy source is not merely a matter of accuracy. It is useful to
distinguish edited, mainstream (or legacy) journalism from the alternative media and
onlineplatforms(blogs,opinioncolumns,partisanoradvocacysites). Mainstreamoutlets
are bound by editorial standards, and their role is to cover the different sides of an issue

Curated retrieval versus open web search5
and to act as a watchdog or “fourth estate”: to hold those in power to account and
to inform the public. The function of alternative outlets is more commonly to express
positionsandspecificviewpointsratherthantodisseminatetraditionalnewsreporting
(Ólafsson and Jóhannsdóttir, 2024). That content can be a legitimate and valuable part of
debate in an open democratic society, but it is not the same as an editorially vetted source
forfactualclaims. Thisdistinctionissharperinasmall mediamarket,whereathinand
fragilepress ismore easilydominated andwhere outsideactors canmore readilyshape
theagenda(Ólafsson,2021;ÓmarsdóttirandÓlafsson,2024);theseconditionsraisethe
stakes of which sources a public AI service amplifies.
These risks are not uniform across languages: audits of chatbots verifying political
claimsfindthataccuracyvariesbylanguageandisweakeroutsidehigh-resourceones
(Kuznetsovaetal.,2025),whichmattersforaserviceoperatinginIcelandic,alanguage
spokenbyonlyabout370,000peopleandcorrespondinglythininhigh-qualitymachine-
readable text. Models are improving in Icelandic, but they still do better when the same
task is posed in English than in Icelandic (Einarsson, 2026b,a), so a public service tool
providing answers in a smaller language starts from a comparative disadvantage, one
that plausibly extends to other low-resource languages in a similar position.
2.3 Retrieval-augmented generation, provenance, and source credibility
RAG was introduced to ground model output in retrieved documents and reduce
unsupported claims (Lewis et al., 2020). Grounding helps, and outright hallucination has
grown less common in capable models, though it has not disappeared (Ji et al., 2023).
Fabricatedandmiscitedreferences,oncerife(WaltersandWilder,2023;Jaźwińskaand
Chandrasekar, 2025), are likewise diminishing as leading models and answer engines
improve at attribution, even if they remain imperfect (Onweller et al., 2026). We treat
these as receding problems and concentrate on one that does not recede with scale: even
a system that cites flawlessly still has to draw on sources that are themselves trustworthy.
This is where the two retrieval paths diverge. A curated corpus lets a service trust its
sourcesbutcannotcovereveryquestion;openwebsearchcanfindmaterialforalmostany
questionbutinheritstheopenweb’sunevenquality. Grounding,inotherwords,moves
theproblemratherthanremovingit. Thequestionbecomeswhetherthesourcesasystem
retrieves and cites would survive scrutiny, and, before that, what we even mean by a
trustworthy source, a point we take up in the discussion. Judging the credibility of a web
sourceis, inanycase, askill readershave alwaysapplied unevenly (Metzger,2007), and
surveysofRAGtrustworthinessnotethatthefieldhasemphasisedaccuracyandefficiency
while the trustworthiness of retrieved sources stays underexamined (Y. Zhou et al., 2024).
2.4 The gap
Two research gaps follow from this literature. First, most evidence on LLM reliability
comesfrom benchmarkprompts ratherthan thesources apublic servicewould actually
cite when answering questions in its domain. Second, few studies cover low-resource
languagesorhigh-stakescivicmomentssuchasreferenda,wheretrustworthymaterial
is scarcer and unreliable material more abundant, as the surge of hyperpartisan and

Curated retrieval versus open web search6
misleading content around the 2016 Brexit referendum illustrated (Bastos and Mercea,
2019; Marshall and Drieschova, 2018). We address both gaps by evaluating, before
deployment, the quality of the sources that an Icelandic-language public service cites
when answering a broad set of questions about a national referendum.
3 Case and context
3.1 Evrópuvefur and the 2026 referendum
Evrópuvefurinn is an Icelandic-language information service about the European Union,
runbytheIcelandicWebofScienceattheUniversityofIceland. Itsremitistogivethe
publiceven-handed,sourcedanswersabouttheEUandIceland’stiestoit. Theservice
has received state funding (it ran on public funding in 2011–2013 and received a smaller
grant again in 2026) but is editorially independent, and that independence is part of what
it offers: answers that are not steered by the government of the day. It was opened in
2011,aroundIceland’searlieraccessionbid(IcelandappliedtojointheEUin2009and
suspended negotiations in 2013), to explain that process to the public, and it then lay
largely dormant for a decade.2
The EU question returned to the agenda in 2026. The 2024 Alþingi election returned a
newcentre-leftgovernment,thefirstadministrationnotopposedtoEUaccessionsince
2013 (Einarsson et al., 2025), which put the question back on the agenda. The government
proposedthevoteinMarch(GovernmentofIceland,2026),andon28May2026theAlþingi
(Iceland’s national parliament) approved a national referendum, scheduled for 29 August
2026, asking whether Iceland should resume accession negotiations with the EU (Alþingi,
2026). The vote would not decide membership; it would decide only whether talks,
dormant since 2013, should restart. Generative AI sharpens the information environment
aroundsuchavote. Evidencefromelectionselsewhereshowsthemechanism,though
none of it concerns Iceland directly: by the 2024 cycle, language models could already
produceelectionmisinformationthatreaderscouldnotdistinguishfromgenuinematerial
(Williams et al., 2025), even if models varied in how readily they complied with such
prompts (Schlicht, 2024). The technology has matured since: today’s models write more
fluently and refuse such prompts more consistently. What has not been settled is whether
the material these systems retrieve and cite can be trusted, the risk a public service must
manage when it answers EU-related questions in a charged referendum period.
3.2 The system
The service answers a question with an LLM grounded in retrieved evidence. In RAG
mode it draws on the Evrópuvefur archive: 742 expert-written, editorially approved
answersby87contributorsacross28topicareas,thelargestbeingEUaffairs,produced
between 2011 and 2013 during Iceland’s earlier accession process and not updated since
(Evrópuvefurinn, 2013). About 670 EU-relevant answers form the curated corpus. Its age
mattersforwhatfollows: itiswhycuratedsourceswereflaggedforstaleness,anditispart
of the coverage gap, since questions about 2026-specific developments often fall outside a
2See the service’s own account of its origins:https://www.evropuvefur.is/svar.php?id=70881.

Curated retrieval versus open web search7
corpusfrozenin2013. Questionsandarticlesareembeddedwith multilingual-e5-large ,
amultilingualencoderthatperformsstronglyontheMassiveTextEmbeddingBenchmark
(Muennighoffetal.,2023),andtheclosestarticlesareretrievedbyapproximatenearest-
neighbour search. Answers are generated with Google’s Gemini models (a Pro model for
most queries, a Flash model as fallback), which were among the best-performing models
inIcelandicatthetimeofthestudy,3withtheretrievedarticlessuppliedascontextand
cited in the answer.
For this evaluation the system answered each question in one of two modes, recorded
foreveryquery. InRAGmodethemodelisgroundedinthecuratedlocalcorpusdescribed
above. Inweb-searchmodethe modelrunsitsownwebsearches, withfull controloverthe
queries it issues and the pages it draws on, and cites external web pages. Its instructions
explicitly steered it toward reputable, primary, and authoritative sources (a guiding list of
domains that esbvaktin.is classifies as “high confidence”, which the prompt asks the
model to prefer but which is not enforced as a hard constraint; Section 4) and away from
blogs, opinion columns, and partisan sites; we give both modes’ prompts verbatim in
Appendix B and return to the effect of that instruction in the discussion. The two modes
giveadirectcontrastbetweenacontrolledsourceofevidence,whichtheinstitutionhas
vetted, and an uncontrolled one drawn from the open web. This contrast is the backbone
of the analysis.
4 Methods
4.1 Question generation
Because the service was not yet public, we could not draw on real user queries, so
we generated a broad, self-contained question set designed to span the public debate.
The source material came from esbvaktin.is , chosen because it is an open, nonprofit,
and transparent EU-referendum fact-checking project built independently by a doctoral
studentattheUniversityofIceland(seeAcknowledgements),notanofficialUniversity
project: it tracks the public discussion around the referendum, collects parliamentary
speeches, fact-checked claims, evidence records, and media reports, and classifies the
domains it draws on by confidence level, a ready-made and documented classification
we could reuse to guide source selection in web-search mode. We weigh the risk that
this introduces (an LLM-assisted aggregator shaping our own pipeline) in Section 7. We
embedded passages from this corpus and clustered them (UMAP for dimensionality
reduction, then HDBSCAN) to surface the recurring themes of the debate. Within
each cluster we chose seed passages by maximal-marginal-relevance selection: a greedy
procedurethatbuildsasetoneitematatime,eachtimeaddingthepassagethatscores
highest on a weighted combination of relevance to the cluster and dissimilarity to the
passages already chosen. Intuitively, it avoids picking several near-identical passages,
so the seeds cover a theme broadly rather than restating its single most typical point.
A generation model (Google Gemini 3 Pro) wrote candidate questions from each seed;
a second model checked that each question stood on its own without referring back to
3Judged against the public Icelandic LLM leaderboard maintained by Miðeind (Miðeind, 2026).

Curated retrieval versus open web search8
a source text, and near-duplicates were removed by embedding similarity. The result
was 287 questions spanning six types: policy reasoning, comparative cases (for example
Norway or the European Economic Area, EEA), competing discourse positions, historical
context, the referendum itself, and common misconceptions. They were deliberately
weighted toward reasoning rather than trivia, to mirror the questions actually circulating
in the debate.
4.2 Data
The 287 generated questions, answered in both modes, produce 574 eligible answers,
evenly split (287 RAG, 287 web search); queries logged during development and testing,
andanswersgeneratedafterthestudywindowforaseparatebatchofquestionsasthe
servicemovedtowardproduction,areexcluded. Fivereviewersproduced551evaluations
covering 449 distinct answers (262 RAG, 187 web search) and 128 source flags. They
also edited 103 answers and marked them ready for publication; whether any of those
is formally published is a later editorial decision, taken outside this study. Reviewers
worked through a shared queue rather than a balanced assignment, so the two modes
were reviewed in unequal numbers, a limitation we return to in Section 7. For the
179 questions reviewed in both modes the two answers were produced from the same
underlying question, each mode wrapping it in its own prompt. This matched subset
lets us compare the modes while holding the question fixed, which controls for question
difficulty: we use McNemar’s test for the binary “answers the question” criterion and the
Wilcoxon signed-rank test for the composite score. We do not run a paired comparison of
source flags, because the available flag reasons differ by mode (Section 4) and so are not
comparable across the pair.
4.3 Expert review instrument
FivedomainexpertsreviewedAI-generatedanswersthroughadedicatedreviewinterface.
The instrument has two independent parts.
Thefirstisaqualityrubricofsevenyes/nocriteriaappliedtotheanswerasawhole:
whether it answers the question, is factually accurate, draws onrelevantsources, is freeof
hallucinations, keeps to an appropriate scope, reads well in Icelandic, and is publishable
with only minor edits. We report the per-criterion pass rates and summarise the rubric as
a composite score, the number of criteria an answer passes, from 0 to 7.
Thesecondisasource-flaggingmechanismthatoperatesonindividualcitedsources
rather than the answer as a whole. A reviewer who judged a cited source unfit flagged it
andrecordedareason,andtheavailablereasonsdifferedbymode,bydesign. Curated
local articles had already passed editorial vetting, so reviewers assessed them only for
currency: theoneapplicablereasonwasoutdated. Webpages,whichenteredananswer
with no prior vetting, were assessed instead for whether they wereuntrustworthyor
irrelevant. This asymmetry reflects the question we set each path. For the curated corpus,
has the vetted material aged? For the open web, did the system reach material worth
citingatall? Italsomeansflagreasonsarenotdirectlycomparableacrossmodes,which
we keep in view when interpreting them. Reviewers could add a free-text comment, and

Curated retrieval versus open web search9
most did(90 of 128flags). We analysethose comments inSection 5.3 withthe help ofan
LLM, to see what reviewers actually objected to. Separating per-source flags from the
whole-answer rubric is what lets us ask whether a good answer can still rest on a bad
source.
4.4 Analysis
We report proportions with Wilson 95% confidence intervals (Wilson, 1927). We compare
modeswiththechi-squaredtest,orFisher’sexacttestwhereexpectedcountsaresmall,
and summarise effect sizes with Cramér’s 𝑉and odds ratios; for the paired subset we
use the Wilcoxon signed-rank test. For inter-rater reliability we use Gwet’s AC1 as the
headline measure (Gwet, 2008), because it is robust to the high-prevalence problem that
distorts Cohen’sand Fleiss’kappa whenmost judgmentsfall inone category; wereport
kappaandKrippendorff’salpha(Krippendorff,2018)alongside it,andreadcoefficients
against conventionalthresholds (Landisand Koch,1977). To categorisethe free-text flag
comments,webuiltafixedsetofreasoncodesbyreadingallofthem,thenhadanLLM
(Gemini 3.5 Flash, with a constrained structured-output schema) assign every applicable
code to each comment; a comment can carry more than one. The figures and statistics
are reproducible from the evaluation export and analysis code described in the Data
availability statement.
4.5 Review procedure
Reviewers signed in to a dedicated web interface and were served one query at a time
in randomised order, with queries that no one had yet evaluated prioritised so that
corpus coverage built up before annotators doubled up on the same items (which is why
inter-rater overlap is sparse; Section 5.8). Each reviewer saw the question and the full
generatedanswer withits inline citations, scored the seven rubriccriteria, and flagged
any cited source in the same pass. The interface presented the rubric items and flag
reasons in English (the flag labels were “Outdated”, “Irrelevant”, and “Not trustworthy”);
the answers under review, and the reviewers’ free-text comments, were in Icelandic.
Because the cited sources were visible (local article links in RAG mode, external URLs
inweb-searchmode),theretrievalmodewasnotblinded;thisisanunavoidablefeature
of source review, since judging a source means seeing it, and we note it as a possible
influence on judgments. Reviewers could not see one another’s evaluations.
4.6 Prompt-ablation experiment
To isolate what the trusted-domain list in the web-search prompt actually does, we ran a
controlled prompt ablation, an experiment that removes one component of a system, here
thetrusted-domainlist,tomeasurewhatthatcomponentcontributes. Itranalongsidethe
expertreview,inApril2026. Eachofthe287studyquestionswasansweredtwicemoreby
theproductionsetup(Gemini3ProthroughthesameAPIrouteandweb-searchplugin
the deployed system uses), with one difference in how citations are captured: instead
ofafree-textanswerwhosereferencelistweparse,theablationconstrainsthemodelto
a structured output with an explicit citation list. The two arms are: once with the full

Curated retrieval versus open web search10
production-style prompt including its trusted-domain list, and once with the “Source
selection” section stripped, everything else identical. Of the 574 requests, 549 completed
(275 with the list, 274 without); failed requests returned no answer and drop out of both
arms. RedirectURLsreturnedbythesearchtoolwereresolvedtotheirtargetpagesbefore
analysis, and every citation was classified ason-list(its host matches a listed domain
exactly or is a subdomain of one) oroff-list. This classification measures compliance with
the list, not trustworthiness: an off-list citation is not necessarily untrustworthy. We also
replicate the audience-profile analysis of Section 5.6 on each arm’s citations.
4.7 Ethics and transparency
The reviewers were five students in the social sciences (three master’s students, one PhD
student,andonebachelorstudent,allattheSchoolofSocialSciences,UniversityofIceland),
selectedfortheirsubject-matterbackgroundinEuropeanandEUaffairsandrecruited
with the assistance of a faculty member in that field (see Acknowledgements). They
werepaidcontributors,notanonymoussurveyparticipants,andwerecompensatedper
item(ISK1500,about EUR10.45,perreviewedanswer). Undertherelevantinstitutional
rules the work did not require formal ethics review. All analysed data were anonymised,
reviewersworkedindependentlyandcouldnotseeoneanother’sevaluations,andthey
understoodthat theirevaluations wouldbe usedto assessandimprove theservice. The
service is funded by Iceland’s Ministry for Foreign Affairs but is editorially independent;
thefundingwasadministeredbyoneoftheauthorsandtheministryhadnoroleinthe
evaluation or its reporting.
5 Findings
Across the 551 evaluations, answers passed a mean of 5.1 of the 7 quality criteria. Our
focusisthe128sourceflagsandwhattheyrevealaboutwherecitizens’answerswould
come from.
5.1Source problems are common, and on the open web the complaint is trust
Reviewers flagged128 cited sources. On the open web,where a sourcecould be flagged
as untrustworthy or irrelevant, trust was the dominant complaint: of 110 web flags, 87
were for untrustworthiness and 23 for irrelevance (Figure 1A). The 18 curated-corpus
flags were, by the design of the instrument, all for being outdated. Because the two paths
were held to different criteria (Section 4), the left panel should be read as the reasons
availabletoeachpath,notasevidencethatthepathsfailincategoricallydifferentways.
Thesubstantiveresultiswithinthewebpath: whenexpertscouldquestionawebsource’s
reliability, they frequently did, and far more often than they found it merely off-topic.
5.2 Web search is where the trust risk concentrates
When reviewers examined a web-search answer, 35% of the time they flagged at least one
of its sources (65 of 187 reviewed web answers); for RAG the figure was 6% (16 of 262)

Curated retrieval versus open web search11
RAG article (local) Web source020406080Number of flags
182387Reasons by source origin
Reason
Outdated
Irrelevant
Untrustworthy
RAG (local) Web search0510152025303540Reviewed answers (%)
6%
(16/262)35%
(65/187)Reviewed answers with a flagged sourceSource flags
Figure1: Sourceflags. (A)Flagreasonsbysourceorigin;thereasonsavailabletoeachpath
differ by design, so this shows the reason distribution within each path, not a discovered
contrast. (B)Shareofreviewedanswerswithatleastoneflaggedsource,bymode(web
65/187, RAG 16/262); because the available reasons differ by mode, this cross-mode
comparison is descriptive.
(Figure 1B). Per source the rate is lower but still substantial: across the 187 reviewed web
answersthemodelcited1,088sources(442distinctURLs),ameanof5.8peranswer,so
the110flagsfallonroughly10%ofcitedsourcesand25%ofdistinctURLs. Theserates
are not a like-for-like comparison with RAG, since web and curated sources were held to
different standards. Becauselocal articles were assessed only forcurrency, the study can
estimateweb-source trustproblemsdirectlybutcannot estimatealike-for-liketrustgap
betweenwebandcuratedsources. Thewebfigurenonethelessstandsonitsown: inmore
than a third of the web answers an expert examined, a cited source was one they judged
untrustworthy or irrelevant.
Theflaggedwebsourcesclusteronarecognisablesetofdomains(Figure2),andthe
flags track source type more than mere frequency. The trusted aggregator esbvaktin.is
was the most-cited domain (84 citations) and was flagged on 10% of them, whereas
alternative, opinion-driven and lightly edited outlets were flagged far more often: the
pages of one broadcaster with a declared anti-EU stance were flagged on 27% of their
citations,andthemost-flaggedtabloid-stylesiteonmorethanhalf. Thecontrasttracks
the mainstream/alternative-media distinction drawn in the Background rather than mere
citation frequency. The problem is less that the web has no relevant pages on these
questionsandmorethatthepagesthesystemreachedwereoftenonesthatexpertswould
not endorse.
5.3 What reviewers objected to
The flag reasons (outdated, irrelevant, untrustworthy) are coarse, so we looked at the
writtencommentsbehindthem. Readingthe comments(90ofthe128flagscarryone),
we built a set of recurring objections and had an LLM tag each comment with every
objection that applied (Section 4); the 77 web comments are summarised in Figure 3. Two

Curated retrieval versus open web search12
0 2 4 6 8 10 12
Number of flagsbjorn.issameyki.isfrettin.isthjodolfur.isis.wikipedia.orgfeykir.isen.wikipedia.orgskandall.isnutiminn.isesbvaktin.isvisir.isutvarpsaga.is
3333334778812Flags by domain, split by reason · web search only · top 12 domainsMost-flagged domains and reasons (web sources)
Reason
Irrelevant
Untrustworthy
Figure 2: Most-flagged web domains, coloured by reason (web sources only).
complaints dominate, and both are about the type of source rather than its topic: the
cited page came from a partisan or otherwise non-objective outlet (17 comments), or it
wasanopinionoreditorialpiece(16). Asecondclusterconcernssourcingpractice: the
answer cited a secondary report when a primary source was available (15), or a page
that itself contained no references (7). The rest spread across recognisable web-quality
problems: personalblogs(7),Wikipedia(6),brokenlinks(6),visiblypoorpresentation
(5),paywalledorotherwiseinaccessiblepages(2),andmaterialinalanguagethatmadeit
hard to verify (3). The picture is consistent with the domain analysis: the model reached
pages that were on topic but editorially weak. The labelling was LLM-assisted but not left
unchecked: the authors read every comment against its assigned codes and corrected the
few that were off, and we reproduce all 77 web comments, an English translation, and
their final codes in Appendix C so the coding can be audited directly.
5.4 Trust is decoupled from answer quality
A natural worry is that flagged answers are simply bad answers, which an ordinary
qualityreviewwouldalreadycatch. Therubricscoressayotherwise. Webanswersfar
outperformed RAG on the single criterion of whether they addressed the question (91.5%
versus48.5%; 𝑝<0.001 ;Figure4A).Thematchedquestionstellthesamestory: among
the179questionsreviewedinbothmodes,webanswered82thatRAGdidnot,against
only 8 the other way (McNemar’s test, 𝑝<0.001 ). The reason is instructive: when the
curated corpus lacked material for a question, the model usually said so rather than
inventing an answer. RAG answers fell back on a stock disclaimer, in essence “according

Curated retrieval versus open web search13
0 2 4 6 8 10 12 14 16
Number of commentsDerived from social mediaPaywalled / inaccessibleHard to verify (language)Outdated informationObscure/unknown outletPoor quality or presentationWikipediaBroken or wrong linkBlog or personal siteSource lacks its own referencesIrrelevant / off-topicShould cite the primary sourceOpinion/editorial piecePartisan or biased source
12333566777151617LLM categorisation of 77 reviewer comments (multi-label) · web sourcesWhat reviewers objected to in web sources
Figure 3: What reviewers objected to in flagged web sources, from an LLM categorisation
of their written comments (multi-label; 77 web comments).
to Evrópuvefur’s material, there is not enough information to answer this”, which we
detected with a keyword pattern and confirmed by reading a sample of the matches.
Among RAG answers that failed the “answers the question” criterion, 87% carried such a
disclaimer;thephrasingappearedin70%ofallRAGanswersbutinonlyabout1%ofweb
answers,whichalmostalwaysreturnedsomethingonpoint. ThelowRAGfigurethus
reflects the curated corpus’s coverage, not the model’s writing. Web answers also scored
significantly higher on appropriate scope (92.8% versus 82.3%; 𝑝<0.001 ), publishability
withminoredits(67.3%versus52.4%; 𝑝=0.001 ),factualaccuracy(80.7%versus71.0%;
𝑝=0.010 ),andfreedomfromhallucinations(88.8%versus80.8%; 𝑝=0.012 ),gapsthat,as
the next paragraph shows, are largely carried by the coverage disclaimers. RAG answers
scored higher on the language-quality criterion, the reviewers’ yes/no judgment that
ananswerreadswellinIcelandic(Section4),at85.4%versus78.9%( 𝑝=0.049 );thetwo
modes did not differ on the relevance of their sources.
BecauseadeclinedRAGanswermechanicallyfailsnotonly“answersthequestion”
but several of the other criteria with it, the raw cross-mode bars understate the quality
oftheanswersRAGactuallyproduces. PanelBofFigure4thereforerecomputesevery
criterion over only the evaluations whose answer passed “answers the question” (159
RAGand204webevaluations),sothereadercansee,criterionbycriterion,whatdropping
the unanswered questions does to the comparison. It reverses much of it. On this
answered-only subset the two modes no longer differ on factual accuracy (89% versus
85%),appropriatescope,orfreedomfromhallucinations(all 𝑝>0.05),andRAGmoves
aheadonrelevantsources(78%versus65%; 𝑝=0.006 ),languagequality(92%versus81%;
𝑝=0.003 ),andpublishabilitywithminoredits(87%versus72%; 𝑝<0.001 );thecomposite

Curated retrieval versus open web search14
0 20 40 60 80 100
Evaluations passing (%)Answers questionPublishable w/ minor editsSources relevantFactually accurateNo hallucinationsAppropriate scopeLanguage quality
************Wilson 95% CI · *p<0.05  **p<0.01  ***p<0.001 (chi-square / Fisher)A. All evaluations (n=551)
0 20 40 60 80 100
Evaluations passing (%)Publishable w/ minor editsSources relevantFactually accurateNo hallucinationsAppropriate scopeLanguage quality
*******‘Answers question ’ excluded (it is the filter)B. Only answers that addressed the question (n=363)
RAG (local) Web search
Figure 4: Quality-criterion pass rates by mode, with Wilson 95% intervals. (A) All
evaluations. (B)Onlyevaluationswhoseanswerpassed“answersthequestion”,withthat
criterion dropped; this panel shows what excluding the unanswered questions, RAG’s
coveragefailuremode,doestoeachremainingcriterion. Starsmarksignificantdifferences
within each panel.
overthesesixcriteriafavoursRAGat5.3versus4.9of6(Mann–Whitney 𝑝<0.001 ). In
other words, when the curated corpus does cover a question, the answer RAG returns
is typically of high quality; the mode’s low aggregate scores are a coverage effect, not
agenerationdeficit. Wetreat this subsetanalysisasexploratory, sinceconditioningon
“answers the question” selects a different set of questions in each mode.
Onthe “answersthequestion” criterion,webanswers rarelyfailed(9%), andthefew
that did were idiosyncratic rather than systematic. To see how, we read all nineteen
failingevaluations(covering eighteenanswers)together withthereviewers’notes. Two
were outright technical breakdowns: in one the model returned only its own search-and-
reasoning scratchpad (“investigating EU fisheries negotiations...”) with no finished
answer, and in another, on the legal duties of a parliamentary article, it pasted a raw
search-tool trace that the reviewer found unreadable. A second group answered a subtly
differentquestionthantheoneasked,usuallybecausethequestioncarriedafalsepremise
or an outdated one: one assumed Iceland had adopted the 2011 draft constitution, which
it has not; another asked about safeguarding “all the nation’s resources” while the answer
addressedonlyfisheries;athirdpresentedasprospectivearegulationIcelandhadalready
implementedin2022. Athirdgroupleanedon asingleweaksource,suchasan opinion
blog opposed to the referendum, that the reviewer judged too thin to count as answering,
andtheremainderwerefactualquibblesinotherwiseordinaryanswers. Noneofthese
werethecoveragedisclaimersthat dominatedtheRAGfailures, sothetwomodesfailed
this criterion for entirely different reasons.
Because the “answers the question” criterion, driven by coverage, dominated the
comparison, we also recomputed the composite over the other six criteria across all
evaluations, unanswered ones included. Much of the gap closed. Across all evaluations
the modes were statistically indistinguishable (4.33 versus 4.70 of 6; Mann–Whitney
𝑝=0.15),andapairedtestonthe179questionsreviewedinbothmodesleftaresidual

Curated retrieval versus open web search15
edge for web (4.75 versus 4.19 of 6; Wilcoxon 𝑝=0.003 ), far smaller than the coverage-
driven gap on “answers the question” itself.
Asharperteststaysinsidethewebpath: doanswerswhosesourceswereflaggedscore
worse than those whose sources were not? On the criteria that capture how an answer
reads, they do not. Web answers with at least one flagged source were statistically no
different from unflagged ones on whether they answered the question (90% versus 93%),
read well in Icelandic (73% versus 83%), stayed in scope (90% versus 95%), or avoided
hallucinations (84% versus 92%; all 𝑝>0.05, Fisher’s exact). Two criteria did separate
them: “sourcesrelevant”(33%versus81%)andtheoverall“publishablewithminoredits”
judgment (47% versus 81%; both 𝑝<0.001 ), and the six-item composite was lower for
flagged answers(4.0 versus5.1 of 6; 𝑝<0.001 ). Thefirst is closetodefinitional, since a
flagged source is often an irrelevant one; the second is more telling, because the reviewer
makingtheoverallpublishabilitycallsensedsomethingamissthatthesurfacecriteriadid
not. So fluency and topical fit carry no signal about whether the sources underneath are
sound, while a reviewer’s overall editorial judgment carries some. Ordinary answer-level
review wouldcatch part of thesource-trust problem and misspart of it, whichis exactly
why source trustworthiness has to be assessed on its own.
5.5 Coverage and source quality by question type
The generated question set was not uniform: it spanned six types, weighted toward
the live debate, with discourse positions (83 questions) and policy reasoning (67) the
largest,followedbyhistoricalcontext(49),misconceptions(45),comparativecases(31),
and referendum-specific questions (12). Breaking the two headline patterns down by
typeshowsthatbothholdacrosstheboardratherthanrestingononekindofquestion
(Figure 5).
The coverage gap is visible in every type. RAG left between a third and three-fifths of
questions unanswered depending on type (highest for discourse positions at 59% and
historical context at 55%, lowest for referendum-specific questions at 31%), whereas web
searchanswerednearlyallofthem(unansweredratesof2–10%acrossthefivelargertypes;
33%forthetinyreferendum-specifictype, whichrestsonnineevaluations). Thepattern
matches the coverage story: the curated corpus, frozen in 2013, is thinnest exactly where
the debate has moved on, such as evolving discourse positions. The source-trust problem
islikewisepresentineverytypeonthewebpath,wheretheshareofreviewedanswers
carryingaflaggedsourcerangedfrom25%to58%,againstatmost10%forRAG.Theweb
flag rate was highest for comparative cases (58%), though the smaller types (comparative,
referendum-specific) rest on few reviewed answers and should be read with that in mind.
No question type was simultaneously well-covered by the curated corpus and free of
web-source flags, which reinforces the coverage–trust trade-off rather than localising it to
one topic.
5.6 The political profile of the cited sources
The expert flags are one lens on source quality; an independent one is who, in the wider
public, actually reads a given outlet. We characterised each Icelandic news outlet the

Curated retrieval versus open web search16
0 20 40 60 80 100
Unanswered (%)Referendum-specificMisconceptionsHistorical contextDiscourse positionsPolicy reasoningComparativeBootstrap 95% CI · share of evaluations failing ‘answers question ’ · RAG vs web searchUnanswered rate by category and mode
Mode
RAG (local)
Web search
0 20 40 60 80 100
Share with a flagged source (%)Referendum-specificMisconceptionsHistorical contextDiscourse positionsPolicy reasoningComparativeBootstrap 95% CI · share of reviewed answers with at least one flagged sourceFlag rate by category and mode
Mode
RAG (local)
Web search
Figure5: Patterns byquestion type. (Left)Unanswered rate(share ofevaluations failing
“answersthequestion”),bytypeandmode. (Right)Shareofreviewedanswerswithat
least one flagged source, by type and mode. Whiskers are bootstrap 95% confidence
intervals (10,000 resamples); the smaller types (comparative, referendum-specific) rest on
few reviewed answers, hence the wide intervals. RAG = blue, web search = teal.
web-search mode cited using the Fjölmiðlanefnd (Media Commission) 2024 survey, a
nationallyrepresentativesurveyofmediauseandpoliticalorientation(Fjölmiðlanefnd,
2024). Followingtheaudience-polarisationmethodofFletcheretal.(2020),usedinthe
Reuters Institute Digital News Report (Newman et al., 2022) and, for Iceland, by Ólafsson
andJóhannsdóttir(2024),weplacedeachoutletontwoaxesbythemake-upofitsaudience:
a left–right axis and, more salient for an EU vote, a nationalism–internationalism axis.
Foreachaxisanoutlet’saudienceskewistheshareofitsreadersatthehighendminus
the share at the low end, read relative to the national population. We then took every
referencethesystemcited(parsedfromeachanswer’sreferencelist)acrossthe287eligible
web-search answers, kept the 153 whose domain mapped to a surveyed outlet (nine
distinct cited outlets, of eleven surveyed), and computed the citation-weighted audience
profile of those references. Curated (RAG) answers cite only the in-house corpus, so this
analysis necessarily concerns the open-web path.
Twopatternsemerge(Figure6). Ontheleft–rightaxis,thesourcesthesystemcited
lean right of the population (citation-weighted skew +0.17against a population near
zero;57%ofmatchedreferenceswenttooutletswithright-leaningaudiences),andthis
tilt held across every question type, strongest for policy-reasoning questions. On the
nationalism–internationalism axis the average cited source matched the (internationalist-
leaning) population rather than shifting it, but the only two outlets whose audiences lean
nationalist, Útvarp Saga and Fréttin.is, were both among the alternative outlets experts
flagged,andÚtvarpSagadrewmoreflagsthananyotheroutlet. Thetwo-dimensional
mapmakesthestructureplain: theheavilyflaggedoutletssittogetherintheright-leaning,
nationalist, low-independence corner (independence here is the survey’s measure, the
share of respondents who rate an outlet independent of political interests), while the
outlets whose audiences lean internationalist (Heimildin, Samstöðin, RÚV) drew few or
no flags. This is independent corroboration, from public survey data rather than our own
reviewers,ofboththeexpertflagsandthemainstream/alternativedistinctionofSection4:
the per-outlet audience profile, perceived independence, citation counts, and flag counts
arereportedinAppendixD.TheclearestexceptionisVísir,amainstreamportalthepublic

Curated retrieval versus open web search17
-0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8
Left  <-->  right (audience)-0.4-0.20.00.20.40.60.8Nationalist  <-->  internationalist (audience)RÚV (0)
mbl.isVísirDVHeimildin
Mannlíf (0)
Útvarp SagaFréttin.isSamstöðin
Viðskiptabl.
Nútíminn(A) Audience political map
-0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4
Deviation from population (+ = right / internationalist)ComparativeHistorical contextMisconceptionsDiscourse positionsReferendum-specificPolicy reasoning(B) Citation skew by question topic
Population = 0
left-right
nat./intl.
102030405060
Rated independent (%)
Political profile of the Icelandic news outlets the AI cited
Figure6: PoliticalprofileoftheIcelandicnewsoutletstheweb-searchmodecited,from
theFjölmiðlanefnd2024survey. (A)Eachoutletplacedbyitsaudience’sleft–rightskew(x)
andnationalist–internationalist skew(y); bubblesize =numberof referencesthe system
gave it, colour = share of the public rating it politically independent, dashed lines =
nationalpopulation. Opencircles(RÚV,Mannlíf)werenotcited. (B)Citation-weighted
audience skew of the cited references relative to the population, by question topic, on
both axes (bootstrap 95% CIs). Positive = right-leaning / internationalist audience.
rates as independent (61%) yet which drew eight flags; reading them shows a mix of
opinion pieces judged untrustworthy and off-topic articles judged irrelevant, rather than
a problem with the outlet as such. This is less surprising than it first appears: Vísir is the
main Icelandic venue for opinion pieces contributed by the general public, which the site
publishes in a section clearly separated from its news reporting and editorial content, and
every untrustworthiness flag on Vísir fell on such a contributed piece rather than on the
newsroom’soutput. Thesesurvey-basedmeasurescaptureanoutlet’saudience,notthe
slant of any individual cited article, and cover only the Icelandic-news subset of citations;
we treat them as exploratory and, given small audiences for the niche outlets, report
bootstrap95%intervals. Oneabsencedeservesitsownsentence: RÚV,thestate-funded
publicbroadcasterandthecountry’smostwidelyusednewssource,wasnevercitedby
the deployed system across the 287 web-search answers.
5.7 Does the source list steer citation behaviour?
The flag analysis showed that the web-search prompt’s trusted-domain list did not keep
weak sources out of the answers. The prompt ablation (Section 4.6) measures directly
how much that list steers the model. The answer is: only weakly. With the list in the
prompt, 21% of the 1,183 citations landed on a listed domain; with the list removed, 12%
of 1,330 did (Figure 7A). The instruction thus doubles compliance, but roughly four in
fivecitationsfalloutsidethelisteitherway. Asystemprompt,onthisevidence,nudges
retrieval; it does not govern it.
The political profile of the cited outlets barely moves. Here, as in Section 5.6, “the

Curated retrieval versus open web search18
population” is the national adult populationas represented bythe Fjölmiðlanefnd 2024
survey sample: an outlet’s audience skew is measured against all respondents, so a
citation-weighted skew equal to the population’s means the cited outlets’ combined
readership is politically indistinguishable from the country as a whole. By that baseline
thecitation-weightedleft–rightskewofthecitedIcelandicoutletsmatchesthepopulation
inbotharms(−0.01withthelist,−0.01without;population 0.00),andthenationalism–
internationalismskewsitsslightlytotheinternationalistsideofthepopulationinboth
(+0.06relativeineach). Removingthelistmainlyaddscitationstothemainstreamoutlets
that already dominate (Vísir from 107 to 121, mbl.is from 51 to 64; Figure 7B), and the
alternativeoutletstheexpertsflaggedmoststaymarginalinbotharms(ÚtvarpSaga10
citationsineach,Fréttin.isnone). RÚVisagainconspicuous: thestatebroadcasterdrew
50 citations in each arm, about 4% of the total, identical with and without the list, modest
for the country’s most widely used news source and its main publicly funded newsroom.
The flat profile also contrasts sharply with the deployed system itself. In production,
the citations leaned right of the population ( +0.17), drew heavily on Útvarp Saga, and
never included RÚV (Section 5.6). The ablation used the same model, the same questions,
thesameweb-searchplugin,andanear-identicalprompt,yetitscitationsshownoneof
this: the profile is neutral, Útvarp Saga is rare, and RÚV appears regularly. Only two
things separate the runs. The ablation collected citations through a structured output
format (with the search tool’s redirect links resolved to their target pages), whereas
production parsesthe reference lists themodel writes intoits free-text answers; and the
runs took place a few days apart. That such small differences move the citation mix
this much is an important finding in its own right: which sources an AI assistant cites
is not a stable property of the model and prompt alone but of the entire configuration
that produces the answer, so a measurement made on one configuration, including ours,
doesnotautomaticallycarryovertoadifferentlyinvoked“same”system. Theprompt,
on the compliance evidence above, is among the weakest levers in that configuration.
Theablation’scitationswerenotexpert-reviewed,sothiscomparisonconcernscitation
patterns, not judged trustworthiness (Section 7).
5.8 Reliability of the expert judgments
We treat the agreement analysis as exploratory, because the overlap supporting it is thin.
Of the 449 reviewed answers, only 82 (18%) were scored by more than one reviewer, and
pairwiseoverlapbetweenreviewersrangesfrom11to22sharedanswers(AppendixA).
This is a deliberate consequence of the review procedure, which prioritised un-reviewed
items so that corpus coverage built up before reviewers doubled up (Section 4), not an
oversight;itdoes,however,meanthecoefficientsbelowshouldbereadasindicativerather
than confirmatory. On those doubly-reviewed answers the whole-answer judgments
heldupunderchancecorrection. Agreementwasstrongestonappropriatescope(Gwet
AC1=0.83)andweakestontheoverall“publishable”decision(AC1 =0.29),withmost
criteria in between. Kappa-based measures collapsed for the high-prevalence criteria.
“Nohallucinations”istheclearestcase: reviewersgavethesameverdicton74%ofanswers,
yet Fleiss’ kappa was negative. This is the kappa paradox. When almost every answer
falls in one category, here nearly all were judged free of hallucinations, two reviewers

Curated retrieval versus open web search19
With source list Without source list020406080100Share of citations (%)
21% on list
12% on list(A) Compliance with the source list
On list (exact host)
On list (subdomain)
Off list
0 20 40 60 80 100 120
Citations (287 questions per arm)Heimildin  (-0.41)Samstöðin  (-0.36)RÚV  (-0.11)Vísir  (-0.02)DV  (+0.10)mbl.is  (+0.13)Mannlíf  (+0.22)Nútíminn  (+0.50)Fréttin.is  (+0.55)Viðskiptabl.  (+0.64)Útvarp Saga  (+0.64)(B) Citations to Icelandic news outlets (ordered by audience left-right skew)
With source list
Without source listPrompt ablation: what does the trusted-source list change?
Figure 7: Prompt ablation: the 287 study questions answered with and without the
trusted-domainlistintheweb-searchprompt(Gemini3Pro,productionAPIrouteand
web-searchplugin,structuredcitationoutput). (A)Shareofcitationsonthelist(exacthost
or subdomain) versus off it, per arm. (B) Citations to the surveyed Icelandic news outlets
per arm, ordered by the left–right skew of each outlet’s audience (in parentheses); filled =
with list, ring = without. Equal counts show as a dot inside a ring (RÚV: 50 in each arm).
will land on the same verdict most of the time simply by both picking the common label.
Kappa treats that high baseline as agreement expected by chance and subtracts it, which
leaves little room to score agreement above chance, so a few disagreements can push the
coefficient to zero or below even when raw agreement is high. Gwet’s AC1 estimates the
chancebaselineinawaythatdoesnotballoonunderskewedprevalence,soitdoesnot
collapse, which is why we read agreement through it (Gwet, 2008). We report the full
comparison in Appendix A.
The source flags, which carry the headline result, have even thinner support for an
agreement estimate. Only 40 of the 128 flags fall on a doubly-reviewed answer, and
on just six answers did two reviewers independently flag a source, too few to estimate
flagreliability. Wethereforetreatflagsasindividualexpertjudgments,notadjudicated
rulings,andreadtheheadlineaccordingly: morethanathirdofreviewedwebanswers
cited a source thata reviewerjudged untrustworthy or irrelevant.
6 Discussion
6.1 Source trustworthiness as a neglected dimension of information quality
Ourcentralobservationissimple. ThesourcesbehindapublicAIservicecanbeassessed,
they vary in trustworthiness, and that variation is rarely surfaced. Services routinely
log queries and answers, but recording whether the cited material would survive expert
review is not, as far as we have seen, standard practice in deployed public services.
Classicinformation-qualityworkisusefulhere. WangandStrong(1996)treatquality as
somethingdataconsumersjudgealongmanydimensionsatonceandgrouptheminto

Curated retrieval versus open web search20
four families: intrinsic quality (accuracy, objectivity, believability, reputation), contextual
quality (relevance, timeliness, completeness, appropriate amount), representational
quality, and accessibility. The trustworthiness of a cited source belongs to the intrinsic
family, close to believability and reputation, and our data show it behaving largely as
a dimension of its own. Within the web path, answers with flagged sources were no
worseonfluency,topicality,orscope;onlyareviewer’soverallpublishabilityjudgment
picked up part of the problem. A quality process that scores the surface of an answer will
thereforemissmuchofthesource-trustproblem,evenwhereacarefuloveralleditorial
call catches some of it.
6.2 Coverage and trust pull in opposite directions
The two paths traded off along a single axis. Open web search reached current, on-point
material, which is why it answered the question far more often, but much of that material
came from alternative, opinion-driven outlets rather than edited mainstream journalism,
and in more than a third of the answers reviewers examined, a cited web source was
flagged. That happened despite an explicit instruction to prefer reputable, primary, and
authoritativesourcesandtoavoidblogs,opinioncolumns,andpartisansites(AppendixB):
the flag rate shows that prompt-level steering, on its own, did not keep weak sources
outoftheanswers,andthepromptablationquantifieshowweakthatleveris,withthe
trusted-domain list raising the share of citations to listed domains only from 12% to
21% (Section 5.7). Independent survey data reinforce the point: the outlets the system
leanedonmostheavily,andthatexpertsflaggedmost,arealsothosethepublicratesas
least politically independent and whose audiences sit furthest to the right and toward
the nationalist pole (Section 5.6). The mirror image of that reliance is as striking: RÚV,
the public broadcaster, the country’s most widely used news source, and the outlet
the surveyed public rates most politically independent, was never cited in any of the
287 web-search answers. A citation mix can thus be skewed in two directions at once,
toward weak sources and away from the strongest ones, and only the first of these shows
up in a flag count. The curated corpus had the mirror-image profile: its sources were
trusted by construction, but it could not cover every question, and when it fell short
the model declined to provide an answer rather than reaching for one. That decline is
worth dwelling on. Faced with a question the corpus did not address, the model did
not improvise; it said the material was not there. The rubric scores this as a failure to
answer,yetitisclosetowhatonewantsfromapublicservice: anhonest“Icannotanswer
thatfrommysources”ratherthanaconfidentguess. Readthisway,RAG’slowquality
score measures the coverage of the curated base more than the competence of the model,
and the response is to widen and maintain that base rather than to loosen the model’s
grounding. Its one recurring source fault, when it did answer, was staleness. The two
pathsarenotasseparateasthedesignimplies: inseveralwebanswersreviewersnoted
that a higher-quality Icelandic source, an Evrópuvefur article, existed but was not the one
the model reached for. (Evrópuvefur’s account of the Maastricht criteria, for instance,
is authoritative enough to be cited directly in a recent University of Iceland economics
report (Kristjánsson et al., 2026).) We had deliberately kept the service’s own corpus
off the source guidance given to the web-search mode (Section 4), precisely so that the

Curated retrieval versus open web search21
evaluation would measure the trustworthiness of the sources the system cited outside
thatdomain. Inproduction, however, addingit, sothatopen-webanswers canfallback
on vetted in-house material, is a concrete, low-cost way to push the trade-off toward trust
without sacrificing coverage. Neither path is safe by default. A curated base buys trust at
the price of coverage and needs constant maintenance; an open-web path buys coverage
at the price of trust and needs some way to tell a reliable page from an unreliable one.
6.3 Responses and their trade-offs
What follows for practice is less obvious than it may seem, and we resist a single
prescription. The most direct response, restricting a public service to an approved list
ofsources,tradesoneriskforanother. Itimprovesreliabilitybutconcentrateseditorial
powerinwhoevermaintainsthelist,anuncomfortablepositionforapublicbodyandone
in tension with open access to information (Ananny and Crawford, 2018). A lighter-touch
alternative is transparency about provenance: making the cited sources, and the basis for
citing them, visible to the reader rather than buried. Content-provenance standards such
asContentCredentials(C2PA)offeronetechnicalroutetocarryandshowwhere material
camefrom(CoalitionforContentProvenanceandAuthenticity,2025). Independentor
expertlabellingofsourcetrustworthiness,keptseparatefromtheinstitutionthatanswers,
is one way to surface that signal without handing any single actor a veto. Two prior
questions complicate all of this. The first is definitional: what makes a source trustworthy
is itself contested, and any labelling scheme must answer it before it can be applied. The
second is jurisdictional: it is not obvious that an AI provider should be adjudicating
source trust at all, since aggressive scrutiny shades into editorial control. A lighter option
keepsthechoicewiththereader,lettingauserconstrain,beforeasking,whichdomains
orkindsofsourceaquerymaydrawon,thoughfewmainstreamsystemsexposesuch
a controltoday. Human review, as practised here, catchesboth failure modes butdoes
not scale to live answering. Keeping curated corpora current addresses staleness but not
webtrust. Theseareoptionswithtrade-offs,notasolution. Ourclaimisnarrower: the
trustworthinessofcitedsourcesshouldbemeasuredanddisclosed,andthechoiceamong
responses is a matter for public deliberation.
6.4 Implications
Fortheory,theresultsarguefortreatingsourcetrustworthinessasafirst-classdimensionof
information quality in AI-mediated public information, alongside accuracy and currency.
For practice, they point to concrete levers: log and audit the sources a service cites, report
source-trustmetricsaspartofroutineevaluation,anddiscloseprovenancetousers. For
procurement and oversight, a public body adopting an LLM service should ask not only
how accurate it is but where its answers come from and how that is checked.
The stakes are highest around elections, and the supply side of the problem is getting
cheaper. LLMs can now produce election misinformation that human readers cannot
reliably distinguish from authentic content, at high volume and negligible cost (Williams
et al., 2025), and analysts have warned since the models’ early days that they would
lowerthepriceoflarge-scaleinfluenceoperations(Goldsteinetal.,2023). Astroturfing,

Curated retrieval versus open web search22
coordinatedcampaignsposingasordinarycitizens,longpredatesthesemodels(Keller
et al., 2020), but generative AI removes its main bottleneck: the writing. The concern
foraserviceliketheonestudiedhereisnotonlythatvotersreadsuchmaterialdirectly,
but that a web-grounded assistant retrieves and cites it. That pathway is no longer
hypothetical. A 2025 audit found that a Kremlin-linked network of sites, publishing
millionsoflow-readershiparticlesayearseeminglyaimedatmachinesratherthanpeople,
haditsclaimsrepeatedbyleadingchatbotsinathirdoftestedresponses(NewsGuard,
2025). Follow-up research attributes such citations less to wholesale poisoning than to
data voids, niche questions on which authoritative coverage is scarce (Alyukov et al.,
2025),whichispreciselytheconditionofasmalllanguageduringacontestedreferendum:
on many of our questions, the trustworthy Icelandic material for a retrieval system to find
is thin or absent. A referendum can plausibly be targeted by flooding the open web with
materialthatanAIassistantwillthenretrieve,cite,andlendaninstitution’scredibility
to. Measuring the trustworthiness of cited sources, as done here, is therefore not only
a quality-assurance exercise; it is a basic defence for the integrity of the information
environment around a vote.
6.5 Future work
Four directions follow directly from this study. First, the source-flagging instrument
we used is labour-intensive and does not scale to live answering; a natural next step is
anautomatedsource-trustvetterthatsurfacesweaksources(toaneditorortothereader)
at answer time rather than in a retrospective audit. This need not involve any model
training: anLLM,givenclearandexplicitcriteriaforwhatmakesasourcetrustworthy,
couldveteachcitedpageonthefly,andourflagcorpus,withitsper-sourcelabelsand
free-textreasons,offersbothawaytospecifythosecriteriaandabenchmark toevaluate
the vetter against. The main caveat is cost. Running an independent vetter alongside
the answering model adds a second pass to every query, but in a public setting, where
the credibility of the answers is the whole point, that added cost may well be worth
paying. Second, the review instrument should add an answer-levelbalancecriterion.
Per-source flags cannot see one-sidedness: every cited source can pass individually while
the answer as a whole presents one side of a contested question. Our flag comments
suggest the risk is real, since opinion and editorial pieces were among the reviewers’
most common objections (Section 5.3), and an answer resting on a single such source
caneditorialisewithouttrippinganyper-sourcecheck;WangandStrong(1996)would
place balance under objectivity, in the same intrinsic-quality family as believability. For a
servicewhosemandateiseven-handedness,balancedeservesacriterionofitsown. Third,
ourremedies(provenancedisclosure,sourcelabelling,user-setdomainconstraints)are
so far argued rather than tested; auser-facing studycould measure how such provenance
signalsandcontrolsactuallyaffectusers’trust,verificationbehaviour,andreliance,which
is what ultimately determines whether transparency helps. Fourth, the design should be
replicatedin other low-resource languages and other high-stakes civic moments (elections,
public-healthemergencies),bothtotesthowfarthecoverage–trusttrade-offgeneralises
and to support the larger, balanced, planned-overlap reviewer design that a confirmatory
agreement estimate would require.

Curated retrieval versus open web search23
7 Limitations
Several limitations bound these results. The evaluation was conducted before public
release, on the system as configured during testing, and the questions were generated to
span the debate rather than drawn from real users, so they may over- or under-represent
what citizens would actually ask. Evaluations were not randomly assigned to modes:
reviewers worked through a shared queue rather than a balanced design, so the two
modes were reviewed in unequal numbers (262 RAG versus 187 web answers) and
the mode comparison is observational rather than controlled. Because the flagging
instrument applied different reasons to each retrieval path, cross-mode flag comparisons
aredescriptive,andtheweb-onlyflagrate(35%)isbestreadonitsownratherthanasalike-
for-like contrast with RAG. The study covers a single service, language, and topic during
one referendum period, which limits generalisation. The 128 flags are a standing backlog
of open flags rather than a closed, adjudicated review, so they record what reviewers
raised, not a final ruling; most were single-reviewer judgments, with toolittle overlap to
estimate their reliability. The reviewer pool is small (five experts), and inter-rater overlap
ismodest. Ourreviewersweresubject-matterexpertsinEUaffairsandmayholdpriors
onthequestion,whichcouldbeperceivedaspro-EU;wemitigatedthisbyaskingthemto
judge cited sources against editorial criteria (trustworthiness, relevance, and currency)
rather than the political position of an answer (the rubric contains no item rewarding
agreement with a conclusion), and by having them work independently and unable to
seeoneanother’sevaluations,butwecannotruleoutthatjudgmentsofwhichsources
count as trustworthy carry some such priors. A related risk attaches to esbvaktin.is :
because it is itself an LLM-assisted aggregator and supplied both our question seeds
and the source guidance for web-search mode, it could in principle propagate its own
selection biases into our pipeline (an LLM feedback loop). We limited this by using
it only as a source map, never as content for answers, and by having reviewers judge
the actual cited pages; a residual risk remains that what we call “trusted” inherits its
editorialjudgments. Becausejudgingasourcemeansseeingit,reviewerscouldtellalocal
articlefromanexternalwebpage,sotheretrievalmodewasnotblindedandmayhave
coloured some judgments. Finally, scanning seven criteria invites multiple comparisons,
so single-criterion 𝑝-values should be read as exploratory. The survey-based political
profiling of cited outlets (Section 5.6) characterises each outlet’saudience, not the slant
of the specific article cited, covers only the Icelandic-news subset of cited sources, and
applies a 2024 survey to a system evaluated in 2026. The prompt ablation (Section 5.7)
used the same model and web-search plugin as the deployed system but elicited citations
through a structured-output schema rather than the free-text reference lists of production
answers, and ran a few days later; its two arms are internally comparable, but it measures
list compliancerather than trustworthiness, andits contrast withthe deployed system’s
citationmixconfoundselicitationformatwithtiming,sowereadthatcontrastasevidence
of configuration sensitivity rather than as a like-for-like comparison. AI capabilities also
change quickly: these findings describe the specific models and the period we studied
and should not be read asfixed properties ofthetechnology. We expect the measurement
approach, more than any single measurement, to carry over.

Curated retrieval versus open web search24
8 Conclusion
PublicinstitutionsareadoptingAItoanswercitizensdirectly,andthesourcesbehindthose
answers are largely invisible, to users and often to the institutions themselves. Evaluating
an independent, government-funded service before its public launch, one answering
EU-related questions ahead of a national referendum, expert reviewers flagged a cited
source in more than a third of the web-search answers they examined, almost always
as untrustworthy or irrelevant. The trust problem was concentrated in the open-web
path, and the trustworthiness of a source could not be read off the quality of the answer:
fluent,on-topicanswersrestedonsourcesexpertswouldnotendorse,whilethecountry’s
most widely used and most trusted news source went entirely uncited. We do not claim a
single remedy. We do claim that source trustworthiness is measurable, that it matters
most where public stakes are highest, and that public AI services should measure and
disclose it rather than assume it.
CRediT authorship contribution statement
Hafsteinn Einarsson:Conceptualization, Methodology, Software, Formal analysis, Data
curation, Writing – original draft, Writing – review & editing, Supervision.Hafsteinn
Birgir Einarsson:Validation, Writing – review & editing.Jón Gunnar Þorsteinsson:
Supervision of reviewers, Validation, Writing – review & editing.Jón Gunnar Ólafsson:
Validation, Data Curation, Writing – review & editing.
Declaration of competing interest
Theauthorsdeclarethattheyhavenoknowncompetingfinancialinterestsorpersonal
relationships that could have appeared to influence the work reported in this paper.
Funding
This work was funded by the Ministry for Foreign Affairs of Iceland. The funder had
no role in the study’s design, conduct, analysis, interpretation, or reporting, or in the
decision to submit the work for publication.
Data availability
Theanonymisedevaluationdata(reviewerratingsandsourceflags)andtheanalysiscode
used to produce the figures and statistics are available from the corresponding author on
reasonable request.

Curated retrieval versus open web search25
Table1: Chance-correctedagreementpercriteriononthe82doubly-reviewedanswers.
AC1 = Gwet’s AC1;𝜅= Fleiss’ kappa;𝛼= Krippendorff’s alpha.
Criterion % agree AC1 Fleiss’𝜅Kripp.𝛼
Answers question 65.9 0.39 0.23 0.25
Factually accurate 75.6 0.61 0.31 0.36
Sources relevant 75.6 0.53 0.49 0.52
No hallucinations 73.5 0.66−0.21−0.14
Appropriate scope 87.8 0.83 0.50 0.55
Language quality 77.2 0.69 0.14 0.13
Publishable, minor edits 63.0 0.29 0.22 0.22
Acknowledgements
Wethanktheexpertreviewerswhoevaluatedtheservice’sanswers;theircontribution
was supported through funding from the Ministry for Foreign Affairs of Iceland. We
would also like to thank Professor Maximilian Conrad, School of Social Sciences at the
University of Iceland, for assisting in locating the expert reviewers, and Brynjólfur Gauti
Guðrúnar Jónsson, doctoral student in statistics at the University of Iceland, who created
esbvaktin.is ,thefact-checkingprojectthatsuppliedoursourcematerialandthetrusted-
domain classification used in web-search mode. We thank Vilborg Ása Guðjónsdóttir
for helpful discussions during the writing of this manuscript, and the Icelandic Media
Commission (Fjölmiðlanefnd) for access to their 2024 survey findings.
Declaration of generative AI use
The authors used generative AI tools to assist with editing, under author supervision.
The authors reviewed and edited all content and take full responsibility for it.
A Inter-rater reliability
Of the 449 reviewed answers, 82 (18%) were scored by more than one reviewer. Pairwise
overlap between the five reviewers is modest, ranging from 11 to 22 commonly-scored
answersperpair,so Table1shouldbereadasexploratory. Thekappaparadoxisvisible
inthehigh-prevalencecriteria: “nohallucinations”has74%rawagreementbutanegative
Fleiss’𝜅,becausealmostalljudgmentsfallinonecategory. Gwet’sAC1isstableunder
that prevalence and is our headline measure.
B Answer-generation prompts
Bothmodessharethesametaskbutdifferinhowtheyaregrounded. TheRAGsystem
promptisreproducedfirst,translatedfromtheIcelandicoriginalasdeployed,thenthe
web-searchprompt(Englishintheoriginal),reproducedwithitscompletetrusted-domain
list. The prompt asks the model to prefer this list, which esbvaktin.is classifies as “high

Curated retrieval versus open web search26
confidence”, but does not enforce it as a hard constraint; the examples named in the main
text ( icelandmonitor.mbl.is and others) are illustrative entries from it, not a preference
ordering. Wedeliberatelykepttheservice’sowncorpus( evropuvefur.is )offthelist,so
that the evaluation would measure the trustworthiness of the sourcesthe system cited
outside that domain; its sister sitevisindavefur.isdoes appear on the list.
RAG mode (system prompt, translated from Icelandic)
You are an expert assistant for Evrópuvefurinn (evropuvefur.is), run by the
University of Iceland. You are an expert on European Union affairs, the European
Economic Area (EEA), and Iceland's relations with Europe.
## Scope
Answer only questions about Europe, the EU, the EEA, European integration, and
Iceland's relations with Europe. Politely decline questions that are clearly out
of scope with a single sentence pointing the user in the right direction.
## Basis for answers
Base your answers on the accompanying articles from Evrópuvefur's knowledge base.
Do not make up facts. If the context does not contain enough information to
answer, say so.
## Citations
Each article you are given is numbered. Use only markdown links of the form
[[N]](source_url). Do not invent numbers that are not among the articles given.
## Language
Answer in the same language as the user's question.
## Style and length
Write clearly, accessibly, and in an academic tone. Typical length: 150-400 words.
Web-search mode (system prompt, full source list)
You are an expert assistant for Evrópuvefurinn (evropuvefur.is), run by the
University of Iceland. You specialize in EU affairs, the EEA, and Iceland's
relations with Europe.
## Web Search Mode
You have access to web search. Use it to find current, accurate information to
answer the user's question.
## Source selection
Prioritize sources from the list below. These are the domains that esbvaktin.is
-- Iceland's EU referendum fact-checking project (University of Iceland) --
classifies as "high confidence" primary or authoritative secondary sources.
Prefer them over blogs, opinion columns, social media, aggregators, or
unfamiliar outlets. Other reputable sources may be used when they add material
the trusted list does not cover, but say so in context and avoid partisan
websites (political parties, advocacy groups) as evidence for factual claims --
cite them only for documenting a party or group's own position.
### EU institutions and legal texts
eur-lex.europa.eu, ec.europa.eu, commission.europa.eu, europarl.europa.eu,

Curated retrieval versus open web search27
consilium.europa.eu, europa.eu, eeas.europa.eu, enlargement.ec.europa.eu,
neighbourhood-enlargement.ec.europa.eu, agriculture.ec.europa.eu,
climate.ec.europa.eu, oceans-and-fisheries.ec.europa.eu,
taxation-customs.ec.europa.eu, cohesiondata.ec.europa.eu, stecf.ec.europa.eu,
eea.europa.eu, ecb.europa.eu, sdw.ecb.europa.eu,
ireland.representation.ec.europa.eu.
### EEA / EFTA
efta.int, eftasurv.int, eftacourt.int.
### Icelandic government, parliament, and institutions
althingi.is, stjornarradid.is, government.is, island.is, sedlabanki.is, cb.is,
orkustofnun.is, mast.is, skatturinn.is, rna.is, ust.is, fjolmidlanefnd.is,
ferdamalastofa.is, stjornlagarad.is.
### Icelandic statistics and academic reference
hagstofa.is, px.hagstofa.is, statice.is, visindavefur.is, uni.hi.is.
### International organizations
oecd.org, data-explorer.oecd.org, imf.org, data.worldbank.org, fao.org, bis.org,
nato.int, coe.int, hdr.undp.org, eeagrants.org, ccpi.org.
### Foreign governments, parliaments, and agencies
gov.uk, legislation.gov.uk, commonslibrary.parliament.uk,
researchbriefings.files.parliament.uk, obr.uk, gov.ie, regeringen.dk, fm.dk,
thedanishparliament.dk, lf.dk, riksdagen.se, ei.se, su.se, stat.fi, mmm.fi,
ssb.no, fiskeridir.no, europa.eda.admin.ch, hnb.hr.
### Icelandic social partners, industry, and polling
sa.is, asi.is, si.is, bondi.is, responsiblefisheries.is, islandsbanki.is,
gallup.is, northstack.is, eurometal.net.
### Scholarly, legal, and reference
doi.org, jstor.org, link.springer.com, avalon.law.yale.edu,
constituteproject.org, en.wikipedia.org (for orientation only -- always prefer
primary sources for citations).
### Reputable news outlets
reuters.com, politico.eu, rte.ie, theskipper.ie, thelocal.com,
nordiclabourjournal.org, grapevine.is, opendemocracy.net, icelandmonitor.mbl.is,
europeanmovement.ie, arcticcircle.org.
## Citations
Use numbered-bracket citations with inline links: [[N]](URL). Assign numbers in
order of first appearance and reuse them. After the answer, add a References
section listing each cited source. Do not invent sources or URLs; only cite
sources you actually retrieved via web search.
C Flag-comment coding audit
Table 2 lists all 77 reviewer comments on flagged web sources, an English translation, and
the reason codes assigned by the LLM and reviewed by the authors. It lets the reader
check the coding against the original evidence.

Curated retrieval versus open web search28
Table2: All77reviewercommentsonflaggedwebsources,withanEnglish
translation and the reason codes assigned by the LLM and reviewed by
the authors.
Comment (Icelandic) Translation (English) Reason codes
Heimildáöðrutungumálieníslensku
og ensku.Source in a language other than Icelandic
and English.foreign language
Vafasömheimildsemerekkifránógu
vönduðum miðill, rithöfundur hefur
viðurkennir að hann notar gervigreind
við skrif.Questionablesourcethatisnotfromasuf-
ficiently reputable outlet; the author has
admitted to using artificial intelligence in
his writing.poor quality
Þettaerbloggsíðasemerekkifræðilega
traust.This is a blog website that is not academi-
cally sound.blog
Þessigreinereinnigekkifræðilegahlut-
laus.This article is also not theoretically neutral. partisan source
Þetta er hlutdræg skoðanagrein sem er
ekki fræðilega traust.This is a biased opinion piece that is not
theoretically sound.opinion piece, par-
tisan source
Ekkitraustvekjandimiðillaðmínumati Not a trustworthy medium in my opinion other
Þó svo að Björn sé virtur stjórnmála-
maður að þá er hann ekki að notast við
neinarheimildirískrifumsínumogþví
væri hægt að nota betri heimild hér.Even though Björn is a respected politician,
he does not use any sources in his writings,
and therefore a better source could be used
here.no references
Vil frumheimild! Want the primary source! cite primary
Hér væri hægt að nota betri heimild,
greininfrekarstuttogvísarekkiíheim-
ildir. Svarið við þessari spurningu
snýstekkiumskoðanirheldurstaðreyn-
dir og því fagmannlegra að hafa peer-
reviewed greinar sem heimildirA better source could be used here; the arti-
cle is rather short and does not cite sources.
The answer to this question is not about
opinionsbutfacts,andthereforeitismore
professional to use peer-reviewed articles
as sources.no references, cite
primary
Vil frumheimild Want original source cite primary
Vil frumheimild Want the primary source cite primary
Þetta er skoðanagrein sem er líklega
ekki fræðilega traust.Thisisanopinionpiecethatisprobablynot
academically sound.opinion piece
Þettaerskoðanagreinsemerekkifræði-
lega traust.This is an opinion piece that is not academi-
cally robust.opinion piece
Þessi grein er ekki hlutlaus (birt af
Evrópuhreyfingunni).This article is not neutral (published by the
European Movement).partisan source
ályktun svarsins byggir á þessu svari
sem er hlutdrægt.Theconclusionoftheresponseisbasedon
this answer, which is biased.partisan source
Þetta er líka skoðanagrein sem tekin
er af hlutdrægum miðli. Betra væri
að vísa beint í þær sérfræðingaskýrslur
sem höfundur greinarinnar nefnir.This is also an opinion piece taken from a
biased source. It would be better to refer
directly to the expert reports mentioned by
the author of the article.opinionpiece,parti-
sansource,citepri-
mary
Þetta er skoðanagrein sem er ekki hlut-
laus og ekki fræðilega traust.This is an opinion piece that is not objective
and lacks academic rigor.opinion piece, par-
tisan source
Bæðiekkitrausturmiðillogsvohefur
þjóðaratkvæðagreiðslaveriðstaðfest29.
ágúst.Both not a reliable source, and also a ref-
erendum has been confirmed for August
29th.outdated
Óaðgengileg heimild, það þarf áskrift
til að lesa greinina.Inaccessible source, a subscription is re-
quired to read the article.paywalled

Curated retrieval versus open web search29
Table 2 – continued
Comment (Icelandic) Translation (English) Reason codes
Wikipedia heimild á öðru tungumáli.
Mjög erfitt að sannreyna þar sem
síðurnar eru ekki alltaf með sömu up-
plýsingar á mismunandi tunugmálum.Wikipediasourceinanotherlanguage. Very
difficulttoverifysincethepagesdonotal-
wayshavethesameinformationindifferent
languages.Wikipedia, foreign
language
Þessi heimild er erfitt að rekja, þetta er
barapdfsemhefurengarupplýsingar
umútgefanda,blaðnénetsíðuoghefur
enga heimildaskrá.This source is difficult to trace; it is just
a PDF that has no information about the
publisher, journal, or website, and has no
bibliography.unknownoutlet,no
references
Saga er ekki hlutlausasta heimildinn
(gagnvart ESB) til þess að vísa í svo
það er kannski ekki nógu áreiðanleg
heimild.Saga is not the most neutral source (regard-
ing the EU) to refer to, so it is perhaps not a
reliable enough source.partisan source
Wikipedia er kannski ekki nógu
áreiðanleg heimild þar sem hægt er að
breyta gögnunum þar inni án þess að
taka það fram.Wikipedia is perhaps not a reliable enough
source, as the data in there can be edited
without it being explicitly stated.Wikipedia
Þetta er skoðanagrein sem er ekki hlut-
laus og ekki fræðilega traust.Thisisanopinionpiecethatisnotneutral
and not academically sound.opinion piece, par-
tisan source
Virkar ekki!!! Does not work!!! broken link
óþarfiogóáreyðanlegíþessusamhengi. unnecessary and unreliable in this context. irrelevant, other
Wikipedia er ekki áreiðanleg heimild
þarsemhversemgeturskrifaðogbreytt
gögnum,aukþesserheimildnotkuná
íslensku útgáfunni ekki alltaf vel rit-
skoðað.Wikipedia is not a reliable source since any-
one can write and edit data; furthermore,
theuseofsourcesintheIcelandicversionis
not always well-reviewed.Wikipedia
Almennterekkitaliðviðeigandiaðnota
Wikipedia sem heimild þar sem ein-
staklingar geta ávallt breytt upplýsin-
gunum.Generally,itisnotconsideredappropriateto
use Wikipedia as a source since individuals
can always change the information.Wikipedia
Þetta virðist aðeins utan fyrir efni
spurningarinnar.Thisseemsslightlyoff-topicforthequestion. irrelevant
Þettavirðistverapersónulegblogsíða.
Hún vísar í mikla tölfræði og góð gögn
en vísar ekki í neinar heimildir, en
er skrifuð af sérfræðingi með mikla
reynslu, svo það er matsatriði hvort
þetta sé nógu traustverðug heimild.This seems to be a personal blog. It refers
toalotofstatisticsandgooddatabutdoes
not cite any sources; however, it is written
byanexpertwithextensiveexperience,so
itisamatterofjudgmentwhetherthisisa
sufficiently reliable source.blog, no references
Flaggaði þessa heimild annarsstaðar
þar sem talað var um staðreyndir en
hún er vel viðeigandi hér þar sem er
verið að tala um hver umræðan í sam-
félaginu sé um þessa fullyrðinguI flagged this source elsewhere where facts
werebeingdiscussed, butitisveryappro-
priateheresincethetopiciswhatthepublic
debate is regarding this claim.other
Þettaerskoðanagreinsemerekkifræði-
lega traust.This is an opinion piece that is not theoreti-
cally sound.opinion piece
Þetta kemur af bloggsíðu sem er ekki
hlutlaus og ekki fræðilega traust.This comes from a blog that is not neutral
and not academically reliable.blog, partisan
source
Þetta er skoðanagrein sem er ekki hlut-
laus og ekki fræðilega traust.This is an opinion piece that is not objective
and lacks academic rigor.opinion piece, par-
tisan source
Þettaer skoðanagreinsem hentarekki
fyrir fræðilega umfjöllun.Thisis anopinionpiecethat isnotsuitable
for academic discussion.opinion piece

Curated retrieval versus open web search30
Table 2 – continued
Comment (Icelandic) Translation (English) Reason codes
Þetta virðist vera aðsend grein á net-
miðli sem notar ekki heimildaskrá, svo
mérfinnsthúnekkinógutraustverðug.This seems to be a contributed article on an
onlineoutletthatdoesnotuseabibliogra-
phy, so I do not find it trustworthy enough.opinion piece, no
references
Ekki áreiðanleg heimild. Þessi heimild
erfrásjálfstæðunetblaðisemopinber-
lega nýtur gervigreind við skrif frét-
tagreina og virðist aðallega hafa einn
höfundsvoþaðvirðistekkihafanógu
góða ritstjórn. Þessi tiltekna grein var
leiðréttfyriraðhafarangarstaðhæfin-
garsvoþaðerekkiviðeigandiaðnota
þetta sem heimild.Notareliablesource. Thissourceisfroman
independent online newspaper that openly
usesartificialintelligencetowritenewsar-
ticles and appears to have mostly a single
author, so it does not seem to have suffi-
cient editorial oversight. This particular
article was corrected for containing false
statements, so it is not appropriate to use it
as a source.poor quality
Þetta er skoðanagrein sem er ekki hlut-
laust né fræðilega traust.Thisis anopinion piecethat isneither neu-
tral nor academically sound.opinion piece, par-
tisan source
Þetta er skoðanagrein sem er líklega
ekki hlutlaus né fræðilega traust.Thisisanopinionpiecethatislikelyneither
neutral nor academically sound.opinion piece, par-
tisan source
ÞaðertilbetriheimildumMaastricht-
skilyrðin á Evrópuvefnum.There is a better source on the Maastricht
criteria on Evrópuvefurinn.cite primary
Ekki nota blogg, ekki áreiðanlegt.
Frekarakademískatexta,greinarívir-
tum blöðum etc.Do not use blogs, they are not reliable.
Rather use academic texts, articles in rep-
utable journals, etc.blog
Ekki hægt að tala um skoðanir ungs
fólks þegar þær koma ekki fram hér.
Aðeins fjallað um skoðanir fullorðinna
í Framsókn á Norðurlandi og Lilja Al-
freðsdóttir er frá Reykjavík svo hún
getur ekki talað fyrir Framsókn út á
landi.Itisnotpossibletotalkabouttheviewsof
young people when they are not presented
here. It only discusses the views of adults
in the Progressive Party in the North, and
Lilja Alfreðsdóttir is from Reykjavík, so she
cannot speak for the Progressive Party in
the countryside.irrelevant
Ekkiviðeigandiþarsemhérerhvorki
talað um Framsókn né við Framsóknar-
mannNot applicable since there is neither men-
tion of Framsókn (the Progressive Party)
here,noristhisaddressedtoamemberof
Framsókn.irrelevant
Væri betra að vitna beint í þessi orð
framkvæmdastjóra Deutsche Bank.Would it be better to directly quote these
words from the CEO of Deutsche Bank?cite primary
Ekkinægilegavelskrifuðgrein,skringi-
leg uppsett, sem er alltaf merki um
óáreiðanleikaNot a well-enough written article, strangely
formatted, which is always a sign of unreli-
ability.poor quality
Sama hér Same here other
Ekki Wikipedia takk, allir og amma
þeirra sem geta skrifað inn á hana.No Wikipedia please, everyone and their
grandmother can write on it.Wikipedia
Lesandiþarfáskrifttilaðlesagreininga,
er það vandamál mögulega?The reader needs a subscription to read the
analysis, could that potentially be a prob-
lem?paywalled

Curated retrieval versus open web search31
Table 2 – continued
Comment (Icelandic) Translation (English) Reason codes
Frumheimildhér,alltofskoðunarglaður
hér. Já, atkvæðagreiðslan snýst um
hvortumræðureigaaðhefjastaðnýju
en ef það á að hefja aftur umræður er
planiðaðfaraíEvrópusambandið. Þú
sækir ekki um starf og ferð í 5 atvin-
nuviðtöl ef þú ætlar þér ekki að taka
starfinu ef það býðst þér.Primarysourcehere,waytooopinionated
here. Yes, the vote is about whether to
resume talks, but if talks are to be resumed,
the plan is to join the European Union. You
don’tapplyforajobandgoto5interviewsif
youdon’tintendtotakethejobifit’soffered
to you.opinion piece, cite
primary
Hér á gervigreindin að nota up-
prunaleguheimildinasemergreinHan-
nesarHólmsteinsþarsemíþessarifrétt
er greinin einungis lauslega þýdd.Here, the AI should use the original source,
which is Hannes Hólmsteinn’s article, as
thisnewspieceisonlyaloosetranslationof
that article.cite primary
Þessiskoðanagreinerekkihlutlausog
ekki fræðilega traust.This opinion piece is notneutral and is not
academically sound.opinion piece, par-
tisan source
Þessi heimild er ekki fræðilega traust. This source is not academically reliable. poor quality
Heimildin er ekki fræðilega traust. The source is not academically reliable. poor quality
Get ekki opnað hlekkinn. I can’t open the link. broken link
Upprunaleg heimild af bloggi, ekki
viðeigandi hérOriginalsourcefromablog,notappropriate
hereblog, irrelevant
Ekki nægilega áreiðanleg þar sem hö-
fundur vísar ekki í heimildir. Vil sjá
betri heimild hér, helst íslenska.Notsufficientlyreliableastheauthordoes
notcitesources. Iwanttoseeabettersource
here, preferably an Icelandic one.no references, cite
primary
Þarf að finna greinina sem grevi-
greindin vísar í, kemur ekki í þessum
hlekk. Effundiðerréttanhlekkerþessi
heimild í lagiThearticlethattheAI referstoneedstobe
found; it does not appear at this link. If the
correct link is found, this source is fine.broken link
Værigottaðverameðnýlegriheimild
um afstöðu Íhaldsflokksins.It would be good to have a more recent
sourceontheConservativeParty’sposition.outdated
Væri betra að finna heimild sem segir
þetta beint en ekki frétt sem er unnin
upp úr Facebook færslu.It would be better to find a source that says
thisdirectly,ratherthananewsarticlebased
on a Facebook post.citeprimary,social
media
Þessi heimild fjallar ekki um þetta. This source does not deal with this. irrelevant
Þessiheimildfjallaraðallegaumáhrifin
sem Brexithafði áDanmörku. Trúlega
til meira viðeigandi heimild sem fjallar
um meira um Bretland.This source mainly deals with the impact
BrexithadonDenmark. Thereisprobably
amorerelevantsourcethatfocusesmoreon
the UK.irrelevant
Veit ekki hversu traust heimild blög-
gfærsla fyrrum ráðherra Sjálfstæðis-
flokksins er.I don’t know how reliable a source a blog
post by a former minister of the Indepen-
dence Party is.partisan source,
blog
AlmennterWikipediaekkitalinnógu
traust heimild til að vitna í.Generally, Wikipedia is not considered a
reliable enough source to cite.Wikipedia
Röngslóð,þaðþyrftiaðtakafrásíðasta
skástrikið svo rétt síða komi upp.Wrong URL, the trailing slash needs to be
removed so the correct page loads.broken link

Curated retrieval versus open web search32
Table 2 – continued
Comment (Icelandic) Translation (English) Reason codes
Heimild1erhvorkiáenskunéíslensku
sem gerir erfitt að meta áreiðanleika,
en auk þess er hún að tala gegn gagn-
rýninni á fjórfrelsinu. Það væri í lagi
ef vitnað væri í þessi mótrök, en það
er óæskilegt að nota þetta sem heimild
umröksemhöfundureraðgagnrýna,
auk þess að heimildinn vísar mjög stut-
tlegaíþessirökogútskýrirþauekkiað
fullu. Það væri miklu áreiðanlegra að
vísaíheimildsemferýtarlegaímálið,
geturútskýrthvaðþessirdómarsnérust
um og útskýrir ýtarlega rökinfrá þessu
sjónarhorni.Source 1 is neither in English nor Icelandic,
which makes it difficult to assess its reliabil-
ity;furthermore,itarguesagainstthecriti-
cism of the four freedoms. This would be
acceptableifthesecounterargumentswere
being cited, but it is undesirable to use this
as a source for arguments that the author
is criticizing, in addition to the fact that the
sourcereferstotheseargumentsverybriefly
and does other not fully explain them. It
would be much more reliable to refer to
a source that goes into the matter in de-
tail,canexplainwhatthesejudgmentswere
about, and thoroughly explains the argu-
ments from this perspective.foreign language,
cite primary
Vil heimildina annarsstaðar frá, formle-
grisíðu/greint.d.,áreiðanlegraþegar
erverið aðtalaum staðreyndirenhún
er flott þegar það er verið að tala um
skiptar skoðanir í seinustu málsgreinI want the source from elsewhere, a more
formal page/article for example, which is
morereliablewhendiscussingfacts,butit
is great when discussing differing opinions
in the last paragraph.opinion piece, cite
primary
Ég hef ekki heyrt um þessa fréttasíðu
áður og er lítið sem kemur upp
þegar ég gúggla nafnið. Mér sýnist
samt að allar upplýsingarnar séu
réttar, en svipaðar upplýsingar koma
líka fram hér í grein frá Evrópu-
vefnum sem mér þykir áreiðanlegri:
https://www.evropuvefur.is/svar.php?id=63420I have not heard of this news site before,
and very little comes up when I google
the name. However, it seems to me
that all the information is correct, but
similar information can also be found
here in an article from Evrópuvefu-
rinn, which I consider to be more reliable:
https://www.evropuvefur.is/svar.php?id=63420unknown outlet
Þettaerskoðanagreinsemerekkifræði-
legatraustnéritrýnd. Aukþesserhö-
fundur hennar yfirlýstur andstæðingur
ESB og er greinin því ekki hlutlaus.Thisisanopinionpiecethatisneitheraca-
demically robust nor peer-reviewed. Fur-
thermore,itsauthorisanavowedopponent
of the EU, and the article is therefore not
unbiased.opinion piece, par-
tisan source
Mögulega úreltar upplýsingar þar sem
húsnæðisverðhefurveriðaðlækkaólíkt
því sem pistlahöfundur spáði fyrir um.Possiblyoutdatedinformation,ashousing
prices have been decreasing, contrary to
what the columnist predicted.outdated
Set spurningamerki við að nota BS-
ritgerð sem trausta heimild.IquestiontheuseofaBSthesisasareliable
source.other
Sem eina heimildin sem svarið vitnar
ítelégekkiaðþessifréttsembyggirá
greiningum hagsmunaðila vera nægi-
lega traust.As the only source cited in the answer, I
do notconsider thisnewsarticle, whichis
based on analyses by stakeholders, to be
sufficiently reliable.partisan source
Hlekkurinn virkar ekki. The link does not work. broken link
Hlekkurinn virkar ekki. The link does not work. broken link
Þetta er bloggsíða sem er ekki ritrýnd
né fræðilega traust.Thisisablogthatisneitherpeer-reviewed
nor academically reliable.blog
Éghefekkiheyrtumþessasíðuáðurog
finnstólíklegtaðhúnséritrýndog/eða
áreiðanleg.I have not heard of this website before
and findit unlikelythat itis peer-reviewed
and/or reliable.unknown outlet

Curated retrieval versus open web search33
Table3: Audiencepoliticalprofile(Fjölmiðlanefnd2024),citationvolume,andexpertflags
for the surveyed Icelandic news outlets. L–R = left–right audience skew; N–I = nationalist
(lower) to internationalist (higher) audience skew; Indep. = % rating the outlet politically
independent; Cited = web-search answers citing it; Flags = expert source flags. RÚV and
Mannlíf were not cited.
Outlet L–R N–I Indep. Cited Flags
Heimildin−0.41+0.6049% 29 0
Samstöðin−0.36+0.5221% 4 0
RÚV−0.11+0.3762% 0 0
Vísir−0.02+0.2761% 33 8
DV+0.10+0.2538% 12 1
mbl.is+0.13+0.1631% 12 1
Mannlíf+0.22+0.2428% 0 0
Nútíminn+0.50+0.1430% 11 7
Fréttin.is+0.55−0.0822% 11 3
Útvarp Saga+0.64−0.2015% 30 12
Viðskiptabl.+0.64+0.3830% 11 0
Table 2 – continued
Comment (Icelandic) Translation (English) Reason codes
Útvarp Saga er þekkt sem fréttamiðill
semeropinberlegagegnaðildaðESB,
svo það er mögulega ekki alveg nógu
áreiðanleg heimild og betra að vísa í
heimildirnar sem fréttin er að vísa í.Útvarp Saga is known as a news outlet that
is officially opposed to EU membership, so
itmaynotbeafullyreliablesource,andit
is better to cite the sources that the article
itself refers to.partisansource,cite
primary
Efnið snýst um vísindalegt málefni,
og þá finnst mér það ekki nógu
traustverðugtaðvísaígreinumefniðá
fréttasíðu sem hefur engar heimildir.Thesubjectmatterisscientific,andtherefore
Idonotfinditcredibleenoughtorefertoan
article on the topic on a news website that
has no sources.no references, cite
primary
D Audience political profile of cited outlets
Table 3 reports, for each surveyed Icelandic news outlet, the political make-up of its
audienceintheFjölmiðlanefnd2024survey(Fjölmiðlanefnd,2024)alongsidehowoftenthe
web-search mode cited it and how many of its cited sources experts flagged (Section 5.6).
Audience skew is the share of an outlet’s readers at the high end of a scale minus the
share at the low end; the national population sits at +0.00on left–right and+0.22on
nationalism–internationalism, so positive left–right values mark a right-leaning audience
and lower nationalism–internationalism values a more nationalist one. Independence is
theshareofrespondentswithanopinionwhoagreetheoutletis“independentofpolitical
interests”. Outlets are ordered left to right by audience left–right skew.

Curated retrieval versus open web search34
References
Alyukov, M., Makhortykh, M., Voronovici, A., Sydorova, M., 2025. LLMs grooming
or data voids? LLM-powered chatbot references to Kremlin disinformation reflect
information gaps, not manipulation. Harvard Kennedy School Misinformation Review
doi:10.37016/mr-2020-187.
Alþingi, 2026. Þingsályktun um þjóðaratkvæðagreiðslu um framhald viðræðna um aðild
Íslands aðEvrópusambandinu. URL: https://www.althingi.is/altext/pdf/157/s/1
245.pdf. 157th legislative session, resolution adopted 28 May 2026.
Ananny,M.,Crawford,K.,2018. Seeingwithoutknowing: Limitationsofthetransparency
idealanditsapplicationtoalgorithmicaccountability. NewMedia&Society20,973–989.
doi:10.1177/1461444816676645.
Bastos, M.T., Mercea, D., 2019. The Brexit botnet and user-generated hyperpartisan news.
Social Science Computer Review 37, 38–54. doi:10.1177/0894439317734157.
Bélanger, F., Carter, L., 2008. Trust and risk in e-government adoption. The Journal of
Strategic Information Systems 17, 165–176. doi:10.1016/j.jsis.2007.12.002.
Bright, J., Enock, F.E., Esnaashari, S., Francis, J., Hashem, Y., Morgan, D., 2025. Generative
AIisalreadywidespreadinthepublicsector: evidencefromasurveyofUKpublicsector
professionals. DigitalGovernment: ResearchandPractice6,1–13. doi: 10.1145/3700140 .
Coalition for Content Provenance and Authenticity, 2025. Content credentials: C2PA
technicalspecification,version2.2. URL: https://spec.c2pa.org/specifications/spe
cifications/2.2/specs/C2PA_Specification. C2PA.
Einarsson, H., 2026a. Cross-lingual mathematical reasoning in LLMs: Evaluating per-
formance on icelandic vs. English problems, in: Proceedings of the RESOURCEFUL
WorkshopatLREC2026,EuropeanLanguageResourcesAssociation(ELRA).pp.89–95.
Einarsson, H.,2026b. MazeEval: Abenchmark fortestingsequential decision-makingin
language models, in: Proceedings of the Fifteenth Language Resources and Evaluation
Conference(LREC2026),EuropeanLanguageResourcesAssociation(ELRA),Palma,
Mallorca, Spain. pp. 407–418. doi:10.63317/4nm93hckcaf2.
Einarsson,H.,Harðarson,Ó.Þ.,Helgason,A.F.,Ólafsson,J.G.,Önnudóttir,E.H.,Þórisdóttir,
H., 2025. The 2024 alþingi election: Is extreme electoral volatility the new norm?
Icelandic Review of Politics and Administration 21, 1–34. URL: https://doi.org/10.1
3177/irpa.a.2025.21.1.1, doi:10.13177/irpa.a.2025.21.1.1.
Evrópuvefurinn,2013. UmEvrópuvefinn: tölfræðiogefnisyfirlit. URL: https://www.ev
ropuvefur.is/svar.php?id=88563. University of Iceland.
Fjölmiðlanefnd,2024. Fjölmiðlakönnun2024: mediause,trust,andpoliticalorientation
in Iceland. Public-opinion survey, fieldwork by Maskína. URL: https://fjolmidlanef
nd.is. media Commission of Iceland; survey microdata provided by the authors.

Curated retrieval versus open web search35
Fletcher,R.,Cornia,A.,Nielsen,R.K.,2020. Howpolarizedareonlineandofflinenews
audiences? Acomparativeanalysisoftwelvecountries. TheInternationalJournalof
Press/Politics 25, 169–195. doi:10.1177/1940161219892768.
Germain,T.,2026. IhackedChatGPTandGoogle’sAI–anditonlytook20minutes. URL:
https://www.bbc.com/future/article/20260218-i-hacked-chatgpt-and-googles-a
i-and-it-only-took-20-minutes. BBC Future, 18 February 2026.
Goldstein,J.A.,Sastry,G.,Musser,M.,DiResta,R.,Gentzel,M.,Sedova,K.,2023.Generative
language models and automated influence operations: Emerging threats and potential
mitigations. doi:10.48550/arXiv.2301.04246. arXiv:2301.04246.
GovernmentofIceland,2026. Governmentproposesreferendumonwhethertoreturn
toaccessiontalkswiththeEU. URL: https://www.government.is/diplomatic-missi
ons/embassy-article/2026/03/06/Government-proposes-referendum-on-whether-t
o-return-to-accession-talks-with-the-EU-/ .MinistryforForeignAffairs,6March
2026.
Gwet, K.L., 2008. Computing inter-rater reliability and its variance in the presence of
high agreement. British Journalof Mathematicaland Statistical Psychology61, 29–48.
doi:10.1348/000711006X126600.
Hemesath, S., Tepe, M., 2024. Public value positions and design preferences toward
AI-based chatbots in e-government: Evidence from a conjoint experiment with citizens
and municipal front desk officers. Government Information Quarterly 41, 101985.
doi:10.1016/j.giq.2024.101985.
Jaźwińska, K., Chandrasekar, A., 2025. AI search has a citation problem. URL: https:
//www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-b
ad-at-citing-news.php . Tow Center for Digital Journalism, Columbia Journalism
Review.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., Fung, P.,
2023. Surveyofhallucinationinnaturallanguagegeneration. ACMComputingSurveys
55, 1–38. doi:10.1145/3571730.
Ju, J., Meng, Q., Sun, F., Liu, L., Singh, S., 2023. Citizen preferences and government
chatbotsocialcharacteristics: Evidencefromadiscretechoiceexperiment. Government
Information Quarterly 40, 101785. doi:10.1016/j.giq.2022.101785.
Keller,F.B.,Schoch,D.,Stier,S.,Yang,J.,2020. PoliticalastroturfingonTwitter: Howto
coordinate a disinformation campaign. Political Communication 37, 256–280. doi: 10.1
080/10584609.2019.1661888.
Krippendorff, K., 2018. Content Analysis: An Introduction to Its Methodology. 4 ed.,
SAGE, Thousand Oaks, CA.
Kristjánsson,K.,Ingólfsson,S.T.,Agnarsson,S.,2026. Svörviðspurningumutanríkisnefn-
dar. Report.HagfræðistofnunHáskólaÍslands.Reykjavík. URL: https://ioes.hi.is/s

Curated retrieval versus open web search36
ites/ioes.hi.is/files/2026-05/Sv%C3%B6r%20vi%C3%B0%20spurningum%20utanr%C3
%ADkisnefndar%2C%20loka%C3%BAtg%C3%A1fa_leidrett%208.05_0.pdf.
Kuznetsova, E., Makhortykh, M., Vziatysheva, V., Stolze, M., Baghumyan, A., Urman, A.,
2025. In generative AI we trust: can chatbots effectively verify political information?
Journal of Computational Social Science 8. doi:10.1007/s42001-024-00338-8.
Landis,J.R.,Koch,G.G.,1977. Themeasurementofobserveragreementforcategorical
data. Biometrics 33, 159–174. doi:10.2307/2529310.
Larsen, A.G., Følstad, A., 2024. The impact of chatbots on public service provision: A
qualitative interview study with citizens and public service providers. Government
Information Quarterly 41, 101927. doi:10.1016/j.giq.2024.101927.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W.t.,Rocktäschel,T., Riedel,S., Kiela,D., 2020. Retrieval-augmentedgeneration
for knowledge-intensive NLP tasks, in: Advances in Neural Information Processing
Systems, pp. 9459–9474.
Li,X.,Wang,J.,2024. Shouldgovernmentchatbotsbehavelikecivilservants? theeffect
of chatbot identity characteristics on citizen experience. Government Information
Quarterly 41, 101957. doi:10.1016/j.giq.2024.101957.
Majithia,N.,Shinde,R.,Maskey,M.,Simperl,E.,Shadbolt,N.,2026. CitizenQuery-UK:
Benchmarking LLM performance in citizen queries about public information on gov.uk.
Technical Report. Open Data Institute. doi:10.61557/HDRO1281.
Marshall,H.,Drieschova,A.,2018. Post-truthpoliticsintheUK’sBrexitreferendum. New
Perspectives 26, 89–105. doi:10.1177/2336825X1802600305.
Metzger,M.J.,2007. MakingsenseofcredibilityontheWeb: Modelsforevaluatingonline
informationandrecommendationsforfutureresearch. JournaloftheAmericanSociety
for Information Science and Technology 58, 2078–2091. doi:10.1002/asi.20672.
Miðeind,2026. IcelandicLLMleaderboard. https://huggingface.co/spaces/mideind/
icelandic-llm-leaderboard. Accessed 15 June 2026.
Muennighoff, N., Tazi, N., Magne, L., Reimers, N., 2023. MTEB: Massive text embedding
benchmark, in: Proceedings of the 17th Conference of the European Chapter of the
Association for Computational Linguistics (EACL), pp. 2014–2037. doi: 10.18653/v1/20
23.eacl-main.148.
Newman, N., Fletcher, R., Robertson, C.T., Eddy, K., Nielsen, R.K., 2022. Reuters Institute
DigitalNewsReport2022.TechnicalReport.ReutersInstitutefortheStudyofJournalism,
University of Oxford. URL: https://reutersinstitute.politics.ox.ac.uk/digital
-news-report/2022.
NewsGuard, 2025. A well-funded Moscow-based global ‘news’ network has infected
western artificial intelligence tools worldwide with Russian propaganda. URL: https:
//www.newsguardtech.com/special-reports/moscow-based-global-news-network

Curated retrieval versus open web search37
-infected-western-artificial-intelligence-russian-propaganda/ . newsGuard
Special Report, 6 March 2025.
OECD, 2024. Governing with Artificial Intelligence: Are Governments Ready? Technical
Report OECD Artificial Intelligence Papers, No. 20. OECD Publishing. doi: 10.1787/26
324bc2-en.
Offenhartz, J., 2024. NYC’s AI chatbot was caught telling businesses to break the law. the
city isn’t taking it down. URL: https://www.nbcnewyork.com/news/local/nycs-ai-c
hatbot-was-caught-telling-businesses-to-break-the-law-the-city-isnt-takin
g-it-down/5287713/. Associated Press, 3 April 2024.
Ólafsson,J.G.,2021.Superficial,shallowandreactive: Howasmallstatenewsmediacovers
politics. Nordicom Review 42, 70–86. URL: https://doi.org/10.2478/nor-2021-0018 ,
doi:10.2478/nor-2021-0018.
Ólafsson, J.G., Jóhannsdóttir, V., 2024. Polarisation, news consumption, and beliefs in
misinformation and conspiracy theories: Early signs of the fragmentation of the public
sphereiniceland. Javnost–ThePublic31,440–458. URL: https://doi.org/10.1080/
13183222.2024.2383905, doi:10.1080/13183222.2024.2383905.
Ómarsdóttir,S.B.,Ólafsson,J.G.,2024. Iceland,in: Lange-Ionatamishvili,E.,Svetoka,S.
(Eds.),Russia’sInformationInfluenceOperationsintheNordic-BalticRegion.NATO
Strategic Communications Centre of Excellence, Riga, pp. 62–73. URL: https://stratc
omcoe.org/publications/russias-information-influence-operations-in-the-nor
dic-baltic-region/314.
Onweller, H., Lumer, E., Huber, A., Ramchandani, P., Subbiah, V.K., Feld, C., 2026. Cited
butnotverified: ParsingandevaluatingsourceattributioninLLMdeepresearchagents.
doi:10.48550/arXiv.2605.06635. arXiv:2605.06635.
Schlicht, E.J., 2024. Evaluating the propensity of generative AI for producing harmful
disinformation during the 2024 US election cycle. doi: 10.48550/arXiv.2411.06120 .
arXiv:2411.06120.
Walters, W.H., Wilder, E.I., 2023. Fabrication and errors in the bibliographic citations
generated by ChatGPT. Scientific Reports 13, 14045. doi: 10.1038/s41598-023-41032-5 .
Wang, R.Y., Strong, D.M., 1996. Beyond accuracy: What data quality means to data
consumers. Journal of Management Information Systems 12, 5–33. doi: 10.1080/074212
22.1996.11518099.
Wardle, C., Derakhshan, H., 2017. Information Disorder: Toward an Interdisciplinary
Framework for Research and Policy Making. Council of Europe report DGI(2017)09.
CouncilofEurope.Strasbourg. URL: https://edoc.coe.int/en/media/7495-informa
tion-disorder-toward-an-interdisciplinary-framework-for-research-and-polic
y-making.html.

Curated retrieval versus open web search38
Williams,A.R.,Burke-Moore,L.,Chan,R.S.Y.,Enock,F.E.,Nanni,F.,Sippy,T.,Chung,Y.L.,
Gabasova,E.,Hackenburg,K.,Bright,J.,Carrasco-Farré,C.,Wu,J.,2025. Largelanguage
models can consistently generate high-quality content for election disinformation
operations. PLOS ONE 20, e0317421. doi:10.1371/journal.pone.0317421.
Wilson, E.B., 1927. Probable inference, the law of succession, and statistical inference.
Journal of the American Statistical Association 22, 209–212. doi: 10.1080/01621459.192
7.10502953.
Zhou, L., Schellaert, W., Martínez-Plumed, F., Moros-Daval, Y., Ferri, C., Hernández-
Orallo,J.,2024a. Largerandmoreinstructablelanguagemodelsbecomelessreliable.
Nature 634, 61–68. doi:10.1038/s41586-024-07930-y.
Zhou, Y., Liu, Y., Li, X., Jin, J., Qian, H., Liu, Z., Li, C., Dou, Z., Ho, T.Y., Yu, P.S., 2024b.
Trustworthiness in retrieval-augmented generation systems: A survey. doi: 10.48550/a
rXiv.2409.10102. arXiv:2409.10102 [cs.CL].