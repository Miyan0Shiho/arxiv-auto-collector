# Mod-Guide: An LLM-based Content Moderation Feedback System to Address Insensitive Speech toward Indigenous Ethnic and Religious Minority Communities

**Authors**: Dipto Das, Achhiya Sultana, Ankit Singh Chauhan, Saadia Binte Alam, Mohammad Shidujaman, Shion Guha, Sunandan Chakraborty, Syed Ishtiaque Ahmed

**Published**: 2026-06-11 14:28:18

**PDF URL**: [https://arxiv.org/pdf/2606.13397v1](https://arxiv.org/pdf/2606.13397v1)

## Abstract
Language operates as a mechanism of both marginalization and resistance, especially for minority communities navigating insensitive and harmful speech online. As content moderation increasingly depends on large language models (LLMs), concerns arise about whether these systems can recognize culturally insensitive speech-language that disregards or marginalizes the cultural and religious perspectives of historically underrepresented communities, often through implicit erasure, misrepresentation, or normative framing, rather than overt hostility. Focusing on Bangladesh's Hindu and Chakma communities -- the country's largest religious and Indigenous ethnic minorities, respectively -- this paper investigates the epistemic limits of LLM-based moderation systems and explores methods for incorporating minority perspectives. We co-created a culturally grounded corpus of insensitive speech with community members and integrated their narratives into moderation pipelines using retrieval augmented generation (RAG). Our tool, Mod-Guide, improves LLM sensitivity to minority viewpoints by leveraging contextual cues derived from lived experience. Through mixed-method evaluations involving both minority and majority participants, we demonstrate that RAG-enhanced moderation responses are more contextually accurate and perceived differently across ethnic lines. This work advances research in human-computer interaction, AI ethics, and social computing by foregrounding restorative justice and hermeneutical inclusion in the design of content moderation systems.

## Full Text


<!-- PDF content starts -->

Mod-Guide : An LLM-based Content Moderation Feedback
System to Address Insensitive Speech toward Indigenous Ethnic
and Religious Minority Communities
Dipto Das
Department of Computer Science
University of Toronto
Toronto, Ontario, Canada
dipto.das@utoronto.caAchhiya Sultana
Independent University Bangladesh
Dhaka, Bangladesh
achhiyasets@iub.edu.bdAnkit Singh Chauhan
Indiana University Indianapolis
Indianapolis, Indiana, United States
ankichau@iu.edu
Saadia Binte Alam
Independent University Bangladesh
Dhaka, Bangladesh
saadiabinte@iub.edu.bdMohammad Shidujaman
Independent University Bangladesh
Dhaka, Bangladesh
shidujaman@iub.edu.bdShion Guha
Faculty of Information
University of Toronto
Toronto, Ontario, Canada
shion.guha@utoronto.ca
Sunandan Chakraborty
Indiana University Indianapolis
Indianapolis, Indiana, United States
sunchak@iu.eduSyed Ishtiaque Ahmed
Department of Computer Science
University of Toronto
Toronto, Ontario, Canada
ishtiaque@cs.toronto.edu
Abstract
Language operates as a mechanism of both marginalization and
resistance, especially for minority communities navigating insen-
sitive and harmful speech online. As content moderation increas-
ingly depends on large language models (LLMs), concerns arise
about whether these systems can recognize culturally insensitive
speech–language that disregards or marginalizes the cultural and
religious perspectives of historically underrepresented communi-
ties, often through implicit erasure, misrepresentation, or norma-
tiveframing,ratherthanoverthostility. FocusingonBangladesh’s
Hindu and Chakma communities – the country’s largest religious
and Indigenous ethnic minorities, respectively – this paper inves-
tigates the epistemic limits of LLM-based moderation systems and
explores methods for incorporating minority perspectives. We co-
created a culturally grounded corpus of insensitive speech with
community members and integrated their narratives into moder-
ation pipelines using retrieval augmented generation (RAG). Our
tool,Mod-Guide,improvesLLMsensitivitytominorityviewpoints
byleveragingcontextualcuesderivedfromlivedexperience. Through
mixed-method evaluations involving both minority and majority
participants, we demonstrate that RAG-enhanced moderation re-
sponses are more contextually accurate and perceived differently
acrossethniclines. Thisworkadvancesresearchinhuman-computer
ThisworkislicensedunderaCreativeCommonsAttribution4.0InternationalLicense.
COMPASS ’26, Virtual Event, USA
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2702-3/2026/07
https://doi.org/10.1145/3811242.3819096interaction,AIethics,andsocialcomputingbyforegroundingrestora-
tive justice and hermeneutical inclusion in the design of content
moderation systems.
CCS Concepts
•Social and professional topics →Race and ethnicity ;Reli-
gious orientation ;Cultural characteristics ;•Human-centered
computing →Interactive systems and tools ;Empirical studies in
collaborative and social computing ; •Applied computing →Doc-
ument management and text processing.
Keywords
Minority, LLM, RAG, content moderation, ethics
ACM Reference Format:
Dipto Das, Achhiya Sultana, Ankit Singh Chauhan, Saadia Binte Alam,
Mohammad Shidujaman, Shion Guha, Sunandan Chakraborty, and Syed
Ishtiaque Ahmed. 2026. Mod-Guide : An LLM-based Content Moderation
Feedback System to Address Insensitive Speech toward Indigenous Eth-
nic and Religious Minority Communities. In ACM SIGCAS/SIGCHI Con-
ference on Computing and Sustainable Societies (COMPASS ’26), July 27–
31, 2026, Virtual Event, USA. ACM, New York, NY, USA, 14pages.https:
//doi.org/10.1145/3811242.3819096
1 Introduction
Language is more than a means of communication and is a form
of power [ 87]. It shapes social hierarchies, legitimizes authority,
and enables the marginalization–a process through which individ-
ualsandgroupsarepushedtotheperipheryofsocietybasedonat-
tributes like race, gender, ethnicity, religion, caste, nationality, lan-
guage,sexualorientation,etc.[ 25]. LinguisticmarginalizationandarXiv:2606.13397v1  [cs.HC]  11 Jun 2026

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
injuries manifest in online communities through hate speech, bul-
lying, political incitement, and other forms of insensitive speech.
In thecontextof this study, wedefine insensitivespeech as the lin-
guistic acts that–while not overtly hateful or profane–disregards,
misrepresents, trivializes, or marginalizes the cultural, religious,
or epistemic values of historically underrepresented communities.
Unlike hate speech, which is often explicit in its hostility or incite-
ment, insensitive speech manifests through dismissive framings,
culturally uninformed generalizations, or normative assumptions
rooted in majority worldviews. Its harm arises not only from in-
tent or content, but from its failure to recognize and respect the
situatedmeanings,livedexperiences,andinterpretiveframeworks
of minority groups.
Most platforms respond to harmful content by enforcing mod-
eration policies through a combination of human moderators and
algorithmic systems [ 49,71]. Recent advances in large language
models (LLMs) have enabled more scalable moderation [ 52,101],
but these models are predominantly shaped by and reinforce ma-
jority perspectives [ 60]. Given the epistemic underrepresentation
of the religious and Indigenous ethnic minorities, whose perspec-
tives and experiences with insensitive or harmful speech might
significantly differ from those of the majority in those LLM-based
content moderation systems, it would likely reinforce the societal
barrier between the majority and minority groups in the case of
understanding each other’s perspectives.
WefocusonBangladesh,wheretheHinduandChakmacommu-
nities are the largest religious and Indigenous ethnic minorities,
respectively [ 14]. Motivated by concepts of hermeneutical injus-
tice [28] and the divide between majority and minority conscious-
ness [22], we collaborated with members from those communities
tocurateacorpusofculturallyinsensitivestatements. Participants
described why specific speech acts were hurtful and problematic,
grounding their explanations in religious texts, oral histories, cul-
tural practices and rituals, lived experiences, and documents from
rights organizations. These insights reflect interpretive resources
thataretypicallyexcludedfromLLMtrainingdata. Tooperational-
ize these perspectives, we introduce Mod-Guide , an LLM-based
moderation feedback tool that uses retrieval-augmented genera-
tion (RAG) to ground moderation responses in this community-
sourcedcorpus. WhileRAGhasshownstrongperformanceacross
a range of NLP tasks [ 59], the significance of our work lies in
groundingRAGinepistemicallymarginalizedperspectivesandeval-
uating its implications for culturally sensitive moderation. We
evaluateMod-Guideusingamixed-methodstudywithparticipants
from the majority and minority communities, comparing its out-
puts to responses from the off-the-shelf GPT-4 model. Our analy-
sis shows that grounding LLM responses in minority perspectives
through RAG significantly affects how harmful speech is inter-
preted and moderated. We also find that the perceived usefulness
ofthesemoderationoutputsvariesbyethnicitybutnotbyreligion.
Thisworkmakestwokeycontributionsthatarewidelyrecognized
in HCI scholarship [ 99]:
•Dataset contribution : a curated and annotated corpus of cul-
turally insensitive speech from minority perspectives.•Artifact contribution : thedesignandevaluationofMod-Guide,
a feedback system that integrates these perspectives into the
workflow of LLM-based moderation.
Thisresearchcontributestoongoingdiscourseinhuman-computer
interaction, AI ethics, and social computing by centering epistem-
ically marginalized communities in data curation and system de-
sign. ItdemonstrateshowLLMscanbemademoresensitivetoplu-
ralisticnormsthroughcommunityparticipationandsocio-technical
design. Thefollowingsectionsdetailthesociolinguisticframingof
marginalization, the construction of our dataset, the design of the
LLM-RAG pipeline, and the empirical evaluation. We conclude by
reflecting on the challenges of scale, the normativity of dataset cu-
ration,andtheimplicationsfordesigntowardcommunity-centered
justice and fair content moderation systems.
2 Literature Review
Thissectionsituatesourworkattheintersectionoflinguisticmarginal-
ization, epistemic injustice, and automated content moderation.
First, we discuss linguistic marginalization and conceptualize in-
sensitive speech as a form of harm shaped by cultural and histor-
ical context. Next, we examine epistemic barriers between ma-
jority and minority communities through the lenses of Du Bois’
notion of the veiland Fricker’s hermeneutical injustice . Finally,
we review research on automated content moderation and large
language models, highlighting limitations in addressing culturally
contextual harms and motivating our community-grounded RAG-
based approach.
2.1 Linguistic Marginalization as Insensitive
Speech
Language plays a crucial role in shaping social hierarchies and
powerdynamics. Itestablishesnormativeandnon-normativeiden-
tities [15]. As such, people are marginalized through language, of-
ten in the form of bullying, hate speech, and threats. Similarly,
religious and ethnic minorities are also vulnerable to linguistic
injuries. Such injury arises not only from offensive speech tar-
geting certain religions and ethnicities but also from the mode
or ways those identities are positioned as dismissed and deval-
ued [15]. In this paper, we focus on linguistic injuries and vulner-
abilities, where exact words may not be explicitly offensive (e.g.,
name-calling), yet their conventional bearing–how words derive
power from historical and social conventions—can come across as
disregarding or diminishing the experiences, identities, practices,
and contexts of religious and ethnic minorities, which we dub as
insensitive speech.
To study the linguistic marginalization of religious and ethnic
minorities in Bangladesh, we need to understand their sociopoliti-
cal contexts. Religious minorities in Bangladesh, particularly Hin-
dus, have long faced marginalization characterized by both histor-
ical and ongoing violence [ 77]. The large-scale communal riots
andthedisproportionatetargetingofHindusduringtheLiberation
War illustrate this pattern [ 6,80]. In recent decades, assaults on
Hindu communities have increased, often fueled by social media
rumorsofreligiousinsultsagainstthemajority[ 30,43,79],suchas
the violence during the 2021 Durga Puja [ 38]. Furthermore, politi-
cal instability worsens the persecution, leading to targeted attacks

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
on Hindus [ 5,45], Christians [ 44], and atheists [ 27,85]. Similarly,
theIndigenousethnicminoritiesinBangladesh(knownas Adivasi)
face marginalization due to their ethnic and cultural differences
fromthemajorityBengalipopulation. Thesecommunities,particu-
larlyintheChittagongHillTracts,haveexperienceddisplacement,
settlement, encroachment on their ancestral lands, ethnocide, and
violence due to the region’s militarization since before the coun-
try’s independence [ 17,40]. Despite a peace accord in 1997, they
continue to struggle for autonomy and basic recognition of indi-
geneity to this day [ 16,68].
Recent scholarships in social computing and ICT for develop-
ment have looked into how these sociopolitical experiences of re-
ligious and ethnic minorities in Bangladesh manifest as everyday
linguistic marginalization in online communities in their interac-
tion with other users and content moderation. For example, [ 77]
explained how social psychology shapes the participation of reli-
gious minorities online, who, due to a fear of isolation, fall into
a spiral of silence, negotiate through the future uncertainties and
presentimpressionoffear,andaccommodatetheircommunication
with religious majority communities. Among the Indigenous com-
munities in Bangladesh, many share religious minority identities,
such as Chakma, Santhals, and Garo, who follow Buddhism, Hin-
duism, and Christianity, respectively [ 95]. Users from these com-
munitieshavemarkedlydifferentexperienceswithhatespeechon
online platforms compared to their peers from the majority com-
munity. The lack of urgency in addressing their experience with
explicitlyprofanespeechcreatesacleardisparityconcerningmem-
bership, rights, and participation as users of online platforms [ 90].
Takingthatintoaccount,effortstoaddressinsensitivespeechwith
conventionalbearingaremorelikelytobeinfluencedbymajoritar-
ianism and, hence, require additional contextual content modera-
tionanddependonincreasedawarenessamongmajorityreligious
and ethnic groups, such as the Bengali Muslims in Bangladesh.
2.2 Epistemic Barriers among Majority and
Minority
Marginalization of minorities often stems from entrenched tribal
stigmasurroundingattributeslikeethnicity,religion,language,and
cultural practices [ 35]. For example, in many contexts, misunder-
standings of minority religions’ practices and beliefs lead to un-
substantiated fear (e.g., Islamophobia [ 4]), misrepresentation (e.g.,
depictingnon-Abrahamicfaithsassatanicorpagan[ 89]), orexclu-
sion. Similarly, immigrants who speak different languages often
face suspicion or hostility, as their speech is perceived as secretive
or exclusionary, reinforcing their marginalization in the form of
xenophobia[ 57]. Scholarsarguethatsuchstigmaandmarginaliza-
tion are not the victims’ attributes but a feature of the society that
imposes it. Through various social processes, minorities’ symbols,
beliefs,practices,andphysicalconditionsaremadenon-normative
in society and are devalued or discredited to such an extent that
they adopt different coping mechanisms [ 35], such as hiding their
identities,avoidingsharingtheirexperiencesorwithdrawingfrom
socialinteractionsoutoffearofisolationandthedesiretoconform
to norms in both online and offline settings [ 77].In this paper, we seek to understand the experiences of reli-
giousandethnicminoritiesbeingmarginalized,ridiculed,andmis-
understood in the Bangladeshi social media sphere by combining
W.E.B. Du Bois’ concept of “the veil” [ 22] and Miranda Fricker’s
notion of hermeneutical injustice [ 28]. These theoretical angles
provide complementary lenses for understanding and addressing
the underlying processes that lead to the minorities’ marginaliza-
tion. DuBois’conceptualizationofthe“veil”highlightshowracial
minorities in the United States experience an imposed separation
that distorts their self-perception and hinders mutual comprehen-
sion across racial divides [ 22]. Recent work [ 77] in the context
of Bangladesh has highlighted how the religious minority commu-
nities feel a comparable divide between themselves and the reli-
gious majority, particularly in how their identities and practices
aremisinterpreted,leadingtoalienationandmarginalization. That
metaphorical veil between the majority and minority groups in
termsofethnicityorreligionfunctionsasanepistemicbarrier,pre-
venting adequate and effective intergroup understanding.
DrawingfromFricker’swork[ 28],thisepistemicdifferencecould
be dubbed hermeneutical injustice, where minority groups strug-
gle to make sense of their experiences due to the lack of neces-
saryconceptualresourceswithinnormativeepistemicframeworks
shapedbyreligiousandethnicmajorities’beliefsandpractices. For
example, theological interpretations (e.g., the role of idols in wor-
ship for Hindu minorities) and dietary practices of the ethnic mi-
nority communities (e.g., consumption of pork, frog, and alcohol)
areconsideredwrongfromtheperspectiveofthemajorityBengali
Muslims’ standpoint [ 77,90,91]. When members of the major-
ity community talk about those beliefs and practices, the minority
groups might deem such comments as stereotypical, condescend-
ing, insulting, and overall insensitive, which reinforces division
and further marginalizes minorities online.
Divisions between majority and minority groups are sustained
by institutionalized ignorance and a lack of empathy [ 22], while
dominant social norms and unconscious biases perpetuate injus-
tice against marginalized communities [ 28]. In online communi-
ties where religious and ethnic minorities encounter insensitive
speech, different moderation and feedback mechanisms could be
implemented with careful attention to the epistemologies of these
groups. More broadly, dismantling these barriers demands inclu-
siveepistemicpractices—encompassingknowledgeproduction,recog-
nition,andvalidation—tovalueminorities’perspectivesandfoster
interfaith communication and mutual understanding. These prac-
ticeswouldultimatelyaddressthepowerasymmetriesexperienced
byreligiousandethnicminoritiesonlinebyshapingthedesignand
governance of sociotechnical systems like online platforms.
2.3 Language Models in Moderating Insensitive
Speech
With the global adoption of online platforms and the diverse com-
munities they host, moderating harmful and insensitive speech
has become a complex sociotechnical challenge. Existing scholar-
ship has shown that perceptions of what constitutes harmful con-
tent and its severity vary significantly across cultural and social
contexts [ 50,81]. While platforms’ “institutional ethics” [ 81] do

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
not want to implement the perspectives of users who think any-
thing that does not pertain to a particular religious belief should
be removed, they rarely make an active effort in addressing the
hermeneutical injustice [ 50], i.e., the structural exclusion of mi-
nority perspectives in defining what counts as harmful. As online
communities grow, platforms must negotiate competing modera-
tionvalues(e.g.,communityidentity),philosophies(e.g.,nurturing
vs. punishing), and implementation styles (e.g., human vs. algo-
rithmic moderation) [ 21,49].
Particularlyfocusedonmoderationphilosophy,Seeringetal.[ 83]
examined how moderation can be conceptualized through differ-
ent metaphors, such as mentoring, law enforcement, and custo-
dianship. These metaphors shape how platforms and moderators
perceive their roles, influencing decisions and ethics about inter-
vention, the balance between users’ autonomy and governance,
and the prioritization of different cultures and values. As the plat-
formsadoptalgorithmicmoderationforthesakeofefficiency,these
societalcomplexitiesareoftenpawnedofftoalgorithmicsystems[ 49].
Language technologies have become central to automated content
moderation systems[ 92,96]. In terms of complexity and sophis-
tication, these systems range from simple keyword filters[ 47,48],
to task-specific models for sentiment analysis and hate speech de-
tection [ 20,72], to foundational large language models (LLMs) de-
ployedatscale[ 42,52,101]. WhilemultilingualLLMshaveshown
promising results in detecting explicit hate speech, fake news, and
discriminatorylanguage[ 51,74,75],theyoftenstrugglewithmore
subtle forms of disinformation and culturally coded insensitivity.
However, LLMs reflect and reinforce dominant cultural norms,
which can lead to representational harms, particularly for non-
Westerncommunities[ 13,34]. Priorresearchhasshownthatthese
models exhibit demographic (e.g., race, gender, nationality, reli-
gion, caste) [ 20,31–33,36], socioeconomic [ 8], and political bi-
ases [1], raising concerns about how automated moderation dis-
proportionatelyimpactsmarginalizedcommunities. Hence,recent
works have attempted to reconceptualize moderation by embed-
ding safety paradigms directly into LLM pipelines [ 9,42], wherein
they have examined how data selection and fine-tuning impacted
LLMs’economicandpoliticalbiases[ 1],howmodelresponsesvary
with culturally sensitive prompts [ 73], and found that persona-
based prompting can improve alignment with specific moderation
goals [54]. Studies highlighted how crowd-sourced data annota-
tion is subject to limited annotator expertise [ 53], dismissal of re-
ligious faiths [ 78], minorities’ underrepresentation [ 88,93], and
disproportionate association of toxicity with minorities [ 98]. Re-
trievalaugmentedgeneration(RAG)–amethodtoenhancelanguage
model outputs by retrieving relevant external documents while
generatingresponses[ 59],canbeaneffectivetechniquetoaddress
the concerns of LLM biases affecting content moderation [ 58,94].
However, there is a dearth of literature that has examined its ef-
fectiveness in moderating content around minority identity, espe-
cially in non-English languages and the Global South contexts.
Our work advances research at the intersection of content mod-
eration,LLMs,andlow-resourcelanguagecommunitiesintwokey
ways. First, we address the dataset challenge by constructing a
culturally grounded corpus of insensitive speech in Bengali, an-
notated and contextualized by members of underrepresented re-
ligious and ethnic minority communities in Bangladesh. Ratherthan relying on crowd-sourced or majority-labels that often ob-
scure minority perspectives, our approach centers the lived expe-
riences, interpretive frameworks, and rationales of those most af-
fectedbymarginalization. Second, webuildoninsightsfromprior
literature that persona-based prompting may help align LLM out-
puts with specific moderation philosophies [ 54,83] and RAG en-
hancesfactualaccuracyandcontextualgrounding[ 46,86]. Weim-
plementedthisinsightinourcontentmoderationfeedbacksystem,
Mod-Guide, in which we prompt an LLM to adopt various moder-
ation roles and ground its responses in the minority community-
sourced corpus using RAG. We evaluated which configurations–
combinationsofpromptsandthepresence/absenceofRAG–produce
more contextually sensitive, factually accurate, and epistemically
inclusive feedback.
3 Methods
This paper is part of a broader study to understand minority com-
munities’ experiences with content moderation in online commu-
nities and develop tools to make those spaces more inclusive and
accessible for these communities [ 21,78,90]. Here, we build on
our findings and community relationships fostered during the ear-
lier phases of our research. Our study proceeded in three stages
(see Figure 1): (1) corpus preparation, (2) development of the Mod-
Guide moderation system, and (3) evaluation of moderation feed-
back.
3.1 Overview
First, we collaborated with 22 members from two minority com-
munities in Bangladesh–Hindu and Chakma–using the asynchro-
nous research community method to construct a corpus of cultur-
ally insensitive speech containing 132 instances and accompany-
ing explanations grounded in community perspectives. Second,
we integrated this corpus into a moderation pipeline that com-
bines persona-based prompting GPT-4 with retrieval-augmented
generation (RAG) to ground moderation feedback in community-
authored explanations. Third, we evaluated the system through a
mixed-method approach consisting of (a) quantitative analysis of
generated responses using text embeddings, (b) assessment of fac-
tual accuracy of generated texts by 2 experts, and (c) a user study
examining the perceived usefulness of moderation feedback with
15 participants from majority and minority communities. The fol-
lowing sections offer further details about each of these stages.
3.2 Author Positionality
Priorresearchhashighlightedhowtheresearchers’identitiesmay
reflexivelyaddressinevitabletensionsandbringaffinitiesintoper-
spectiveinstudyingmarginalizedcommunities[ 61,82]. Therefore,
weconsideritessentialtosituatethisworkonmarginalizedminor-
ity communities in the Global South in relation to the researchers’
positionality. Among all authors (2 women and 6 men), five were
bornandraisedinBangladesh, whiletheotherthreewerefromIn-
dia. Exceptforoneauthor(whoisfromaNorthIndianethnicback-
ground), all authors belong to the Bengali ethnolinguistic group.
Three authors identify as Bengali Hindus (the lead author from
an underprivileged caste in Bangladesh, the rest from a dominant

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
1. Corpus Preparation
ARC method
22 minority participants
Hindu, Chakma
Output:
Insensitive speech instances +
community explanations2. Mod-Guide Development
GPT-4
Persona prompting
RAG with community corpus
Output:
Moderator-like feedback on input texts3. Evaluation
Text embedding analysis
Expert Feedback (N=2)
User Study (N=15)
Output:
Textual difference, factual accuracy, useful-
ness assessment
Figure 1: Methods overview.
caste in West Bengal, India), and four authors were born in Mus-
lim communities. In addition to their varied sociocultural perspec-
tives, all authors’ backgrounds in computer science, with different
authors’ prior research with marginalized communities, text min-
ing, and data science, have informed and guided the motivation
and execution of this study.
4 Corpus Preparation to Understand Minority
Hermeneutics
We collected the corpus through the Asynchronous Remote Com-
munity(ARC)method[ 63]. Priorresearchhasusedthismethodto
engage with participants when in-person communication is diffi-
cult to arrange due to population distribution [ 62], stigma [ 64], or
fear of isolation [ 97]. Over a month, we nudged discussions from
those groups weekly to sustain engagement while allowing flexi-
bility for participants to share instances of social media posts they
found culturally insensitive, about which they have found that the
religious and ethnic majority communities have different percep-
tions. In doing so, our corpus prioritizes minority hermeneutics–
interpretation of their practices, experiences, values, and beliefs
from their own perspectives, over being shaped or constrained by
majoritarian normative societal views.
4.1 ARC Participants
In this paper, we focus on the religious minority Hindu commu-
nity and the Indigenous ethnic minority Chakma community in
Bangladesh. We recruited participants aged 18 years and older by
sharingtherecruitmentmaterialsandadditionalinformationwith
our personal networks, through Facebook advertisements, and by
reaching out to participants from our previous studies involving
thesecommunities. Wealsocontactedtheadministratorsandmod-
erators of local Facebook groups dedicated to these minority com-
munities, asking their permission to post the call for participation
in those groups. We asked the respondents to the study’s adver-
tisements to self-identify key characteristics such as gender, caste,
age, and their places of upbringing and current residence, which
priorstudiesfoundtohavedifferingexperienceswithintheHindu
and Chakma communities [ 77,90]. Our ARCs with these partici-
pants included 11 from the religious minority Hindu community
(7 male and 4 female) and 11 from the Indigenous ethnic minority
Chakmacommunity(2male,4female,and5didnotrespondtothe
question asking their gender). Both ARCs had more members, but
those who did not post at least once in the groups were excludedfrom the reported counts. While our religious minority partici-
pants came from various parts of the country, most of our ethnic
minority participants were from the Chittagong Hill Tracts (CHT)
region, where most Indigenous ethnic minority communities live.
Most of our Hindu participants were from underprivileged sched-
uled castes ( tafsili jati ) [84]. In addition to reflecting the general
demographic pattern of Hindu communities in Bangladesh, the
higher representation of participants from underprivileged castes
also resists the Brahminical and casteist interpretations of Hindu
beliefs and practices in our corpus.
4.2 Procedure
Similar to previous ARC studies [ 63,64,97], based on our partici-
pants’ preferences, we used a secret Facebook group and a secret
WhatsApp group, respectively, to interact with the former and the
latter minority groups. Hosting the ARCs on these online plat-
forms minimized the need to familiarize participants with a new
system [ 39,62]. All participants had existing Facebook and What-
sAppaccountsthattheyusedtoparticipateinthestudy,thusmain-
taining platform-related risks similar to those participants regu-
larly assume while using these communication channels. After
completing our informed consent procedure and orienting them
with a code of conduct, we invited them to join the groups. From
25/10/2024to23/11/2024,wemaintainedengagementthroughweekly
elicitation while allowing for flexibility.
However, a few participants either did not actively engage or
ceased participating after the first couple of weeks in those Face-
book and WhatsApp groups, which is a pattern of attrition and
participation consistent with previous ARC studies [ 76,97]. The
other participants responded to our prompts by sharing examples
of textual posts, comments, images, and videos they perceived as
insensitive to their religious and Indigenous ethnic identities, cul-
tures, rituals, and practices. We specifically sought instances that
were often dismissed as non-problematic by the religious and eth-
nicmajoritycommunities,astheparticipantsexperiencedthrough
interacting with friends and acquaintances in those communities
orhavingtheirreportsofsuchcontentoverlookedbycontentmod-
eration systems on online platforms. We also asked the partici-
pantstoexplainwhytheyfoundthecontentsinsensitive,referenc-
ingsourcessuchasthescripturesofthereligiousminoritycommu-
nities, national and international resolutions regarding the rights
and concerns of the Indigenous ethnic minorities, and their lived
experiences and understanding of their respective communities.

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
The participants also engaged with and built upon each others’ re-
sponses. The first two authors monitored the groups to ensure
compliance with the code of conduct and asked follow-up ques-
tions to nudge the participants to share additional details.
Our participants participated in the ARCs using both Bengali
and English. To streamline the corpus, we translated all written
communication into Bengali, the national language of Bangladesh.
Participants occasionally shared screenshots of social media posts
they considered insensitive. We utilized optical character recog-
nition (OCR) to convert those images into Bengali text. Similarly,
any videos shared by participants as examples were transcribed
into Bengali text. For web URLs provided by participants as in-
stances of insensitive speech, we transcribed the content into Ben-
gali. We anonymized all these contents before OCR and transcrip-
tion. For common examples of insensitive speech, some partici-
pants shared links to online repositories containing writings and
references explaining why such remarks are inappropriate. We
scraped the web pages in those cases, excluding non-textual con-
tent (e.g., HTML tags, URLs). While allowing the participants to
share screenshots, URLs, and videos made it easier for them to
share the examples of insensitive speech they encountered, using
OCR,transcription,andcleaningnon-naturallanguagecomponents
enabledtextualstandardization,allowingustoconvertimage-based
content into analyzable text for inclusion in both the RAG corpus
andthemoderationevaluationpipelineandallowedustopreserve
contextually rich, vernacular examples that participants consid-
ered important.
We gathered 53 instances of insensitive speech directed at the
religious minority Hindu community and 79 instances targeting
the indigenous ethnic minority Chakma community, organizing
them into two separate spreadsheets. Each spreadsheet contains
two columns: one listing examples of insensitive speech and the
other explaining their inappropriateness. Let’s consider the fol-
lowingexampletext (later referred as Insensitive Speech Example-1)
that Hindu participants in our ARCs found to be culturally insen-
sitive.
িকছুমানবতারেফিলওয ়ালােদরেদখেতিছ,মূি র্ত​পাহারা
িদেতমিন্দন্েরযােচ্ছ।মূির্ত​পাহারােদওয়ারজনযঈমান
আিনওনাই,মূির্ত​পাহারারপেক্ষআিমনাই।ভাঙ্গালাগেল
ডাকিদেয়ন(I have been seeing some vendors of hu-
manism who are going to temples to guard the idols.
Ididnotbring imaan(faithinIslam)forguardingthe
idols, [and] I am not in favor of guarding the idols.
Call [me] if those [idols] need to be broken.)
Since this example text was collected from the post of a user
belonging to the religious majority, it reflects their cultural value
andbelief: theprohibitiononidolworshipinIslam. Incontrast, in
the Hindu faith, idols are viewed as a medium for worship. Conse-
quently, a few of our participants pointed out the aforementioned
text that was recently well-circulating in the Bangladeshi social
media sphere as insensitive speech. They also explained why they
consider it culturally insensitive from different angles. For exam-
ple, while some participants explained the relevance of idols in
HinduritualsbasedonreferencesfromHinduscriptures,someoth-
ers presented arguments informed by their observations of social
practices in different religions. For example, an ARC participantshared the following explanations for why the above text was in-
sensitive based on different schools of thought within Hinduism:
There are many formless-theist communities in the
world who do not believe in incarnations and do not
require any tangible deity or symbol for worship or
spiritual practice. Again, some who accept formless-
theismstillacknowledgethenecessityofsymbols(such
as Om, the Dharma Wheel, or the Star of David) in
certain contexts. While they do not accept an exter-
nal image/idol of God, they still mentally envision
some form or symbol within their hearts. On this
matter,SwamiVivekanandaoncesaid: “Twotypesof
peopledonotrequireformsoridols–thosewhohave
noconcernforreligionat all, and theenlightenedbe-
ings who have transcended all such states. We exist
somewhere in between these two conditions. Inter-
nally and externally, we need some form of an idol
or image.”
We emphasize that our work does not seek to evaluate differ-
ent theological beliefs and practices. Rather, we aim to highlight
how various cultural, religious, and social values influence peo-
ple’s perceptions of content sensitivity and the roles they expect
moderators to fulfill. Hence, we will use this corpus of speech the
minority communities viewed as culturally insensitive and the ra-
tionales behind such perceptions to inform LLM-based automated
content moderation.
5Mod-Guide : Persona-based LLM Prompting
and RAG Pipeline for Moderation Feedback
This paper investigates the effectiveness of large language mod-
els (LLMs) in moderating insensitive speech directed at religious
and ethnic minority communities in Bangladesh, which is often
based on stereotypes and deepens the cultural divide between the
majority and minority communities in the country. Drawing on
Du Bois [ 22], we refer to that as the veil. We examined OpenAI’s
GPT-4 in particular. Additionally, we explore retrieval-augmented
generation (RAG) [ 59] based on community insights, with content
moderation in mind. We chose RAG over other approaches, such
as few-shot prompting or fine-tuning, to ensure interpretability,
adaptability,andalignmentwithcommunityperspectives. RAGal-
lowsgeneratedtextstobedirectlygroundedinretrievable,community-
authoredexplanations,preservingtraceabilityandculturalnuance[ 18].
Unlikefine-tuning,whichembedsknowledgeirreversiblyintomodel
weights, RAG supports modular updates as community insights
evolve. This approach preserves traceability, allows the corpus to
evolveascommunitiescontributeadditionalinsights,andsupports
modular updates as new examples are collected. Although RAG
introduces computational overhead compared to simple prompt-
ing, it offers an interpretable mechanism for integrating minority
hermeneutics into moderation feedback, which aligns with the ex-
ploratoryanddesign-orientedobjectivesofthiswork. Weprompted
these systems in different ways to shape their responses by defin-
ing their role as content moderators (e.g., nurturing, governing)
and persona (e.g., teacher, judge) to build a moderation feedback
system called “Mod-Guide.”

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
In this section, we will first discuss the different prompts, then
explaintheLLMandRAGpipeline,andfinallydiscusstheinterface
with examples.
5.1 Prompting
We used five prompts in the pipeline. In all these cases, the objec-
tive was the same: to understand whether a piece of text would
potentially be perceived as hate speech, hurtful, or culturally sen-
sitive. Those prompts featured different roles of moderators of on-
line interaction. Drawing on Seering et al. [ 83], we considered
five different roles content moderators play in moderating online
communities, such as (a) nurturing and supporting, (b) overseeing
and facilitating, (c) governing and regulating, (d) managing, and
(e) fighting for communities. Compared to other metaphoric con-
ceptualizations of moderation [ 65,102], this offers a computation-
ally tractable and interactionally diverse set of metaphors that can
be readily translated into promptable roles for language models.
Hence, based on Seering et al.’s recommendations [ 83], for these
different roles, we chose the metaphors teacher, mediator, judge,
representative, and protector, respectively.
•Prompt 1: As a supportive and nurturing content moderator like
a teacher, evaluate whether this text could be perceived as hate
speech, hurtful, orculturallyinsensitive. Considerifitmarginal-
izes,reinforcesstereotypes,orexcludesanygroup. Ifso,provide
constructive feedback by identifying concerns, explaining why
theymaybeproblematic, andsuggestingmoreinclusivealterna-
tives.
•Prompt 2: As a moderator who oversees and facilitates like a
mediator, consider whether this text could be perceived as hate
speech, hurtful, or culturally insensitive during interactions be-
tweenmajorityandminoritycommunitiesbasedonreligionand
ethnicity. If it might escalate tension, explain different perspec-
tivesandsuggestrevisionsthatpromoterespectfulandinclusive
dialogue.
•Prompt 3: Asacontentmoderatorwhogovernsandregulateslike
a judge, determine whether this text constitutes hate speech, is
hurtful, or culturally insensitive. Decide if it violates principles
of fairness, dignity, or inclusivity, and provide a clear ruling to
retain or remove the content.
•Prompt 4: Asamoderator familiarwith religiousand ethnicrela-
tionships in Bangladesh, assess whether this text represents the
country’s broader societal values. Consider if it could be per-
ceived as hate speech, hurtful, or culturally insensitive to mem-
bersofanycommunity. Providefeedbackbyhighlightingpoten-
tialissuesandsuggestingwaystofosterrespectfulandinclusive
dialogue.
•Prompt 5: As a content moderator who protects, advocates, and
looks out for religious and ethnic minorities like Hindus and
Chakmas, examine if this text could be perceived as hate speech,
hurtful, or culturally insensitive to them. Instead of reinforc-
ing stereotypes, erasing voices, or contributing to harm against
thesemarginalizedgroups,explainhowitcancenterrespectand
inclusion.
Weaddedanextrainstructiontoallfiveprompts—“Answerbriefly
and translate that in the Bengali language before responding”—
after observing that the LLMs, with or without RAG, tended torespondprimarilyinEnglishevenwhenpromptedinBengali. This
addition was intended to ensure that the feedback would be gener-
ated in Bengali.
5.2 LLM and RAG Pipeline
The RAG and LLM pipeline consisted of a data preprocessing and
ingestionphase,apromptingsteptodefinethetasksofthecontent
moderator, and the LLM or RAG component (see Figure 2). We
developed and operated the pipeline between December 2024 and
January 2025.
Figure 2: Prompt, LLM, and RAG pipeline.
To evaluate the LLM and RAG, we designed five prompts, as de-
scribed above, that embodied distinct moderator metaphors, each
reflecting a different moderation approach. The off-the-shelf LLM
we are using is GPT-4 from OpenAI, which supports controlled re-
trieval, where it is up to the language model to decide if retrieval
is necessary. We designed the script to do forced retrieval using a
separate system prompt, where we used the five prompts outlined
earlier to define the persona of the LLM-based content modera-
tion (see path 1 in Figure 2). Under the hood, OpenAI generates a
small query based on the prompt that triggers a retrieval tool call.
Next, we generated evaluation questions, where we asked if an
example from our corpus could be considered insensitive speech
by religious or ethnic minorities. Then, we asked these evalua-
tion questions to the LLM (see path 2 in Figure 2). The retrieval
tool performs a similarity search against this query in the vec-
torstore,whichcontainsembeddingsofknowledgecollectedfrom
the minority communities. The corpus collected from the minor-
ity communities provides additional cultural and situational con-
text, along with explanations of why these communities perceive
certain example texts as insensitive. The retrieved information is
then processed based on the system prompt from earlier to gen-
erate an output. The data is then processed through a pipeline
to build a retrieval-augmented generation (RAG) component us-
ing LangChain, allowing the LLM to reference it during inference.
Based on the general length of our pairs of example text and the
corresponding explanation of that being culturally insensitive, we
used recursive character text splitting with chunk size=512 and
k=2 so that the embeddings do not lose context, and both the text
and the explanation are retrieved if the pair is split between two

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
different chunks. We asked the same evaluation questions to the
LLM (see path 3 in Figure 2), but this time, it could utilize RAG.
Thus, we obtained two sets of responses–one from the standalone
LLMandanotherfromtheRAG-enhancedsystem,enablingacom-
parative evaluation of their effectiveness.
5.3 Interactive Interface
We developed an interactive user interface (UI) around our LLM
pipeline, enabling users to receive feedback on their texts while
leveraging RAG based on the community-sourced corpus and ex-
ploringdifferentmoderationpersonaswithoutrequiringpriorknowl-
edge of these mechanisms and prompt engineering. We chose a
web-based interactive interface due to its platform independence
andeaseofaccessacrossdifferentdevices. First,wecreatedahigh-
fidelityprototypeinFigma,whichservedasablueprintandguided
the UI’s development process and maintained design consistency
throughout the project. Then, we developed the final interface us-
ing React.js. Its use in the front end enhances performance due
to the framework’s virtual document object model and facilitates
seamless updates, resulting in a dynamic and responsive user ex-
perience. In the back end, we handled server-side logic and API
calls using Python.
Werefertotheintegratedplatform—comprisingthecorpus, the
LLMpipelinewithRAGandpromptvariations,andtheuserinterface—
asMod-Guide . This tool assists users in online communities to
identifyandavoidculturallyinsensitivespeech,simulatingtherole
of a content moderator. For instance, when we input the Insensi-
tive Speech Example-1 discussed in the previous section and asked
Mod-Guide to respond in the role of a mediator, it generated the
Bengali feedback shown in Figure 3that translates as:
Figure 3: Feedback from Mod-Guide’s in Mediator role.
Thistextcouldberegardedasinsensitiveoroffensive
to certain religious or cultural communities, specif-
ically those who value idol worship or temple prac-
tices. Thestatementdismissesthesignificanceofidol
protection and implies disrespect towards the faith
associated with these practices. To defuse potential
tensions, it is advisable to rephrase the statement tofocusonpromotingmutualrespectfordiversebeliefs.
A possible revision could be emphasizing the impor-
tance of understanding and respecting each other’s
religious practices, fostering a community where di-
versity in beliefs can coexist peacefully.
Thisfeedbackadoptssomehigh-levelinsightsandsimilarword-
ing from the explanations provided by the minority community
members in the corpus. However, the questions remain whether
theresponsesbecomesignificantlydifferentifLLMusesRAGbased
on the community-sourced corpus, whether the responses are fac-
tuallycorrect,andhowusersfromminorityandmajorityreligions
and ethnicities find those responses useful.
6 Evaluation of Moderation Feedback
We adopted a mixed-method evaluation approach in our study,
where we considered content moderation persona, whether the
community knowledge corpus was provided for RAG, and which
LLM model was used as independent factors. We compared the ef-
fectivenessoftheircombinationsinmoderatinginsensitivespeech
toward religious and ethnic minorities, in other words, addressing
hermeneutical differences of these communities with the majority
religious and ethnic group in the country. We evaluated the mod-
eration feedback based on three criteria by asking the following
questions in the evaluation phase:
(1)Difference in textual response:
(a)HowdovariouspromptsimpacttextgenerationinLLM-based
content moderation?
(b)How does the use of RAG impact text generation in LLM-
based content moderation?
(2)Factual accuracy: Is the feedback generated in LLM-based
content moderation, both without and with RAG, factually ac-
curate?
(3)Users’ perceived usefulness: How do people’s demographic
backgrounds and the persona of LLM-based content modera-
tion influence the perceived usefulness of the feedback?
6.1 Quantitative Analysis of Textual
Differences
To analyze textual differences and similarities between responses
generatedbyoff-the-shelfLLMGPT-4andthosegeneratedthrough
RAG with community-generated knowledge as context, we em-
ployedBERTScore,whichleveragescontextualembeddingstomea-
sure token similarity to offer strong alignment with human judg-
mentsandgreaterrobustnesstoadversarialparaphrasescompared
to traditional text generation metrics [ 103]. However, there is
a dearth of research on whether a metric like BERTScore works
well for low-resource languages like Bengali. While future NLP
research should look into the cross-language applicability of this
metric,ourevaluationtriedtoaddressthisconcernbyusingamul-
tilingual BERT model.
Tocomparewhetherandhowfivedifferentcontentmoderation
personas (reflected through prompts) influence the generated re-
sponsesfromtheLLM,weanalyzedtheresponses’varianceacross
differentprompts. First,weusedthe distiluse-base-multilingual
sentence encoder to find the embeddings of the responses gener-
ated for prompts reflecting different moderation personas. Then,

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
wecalculatedtheEuclideandistancesoftheembeddingsfordiffer-
ent pairs of prompts. Based on whether or not the distance scores
maintained normality in the Shapiro-Wilk test, we used a series of
parametricpairedt-testsornon-parametricWilcoxonsigned-rank
tests, respectively, to compare responses for ten pairs of persona
prompts based on the Euclidean distances of their embeddings.
In answering evaluation question 1(a), our null hypothesis was:
“There is no significant difference in the text generated by LLMs,
measuredbytheEuclideandistanceoftheirembeddings,forprompts
reflectingdifferentcontentmoderationpersonas.”WithBonferroni
correction, our results for all pairs of prompts ( 𝑝 < 10−22) pro-
vided strong evidence that there is a significant difference in the
text generated by LLMs for prompts reflecting different modera-
tion personas.
To answer question 1(b), we tested the influence of the use of
RAG on text generation using a similar approach. Since the dis-
tancesoftheembeddingsoftextsgeneratedbyoff-the-shelfGPT-4
from OpenAI and with RAG did not follow a normal distribution,
we used the Wilcoxon signed-rank test. Assuming a null hypothe-
sis: “ThereisnosignificanteffectofusingRAGontheresponsesof
theLLMs”. Weobtained 𝑝 = 3.3𝑒 −54 ,basedonwhichwerejected
the null hypothesis, i.e., we found strong evidence of RAG based
on community-sourced corpus affecting the generated texts.
6.2 Qualitative Analysis Responses’ Factual
Accuracy
There exist few studies focused on evaluating the factual accuracy
oflong-formtextgeneratedbyLLMswithoutanyhumaneffort[ 67].
Due to considerable disparities in resources and online presence,
these approaches remain unusable in non-English languages, like
Bengali. Moreover, especially in contexts of minority religious
faiths and Indigenous ethnic practices, where interpretations are
crucial,evaluationofmodelsbyhumanparticipantsismoreappro-
priate.
6.2.1 Expert Participants. We recruited two expert participants,
one from each minority community, through convenience sam-
pling[26]. Theexpert(E1)fromthereligiousminorityHinducom-
munity was 35 years old man. He was from an underprivileged
Hindu caste. He obtained a ( kabyotirtho ) certification from the
Bangladesh Sanskrit and Pali Education Board, demonstrating his
extensive knowledge of Hindu beliefs and scriptures. In addition,
hewasknowledgeableaboutlocalHindupracticesandexperiences
through his role as an administrator and moderator, and his in-
volvement in various social welfare initiatives aimed at religious
minorities. His background positions him as an expert who could
evaluateMod-guide’soutputswithoutreinforcingcasteistperspec-
tives. The expert (E2) from the ethnic minority community was a
32-year-old man. He has worked on issues affecting Indigenous
ethnic minority communities. Besides collaborating with commu-
nity members in the CHT region and the activist groups in the
national capital, he has also served as a young representative on
Indigenous rights at international venues. These participants did
not take part in the earlier corpus generation phase but were well
familiarwiththeirrespectivecommunities’cultures. Wepresented
them with ten randomly selected posts’ responses and explana-
tions generated in LLM-based content moderation, from GPT-4without and with RAG, and inquired whether the explanations
were factually accurate and where the LLMs’ responses were lack-
ing. Followingsharingtherandomsampleofresponsesasaspread-
sheet, the first author regularly communicated with the partici-
pants asynchronously over a week. To analyze their feedback, we
used iterative thematic coding, which is widely used in human-
computerinteractionresearch[ 12,66]. Inthisapproach,weidenti-
fied codes–identities, groups, topics, or issues that appeared repet-
itively across multiple iterations. We later aggregated the related
codes into broader themes.
6.2.2 Expert Feedback. The expert participant from the religious
minority community (E1) believed that the information provided
inmostresponsesfromtheLLMswassomewhatcorrect. However,
the responses obtained directly from GPT-4 were shallow com-
pared to the ones generated by augmenting its responses through
retrievalfromcommunity-sourceddata. Forexample,forthestate-
ment “Hindus should not worship idols” , participant E1 said,
I find the first response [from GPT-4] to somewhat
lack in depth. It correctly emphasizes the need to
respect and understand religious beliefs but does not
addressthecentraltopic[roleofidols]. [But,]thesec-
ond response: (“Some Hindus consider idol worship as
a way of expressing their devotion, a means of connect-
ing their souls to God.”) [from RAG] provides a more
nuanced perspective. While the verse mentioned is
correct, it is translated literally. It could be inter-
preted to recognize different theological traditions
within Hinduism regarding the role of idols in wor-
shiping.
The expert acknowledged that there are a few schools of Hindu
faith that do not permit idols in worship, but a general critique
of that comes across as culturally insensitive for Hindu minorities,
which is correctly mentioned in responses from LLM without and
with RAG. His explanation highlights that while the cited Vedic
verse mentioned in the text is accurate in a literal sense, its inter-
pretation varies across different Hindu sects. Therefore, prescrib-
ing a singular viewpoint, especially by someone who is not a reli-
gious minority, can be insensitive to diverse Hindu cultures.
However,inseveralotherinstances,theRAGbasedonthecommunity-
sourcedcorpusimprovedtheinformationqualityoftheresponses,
which directly influenced the accuracy of the feedback. For exam-
ple, evaluating the following statement made by a user from Ben-
galimajoritycommunity: “Those of you who are demanding that the
army be removed from the hills, you don’t even know that you don’t
know much!” , GPT-4 decided that “This text does not constitute
hate speech, is not hurtful, nor culturally insensitive.” Our expert
fromtheethnicminoritycommunity(E2)arguedthatthisresponse
does not take the experiences and perspectives of the Indigenous
ethnic communities in the Chittagong hill tracks. In contrast, the
response from LLM with RAG was “This statement could be per-
ceived as dismissive or condescending towards those requesting
the withdrawal of forces from hill areas. … For a more respectful
dialogue, consider framing it as a call for understanding and dia-
logue rather than an outright dismissal: Those who are demanding
to withdraw the army from the mountains, let us all discuss together

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
and try to know more. ”, which E2 found more insightful and accu-
rate.
In some cases, LLM, without and with RAG, fails to capture dif-
ferent problematic aspects of insensitive speech. Let’s consider
the following screenshot (see Figure 4) shared by one of our ARC
participants. After applying OCR on this, we retained only the
text but not the image. The text uses the term, ‘upojati’ ( 'উপজা-
িত', literal translation: sub-nation), which is often used as a slang
for the ethnic minorities, which they find offensive [ 91]. Partici-
pant E2 also focused on the text’s use of vulgar language (“cdi” is
a Romanized Bengali internet slang that means “fuck”) targeted at
Indigenouswomen. LLM’sresponsediscussedandreflectedonthe
former issue and recommended the “use [of] precise and accepted
terminology that members of these communities identify with. In
Bangladesh, ‘Adibashi’ or ‘Indigenous Peoples’ might be more ap-
propriate than ‘upojati’.” However, neither the use of GPT-4 nor
the use of RAG on top of that focused on the latter issue. This
shortcoming might be a result of not having enough context, pos-
sibly obtainable from the image or LLM’s systematic overlooking
of Indigenous women’s concerns.
Figure 4: A screenshot shared by an ARC participant.
6.3 Quantitative Analysis of Perceived
Usefulness
We conducted a quantitative evaluation to understand whether in-
dividuals from various religious and ethnic backgrounds find the
feedback from LLM-based content moderation useful and which
persona they prefer.
6.3.1 User Study. For this phase, we recruited a combination of
15 participants from the ethnic and religious majority and minor-
itycommunities,suchasBengalis,non-BengaliIndigenousgroups,
Muslims, and Hindus. Among those from the minority communi-
ties,threeparticipantsalsotookpartinthecorpuscollectionorfac-
tual accuracy evaluation phases. For a randomly selected sample
of texts, we presented the participants with feedback from LLMs
with five different content moderation prompts. To avoid possi-
ble inconsistencies among participants in interpreting Likert scale
levels [19], we asked them to identify the feedback they perceivedto be the most useful and explain why they found those more use-
ful compared to others. We analyzed how the demographic back-
groundandthecontentmoderationpersonaadopted(reflectedthrough
the prompts) influence the perceived usefulness of the LLMs’ feed-
back using the 𝜒2test with 𝛼 = 0.05 .
6.3.2 Usefulness of Persona and RAG-based Feedback. Intwosepa-
rate tests focusing on demographic attributes, religion and ethnic-
ity, we considered Bengali Hindus as the religious minority and
ethnic majority, respectively. Based on our data, we did not find
evidence( 𝑝 = 0.596 )toclaimthatthereisasignificantrelationship
between the participants’ religious identity and responses from
which persona they found useful. However, our data suggested
that there is a relationship ( 𝑝 = 0.0104 ) between whether the par-
ticipants were from the ethnic majority or the ethnic minority In-
digenous community and the response resulting from which mod-
eration persona prompt they found the most useful. We allowed
the participants to include small notes about the criteria they con-
sidered to decide the “usefulness” of the responses. Our partici-
pants shared that they prioritized factors such as empathic and in-
clusive language, promoting education and contextual awareness,
etc. However, deeper qualitative studies in the future should look
into whether and how different linguistic and informative aspects
are prioritized across demographic variations.
7 Limitations and Future Work
While this paper makes conceptual, technical, and methodologi-
cal contributions to the design of culturally sensitive moderation
systems, it has several limitations that warrant acknowledgment.
Sincethispaperisoneoftheinitialoutcomesofalargerprojectfo-
cusedonminoritycommunities’experienceswithcomputingtech-
nologies in the Global South, we also outline later in this section
how we plan to address those shortcomings in our future work.
First, the dataset used in Mod-Guide, while rich in contextual
and narrative depth, is relatively small. Such a limited size may
constrainthediversityofinsensitivespeechpatternscapturedand
reducetherecallcapacityofsemanticretrievalintheRAGpipeline.
It may also limit generalizability to other minority communities
in Bangladesh or to other communities across the Global South.
Second, the effectiveness of RAG depends on the semantic quality
of retrieved documents. While we used multilingual embeddings
to enable retrieval in Bengali, concerns about uneven embedding
qualityremain,especiallygiventhelow-resourcestatusofBengali
in NLP. Moreover, our RAG-based pipeline’s computational over-
headfortaskslikevectorstores, chunking, andretrievaltoolsmay
not be readily available in resource-constrained settings. Third,
the factual accuracy assessment in our paper involved only two
expert participants, which, while insightful, may introduce subjec-
tivity and reduce the evaluation’s robustness. Similarly, the use-
fulness study involved a small participant pool, and demographic
coverage was uneven across ethnic and religious groups. Our fu-
ture work to improve the tool will expand these evaluations by
including more diverse participants, and employing both qualita-
tive and quantitative measures (e.g., inter-rater reliability, Likert
ratings) to triangulate user perceptions. Fourth, we also acknowl-
edge our concerns about using OCR to extract Bengali text from
screenshots submitted by participants. While it was necessary to

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
incorporatereal-worldcontentthatoftencirculatesasimages,this
process may introduce errors or mistranscriptions (e.g., OCR limi-
tations on low-resolution images).
Finally, while our focus on two specific minority communities
in Bangladesh–Hindus and Chakmas enables rich, context-aware
analysis, it limits the applicability of findings to other religious,
ethnic, or linguistic groups. Additionally, even within the focal
communities, there exists internal diversity (e.g., caste, gender, re-
gionaldialects)thatoursamplemaynotfullycapture. Thus, while
oursystemdemonstratespromise,itsoutputsshouldbeinterpreted
as community-situated rather than universally representative. We
also recognize that moderation decisions, even when community-
informed, may reproduce power asymmetries or unintentionally
essentializeminorityidentities. Interpretationsofwhatconstitutes
“insensitive” speech are context-dependent and contested. Thus,
while Mod-Guide foregrounds community narratives, it must re-
main adaptable to revision, contestation, and critique through on-
goingparticipatorydesign. Ourfutureworkwillexpandthecommunity-
sourcedcorpustoincludeadditionalminoritygroupsinBangladesh,
includingBuddhistandChristiancommunities,aswellasotherIn-
digenous ethnic minority groups, such as the Marma and Santal
peoples. Further engagement within the Hindu and Chakma pop-
ulations could also examine intra-community variations, as identi-
fiedabove,toavoidessentializingminorityperspectives. Addition-
ally, future studies should investigate how corpus size and compo-
sitionaffectthequalityandcontextualaccuracyofRAG-generated
feedback in faith-based and culturally sensitive cases.
8 Discussion
We have described how we collaborated with two religious and
ethnic minority communities in Bangladesh to collect a corpus of
insensitive speech, how we used different moderation personas
to generate decisions and feedback on those examples of insen-
sitive speech from GPT-4 model and how we informed the LLM
through a RAG pipeline regarding the community-sourced expla-
nationsaboutwhythoseexamplesmightcomeacrossasculturally
insensitive for Bangladeshi Hindu and Chakma communities, and
evaluated the impact of different persona and community-sourced
explanation on LLMs’ text generation and their truthfulness and
usefulnessforusersfromdifferentdemographicbackgrounds. Mir-
roring that flow, in this section, we are going to reflect on how we
should regard the sizes and labeling of datasets collected through
collaboration with minority communities, why moderating, be it
human-run or LLM-based, content related to minority identities
andexperiencesshouldadoptarestorativejusticeperspective,and
how algorithmic audits should adopt explainability measures be-
sides their focus on biases.
8.1 Rethinking Dataset on Minorities as
Prototypical Resources
Compared to the vast amount of data traditionally used to train
LLMs [11], our corpus sourced from religious and ethnic minority
communities could be characterized as quite small and could be
viewed as a limitation of our study. However, dismissing thesecommunity contributions solely because of their size risks rein-
forcing epistemic erasure, where marginalized voices are system-
atically excluded from the development and evaluation of AI sys-
tems. This exclusion aligns with what Appadurai [ 7] describes as
ideocide–the systematic annihilation of the ethical and epistemo-
logical frameworks of marginalized groups. For example, how the
interpretation and labeling of a text about idol worship as “cul-
turally insensitive” vary between Hindu communities and Muslim
communities based on their distinct religious values and beliefs.
Let’s think of moderation in online communities as determining
the permissibility of content based on morality and ethics. We
need to consider whose ethics [ 2] are being guided by and whose
intelligence the AI systems, particularly those used for content
moderation, reinforce [ 3]. In the context of LLM training, the
scarcity of data from minorities is not just a technical issue but
also a reflection of broader socio-political inequalities in knowl-
edge production. Recognizing the limited number of example so-
cial media posts in our corpus that Bangladeshi religious and In-
digenous ethnic minorities find culturally insensitive, along with
the corresponding explanations of these views in our corpus, we
argue that the size of such a community-sourced corpus should
be viewed as a “prototype-based category” [ 55]. This definition
should not depend on straightforward rules about whether a cor-
pus is categorized as big or small based on the number of data
instances; instead, it should focus on their prototypical members–
similar to how a robin is a better example of a bird than an emu
or penguin. Similarly, a corpus that includes examples of cultur-
ally insensitive speech according to a wider range of religious mi-
norities, such as Hindus, Buddhists, and Christians, as well as In-
digenous ethnic groups like the Chakma, Marma, Garo, and San-
thal, would be a more comprehensive community-sourced corpus
compared to ours, which focuses solely on the Hindu and Chakma
communities. Therefore, while we recognize the need for future
worktoexpandanddiversifythesecorporathroughsustainedcom-
munity partnerships, we emphasize that datasets and corpora ob-
tained through collaboration with minority communities should
be viewed as prototypical examples that can be enhanced rather
than dismissed due to their small size.
8.2 Content Moderation for Restorative Justice
Scholars in social computing have studied content moderation on
online platforms as an exercise of discipline and punishment [ 21,
83]. However, recent works with Bangladeshi minority communi-
tiesrecommendthatthedesignandinteractioninonlinecommuni-
ties should promote restorative justice–an approach to addressing
harm that emphasizes healing, accountability, and repairing rela-
tionships rather than focusing solely on punishment [ 100]. This
approach involves dialogue among those affected–victims, offend-
ers, and the community to foster understanding and find mutually
agreed-uponresolutions. Itcanprovideaneffectiveframeworkfor
addressing the lack of intercultural knowledge between majority
and minority groups and for building trust among them. Rather
than relying on stereotypes and overlooking hermeneutical differ-
ences, ourapproachtoeducatingthemajorityreligiousandethnic
groups about the perspectives and experiences of minorities can
help build trust and lead toward restorative justice. Recognizing

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
diverse epistemologies instead of privileging majority worldviews
through LLM-based content moderation, community-sourced cor-
pora, such as those used to refine LLM-based moderation for re-
flectingtheperspectivesofBangladeshireligiousminorityHindus
and Indigenous ethnic minority Chakmas, can act as a form of
restorativeintervention,fosteringinterculturalknowledge-sharing
and shared meaning-making. Additionally, different moderation
personas (e.g., teacher, mediator) would facilitate conversations
within the community and enhance cultural awareness instead of
viewing users from different religions and ethnicities through a
dichotomy of victims and offenders. By integrating restorative
justice with AI ethics, social computing research can conceptual-
ize LLM-based content moderation systems that protect minority
groups,repairepistemicharms,andfosteronlinecommunitiesthat
promotetrustandreconciliationacrossculturalandreligiousveils.
8.3 Bias to Explainability in Algorithmic
Audits
Scholarshipsacrossdifferentfields, includinghuman-computerin-
teraction, social computing, algorithmic fairness, and natural lan-
guageprocessing, haveincreasinglyfocusedonbiasesinlanguage
technologies [ 20,70] and how they manifest in downstream ap-
plications [ 37,56]. Many of these studies use algorithmic audits
asamethodologicalapproach—empiricalinvestigationsthatexam-
ine public algorithmic systems for potentially problematic behav-
iors [10]. A central criterion these audits focus on is bias, defined
as the systematic and unfair discrimination by computing systems
against certain individuals or groups in favor of others [ 29], with
mitigation often framed as the relevant objective. When algorith-
mic systems, like LLM, are used in content moderation, it is essen-
tial to identify and address biases related to religious and ethnic
identities. However, ensuring transparency in decision-making
is equally important. Without clear explanations for moderation
choices,perceptionsoffavoritismmayarise. Forexample,Dasand
colleagues found that given the postcolonial relationship among
different religions in the region, when there is not enough clari-
fication, users from Bengali Hindu communities accused Quora’s
moderation of favoring Bengali Muslims, while users from the lat-
tergroupbelievedtheplatform’sdecisionswereinfluencedbyand
privileged the former [ 21]. This challenge of addressing biases
withadequateexplanationbecomesevenmorecomplexwhenmod-
eratingdiscussionsaboutreligiousbeliefsandculturalrituals. Given
this complexity, automated content moderation systems that rely
on AI should incorporate principles of explainable AI [ 23,69] to
improve interpretability. Keeping this concern in mind, in our
study, we chose RAG compared to few-shot prompting since the
former offers greater transparency and scalability, especially in
low-resource settings where examples must remain auditable and
epistemicallyaccountable[ 9,24]. Furthermore,auditsshouldbroaden
their focus beyond identifying and addressing bias to also include
explainability metrics [ 41], particularly in the downstream appli-
cations of LLMs, like in content moderation.9 Conclusion
Our paper develops a corpus of insensitive speech that may not be
directly hostile like hate speech but reinforces stereotypes, disre-
gards cultural values or marginalizes the perspectives of religious
andethnicminoritiesinBangladesh. Throughatoolwedeveloped
called “Mod-Guide” that poses different moderation roles and per-
sonas, we evaluated whether augmenting GPT-4’s text generation
by retrieving information from community-sourced explanations
can provide significantly different, accurate, and more useful in-
sights for users from diverse backgrounds compared to directly
using OpenAI’s GPT-4. While our approach offers a promising
pathway for fostering pluralistic understanding among religious
and ethnic majorities and minorities, challenges remain, includ-
ing the scalability of incorporating diverse perspectives. Future
work should examine reasoning in RAG, explore interdisciplinary
collaborations, and expand participatory approaches to improve
alignment between LLMs and other marginalized minority com-
munities.
Acknowledgments
ThisstudywaspartiallysupportedbytheInstituteofHealthEmer-
genciesandPandemicsPostdoctoralFellowship,theSchoolofCities
UrbanChallengeGrantattheUniversityofToronto,andanNSERC
Discovery Grant.
References
[1]AhmedAgiza,MohamedMostagir,andSheriefReda.2024. Politune: Analyzing
the impact of data selection and fine-tuning on economic and political biases
in large language models. In Proceedings of the AAAI/ACM Conference on AI,
Ethics, and Society , Vol. 7. 2–12.
[2]Syed Ishtiaque Ahmed. 2022. Situating ethics: A postsecular perspective for
HCI. Interactions 29, 4 (2022), 84–86.
[3]Syed Ishtiaque Ahmed. 2022. Whose intelligence? Whose ethics?: Eth-
ical pluralism and decolonizing AI. https://www.youtube.com/watch?v=
ReSbgRSJ4WY . last accessed: Feb 22, 2025.
[4]Chris Allen. 2016. Islamophobia . Routledge.
[5]The Prothom Alo. 2024. 5–20 August: 1068 minority homes and businesses
attacked (translated). https://www.prothomalo.com/bangladesh/6bm2lfn7bz .
last accessed: Feb 21, 2025.
[6]Tahmima Anam. 2013. Pakistan’s State of Denial. https://www.nytimes.com/
2013/12/27/opinion/anam-pakistans-overdue-apology.html . Last accessed:
July 7, 2023.
[7]Arjun Appadurai. 2015. Fear of Small Numbers. Writing Religion: The Case for
the Critical Study of Religion (2015), 73–95.
[8]Mina Arzaghi, Florian Carichon, and Golnoosh Farnadi. 2024. Understanding
IntrinsicSocioeconomicBiasesinLargeLanguageModels.In Proceedings of the
AAAI/ACM Conference on AI, Ethics, and Society , Vol. 7. 49–60.
[9]AnneArzberger,StefanBuijsman,MariaLuceLupetti,AlessandroBozzon,and
Jie Yang. 2024. Nothing Comes Without Its World–Practical Challenges of
Aligning LLMs to Situated Human Values through RLHF. In Proceedings of the
AAAI/ACM Conference on AI, Ethics, and Society , Vol. 7. 61–73.
[10]Jack Bandy. 2021. Problematic machine behavior: A systematic literature re-
viewofalgorithmaudits. Proceedings of the acm on human-computer interaction
5, CSCW1 (2021), 1–34.
[11]Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret
Shmitchell. 2021. On the dangers of stochastic parrots: Can language models
betoobig?.In Proceedings of the 2021 ACM conference on fairness, accountability,
and transparency . 610–623.
[12]Robert Bowman, Camille Nadal, Kellie Morrissey, Anja Thieme, and Gavin Do-
herty. 2023. Using thematic analysis in healthcare HCI at CHI: A scoping re-
view.In Proceedings of the 2023 CHI Conference on Human Factors in Computing
Systems. 1–18.
[13]Venetia Brown, Retno Larasati, Aisling Third, and Tracie Farrell. 2024. A Qual-
itative Study on Cultural Hegemony and the Impacts of AI. In Proceedings of
the AAAI/ACM Conference on AI, Ethics, and Society , Vol. 7. 226–238.
[14]Bangladesh Statistics Bureau BSB. 2022. Preliminary Report on Population
and Housing Census 2022 : English Version. https://sid.portal.gov.bd/

Mod-Guide: An LLM-based Content Moderation Feedback System COMPASS ’26, July 27–31, 2026, Virtual Event, USA
sites/default/files/files/sid.portal.gov.bd/publications/01ad1ffe_cfef_4811_
af97_594b6c64d7c3/PHC_Preliminary_Report_(English)_August_2022.pdf .
[Accessed: Jan 25, 2025].
[15]Judith Butler. 2021. Excitable speech: A politics of the performative . routledge.
[16]BhumitraChakma.2008. Assessingthe1997Chittagonghilltractspeaceaccord.
Asian Profile 36, 1 (2008), 93.
[17]Bhumitra Chakma. 2010. The post-colonial state and minorities: ethnocide in
the Chittagong Hill Tracts, Bangladesh. Commonwealth & comparative politics
48, 3 (2010), 281–300.
[18]JiaweiChen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
languagemodelsinretrieval-augmentedgeneration.In Proceedings of the AAAI
Conference on Artificial Intelligence , Vol. 38. 17754–17762.
[19]RobertACumminsandEleonoraGullone.2000.Whyweshouldnotuse5-point
Likert scales: The case for subjective quality of life measurement. In Proceed-
ings, second international conference on quality of life in cities , Vol. 74. 74–93.
[20]DiptoDas,ShionGuha,JedRBrubaker,andBryanSemaan.2024. The“Colonial
Impulse”ofNaturalLanguageProcessing: AnAuditofBengaliSentimentAnal-
ysis Tools and Their Identity-based Biases. In Proceedings of the 2024 CHI Con-
ference on Human Factors in Computing Systems . 1–18.
[21]Dipto Das, Carsten Østerlund, and Bryan Semaan. 2021. ” Jol” or” Pani”?: How
Does Governance Shape a Platform’s Identity? Proceedings of the ACM on
Human-Computer Interaction 5, CSCW2 (2021), 1–25.
[22]William Edward Burghardt Du Bois. 2015. Souls of black folk . Routledge.
[23]Upol Ehsan, Q Vera Liao, Michael Muller, Mark O Riedl, and Justin D Weisz.
2021. Expanding explainability: Towards social transparency in ai systems. In
Proceedings of the 2021 CHI conference on human factors in computing systems .
1–19.
[24]Upol Ehsan and Mark O Riedl. 2020. Human-centered explainable ai: Towards
a reflective sociotechnical approach. In International Conference on Human-
Computer Interaction . Springer, 449–466.
[25]Sheena Erete, Aarti Israni, and Tawanna Dillahunt. 2018. An intersectional
approach to designing in the margins. Interactions 25, 3 (2018), 66–69.
[26]Ilker Etikan, Sulaiman Abubakar Musa, Rukayya Sunusi Alkassim, et al. 2016.
Comparison of convenience sampling and purposive sampling. American jour-
nal of theoretical and applied statistics 5, 1 (2016), 1–4.
[27]Agence France-Presse. 2015. American atheist blogger hacked to death in
Bangladesh — theguardian.com. https://www.theguardian.com/world/2015/
feb/27/american-atheist-blogger-hacked-to-death-in-bangladesh . Last ac-
cessed July 7, 2023.
[28]Miranda Fricker. 2007. Epistemic injustice: Power and the ethics of knowing .
Oxford University Press.
[29]Batya Friedman and Helen Nissenbaum. 1996. Bias in computer systems. ACM
Transactions on information systems (TOIS) 14, 3 (1996), 330–347.
[30]Sumit Ganguly. 2021. Bangladesh’s Deadly Identity Crisis. https:
//foreignpolicy.com/2021/10/29/bangladesh-communal-violence-hindu-
muslim-identity-crisis/ . Last accessed: July 7, 2023.
[31]Sourojit Ghosh. 2024. Interpretations, Representations, and Stereotypes of
Caste within Text-to-Image Generators. In Proceedings of the AAAI/ACM Con-
ference on AI, Ethics, and Society , Vol. 7. 490–502.
[32]Sourojit Ghosh and Aylin Caliskan. 2023. Chatgpt perpetuates gender bias
in machine translation and ignores non-gendered pronouns: Findings across
bengali and five other low-resource languages. In Proceedings of the 2023
AAAI/ACM Conference on AI, Ethics, and Society . 901–912.
[33]Sourojit Ghosh and Aylin Caliskan. 2023. ’Person’== Light-skinned, Western
Man, and Sexualization of Women of Color: Stereotypes in Stable Diffusion.
arXiv preprint arXiv:2310.19981 (2023).
[34]Sourojit Ghosh, Pranav Narayanan Venkit, Sanjana Gautam, Shomir Wilson,
and Aylin Caliskan. 2024. Do Generative AI Models Output Harm while Rep-
resenting Non-Western Cultures: Evidence from A Community-Centered Ap-
proach. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society ,
Vol. 7. 476–489.
[35]E. Goffman. 2009. Stigma: Notes on the Management of Spoiled Identity . Touch-
stone.
[36]Kimia Hamidieh, Haoran Zhang, Walter Gerych, Thomas Hartvigsen, and
Marzyeh Ghassemi. 2024. Identifying implicit social biases in vision-language
models. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society ,
Vol. 7. 547–561.
[37]David Hartmann, Amin Oueslati, and Dimitri Staufer. 2024. Watching the
Watchers: A Comparative Fairness Audit of Cloud-based Content Moderation
Services. arXiv preprint arXiv:2406.14154 (2024).
[38]Mubashar Hasan. 2021. Minorities under attack in Bangladesh. https://www.
lowyinstitute.org/the-interpreter/minorities-under-attack-bangladesh . Last
accessed: July 7, 2023.
[39]Emma Heywood, Beatrice Ivey, and Sacha Meuter. 2024. Reaching hard-to-
reachcommunities: usingWhatsApptogiveconflict-affectedaudiencesavoice.
International Journal of Social Research Methodology 27, 1 (2024), 107–121.
[40]Glen Hill and Kabita Chakma. 2022. Muscular nationalism, masculinist mili-
tarism: the creation of situational motivators and opportunities for violenceagainst the Indigenous peoples of the Chittagong Hill Tracts, Bangladesh. In-
ternational Feminist Journal of Politics 24, 4 (2022), 519–543.
[41]Robert R Hoffman, Shane T Mueller, Gary Klein, and Jordan Litman. 2018.
Metrics for explainable AI: Challenges and prospects. arXiv preprint
arXiv:1812.04608 (2018).
[42]Sedat İnan, Hasan Çetin, and Nurettin Yakupoğlu. 2024. Spring water anom-
alies before two consecutive earthquakes (M w 7.7 and M w 7.6) in Kahraman-
maraş(Türkiye)on6February2023. Natural Hazards and Earth System Sciences
24, 2 (2024), 397–409.
[43]Amnesty International. 2021. Bangladesh: Protection of Hindus and others
must be ensured amid ongoing violence. https://www.amnesty.org/en/
latest/news/2021/10/bangladesh-protection-of-hindus-and-others-must-be-
ensured-amid-ongoing-violence/ . Last accessed: July 7, 2023.
[44]Minority Rights Group International. 2018. Christians. https://minorityrights.
org/minorities/christians-6/ . Last accessed: July 7, 2023.
[45]The Daily Ittefaq. 2014. Attacks on minorities continue. https:
//web.archive.org/web/20140110191737/http://www.clickittefaq.com/more-
stories/attacks-minorities-continue/ . Last accessed: July 7, 2023.
[46]Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with
generative models for open domain question answering. arXiv preprint
arXiv:2007.01282 (2020).
[47]Shagun Jhaver, Iris Birman, Eric Gilbert, and Amy Bruckman. 2019. Human-
machinecollaborationforcontentregulation: Thecaseofredditautomoderator.
ACM Transactions on Computer-Human Interaction (TOCHI) 26, 5 (2019), 1–35.
[48]Shagun Jhaver, Quan Ze Chen, Detlef Knauss, and Amy X Zhang. 2022. De-
signingwordfiltertoolsforcreator-ledcommentmoderation.In Proceedings of
the 2022 CHI conference on human factors in computing systems . 1–21.
[49]JialunAaronJiang,PeipeiNie,JedRBrubaker,andCaseyFiesler.2023. Atrade-
off-centeredframeworkofcontentmoderation. ACM Transactions on Computer-
Human Interaction 30, 1 (2023), 1–34.
[50]Jialun Aaron Jiang, Morgan Klaus Scheuerman, Casey Fiesler, and Jed R
Brubaker. 2021. Understanding international perceptions of the severity of
harmful content online. PloS one 16, 8 (2021), e0256762.
[51]Hellen Koka, Solomon Langat, Francis Mulwa, James Mutisya, Samuel Owaka,
Millicent Sifuna, Juliette R Ongus, Joel Lutomiah, and Rosemary Sang. 2024.
CombiningMorphologicalandMolecularToolsCanEnhanceTickSpeciesIden-
tification for Improved Tick-Borne Disease Surveillance Among Pastoral Com-
munities in Kenya. Vector-Borne and Zoonotic Diseases (2024).
[52]Mahi Kolla, Siddharth Salunkhe, Eshwar Chandrasekharan, and Koustuv Saha.
2024. Llm-mod: Can large language models assist content moderation?. In Ex-
tended Abstracts of the CHI Conference on Human Factors in Computing Systems .
1–8.
[53]Shanu Kumar, Gauri Kholkar, Saish Mendke, Anubhav Sadana, Parag Agrawal,
and Sandipan Dandapat. 2024. Socio-Culturally Aware Evaluation Framework
for LLM-Based Content Moderation. arXiv preprint arXiv:2412.13578 (2024).
[54]Louis Kwok, Michal Bravansky, and Lewis D Griffin. 2024. Evaluating cultural
adaptability of a large language model via simulation of synthetic personas.
arXiv preprint arXiv:2408.06929 (2024).
[55]George Lakoff. 2007. Cognitive models and prototype theory. The cognitive
linguistics reader (2007), 130–167.
[56]MichelleSLam, MitchellLGordon, DanaëMetaxa, JeffreyTHancock, JamesA
Landay,andMichaelSBernstein.2022. End-useraudits: Asystemempowering
communitiestoleadlarge-scaleinvestigationsofharmfulalgorithmicbehavior.
Proceedings of the ACM on Human-Computer Interaction 6,CSCW2(2022),1–34.
[57]Erika Lee. 2019. America for Americans: A history of xenophobia in the United
States. Basic Books.
[58]Maxyn Leitner, Rebecca Dorn, Fred Morstatter, and Kristina Lerman. 2025.
Characterizing Network Structure of Anti-Trans Actors on TikTok. arXiv
preprint arXiv:2501.16507 (2025).
[59]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-
intensivenlptasks. Advances in neural information processing systems 33(2020),
9459–9474.
[60]Tianlin Li, Xiaoyu Zhang, Chao Du, Tianyu Pang, Qian Liu, Qing Guo, Chao
Shen, and Yang Liu. 2024. Your large language model is secretly a fairness
proponent and you should prompt it like one. arXiv preprint arXiv:2402.12150
(2024).
[61]Calvin A Liang, Sean A Munson, and Julie A Kientz. 2021. Embracing four
tensions in human-computer interaction research with marginalized people.
ACM Transactions on Computer-Human Interaction (TOCHI) 28, 2 (2021), 1–47.
[62]Haley MacLeod, Grace Bastin, Leslie S Liu, Katie Siek, and Kay Connelly. 2017.
”BeGratefulYouDon’tHaveaRealDisease”UnderstandingRareDiseaseRela-
tionships. In Proceedings of the 2017 CHI Conference on Human Factors in Com-
puting Systems . 1660–1673.
[63]Haley MacLeod, Ben Jelen, Annu Prabhakar, Lora Oehlberg, Katie A Siek, and
KayConnelly.2016. Asynchronousremotecommunities(ARC)forresearching

COMPASS ’26, July 27–31, 2026, Virtual Event, USA Das, Sultana, Chauhan, Alam, Shidujaman, Guha, Chakraborty, and Ahmed
distributed populations.. In PervasiveHealth . 1–8.
[64]Juan F Maestre, Haley MacLeod, Ciabhan L Connelly, Julia C Dunbar, Jordan
Beck, Katie A Siek, and Patrick C Shih. 2018. Defining through expansion:
conductingasynchronousremotecommunities(arc)researchwithstigmatized
groups. In Proceedings of the 2018 CHI Conference on Human Factors in Comput-
ing Systems . 1–13.
[65]J Nathan Matias. 2019. The civic labor of volunteer moderators online. Social
Media+ Society 5, 2 (2019), 2056305119836778.
[66]Nora McDonald, Sarita Schoenebeck, and Andrea Forte. 2019. Reliability and
inter-rater reliability in qualitative research: Norms and guidelines for CSCW
and HCI practice. Proceedings of the ACM on human-computer interaction 3,
CSCW (2019), 1–23.
[67]Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei
Koh,MohitIyyer,LukeZettlemoyer,andHannanehHajishirzi.2023. Factscore:
Fine-grained atomic evaluation of factual precision in long form text genera-
tion. arXiv preprint arXiv:2305.14251 (2023).
[68]Mashfiq Mizan and Arafat Rahaman. 2025. Removal of word ‘adi-
vasi’: Indigenous group attacked at NCTB; 20 hurt — thedailystar.net.
https://www.thedailystar.net/news/bangladesh/news/removal-word-adivasi-
indigenous-group-attacked-nctb-20-hurt-3799851 . Last accessed 21-02-2025].
[69]Sina Mohseni, Niloofar Zarei, and Eric D Ragan. 2021. A multidisciplinary
survey and framework for design and evaluation of explainable AI systems.
ACM Transactions on Interactive Intelligent Systems (TiiS) 11, 3-4 (2021), 1–45.
[70]Jakob Mökander, Jonas Schuett, Hannah Rose Kirk, and Luciano Floridi. 2024.
Auditing large language models: a three-layered approach. AI and Ethics 4, 4
(2024), 1085–1115.
[71]Maria D Molina and S Shyam Sundar. 2022. When AI moderates online con-
tent: effectsofhumancollaborationandinteractivetransparencyonusertrust.
Journal of Computer-Mediated Communication 27, 4 (2022), zmac010.
[72]Marzieh Mozafari, Reza Farahbakhsh, and Noël Crespi. 2020. Hate speech de-
tection and racial bias mitigation in social media based on BERT model. PloS
one15, 8 (2020), e0237861.
[73]Abhijit Mukherjee, Poulomee Coomar, Soumyajit Sarkar, Karen H Johannes-
son, Alan E Fryar, Madeline E Schreiber, Kazi Matin Ahmed, Mohammad Ayaz
Alam, Prosun Bhattacharya, Jochen Bundschuh, et al. 2024. Arsenic and other
geogenic contaminants in global groundwater. Nature Reviews Earth & Envi-
ronment 5, 4 (2024), 312–328.
[74]Richard R Orlandi, Todd T Kingdom, Timothy L Smith, Benjamin Bleier, Adam
DeConde, Amber U Luong, David M Poetker, Zachary Soler, Kevin C Welch,
Sarah K Wise, et al. 2021. International consensus statement on allergy and
rhinology: rhinosinusitis 2021. In International forum of allergy & rhinology ,
Vol. 11. Wiley Online Library, 213–739.
[75]Flor Miriam Plaza-del Arco, Debora Nozza, Dirk Hovy, et al. 2023. Respectful
or toxic? using zero-shot learning with language models to detect hate speech.
InThe 7th workshop on online abuse and harms (woah) . Association for Compu-
tational Linguistics.
[76]AnnuSiblePrabhakar,LuciaGuerra-Reyes,VanessaMKleinschmidt,BenJelen,
Haley MacLeod, Kay Connelly, and Katie A Siek. 2017. Investigating the suit-
ability of the asynchronous, remote, community-based method for pregnant
and new mothers. In Proceedings of the 2017 CHI Conference on Human Factors
in Computing Systems . 4924–4934.
[77]Mohammad Rashidujjaman Rifat, Dipto Das, Arpon Poddar, Mahiratul Jannat,
Robert Soden, Bryan Semaan, and Syed Ishtiaque Ahmed. 2024. The Politics
of Fear and the Experience of Bangladeshi Religious Minority Communities
Using Social Media Platforms. Proceedings of the ACM on Human-Computer
Interaction 8, CSCW2 (2024), 1–32.
[78]Mohammad Rashidujjaman Rifat, Abdullah Hasan Safir, Sourav Saha, Ja-
hedul Alam Junaed, Maryam Saleki, Mohammad Ruhul Amin, and Syed Ish-
tiaque Ahmed. 2024. Data, Annotation, and Meaning-Making: The Politics of
Categorization in Annotating a Dataset of Faith-based Communal Violence. In
Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Trans-
parency. 2148–2156.
[79]Sajal Roy, Ashish Kumar Singh, et al. 2023. Sociological perspectives of social
media,rumors,andattacksonminorities: EvidencefromBangladesh. Frontiers
in Sociology 8 (2023), 1067726.
[80]Tanika Sarkar and Sekhar Bandyopadhyay. 2017. Calcutta: The stormy decades .
Taylor & Francis.
[81]Morgan Klaus Scheuerman, Jialun Aaron Jiang, Casey Fiesler, and Jed R
Brubaker. 2021. A framework of severity for harmful content online. Proceed-
ings of the ACM on Human-Computer Interaction 5, CSCW2 (2021), 1–33.
[82]Ari Schlesinger, W Keith Edwards, and Rebecca E Grinter. 2017. Intersectional
HCI: Engaging identity through gender, race, and class. In Proceedings of the
2017 CHI conference on human factors in computing systems . 5412–5427.
[83]Joseph Seering, Geoff Kaufman, and Stevie Chancellor. 2022. Metaphors in
moderation. New Media & Society 24, 3 (2022), 621–640.
[84]Dwaipayan Sen. 2018. The decline of the caste question: Jogendranath Mandal
and the defeat of Dalit politics in Bengal . Cambridge University Press.[85]Samira Shackle. 2018. Atheist bloggers in Bangladesh are still under threat
— New Humanist. https://newhumanist.org.uk/articles/5386/atheist-bloggers-
in-bangladesh-are-still-under-threat . Last accessed July 7, 2023.
[86]Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike
Lewis,LukeZettlemoyer,andWen-tauYih.2023. Replug: Retrieval-augmented
black-box language models. arXiv preprint arXiv:2301.12652 (2023).
[87]Divyanshu Kumar Singh, Dipto Das, and Bryan Semaan. 2025. The Power of
Language: Resisting Western Heteropatriarchal Normative Writing Standards.
InProceedings of the CHI Conference on Human Factors in Computing Systems
(CHI ’25) . Association for Computing Machinery, New York, NY, USA. doi:10.
1145/3706598.3714073
[88]Shanshan Song, Micaela Ashton, Rebecca Hahn Yoo, Zoljargal Lkhagvajav,
Robert Wright, Debra JH Mathews, and Casey Overby Taylor. 2025. Partic-
ipant Contributions to Person-Generated Health Data Research Using Mobile
Devices: ScopingReview. Journal of medical Internet research 27(2025),e51955.
[89]Sharada Sugirtharajah. 2004. Imagining Hinduism: A postcolonial perspective .
Routledge.
[90]Achhiya Sultana, Dipto Das, Saadia Binte Alam, Mohammad Shidujaman, and
SyedIshtiaqueAhmed.2024. ACivics-orientedApproachtoUnderstandingIn-
tersectionally Marginalized Users’ Experience with Hate Speech Online. arXiv
preprint arXiv:2410.14950 (2024).
[91]Sharifa Sultana, Rokeya Akter, Zinnat Sultana, and Syed Ishtiaque Ahmed.
2022. TolerationFactors: The Expectations of Decorum, Civility, and Certainty
on Rural Social Media. In Proceedings of the 2022 International Conference on
Information and Communication Technologies and Development . 1–14.
[92]Heng Sun and Wan Ni. 2022. Design and Application of an AI-Based Text
Content Moderation System. Scientific Programming 2022, 1 (2022), 2576535.
[93]James Thorne. 2022. Data-efficient autoregressive document retrieval for fact
verification. arXiv preprint arXiv:2211.09388 (2022).
[94]DimitrisTsirmpas, IonAndroutsopoulos, andJohnPavlopoulos.2025. Scalable
Evaluation of Online Moderation Strategies via Synthetic Simulations. arXiv
preprint arXiv:2503.16505 (2025).
[95]The World In Us. n.d.. Indigenous Peoples of Bangladesh — The World in Us
— theworldinus.org. https://www.theworldinus.org/blog/indigenous-peoples-
of-bangladesh . [Accessed 21-02-2025].
[96]Sahaj Vaidya, Jie Cai, Soumyadeep Basu, Azadeh Naderi, Donghee Yvette
Wohn, and Aritra Dasgupta. 2021. Conceptualizing visual analytic interven-
tions for content moderation. In 2021 IEEE Visualization Conference (VIS) . IEEE,
191–195.
[97]Ashley Marie Walker and Michael A DeVito. 2020. ”’More gay’fits in better”:
Intracommunity Power Dynamics and Harms in Online LGBTQ+ Spaces. In
Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems .
1–15.
[98]Michael Wiegand, Josef Ruppenhofer, and Elisabeth Eder. 2021. Implicitly abu-
sivelanguage–whatdoesitactuallylooklikeandwhyarewenotgettingthere?.
InProceedings of the 2021 Conference of the North American Chapter of the Asso-
ciation for Computational Linguistics: Human Language Technologies . 576–587.
[99]JacobOWobbrock.2012. SevenresearchcontributionsinHCI. Intelligence 174,
12-13 (2012), 910–950.
[100]Sijia Xiao, Shagun Jhaver, and Niloufar Salehi. 2023. Addressing interpersonal
harm in online gaming communities: The opportunities and challenges for a
restorative justice approach. ACM Transactions on Computer-Human Interac-
tion30, 6 (2023), 1–36.
[101]Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza
Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Rad-
harapu, et al. 2024. Shieldgemma: Generative ai content moderation based on
gemma. arXiv preprint arXiv:2407.21772 (2024).
[102]Alice Qian Zhang, Kaitlin Montague, and Shagun Jhaver. 2023. Cleaning up
the streets: Understanding motivations, mental models, and concerns of users
flagging social media posts. arXiv preprint arXiv:2309.06688 (2023).
[103]Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav
Artzi. 2019. Bertscore: Evaluating text generation with bert. arXiv preprint
arXiv:1904.09675 (2019).