# Artificial Intelligence Agents in Music Analysis: An Integrative Perspective Based on Two Use Cases

**Authors**: Antonio Manuel Martínez-Heredia, Dolores Godrid Rodríguez, Andrés Ortiz García

**Published**: 2025-11-17 23:46:47

**PDF URL**: [https://arxiv.org/pdf/2511.13987v1](https://arxiv.org/pdf/2511.13987v1)

## Abstract
This paper presents an integrative review and experimental validation of artificial intelligence (AI) agents applied to music analysis and education. We synthesize the historical evolution from rule-based models to contemporary approaches involving deep learning, multi-agent architectures, and retrieval-augmented generation (RAG) frameworks. The pedagogical implications are evaluated through a dual-case methodology: (1) the use of generative AI platforms in secondary education to foster analytical and creative skills; (2) the design of a multiagent system for symbolic music analysis, enabling modular, scalable, and explainable workflows.
  Experimental results demonstrate that AI agents effectively enhance musical pattern recognition, compositional parameterization, and educational feedback, outperforming traditional automated methods in terms of interpretability and adaptability. The findings highlight key challenges concerning transparency, cultural bias, and the definition of hybrid evaluation metrics, emphasizing the need for responsible deployment of AI in educational environments.
  This research contributes to a unified framework that bridges technical, pedagogical, and ethical considerations, offering evidence-based guidance for the design and application of intelligent agents in computational musicology and music education.

## Full Text


<!-- PDF content starts -->

Artificial Intelligence Agents in Music Analysis:
An Integrative Perspective Based on Two Use
Cases
Antonio Manuel Mart´ ınez Heredia∗1, Dolores Godrid Rodr´ ıguez2,
and Andr´ es Ortiz Garc´ ıa†1
1Dpto. Ingenier´ ıa de Comunicaciones, Universidad de M´ alaga,
Campus de Teatinos, 29071, M´ alaga, Spain
2Universidad Rey Juan Carlos
November 2025
Abstract
This paper presents an integrative review and experimental validation
of artificial intelligence (AI) agents applied to music analysis and educa-
tion. We synthesize the historical evolution from rule-based models to
contemporary approaches involving deep learning, multi-agent architec-
tures, and retrieval-augmented generation (RAG) frameworks. The ped-
agogical implications are evaluated through a dual-case methodology: (1)
the use of generative AI platforms in secondary education to foster analyt-
ical and creative skills; (2) the design of a multiagent system for symbolic
music analysis, enabling modular, scalable, and explainable workflows.
Experimental results demonstrate that AI agents effectively enhance
musical pattern recognition, compositional parameterization, and educa-
tional feedback, outperforming traditional automated methods in terms
of interpretability and adaptability. The findings highlight key challenges
concerning transparency, cultural bias, and the definition of hybrid eval-
uation metrics, emphasizing the need for responsible deployment of AI in
educational environments.
This research contributes to a unified framework that bridges techni-
cal, pedagogical, and ethical considerations, offering evidence-based guid-
ance for the design and application of intelligent agents in computational
musicology and music education.
Keywords: artificial intelligence, music analysis, education, deep learning,
multi-agent systems
∗ORCID: 0000-0002-8518-0981
†ORCID: 0000-0003-2690-1926
1arXiv:2511.13987v1  [cs.AI]  17 Nov 2025

1 Introduction
Despite considerable technological innovation, comprehensive reviews synthe-
sizing the application and evolution of artificial intelligence (AI) in the field of
music analysis remain scarce.
Although early studies on computer-assisted composition and rule-based
analysis established a foundation for the automated exploration of musical form
and content Hiller (1959), there is still a limited body of literature addressing
the complete progression from traditional algorithms to recent AI-driven models
and hybrid systems. Pioneering work such as Miranda’s Miranda (2021), un-
derscores the influence of AI, supercomputing, and evolutionary computation in
shaping the first computational tools for creation. Recent reviews (Wang et al.
(2024); Lerch et al. (2025)) focus on intelligent music generation systems. How-
ever, a systematic integration of these historical advances with state-of-the-art
AI methodologies and musical analysis is largely absent.
In the last decade, deep learning frameworks—including convolutional neural
networks, recurrent neural networks, and transformer architectures—have led to
breakthroughs in music information retrieval. Recent advances also encompass
interactive harmony tutoring (Ventura, 2022), structural music analysis (Min
et al., 2022), intelligent education systems (Han et al., 2023), and the use of
Large Language Model (LLM)-powered teachable agents in music pedagogy (Jin
et al., 2025). However, few studies provide an overarching review that brings
together these disparate lines of inquiry or highlights the shift from task-specific
solutions to holistic, explainable, and context-aware analysis.
More recently, retrieval-augmented generation (RAG) models have intro-
duced new standards of context awareness and interpretability by combining
generative modeling with adaptive information retrieval. However, the liter-
ature lacks syntheses that discuss the joint impact of RAG with earlier and
complementary AI-based approaches. A particularly groundbreaking develop-
ment is the emergence of AI agents: autonomous systems that analyze, generate,
and interact with music or learners using symbolic and audio data. The present
review seeks to address this gap by providing a unified synthesis of the evolution
of AI in music, from its historical roots to current advances in deep learning,
agent-based systems, and RAG methodologies. We aim to clarify methodologi-
cal innovations, highlight key research, and evaluate the implications for music
research, analysis, and education going forward. The principal themes are the
following:
•The evolution of AI methods in music analysis.
•The principles and applications of RAG methodologies.
•Recent progress in deep learning and agent-based systems.
•Implications for musicological research and pedagogy.
This review aims to inform researchers, educators, and practitioners of current
trends and emerging directions in computational musicology, thus filling a crit-
ical gap in the current academic landscape.
This paper is organized as follows: Section I (Introduction) outlines the his-
torical context and motivation to synthesize AI developments in music analysis.
2

Section II traces the evolution of AI methods. Section III explains the method-
ology followed in this study. Section IV highlights two cases: the first is an
exercise conducted with secondary students, and the second is an architecture
designed to perform analysis. This section also details case studies and experi-
mental results. Section V (Conclusions) summarizes key information and future
research directions.
The novelty of this review lies in the bridge between historical, technical,
and pedagogical aspects of AI in music analysis through dual empirical case
studies.
2 Related work
In the field of music analysis, the evaluation of different types of music poses a
significant challenge. One approach involves subjective evaluation, wherein in-
dividuals analyze music according to their own criteria. Another avenue involves
establishing metrics and applying them systematically.
Achieving the right balance between human feedback and quantitative indi-
cators is essential. Human feedback provides invaluable insight into expressive
and aesthetic aspects, often capturing nuances that computational methods may
overlook. In contrast, quantitative indicators, such as accuracy, precision, re-
call, or diversity metrics, ensure reproducibility and objectivity when comparing
systems. Integrating both perspectives enables comprehensive assessments and
supports the development of robust, user-oriented music analysis tools.
One of the main challenges in the objective evaluation of music analysis
systems lies in the absence of a single universally accepted criterion to evaluate
their performance.
This study focuses on Western music analysis to provide a more unified ap-
proach for studying, appreciating, and understanding music. The characteristics
of symbolic generation, structure, artistic expression, and aesthetics must there-
fore be examined through automatic methods. The computational evaluation
of these characteristics has been revolutionized by recent advances in artificial
intelligence.
Deep learning architectures now represent the main generation methods for
music composition systems. Enhancing the interpretability and controllability
of deep networks is one of the future research directions for music generation
technology Wang et al. (2024), which is particularly relevant given the need for
objective but musically meaningful evaluation metrics discussed earlier.
Multiple studies provide converging evidence that language and music share
a cognitive system for structural integration, although the exact nature of this
shared system remains complex and nuanced. Experimental evidence includes
(Fedorenko et al., 2009), who found an interaction between linguistic and musical
complexity in which comprehension accuracy changed when linguistic structures
were paired with musically unexpected elements.
2.1 The evolution of AI methods in music analysis
Early computational approaches to music analysis relied heavily on rule-based
systems and expert heuristics, notably exemplified in Hiller and Isaacson’s
3

computer-generated music experiments (Hiller, 1959) and the generative the-
ory of tonal music (Lerdahl and Jackendoff, 1996).
With the advent of the Music Information Retrieval (MIR) field—a mul-
tidisciplinary research area focused on developing innovative methods to ana-
lyze, search, and organize digital music data (Burgoyne et al., 2015)—the focus
shifted to designing features and algorithms—ranging from k-nearest neighbors
to support vector machines—that could automatically classify, segment, or de-
scribe large corpora of music recordings and symbolic data.
The subsequent explosion of large-scale annotated datasets—such as Mu-
sical Instrument Digital Interface (MIDI)-based datasets, the MIDI and Au-
dio Edited for Synchronous TRacks and Organization (MAESTRO) (Zhang,
2025), the Guitar-Aligned Performance Scores (GASP) (Riley et al., 2024), and
MusicNet—and the increasing power of neural architectures ultimately enabled
more flexible and data-driven models. Deep learning paradigms, by capturing
complex temporal and harmonic structures in audio and scores, have largely su-
perseded earlier approaches, although concerns around explainability and bias
persist. Hybrid models that combine symbolic reasoning with deep neural sys-
tems are also an emerging trend, with the aim of leveraging the strengths of
both paradigms.
A final, cutting-edge trend involves the direct application of Natural Lan-
guage Processing (NLP) models to symbolic music. This paradigm, which treats
musical scores as a sequential language, leverages successful architectures in text
analysis to understand musical syntax and semantics. Mel2Word is an example
of this NLP-based approach, which transforms melodies into text-like represen-
tations using Byte Pair Encoding (BPE) (Park et al., 2024).
2.2 The principles and applications of RAG methodolo-
gies
RAG models represent a paradigm shift in AI, combining generative neural
models with external information retrieval modules to overcome the inherent
limitations of purely parametric models. Unlike traditional approaches that rely
exclusively on knowledge encoded during training, RAG architectures dynam-
ically retrieve relevant information from external knowledge bases at inference
time, enabling systems to access up-to-date information, reduce hallucinations,
and provide verifiable sources for their outputs(Gao et al., 2023).
In the music domain, RAG enables systems to ground generation in large
corpora of musicology texts, symbolic scores, or audio datasets, thus enhanc-
ing both factuality and interpretability. The retrieval component typically em-
ploys dense vector representations, obtained through encoders such as BERT or
domain-specific music embeddings, to identify relevant passages from indexed
collections. The generative component then conditions its output on both the
user query and the retrieved context, allowing for more informed and contextu-
ally appropriate responses.
The applications of RAG in music span multiple modalities and tasks. Context-
aware automatic music annotation systems leverage RAG to provide rich reference-
based descriptions of musical works, drawing from historical analyses and the-
oretical literature. Personalized analysis reports can be generated by retrieving
relevant excerpts from pedagogical materials tailored to the user’s level and in-
terests. In an ”explainable” generative composition, generation is coupled with
4

textual or symbolic references justifying the output, making the creative pro-
cess more transparent and pedagogically valuable. RAG frameworks are also
being developed to adapt general-purpose LLMs for text-only music question
answering (MQA) tasks, as demonstrated by MusT-RAG (Kwon et al., 2025),
which employs a specialized retrieval mechanism over music-specific knowledge
bases.
Beyond these applications, RAG methodologies facilitate multimodal music
understanding by bridging symbolic, audio, and textual representations. For
instance, a system might retrieve relevant score passages when analyzing an
audio recording or access theoretical explanations when generating harmonic
progressions. This approach bridges the gap between black-box generation and
musicological transparency, offering a path toward AI systems that can engage
in informed musical discourse. As such, RAG is poised to redefine interactive
systems in music analysis, composition pedagogy, and computational musicol-
ogy, enabling tools that are powerful and accountable to established musical
knowledge.
2.3 Recent progress in deep learning and agent-based sys-
tems
Recent advances in deep learning and agent-based methodologies have led to the
development of intelligent music analysis and composition tools. In particular,
WeaveMuse offers an open source multi-agent framework that supports iterative
analysis, synthesis, and rendering processes across diverse modalities, including
text, symbolic notation, and audio (Karystinaios, 2025). Similarly, MusicAgent
utilizes powered workflows powered by large language model to orchestrate a
wide array of music-related tools, allowing the automatic decomposition of com-
plex user requests into manageable subtasks (Yu et al., 2023). These innovations
illustrate the growing capacity of AI-driven agents to handle multifaceted musi-
cal information and automate tasks that previously required substantial human
expertise.
The musical agents MACAT and MACataRT(Lee and Pasquier, 2025), pio-
neering a novel framework for creative collaboration in interactive musicmaking.
These real-time generative AI systems use corpus-based synthesis and small-data
training to function as responsive, artist-in-the-loop co-creators, prioritizing the
preservation of expressive nuances.
2.4 Implications for Musicological Research and Pedagogy
Zhang investigates various methodologies for the integration of AI in musicologi-
cal research, with particular emphasis on the theoretical and practical challenges
of adapting structural analysis to this context Zhang (2025). AI techniques can
facilitate knowledge extraction processes that support musicologists and educa-
tors in interpreting, organizing, and teaching complex musical phenomena.
The integration of AI-driven analytical tools into musicological research in-
troduces both methodological opportunities and epistemological questions. On
the one hand, computational methods enable large-scale corpus studies that
would be impractical through traditional manual analysis, revealing patterns
and stylistic trends across extensive repertoires. Machine learning models can
5

identify subtle relationships between musical parameters, detect influences be-
tween composers and periods, and propose novel analytical perspectives that
complement human expertise. However, these approaches raise fundamental
questions about the nature of musical understanding: to what extent can com-
putational pattern recognition capture the cultural, historical, and esthetic di-
mensions that musicologists consider essential?
In pedagogical contexts, AI systems offer transformative potential for music
education at multiple levels. Intelligent tutoring systems can provide person-
alized feedback on compositional exercises, adapting their guidance to individ-
ual student needs and learning trajectories. Interactive analytical tools can
help students visualize complex theoretical concepts, such as harmonic func-
tion, motivic development, or formal structure, through dynamic, multimodal
representations. Furthermore, AI-generated examples and counterexamples can
illustrate stylistic principles or theoretical rules in ways that enrich traditional
pedagogical materials. However, the deployment of such systems requires careful
consideration of pedagogical philosophy: educators must balance the efficiency
of automated feedback with the irreplaceable value of human mentorship and
critical dialog.
The explainability challenge represents a crucial concern for both research
and pedagogy. Black-box models that produce accurate analytical results with-
out transparent reasoning processes risk creating a disconnect between compu-
tational output and musical understanding. This is where RAG and other in-
terpretable AI methodologies become particularly valuable, as they can provide
traceable connections between analytical conclusions and established musicolog-
ical knowledge. For educators, this transparency is essential: students must not
only receive correct analytical interpretations but also understand the reasoning
pathways that lead to them.
In addition, the use of AI in musicology requires critical reflection on the
issues of bias, representation, and canon formation. Training datasets predom-
inantly featuring Western art music can perpetuate existing biases and limit
the applicability of AI tools to diverse musical traditions. Musicologists and
educators have a responsibility to advocate for more inclusive datasets and to
critically examine the assumptions embedded in computational models. This
critical engagement can itself become a valuable pedagogical opportunity, teach-
ing students to question technological solutions and to recognize the cultural
contingency of analytical frameworks.
In the future, the successful integration of AI into musicological research
and pedagogy will require ongoing collaboration between technologists, musi-
cologists, and educators. This interdisciplinary dialog should address not only
the technical capabilities but also the ethical, epistemological, and practical di-
mensions of AI adoption. By approaching these tools with both enthusiasm
for their potential and critical awareness of their limitations, the musicological
community can harness AI to enhance—rather than replace—the rich traditions
of musical scholarship and teaching.
3 Methodology
This study employs a dual-case methodology to examine AI applications in mu-
sic analysis from complementary pedagogical and technical perspectives. The
6

first case explores the integration of generative AI tools in secondary music edu-
cation, while the second presents a technical framework for automated symbolic
music analysis through multi-agent systems.
3.1 Case Study 1: Generative AI in Secondary Education
The first case investigates the pedagogical application of generative AI tools in
developing structural analysis skills of music among secondary school students.
The methodology comprises three interconnected phases designed to bridge an-
alytical understanding and creative practice.
3.1.1 Participants and Context
A total of 200 secondary students (ages 12–16) participated in a series of struc-
tured workshop sessions. The activity was organized in consecutive 1-hour
blocks, with approximately 30 students per group. During a 6-hour event, all
participants engaged in the analysis of contemporary popular songs, focusing
on formal structure, harmony, rhythm, and lyrics.
3.1.2 Phase 1: Structural Analysis
Students engaged in systematic analysis of contemporary popular music to de-
velop foundational skills in formal recognition:
•Participants:Secondary education students.
•Musical corpus:Contemporary popular music songs selected for their
clear formal structures.
•Analytical framework:Students examined and identified structural ele-
ments including verse-chorus patterns, bridges, pre-choruses, instrumental
sections, and overall form (e.g., ABABCB).
•Learning objectives:Development of analytical skills to recognize for-
mal patterns, harmonic progressions, melodic characteristics, and lyrical
themes.
3.1.3 Phase 2: Creative Parameter Definition
Building upon analytical insights, students collaboratively designed specifica-
tions for original compositions:
•Collaborative design:Students worked in groups to define parameters
for an original composition based on their structural analysis.
•AI tool used:ChatGPT1as an interactive assistant to refine composi-
tional specifications.
•Definition ofStructural form, lyrical themes, mood/atmosphere, tempo
indications, instrumentation preferences, and stylistic characteristics.
1https://chat.openai.com
7

•Pedagogical focus:Translation of analytical understanding into creative
specifications.
In each session, students collaboratively used ChatGPT to generate lyrics and
refine compositional parameters during a 30-minute segment.
3.1.4 Phase 3: AI-Assisted Generation and Evaluation
The final phase involved iterative generation and critical assessment of AI-
produced musical outputs:
•Generation tools:Suno2and Music.ai3platforms used to generate mu-
sical outputs based on student-defined parameters. Each generated lyric
served as input for both platforms, and each platform produced two audio
outputs, for a total of four pieces per group.
•Iterative process:Students refined prompts and parameters based on
initial results.
•Critical evaluation:Comparative analysis between the intended speci-
fications and AI-generated outputs.
•Assessment criteria:Fidelity to structural parameters, musical coher-
ence, creative quality, and alignment with student expectations.
3.1.5 Data Collection and Evaluation Procedures
The evaluation procedure combined multiple data sources to assess both learning
outcomes and AI tool effectiveness:
Objective metrics:
•Melodic similarity (Dynamic Time Warping – DTW).
•Harmonic coherence.
•Rhythmic diversity (entropy).
Subjective ratings:
•All students participated in blind listening sessions.
•Scoring each output on 5-point Likert scales across five qualitative dimen-
sions:
–Expressiveness.
–Stylistic Accuracy.
–Harmonic Coherence.
–Rhythmic Diversity.
–Overall Appeal.
2https://suno.ai
3https://music.ai
8

Qualitative data:
•Analytical documentation and student parameter specifications.
•Generated musical outputs and prompt iteration histories.
•Reflective assessments comparing analytical intentions with generative re-
sults.
•Evaluation of pedagogical effectiveness in developing analytical skills, crit-
ical thinking about AI capabilities and limitations, and understanding the
relationship between analytical and creative processes.
3.2 Case Study 2: Agent-Based Architecture for Symbolic
Analysis
The second case presents a technical framework for automated music analysis
through a specialized multi-agent system. This computational approach ad-
dresses the complexity of music analysis by distributing analytical tasks among
specialized agents.
3.2.1 Rationale for Multi-Agent Approach
Automated symbolic music analysis poses a significant challenge due to the in-
herently multidimensional and structural nature of music, which encompasses
harmonic, melodic, rhythmic, and formal aspects. Employing a multi-agent sys-
tem architecture provides an effective solution by enabling specialized modules
to handle specific analytical tasks, thereby enhancing the accuracy and inter-
pretive depth of the system. The choice of a multi-agent approach is grounded
in several key advantages:
•Specialization and Modularity:Dividing the analysis into specialized
agents allows independent handling of each musical dimension (harmonic,
melodic, rhythmic, formal), facilitating continuous improvement of each
component without affecting the overall system (Wooldridge, 2009).
•Coordination and Synthesis:An integration layer enables the coher-
ent combination of partial results from different agents, producing com-
prehensive analytical reports that reflect the interaction of diverse musical
elements. This mimics expert human interpretative processes (Jennings,
2000).
•Scalability and Flexibility:Implementing this architecture on plat-
forms such as LangGraph or n8n leverages modern agent orchestration
tools that support distributed, adaptive workflows, allowing the system to
scale and adapt to varying analytical complexities and workloads (Jensen
and Chase, 2023).
•Validation and Robustness:Incorporating agents dedicated to com-
paring outputs against expert musicological analyzes ensures the system’s
reliability and establishes trustworthiness for academic and professional
applications (Conklin, 2016).
9

This multi-agent framework advances computational music analysis by ap-
proximating sophisticated interpretive models through collaborative, special-
ized agents, supported by flexible and automated architectures that facilitate
advanced artificial intelligence techniques.
3.2.2 System Architecture and Components
System architecture:Multi-agent framework with modular design that en-
ables parallel processing and specialized analytical functions.
Input processing:Symbolic music representations in standard formats (MIDI,
MusicXML, kern, or similar structured formats).
Agent specialization:Dedicated agents are defined for distinct analytical
dimensions: structural analysis (formal segmentation and architectural out-
line), stylistic analysis (historical period attribution, instrumentation, ornamen-
tation), and harmonic analysis (chord identification, harmonic function, tonal
centers). This approach facilitates modular evaluation and coherent synthesis in
music analysis, enabling the system to approximate expert-level interpretation
through the collaboration of specialized autonomous modules.
Integration layer:Coordination mechanism enabling inter-agent communi-
cation, conflict resolution, and synthesis of partial analyzes into coherent inter-
pretations.
Output generation:Comprehensive analytical reports integrate insights from
multiple specialized agents, presented in human-readable and machine-processable
formats.
3.2.3 Workflow Stages
The multi-agent system for symbolic music analysis, creative recomposition, and
evaluation is structured in the following stages:
1.Input Processing Agent:Transcribes and segments symbolic or audio
input to extract musical phrases.
2.Analysis Agent:Identifies harmonic, rhythmic, and formal patterns,
and generates an annotated report in MusicXML.
3.Generation Agent:Uses extracted features to generate new material
using style transfer models, imposing compositional constraints.
4.Evaluation Agent:Assesses compliance with musical rules and aggre-
gates expert and user feedback.
3.2.4 Agent-Based Analysis: Structure, Principal Harmony, and Style
The implemented multi-agent system performs musical analysis in three au-
tonomous stages, closely simulating expert human procedures.
10

1.Structural Agent:Segments the piece, detecting main sections based on
patterns of textural change and repetition. The agent produces a global
architectural outline, labeling formal segments such as introduction, expo-
sition, development, reprise, or coda according to widely accepted models.
2.Principal Harmony Agent:Extracts the dominant key and recognizes
modulations via chord classification algorithms and tonal trajectory analy-
sis. This module identifies primary harmonic progressions (e.g., I-IV-V-I),
points of modulation, and signature chords such as sevenths or secondary
dominants, producing a detailed map of harmonic flows across the work.
3.Stylistic Agent:Compares the obtained structural and harmonic pat-
terns against a curated database of tagged historical examples. It com-
putes likelihoods for period attribution (Baroque, Classical, Romantic,
etc.) and augments its prediction with meta-information regarding instru-
mentation and ornamentation. The agent’s inference yields a probabilistic
profile of the work’s stylistic features.
Through this procedure, the system delivers an automated synthesis of struc-
ture, principal harmony, and stylistic characteristics, mirroring expert analytical
workflow and enabling direct quantitative comparison between human and AI-
generated analyses.
3.2.5 Dataset: Representative 18th-Century Musical Repertoire: A
Curated Dataset Overview
The following dataset of fifty representative musical works from the 18th century
has been compiled to support comparative analytical studies in style, form, and
historical evolution. It includes late Baroque idioms, the galant transition,
empfindsamer elements, and the full emergence of Classical structures. The
selection is intentionally diverse, covering instrumental, vocal, sacred, secular,
chamber, orchestral, and operatic genres.
The corpus begins withJohann Sebastian Bach, whoseWell-Tempered
Clavier(Books I and II),Brandenburg Concertos,St. Matthew Passion,Mass
in B minor,The Art of Fugue,Musical Offering, and solo suites and partitas
exemplify the contrapuntal and rhetorical density of the late Baroque. Together,
these works provide a foundation for examining polyphonic technique, motivic
transformation, and pre-Classical harmonic practice.
Works byGeorge Frideric Handel, includingMessiah,Water Music,Mu-
sic for the Royal Fireworks, and operas such asRinaldoandGiulio Cesare,
highlight the synthesis of Italian, French, and English traditions, offering ma-
terial for studies in orchestration, dramatic construction, and large-scale vocal
form.
In contrast,Domenico Scarlatti’skeyboard sonatas (K.1–555), hisStabat
Mater, andMissa quatuor vocumillustrate the galant aesthetic, characterized by
clear phrase structures and idiomatic keyboard writing that anticipate Classical
norms.
The empfindsamer Stil is represented throughC. P. E. Bach’skeyboard
concertos, symphonies (Wq 182), and early sonatas, embodying heightened emo-
tional expressivity, irregular phrasing, and an evolving relationship between
11

soloist and ensemble. These works serve as a bridge between Baroque complex-
ity and Classical clarity.
A significant portion of the dataset focuses onJoseph Haydn, whose sym-
phonies (e.g., No. 94 “Surprise,” No. 104 “London”), quartets (Op. 33 No.
2 “The Joke”), oratorios (The Creation,The Seasons), the Trumpet Concerto,
and the “Gipsy” Trio exemplify mature Classical form. Haydn’s oeuvre provides
rich material for studying sonata form, monothematicism, motivic economy, and
the standardization of the symphony and quartet.
The Classical vocabulary reaches its height inWolfgang Amadeus Mozart,
represented here by major symphonies (No. 40 and No. 41 “Jupiter”), piano
concertos (Nos. 20 and 21), operas (The Marriage of Figaro,Don Giovanni,
The Magic Flute), theRequiem, the Clarinet Quintet, and the wind serenade
Gran Partita. These works are central to inquiries into thematic integration,
formal balance, instrumental color, and operatic dramaturgy.
Additional diversity is provided by figures such asLuigi Boccherini,Gluck,
Pergolesi,Stamitz,Paisiello,Rameau, andSalieri, highlighting regional
variation and the stylistic plurality of 18th-century Europe.
Table 1 summarizes the selected composers, their dates, stylistic affiliations,
and representative works. This dataset offers a broad yet coherent panorama of
the century’s musical developments, supporting both quantitative corpus anal-
ysis and detailed qualitative examinations of form, harmony, texture, and or-
chestration.
3.3 Corpus of Analysis
A representative corpus comprising fifty musical works from the eighteenth cen-
tury was utilized, spanning various instrumental and vocal genres primarily from
the Western academic tradition. The selected repertoire includes compositions
by J. S. Bach, Handel, Scarlatti, C. P. E. Bach, Haydn, Mozart, Boccherini,
Gluck, Pergolesi, Stamitz, Paisiello, Rameau, and Salieri. The works reflect
both stylistic and formal diversity, thus allowing the scrutiny of structural, har-
monic, and stylistic parameters in varied contexts.
3.4 Additional Quantitative Evaluation Metrics
In addition to the standard metrics previously described, the evaluation incor-
porated timbral similarity analysis using spectral descriptors, motif complexity
assessment based on automated counting of recurrent motives, and statistical
formal variability analysis utilizing Shannon diversity indices. Each work was
evaluated using pattern-matching algorithms, coherence and harmonic richness
metrics, and measures for rhythmic diversity and formal complexity, thus com-
plementing conventional DTW and Music21 approaches.
3.5 Expert Validation Procedures
A panel of six musicologists and domain-specialist educators performed blind
validations to contrast multi-agent system reports against their own manual
analyzes. The validation criteria used a five-point Likert scale that addressed
accuracy, interpretive depth, and stylistic appropriateness. Qualitative feed-
back was gathered with respect to utility, observable bias, and potential im-
12

Composer Dates Style Representative Works
J. S. Bach 1685–1750 Late Baroque Well-Tempered Clavier, Bran-
denburg Concertos, St. Matthew
Passion, Mass in B minor, The
Art of Fugue, Musical Offering,
Solo Suites and Partitas
Handel 1685–1759 Late Baroque Messiah, Water Music, Fire-
works Music, Rinaldo, Giulio Ce-
sare
D. Scarlatti 1685–1757 Galant
BaroqueKeyboard Sonatas (K.1–555),
Stabat Mater, Missa quatuor
vocum
C. P. E. Bach 1714–1788 Empfindsamer
StilKeyboard Concertos, Sym-
phonies (Wq 182), Prussian
Sonatas
Haydn 1732–1809 Classical Symphony No. 94 “Surprise”,
Symphony No. 104 “London”,
Quartet Op.33 No.2 “The Joke”,
The Creation, Trumpet Con-
certo
Mozart 1756–1791 Classical Symphonies No. 40 & 41
“Jupiter”, Piano Concertos No.
20, 21, The Marriage of Figaro,
Don Giovanni, Magic Flute, Re-
quiem, Clarinet Quintet
Boccherini 1743–1805 Classical Quintet “Fandango” G.448,
Cello Concerto G.482, Sym-
phony “La casa del diavolo”
Gluck 1714–1787 Opera Reform Orfeo ed Euridice, Alceste,
Iphig´ enie en Tauride
Pergolesi 1710–1736 Galant Stabat Mater, La serva padrona
Stamitz 1717–1757 Mannheim
SchoolSymphonies, orchestral dynamic
innovations
Paisiello 1740–1816 Opera Buffa Il barbiere di Siviglia
Rameau 1683–1764 French
BaroqueLes Indes galantes, harpsichord
pieces, trag´ edies lyriques
Salieri 1750–1825 Classical Armida, various operatic and sa-
cred works
Table 1: Representative 18th-Century Composers and Selected Works
provements with respect to interpretive autonomy and correction of systematic
errors.
13

3.6 Comparative Analysis: System Results vs. Reference
Human Analyses
Discrepancies and overlaps in the detection of formal sections, harmonic pro-
gressions, and stylistic attributions were systematically analyzed. The compar-
ative results were summarized by tabulating the percentage of match in formal
segmentation, agreement in tonal identification, and expert assessments on the
capacity of the system to approximate human evaluation standards. The dis-
cussion integrates factors such as the explainability of the system, adaptation to
diverse repertoires, and identification of current limitations in AI-based music
analysis.
4 Case Studies and Experimental Results
To illustrate practical applications and evaluative challenges of AI-driven mu-
sic analysis, we present two complementary case studies: the first examines
generative AI integration in secondary education; the second demonstrates a
multi-agent system for symbolic analysis.
4.1 Pop Music Analysis and AI Generation in Secondary
Education
Context
In each session, the students collaboratively used ChatGPT to generate lyrics
and refine compositional parameters during a 30-minute segment. Each gener-
ated lyric served as input for both Suno4and Music.AI5platforms, and each
platform produced two audio outputs, for a total of four pieces per group.
The evaluation procedure combined:
•Objective metrics:melodic similarity (DTW), harmonic coherence, and
rhythmic diversity (entropy), calculated using the Music21 toolkit across
all generated pieces.
•Subjective ratings:all students participated in blind listening sessions,
scoring each output on 5-point Likert scales across five qualitative dimen-
sions.
4.1.1 Results
Table 2 summarizes the evaluation result for the four generated pieces (two per
platform, mean±SD,∗p <0.05,∗∗p <0.01, two-tailed t-test,n= 200 student
evaluators):
Objective metrics indicated that Suno achieved closer melodic adherence
(DTW: 0.34 vs. 0.48) and greater harmonic coherence (8.2/10 vs. 7.4/10),
while Music.AI exhibited greater rhythmic variety (entropy: 3.8 vs. 3.2 bits).
Key student insights:
•85% reported improved analytical articulation skills.
4https://suno.ai
5https://music.ai
14

Table 2: Comparative Evaluation: Suno vs. Music.AI
Criterion Suno Music.AI p-value
Expressiveness 4.0±0.6 4.3±0.5 0.042∗
Stylistic Accuracy 4.4±0.4 3.8±0.7 0.001∗∗
Harmonic Coherence 4.5±0.5 4.0±0.6 0.003∗∗
Rhythmic Diversity 3.9±0.7 4.2±0.6 0.089
Overall Appeal 4.1±0.6 4.4±0.5 0.028∗
•90% recognized the importance of precise parameter specification.
•75% acknowledged the gap between analytical and creative processes.
Finally, teachers highlighted the usefulness of tools to support personalized
learning and promote student autonomy in the classroom.
4.1.2 Discussion
The larger, diversified sample (n= 200) supports stronger generalization of
results. The findings confirm a trade-off between fidelity and creative variabil-
ity: Suno prioritized stylistic and structural coherence, while Music.AI facili-
tated wider exploratory approaches. The broader evaluation suggests that stu-
dent analytical frameworks and background strongly influenced their subjective
judgments. The iterative process fostered metacognitive skills but also revealed
pedagogical challenges such as over-reliance on AI and potential reinforcement
of dataset biases.
4.2 Agent-Based Workflow for Symbolic Music Analysis
Expanding on Section 3.2.4, we present a multi-agent framework for symbolic
analysis, designed to replicate expert practices through modular, specialized
agents.
The multi-agent system described above is deployed in the evaluation of the
18th-century corpus. Each work is processed by three autonomous agents—
•Structural Agent: Segments the piece and detects main sections based on
textural change and repetition, generating a global architectural outline.
•Harmonic Agent: Extracts the dominant key and recognizes modulations,
mapping harmonic flows and significant progressions.
•Stylistic Agent: Attributes historical period, instrumentation, and or-
namentation likelihoods by comparing patterns with a curated reference
database.
The results table (Table 3) presents the outcomes of these analyses across the
entire repertoire. The analytical modules demonstrate high overall consistency,
simulating expert human workflow and enabling direct quantitative comparison.
While most outputs are coherent, a minority of cases reveal “hallucinations”
(inaccuracies or over-interpretations) that remain generally congruent with the
logic defined in agent specialization.
15

Figure 1: Music Analysis Agent System
The proposed modular architecture for symbolic music analysis is illustrated
in Figure 1. At the core of the system, a centralCoordinatorAgentorches-
trates the workflow by distributing the musical score to three specialized agents,
each responsible for a distinct facet of analysis:
TheStructuralAgentperforms formal segmentation and identifies global
architectural outlines of the piece.
TheStylisticAgentevaluates historical period, instrumentation, and orna-
mentation, associating the work with stylistic trends.
TheHarmonicAgentextracts chordal progressions, recognizes harmonic
functions, and identifies tonal centers.
The specialized outputs are then integrated by the coordinator, enabling a
comprehensive and interpretable multi-dimensional analysis. This architecture’s
modular separation and well-defined communication pathways facilitate scala-
bility and are compatible with orchestrators such as LangGraph, supporting
reproducible and extensible automated music analysis workflows.
4.2.1 Results
The results obtained from the structural, stylistic, and harmonic analysis agents
are generally consistent across the examined musical works. In some cases,
agents produced hallucinations—analytical inferences not entirely supported by
the musical data. These instances, however, remain aligned with the designed
agent logic and do not undermine the overall agreement among the agents.
Table 3 presents a comparative summary demonstrating high structural and
harmonic congruence, with noted hallucinations flagged in grey.
16

Table 3: Comparative multi-agent analysis for the dataset: consis-
tency and hallucination notes for each work.
Composer Work Structural Anal-
ysisStylistic Analy-
sisHarmonic Anal-
ysisHallucination Note
J. S. Bach Well-Tempered Clavier Consistent Consistent Consistent -
J. S. Bach Brandenburg Concertos Consistent Consistent Minor error Harmony mislabel
J. S. Bach St. Matthew Passion Consistent Consistent Consistent -
J. S. Bach Mass in B minor Minor error Consistent Consistent Structure split missed
J. S. Bach The Art of Fugue Consistent Consistent Consistent -
J. S. Bach Musical Offering Consistent Consistent Consistent -
J. S. Bach Solo Suites and Partitas Minor error Consistent Consistent Formal segmentation
Handel Messiah Consistent Consistent Hallucination Tonal ambiguity
Handel Water Music Consistent Hallucination Consistent Mixed stylistic attribu-
tion
Handel Fireworks Music Consistent Consistent Consistent -
Handel Rinaldo Hallucination Consistent Consistent Period label ambiguity
Handel Giulio Cesare Consistent Hallucination Minor error Stylistic attribution
D. Scarlatti Keyboard Sonatas Consistent Consistent Consistent -
D. Scarlatti Stabat Mater Consistent Consistent Consistent -
D. Scarlatti Missa quatuor vocum Consistent Consistent Minor error Harmony mislabel
C. P. E. Bach Keyboard Concertos Consistent Consistent Minor error Emotional nuance
missed
C. P. E. Bach Symphonies (Wq 182) Minor error Consistent Consistent Structure split missed
C. P. E. Bach Prussian Sonatas Consistent Consistent Consistent -
Continued on next page
17

Table 3 – continued from previous page
Composer Work Structural Anal-
ysisStylistic Analy-
sisHarmonic Anal-
ysisHallucination Note
Haydn Symphony No. 94 “Sur-
prise”Consistent Consistent Minor error Motif missed
Haydn Symphony No. 104 “Lon-
don”Consistent Consistent Consistent -
Haydn Quartet Op.33 No.2 “The
Joke”Consistent Consistent Hallucination Modulation misde-
tected
Haydn The Creation Consistent Consistent Consistent -
Haydn Trumpet Concerto Consistent Consistent Consistent -
Mozart Symphony No. 40 Consistent Consistent Consistent -
Mozart Symphony No. 41
“Jupiter”Consistent Consistent Minor error Formal repetition mis-
labeled
Mozart Piano Concertos No. 20,
21Consistent Hallucination Consistent Stylistic ambiguity
Mozart Marriage of Figaro Consistent Consistent Hallucination Tonal center error
Mozart Don Giovanni Consistent Minor error Consistent Famous aria misseg-
mented
Mozart Magic Flute Consistent Consistent Consistent -
Mozart Requiem Consistent Hallucination Consistent Style attribution un-
certainty
Mozart Clarinet Quintet Minor error Consistent Consistent Emotional phrasing
missed
Boccherini Quintet ”Fandango”
G.448Consistent Consistent Minor error Harmony mislabel
Continued on next page
18

Table 3 – continued from previous page
Composer Work Structural Anal-
ysisStylistic Analy-
sisHarmonic Anal-
ysisHallucination Note
Boccherini Cello Concerto G.482 Consistent Consistent Consistent -
Boccherini Symphony ”La casa del di-
avolo”Consistent Consistent Hallucination Stylistic period confu-
sion
Gluck Orfeo ed Euridice Consistent Hallucination Consistent Style conflict (Opera
Reform)
Gluck Alceste Consistent Consistent Consistent -
Gluck Iphig´ enie en Tauride Hallucination Consistent Consistent Structure divergence
Pergolesi Stabat Mater Consistent Hallucination Consistent Style ambiguity
(Galant)
Pergolesi La serva padrona Consistent Consistent Consistent -
Stamitz Symphonies Consistent Minor error Consistent Dynamic motif missed
Stamitz Orchestral innovations Consistent Consistent Consistent -
Paisiello Il barbiere di Siviglia Minor error Consistent Hallucination Style/period conflict
Rameau Les Indes galantes Consistent Consistent Minor error Harmony ambiguity
Rameau Harpsichord pieces Consistent Consistent Consistent -
Rameau Trag´ edies lyriques Consistent Hallucination Consistent Period mix-up
Salieri Armida Hallucination Consistent Consistent Formal segmentation
error
Salieri Other operatic/sacred
worksConsistent Minor error Consistent Style attribution sub-
tlety
19

The agent-based analytical framework was applied to the full 18th-century
corpus, with each piece independently examined by the Structural, Stylistic,
and Harmonic agents. Across the dataset, the overall agreement between the
three agents was high: most works exhibit consistent analytic outputs, espe-
cially concerning major structural boundaries, characteristic stylistic markers,
and principal harmonic progressions. Despite this overall consistency, occasional
discrepancies—labelled as “hallucinations” in Table 3—arose. These typically
reflected either over-segmentation by the Structural agent in ambiguous pas-
sages, stylistic misattribution in transitional works, or harmonic mislabelling in
pieces with complex modulations. Importantly, such isolated inconsistencies did
not undermine the analytical coherence of the multi-agent system, but rather
revealed well-delimited cases where automated inference remains challenging
and human expert supervision would be advisable. The summary of results and
the precise nature of all observed “hallucinations” are provided for each work
in Table 3 on the following pages.
4.2.2 Discussion
The high overall consistency in agent output highlights the robustness of the
modular multi-agent design described in Section 3.2.2. However, specific lim-
itations emerge, particularly in works that exhibit transitional stylistic char-
acteristics or atypical harmonic structures. These “hallucinations” —while
few—illuminate the boundaries of current automated analysis and underscore
the continuing importance of expert musicological interpretation. The modular
structure enables granular diagnostic review and targeted refinement, setting a
foundation for future extensions (e.g., new analytical agents, multimodal inte-
gration). Overall, the system promises substantial efficiency gains and supports
explainable and reproducible music analysis, but optimum results will depend
on a synergetic integration of AI and human expertise.
5 Conclusions
The study highlights the interaction of analytical accuracy, creative potential,
and educational value in AI-driven music analysis.
This paper presents a comprehensive overview of contemporary develop-
ments in music analysis, detailing algorithmic methodologies and the various
frameworks for musical representation and evaluation. By elucidating the dis-
tinguishing characteristics and analytical methods within Western musical tra-
dition, the study brings to light the unique opportunities and inherent challenges
that face automated music analysis. The findings point to promising avenues
for future research, with implications for enriching scholarly discourse and en-
hancing listeners’ understanding of music.
Both cases provide complementary perspectives: the first examines human-
AI interaction in educational contexts, while the second explores autonomous AI
capabilities in specialized analytical tasks. Together, they illustrate the current
state and potential trajectories of AI integration in musicological practice.
These results are consistent with the collective experience of the research
team in the design, evaluation, and critical review of AI-driven music analy-
sis systems. The methodology and limitations discussed here reflect both the
20

empirical findings of this study and our ongoing engagement with advanced com-
putational tools in musicology. Future research will continue to integrate human
expertise and automated frameworks to address the complexities encountered
in polyphonic and stylistically ambiguous repertoires.
References
Burgoyne, J. A., Fujinaga, I., and Downie, J. S. (2015).Music Information
Retrieval, chapter 23, pages 213–228. Wiley-Blackwell.
Conklin, D. (2016). Music generation from statistical models.Proceedings of
the AISB Symposium on AI and Music, pages 30–35.
Fedorenko, E., Patel, A., Casasanto, D., et al. (2009). Structural integration
in language and music: Evidence for a shared system.Memory & Cognition,
37(1):1–9.
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
H., and Wang, H. (2023). Retrieval-augmented generation for large language
models: A survey.arXiv preprint arXiv:2312.10997, 2(1).
Han, X., Chen, F., Ullah, I., and Faisal, M. (2023). An evaluation of ai-based
college music teaching using ahp and moora.Soft Computing, pages 1–11.
Hiller, L. A. (1959). Computer music.Scientific American, 201(6):109–121.
Jennings, N. R. (2000). On agent-based software engineering.Artificial Intelli-
gence, 117(2):277–296.
Jensen, E. and Chase, H. (2023). Langgraph: A framework for building stateful,
multi-actor applications with llms. InProceedings of the Workshop on Large
Language Models and Applications, pages 1–10. ACM.
Jin, L., Lin, B., Hong, M., Zhang, K., and So, H.-J. (2025). Exploring the
impact of an llm-powered teachable agent on learning gains and cognitive
load in music education.arXiv preprint arXiv:2504.00636.
Karystinaios, E. (2025). Weavemuse: An open agentic system for multimodal
music understanding and generation.arXiv preprint arXiv:2509.11183.
Kwon, D., Doh, S., and Nam, J. (2025). Must-rag: Musical text question answer-
ing with retrieval augmented generation.arXiv preprint arXiv:2507.23334.
Lee, K. J. M. and Pasquier, P. (2025). Musical agent systems: Macat and
macatart.arXiv preprint arXiv:2502.00023.
Lerch, A., Arthur, C., Bryan-Kinns, N., Ford, C., Sun, Q., and Vinay, A. (2025).
Survey on the evaluation of generative models in music.ACM Computing
Surveys.
Lerdahl, F. and Jackendoff, R. S. (1996).A Generative Theory of Tonal Music.
The MIT Press, Cambridge, MA, reissue, with a new preface edition.
21

Min, J., WANG, L., PANG, J., HAN, H., Li, D., ZHANG, M., and and, Y. H.
(2022). Application and research of monte carlo sampling algorithm in mu-
sic generation.KSII Transactions on Internet and Information Systems,
16(10):3355–3372.
Miranda, E. R. (2021).Handbook of artificial intelligence for music. Springer.
Park, S., Choi, E., Kim, J., and Nam, J. (2024). Mel2word: A text-based melody
representation for symbolic music analysis.Music & Science, 7. Original work
published 2024.
Riley, X., Guo, Z., Edwards, D., and Dixon, S. (2024). Gaps: A large and
diverse classical guitar dataset and benchmark transcription model.arXiv
preprint arXiv:2408.08653.
Ventura, M. D. (2022). A self-adaptive learning music composition algorithm
as virtual tutor. In Maglogiannis, I., Iliadis, L., Macintyre, J., and Cortez,
P., editors,Artificial Intelligence Applications and Innovations, pages 16–26,
Cham. Springer International Publishing.
Wang, L., Zhao, Z., Liu, H., Pang, J., Qin, Y., and Wu, Q. (2024). A review
of intelligent music generation systems.Neural Computing and Applications,
36(12):6381–6401.
Wooldridge, M. (2009).An Introduction to MultiAgent Systems. John Wiley &
Sons, Chichester, UK, 2nd edition.
Yu, D., Song, K., Lu, P., He, T., Tan, X., Ye, W., Zhang, S., and Bian, J. (2023).
Musicagent: An ai agent for music understanding and generation with large
language models.arXiv preprint arXiv:2310.11954.
Zhang, M. (2025). Advancing deep learning for expressive music composition
and performance modeling.Scientific Reports, 15(1):28007.
22