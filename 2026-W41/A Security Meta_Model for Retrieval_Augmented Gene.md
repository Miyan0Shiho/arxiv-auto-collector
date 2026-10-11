# A Security Meta-Model for Retrieval-Augmented Generation Systems

**Authors**: Steve Nouyep, Sébastien Salva, Maxime Puys

**Published**: 2026-10-08 13:03:42

**PDF URL**: [https://arxiv.org/pdf/2610.11893v1](https://arxiv.org/pdf/2610.11893v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems extend large language models (LLMs) with external knowledge through a multi-stage pipeline. While this architecture can improve the factual grounding of generated answers, it introduces structural attack surfaces that extend beyond those of standalone LLMs. In this paper, we introduce a security meta-model that captures explicit causal relationships between RAG surfaces, attacks, weaknesses, risks, and CIA impact (Confidentiality, Integrity, Availability). Its purpose is to provide security engineers with a structured and user-friendly framework for gathering and assessing the risks, weaknesses, and mitigations relevant to their RAG deployment. We designed the meta-model through an iterative, structured analysis of 43~publications (2023--2026) and instantiated it as a catalog populated with the security threats and remediations reported in the literature. Filtering the catalog according to a deployment configuration produces a risk profile containing the risks applicable to that deployment. An interactive web visualizer lets users navigate the catalog as a graph, follow causal chains, and explore stakeholder-specific views. Analysis of the catalog revealed a persistent imbalance between attack-focused and defense-focused research, a concentration of threats at ingestion, and coverage gaps affecting output integrity. Coverage is assessed against the OWASP LLM Top~10, and operational applicability is illustrated across textual, graph-based, and multimodal RAG configurations.

## Full Text


<!-- PDF content starts -->

A Security Meta-Model for Retrieval-Augmented
Generation Systems
Steve Nouyep, Sébastien Salva, and Maxime Puys
Université Clermont Auvergne, CNRS, LIMOS, 63000 Clermont-Ferrand, France
{steve_durand.nouyep_noundjio, sebastien.salva, maxime.puys}@uca.fr
Abstract. Retrieval-AugmentedGeneration(RAG)systemsextendlarge
language models (LLMs) with external knowledge through a multi-stage
pipeline. While this architecture can improve the factual grounding of
generated answers, it introduces structural attack surfaces that extend
beyond those of standalone LLMs. In this paper, we introduce a security
meta-model that captures explicit causal relationships between RAG
surfaces, attacks, weaknesses, risks, and CIA impact (Confidentiality,
Integrity, Availability). Its purpose is to provide security engineers with a
structured and user-friendly framework for gathering and assessing the
risks, weaknesses, and mitigations relevant to their RAG deployment.
We designed the meta-model through an iterative, structured analysis of
43 publications (2023–2026) and instantiated it as a catalog populated
with the security threats and remediations reported in the literature.
Filtering the catalog according to a deployment configuration produces
a risk profile containing the risks applicable to that deployment. An
interactive web visualizer lets users navigate the catalog as a graph,
follow causal chains, and explore stakeholder-specific views. Analysis of
the catalog revealed a persistent imbalance between attack-focused and
defense-focused research, a concentration of threats at ingestion, and
coverage gaps affecting output integrity. Coverage is assessed against the
OWASP LLM Top 10, and operational applicability is illustrated across
textual, graph-based, and multimodal RAG configurations.
Keywords:RAG·Threat Modeling·Meta-model·Risk Filtering
1 Introduction
Retrieval-Augmented Generation (RAG) systems [15] have emerged as a widely
adopted approach for enhancing Large Language Models (LLMs) with external
knowledge sources. By combining retrieval and generation, they aim to improve
factual grounding and domain adaptation in applications ranging from enterprise
knowledge management to healthcare and e-commerce. However, the architec-
tural decomposition that enables these benefits also introduces new security
considerations. Unlike standalone LLMs, RAG systems expose diverse pipeline
stages (document ingestion, indexing, retrieval, and generation), each of which
may constitute a distinct attack surface.
arXiv:2610.11893v1  [cs.CR]  8 Oct 2026

2 Steve Nouyep, Sébastien Salva, and Maxime Puys
Several studies and surveys [2,4,6,23,26] examine these attacks, propose
mitigation strategies, or discuss their associated risks. While these contributions
provide valuable insights into RAG security threats, they often address individual
aspects in isolation, contributing to a growing body of (often complex) docu-
mentation. As a result, it becomes increasingly difficult for security engineers to
identify and select the most appropriate solution for a given context.
These solutions either address LLM application security generically (e.g.,
OWASP LLM Top 10 [23]) or analyze selected RAG threat classes; what remains
less explored is a structured modeling approach that (i) systematically links
pipeline components to weaknesses, attacks, risks, and Confidentiality, Integrity,
Availability (CIA) impact, and (ii) adapts the assessment to concrete deployment
configurations.
To help security engineers design secure RAG systems, it is essential to
provide a structured and coherent view linking together RAG system types,
attacks, weaknesses, risks, and mitigation strategies, while ensuring that this
perspective remains adaptable to heterogeneous deployment contexts.
Contributions.To address this need, we propose in this paper:
1.A security meta-modelthat specifies entities and relations capturing
structural relationships among RAG components, attacks, weaknesses, risks,
and CIA impact (Section 4), instantiated as a security catalog covering 3
RAG types, 8 attack surfaces, 38 attacks, 12 weaknesses, 41 risks, and 72
mitigations. It is publicly available at https://anonymous.4open.science/
r/rag-mmsecu/.
2.Acontext-dependentfilteringmechanismthat,givenaRAGdeployment
configuration (e.g., network exposure, data source type), identifies the subset
of applicable risks and produces a configuration-specific risk profile, illustrated
on a multimodal e-commerce deployment (Section 5.2).
3.Four complementary taxonomic viewssupporting distinct stakeholder
roles (architect, CISO, pentester, DPO) (Section 5).
4.An interactive web visualizerthat operationalizes the meta-model: it
renders the catalog as a navigable graph, lets users follow causal chains,
applies the deployment filtering interactively, projects the four stakeholder
views, and validates newly drafted attacks against the integration rules.
We designed the meta-model from the entity and relation types extracted
through a structured analysis of 43 publications (2023–2026, plus the foundational
RAG paper from 2020 [15]). We then instantiated it as a catalog populated
with the security data collected from these publications, and derived the four
stakeholder views from this catalog. This structured integration also revealed
cross-cutting patterns, including a systematic imbalance between attack and
defense research (Section 6).
Outline.The remainder of the paper is structured as follows. Section 2
presents background notions and reviews related work. Section 3 details the
corpus construction. Section 4 introduces the meta-model, its iterative derivation,
and its instantiation into the security catalog. Section 5 presents context- and

A Security Meta-Model for Retrieval-Augmented Generation Systems 3
Fig. 1.The web visualizer rendering the security catalog as a causal chain, here focused
on document-ingestion attacks: RAG type →surface →attacks →exploited weakness
→mitigations. Per-column counters show the narrowing effect of the projection (e.g.,
12 of 38 attacks). Filtering and views are detailed in Sections 5.2 and 5.
role-related views of the catalog, exposed through an interactive web visualizer.
Section 6 discusses quality assessment, cross-cutting observations, a research
agenda, limitations, and threats to validity. Section 7 concludes and outlines
future directions.
2 Background and Related Work
2.1 RAG System Architecture and Analysis
RAG systems combine LLMs with an external retrieval mechanism to ground
responses in external corpora [15].
Module decomposition.Gao et al. [9,10] identify three paradigms (Naive,
Advanced, Modular) and, in their Modular RAG framework, define six module
types (Indexing, Pre-/Post-retrieval, Retrieval, Generation, Orchestration) with
14 sub-modules and 40+ operators. We derive from this framework a security-
oriented decomposition of 11 modules (Figure 2) by applying two refinements:
1.Trust-boundary split of Indexing.Gao et al.’s Indexing type is split
into three modules: document acquisition (M1), preprocessing (M2), and
vector indexing (M3). From a functional standpoint they form a single stage,
but from a security standpoint M1straddlesthe external trust boundary
(attacker-supplied documents), whereas M2 and M3 are internal and expose
no direct attack surface. Two conditional sub-modules extend M3: M3.graph
(knowledge-graph indexing, active when type=GRAPHRAG) and M3.multi
(multimodal encoding, active when type=MULTIMODAL).
2.Cross-cutting modules.Two modules operate orthogonally to the sequen-
tial pipeline:conversational memory(which Gao et al. identify but do not

4 Steve Nouyep, Sébastien Salva, and Maxime Puys
Fig. 2.RAG pipeline: 11 modules (M1–M7 sequential, M3.graph/M3.multi conditional,
two cross-cutting), 8 attack surfaces, and data flow. Blue = external-facing; gray =
internal; green/orange = conditional (GraphRAG/Multimodal); yellow = cross-cutting;
white dashed = I/O.
include in their catalog) andagentic orchestration(corresponding to Gao
et al.’s Orchestration type). Both are optional, activated by deployment
properties (Section 5.2), and each exposes a dedicated attack surface when
enabled.
The resulting 11 modules expose 8 attack surfaces (S-DOC, S-QUERY, S-RETR,
S-GEN, S-GRAPH, S-MULTI, S-MEM, S-ORCH); three internal modules (M2,
M3, M6) expose no direct surface but can be targeted indirectly via upstream
entry points [1,30].
RAG type grouping.While the three RAG paradigms cited previously (Naive,
Advanced, Modular) differ operationally, our analysis of their operators [10]
indicates that they expose comparable attack surfaces and CIA impact patterns,
when neither graph-based indexing nor multimodal encoders are employed. We
therefore group them under the umbrella termTEXTUAL, distinguishing
them fromGRAPHRAGandMULTIMODAL, which introduce additional
components and specific attack surfaces. This grouping reflects a risk-oriented
abstraction rather than an architectural claim of equivalence; it independently
converges with the modalities dimension of Nikishina et al. [21].
Trust boundaries.From a security standpoint, two trust boundaries are es-
sential. First, external documents entering M1 may be attacker-controlled and
persist in storage. Second, retrieved passages (M5) are integrated into the LLM
prompt (M7): any attacker-controlled content in the corpus can therefore directly
influence the model’s output, which is the structural basis for indirect prompt in-
jection attacks. These properties motivate the need for a security model explicitly
aligned with pipeline stages.

A Security Meta-Model for Retrieval-Augmented Generation Systems 5
2.2 Threat Modeling Foundations
Modeling RAG security requires bridging information-security risk management
with software weakness enumeration. We adopt the ISO/IEC 27000 and 27005
terminology [12,13], distinguishing threats, vulnerabilities, assets, and risks. In
this context, risk is treated as anintegrative conceptthat captures the combined
effect of a threat exploiting a vulnerability and producing an impact on an asset.
This integrative role motivates our decision to model risks explicitly rather than
treating attacks or weaknesses in isolation.
CAPEC patterns [17] and CWE identifiers [18] are widely used classifica-
tion systems for attack patterns and software weaknesses, providing a common
vocabulary across security tools and analyses. NIST SP 800-30 [22] models risk as-
sessment as a causal chain from threat source to impact. Both serve as structural
anchors for the meta-model of Section 4.
Finally, the OWASP LLM Top 10 [23] provides an external benchmark for
assessing coverage of RAG-relevant categories, while targeting LLM applications
more broadly rather than RAG architectures specifically, and its entries carry no
standard weakness identifiers. MITRE ATLAS [19] provides a complementary
taxonomy of adversarial ML tactics and techniques, but targets ML systems more
generally and does not model RAG pipeline stages or deployment-dependent
applicability. Our attacks could similarly be mapped to ATLAS techniques, but
we leave this alignment for future work.
2.3 RAG and LLM Security Frameworks
Existing work on RAG and LLM security can be grouped into (i) risk-oriented
frameworks [2,23], (ii) formally defined threat models [4], and (iii) descriptive
surveys of attacks and defenses [6,26].
The OWASP LLM Top 10 (2025) [23] provides a practitioner-oriented,
consensus-based ranking of ten risk categories for LLM applications. While
widely adopted, it targets LLM applications generically: RAG pipeline stages
are not modeled, categories are not mapped to CWE/CAPEC identifiers, and
retrieval-related risks (e.g., data poisoning and sensitive information disclosure)
appear as subcases of broader categories rather than architecturally distinct
threats.
In 2025, Arzanipour et al. [4] propose a formal threat model for RAG sys-
tems. They identify three confidentiality-oriented attack classes (membership
inference (DL-MIA), content extraction, and data poisoning) and formalize them
through game-based security definitions with an adversary-challenger paradigm.
Their adversary taxonomy distinguishes four types along two axes: API-level vs.
knowledge-base access and black-box vs. informed. The scope remains limited
to confidentiality-focused threats; broader integrity and availability threats are
outside the model. Differential privacy is the primary formally analyzed de-
fense, while other mitigations are discussed conceptually. The model provides no
CWE/CAPEC mapping, RAG-variant differentiation, or deployment awareness.

6 Steve Nouyep, Sébastien Salva, and Maxime Puys
Ammann et al. (2025) [2] present a layered risk assessment framework identify-
ing eleven RAG-specific risks and thirteen mitigations in a binary risk-mitigation
matrix. It combines IT baseline, AI/LLM, and RAG-specific layers, anchored in
OWASP, NIST RMF, and ISO 27001, and is demonstrated on three use cases.
The approach is component-centric: risks map directly to pipeline elements with-
out typed intermediate entities (attacks, weaknesses) or explicit causal chains.
It also provides no CWE/CAPEC-level mappings, RAG-type distinction, or
deployment-based risk filtering.
Wang et al. (2025) [26] survey RAG security from a full-chain perspective,
identifying four threat classes (adversarial attacks, data poisoning, backdoor
attacks, and privacy leaks) and eight defense techniques organized into three
layers: input hardening, data protection, and privacy preservation. They identify
thePhantomtrigger-based backdoor as an emerging single-document threat.
However, the survey does not follow an explicit systematic review protocol, relate
threats to security properties, or distinguish RAG pipeline stages; multimodal
RAG is only discussed as future work.
Bodea et al. (2026) [6] present the most closely related work: a systematic
literature review of 72 publications that develops a taxonomy of RAG privacy
risks (leakage and adversarial manipulation) and a mitigation catalog scored by
relevance and maturity. Its scope is restricted to confidentiality, with integrity and
availability outside its boundaries. It covers only naive textual RAG, with variant
architectures identified as an open gap, and does not provide CWE/CAPEC
mapping, deployment filtering, or explicit causal chains.
Table 1 summarizes these frameworks using ten criteria, covering scope
(RAG specificity, number of publications analyzed), modeling depth (explicit
specification, causal chain, CWE/CAPEC mapping), and operational support
(RAG types, deployment filtering, mitigations, taxonomic views). Our work
complements them in three ways. First, it provides explicit structural modeling
across RAG pipeline stages using typed entities and relations (rowsExplicit
specificationandExplicit causal chain); the former is matched only by Arzanipour
etal.,whilethelatterisatbestpartialintheotherframeworks.Second,itprovides
traceability to CWE/CAPEC identifiers, which none of the compared frameworks
provides. Third, it provides deployment-aware filtering that adapts the risk profile
to a given RAG configuration (rowDeployment filtering, with six deployment
properties). It also covers all three CIA dimensions, the only entry in Table 1 to
do so (rowCIA coverage).
2.4 Broader Systematizations and Empirical Studies
Severalrecentsystematizationsaddressrelatedbutbroaderscopes.Wangetal.[27]
empirically analyze 529 vulnerabilities across 77 LLM projects, but treat RAG
as one application-layer component among many. Duddu et al. [8] show that
combining ML defenses may degrade security, a principle relevant to RAG but
studied only on classifiers. Schroer et al. [24] examine offensive AI capabilities,
while Hong et al. [11] categorize prompt security for standalone LLMs. None

A Security Meta-Model for Retrieval-Augmented Generation Systems 7
Table 1.Positioning relative to existing work.
Criterion OWASP Arzan. Amm. WangcBodea Ours
RAG-specific Partial✓ ✓ ✓ ✓ ✓
Explicit specification —✓— — —✓
Provable security —✓— — — —
Explicit causal chain — Partial Partial — —✓
CWE/CAPEC mapping — — — — —✓
RAG types — — — — —✓(3)
Deployment filtering — — — — —✓(6)
Mitigations✓DPa✓(13)✓(8)✓ ✓(72)
Taxonomic views — — — — —✓(4)
Publications analyzed Ind.b∼15∼20 25 72 43
Methodology Consensus Formal Risk assess. Narr. survey SoK Iter. taxonomy
CIA coverage C,I C,I C,I C,I C C,I,A
aDefensive-posture recommendations only (no specific countermeasures).
bCommunity consensus, not a systematic literature corpus.
cWang et al. [26] (RAG security survey); distinct from [27,28].
provides a unified structural model of RAG-specific threats across the three CIA
dimensions.
Empirical evidence.Recent experiments converge on three findings that moti-
vate a structured security model. First, retrieval reshapes rather than reduces
risk: RAG increases unsafe outputs in 8 of 11 models, even from safe docu-
ments [3], and shifts exposure from training data to retrieval data [29]. Second,
both integrity and confidentiality are practically exploitable: knowledge-base
poisoning achieves high success rates under black-box access [30], while implicit
knowledge extraction succeeds through benign-looking queries [28]. Benchmark-
ing [16] and certified bounds [14] further support the practical relevance of these
threats. Third, contextual factors such as access model, underlying LLM, and
data provenance affect severity, but their interplay has not been systematically
studied.
3 Corpus Construction and Scope
We conducted a structured literature analysis targeting RAG-specific security
publications. The objective was structural representativeness, that is, identifying
sufficient patterns to derive a coherent meta-model rather than the exhaustiveness
of a formal systematic literature review.
Search strategy.We searched three sources: Google Scholar, IEEE Xplore, and
arXiv. We executed six keyword queries on arXiv (title search, Computer Science
categories, cross-listed papers included) over January 2020–February 2026:“RAG
attack,” “RAG poisoning,” “RAG privacy,” “RAG security,” “prompt injection
AND RAG,”and“RAG membership inference.”These queries returned 67 results
before deduplication (31, 11, 10, 10, 3, and 2 respectively); all security-relevant
results date from 2024 onward. Complementary searches on IEEE Xplore and
Google Scholar captured published work. When the full text was inaccessible,
the corresponding arXiv preprint was used, giving 78 articles in total.

8 Steve Nouyep, Sébastien Salva, and Maxime Puys
Selectionprocess.Aftercross-queryandcross-databasededuplication,50unique
candidates remained. Backward snowballing identified additional publications
not captured by keyword queries, including papers using compound names such
as SafeRAG, BadRAG, and BiasRAG, as well as foundational work predating
2024. Inclusion required at least one RAG-specific attack or defense with suf-
ficient technical detail to identify the exploited pipeline surface and resulting
CIA impact. Papers addressing LLM security without retrieval, or RAG quality
without a security dimension, were excluded.
Final corpus.The retained corpus comprises43 publications: 31 attack-
focused (72%), 9 defense-focused (21%), and 3 frameworks (7%). By venue,
14 are arXiv-only preprints (33%), 19 appear in major conferences (AAAI, ACL,
ACM CCS, EMNLP, ICML, ICLR, USENIX Security, WWW; 44%), and 10 in
journals or workshops (23%). We mitigate the preprint share by cross-checking
published versions where available and applying the same inclusion threshold
to all publications. Extracted entities are additionally grounded in CWE and
CAPEC, independent of publication maturity. Normative references (ISO 27005,
NIST SP 800-30, CWE, OWASP) provide further structural anchors.
This corpus serves two purposes in the remainder of the paper: deriving the
structure of the meta-model (Section 4.1) and populating its catalog (Section 4.4).
4 The RAG Security Meta-Model
We use the termmeta-modelin the sense of Bézivin [5], that is, a model defining
the structure (entities, relations, constraints) in which concrete models are
instantiated. Our meta-model does not describeonespecific RAG system, but
the common structure underlying security models of RAG systems, based on the
literature review in Section 3. It is extensible, as new attacks can be added without
changing the structure, and actionable, as a RAG configuration automatically
produces a configuration-specific risk profile.
We deliberately specify the meta-model using an entity–relationship style
rather than heavyweight metamodeling standards (MOF/EMF) or OWL ontolo-
gies: ER diagrams and property graphs are familiar to security practitioners,
and the integration rules that populate the model (Section 4.4) require only
boolean criteria over typed links. This choice does not prevent a future OWL
formalization if automated inference becomes necessary.
Structural specification.Following Bézivin [5], we define the meta-model as a
tripleM= (E, R, C ), where Eis the set of entity types (RAG, Surface, Attack,
Weakness, Risk, Mitigation, CIA), Ris the set of directed relation types linking
them (e.g.,exposes,enables,exploits,generates,affects,mitigates,remediates),
andCis the set of structural constraints that valid instances must satisfy:
–Every Attack must be enabled by at least one Surface.
–Every Attack must exploit at least one Weakness.
–Every Attack must generate at least one Risk.

A Security Meta-Model for Retrieval-Augmented Generation Systems 9
Fig. 3.Entity-relationship model: attributes, types, and N:M cardinalities. Bold =
type-discriminant; P1–P6 drive risk filtering (Section 5.2).
–Every Risk must affect at least one CIA dimension (Confidentiality, Integrity,
or Availability).
Together, these constraints ensure that attacks, weaknesses, and risks are
structurally connected to the architecture and its security impact.
4.1 Iterative Construction of the Meta-Model
The meta-model structure was derived following the iterative taxonomy method
of Nickerson et al. [20]. We first defined themeta-characteristicshared by all
model elements:linking each RAG architectural surface to its CIA impact through
a causal chain. An initial conceptual-to-empirical iteration seeded the entity
types from ISO 27005 risk terminology and Ammann et al.’s component-centric
framework [2]; subsequent empirical-to-conceptual iterations refined this skele-
ton against the corpus. This process led to splitting indexing into acquisition,
preprocessing, and indexing modules with distinct trust boundaries; introduc-
ing conditional surfaces (S-GRAPH, S-MULTI, S-MEM, S-ORCH) for type-
and deployment-specific attacks; adding the Mitigation entity and itsmiti-
gates/remediatesdistinction; and deriving deployment properties P1–P6 from
recurring applicability conditions.
We stopped when Nickerson et al.’s objective ending conditions were met: all
corpus publications had been examined and the last iteration introduced no struc-
tural additions. Structural saturation was reached after the first 35 publications:
the remaining eight confirmed existing categories but introducedno new entity
or relation types. Even when restricted to the 29 peer-reviewed publications, the
corpus still instantiates all seven entity and seven relation types, supporting the
structural coverage of the resulting meta-model shown in Figure 3.

10 Steve Nouyep, Sébastien Salva, and Maxime Puys
Table 2.RAG types and exclusive risks.
Type Description Module Exclusive risks
TEXTUAL Naive+Advanced+Modular Base (M1–M7) —
GRAPHRAG + Knowledge Graph M3.graph (S-GRAPH) R17
MULTIMODAL + Non-textual encoders M3.multi (S-MULTI) R24, R25, R29, R39
4.2 RAG Types and Modules
Three RAG types are defined by a risk bifurcation criterion: a type is created if
and onlyif itintroduces risks absent from othertypes (Table 2;formal justification
in Section 5.2).
Justification for TEXTUAL grouping.The three operational variants of
text-based RAG (Naive, Advanced, Modular [10]) expose identical attack surfaces
and CIA impact patterns in our corpus when graph and multimodal components
are absent. We therefore group them as TEXTUAL, creating distinct types only
when new risks emerge: knowledge-graph indexing introduces graph poisoning
(M3.graph), and multimodal encoding introduces cross-modal attacks (M3.multi).
The complete module-to-surface mapping is available in the supplementary
material (https://anonymous.4open.science/r/rag-mmsecu/).
4.3 Causal chain
The seven entity types form a causal chain: a RAG systemexposessurfaces,
whichenableattacks; attacksexploitweaknesses andgeneraterisks, whichaffect
CIA dimensions; mitigationsmitigaterisks andremediateweaknesses. Figure 3
illustrates the attribute schema. The causal-chain design follows NIST SP 800-
30 [22], which models risk assessment as a chain from threat source to impact: the
meta-model makes this chain explicit by linking RAG pipeline surfaces to attacks,
weaknesses, and CIA impacts. Themitigatesrelation is contextual (which risk
is reduced), whileremediatesis structural (which weakness is eliminated); only
preventive mitigations carry both relations, detective and reactive mitigations
carrying onlymitigates(IR8, Section 4.4).
4.4 Security Data Integration and Catalog
Eight integration rules (IR1–IR8, Table 3) specify how links between entities
are constructed when populating the meta-model. They are organized into three
groups, each with a distinct objective and verification criteria.
Group 1: Anchoring attacks in the architecture(IR1–IR3).Objective:determine
which surfaces are exposed by a given RAG configuration, which attacks they
enable,andwhichweaknessesthoseattacksexploit.IR1requiresthecorresponding
module to be active and the surface accessible; IR2 requires an entry vector
documented in at least one publication; IR3 requires a traceable CAPEC →CWE
mapping or an explicit root-cause analysis. This mapping ensures interoperability

A Security Meta-Model for Retrieval-Augmented Generation Systems 11
Table 3.Integration rules for instance construction.
# Relation Link CriterionAnchoringIR1 exposes RAG→Surf. Module activeandsurface
accessible
IR2 enables Surf.→Att. Entry vector in≥1 publication
IR3 exploits Att.→Weak. CAPEC→CWE mapping or
root-causeConseq.IR4 generates Att.→Risk 1 primary + 0..N secondary
IR5 assigns⋆Risk.assetConsequence heuristic
IR6 affects Risk→CIA Discl.→C; modif.→I;
denial→ADef.IR7 mitigates Mit.→Risk Reduces likelihood or impact⋆⋆
IR8 remediates Mit.→Weak. Preventive mitigations only
(43/72)
⋆Attribute assignment, not an inter-entity relation (Asset/∈E).
⋆⋆5 evidence sources: attack papers, CWE, defense papers, standards, deduction.
with established security taxonomies; it is not strictly one-to-one, as some ML-
specific integrity issues are only coarsely represented in CWE and some RAG-
specific attacks are only partially covered. A link is created only when its criterion
is satisfied; otherwise, the entity remains in the catalog but unconnected.
Group 2: Deriving consequences(IR4–IR6).Objective:derive the risks resulting
from each attack, assign the affected asset, and determine the CIA impact. IR4
produces one primary risk per attack, corresponding to the documented direct
consequence, plus zero or more secondary risks for reported cascading effects.
IR5 assigns the affected asset as a Risk attribute rather than an entity, keeping
the meta-model focused on the causal chain while supporting the asset-based V4
view. IR6 maps disclosure, modification, and denial to Confidentiality, Integrity,
and Availability, respectively, following the standard information-security impact
classes [13]. Every attack must generate at least one risk, and every risk must
affect at least one CIA dimension.
Group 3: Linking defenses(IR7–IR8).Objective:connect mitigations to the
risks they reduce and the weaknesses they remediate. IR7 requires evidence
from one of five sources (attack-paper defenses, CWE recommendations, defense
papers, standards, or logical deduction) that the mitigation can reduce likelihood
or impact. IR8 links onlypreventivemitigations (43/72) to weaknesses, since
detective and reactive measures address risks without eliminating their underlying
cause.Eachlinkcarriesatraceabilitytag;mitigationswithoutsupportingevidence
remain in the catalog but are not linked to a risk.
The resulting catalog.Applying these rules to the 43 publications of Section 3
produces thecatalog, the instance of Mcontaining all extracted entities and
their relations: 3 RAG categories, 8 surfaces, 38 attacks, 12 weaknesses, 41 risks,
72 mitigations, and the 3 CIA dimensions. The attacks fall into five categories:
Poisoning (16), Exfiltration (11), Inference (5), Injection (4), and Degradation (2);
poisoning is the largest (42%), followed by exfiltration (29%).
Each risk is characterized by an attack, an affected asset, and one or more
CIA dimensions, across five assets: CORPUS (20), OUTPUT (12), MEMBER-

12 Steve Nouyep, Sébastien Salva, and Maxime Puys
Table 4.Illustrative causal chains (one per major attack category).
Surface Attack CWE Risk Asset CIA Mit.(#)
S-DOC Corpus poison. 345 R8 CORPUS I MI3, MI14, MI17 (11)
S-GEN Indirect inj. 1427 R9 OUTPUT I MI4, MI5, MI16 (7)
S-RETR Data exf. 200 R2 CORPUS C MI0, MI9, MI27 (15)
S-RETR Memb. inf. 200 R4 MEMB C MI0, MI1, MI62 (4)
Mit.(#)=Mitigations (count). MEMB=MEMBERSHIP. C=Confidentiality, I=Integrity.
SHIP (5), EMBEDDINGS (3), and PROMPTS (1). Integrity accounts for 20 risks,
confidentiality for 17, and availability for 1; three span multiple CIA dimensions,
and three (R0, R26, R28) capture emerging or organizational risk scenarios
anchored in normative sources rather than directly derived from a cataloged
attack. The mitigations divide into 43 preventive, 20 detective, and 9 reactive,
classified following ISO 27001 and NIST CSF.
The catalog includes 12 CWE mappings covering all five attack categories:
INJ maps to CWE-1427/74/693/863, POI to CWE-345/94/1321, INF to CWE-
200/203, EXF to CWE-200/212, and DEG to CWE-400/770. Seven of the eight
surfaces are exercised by at least one attack; S-ORCH is included for extensibility
but has no dedicated attack in the current corpus. The catalog is stored as a
property graph, and its structural constraints are checked mechanically at import
time. Two derived artifacts must not be confused with the catalog: filtering it with
a specific RAG deployment configuration (Section 5.2) produces arisk profile,
the subset of risks applicable to that deployment; and the stakeholderviewsof
Section 5 are projections of the catalog, not new instances.
Table 4 gives four representative causal chains covering the main attack categories
(POI, INJ, EXF, and INF); degradation attacks (DEG) follow the same structure
but primarily target availability.
Walkthrough.To illustrate how the integration rules apply sequentially to
construct an explicit causal chain, consider the first chain in Table 4: IR1 deter-
mines that surface S-DOC is exposed (module M1 active); IR2 links S-DOC to
the corpus-poisoning attack (entry vector documented in [30]); IR3 maps that
attack to CWE-345 via its CAPEC entry; IR4 generates risk R8 as the primary
consequence (each attack produces exactly one primary risk, plus zero or more
secondary risks when a single technique causes cascading effects); IR5 assigns
asset CORPUS to R8; IR6 labels R8 as Integrity; and IR7 links the 11 mitigations
whose evidence (from attack-paper defenses, CWE recommendations, dedicated
defense papers, standards, or logical deduction) shows they reduce R8’s likelihood
or impact. The same sequence can be applied to any new attack without changing
the meta-model structure (Section 5.2 details the deployment properties that IR1
depends on).
5 Views Related to Context and Role
The full catalog can be covered, but it is difficult to navigate as a whole, as
most security questions concern only a subset of it. We therefore provide an

A Security Meta-Model for Retrieval-Augmented Generation Systems 13
Table 5.RAG deployment properties.
# Property Values Semantics
P1typeTEXTUAL, GRAPHRAG, MULTIMODAL Architecture
P2ingestion_sourceINTERNAL, AUTHENTICATED, OPEN Ingestion control
P3network_exposureISOLATED, INTERNAL, PUBLIC Network exposure
P4memory_enabledtrue / false Conversation history
P5llm_locationON_PREMISE, CLOUD, HYBRID LLM location
P6agentictrue / false Action execution
interactive web visualizer (Section 5.1) with purpose-builtviews: context-related
views (Section 5.2) filter the catalog by deployment configuration, while role-
related views (Section 5.3) present the elements relevant to a given stakeholder
role.
5.1 The Interactive Web Visualizer
Both view families are exposed through an interactive web visualizer that renders
the catalog as a navigable graph (Figure 1). From it, a user follows causal
chains from surfaces to CIA impact, switches between the role-related views of
Section 5.3, applies the context-related filtering of Section 5.2 with risks relabeled
live as the deployment configuration changes, and drafts new attacks that are
validatedagainsttheintegrationrulesbeforetheyenterthecatalog.Thevisualizer,
together with the full catalog and the datasets and scripts that generate it, is
publicly available athttps://anonymous.4open.science/r/rag-mmsecu/.
5.2 Views Related to Context
Not every threat applies to every RAG configuration: an isolated system has
no network-based risks, while a text-only pipeline has no multimodal threats.
We therefore complement the catalog with context-dependent filtering, which
dynamically annotates risks according to a deployment configuration without
modifying the meta-model structure.
Deployment properties.A RAG deployment is characterized by six properties
(P1–P6) that determine risk applicability (Table 5). They were derived bottom-up
from the 41 risks by grouping deployment conditions with the same security im-
plication. For example, open Internet sources and untrusted third-party suppliers
are both captured by P2=OPEN. A property was retained only if it produces at
least one non-NORMAL effect in Table 6; parameters affecting severity alone
were excluded.
Choice of risk as segmentation criterion.We considered weakness (CWE),
surface, attack, and risk as candidate criteria; Table 10 (Appendix B) compares
their discriminant power. CWE has no type-specific items, while surfaces have
only 2 of 8; risk and attack each have 5 type-specific items. Risk is preferred
because it carries strictly more information: Risk =f(Attack,Weakness,Asset,
CIA). A risk thus subsumes its attack, while the converse does not hold.

14 Steve Nouyep, Sébastien Salva, and Maxime Puys
Table 6.Filtering table: property×value→effect.
Property Value Effect Risks
type̸=GRAPHRAG ELIMINATES R17
type̸=MULTIMODAL ELIMINATES R24, R25, R29, R39
ingestion_source=INTERNAL MITIGATES R8, R11, R19, R20
network_exposure=ISOLATED ELIMINATES R22, R40
memory_enabled=false ELIMINATES R16
llm_location∈{CLOUD, HYBRID} AGGRAVATES R2, R6, R7
agentic=true ADDS Surface S-ORCH
Algorithm 1Context-dependent risk filtering.
1:functionApplicableRisks(config,Rules)
2:label[r]←normalfor eachriskrin the catalog
3:for all(prop,op,value,effect,targets)inRulesdo
4:ifconfig[prop]op valuethen
5:ifeffect=addsthen
6:activatesurfacetargetsand its cataloged risks
7:else
8:label[r]←effectfor eachr∈targets
9:return{(r,label[r]) :label[r]̸=eliminated}
Filtering algorithm.Each property value produces one of four effects: ELIM-
INATES removes an impossible risk (e.g., graph-specific risks in a text-only
pipeline); MITIGATES reduces its likelihood; AGGRAVATES increases its im-
pact (e.g., a cloud-hosted LLM increases confidentiality exposure); and ADDS
activates an additional attack surface (e.g., agentic capabilities activate S-ORCH).
Each risk therefore receives one of four labels (eliminated,mitigated,aggra-
vated, ornormal) that stakeholders can use to prioritize analysis; the first three
correspond directly to the effects above. Theaddseffect operates on surfaces
rather than risks: the risks of an added surface, if any, enter the profile with the
normallabel unless another rule applies. The rules in Table 6 are processed
generically by Algorithm 1: new rules extend the table without changing the
procedure, and new deployment properties can be added without altering the
meta-model schema. The web visualizer implements this filtering interactively,
relabeling risks as the user changes the deployment configuration.
Illustrative example.Consider a multimodal e-commerce search engine config-
ured as: type=MULTIMODAL, ingestion=OPEN, network=PUBLIC, memory
enabled, LLM=CLOUD, non-agentic. Applying Algorithm 1, only R17 (graph-
specific poisoning) is eliminated because the system does not use knowledge-graph
indexing. No risks are contextually mitigated because ingestion is open, while
three confidentiality risks (R2, R6, R7) are aggravated because the cloud-hosted
LLM involves data processing by an external provider. The result is40 applica-
ble risks out of 41, the most exposed profile in our catalog. This illustrates
that the primary value of filtering is not risk elimination but risk differentiation:
even in this worst-case configuration, the algorithm distinguishes structurally

A Security Meta-Model for Retrieval-Augmented Generation Systems 15
Table 7.Taxonomic views: projection per stakeholder role.
View Audience Question Projected Hidden
V1 Architect “What weaknesses?” RAG, Surf., Att., Weak., Mit. Risk, CIA
V2 CISO “What to do?” Att., Weak., Risk, Mit. RAG, Surf., CIA
V3 Pentester “Full chain” RAG, Surf., Att., Weak., Risk, Mit. CIA
V4 DPO “What risks?” Risk, CIA, Mit.(+asset) RAG, Surf., Att., Weak.
impossible risks from aggravated ones, enabling practitioners to prioritize their
analysis.
5.3 Views Related to Role
In practice, no single stakeholder requires the full catalog at once. Different
stakeholders need distinct views, emphasizing elements and relations relevant to
their role. For example, a security architect deciding which CWE weaknesses to
remediate before deployment does not require detailed CIA-level impact data,
whereas a data protection officer assessing GDPR exposure does not need the full
surface-to-weakness dependency chain. We therefore define aviewas a projection
of the catalog that retains only the entity types and relations relevant to a given
role, producing a self-contained subgraph with reduced cognitive load.
Table 7 summarizes four views, one per stakeholder role. Each row specifies
which entity types and relations areprojected(visible in the subgraph) and which
arehidden(abstracted away). The projected elements define the subgraph that
stakeholders navigate; hidden elements remain in the underlying catalog but are
not required for the role’s primary question. We describe each view below.
V1-Architect (5 entities, 4 relations).This view focuses on the architectural
perspective by projecting the chain RAG →Surface →Attack →Weakness
←Mitigation through the relationsexposes,enables,exploits, andremediates.
Risk and CIA are hidden, allowing the architect to reason directly about how
architectural components expose attack surfaces that lead to weaknesses and
their available remediations.Question: where are the architectural entry points
that may lead to risks?By navigating the pipeline from system components to
weaknesses, this view helps identify structural security issues early in the design.
Figure 1 (page 3) shows a partial V1 projection forTextualRAG, restricted to
surface S-DOC and weakness CWE-345; the full V1 subgraph contains 78 nodes
and 140 relationships.
V2-CISO (4 entities, 4 relations).This view focuses on the risk-mitigation
perspective by projecting the mapping Attack →Weakness →Risk←Mitigation
(with Mitigation also →Weakness) through the relationsexploits,generates,
mitigates, andremediates. RAG, Surface, and CIA are hidden, allowing the CISO
to reason directly about which defenses address which risks.Question: which
mitigations exist and where are the coverage gaps?Risks are grouped by category
and linked to their available mitigations, making missing protections immediately

16 Steve Nouyep, Sébastien Salva, and Maxime Puys
visible. Five risks (R17, R33, R34, R35, R36) currently have no documented
mitigation.
V3-Pentester (6 entities, 6 relations).This view combines V1 and V2 to
expose the complete chain RAG →Surface →Attack →Weakness →Risk, while
Mitigation connects both to Weakness (remediates) and Risk (mitigates). Only
CIA is hidden.Question: how can a structured security assessment be planned?
The pentester navigates the graph in both directions: forward along the V1 path
to identify potential entry points, and backward along the V2 path to verify
deployed defenses.
V4-DPO (3 entities, 2 relations, + asset filter).This view projects Risk →
CIA and Mitigation →Risk through the relationsaffectsandmitigates, filtered
by the target asset (CORPUS, OUTPUT, MEMBERSHIP, EMBEDDINGS,
PROMPTS). RAG, Surface, Attack, and Weakness are hidden, enabling the DPO
to reason directly in terms of data assets and CIA impact rather than technical
attack chains.Question: for a given asset, which risks apply and what mitigations
exist?CORPUS concentrates 49% of all risks (20/41), while OUTPUT exhibits
the highest mitigation gap (25%: 3 risks without mitigation out of 12), making
this view directly usable as input to GDPR Data Protection Impact Assessments.
The four views are complementary projections, not prescriptive steps: each
stakeholder selects the view matching their role and obtains a self-contained
subgraph. Because a view is simply a projection defined by a subset of entity
types and relations, practitioners may define additional views tailored to their
own needs.
6 Discussion
6.1 Quality Assessment
We examine the meta-model along six quality criteria, drawn from taxonomy
evaluation literature [20,25]:coverage(does the catalog span the domain?),
determinism(do the rules produce a unique result?),repeatability(can the process
be reproduced?),unambiguity(is each entity clearly defined?),usefulness(does it
serve its stakeholders?), andnavigability(can users find relevant information?).
Coverage against OWASP LLM Top 10.Table 8 maps each OWASP LLM
Top 10 (2025) entry [23] to our meta-model entities. Seven entries are fully
covered; two entries are partially covered, meaning that our catalog captures
their RAG-specific manifestation (e.g., poisoning via document ingestion for
Supply Chain, or the orchestration surface for Excessive Agency), but does not
address their broader LLM-generic aspects (e.g., training-time attacks or plugin
vulnerabilities); Unbounded Consumption falls outside our scope as it concerns
LLM resources rather than RAG-specific threats.
Determinism, Unambiguity, and Repeatability.Entity definitions are
grounded in established standards and taxonomies (ISO 27005 for risk, and
CWE/CAPEC for weaknesses and attack patterns), reducing ambiguity. The

A Security Meta-Model for Retrieval-Augmented Generation Systems 17
Table 8.Coverage of OWASP LLM Top 10 (2025) by the meta-model.
OWASP Entry Title Mapping Risks Cov.
Prompt Injection (01) INJ CWE-1427/94; S-GEN R9, R10 Full
Sensitive Info Discl. (02) EXF/INF CWE-200/212 R2, R4–R7 Full
Supply Chain (03) POI S-DOC; CWE-345 R8, R11 Part.†
Data/Model Poison. (04) POI CWE-345/1321; S-DOC R8, R11–R13, R15 Full
Improper Output (05) POI/DEG CWE-345/770; S-DOC R34–R36 Full
Excessive Agency (06) — S-ORCH; P6=agentic —‡Part.‡
Prompt Leakage (07) EXF CWE-200; S-GEN R6 Full
Vector/Embed. (08) POI/EXF S-RETR; CWE-345 R8, R2 Full
Misinformation (09) POI/INJ S-DOC/S-GEN R19, R34 Full
Unbound. Consump. (10) — (LLM resource mgmt.) — OOS
†Beyond doc. ingestion = out of scope.‡Covered via S-ORCH; no dedicated risk IDs. OOS = Out of Scope.
eight integration rules (IR1–IR8) are declarative: each specifies a boolean crite-
rion (Table 3) that must be satisfied for a relation to be created. For instance,
IR2 creates anenableslink between a surface and an attack only if at least
one publication documents the corresponding entry vector. Once the extracted
entities and evidence are fixed, the declarative rules deterministically produce the
corresponding set of links. Repeatability follows from the same explicit process:
the construction and stopping conditions of Section 4.1 make the derivation
reproducible. This determinism concerns theapplicationof the rules to a given
corpus; its relation to the completeness of that corpus is discussed in Section 6.4.
Usefulness and Navigability.Usefulness and navigability are supported by the
interactive web visualizer (Section 5.1), which turns the catalog into navigable,
filterable views. Stakeholders can explore causal chains, filter risks by deployment
context, or project the catalog onto their role, reaching relevant information
without traversing the full catalog. Empirical validation with practitioners is left
to future work (Section 6.5).
Limitations of extraction.Entity and relation extraction may introduce sub-
jectivity due to the interpretative nature of the process. We mitigate this through:
(i) a systematic grounding in CWE/CAPEC identifiers; (ii) explicit integration
rules (IR1–IR8); and (iii) four structural constraints verified programmatically:
every attack must be linked to at least one surface (C1), one weakness (C2), and
one risk (C3), and every risk must affect at least one CIA dimension (C4). These
constraints are checked automatically during catalog generation, rejecting any
entry that violates the meta-model schema.
6.2 Cross-Cutting Observations
By aggregating 43 publications into a single connected structure, the catalog
reveals patterns that individual studies cannot capture.
Offensive/defensive imbalance.Of 43 publications, 31 are attack papers (72%)
versus 9 defense papers (21%). Many mitigations originate from the “defenses”
section of attack papers rather than from systematically evaluated defense work,
and several have already been bypassed: perplexity-based detectors [1] by adaptive
attacks [16], and similarity filtering by reformulated queries [7]. Bodea et al. [6]

18 Steve Nouyep, Sébastien Salva, and Maxime Puys
Table 9.Per-asset gap analysis.
Asset Risks Gaps Gap ratio
CORPUS 20 2 (R17,R33) 10%
OUTPUT 12 3 (R34,R35,R36)25%
MEMBERSHIP 5 0 0%
EMBEDDINGS 3 0 0%
PROMPTS 1 0 0%
independently report a similar 56%/36% attack-to-defense imbalance in a privacy-
focused RAG corpus of 72 publications.
Ingestion as structural hotspot.Surface S-DOC concentrates 16 of 38 at-
tacks (42%), 14 of them poisoning attacks. Unlike transient injection attempts,
poisoning can persist in the corpus until detected or removed, making ingestion
control critical. With trusted internal sources (P2=INTERNAL), four poison-
ing risks are contextually mitigated; with open ingestion (P2=OPEN), none
of the 16 poisoning attacks is mitigated by the ingestion property alone, and
their applicability still depends on the other deployment properties. This leaves
open-ingestion configurations particularly exposed to integrity threats.
Output under-protection and coverage gaps.Table 9 shows that OUTPUT
(the answer returned by the RAG system) has the highest gap ratio (25%): R34
(inter-context conflict), R35 (soft ad injection), and R36 (white DoS), all common
to all RAG types, have no incomingmitigatesedge (Table 11). Two gaps also
affect CORPUS: R17 (graph poisoning, GRAPH-specific) and R33 (silver noise
injection, common to all RAG types). This may reflect a publication bias or
limitations of the available corpus: output attacks are perceived as less novel or
impactful than poisoning attacks and receive less research attention.
6.3 Practitioner Use Cases
Each view supports a distinct decision at a specific stage of deployment.At
design time, architects (V1) compare deployment configurations (P1–P6) and
assess risks before implementation; for example, internal ingestion contextually
mitigates four poisoning risks.In operation, CISOs (V2) prioritize applicable
risks and identify coverage gaps.During security assessment, pentesters (V3)
derive configuration-specific test scopes from reachable attack chains; an internal,
isolated deployment eliminates two risks and contextually mitigates four poisoning
risks.Across these stages, DPOs (V4) obtain per-asset information supporting
GDPR and NIS2 assessments, including information relevant to cross-border
data transfers (GDPR Art. 46).
The web visualizer supports these use cases by projecting views, filtering
risks by configuration, and exposing causal chains. It also allows new attacks to
be drafted and checked against the integration rules for inclusion in the cata-
log. Whether these views effectively support practitioners remains an empirical
question, left for future work in Section 6.5.

A Security Meta-Model for Retrieval-Augmented Generation Systems 19
6.4 Limitations and Threats to Validity
Catalog limitations.The catalog reflects the literature available and may
require periodic updates; its extensibility (IR2–IR4, IR7) allows new threats to
be integrated without structural change. CWE coverage remains coarse: 14 of the
16 poisoning attacks map solely to CWE-345, reflecting CWE’s limited coverage
of ML-specific integrity flaws rather than a limitation of our meta-model.
Internal rigor and completeness.The integration rules ensure consistent
extraction: given the same corpus, the same causal chains are derived. They do
not, however, guarantee completeness. Publication bias, coarse CWE coverage,
and genuinely unknown threats limit what a literature-based approach can
capture. Completeness therefore holds only with respect to the documented
threat landscape, a blind spot the catalog makes explicit through R0.
RAG specialization.RAG is deliberately scoped because deployment-aware
filtering and surface-level causal chains require a concrete architecture. The
catalog does not transfer directly to other LLM applications, but the meta-
model structure and construction process are architecture-agnostic and could be
instantiated for, e.g., agentic pipelines.
Methodological limitations.Attacks are modeled individually, so attack
combinations (e.g., chaining an injection with an exfiltration) are not represented,
although real-world adversaries may exploit such sequences.
Threats to validity.Internal: entity and relation extraction involves manual
judgment by a single coder, mitigated by systematic CWE/CAPEC grounding
and the boolean criteria of Section 6.1; inter-coder agreement was not formally
measured.External: the meta-model covers the three RAG types but has not yet
been validated on production deployments, and the four views await practitioner
evaluation (Section 6.5).Construct: definitions are grounded in ISO 27005, CWE,
and CAPEC, with the attack/risk distinction applied consistently.
6.5 Perspectives
The previous observations point to three priority directions. 1) Closing the
persistent coverage gaps: graph poisoning (R17) has no documented mitigation
despite the rapid adoption of GraphRAG [9]; silver noise (R33) targets the
CORPUS asset shared by all types; and the output-integrity cluster (R34–R36)
combines a 25% coverage gap with a probable publication bias. 2) Re-evaluating
mitigations proposed in attack papers, many of which lack systematic validation
against adaptive adversaries. 3) Empirical validation: a task-based user study
in which practitioners of each role answer their Section 6.3 question on a real
deployment.
7 Conclusion
We have presented a structurally specified security meta-model for RAG systems
that, unlike existing frameworks, integrates an explicit causal chain from architec-
ture to CIA impact, bridges RAG-specific threats to standardized weakness and

20 Steve Nouyep, Sébastien Salva, and Maxime Puys
attack-pattern identifiers (CWE/CAPEC), and adapts to deployment context
through property-based risk filtering. Eight declarative integration rules verify
and constrain how data from the literature is incorporated into a RAG security
catalog.
Its practical value lies in three mechanisms: the filtering algorithm turns a
large catalog into a deployment-specific risk profile; the four taxonomic views let
each stakeholder reach the information relevant to their role; and the structural
constraints, reinforced by the integration rules, enforce structural consistency as
the catalog grows (with the limits discussed in Section 6.4). The cross-cutting anal-
ysis further reveals actionable patterns, notably the persistent under-protection
of output integrity and the need to re-evaluate proposed defenses.
Three directions extend this work (Section 6.5): closing the five persistent
coverage gaps, re-evaluating mitigations whose effectiveness remains unvalidated,
and empirically validating the meta-model on production deployments with
practitioner evaluation of the views.
Ethical statement.This work systematizespublicly documentedattacks; no
new exploit is developed.
Open science.Supplementary material ( https://anonymous.4open.scienc
e/r/rag-mmsecu/ ) includes the full CSV database, regeneration scripts, the
pregenerated Neo4j database, and the self-contained web visualizer supporting
role-based views, causal-chain exploration, deployment filtering, and IR-validated
attack drafting. Long-term Zenodo storage will be registered upon acceptance.
Acknowledgement.ThisworkwassupportedbytheIndustrialChaironReliable
and Confident Use of LLMs1and by the MIAI Cluster, France 2030 (ANR-23-
IACL-0006).
References
1.Abdelnabi, S., Greshake, K., Mishra, S., Endres, C., Holz, T., Fritz, M.: Not what
you’ve signed up for: Compromising real-world LLM-integrated applications with
indirect prompt injection. In: Proceedings of the 16th ACM Workshop on Artificial
Intelligence and Security (AISec). pp. 79–90 (2023). https://doi.org/10.1145/
3605764.3623985, arXiv:2302.12173
2.Ammann, L., Ott, S., Landolt, C.R., Lehmann, M.P.: Securing RAG: A risk
assessment and mitigation framework. In: IEEE Swiss Conference on Data Science
(SDS). pp. 127–134 (2025). https://doi.org/10.1109/SDS66131.2025.00024 ,
arXiv:2505.08728
3.An, B., Zhang, S., Dredze, M.: RAG LLMs are not safer: A safety analysis of
retrieval-augmented generation for large language models. In: Annual Conference
of the Nations of the Americas Chapter of the Association for Computational
Linguistics (NAACL). pp. 5444–5474 (2025). https://doi.org/10.18653/v1/20
25.naacl-long.281, arXiv:2504.18041
1https://uca-fondation.fr/chaires/chaire-sur-lutilisation-fiable-et-d
e-confiance-des-grands-modeles-de-langage-llms/

A Security Meta-Model for Retrieval-Augmented Generation Systems 21
4.Arzanipour, A., Behnia, R., Ebrahimi, R., Dutta, K.: RAG security and privacy:
Formalizing the threat model and attack surface. In: IEEE International Conference
on Data Mining Workshops (ICDMW). pp. 1387–1394 (2025). https://doi.org/
10.1109/ICDMW69685.2025.00165, arXiv:2509.20324
5.Bézivin, J.: On the unification power of models. Software and Systems Modeling
4(2), 171–188 (2005).https://doi.org/10.1007/s10270-005-0079-0
6.Bodea, A.E., Meisenbacher, S., Klymenko, A., Matthes, F.: SoK: Privacy risks and
mitigations in retrieval-augmented generation systems. In: IEEE Conference on
Secure and Trustworthy Machine Learning (SaTML) (2026), arXiv:2601.03979
7.Di Maio, C., Cosci, C., Maggini, M., Poggioni, V., Melacci, S.: Pirates of the RAG:
Adaptively attacking LLMs to leak knowledge bases. In: European Conference on
Artificial Intelligence (ECAI). pp. 4041–4048 (2025). https://doi.org/10.3233/
FAIA251293, arXiv:2412.18295
8.Duddu, V., Szyller, S., Asokan, N.: SoK: Unintended interactions among machine
learning defenses and risks. In: IEEE Symposium on Security and Privacy (S&P)
(2024), arXiv:2312.04542; Distinguished Paper Award
9.Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 (2024)
10.Gao,Y.,Xiong,Y.,Wang,M.,Wang,H.:ModularRAG:TransformingRAGsystems
into LEGO-like reconfigurable frameworks. arXiv preprint arXiv:2407.21059 (2024)
11.Hong,H.,Wu,S.,Feng,S.,Naderloui,N.,Yan,S.,Zhang,J.,Arastehfard,A.,Huang,
H., Hong, Y.: SoK: Systematizing LLM prompt security: Taxonomies, datasets, and
unified evaluation of attacks and defenses. arXiv preprint arXiv:2510.15476 (2025)
12.ISO/IEC: Information technology — security techniques — information security
management systems — overview and vocabulary (ISO/IEC 27000:2018) (2018)
13.ISO/IEC: Information security, cybersecurity and privacy protection — guidance
on managing information security risks (ISO/IEC 27005:2022) (2022)
14.Kang, M., Gürel, N.M., Yu, N., Song, D., Li, B.: C-RAG: Certified generation risks
for retrieval-augmented language models. In: International Conference on Machine
Learning (ICML). pp. 22963–23000 (2024), arXiv:2402.03181
15.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Advances in Neural Information
Processing Systems (NeurIPS) (2020), arXiv:2005.11401
16.Liang, X., Niu, S., Li, Z., et al.: SafeRAG: Benchmarking security in retrieval-
augmented generation of large language model. In: Annual Meeting of the As-
sociation for Computational Linguistics (ACL). pp. 4609–4631 (2025). https:
//doi.org/10.18653/v1/2025.acl-long.230, arXiv:2501.18636
17.MITRE: Common attack pattern enumeration and classification (CAPEC) (2024),
https://capec.mitre.org/
18.MITRE: Common weakness enumeration (CWE) (2024), https://cwe.mitre.org/
19.MITRE: ATLAS: Adversarial threat landscape for artificial-intelligence systems
(2025),https://atlas.mitre.org/
20.Nickerson, R.C., Varshney, U., Muntermann, J.: A method for taxonomy develop-
ment and its application in information systems. European Journal of Information
Systems22(3), 336–359 (2013).https://doi.org/10.1057/ejis.2012.26
21.Nikishina,I.,Sevgili,Ö.,Li,M.M.,Biemann,C.,Semmann,M.:Creatingataxonomy
for retrieval augmented generation applications. arXiv preprint arXiv:2408.02854
(2025)

22 Steve Nouyep, Sébastien Salva, and Maxime Puys
22.NIST: SP 800-30 rev. 1: Guide for conducting risk assessments. Tech. rep., National
Institute of Standards and Technology (2012)
23.OWASP: Top 10 for LLM applications 2025 (2025), https://owasp.org/www-pro
ject-top-10-for-large-language-model-applications/
24.Schröer, S.L., Apruzzese, G., Human, S., Laskov, P., et al.: SoK: On the offensive
potential of AI. In: IEEE Conference on Secure and Trustworthy Machine Learning
(SaTML). pp. 247–280 (2025). https://doi.org/10.1109/SaTML64287.2025.0002
1, arXiv:2412.18442
25.Unterkalmsteiner, M., Abdeen, W.: A compendium and evaluation of taxonomy
quality attributes. Expert Systems40(1), e13098 (2023). https://doi.org/10.1
111/exsy.13098
26.Wang, C., Li, H., Song, W., Lin, Y.: Retrieval-augmented generation: A survey
of security challenges and countermeasures. In: IEEE International Conference
on Privacy Computing and Data Security (PCDS). pp. 210–217 (2025). https:
//doi.org/10.1109/PCDS65695.2025.00037
27.Wang, S., Zhao, Y., Liu, Z., Zou, Q., Wang, H.: Demystifying LLM supply chain
vulnerabilities in the wild: Distribution, root cause, and real-world impact. In: Asia-
Pacific Symposium on Internetware (2026), arXiv:2502.12497; initially circulated as
“SoK: Understanding Vulnerabilities in the Large Language Model Supply Chain”
28.Wang, Y., Qu, W., Zhai, S., Jiang, Y., Liu, Z., Liu, Y., Dong, Y., Zhang, J.:
Silent leaks: Implicit knowledge extraction attack on RAG systems through benign
queries. In: International Conference on Learning Representations (ICLR) (2026),
arXiv:2505.15420;https://openreview.net/forum?id=zfVICPB5Sv
29.Zeng, S., Zhang, J., He, P., Xing, Y., Liu, Y., Xu, H., Ren, J., Wang, S., Yin, D.,
Chang, Y., Tang, J.: The good and the bad: Exploring privacy issues in retrieval-
augmented generation (RAG). In: Findings of the Association for Computational
Linguistics: ACL 2024. pp. 4505–4524 (2024). https://doi.org/10.18653/v1/20
24.findings-acl.267, arXiv:2402.16893
30.Zou, W., Geng, R., Wang, B., Jia, J.: PoisonedRAG: Knowledge corruption attacks
to retrieval-augmented generation of large language models. In: USENIX Security
Symposium (2025), arXiv:2402.07867
Supplementary Material
This supplementary material provides a full-catalog overview (Appendix A) and
additional tables referenced from the discussion (Appendix B). It is optional:
reviewers are not required to read it.
A Full Catalog Overview
Figure 4 renders the complete catalog in one view: the 3 RAG types, 8 surfaces,
38 attacks, 12 weaknesses, 41 risks, 3 CIA dimensions and 72 mitigations with
all their relations. Long node names are truncated to fit; full names, descriptions
and sources are available interactively in the web visualizer and as zoomable
vector exports in the supplementary material ( https://anonymous.4open.scie
nce/r/rag-mmsecu/).

A Security Meta-Model for Retrieval-Augmented Generation Systems 23
Fig. 4.The complete catalog (177 nodes), as rendered by the web visualizer under the default deployment configuration. Columns follow
the causal chain RAG type →Surface →Attack →Weakness / Risk →CIA, with Mitigations linked to the risks they mitigate. Markers:
“!” =coverage gap (risk without mitigation); “↑” =risk aggravated by the default configuration; dashed=not applicable to it.

24 Steve Nouyep, Sébastien Salva, and Maxime Puys
B Additional Tables
Table 10.Discriminant power of candidate segmentation criteria.
Criterion Total Shared Type-specific Power
Weakness (CWE) 12 12 0 None
Surface 8 6 2 Low
Attack 38 33 5 Medium
Risk41 36 5 Medium+CIA
Table 11.Persistent coverage gaps.
Risk Category Type Explanation
R17 (Graph poison.) POI GRAPH only No triplet validation
R33 (Silver noise) POI ALL No noise filtering
R34 (Inter-context) POI ALL No contradiction detection
R35 (Soft ad) POI ALL No semantic filtering
R36 (White DoS) DEG ALL No defense