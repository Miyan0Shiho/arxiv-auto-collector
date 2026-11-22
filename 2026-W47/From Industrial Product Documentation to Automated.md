# From Industrial Product Documentation to Automated Digital Product Passport Generation: An Experimental Study of Large Language Models

**Authors**: Marco Buecheler, Grant Richards, Nathan Hartman

**Published**: 2026-11-18 15:57:39

**PDF URL**: [https://arxiv.org/pdf/2611.24998v1](https://arxiv.org/pdf/2611.24998v1)

## Abstract
Manufacturing companies that are planning to sell certain products in the European Union (EU) will be required to provide mandatory digital product passports (DPPs). The use of DPPs is to provide standardized product information throughout multiple product life cycles in a circular economy. Required passport information, such as technical specifications, material composition, and environmental product footprint, is available in manufacturers' data sources but is scattered across different heterogeneous systems, documents, and data formats. The given paper investigates using large language models (LLMs) to automate the generation of DPPs based on the asset administration shell (AAS). A retrieval-augmented generation (RAG) approach is used to extract product information from industrial documentation and generate populated AAS instances. Product documentation is processed and divided into text chunks. The text chunks are then stored in a persistent ChromaDB vector database using vector embeddings. The retrieval process combines semantic and lexical search to identify relevant information based on standardized AAS submodel template requirements. The retrieved information is provided to five LLMs to generate structured AAS instances. The experiments cover different industrial products and AAS submodels. Four levels of prompt guidance are investigated to assess their influence on generation quality. The generated passports are evaluated against a ground truth. The experimental evaluation includes 4,560 executions across 76 product-submodel combinations. Of the generated passport outputs, 97.4% are valid JSON, while 44.9% pass the full AAS validation procedure. Under deterministic scoring, mean completeness, conditional accuracy, and overall field correctness are 49.6%, 58.3%, and 30.8%, respectively. An LLM-as-a-judge yields better scores of 66.5%, 88.9%, and 60.4% for the same outputs.

## Full Text


<!-- PDF content starts -->

Available online at www.sciencedirect.com
Manufacturing Letters
Preprint
55th SME North American Manufacturing Research Conference (NAMRC 55, 2027)
From Industrial Product Documentation to Automated Digital Product
Passport Generation: An Experimental Study of Large Language Models
Marco Buechelera,∗, Grant Richardsa, Nathan Hartmana
aPurdue University, School of Engineering Technology, Knoy Hall of Technology 150, 401 N Grant Street, West Lafayette, IN 47906, USA
Abstract
Manufacturing companies that are planning to sell certain products in the European Union (EU) will be required to provide mandatory digital
product passports (DPPs). The use of DPPs is to provide standardized product information throughout multiple product life cycles in a circular
economy. Required passport information, such as technical specifications, material composition, and environmental product footprint, is available
in manufacturers’ data sources but is scattered across different heterogeneous systems, documents, and data formats. Manually collecting and
structuring the required information for a DPP requires considerable effort, particularly when following standardized information models. The
given paper investigates using large language models (LLMs) to automate the generation of DPPs based on the asset administration shell (AAS).
A retrieval-augmented generation (RAG) approach is used to extract product information from industrial documentation and generate populated
AAS instances. Product documentation is processed and divided into text chunks. The text chunks are then stored in a persistent ChromaDB
vector database using vector embeddings. The retrieval process combines semantic and lexical search to identify relevant information based on
standardized AAS submodel template requirements. The retrieved information is provided to five LLMs to generate structured AAS instances.
The experiments cover different industrial products and AAS submodels, including ‘Digital Nameplate’, ‘Technical Data’, ‘Carbon Footprint’,
and battery-specific passport submodels. Four levels of prompt guidance are investigated to assess their influence on generation quality.
The generated passports are evaluated against a ground truth. The experimental evaluation includes 4,560 executions across 76 prod-
uct–submodel combinations, five LLMs, four prompt configurations, and three repetitions per condition. Of the generated passport outputs, 97.4%
are valid JSON, while 44.9% pass the full AAS validation procedure. Under deterministic scoring, mean completeness, conditional accuracy, and
overall field correctness (OFC) are 49.6%, 58.3%, and 30.8%, respectively. An LLM-as-a-judge yields better scores of 66.5%, 88.9%, and 60.4%
for the same outputs. Deterministic OFC increases from 13.4% with ‘Low’ prompt guidance to 30.2% when the target AAS template is provided
and approximately 40% with additional guidance.
©2027 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY-NC-ND license
(http://creativecommons.org/licenses/by-nc-nd/4.0/)
Peer-review under responsibility of the scientific committee of the NAMRI/SME.
Keywords:Digital Product Passport; Digital Battery Passport; Asset Administration Shell; Retrieval-Augmented Generation; Large Language Models; Circular
Economy
1. Introduction
Circular manufacturing aims to retain the value of products,
individual components, and materials [1]. Decisions throughout
these activities depend on the availability of product informa-
tion. Technical specifications may determine whether a prod-
∗Corresponding author. Tel.:+1 (765) 494-9099.
E-mail address:mbuechel@purdue.edu (Marco Buecheler).uct’s components can be reused elsewhere. Information on ma-
terial composition is useful in disassembly and recycling pro-
cesses. Environmental data, including product carbon footprint,
can be used to assess the environmental implications of product
life cycle activities. However, the needed product information
is often scattered across many heterogeneous data systems in
an organization. Moreover, the data originates from different
activities and stakeholders in a company’s value chain.
2213-8463©2027 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY-NC-ND license
(http://creativecommons.org/licenses/by-nc-nd/4.0/)
Peer-review under responsibility of the scientific committee of the NAMRI/SME.
arXiv:2611.24998v1  [cs.SE]  18 Nov 2026

M. Buecheler et al./Preprint2
Nomenclature
AAS Asset Administration Shell
CAD Computer-Aided Design
DBP Digital Battery Passport
DPP Digital Product Passport
DSR Design Science Research
ESPR Ecodesign for Sustainable Products Regulation
EU European Union
IDTA Industrial Digital Twin Association
LLM Large Language Model
OCR Optical Character Recognition
OFC Overall Field Correctness
RAG Retrieval-Augmented Generation
RQ Research Question
RRF Reciprocal Rank Fusion
To streamline the data for circular economy purposes, the
European Union (EU) will require companies selling prod-
ucts in the EU to provide digital product passports (DPPs).
These DPPs are product records that offer an approach to make
product data available throughout a product’s (multiple) life
cycle(s). The EU has introduced a regulatory framework for
DPPs through the ‘Ecodesign for Sustainable Products Regu-
lation (ESPR)’ [2]. The regulation establishes a framework for
product-specific information requirements intended to support
environmental sustainability and circularity.
The ‘EU Batteries Regulation’ establishes digital battery
passport (DBP) requirements for industrial batteries [3]. De-
pending on the product and applicable requirements, passports
may include information on product identification, technical
characteristics, material composition, environmental impacts,
and other life cycle-related information [4–6].
To meet the information requirements of a DPP or a DBP,
respectively, the required product documentation often needs
to be extracted from heterogeneous data sources. For many
companies, this process can be time-consuming and requires
manual effort as well as deep product knowledge.
Recent advances in large language models (LLMs) offer op-
portunities to automate the extraction and structuring of product
information required for DPPs. Previous research highlights the
ability of LLMs to interpret unstructured data and extract se-
mantically relevant information [7–9]. However, there are lim-
ited quantitative results on the performance of LLMs in the
field. Therefore, the given study evaluates the automated cre-
ation of DPPs with five different LLMs and multiple industrial
product documents from different manufacturers. Eight prod-
uct categories are part of the experimental design. To standard-
ize the passport creation process, the LLMs are prompted to
create DPPs based on the data structure of an asset administra-
tion shell (AAS). The study uses retrieval-augmented genera-
tion (RAG) to identify relevant product information in a vec-
tor database, with retrieval queries derived from standardized
AAS submodel requirements. The experiments cover differentstandardized AAS submodels, including ‘Digital Nameplate’,
‘Technical Data’, ‘Carbon Footprint’, and also battery-specific
passport submodels [10]. Four levels of prompt guidance are in-
vestigated to determine their influence on the generation results.
The generated AAS instances are evaluated for structural va-
lidity, information completeness, conditional accuracy, overall
field correctness (OFC), and consistency across repeated runs.
Particular attention is given to differences between submodels
that require technical, environmental, and material-related in-
formation.
As a result, the study wants to answer the following four
research questions (RQs) quantitatively:
•RQ1:To what extent can LLMs generate complete,
accurate, and structurally valid AAS instances from
heterogeneous industrial product documentation using
requirement-driven RAG for automated DPP generation?
•RQ2:How do LLM selection and prompt guidance affect
the quality of generated DPPs?
•RQ3:How does generation performance differ across in-
dustrial product categories and standardized AAS sub-
models?
•RQ4:How consistent are the generated AAS instances
across repeated runs under identical experimental condi-
tions?
The following Section 2 outlines how the current study re-
lates to already existing work and adds to the current knowl-
edge of research. Section 3 focuses on the methodology of the
given automated DPP generation pipeline, and Section 4 out-
lines the experimental design of the LLM evaluation study. The
exact evaluation metrics that are being evaluated in each of the
RQs are explained in Section 5. The findings of the study are
pointed out in Section 6. The last two Sections 7 and 8, discuss
and conclude the findings and outline needs for future research
activities.
2. Background and Related Work
Multiple research fields need to be considered for the given
paper. Therefore, the background and related work section out-
lines the state-of-the-art regarding DPP information exchange
requirements for circular manufacturing, data representations
for DPPs, and LLM-based DPP/AAS generation approaches
(including the storage and retrieval of product evidence). Lastly,
identified research gaps are summarized.
2.1. DPP Interoperability Needs for Circular Manufacturing
Product passports are digital data records that consolidate
product-relevant information generated across the entire value
chain, from raw material extraction through manufacturing,
use, and end-of-life [11–13]. Research on DPP systems iden-
tifies the collection and exchange of product life cycle informa-
tion as a central implementation challenge [6]. Industrial bat-
teries are the first product category where a passport will be

M. Buecheler et al./Preprint3
required [3]. The ‘Battery Pass Consortium’s’ content guide
translates battery-passport requirements into practical infor-
mation categories. These contributions establish that passport
preparation involves information from different activities and
stakeholders rather than a single product document [4].
2.2. Standardized Data Representations
The AAS provides a standardized approach for represent-
ing industrial assets [14]. An AAS identifies the represented
asset and references submodels that describe specific infor-
mation about that asset. The Industrial Digital Twin Associa-
tion (IDTA) publishes standardized AAS submodel templates
to support interoperability between different organizations and
systems. Each submodel contains elements such as properties,
collections, and references [15]. As a result, a product’s in-
formation is represented in a structured and machine-readable
form.
The information requirements and structures differ across
submodels. Product identification and manufacturer informa-
tion are recorded in the ‘Digital Nameplate’. The ‘Techni-
cal Data’ submodel describes technical product characteris-
tics(including standardized property definitions and units). En-
vironmental impact information is found in the ‘Carbon Foot-
print’ submodel. For batteries, the ‘Battery Material Composi-
tion’ submodel informs about battery materials and substances.
The following Section 2.3 reviews previous research on the
automated generation of AAS submodels using LLMs.
2.3. AAS generation with LLMs
Xia et al. [16] examine automated mapping between het-
erogeneous information models and standardized AAS rep-
resentations using semantic fingerprints. Their approach uses
language-model-based semantic representations to identify re-
lationships between source properties and standardized infor-
mation elements.
In subsequent work, Xia et al. [9] explore AAS genera-
tion using LLM agents and compare different models and
a retrieval-augmentation mechanism for ECLASS selection.
Their work establishes LLM-based transformation of technical
documentation into digital-twin representations.
Shi et al. [17] address the mapping of information between
existing industrial data systems and AAS representations. The
work uses semantic matching between source information and
standardized AAS property definitions. Additionally, the use
of classification systems such as ECLASS is included. The
approach is relevant to enhancing information interoperability,
as equivalent technical properties are often described using
different terminology in different stakeholders’ systems.
More recently, B ¨ucheler et al. [18] investigate the generation
of regulatory compliance artifacts using LLMs, focusing on
DBPs under the ‘ESPR’ and ‘EU Batteries Regulation’. Five
LLMs are evaluated using different levels of prompt guidance
and multiple repeated executions per experimental condition.For battery passports, the study uses an AAS ‘Battery Material
Composition’ submodel to evaluate the completeness and
consistency of generated information. The results show high
levels of required-field inclusion and consistency for most
models, with relatively small differences between prompt
configurations.
Groß and Heidrich [19] propose AAS-RAIL, which uses
previously modeled AAS instances to generate extraction
guidance for retrieving technical property values from product
datasheets. AAS-RAIL retrieves property-specific extraction
helpers generated from existing AAS instances and their as-
sociated datasheets. These helpers describe how values should
be identified and formatted. A benefit is that manufacturer-
specific conventions are taken into account without having to
fine-tune the LLM. Their evaluation focuses on the ‘Technical
Data’ submodel extraction and reports improvements over a
fixed few-shot baseline. A downside of the approach is that
constructing the retrieval requires previously modeled AAS
examples [19].
V ogel et al. [20] evaluate the extraction of technical prod-
uct information from manufacturer datasheets using LLMs
and standardized property definitions. The authors distinguish
between extracting available product properties followed by
semantic annotation and identifying values based on prede-
fined property definitions. The authors use the open-source
basyx-pdf-to-aaslibrary to extract technical properties
from manufacturer datasheets. They base their extraction on
predefined property definitions, such as those from ECLASS.
The extracted values are compared with the information con-
tained in previously populated AAS ‘Technical Data’ submod-
els from several manufacturers to evaluate extraction accuracy.
2.4. Summary of Identified Research Gaps
Previous research has demonstrated the feasibility of
using LLMs for technical information extraction and AAS
generation. However, the extent to which requirement-driven
RAG can generate complete, factually correct, and structurally
valid AAS instances from heterogeneous industrial product
documentation remains insufficiently investigated. Existing
approaches primarily focus on technical submodels. However,
DPP applications require information from various domains
such as product identification, material composition, and
environmental information. Consequently, further contribu-
tions must be distinguished by their product data retrieval
mechanisms rather than a general claim to introduce automated
AAS generation. This motivates the evaluation of requirement-
driven RAG for automated DPP generation effectiveness across
multiple standardized AAS submodels and industrial product
categories (RQ1).
Furthermore, the influence of LLM selection and prompt
guidance on generation quality across different AAS submod-
els has not been systematically established. Although previous
studies have compared LLMs and investigated prompting

M. Buecheler et al./Preprint4
strategies, it remains unclear whether their findings extend
to submodels with different information requirements and
structures across product categories. The present study there-
fore examines differences across LLM prompt configurations
(RQ2), as well as variations in generation performance across
product categories and standardized submodels (RQ3).
Finally, the reproducibility of generated AAS instances
requires further investigation. Previous evaluations have con-
sidered output consistency, but limited evidence is available
on the stability of field presence and generated values across
repeated executions of requirement-driven RAG for different
AAS submodels. The study therefore evaluates cross-run
consistency under identical experimental conditions (RQ4).
How the present study investigates the automated generation
of AAS-based DPP information across heterogeneous industrial
products and standardized submodels is explained in further de-
tail in the following Section 3.
3. Methodology
The study follows a design science research (DSR) approach
to guide the development and evaluation of a requirement-
driven RAG pipeline for automated DPP creation. In DSR,
practical problems are addressed through the systematic
development and evaluation of an artifact [21]. The artifact is
based on the current state of knowledge while also contributing
to new scientific results itself [21, 22].
The proposed approach transforms industrial product docu-
mentation into structured AAS representations for automated
DPP creation. It combines document ingestion, requirement-
driven retrieval (based on the specific AAS submodel un-
der investigation), standardized AAS-based passport generation
through an LLM, as well as a final output validation check.
The generated output is an ‘AAS 3.2 Environment’ serialized
in JSON format. Each generation task is defined by a selected
product, its associated source documentation, and a target stan-
dardized AAS submodel. The submodel determines both theinformation required for retrieval and the preferred structure of
the generated output.Figure1 shows the individual components
of the pipeline and how product data flows through the pipeline.
3.1. Product Data Sources and Data Preprocessing
The first stage of the pipeline preprocesses industrial doc-
uments and converts the data content into text suitable for in-
formation retrieval and LLM processing. The implementation
includes document readers for supported file formats, includ-
ing PDF and Word documents, spreadsheets, text and Mark-
down files, or specialized formats such as computer-aided de-
sign (CAD) files or drawings.
The tools PyMuPDF4LLM and PyMuPDF are primarily
used to process documents with extractable text. Optical Char-
acter Recognition (OCR) is available as a fallback when the
document does not contain sufficient extractable text. The goal
is to preserve relevant product information as written in the
source documents. Such information includes technical prop-
erties, numerical values, units, and other product descriptions.
For each experimental condition, the source documents are
assigned to the corresponding product.
3.2. Document Chunking, Embedding, and Vector Storage
Following preprocessing, the extracted document text is
divided into chunks with a target length of 1,400 charac-
ters and an overlap of 200 characters to preserve informa-
tion spanning chunk boundaries. Each chunk is converted into
a vector embedding using the sentence-transformers model
all-MiniLM-L6-v2. The resulting embeddings are stored in
a persistent ChromaDB vector database. The vector database
is configured for cosine-distance similarity retrieval. Each
database record contains the original text chunk and metadata
identifying its source document and chunk position.
The persistent database enables the indexed documentation
to be reused across experimental executions without repeat-
ing the embedding process for each LLM request. Retrieval
is restricted to the source documents assigned to the current
product to prevent the inclusion of unrelated product informa-
tion. To support the comparability of the experiments, the same
1.Documents2.Spread-sheets3.Plain text4.CAD and drawing files•PyMuPDF-4LLM  •PyMuPDF•OCR1. Product Data Sources & Data PreprocessingStandardized product data output by LLM4. LLM-based DPP/AAS Generation5. Output Validation & StorageValidation of created AAS standardized DPPs with python scripts1.Chunk product evidence2.Create embeddings3.Store in vector database1.Requirement aware retrieval2.Query by AAS field / submodel need3.Re-rank retrieved evidence chunks1.GPT-5.6 Sol2.Claude Fable 53.Llama 4 Scout 17B 16E Instruct4.Qwen-3.5-397B-A17B5.Mistral Large 4Generated AAS-based DPP environments with validation results
2. Document Chunking, Embedding & Vector Storage
PDF, DOCX, XLSX, MD, TXT,STEP, DXFTXTText chunks and vector embeddingsJSONPrompt + retrieved evidence in TXTRanked text chunks + metadataJSON + validation reportValidated JSONData format changes as product data flows through pipeline3. Requirement-driven Retrieval
Figure 1. Overview of the proposed DPP/AAS generation pipeline.

M. Buecheler et al./Preprint5
indexed documentation and retrieval configuration are main-
tained across LLMs and prompt conditions. If inputs exceed
all-MiniLM-L6-v2’sdefault limit of 256 word pieces, input
truncation occurs. Some chunks may exceed the limit as chunk
boundaries are defined in characters rather than tokens. As a
result, information is potentially excluded near the end of the
vector representation. The complete chunk text remains stored
in ChromaDB and accessible to lexical retrieval. The potential
impact of embedding truncation on retrieval performance will
also be evaluated by examining the retrieved evidence included
in the prompts provided to the LLMs.
3.3. Requirement-Driven Retrieval
The retrieval process begins with the standardized AAS sub-
model selected for the experiment. The pipeline identifies rele-
vant submodel elements and uses their identifiers to generate re-
trieval queries. The identifiers are converted into readable field
descriptions and combined with product-specific information.
For each selected field, the pipeline generates queries that in-
clude the product designation and the corresponding AAS field
name, together with field-specific queries that do not repeat the
product designation. Additional queries provide general prod-
uct and submodel information. The retrieval process combines
dense semantic search using the ChromaDB embeddings with
lexical retrieval based on BM25. Both methods produce candi-
date document chunks that may contain information relevant to
the selected submodel requirement. The candidate rankings are
combined using reciprocal rank fusion (RRF). This method as-
signs a combined ranking score based on the position of each
chunk in the individual retrieval lists. After rank fusion, a cross-
encoder reranking model, ms-marco-MiniLM-L6-v2, evaluates
the relevance of candidate chunks to the corresponding require-
ment. Unlike vector similarity search, the cross-encoder pro-
cesses the query and candidate text together to produce a rele-
vance score.
The retrieval configuration selects one main evidence chunk
for each requirement when a suitable candidate is available.
To preserve nearby information (such as table headings, units,
and explanatory text), preceding and following chunks are also
included. Duplicate chunks are removed when constructing
the final retrieved evidence. The retrieval process follows a
requirement-driven strategy. It does not select a fixed global
number of chunks for the complete submodel. Each submodel
requirement is used to retrieve relevant information. This is
intended to reduce the risk that frequently mentioned prod-
uct characteristics dominate the retrieved evidence while less
frequently documented properties are omitted. For applicable
‘Technical Data’ conditions, the implementation also incor-
porates ECLASS-related information to support standardized
technical property identification.
3.4. LLM-based DPP/AAS Generation
Following retrieval, the selected document chunks are pro-
vided to the LLM together with the generation instructions as-
signed to the experimental condition. The generation task re-quires the model to extract product-specific information and
create a populated AAS Environment in JSON format. The in-
structions specify that the generated environment must contain
the AAS and the target submodel, and that the AAS must refer-
ence the generated submodel. The models are instructed to use
only values supported by the provided product documentation
and not to invent missing information.
The information supplied to the model and the amount of
generation guidance vary according to the assigned prompt con-
figuration. The five investigated LLMs and four prompt config-
urations are described in Section 4.
3.5. Output Validation and Storage
The generated AAS environments are evaluated through sep-
arate validation and scoring procedures. The first stage deter-
mines whether the generated response can be parsed as a valid
JSON file. Successful parsing establishes syntactic validity but
does not establish compliance with the AAS metamodel. There-
fore, a subsequent validation step evaluates the generated pass-
port environment against an AAS 3.2 JSON schema and also
performs additional structural checks. These checks include the
consistency of AAS elements, references between the AAS and
its submodel, and uniqueness requirements for applicable iden-
tifiers. The generated outputs are retained together with their
execution metadata and validation results. Field-level quality
is evaluated using two scoring approaches applied to the same
4,560 generated outputs. The first approach is a determinis-
tic rule-based comparison against the reference AAS environ-
ments, and the second approach is an LLM-as-a-judge that re-
ports evaluated field presence and correctness separately. The
two evaluation approaches allow analysis of differences in scor-
ing outcomes without changing the generated outputs. For mal-
formed JSON responses containing recoverable field informa-
tion, the recovered fields are included in field-level scoring,
while JSON and AAS validity are assessed on the original out-
puts. The exact evaluation procedure is explained in Section 5.
4. Experimental Design
4.1. Product Dataset and AAS Submodels
The experimental dataset consists of industrial product docu-
mentation associated with different product categories and man-
ufacturers. A total of 35 products are selected, resulting in
76 product–submodel combinations across eight product cat-
egories. The investigated products include industrial batteries,
diesel engines, unmanned aerial vehicles, electrical compo-
nents, and electric motors. The product data is ingested in dif-
ferent document formats with different structures and levels of
information content (see Table 1 on Page 6).
Different standardized submodels are selected according
to the available product information and the intended pass-
port representation. The experiments include general ‘Digital
Nameplate’ and ‘Technical Data’ submodels, ‘Carbon Foot-
print’ submodels, and battery-specific submodels covering bat-

M. Buecheler et al./Preprint6
Table 1. Data formats tested in the pipeline
Category Formats
Documents PDF, DOCX
Spreadsheets XLSX
Plain text MD, TXT
CAD and drawing files STEP, DXF
tery nameplate information, technical characteristics, product
carbon footprint, and material composition.
Each product–submodel condition is associated with an
experiment definition identifying the product, the permitted
source documents, and the target AAS template. Table 2 gives
an overview of product categories, tested submodels, number of
product-submodel conditions, and resulting generation runs.
Table 2. Overview of experimental conditions
Product Category Submodels Tested Conditions Runs
Battery Battery Digital
Nameplate, Battery
Technical Data, Battery
Material Composition,
Battery Product Carbon
Footprint7 420
Car Digital Nameplate,
Carbon Footprint2 120
Electric Motor Digital Nameplate,
Technical Data2 120
Electrical
ComponentDigital Nameplate,
Technical Data, Carbon
Footprint44 2,640
Engine Digital Nameplate,
Technical Data, Carbon
Footprint13 780
Flight Controller Digital Nameplate,
Technical Data2 120
Unmanned Aerial
VehicleDigital Nameplate,
Technical Data4 240
Ventilation Unit Digital Nameplate,
Carbon Footprint2 120
Total 76 4,560
4.2. Experimental Factors
Five LLMs are evaluated using the outlined requirement-
driven RAG architecture. The selected LLMs include closed
and open-weight models and are shown in Table 3.
Table 3. Overview of evaluated LLMs
Name Developer Type
GPT-5.6 Sol OpenAI Closed
Claude Fable 5 Anthropic Closed
Llama 4 Scout Meta Open
Qwen3.5-397B-A17B Alibaba Open
Mistral Large 4 Mistral AI Open
The experiments investigate the influence of LLM selec-
tion and prompt guidance on the generated AAS instances. Thefour prompt configurations, ‘Low’, ‘Baseline’, ‘Medium’, and
‘High’, are applied to each model. Table 4 provides an overview
of the four prompt configurations that are being investigated.
Table 4. Context levels provided to the model in the generation task.
Level Information provided to the model
Low Generation task, product information, and re-
trieved document evidence. The full target sub-
model template is not provided.
Baseline Low-level inputs plus the target standardized AAS
submodel template.
Medium Baseline inputs plus the template’s accompanying
documentation.
High Medium-level inputs plus a written summary of the
AAS metamodel.
The full model and prompt configurations can be found in
the GitHub repository, which is linked in Appendix A.
Each experimental condition is executed three times to ex-
amine variability across repeated generations. The planned
experiment therefore comprises the product–submodel con-
ditions, five models, four prompt configurations, and three
repeated runs per condition. For the 76 registered product-
submodel conditions, this corresponds to 4,560 generation at-
tempts if all combinations are completed, as Eq. (1) shows.
N=N exp×N LLM×N prompt×N rep=76×5×4×3=4,560 (1)
4.3. Ground Truth Construction
The generated AAS instances are compared against manu-
ally prepared ground truth references. Each reference contains
the expected AAS and one target submodel populated using
the corresponding product documentation. The reference data
are associated with source-supported values and the relevant
AAS information structure. Required AAS elements are re-
tained (even without a source-supported value) according to the
cardinality constraints of the corresponding IDTA templates.
The evaluation distinguishes fields for which a source-
supported value is available from fields for which the reference
explicitly expects an empty value. This distinction is important
because a missing source value should not be replaced with an
invented value to increase the apparent completeness of the gen-
erated submodel.
The ground-truth environments are used to evaluate the pres-
ence, structure, and correctness of individual generated fields.
The exact scoring procedure is explained in Section 5.
4.4. Experimental Procedure
For each experimental condition, the applicable product doc-
umentation is retrieved from the persistent vector database us-
ing queries derived from the selected AAS submodel require-
ments (as explained in Section 3).
The retrieved evidence is combined with the assigned
prompt configuration and submitted to the selected LLM. The

M. Buecheler et al./Preprint7
generation process is repeated three times using the same ex-
perimental settings. Each response is saved together with its
execution metadata, including the selected model, prompt con-
figuration, and product–submodel condition. After generation,
the responses are processed by the validation and scoring pro-
cedures.
5. Evaluation Metrics
The generated AAS environments are evaluated for output
validity, information completeness, conditional accuracy, OFC,
and consistency across repeated executions. The two scoring
approaches used in the study are explained in the following
Subsection.
5.1. Deterministic and LLM-as-a-Judge Evaluation
Two independent scoring approaches are used to evaluate
the generated AAS instances against the manually constructed
ground-truth environments.
The first approach, deterministic scoring, relies on prede-
fined rules to match generated fields with the corresponding
reference fields and assess their correctness. Field presence
is determined through structural matching. Value correctness
is evaluated using rule-based comparisons of normalized
representations, including text, numerical values, dates, or
units. These comparisons follow predefined criteria.
The second approach, LLM-as-a-judge, uses a separate lan-
guage model (OpenAI’s GPT-6.1 Terra) to evaluate the gener-
ated fields against the ground-truth reference. The judge inde-
pendently determines whether each required field is present and
whether its generated value is correct. This approach can po-
tentially recognize semantically equivalent information that is
expressed differently from the reference. However, it may also
introduce variability or more permissive judgments [23]. The
judge’s decisions are independent of the deterministic scoring
results. Comparing the two approaches therefore provides in-
sight into the influence of the evaluation method on the mea-
sured quality of automatically generated AAS instances. Both
approaches use the same required reference fields and the same
mathematical definitions of completeness, conditional accu-
racy, and OFC, presented below.
5.2. Completeness Scoring
Completeness evaluates the proportion of required submodel
fields populated in a generated AAS instance. For a single exe-
cution, completenessCis calculated as
C=1
FFX
i=1δ(ˆvi),(2)whereFis the number of required fields, ˆv iis the generated
value of fieldi, and
δ(ˆvi)=1,if fieldiis populated with a non-null value,
0,otherwise.(3)
A populated value contributes to completeness irrespective
of whether the value is factually correct. Therefore, complete-
ness and accuracy are reported separately because a populated
field may contain an incorrect value.
5.3. Conditional Accuracy Scoring
Conditional accuracy assesses the correctness of populated
fields against the ground truth. Comparisons are carried out af-
ter appropriate normalization of text, dates, numeric formats,
and units. For a single execution, conditional accuracyAis de-
fined as
A=1
FAFAX
i=1δ(ˆvi,v∗
i),(4)
whereF Adenotes the number of ground-truth fields populated
in the generated AAS,v∗
iis the reference value, and
δ(ˆvi,v∗
i)=1,if the comparison criterion is met,
0,otherwise.(5)
All populated ground-truth fields are included in this calcu-
lation. Fields for which correctness cannot be established re-
ceive no correctness credit. Conditional accuracy is undefined
ifF A=0.
5.4. Overall Field Correctness Scoring
The OFC measures the proportion of all required fields that
are both populated and correctly extracted. Missing or incor-
rectly populated fields are counted as failures:
OFC=1
FFX
i=1δ(ˆvi,v∗
i),(6)
where the indicator in Eq. (6) is 1 only if the required field is
populated and its value satisfies the ground-truth comparison
criterion. When completeness and conditional accuracy refer to
the same set of required fields,
OFC=C×A.(7)

M. Buecheler et al./Preprint8
Thus, OFC combines completeness and conditional accu-
racy in one measure.
5.5. Consistency Across Repeated Runs
Consistency is evaluated fork=3 repeated executions of
the same experimental condition. Consistency is assessed using
cross-run field coverage, presence stability, and value consis-
tency.
For fieldi, cross-run coverageD iis
Di=1
kkX
j=1δ
ˆv(j)
i
,(8)
whereδ(ˆv(j)
i) is 1 if fieldiis populated with a non-null value in
runjand 0 otherwise.
Presence stabilityS iis
Si=1,D i=0 orD i=1,
0,otherwise.(9)
A field that is absent in all runs is therefore stable in its
absence, but has zero cross-run coverage. Value consistency is
evaluated among runs in which fieldiis complete and provides
a comparable normalized value. Letn idenote the number of
these runs. Value consistency is defined as
Vi=fmode,i
ni,n i≥2.(10)
Here,f mode,i is the frequency of the most common normalized
value. Missing or incomplete fields are not considered when
calculating value consistency. If fewer than two comparable
runs are available, value consistency is reported as not appli-
cable. The measures describe how consistently the model pro-
duces field values across repeated runs, irrespective of their cor-
rectness against the ground truth.
6. Results
6.1. DPP Generation and Structural AAS Validation
The experimental evaluation comprises 4,560 generation at-
tempts across 76 product–submodel combinations, five LLMs,
and four prompt configurations. Of these, 4,443 outputs
(97.4%) are syntactically valid JSON. A total of 2,205 out-
puts (48.4%) passed the AAS 3.2 schema validation, while
2,049 outputs (44.9%) passed the additional structural vali-
dation checks. The highest full-validator pass rate across all
prompt context levels is obtained by GPT-5.6 Sol (91.1%),
followed by Claude Fable 5 (66.9%) and Qwen 3.5 (53.4%).
Lower pass rates are observed for Mistral Large 4 (12.8%) andLlama 4 Scout (0.4%). Table 5 gives a summary of the valida-
tion results.
Table 5. Validation results of generated passports across all generated outputs.
ModelValid
JSONAAS
SchemaFull AAS
Validator
Claude Fable 5 100.0% 66.9% 66.9%
GPT-5.6 Sol 100.0% 91.3% 91.1%
Llama 4 Scout 100.0% 6.6% 0.4%
Mistral Large 4 87.3% 15.8% 12.8%
Qwen 3.5 99.9% 61.2% 53.4%
Overall 97.4% 48.4% 44.9%
The validation results also differ across prompt configura-
tions (see Table 6). GPT-5.6 Sol achieves 79.8% full AAS va-
lidity even under ‘Low’ guidance, suggesting possible prior fa-
miliarity with AAS structures. However, this cannot be con-
firmed without knowledge of its training data. Claude Fable
5 declines from ‘Medium’ context (90.4%) to ‘High’ context
(86.0%). Qwen 3.5 shows a similar decrease from 76.3% to
72.4%. The results suggest that extra guidance may introduce
competing instructions and reduce structural compliance. How-
ever, the found declines do not establish a causal relationship.
Table 6. Full AAS validator pass rates by LLM and prompt configuration.
Model Low Baseline Medium High Overall
Claude Fable 5 1.8% 89.5% 90.4% 86.0%66.9%
GPT-5.6 Sol 79.8% 96.1% 94.3% 94.3%91.1%
Llama 4 Scout 0.0% 1.8% 0.0% 0.0%0.4%
Mistral Large 4 0.9% 16.7% 15.8% 18.0%12.8%
Qwen 3.5 3.1% 61.8% 76.3% 72.4%53.4%
Overall 17.1% 53.2% 55.4% 54.1% 44.9%
To conclude, the differences between the three validation
stages indicate that syntactically valid JSON outputs do not nec-
essarily satisfy the structural requirements of an AAS Environ-
ment. Structural validity must therefore be evaluated separately
from the successful generation of JSON output.
6.2. Completeness, Conditional Accuracy and OFC
The generated instances are evaluated against the corre-
sponding ground truth AAS environments. Figure 2 shows the
aggregated completeness scores in % for the five models and
four prompt context configurations.
Low Baseline Medium High
Prompt context0%20%40%60%80%100%Completeness
Claude Fable 5
GPT-5.6 Sol
Llama 4 Scout
Mistral Large 4
Qwen3.5-397B-A17B
Figure 2. Completeness by LLM and prompt context.

M. Buecheler et al./Preprint9
Across all experimental conditions, the mean completeness
is 49.6%, showing that approximately half of the required
reference fields are populated. Across all generated AAS
environments, the mean additional field count is 26 fields
per run, representing fields not present in the corresponding
ground-truth environments.
Conditional accuracy among complete fields is 58.3%. This
indicates that slightly more than half of the evaluated values
are classified as correct by the deterministic scoring procedure.
Conditional accuracy results are shown in Figure 3.
Low Baseline Medium High
Prompt contextClaude
Fable 5
GPT-5.6
Sol
Llama 4
Scout
Mistral
Large 4
Qwen3.5-
397B-A17BModels51.9% 73.4% 73.9% 72.7%
49.0% 54.5% 70.3% 68.8%
35.2% 47.0% 41.7% 41.1%
26.8% 47.9% 65.9% 64.3%
44.7% 49.2% 64.4% 63.1%
0%20%40%60%80%100%
Score
Figure 3. Conditional accuracy by LLM and prompt context.
The mean OFC is 30.8% (see Table 7). This lower result re-
flects the combined influence of missing fields and incorrect or
unmatched values, providing a more comprehensive assessment
of generation performance than conditional accuracy alone.
Table 7. Mean OFC by LLM and prompt configuration.
Model Low Baseline Medium High Overall
Claude Fable 5 32.8% 52.1% 52.2% 51.7%47.2%
GPT-5.6 Sol 23.4% 34.7% 52.6% 52.1%40.7%
Llama 4 Scout 0.2% 8.9% 10.7% 9.7%7.4%
Mistral Large 4 5.4% 27.0% 39.4% 41.5%28.3%
Qwen 3.5 5.0% 28.2% 44.9% 44.4%30.6%
Overall 13.4% 30.2% 40.0% 39.9% 30.8%
6.3. Differences Between Submodels and Product Categories
Conditional accuracy is also compared across eight indus-
trial product categories and the investigated AAS submodels.Figure 4 shows the results by product category and LLM.
Claude Fable 5 achieved the highest conditional accuracy
for cars (90.8%), while GPT-5.6 Sol achieved its highest
value for ventilation products (88.1%). Across the five mod-
els, lower values are generally observed for diesel engines,
electric motors, and electrical components. Nevertheless, the
pattern differs between models as well. For example, Qwen
3.5 achieves 77.0% for batteries but only 45.6% for flight
controllers.
Table 8 presents conditional accuracy for the general AAS
submodel types. For ‘Carbon Footprint’, the highest values
are obtained by GPT-5.6 Sol (99.1%), Qwen 3.5 (89.1%), and
Claude Fable 5 (91.7%). The results for ‘Technical Data’ are
lower, ranging from 40.3% for Llama 4 Scout to 57.3% for
Claude Fable 5.
Table 8. Mean conditional accuracy by LLM and general AAS submodel.
ModelCarbon
FootprintDigital
NameplateTechnical
Data Overall
Claude Fable 5 91.7% 74.9% 57.3%67.9%
GPT-5.6 Sol 99.1% 61.8% 52.3%59.7%
Llama 4 Scout 51.9% 44.3% 40.3%42.8%
Mistral Large 4 76.4% 51.8% 48.0%51.5%
Qwen 3.5 89.1% 54.9% 53.3%56.3%
Overall 86.1% 59.1% 50.8% 56.9%
Table 9 presents conditional accuracy for the battery AAS
submodel types. The battery-specific submodels exhibited sub-
stantial variation between models. For example, Llama 4 Scout
achieved 97.4% for ‘Battery Material Composition’ but 0.0%
for ‘Battery Digital Nameplate’, while Mistral Large 4 reached
100.0% for ‘Battery Product Carbon Footprint’.
Table 9. Mean conditional accuracy by LLM and battery AAS submodel.
ModelBattery
Material
CompositionBattery
Digital
NameplateBattery
Product
Carbon
FootprintBattery
Technical
Data Overall
Claude Fable 5 82.0% 95.0% 33.3% 54.8%71.0%
GPT-5.6 Sol 77.2% 77.3% 72.2% 73.3%75.4%
Llama 4 Scout 97.4% 0.0% 33.3% 42.0%52.0%
Mistral Large 4 71.7% 77.1% 100.0% 66.2%75.7%
Qwen 3.5 92.0% 80.3% 75.0% 59.7%77.0%
Overall 82.8% 76.5% 62.5% 59.8% 71.5%
Battery Car Diesel Electric Motor Electrical
ComponentFlight
ControllerUAV Ventilation
Product categoriesClaude
Fable 5
GPT-5.6
Sol
Llama 4
Scout
Mistral
Large 4
Qwen3.5-
397B-A17BModels71.0% 90.8% 73.6% 71.3% 64.0% 69.0% 81.2% 70.8%
75.4% 76.6% 58.7% 63.8% 56.9% 54.6% 75.0% 88.1%
52.0% 55.6% 33.7% 20.2% 45.0% 51.0% 54.0% 41.7%
75.7% 70.1% 58.2% 44.4% 47.5% 51.8% 67.3% 59.0%
77.0% 74.9% 62.4% 58.3% 53.5% 45.5% 57.3% 70.6%
0%20%40%60%80%100%
Score
Figure 4. Mean conditional accuracy by LLM and industrial product category.

M. Buecheler et al./Preprint10
6.4. Consistency Across Repeated Runs
Consistency is assessed using cross-run field coverage,
presence stability, and value consistency. The analysis iden-
tifies fields that are consistently populated, fields that are
repeatedly omitted, and fields for which generated values
varied across repeated executions. These results complement
the field-correctness evaluation, as repeated generation of the
same value does not necessarily mean that the value is correct.
Mean cross-run field coverage is 49.6%, with required fields
populated in approximately half of the repeated executions.
GPT-5.6 Sol achieves the highest average coverage (65.4%),
followed by Claude Fable 5 (64.6%), while Llama 4 Scout
shows substantially lower coverage (18.8%). Table 10 summa-
rizes the cross-run field coverage results.
Table 10. Mean cross-run field coverage by LLM and prompt configuration.
Model Low Baseline Medium High Overall
Claude Fable 5 54.3% 68.2% 67.7% 68.3%64.6%
GPT-5.6 Sol 51.2% 66.1% 72.2% 72.0%65.4%
Llama 4 Scout 0.4% 20.2% 28.9% 25.7%18.8%
Mistral Large 4 15.9% 59.1% 57.2% 61.6%48.5%
Qwen 3.5 9.7% 60.3% 66.0% 67.4%50.8%
Overall 26.3% 54.8% 58.4% 59.0% 49.6%
Mean field presence stability is 90.5% across all experi-
mental conditions, indicating that most required fields are ei-
ther consistently populated or consistently omitted across the
three runs. The highest stability is observed for Claude Fable 5
(95.5%) and Llama 4 Scout (95.1%) (see Table 11). However,
high stability does not mean successful information extraction.
Table 11. Field stability by LLM and prompt configuration.
Model Low Baseline Medium High Overall
Claude Fable 5 95.8% 96.4% 95.7% 94.1%95.5%
GPT-5.6 Sol 85.1% 87.6% 93.9% 93.1%89.9%
Llama 4 Scout 99.0% 93.5% 93.2% 94.6%95.1%
Mistral Large 4 79.7% 85.5% 70.7% 78.1%78.5%
Qwen 3.5 89.1% 93.9% 95.1% 95.6%93.4%
Overall 89.7% 91.4% 89.7% 91.1% 90.5%
Mean value consistency is 93.1% across eligible experimen-
tal conditions (see Figure 5).
Low Baseline Medium High
Prompt contextClaude
Fable 5
GPT-5.6
Sol
Llama 4
Scout
Mistral
Large 4
Qwen3.5-
397B-A17BModels90.0% 93.6% 93.7% 92.8%
89.0% 93.2% 93.6% 94.0%
100.0% 97.4% 97.1% 99.3%
94.0% 90.5% 90.8% 90.3%
94.7% 93.1% 92.2% 92.6%
0%20%40%60%80%100%
Score
Figure 5. Value consistency by LLM and prompt context.Scores range from 89.0% to 100.0%. This indicates that
comparable populated fields generally contained the same val-
ues across repeated executions. Llama 4 Scout achieved the
highest consistency across all prompt configurations. The result
does not necessarily indicate high completeness or accuracy.
6.5. Deterministic and LLM-as-a-Judge Scoring Differences
The LLM-as-a-judge procedure returned higher overall com-
pleteness (66.5%, compared with 49.6% under deterministic
scoring), conditional accuracy (88.9% versus 58.3%), and OFC
(60.4% versus 30.8%). The OFC difference is 29.6 percentage
points for the same 4,560 generation attempts (also see Fig-
ure 6). In the paired run-level comparison, judge-based OFC
is higher in 3,807 cases, equal in 744, and lower in nine. The
scores therefore reflect substantial differences between the eval-
uators rather than changes to the generated AAS instances.
Strict deterministic field matching may underestimate extrac-
tion quality (semantically equivalent properties may be re-
jected). However, without further human validation, the higher
LLM-as-a-judge scores cannot be considered more accurate.
The retrieval evaluation achieved an evidence recall of 98%,
indicating that most source-supported ground-truth values are
represented in the retrieved context.
Claude
Fable 5GPT-5.6
SolLlama 4
ScoutMistral
Large 4Qwen3.5-
397B-A17B
Models0%20%40%60%80%100%Overall field correctness (OFC)Deterministic
LLM judge
Figure 6. OFC Deterministic and LLM-as-a-Judge.
6.6. Prompt Reduction and Token Consumption
The requirement-driven RAG approach reduces prompt
length by selecting relevant information instead of providing
the complete source documentation (see Figure 7).
Low Baseline Medium High
Prompt context0%20%40%60%80%100%Prompt character reduction
Figure 7. Weighted mean prompt character reduction by prompt context.
The weighted mean prompt character reduction ranged from
26.2% for ‘High’ to 61.4% for ‘Low’ guidance. ‘Baseline’

M. Buecheler et al./Preprint11
and ‘Medium’ achieved reductions of 31.4% and 26.6%,
respectively. The product documentation used in the given
study largely fit within the context windows of the evaluated
LLMs. However, industrial applications involving extensive
product documentation and life cycle data may exceed these
limits. Requirement-driven RAG therefore provides a scalable
approach by restricting the supplied context to information rel-
evant to the target AAS submodel. By reducing the amount of
supplied information, RAG can help lower token consumption
and associated inference costs.
The token analysis additionally reveals differences be-
tween LLMs, with Claude Fable 5 consuming approximately
69,000 total tokens per execution, compared with approxi-
mately 39,000 for Llama 4 Scout (see Figure 8).
Claude
Fable 5GPT-5.6
SolLlama 4
ScoutMistral
Large 4Qwen3.5-
397B-A17B
Models010,00020,00030,00040,00050,00060,00070,000Mean total tokens per run
Figure 8. Mean total tokens per run by LLM.
7. Discussion
7.1. Automated DPP Creation in a Circular Economy
The experimental results provide evidence concerning the
extent to which existing industrial documentation can be used
to generate standardized AAS-based product information.
The proposed approach reduces the dependence on manual
extraction and mapping by using standardized submodel
requirements to guide information retrieval and generation. It
is therefore applicable to organizations that possess product
documentation but have not yet established repositories of
populated AAS instances.
However, automatically generated AAS instances are only
one part of a functioning DPP infrastructure. Their use in circu-
lar manufacturing also requires reliable product identification,
data traceability, access control, and life cycle information man-
agement.
7.2. Product Data Origins
Product information can originate from various systems and
have heterogeneous formats. The implemented pipeline retains
information about retrieved source documents and document
chunks, but does not establish complete traceability to the au-
thoritative information source. For industrial deployment, addi-
tional mechanisms would be needed to maintain relationships
between generated AAS values, original source records, prod-
uct revisions, and subsequent life cycle updates.7.3. Limitations
First, the evaluation is limited to the selected industrial prod-
ucts, source documents, and standardized AAS submodels. De-
spite covering eight product categories, the unbalanced dataset
limits the generalizability of the findings. Additionally, the
completeness and accuracy metrics depend on the correctness
of the manually created ground truths.
Second, the pipeline only processes static documents. The
experiments consequently do not assess live data exchange,
continuous passport updates, or the handling of life cycle in-
formation generated during manufacturing.
Third, the evaluation methods used differ. Deterministic
scoring is reproducible but may fail to recognize semantically
equivalent values. An LLM-as-a-judge allows for more flexible
comparisons but may introduce less reproducible judgments.
Finally, the study does not establish whether requirement-
driven RAG outperforms alternative retrieval approaches.
8. Conclusion
The paper showed a pipeline to automatically generate
AAS-based DPPs and DBPs with the help of LLMs. The
results provide a basis for assessing the suitability of the inves-
tigated LLMs for preparing standardized product passports.
Across 4,560 generations, 97.4% of responses are valid
JSON, but only 44.9% passed the full AAS validator. Under
deterministic scoring, mean completeness, conditional accu-
racy, and OFC are 49.6%, 58.3%, and 30.8%, respectively. The
corresponding LLM-as-a-judge scores are 66.5%, 88.9%, and
60.4%. The 29.6 percentage-point difference in mean OFC
between deterministic and LLM-based scoring shows that
evaluation design influences reported performance.
Four prompt context levels show differences in the creation
performance of the passports under investigation. Deterministic
OFC increases from 13.4% for Low guidance to 30.2% for
‘Baseline’ and approximately 40% at ‘Medium’ and ‘High’,
indicating that providing the AAS template is associated
with better field-level results without establishing statistical
significance. Claude Fable 5 achieves the highest deterministic
OFC (47.2%), while GPT-5.6 Sol achieves the highest full AAS
validator pass rate (91.1%). Field presence stability averages
90.5%, but is also increased by consistently absent fields. The
value consistency score averages 93.1%.
For circular manufacturing, the pipeline can help turn
scattered product data into standardized DPPs that inform com-
ponent reuse and material recovery. The use of standardized
AAS submodels gives product stakeholders an interoperable
product information exchange format across life cycles.
Future work should use more balanced datasets and link
DPP fields to authoritative product records. These links can aid
LLM-assisted DPP updates after products leave a factory.

M. Buecheler et al./Preprint12
Appendix A. Supplementary Research Materials
The supplementary research materials are available at:
https://github.com/mbuecheler/
automated-LLM-generated-DPPs
References
[1] P. Morseletto, Targets for a circular economy, Resources, Conservation
and Recycling 153 (2020) 104553.
doi:10.1016/j.resconrec.2019.104553.
[2] European Commission, Regulation (EU) 2024/1781 of the European
Parliament and of the Council of 13 June 2024 establishing a framework
for the setting of ecodesign requirements for sustainable products,
amending Directive (EU) 2020/1828 and Regulation (EU) 2023/1542 and
repealing Directive 2009/125/EC (Text with EEA relevance) (2024) [cited
2026-05-02].
URLhttps://eur-lex.europa.eu/eli/reg/2024/1781/oj/eng
[3] European Commission, Regulation (EU) 2023/1542 of the European
Parliament and of the Council of 12 July 2023 concerning batteries and
waste batteries, amending Directive 2008/98/EC and Regulation (EU)
2019/1020 and repealing Directive 2006/66/EC (Text with EEA
relevance) (2023) [cited 2026-02-06].
URLhttps://eur-lex.europa.eu/eli/reg/2023/1542/oj
[4] Battery Pass Consortium, Battery Passport Content Guidance, version 1.1
(2023) [cited 2026-04-30].
URLhttps:
//thebatterypass.eu/assets/images/content-guidance/pdf/
2023_Battery_Passport_Content_Guidance.pdf
[5] European Commission, Guidance Document: Digital Batteries Passport –
data points by category (2026) [cited 2026-11-02].
URLhttps://single-market-economy.ec.europa.eu/
document/download/cd1e5e6c-4a4a-4b99-995a-49eb6916187e_
en?filename=Digital%20Batteries%20Passport%20-%20data%
20point%20by%20category.pdf
[6] M. Jansen, T. Meisen, C. Plociennik, H. Berg, A. Pomp, W. Windholz,
Stop Guessing in the Dark: Identified Requirements for Digital Product
Passport Systems, Systems 11 (3) (2023).
doi:10.3390/systems11030123.
[7] P. Chen, Y . Zhang, M. Cafarella, D. Roth, Can we Retrieve Everything All
at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method
(2025).doi:10.48550/arXiv.2501.18539.
[8] A. Menon, O. Serban, An Automated LLM-based Pipeline for
Asset-Level Database Creation to Assess Deforestation Impact, in:
K. Dutia, P. Henderson, M. Leippold, C. Manning, G. Morio,
V . Muccione, J. Ni, T. Schimanski, D. Stammbach, A. Singh, A. R. Su,
S. A. Vaghefi (Eds.), Proceedings of the 2nd Workshop on Natural
Language Processing Meets Climate Change (ClimateNLP 2025),
Association for Computational Linguistics, Vienna, Austria, 2025, pp.
142–167.doi:10.48550/arXiv.2505.05494.
[9] Y . Xia, Z. Xiao, N. Jazdi, M. Weyrich, Generation of Asset
Administration Shell With Large Language Model Agents: Toward
Semantic Interoperability in Digital Twins in the Context of Industry 4.0,
IEEE Access 12 (2024) 84863–84877.
doi:10.1109/ACCESS.2024.3415470.
[10] Industrial Digital Twin Association e.V ., Registered AAS Submodel
Templates (2026) [cited 2026-07-06].
URLhttps:
//industrialdigitaltwin.org/en/content-hub/submodels
[11] T. Adisorn, L. Tholen, T. G ¨otz, Towards a Digital Product Passport Fit for
Contributing to a Circular Economy, Energies 14 (8) (2021) 2289.
doi:10.3390/en14082289.
[12] J. Walden, A. Steinbrecher, M. Marinkovic, Digital Product Passports as
Enabler of the Circular Economy, Chemie Ingenieur Technik 93 (11)
(2021) 1717–1727.doi:10.1002/cite.202100121.[13] M. Jansen, B. Gerstenberger, J. Bitter-Krahe, H. Berg, J. Sebesty ´en,
J. Schneider, Current Approaches to the Digital Product Passport for a
Circular Economy, Tech. Rep. 198, Wuppertal Institut f ¨ur Klima, Umwelt,
Energie, Wuppertal, publication Title: Wuppertal paper V olume: 198
(2022).doi:10.48506/opus-8042.
[14] A. Alexopoulos, G. Kalogeras, K. Koutras, A. Kalogeras, Why Asset
Administration Shells: A Survey on Uses and Challenges, IEEE Access
13 (2025) 126582–126609 [cited 2026-02-05].
doi:10.1109/ACCESS.2025.3589931.
URLhttps://ieeexplore.ieee.org/document/11082140
[15] Industrial Digital Twin Association e.V ., Specification of the Asset
Administration Shell Part 1: Metamodel, IDTA Number: 01001, Version:
3.2 (2026) [cited 2026-09-18].
URLhttps://industrialdigitaltwin.io/
aas-specifications/IDTA-01001/v3.2/part-1-metamodel.pdf
[16] Y . Xia, N. Jazdi, M. Weyrich, Automated generation of Asset
Administration Shell: a transfer learning approach with neural language
model and semantic fingerprints, in: 2022 IEEE 27th International
Conference on Emerging Technologies and Factory Automation (ETFA),
2022, pp. 1–4.doi:10.1109/ETFA52439.2022.9921637.
[17] D. Shi, O. Meyer, M. Oberle, T. Bauernhansl, Dual data mapping with
fine-tuned large language models and asset administration shells toward
interoperable knowledge representation, Robotics and
Computer-Integrated Manufacturing 91 (2025) 102837.
doi:10.1016/j.rcim.2024.102837.
[18] M. B ¨ucheler, A. Watson, G. Richards, From Regulation to
Implementation: A Critical Evaluation of LLM-Assisted Regulatory
Compliance in Industry (2026).doi:10.48550/arXiv.2608.21317.
[19] J. Groß, J. Heidrich, AAS-RAIL: Improving Information Extraction for
Asset Administration Shells through Retrieval-Augmented In-Context
Learning (Sep. 2026).doi:10.48550/arXiv.2609.07334.
[20] J. A. V ogel, I. Garmaev, M. Ristin, C. Barth, A. Bayha, S. Gr ¨uner,
N. Braunisch, Extraction of technical data using LLMs Experience Report
and Evaluation based on Asset Administration Shells, Tech. rep., VDI
Verlag GmbH, conference Name: 26. Leitkongress der Mess- und
Automatisierungstechnik (2025).
doi:10.51202/9783181024577-153.
[21] A. Hevner, A Three Cycle View of Design Science Research,
Scandinavian Journal of Information Systems 19 (Jan. 2007) [cited
2026-02-04].
URLhttps://aisel.aisnet.org/sjis/vol19/iss2/4/
[22] A. Hevner, A. R, S. March, S. T, Park, J. Park, Ram, Sudha, Design
Science in Information Systems Research, Management Information
Systems Quarterly 28 (2004) 75 [cited 2026-02-04].
URLhttps://wise.vub.ac.be/sites/default/files/thesis_
info/design_science.pdf
[23] A. S. Thakur, K. Choudhary, V . S. Ramayapally, S. Vaidyanathan,
D. Hupkes, Judging the Judges: Evaluating Alignment and Vulnerabilities
in LLMs-as-Judges (Aug. 2025).doi:10.48550/arXiv.2406.12624.