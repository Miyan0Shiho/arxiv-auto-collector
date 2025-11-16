# fastbmRAG: A Fast Graph-Based RAG Framework for Efficient Processing of Large-Scale Biomedical Literature

**Authors**: Guofeng Meng, Li Shen, Qiuyan Zhong, Wei Wang, Haizhou Zhang, Xiaozhen Wang

**Published**: 2025-11-13 06:31:16

**PDF URL**: [https://arxiv.org/pdf/2511.10014v1](https://arxiv.org/pdf/2511.10014v1)

## Abstract
Large language models (LLMs) are rapidly transforming various domains, including biomedicine and healthcare, and demonstrate remarkable potential from scientific research to new drug discovery. Graph-based retrieval-augmented generation (RAG) systems, as a useful application of LLMs, can improve contextual reasoning through structured entity and relationship identification from long-context knowledge, e.g. biomedical literature. Even though many advantages over naive RAGs, most of graph-based RAGs are computationally intensive, which limits their application to large-scale dataset. To address this issue, we introduce fastbmRAG, an fast graph-based RAG optimized for biomedical literature. Utilizing well organized structure of biomedical papers, fastbmRAG divides the construction of knowledge graph into two stages, first drafting graphs using abstracts; and second, refining them using main texts guided by vector-based entity linking, which minimizes redundancy and computational load. Our evaluations demonstrate that fastbmRAG is over 10x faster than existing graph-RAG tools and achieve superior coverage and accuracy to input knowledge. FastbmRAG provides a fast solution for quickly understanding, summarizing, and answering questions about biomedical literature on a large scale. FastbmRAG is public available in https://github.com/menggf/fastbmRAG.

## Full Text


<!-- PDF content starts -->

fastbmRAG: A Fast Graph-Based RAG Framework for Efficient Processing of
Large-Scale Biomedical Literature
Guofeng Meng1, Li Shen1, Qiuyan Zhong1, Wei Wang1, Haizhou Zhang1, and Xiaozhen Wang1
1Changchun GeneScience Pharmaceuticals Co., Ltd. Shanghai
Abstract
Large language models (LLMs) are rapidly transforming various domains, including biomedicine and healthcare, and
demonstrate remarkable potential from scientific research to new drug discovery. Graph-based retrieval-augmented gener-
ation (RAG) systems, as a useful application of LLMs, can improve contextual reasoning through structured entity and
relationship identification from long-context knowledge, e.g. biomedical literature. Even though many advantages over
naive RAGs, most of graph-based RAGs are computationally intensive, which limits their application to large-scale dataset.
To address this issue, we introduce fastbmRAG, anfastgraph-basedRAGoptimized forbiomedicalliterature. Utilizing
well organized structure of biomedical papers, fastbmRAG divides the construction of knowledge graph into two stages,
first drafting graphs using abstracts; and second, refining them using main texts guided by vector-based entity linking,
which minimizes redundancy and computational load. Our evaluations demonstrate that fastbmRAG is over 10x faster
than existing graph-RAG tools and achieve superior coverage and accuracy to input knowledge. FastbmRAG provides a
fast solution for quickly understanding, summarizing, and answering questions about biomedical literature on a large scale.
FastbmRAG is public available inhttps://github.com/menggf/fastbmRAG.
1 Introduction
Large language models (LLMs), such as GPT-4, have demonstrated their remarkable capabilities in understanding and
generating natural language with human-like performance [1]. These models are extensively trained on massive amounts
of knowledge and are capable to generate expert-like responses in various domains [2]. Now, LLMs has been widely used
in biomedical research by enhancing data analysis, automating literature reviews, and improving clinical decision-making
[3, 4]. This significantly accelerates the efficiency of researchers to acquire interest knowledge [5]. However, LLMs are also
challenged by incomplete or even invalid responses, known as the ”hallucination” due to the biases inherent in their training
data [6, 7, 8].
Retrieval-augmented generation (RAG) is a technique that enables LLMs to retrieve and incorporate new information
into their responses [9]. RAG systems retrieve relevant documents or messages from some source data instead of requiring the
model to memorize all knowledge, allowing them to generate more grounded and accurate responses [10]. This is especially
useful to improve the performance of LLMs to handle domain-specific knowledge [11, 12]. However, performance of naive
RAG approaches typically relies on proper chunk partitioning of documents and effective retrieval of related chunks according
user’s queries [13, 14]. When it comes to large-scale domain-specific knowledge, naive RAG often retrieves irrelevant or low-
quality data chunks (low precision) or fails to retrieve all relevant information (low recall) [15, 16]. This limits the application
of naive RAG systems to integrate too much knowledge. In biomedical domain, there are millions of published literatures,
which usually contain large amounts of fragmented information. Comprehensive integration is more necessary than simple
retrieval.
Recent advances in graph-based retrieval augmented with generation (RAG), such as GraphRAG [17] and LightRAG [18],
have introduced graph-based representations to organize the entities (e.g., genes or diseases) and their relationships. These
systems use knowledge graphs to improve contextual understanding and reasoning. This results in more comprehensive,
accurate, and explainable AI responses, especially for complex queries and the domains with intricate relationships [19, 20].
Compared to naive RAG, graph-based RAG can integrate information from multiple documents seamlessly and uncover
hidden insights and connections among them [21]. LightRAG, a lightweight model, has been optimized for retrieval accuracy
and efficiency and achieves a better performance. However, LightRAG still faces limitations when scaling to large biomedical
data. The graph construction and inference steps are often computationally expensive and not optimized for domain-specific
document structures [22]. This limits the application of graph-based RAG tools in scientific research.
To address these challenges, we present afastgraph-basedRAGtool designed for processing large-scalebiomedical
literature, named asfastbmRAG. It utilizes a similar framework of LightRAG by incorporates graph structures into text
indexing and retrieval processes. The novel concept of fastbmRAG is to construct knowledge graphs in two steps: first,
building a draft knowledge graph using only abstracts, and second, refining the relationship description of graph using both
the entity relationships of the draft graph and the main texts of scientific papers. In the latter step, a vector-based search
1arXiv:2511.10014v1  [q-bio.QM]  13 Nov 2025

links entities in the main text, reducing the computational requirements and keeping the input chunk size within a proper
range. FastbmRAG greatly reduces redundancy and accelerates information extraction in biomedical paper processing.
Our evaluation shows that fastbmRAG is more than 10 times faster than LightRAG. using large-scale disease-related papers,
fastbmRAG generate outputs with better coverage to disease knowledge and better description accuracy of related biomedical
papers. Overall, fastbmRAG offers a fast solution for quickly understanding, summarizing, and answering questions about
biomedical literature on a large scale.
2 Methods
2.1 Disease-related papers
We searched the PubMed database for disease names using queries such as ”endometriosis [title/abstract] AND free full text
[sb]” and downloaded the papers’ PMIDs. We downloaded the full texts of these papers from the Europe PMC FTP site in
XML format. We then processed the XML files using the tidyPMC package. We extracted the sections of ’title’, ’authors’,
’journal’, ’paper ID’, ’year’, ’abstract’, and ’main text’. Here, ’paper ID’ can be a PMC ID or a PMID, and ’main text’ may
contain different sections. We stored this information in a Pandas DataFrame, with each paper as a row. The data frame
containing all the papers is written to a text file.
2.2 LLM model
Under the default settings, fastbmRAG uses Ollama (v0.7.1) as the LLM backend. The Phi4 (16B) model, developed by
Microsoft Research, was used to extract entities and their relationships from the input text using prompts. The ’mxbai-
embed-large’ embedding model was used to generate the embedded vector of the main text chunks or queries. Vector similarity
was calculated using cosine similarity.
2.3 Implement of fastbmrag and LightRAG
Two models were implemented on an Ubuntu server with one RTX 4090 GPU (24 GB) and 1 TB of ROM. LightRAG (v1.3.8)
was cloned from GitHub and installed locally. The demo script in the example directory was modified for ollama to ensure
that LightRAG uses the same LLM and embedding model as fastbmRAG. The other parameters are set as the default. The
main texts of all the papers were read into a Python DataFrame, and then they were indexed using the ’insert’ function.
2.4 Knowledge graph and vector database
The knowledge graph is generated through the retrieval of entities and relationships from scientific papers. All edges are
directional, with a source node and a target node. The source and target nodes are the source and target entities generated
by the LLM. Each node has an ’entity type’ attribute, such as ’gene,’ ’disease,’ ’mutation,’ ’drug,’ and other types. Node
weights are calculated based on the nodes’ degree in the knowledge graph and normalized according to the total number
of nodes. Edges have two attributes: ’relationship type’ and ’relationship description’. The ’relationship type’ attribute is
used during global queries, while the ’relationship description’ attribute records the most important information about entity
pairs. The contents of the ”relationship description” attribute are embedded into vectors and used for similarity searching
of queries. Edge weights are calculated based on node weights. By default, edge weight is calculated as the minimum weight
of the source and target nodes. The edge weight indicates the confidence in the entity relationship. There are also other
edge attributes, e.g., paper id, journal, authors, and other related information. The knowledge graph is stored in a vector
database, e.g., Qdrant.
2.5 Query the knowledge graph
Unlike other graph-based RAGs, fastbmRAG does not rely on users’ specifications for global or local retrieval. Instead, our
tool uses an LLM to parse the input query and extract information such as gene or disease names and potential output types.
These correspond to the source or target node names and the ’relationship type’ edge attribute in the knowledge graph.
These can then be used to filter out irrelevant edges. This helps control the scale of the subgraph for semantic searches.
Additionally, due to the existence of numerous unvalidated relationships, users can optionally set a threshold for edge weights
to further filter low-confidence edges. This can further reduce the scale of the subgraph.
The query is performed by searching for vector similarity between the embedded query vector and the database records for
the ’relationship description’ of the edges. To ensure sufficient coverage of scientific papers, fastbmRAG allows the selection
of an unlimited number of matched descriptions or papers. The outputs from each matched description are summarized into
one integrated answer.
2

Figure 1: The flowchart of fastbmRAG to build knowledge graph
2.6 Answer generation and quality assessment
We applied fastbmRAG to real scientific papers and evaluated its output by manually checking the quality of the retrieval
and generation. As described in the previous section, we used 400 open-access papers related to diseases as test dataset. We
built an index with the default settings and queried it with five questions that represented some common queries in scientific
research. For endometriosis, these questions are: (1) Which genes are up- or down-regulated in endometriosis? (2) Which
pathways contribute to endometriosis? (3) Which mutations contribute to endometriosis? (4) What is the role of IL1B in
endometriosis? (5) What is the causal mechanism of endometriosis? The answers generated by fastbmRAG and other LLM
models are evaluated by manually checking them in the original paper.
2.7 Software and availability
fastbmRAG is implemented in Python 3.11 or higher. There are two modes: ’update’ and ’query’. The ’update’ mode is
used to create an new collection or add new documents to an existing collection of vector database. The documents should
be a text file in CSV format. It should have at least three columns: ’abstract’, ’main text’ and ’paper id’. If there are more
columns, they are used for additional information. Each element of the ’main text’ column should be either a list of strings
in the format ’[str1, str2]’ or a string separated by ’∖n’. ’paper id’ is a unique ID for each paper. If a paper ID exists in the
collection, the corresponding paper will be ignored. To update the collection, use the following command:
python main.py --job update --document input.csv
--collection_name collection_name
--working_dir directory_path
Here, ‘collection name’ and ‘working dir’ specify the collection name and directory to store collection.
Another mode is ‘query’. It is used to query the collection.
python main.py --job query --collection_name collection_name
--working_dir directory_path
--question ’your question’
The source code of fastbmRAG is public available inhttps://github.com/menggf/fastbmRAG
3 Results
3.1 Workflow of fastbmRAG
Unlike internet documents, scientific papers are usually well-structured and include sections such as abstract, introduction,
methods, results, and discussion. The abstract provides a brief yet comprehensive overview of the entire paper, while the main
text has detailed yet redundant information about the key findings. FastbmRAG leverages these features of scientific papers
to construct an entity-relationship knowledge graph in two steps. Figure 1 describes the flowchart of the entire pipeline.
3

Table 1: Time consumption of fastbmRAG and LightRAG in indexing biomedical papers
Disease*No. selected Papers fastbmRAG LightRAG
endometriosis 400 1.8 h 18.6 h
psoriatic arthritis 400 1.6 h 18.1 h
Parkinson’s disease 400 2.1 h 25.3 h
Ovarian cancer 400 1.9 h 22.2 h
COVID-19 400 1.5 h 18.5 h
Average 400 1.78 h 20.54 h
*Evaluation is repeated for three times and the average computational time is reported.
In step one, fastbmRAG uses a similar strategy of LightRAG by incorporates graph structures into text indexing processes
by entity and relationship extraction. In detail, fastbmRAG uses an LLM to process the abstracts of scientific papers and
extract entities, which can be genes, diseases, drugs, animal models, mutations, pathways, or other related types. Based
on their role, these entities can be either source or target entities, constituting the source or target nodes of the knowledge
graph. LLM summarizes the descriptions of the relationships between source and target entities according to the knowledge
in the abstracts. However, these descriptions may be concise and lack details. To refine these descriptions, fastbmRAG first
generates a general query for each pair of source and target entities. In Step 2, the main texts of the scientific papers are
splitted into paragraphs or fixed-size chunks. If a fixed size is set, overlapping chunks are necessary. FastbmRAG searches
the query question against the embedded main text to find chunks related to the source and target entities. The LLM refines
the descriptions of entity pairs using the selected chunks.
After two steps, a knowledge graph is constructed, including entities and their relationships. This graph has the following
information: source entity, source entity type, target entity, target entity type, relationship description, relationship type,
and additional information such as journal, authors, paper ID, publication year, and weights. To facilitate graph query,
entity names should be standardized. Gene names are transformed into Human Gene Nomenclature Committee (HGNC)
symbols, and disease names are transformed into Medical Subject Headings (MeSH) terms. This step is usually performed
by an LLM. The knowledge graph is stored in a vector database for querying.
During the query stage, the LLM processes the question to extract the query gene, query disease, and output types. This
information is then used as filters to improve the precision of the output and reduce the scale of the information graph. The
query vector is then searched for relationship descriptions of the entities to retrieve related answers from related papers.
Finally, the LLM summarizes these answers into a final output.
3.2 Efficiency evaluation for indexing biomedical literature
The primary objective of fastbmRAG is to enhance the efficiency of graph-based RAG systems for indexing large-scale
biomedical literature. To evaluate its effectiveness, we collected 400 open-access papers on various diseases from the PubMed
database. We built a knowledge graph using the following Python commands:
>import fastbmrag.fastbmrag as fastbmrag
>rag=fastbmrag.RAG(working_dir="endometriosis", collection_name="endometriosis")
>rag.insert_paper(documents)
Here, the ’documents’ is a pandas dataframe with column of ’abstract’, ’main text’, ’paper id’, ’article title’, ’year’,
’author’, ’journal’, ’weight’.
We used fastbmRAG to index 400 exemplary full-text research papers on five diseases: endometriosis, psoriatic arthritis,
Parkinson’s disease, ovarian cancer, and COVID-19. Table 1 shows the evaluation results. On average, fastbmRAG indexed
400 papers in 1.78 hours, or about 16 seconds per paper, on a Linux server with one RTX 4090 GPU. For comparison,
LightRag, a popular, general-purpose graph-based RAG model, was implemented to process the same papers using the same
LLM and embedding models on the same server. With the default settings, LightRag took approximately 20.54 hours to
process 400 papers, making fastbmRAG over 10 times faster. These results suggest that fastbmRAG is more efficient at
indexing biomedical literature.
3.3 Outcome evaluation
We evaluated FastbmRAG using 7,517 open-access, full-text papers related to endometriosis. After indexing, we used it
to answer some endometriosis-related questions and evaluated the coverage and accuracy of the results. The first question
was, ”Which genes are up- or down-regulated in endometriosis?” Figures 2 and Table S1 show the output of fastbmRAG
under different queries. Compared to the outputs of LightRag and the phi4 model, fastbmRAG collected more up- or down-
regulated genes and classified them into different categories. FastbmRAG also recognized the importance of other covariates
(e.g., tissue regions, hormonal treatment, menstrual stage, and other factors that affect gene expression) and discussed them
4

Figure 2: Outputs of fastbmRAG, LightRAG and Phi4 model
5

in its output. We notices some difference in output of fastbmRAG and LightRAG. For examples, LightRag reported down-
regulation of PTGS2. Our manual investigation found that only 9 open-access full-text papers reported the association of
PTGS2 with endometriosis and no paper reported the differential expression status of PTGS2 under disease status. This is
the reason why fastbmRAG fails to identify some genes. We manually verified the output genes of fastbmRAG and found
that the reported up- or down-regulated genes were all generated according to the source papers. We performed the same
evaluation with other questions, and fastbmRAG always generated more detailed outputs. Overall, our evaluation suggests
that fastbmRAG performs well in generating answers according to the user’s query with good coverage and accuracy.
4 Conclusion and Discussion
FastbmRAG is a new graph-based RAG tool, characterized by (a) fast-speed to process source dataset and (b) specifically
designed for biomedical papers. Unlike GraphRAG and LightRAG, fastbmRAG is specifically designed for biomedical lit-
erature. It utilizes the well-organized structure of scientific papers, such as the abstract and main text. FastbmRAG use
a similar framework of LightRAG by incorporates graph structures into text indexing and retrieval processes. It builds a
knowledge graph in two steps. First, it extracts entities from the abstract text. Then, it builds a knowledge graph using
only the abstract text, which is usually shorter than the main text but contains all the information from the main text. The
knowledge graph generated using abstracts usually contains the key entities and entity relationships but has fewer detailed
descriptions of the entity relationships. In the next step, fastbmRAG refines the entity relationships using the main text.
Using semantic searching, fastbmRAG extracts related paragraphs from the main text — usually two to three paragraphs
— and refines the relationship descriptions of the existing knowledge graph with these paragraphs. This makes fastbmRAG
more efficient at building knowledge graphs than general graph-based RAGs. When applied to real data, our evaluation
suggested that fastbmRAG indexed scientific papers about ten times faster than LightRAG.
Biomedical papers always have limited number of entities and types. In biomedical papers, genes and diseases are
key entities, and their names are standardized as specific terms (e.g., HGNC gene symbols and Disease Ontology terms).
Transforming the alias names into standard names allows fastbmRAG to filter the knowledge graph based on the exact
matching of disease and gene names. This greatly increases querying efficiency and generates more precise responses to users’
questions. Our evaluation also shows that fastbmRAG can capture more precise information from the input papers according
to the user’s query.
FastbmRAG also has some limitations. Although the settings make fastbmRAG more efficient to process biomedical
papers, it may miss the knowledge that only exists in the in main texts or that authors did not give enough description. One
example is PTGS2 that has been discussed in above section. Current version of fastbmRAG still needs many improvement,
e.g. integrating reasoning in the outputs to generate more confident results.
5 Acknowledgements
This work is supported by Changchun GeneScience Pharmaceuticals Co., Ltd..
References
[1] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie
Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-
Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman,
Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea
Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen,
Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing
Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila
Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Sim´ on Posada Fishman,
Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh,
Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane
Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan
Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu
Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn,
Heewoo Jun, Tomer Kaftan,  Lukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan,
Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight,
Daniel Kokotajlo,  Lukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal
Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin,
Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning,
6

Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney,
Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey
Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk,
David M´ ely, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long
Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo,
Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres,
Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell,
Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond,
Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders,
Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard,
Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin,
Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers,
Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng,
Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cer´ on Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea
Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila
Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich,
Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming
Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang,
William Zhuk, and Barret Zoph. Gpt-4 technical report, 2023.
[2] Arnon Ilani and Shlomi Dolev. Invited paper: Common public knowledge for enhancing machine learning data sets.
InProceedings of the 5th workshop on Advanced tools, programming languages, and PLatforms for Implementing and
Evaluating algorithms for Distributed systems, ApPLIED 2023, pages 1–10. ACM, June 2023.
[3] Abubakari Ahmed, Aceil Al-Khatib, Yap Boum, Humberto Debat, Alonso Gurmendi Dunkelberg, Lisa Janicke Hinchliffe,
Frith Jarrad, Adam Mastroianni, Patrick Mineault, Charlotte R. Pennington, and J. Andrew Pruszynski. The future of
academic publishing.Nature Human Behaviour, 7(7):1021–1026, July 2023.
[4] Zhiyong Lu, Yifan Peng, Trevor Cohen, Marzyeh Ghassemi, Chunhua Weng, and Shubo Tian. Large language models in
biomedicine and health: current research landscape and future directions.Journal of the American Medical Informatics
Association, 31(9):1801–1811, August 2024.
[5] Huizi Yu, Lizhou Fan, Lingyao Li, Jiayan Zhou, Zihui Ma, Lu Xian, Wenyue Hua, Sijia He, Mingyu Jin, Yongfeng Zhang,
Ashvin Gandhi, and Xin Ma. Large language models in biomedical and health informatics: A review with bibliometric
analysis.Journal of Healthcare Informatics Research, 8(4):658–711, September 2024.
[6] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models
using semantic entropy.Nature, 630(8017):625–630, June 2024.
[7] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language
models, 2024.
[8] Jia-Yu Yao, Kun-Peng Ning, Zhen-Hui Liu, Mu-Nan Ning, Yu-Yang Liu, and Li Yuan. Llm lies: Hallucinations are not
bugs, but features as adversarial examples, 2023.
[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨ uttler,
Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for
knowledge-intensive nlp tasks, 2020.
[10] Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He, Luna K. Qiu, and Lili Qiu. Retrieval augmented generation (rag)
and beyond: A comprehensive survey on how to make your llms use external data more wisely, 2024.
[11] R´ obert Lakatos, P´ eter Pollner, Andr´ as Hajdu, and Tam´ as Jo´ o. Investigating the performance of retrieval-augmented
generation and domain-specific fine-tuning for the development of ai-driven knowledge-based systems.Machine Learning
and Knowledge Extraction, 7(1):15, February 2025.
[12] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey
on rag meeting llms: Towards retrieval-augmented large language models. InProceedings of the 30th ACM SIGKDD
Conference on Knowledge Discovery and Data Mining, KDD ’24, pages 6491–6501. ACM, August 2024.
[13] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu
Wang, and Enhong Chen. A survey on knowledge-oriented retrieval-augmented generation, 2025.
[14] Rafael Teixeira de Lima, Shubham Gupta, Cesar Berrospi, Lokesh Mishra, Michele Dolfi, Peter Staar, and Panagiotis
Vagenas. Know your rag: Dataset taxonomy and generation strategies for evaluating rag systems, 2024.
7

[15] Philippe Laban, Alexander R. Fabbri, Caiming Xiong, and Chien-Sheng Wu. Summary of a haystack: A challenge to
long-context llms and rag systems, 2024.
[16] Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. Long context vs. rag for llms: An evaluation and revisits, 2025.
[17] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused
summarization, 2024.
[18] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation,
2024.
[19] Yuxin Dong, Shuo Wang, Hongye Zheng, Jiajing Chen, Zhenhong Zhang, and Chihang Wang. Advanced rag models with
graph structures: Optimizing complex knowledge reasoning and text generation. In2024 5th International Symposium
on Computer Engineering and Intelligent Communications (ISCEIC), pages 626–630. IEEE, November 2024.
[20] Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, and Siqiang Luo. Graph-based approaches and
functionalities in retrieval-augmented generation: A comprehensive survey, 2025.
[21] Julien Delile, Srayanta Mukherjee, Anton Van Pamel, and Leonid Zhukov. Graph-based retriever captures the long tail
of biomedical knowledge, 2024.
[22] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen,
Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented generation for customized large language models,
2025.
8