# FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems

**Authors**: Val Andrei Fajardo, David B. Emerson, Amandeep Singh, Veronica Chatrath, Marcelo Lotif, Ravi Theja, Alex Cheung, Izuki Matsuba

**Published**: 2025-06-10 19:42:07

**PDF URL**: [http://arxiv.org/pdf/2506.09200v2](http://arxiv.org/pdf/2506.09200v2)

## Abstract
Retrieval-augmented generation (RAG) systems have been shown to be effective
in addressing many of the drawbacks of relying solely on the parametric memory
of large language models. Recent work has demonstrated that RAG systems can be
improved via fine-tuning of their retriever and generator models. In this work,
we introduce FedRAG, a framework for fine-tuning RAG systems across centralized
and federated architectures. FedRAG supports state-of-the-art fine-tuning
methods, offering a simple and intuitive interface and a seamless conversion
from centralized to federated training tasks. FedRAG is also deeply integrated
with the modern RAG ecosystem, filling a critical gap in available tools.

## Full Text


<!-- PDF content starts -->

arXiv:2506.09200v2  [cs.LG]  12 Jun 2025FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation
Systems
Val Andrei Fajardo1David B. Emerson1Amandeep Singh1Veronica Chatrath1
Marcelo Lotif1Ravi Theja Desetty2Chi Ho Cheung2
Izuki Matsuba2
1Vector Institute, Toronto ON M5G 0C6, Canada
2Independent Researcher, Toronto, Canada
Abstract
Retrieval-augmented generation (RAG) systems have
been shown to be effective in addressing many of the
drawbacks of relying solely on the parametric memory of
large language models. Recent work has demonstrated
that RAG systems can be improved via fine-tuning of
their retriever and generator models. In this work, we
introduce FedRAG, a framework for fine-tuning RAG
systems across centralized and federated architectures.
FedRAG supports state-of-the-art fine-tuning methods,
offering a simple and intuitive interface and a seamless
conversion from centralized to federated training tasks.
FedRAG is also deeply integrated with the modern RAG
ecosystem, filling a critical gap in available tools.
1 Introduction
Large Language Models (LLMs) have demonstrated a
remarkable capacity to perform a diverse set of tasks,
despite standard pre-training only incorporating a next-
token prediction objective [ 4,14,7,30,27]. Prompt
engineering techniques, such as in-context learning, chain-
of-thought (CoT), and self-consistency have been shown
to further improve the performance of pre-trained LLMs
in many settings [ 4,35,33]. However, there are tasks, par-
ticularly knowledge-intensive ones, where relying solely
on information encoded in an LLM’s parameters, often
referred to as parametric memory, may lead to factual
inaccuracies in model responses, colloquially named hal-
lucinations [ 16]. Hallucinations issues can become more
prevalent when models attempt to address queries beyond
their knowledge cutoffs [10, 26].
Retrieval-augmented generation (RAG) is a popular
methodology that aims to address this specific draw-
back [ 20,29]. Specifically, RAG systems present relevant
non-parametric knowledge drawn from external systems
alongside queries in the form of additional input to an
LLM. Perhaps the most widely used form of this involves
computing an embedding of right-sized chunks of the
non-parametric knowledge and storing these in a vectorstore like Qdrant, Chroma or Pinecone for future search
and retrieval [ 12]. More elaborate designs of RAG sys-
tems have also been developed, such as those utilizing
knowledge graphs [28].
Recent studies have shown that fine-tuning RAG sys-
tems can lead to even greater performance improve-
ments [ 22,36,6]. That is, through fine-tuning, the over-
all RAG system, comprised of a generator, retriever and
knowledge store, may be adapted to work more cohesively
as a single unit in performing tasks.
The RAG ecosystem of today is quite vibrant and
includes a wide range of options for LLMs, retrieval
models, and re-rankers, among other components [ 18,
24,3,34,37] being offered by organizations operating
under both open- as well as closed-source business models.
There are also more than a few storage, observability,
and evaluation solutions that developers can choose from
in order to build an end-to-end RAG production. Finally,
popular RAG frameworks such as LlamaIndex [ 23] and
LangChain [ 5] offer users the ability to rapidly assemble
and experiment with diverse RAG system configurations,
ultimately aiding in the discovery of optimal designs.
Yet, to the best of our knowledge, there are few, if any,
frameworks that help simplify RAG fine-tuning, while
remaining well integrated with other available tools and
resources in the ecosystem.
The work presented here aims to directly fill this gap.
Specifically, we introduce FedRAG, a framework for fine-
tuning RAG systems across both centralized and feder-
ated architectures.1Decentralized designs for LLM train-
ing and deployment are becoming increasingly important,
as evidenced by popular initiatives like Anthropic’s Model
Context Protocol (MCP) [ 1] and Google’s Agent2Agent
Protocol [ 13]. Moreover, in settings where data privacy
prevents centralizing datasets, decentralized training tech-
niques like federated learning (FL) become an indispens-
able tool for improving RAG systems.
1Library Code: https://github.com/VectorInstitute/fed-
rag
1

2 Related Work
2.1 Fine-tuning RAG Systems
As discussed above, RAG systems are comprised of a num-
ber of components, some of which are driven by trainable
models. This work specifically focuses on two main com-
ponents: the generator, which is responsible for text
generation; and the retrieval model, which maps queries
or prompts into a form, commonly a high-dimensional
embedding vector, used to retrieve related context from
a knowledge store.
Several studies have focused on generator training via
instruction fine-tuning, for which the instruction exam-
ples include context retrieved by the retriever from the
knowledge store. Lin et al. [22]refer to this approach as
Retrieval-Augmented Language Model Training (RALT).
A similar generator fine-tuning approach called Retrieval-
Augmented Fine-Tuning (RAFT) was utilized in [ 36].
RAFT differs from RALT in that the instruction examples
also include LLM generated CoT passages with reasoning
traces linking the response to the query. Finally, in line
with recent trends in reasoning LLMs [ 19], Chen et al. [6]
introduce ReSearch, which follows a reinforcement learn-
ing approach similar to that used in DeepSeek-R1 [ 8].
With ReSearch, the LLM learns to generate long CoT
passages that incorporate search and retrieval from a
knowledge store. In so doing, the generator is adapted to
cycle between reasoning and retrieval, potentially multi-
ple times.
Fewer studies exist considering retriever fine-tuning.
In the same work that produced RALT, the authors also
introduce Language Model Supervised Retriever Train-
ing (LSR). In LSR, the retrieval scores of retrieved text
chunks as well as corresponding target sequence proba-
bilities produced by the generator model, conditioned on
the context of each retrieved chunk, form two distribu-
tions whose distance, measured by the Kullback-Leibler
divergence, is minimized.
2.2 Federated Learning for LLM Appli-
cations
Recently, Flower Labs developed the first federally pre-
trained LLM called FlowerLLM. Further, efforts support-
ing federated LLM tuning have been undertaken [ 31].
In the context of RAG systems, however, work has pri-
marily focused on decentralized inference rather than
federated fine-tuning or training. A notable example
published by Flower Labs demonstrates RAG inference
in a federated setting.2Similarly, LlamaIndex has also
developed a library extension to their framework called
llama-index-networks [11] for decentralized RAG in-
ference. To the best of our knowledge, no existing tools
provide a simple interface for converting centralized RAG
fine-tuning to federated tasks.
2https://flower.ai/docs/examples/fedrag.html3Philosophy and Design Princi-
ples
In this section, we describe the core philosophy and design
principles that guide the development of FedRAG. These
principles address the challenges identified in the previous
section and inform implementation decisions.
3.1 Philosophy
We endeavour to build FedRAG for researchers and prac-
titioners alike in a manner that makes applying state-of-
the-art fine-tuning techniques to RAG systems simple,
yet highly effective, irrespective of whether the system is
centralized or federated. Moreover, we seek to make re-
searching new methods for RAG fine-tuning more efficient
and scientifically rigorous by promoting reproducibility.
This is achieved through designing flexible components
and appropriate tools that allow users to easily replicate
and disseminate their RAG systems, as well as extend
the library with custom trainers, losses, benchmarks, and
more.
3.2 Design Principles
Advanced RAG Fine-Tuning: Comprehensive sup-
port for state-of-the-art RAG fine-tuning methods that
can be federated with ease.
This principle is central to advancing the state of knowl-
edge in RAG methods. By implementing and supporting
frontier techniques in RAG fine-tuning, while simultane-
ously making federation straightforward and accessible,
researchers are enabled to develop and evaluate novel
approaches in a systematic and reproducible fashion. At
the same time, such methods transfer smoothly to decen-
tralized systems.
Work With Your Tools: Seamless integration with
popular frameworks including HuggingFace, Unsloth, and
LlamaIndex, to become deeply embedded in the RAG
ecosystem and beyond.
By designing FedRAG as a deeply embedded frame-
work within the existing RAG ecosystem, barriers to
adoption are significantly reduced for both practitioners
and researchers. Integrations into popular frameworks
and libraries, such as those mentioned above, allow users
to leverage familiar tools and workflows while gaining
access to advanced fine-tuning capabilities. Finally, ex-
tensive ecosystem compatibility facilitates discoverability
and further adoption of new methods and results, thus
maximizing the impact and reach of research advance-
ments.
Lightweight Abstractions: Clean, intuitive abstrac-
tions that simplify RAG fine-tuning while maintaining
full flexibility and control.
2

We seek to provide developers with an intuitive inter-
face and abstractions that are easy to work with, cus-
tomize, and extend. Lowering the learning curve to use
FedRAG, while simultaneously increasing its utility and
effectiveness, is essential to providing a pleasant develop-
ment experience. This approach enables practitioners and
researchers to focus their efforts entirely on the challenges
of designing and experimenting with methods for improv-
ing RAG systems rather than wrestling with complex
implementation details.
4 FedRAG Framework Overview
This section provides a more detailed overview of the
FedRAG library.
4.1 Library Organization
FedRAG incorporates a modular design, consisting of
several modules with clear and intuitive separation of
concerns. Table 1 presents non-exhaustive overview of
the key modules and their responsibilities.
Module Description
core Core types i.e., RAGSystem
evals Evaluation metrics and benchmarks
fltasks Federated learning task definitions
generators Generator types
knowledge stores Data storage
retrievers Retriever types
trainers Trainer types
Table 1: Key modules in FedRAG and their responsibili-
ties.
4.2 Standard Usage Patterns
Building a RAG System: We first introduce
the main classes that FedRAG offers and with
which users will often work. We begin with the
core.RAGSystem class, which is comprised of three
parts, namely: knowledge stores .KnowledgeStore ,
retrievers .Retriever , and generators .Generator . Fig-
ure 1 provides a code snippet on how to assemble a
RAGSystem with FedRAG.
RAG Fine-Tuning: After the essential RAG compo-
nents are constructed, the system can be trained using the
fine-tuning abstractions offered by the library. FedRAG
provides various trainers .Trainer classes distinguished
by their methodology and which model, generator or
retriever, they target. A typical pattern for performing
retriever or generator training is provided in Figure 2.
There, the manager is an orchestrator object that bears
the responsibility of preparing the target model for train-
ing and ensuring the other model is frozen. Note that
both generator and retriever trainer objects also expose a# Flat imports are supported
from fed_rag import (
RAGSystem,
RAGConfig,
HFSentenceTransformerRetriever,
UnslothFastModelGenerator,
QdrantKnowledgeStore
)
knowledge_store = QdrantKnowledgeStore()
generator = UnslothFastModelGenerator(
"unsloth/gemma-3-4b-it",
)
retriever = HFSentenceTransformerRetriever(
"nthakur/dragon-plus-query-encoder",
"nthakur/dragon-plus-context-encoder",
)
# Assemble rag_system
rag_system = RAGSystem(
knowledge_store=knowledge_store,
generator=generator,
retriever=retriever,
rag_config=RAGConfig(top_k=2)
)
# Executes the typical RAG pipeline
response = rag_system.query("What are tulips?")
Figure 1: Creating a RAGSystem with FedRAG. Not
depicted here, but the retriever is also used to popu-
late embedded reference chunks into the knowledge store ,
prior to querying.
train() method that can be called without using manager
providing an interface similar to that of HuggingFace.
Federated Fine-Tuning: With the centralized fine-
tuning pattern established, we show the simple process
for converting the previous task to a federated one. This
is done by extracting an fltasks .FLtask object from
themanager . This is demonstrated in Figure 3. Each
client has its own fine-tuning dataset and contribute to
the tuning process in a decentralized manner and updates
are combined with federated averaging [25].
Evaluation and Benchmarking: In this final pat-
tern demonstration, we show how benchmarking a RAG
system can be achieved. Figure 4 depicts an intu-
itive benchmarking pattern where an evals .Benchmarker
runs the desired evals .Benchmark using the chosen
evals .EvaluationMetric.
These patterns demonstrate the consistent API de-
sign of FedRAG, enabling users to seamlessly transition
between RAG system development, central and decen-
tralized fine-tuning, and evaluation with minimal code
changes.
3

... # Keep code from Figure 1
from fed_rag.trainers import (
HuggingFaceTrainerForRALT
HuggingFaceTrainerForLSR
)
from fed_rag.trainer_managers import (
HuggingFaceRAGTrainerManager
)
from datasets import Dataset
# Train datasets are examples of (query,
# response) pairs
train_dataset = Dataset.from_dict(
{
"query": [...],
"response": [...]
}
)
generator_trainer = HuggingFaceTrainerForRALT(
rag_system=rag_system,
train_dataset=train_dataset,
)
retriever_trainer = HuggingFaceTrainerForLSR(
rag_system=rag_system,
train_dataset=train_dataset,
)
manager = HuggingFaceRAGTrainerManager(
mode="retriever", # can be generator
retriever_trainer=retriever_trainer,
generator_trainer=generator_trainer,
)
# Train
train_result = manager.train()
Figure 2: Fine-tuning a RAGSystem with FedRAG.
... # Keep code from Figures 1 and 2
import flwr as fl # The FL backend
# fl_task
fl_task = manager.get_federated_task()
# Build fl server and client
server = fl_task.server(
model=retriever_trainer.model
)
client = fl_task.client(
model=retriever_trainer.model,
train_dataset=train_dataset,
)
# Spin up client and server using flwr
fl.start_server(server)
fl.start_client(client)
Figure 3: Federated fine-tuning of a RAGSystem with
FedRAG.
4.3 Integrations
In this section, we briefly outline the existing integra-
tions to popular tools and frameworks within the RAG
ecosystem. Of the integrations listed in Table 2, only the... # Keep code from Figures 1 and 2
import fed_rag.evals.benchmarks as benchmarks
from fed_rag.evals import (
Benchmarker,
ExactMatchEvaluationMetric,
)
benchmarker = Benchmarker(rag_system=rag_system)
mmlu = benchmarks.HuggingFaceMMLU(streaming=True)
metric = ExactMatchEvaluationMetric()
# Run benchmark with first 3 examples only
result = benchmarker.run(
benchmark=mmlu,
metric=metric,
is_streaming=True,
num_examples=3,
agg="avg",
)
Figure 4: Benchmarking with FedRAG.
LlamaIndex integration had not been represented in the
preceding patterns of Figures 1–4. FedRAG supports a
bridge to convert a RAGSystem object to a LlamaIndex
RAG system equivalent, thus enabling users to leverage
their powerful inference features and ecosystem. These in-
tegrations allow FedRAG users to leverage existing tools
while gaining RAG fine-tuning capabilities, aligning with
theWork With Your Tools design principle in Section 3.2.
Library Integration
HuggingFace Generators, retrievers, datasets
Unsloth Fast fine-tuning of generators
Qdrant Storage solution for knowledge
LlamaIndex Bridge to inference object
Table 2: Currently supported integrations in FedRAG.
5 Future Work and Conclusions
In this paper, we introduced FedRAG, a framework for
fine-tuning RAG systems across both centralized and
federated architectures that offers state-of-the-art fine-
tuning methods and fills a critical gap within the RAG
ecosystem. In Appendix A, a lightweight experiment is
presented. The results confirm that the framework can
be used to successfully and flexibly execute RAG fine-
tuning tasks. The experimental code and a containerized
image of the knowledge store is released with this paper
to facilitate reproducibility.
In terms of future development, we have several excit-
ing and impactful additions on the development roadmap.
For example, an MCP RAG system and companion MCP
knowledge store will soon be integrated into the frame-
work, which will pave the way for studying the effects of
adapting RAG systems to knowledge provided by third-
4

party MCP providers. Additional high-priority develop-
ment items are presented in Table 4 of Appendix B. We
are eager to continue the development of FedRAG and
believe that it will enable researchers and practitioners to
more easily explore advanced RAG fine-tuning techniques
in both centralized and federated settings.
References
[1]Anthropic. Model context protocol, 2024. URL
https://github.com/modelcontextprotocol/
modelcontextprotocol .
[2]Jonathan Berant, Andrew Chou, Roy Frostig, and
Percy Liang. Semantic parsing on freebase from
question-answer pairs. In Proceedings of the 2013
conference on empirical methods in natural language
processing , pages 1533–1544, 2013.
[3]Vladimir Blagojevic. Enhancing RAG pipelines
in Haystack: Introducing diversityranker
and lostinthemiddleranker , 2023. https:
//towardsdatascience.com/enhancing-rag-
pipelines-in-haystack-45f14e2bc9f5 .
[4]T. Brown, B. Mann, N. Ryder, M. Subbiah, J.D.
Kaplan, P. Dhariwal, A. Neelakantan, and et al. Lan-
guage models are few-shot learners. In Advances in
Neural Information Processing Systems , volume 33,
pages 1877–1901, 2020.
[5]Harrison Chase. LangChain, October 2022. URL
https://github.com/langchain-ai/langchain .
[6]Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie
Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z. Pan,
Wen Zhang, Huajun Chen, Fan Yang, Zenan Zhou,
and Weipeng Chen. Research: Learning to reason
with search for llms via reinforcement learning, 2025.
URL https://arxiv.org/abs/2503.19470 .
[7]Cyril Chhun, Pierre Colombo, Fabian M. Suchanek,
and Chlo´ e Clavel. Of human criteria and automatic
metrics: A benchmark of the evaluation of story gen-
eration. In Nicoletta Calzolari, Chu-Ren Huang,
Hansaem Kim, James Pustejovsky, Leo Wanner,
Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lu-
cia Donatelli, Heng Ji, Sadao Kurohashi, Patrizia
Paggio, Nianwen Xue, Seokhwan Kim, Younggyun
Hahm, Zhong He, Tony Kyungil Lee, Enrico Santus,
Francis Bond, and Seung-Hoon Na, editors, Proceed-
ings of the 29th International Conference on Com-
putational Linguistics , pages 5794–5836, Gyeongju,
Republic of Korea, October 2022. International Com-
mittee on Computational Linguistics. URL https:
//aclanthology.org/2022.coling-1.509/ .[8]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei
Zhang, and et al. Deepseek-r1: Incentivizing rea-
soning capability in llms via reinforcement learning,
2025. URL https://arxiv.org/abs/2501.12948 .
[9]Tim Dettmers, Artidoro Pagnoni, Ari Holtzman,
and Luke Zettlemoyer. Qlora: Efficient finetuning
of quantized llms. Advances in neural information
processing systems , 36:10088–10115, 2023.
[10]Omkar Dige, John Willes, and D. B. Emerson.
Evaluating RAG system performance: The impact
of knowledge cut-off and fine-tuning. In Adap-
tive Foundation Models: Evolving AI for Person-
alized and Efficient Learning , 2024. URL https:
//openreview.net/forum?id=2K6I3317QV .
[11]Val Andrei Fajardo, Jerry Liu, Logan
Markewich, Simon Suo, Haotian Zhang,
and Sourabh Desai. LlamaIndex Networks,
11 2024. URL https://github.com/run-
llama/llama_index/llama-index-networks .
Extension for LlamaIndex.
[12]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie
Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on rag meeting llms: To-
wards retrieval-augmented large language models.
InProceedings of the 30th ACM SIGKDD Con-
ference on Knowledge Discovery and Data Mining ,
KDD ’24, page 6491–6501, New York, NY, USA,
2024. Association for Computing Machinery. ISBN
9798400704901. doi: 10.1145/3637528.3671470. URL
https://doi.org/10.1145/3637528.3671470 .
[13]Google. Agent2agent (a2a) protocol, 2025. URL
https://github.com/google/A2A .
[14]Aaron Grattafiori, Abhimanyu Dubey, Abhinav
Jauhri, Abhinav Pandey, and et al. The llama 3
herd of models, 2024. URL https://arxiv.org/
abs/2407.21783 .
[15]Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Song, and Jacob Stein-
hardt. Measuring massive multitask language under-
standing. arXiv preprint arXiv:2009.03300 , 2020.
[16]Lei Huang, Weijiang Yu, Weitao Ma, Weihong
Zhong, Zhangyin Feng, Haotian Wang, Qianglong
Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and
Ting Liu. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Trans. Inf. Syst. , 43(2), Jan-
uary 2025. ISSN 1046-8188. doi: 10.1145/3703155.
URL https://doi.org/10.1145/3703155 .
[17]Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
5

Grave. Few-shot learning with retrieval augmented
language models. arXiv preprint arXiv:2208.03299 ,
1(2):4, 2022.
[18]Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval
for open-domain question answering. In Bonnie
Webber, Trevor Cohn, Yulan He, and Yang Liu, edi-
tors, Proceedings of the 2020 Conference on Em-
pirical Methods in Natural Language Processing
(EMNLP) , pages 6769–6781, Online, November 2020.
Association for Computational Linguistics. doi:
10.18653/v1/2020.emnlp-main.550. URL https:
//aclanthology.org/2020.emnlp-main.550 .
[19]Komal Kumar, Tajamul Ashraf, Omkar Thawakar,
Rao Muhammad Anwer, Hisham Cholakkal,
Mubarak Shah, Ming-Hsuan Yang, Phillip H. S. Torr,
Fahad Shahbaz Khan, and Salman Khan. Llm post-
training: A deep dive into reasoning large language
models, 2025. URL https://arxiv.org/abs/2502.
21321 .
[20]Patrick Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau
Yih, Tim Rockt¨ aschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for
knowledge-intensive NLP tasks. In Proceedings of
the 34th International Conference on Neural Infor-
mation Processing Systems , NIPS ’20, Red Hook,
NY, USA, 2020. Curran Associates Inc. ISBN
9781713829546.
[21]Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas
Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih,
and Xilun Chen. How to train your dragon: Diverse
augmentation towards generalizable dense retrieval.
arXiv preprint arXiv:2302.07452 , 2023.
[22]Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia
Shi, Maria Lomeli, Richard James, Pedro Rodriguez,
Jacob Kahn, Gergely Szilvasy, Mike Lewis, et al. Ra-
dit: Retrieval-augmented dual instruction tuning. In
The Twelfth International Conference on Learning
Representations , 2023.
[23]Jerry Liu. LlamaIndex, November 2022. URL https:
//github.com/jerryjliu/llama_index .
[24]Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao,
and Nan Duan. Query rewriting in retrieval-
augmented large language models. In Houda
Bouamor, Juan Pino, and Kalika Bali, editors, Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing , pages 5303–
5315, Singapore, December 2023. Association for
Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.322. URL https://aclanthology.
org/2023.emnlp-main.322 .
[25]H. B. McMahan, E. Moore, D. Ramage, S. Hamp-
son, and B. A. y Arcas. Communication-efficient
learning of deep networks from decentralized data.
Proceedings of the 20th International Conference
on Artificial Intelligence and Statistics (AISTATS) ,
2017.
[26]Oded Ovadia, Menachem Brief, Moshik Mishaeli,
and Oren Elisha. Fine-tuning or retrieval? com-
paring knowledge injection in llms, 2024. URL
https://arxiv.org/abs/2312.05934 .
[27]Debjit Paul, Mete Ismayilzada, Maxime Peyrard,
Beatriz Borges, Antoine Bosselut, Robert West, and
Boi Faltings. REFINER: Reasoning feedback on in-
termediate representations. In Yvette Graham and
Matthew Purver, editors, Proceedings of the 18th
Conference of the European Chapter of the Asso-
ciation for Computational Linguistics (Volume 1:
Long Papers) , pages 1100–1126, St. Julian’s, Malta,
March 2024. Association for Computational Lin-
guistics. URL https://aclanthology.org/2024.
eacl-long.67/ .
[28]Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. Graph retrieval-augmented generation: A
survey, 2024. URL https://arxiv.org/abs/2408.
08921 .
[29]Ori Ram, Yoav Levine, Itay Dalmedigos, Dor
Muhlgay, Amnon Shashua, Kevin Leyton-Brown,
and Yoav Shoham. In-context retrieval-augmented
language models. Transactions of the Associa-
tion for Computational Linguistics , 11:1316–1331,
2023. doi: 10.1162/tacl a00605. URL https:
//aclanthology.org/2023.tacl-1.75 .
[30]Jie Ren, Yao Zhao, Tu Vu, Peter J. Liu, and Bal-
aji Lakshminarayanan. Self-evaluation improves
selective generation in large language models. In
Javier Antor´ an, Arno Blaas, Kelly Buchanan, Fan
Feng, Vincent Fortuin, Sahra Ghalebikesabi, An-
dreas Kriegler, Ian Mason, David Rohde, Francisco
J. R. Ruiz, Tobias Uelwer, Yubin Xie, and Rui
Yang, editors, Proceedings on ”I Can’t Believe It’s
Not Better: Failure Modes in the Age of Founda-
tion Models” at NeurIPS 2023 Workshops , volume
239 of Proceedings of Machine Learning Research ,
pages 49–64. PMLR, 16 Dec 2023. URL https:
//proceedings.mlr.press/v239/ren23a.html .
[31]Lorenzo Sani, Alex Iacob, Zeyu Cao, Royson Lee,
Bill Marino, Yan Gao, Dongqi Cai, Zexi Li, Wanru
Zhao, Xinchi Qiu, and Nicholas D. Lane. Photon:
Federated llm pre-training, 2024. URL https://
arxiv.org/abs/2411.02908 .
6

[32]Hugo Touvron, Louis Martin, Kevin Stone, Peter
Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava,
Shruti Bhosale, et al. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 , 2023.
[33]Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V.
Le, Ed H. Chi, Sharan Narang, Aakanksha Chowd-
hery, and Denny Zhou. Self-consistency improves
chain of thought reasoning in language models. In
The Eleventh International Conference on Learning
Representations, ICLR 2023, Kigali, Rwanda, May
1-5, 2023 , 2023. URL https://openreview.net/
forum?id=1PL1NIMMrw .
[34]Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu,
Ruiyi Zhang, and Tyler Derr. Knowledge graph
prompting for multi-document question answering,
2023. URL https://arxiv.org/abs/2308.11730 .
[35]Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi,
Quoc V. Le, and Denny Zhou. Chain-of-thought
prompting elicits reasoning in large language models.
InProceedings of the 36th International Conference
on Neural Information Processing Systems , NIPS
’22, Red Hook, NY, USA, 2022. Curran Associates
Inc. ISBN 9781713871088.
[36]Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E Gon-
zalez. Raft: Adapting language model to domain
specific rag. In First Conference on Language Mod-
eling, 2024.
[37]Shengyao Zhuang, Bing Liu, Bevan Koopman, and
Guido Zuccon. Open-source large language mod-
els are strong zero-shot query likelihood models
for document ranking. In Houda Bouamor, Juan
Pino, and Kalika Bali, editors, Findings of the As-
sociation for Computational Linguistics: EMNLP
2023, pages 8807–8817, Singapore, December 2023.
Association for Computational Linguistics. doi:
10.18653/v1/2023.findings-emnlp.590. URL https:
//aclanthology.org/2023.findings-emnlp.590 .
7

A Example: RA-DIT
In their work, Lin et al. [22]conducted various experiments studying the effectiveness of RALT and LSR fine-tuning
methods. Their experiments revealed that applying RALT or LSR individually leads to performance gains, but the
greatest gain comes after applying both RALT and LSR in succession. They termed this combination of fine-tuning
techniques Retrieval-Augmented Dual Instruction Tuning (RA-DIT). In order to illustrate the potential of the FedRAG
framework, we aim to reproduce a lightweight version of their experiments. The rest of this appendix outlines the
RAG system specifications, details on fine-tuning as well as evaluation setup, and finally the results of the experiments.
A.1 RAG System
A.1.1 Knowledge Store & Retriever
We use text chunks from the December 2021 Wikipedia dump released by Izacard et al. [17]. This re-
lease includes two files, infobox.jsonl and text-list-100-sec.jsonl , which can be downloaded from the
facebookresearch/atlas GitHub repository.3For the knowledge store, we use the first 10M text passages provided
intext-list-100-sec.jsonl with no further preprocessing with the exception of concatenating the title ,section ,
andtext fields for each passage.
For the retriever, we use DRAGON+ [ 21]. More specifically, we use the SentenceTransformer versions of this
dual-encoder model available on HuggingFace.4,5The context encoder of DRAGON+ is used to encode the 10M text
chunks prior to loading the embeddings into the knowledge store.
A.1.2 Generator
For the generator LLM, we use a quantized (4-bit) Llama2-7B [ 32]. We specifically use the official version of this
model available on HuggingFace,6with the load in4bit parameter set to True .
A.2 Fine-tuning & Evaluation
For the fine-tuning dataset, we use Web Questions [ 2] available on HuggingFace.7We apply QLoRA [ 9] fine-tuning
with only the RALT objective by making use of a trainers .HuggingFaceTrainerForRALT object and supplying it a
generators .HFPeftModelGenerator. The latter is used to load a PeftModel available on HuggingFace.8
For evaluation, we use the test split of the global facts subset of MMLU [ 15] available on HuggingFace, which
has exactly 100 data points.9Similar to Lin et al. [22], we apply 5-shot in-context learning for each evaluation example.
The few-shot examples are randomly drawn from the validation split and held fixed throughout evaluation. For
performance measurement, we use the exact match metric.
A.3 Results
We report the results of two separate fine-tuning runs in Table 3. For comparison, we also report the performance of
the same RAG system but without any fine-tuning applied at all.
Method Run 1 Run 2 Average
Without fine-tuning 17 .0 27 .0 22 .0
RALT fine-tuning 27 .0 34 .0 30 .5
Table 3: RA-DIT inspired experiment demonstrating the effect of RALT fine-tuning on exact match performance for
the MMLU ( global facts ) benchmark.
The results of our lightweight experiment corroborate the findings of Lin et al. [22]and align with results from other
work [ 36,6]. That is, RAG fine-tuning can lead to significant performance gains. Note that the observed variability
in runs is due to the sampling parameters used for generation.
3https://github.com/facebookresearch/atlas
4https://huggingface.co/nthakur/dragon-plus-query-encoder
5https://huggingface.co/nthakur/dragon-plus-context-encoder
6https://huggingface.co/meta-llama/Llama-2-7b-hf
7https://huggingface.co/datasets/stanfordnlp/web_questions
8https://huggingface.co/Styxxxx/llama2_7b_lora-quac
9https://huggingface.co/datasets/cais/mmlu
8

B Development Roadmap
In this section, we provide a portion of our development roadmap that includes the high-priority items, which we
deem to be highly impactful for our users.
Item Description
MCP knowledge store An MCP client that can be used to receive knowledge context from third-
party MCP servers.
MCP RAG system A specialized RAG system class that is able to retrieve knowledge from the
MCP Knowledge Store and subsequently supply it to the generator.
ReSearch generator trainer An implementation of ReSearch [ 6], i.e., equipping generator LLMs with
reasoning infused with search.
Improved retriever trainers New training objectives for language model supervised retriever fine-tuning.
Advanced federated learning Support for more advanced federated learning techniques.
LangChain integration Bridge to LangChain inference objects.
General optimizations Optimizations for batch RAG system querying and concurrent benchmarking.
Table 4: Development roadmap: high-priority items for FedRAG.
9