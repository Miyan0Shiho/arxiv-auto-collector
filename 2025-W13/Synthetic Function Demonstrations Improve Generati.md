# Synthetic Function Demonstrations Improve Generation in Low-Resource Programming Languages

**Authors**: Nick McKenna, Xinnuo Xu, Jack Williams, Nick Wilson, Benjamin Van Durme, Christian Poelitz

**Published**: 2025-03-24 15:09:03

**PDF URL**: [http://arxiv.org/pdf/2503.18760v1](http://arxiv.org/pdf/2503.18760v1)

## Abstract
A key consideration when training an LLM is whether the target language is
more or less resourced, whether this is English compared to Welsh, or Python
compared to Excel. Typical training data for programming languages consist of
real program demonstrations coupled with human-written comments. Here we
present novel approaches to the creation of such data for low resource
programming languages. We generate fully-synthetic, textbook-quality
demonstrations of common library functions in an example domain of Excel
formulas, using a teacher model. We then finetune an underperforming student
model, and show improvement on 2 question-answering datasets recast into the
Excel domain. We show advantages of finetuning over standard, off-the-shelf RAG
approaches, which can offer only modest improvement due to the unfamiliar
target domain.

## Full Text


<!-- PDF content starts -->

Synthetic Function Demonstrations Improve Generation
in Low-Resource Programming Languages
Nick M cKenna1Xinnuo Xu1Jack Williams1Nick Wilson1
Benjamin Van Durme2Christian Poelitz1
1Microsoft Research2Microsoft
cpoelitz@microsoft.com
Abstract
A key consideration when training an LLM
is whether the target language is more or less
resourced, whether this is English compared
to Welsh, or Python compared to Excel. Typ-
ical training data for programming languages
consist of real program demonstrations cou-
pled with human-written comments. Here we
present novel approaches to the creation of such
data for low resource programming languages.
We generate fully-synthetic, textbook-quality
demonstrations of common library functions in
an example domain of Excel formulas, using
a teacher model. We then finetune an under-
performing student model, and show improve-
ment on 2 question-answering datasets recast
into the Excel domain. We show advantages
of finetuning over standard, off-the-shelf RAG
approaches, which can offer only modest im-
provement due to the unfamiliar target domain.
1 Introduction
The process of pretraining LLMs on code has
improved significantly, supporting high accuracy
single-shot generation for general-purpose pro-
gramming languages (PLs) like Python, when
prompted with a variety of natural- and program-
ming language inputs (OpenAI et al., 2024b; Ab-
din et al., 2024; Yang et al., 2025). These high-
resource PLs (HRPLs) profit due to the abundance
of training data, which is often human-annotated
with explanatory comments (Kocetkov et al., 2022;
Lozhkov et al., 2024).
Comparatively, LLM generations in Low-
Resource Programming Languages (LRPLs) are
of significantly lower quality due to limited train-
ing data (Joel et al., 2024). In this paper, we tackle
generation in LRPLs through synthetic data finetun-
ing. Previous work directly augments real-world-
like data using SoTA LLMs (Honovich et al., 2023;
Yoo et al., 2021), yet even the performance of SoTA
LLMs on LRPL generation is unreliable, making
direct augmentation challenging.
A: To find the number of wins listed in the table, we need to count the rows 
where the result starts with 'W' .
We can use the COUNTIFS function to count cells that meet multiple 
criteria.
The first criteria will check if the result starts with 'W' .
The range for this criteria will be the 'Result' column (D2:D11).
```excel
=COUNTIFS(D2:D11, "W*")
```A B C D
1 Date Opponent Site Result
2 September 24 Millsaps* Denny Field • Tuscaloosa, AL W 54 -0
3 October 2 At Vanderbilt DudleyField  • Nashville, TN W 19 -7
4 October 9 At Mississippi A&M Meridian Fairgrounds • Meridian, MS W 26 -7
Q: How many wins are listed?(More rows… )
>>> Execution: 9
Figure 1: Finetuning on synthetic data improves adap-
tation to the low-resource domain. Qwen2.5-coder 3B
learns to read the table and compose a formula with
flexible string formatting, executing to the answer.
Fortunately, many LRPLs have publicly avail-
able human-written documentation detailing lan-
guage specifications and library function usage.
Instead of augmenting real-world-like data from
the vast space of possible programs, we generate
textbook-quality demonstrations of library func-
tions based on this documentation (Gunasekar et al.,
2023), scaled only to the library size. Our goal is
for models fine-tuned on this augmented data to (1)
efficiently capture the syntax and semantics of the
target LRPL, and (2) leverage shared knowledge
of programming concepts, learned during standard
pretraining on HRPLs, to reason through complex
problems.
LRPLs are numerous and widespread. Domain-
Specific Languages (DSLs) and many general-
purpose languages lack sufficient public training
data, and novel APIs in HRPLs can also be under-
resourced. In this paper, we focus on Excel (Zhao
et al., 2024), widely used by millions, yet still lack-
ing large-scale real-world data that pairs human-
1arXiv:2503.18760v1  [cs.CL]  24 Mar 2025

written Excel formulas with tables and queries for
generative model training.
In our experiments, we task models with answer-
ing table-based questions by generating Excel for-
mulas. The generation is evaluated by comparing
the results of formula execution to the ground truth
answers. The results show that: (1) providing Excel
function documentation along with the table-based
question as a prompt (RAG) does not significantly
improve generation accuracy; (2) “continued train-
ing” on the original documentation alone does not
enhance performance; (3) finetuning with our syn-
thetic textbook examples improves table-QA ac-
curacy by more than 10% for most models; (4)
while coding skills acquired from LRPLs during
pre-training cannot be directly applied to gener-
ate Excel formulas, they still make the fine-tuning
process on Excel more effective.
Our contributions are the following:
1.We develop a protocol for efficiently gener-
ating synthetic training examples of library
functions for a target LRPL (Excel formulas).
2.Using two open LLM model families (Qwen
2.5 and Llama 2) across two table-based QA
datasets (WikiTQ and TAT-QA), we show that
finetuning on our synthetic textbook examples
significantly improves models’ performance
on LRPL generation, even compared to stan-
dard RAG approaches using oracle retrieval.
We also show that HRPL code specialization
in model pretraining helps in domain adapta-
tion to a new LRPL.
3.Analysis shows a limitation of our approach:
training on individual function demonstrations
biases models toward generating simple pro-
grams; without complex demonstrations, mod-
els become limited by the expressivity of the
function library rather than the language.
2 Example Domain and Datasets
LRPLs lack publicly available data. Excel, while
popular, is mainly used in enterprise settings where
tabular data is sensitive and private, making real-
world examples that pair human-written Excel for-
mulas with tables and queries extremely rare online.
This makes Excel an interesting example case.
Excel formulas are not often tested with dedi-
cated datasets, so we recast existing table-basedQA datasets WikiTQ1andTAT-QA2into the Ex-
cel domain: we lightly preprocess and embed tables
into Excel spreadsheets, and query models to an-
swer questions using Excel formulas. To ensure the
table-based QA problems are solvable in Excel, we
use OpenAI o1 (OpenAI et al., 2024a) to generate
formulas for each problem, retaining only those
whose execution output matches the ground truth
answer. The formulas produced by o1 serve as the
oracle Excel solution for each sample. As a result,
we obtain 639 problems from WikiTQ and 459
from TAT-QA . In experiments we evaluate model
performance using program execution match (EM)
to the original dataset sample labels.
3 Synthetic Training Data Generation
Due to the absence of publicly available real-world
data, SoTA LLMs’ performance on Excel genera-
tion is unreliable, making data augmentation chal-
lenging (Joel et al., 2024). Fortunately, Excel (as
well as other LRPLs) have public documentation
of language specifications and library functions.
We leverage this documentation by augmenting a
teacher model to generate a high-quality synthetic
curriculum for teaching a student model. We expect
the student model to learn the Excel basic grammar
and functionality from our curriculum while trans-
ferring problem-solving skills from its pretraining
on HRPLs.
Our pipeline requires minimal Excel-specific re-
sources and can be adapted to another LRPLs:
(I) Prepare the Function Library Excel For-
mula language contains 505 functions, yet many
are unused due to niche functionality or being out-
dated. We first collect a library of useful functions
and documentation. We gather publicly available
spreadsheets from the web, similarly to Fisher and
Rothermel (2005), and use them to estimate a dis-
tribution of real function usage. We sample the 100
most frequently used functions and download their
documentation from public Microsoft help pages.
(II) Sample Relevant Data Contexts Following
(Zhang et al., 2025) we seek to ground our syn-
thetic training data in realistic data tables resem-
bling the target evaluation format. For simplicity
of the method we seek easily accessible data. The
1WikiTableQuestions (Pasupat and Liang, 2015) contains
tables from Wikipedia annotated by humans with questions
and answers in natural language.
2A financial table-QA dataset (Zhu et al., 2021).
2

WikiTableQuestions dataset contains real data ta-
bles extracted from Wikipedia, and we randomly
sample 10 tables per function from the training
split3. We query our teacher model GPT-4o to
pick a good table for each function, such that the
function may be executable on the table. See ap-
pendix A for prompt and further details. Manual
inspection of table assignments shows good quality,
and GPT-4o matches numeric functions to numeric
data, string functions to string data, etc.
(III) Generate Synthetic Problems We gener-
ate synthetic demonstrations of function semantics,
systematically showing the effects of individual ar-
guments. We query GPT-4o ( Q()) to produce data
samples demonstrating a target function fiwith
documentation diand table ti:
Q(fi, di, ti)→Si
Si={(q, e, a )j}j∈1..|Si|
Even the teacher model may not be an expert in
the LRPL, so diprovides the semantics of fi, and
the table tiprovides grounding to generate natural
questions. For one input, teacher GPT-4o produces
an output set of samples Si. Each sample contains
a question qin natural language, which requires
the execution of fiontito solve. We also generate
a step by step explanation eof how to solve the
problem, and an executable answer formula a.
We instruct the teacher to rotate through each
function argument and produce one demonstration
for each (resulting in the set Si). This allows us
to demonstrate the usage of individual optional
arguments. Appendix B shows examples.
(IV) Validate Generations Each generated sam-
ple constitutes a textbook-quality demonstration
of basic function semantics in the target domain.
However, since samples are synthetically gener-
ated using an AI model, they are unverified for cor-
rectness. We improve data quality by employing
post-generation validation. First, for each sample
we execute the generated answer formula a, and
discard any sample which fails to execute. Second,
we feed generated samples back into the teacher
model and generate parallel solutions in Python to
the same questions. We do not know the correct
answer to a synthetic question without human label-
ing, but we assume Python (as an HRPL) will have
the most reliable generations (Joel et al., 2024). We
3Our test set uses the heldout test split with separate tables.keep all samples with matching Excel and Python
executed values. See Appendix C for details.
4 Enhancing Low-Resource Generation
4.1 Models
We selected student model families that are (1)
open models, (2) available in multiple sizes, and
most importantly, (3) have corresponding code-
finetuned versions.
Qwen 2.5 A recent family with notable perfor-
mance across tasks, including analytical reasoning
(Yang et al., 2025). We use Qwen2.5 3B, 14B, and
Qwen2.5-Coder 3B, 14B.
Llama 2 A popular family with corresponding
code-finetuned models (Touvron et al., 2023). We
use Llama2 7B, 13B, and CodeLlama 7B, 13B.
4.2 RAG
RAG (Retrieval-Augmented Generation) enhances
LLMs performance by retrieving relevant informa-
tion from external sources and incorporating it into
the model’s prompts. In the RAG Allsetting, we
provide the student models with the 100 most fre-
quently used Excel function signatures and their
descriptions (see Section 3) before each question.
To set an upper bound, RAG Oracle replaces the 100
functions with only those included in the oracle Ex-
cel solution for each question (see Section 2).
4.3 Fine-tuning
We generate 6,440 quality-validated samples using
the pipeline described in Section 3, covering the top
100 most used Excel functions and their argument
permutations. We then finetune the student models
introduced in Section 4.1 on the synthetically gen-
erated data ( FTSyn−QA). The training is performed
until performance peaks on a heldout dataset of 100
questions from WikiTQ dev (typically within 1-3
epochs). For details, see Appendix D.
We also introduce two finetuning baselines: in
FTDocwe finetune student models on the raw
top 100 most used Excel function documentation,
which includes simple examples paired with toy
tables. However, since the function documentation
does not align with the data format in the down-
stream task, WikiTQ and TAT-QA, FTDoc−QAfine-
tunes the student models on the same documenta-
tion examples, but reformatted into a QA format
using GPT-4o. See Appendix E for more details.
3

Base Model RAG All RAG Oracle FTDoc FTDoc−QA FTSyn−QA
GPT-4o 79.19 78.25 84.19 - - -
Qwen2.5-coder 3B 15.34 13.77 18.94 13.62 15.34 28.64
Qwen2.5-coder 14B 46.95 44.76 53.52 40.22 46.95 54.93
Qwen2.5 3B 14.87 12.21 17.84 13.62 14.55 20.81
Qwen2.5 14B 50.70 47.57 49.45 46.64 49.92 51.02
CodeLlama-Instruct 7B 0.47 0.31 0.47 0.78 9.23 4.85
CodeLlama-Instruct 13B 10.80 13.15 16.12 12.21 12.83 21.28
Llama2 7B 0.63 0.16 1.10 0.63 3.91 8.61
Llama2 13B 1.72 0.31 1.56 0.47 4.54 15.49
Table 1: Evaluation results on our subset of WikiTableQuestions. Execution Match (EM) measures the percentage
of programs which execute to the correct answer. We test (a) base models; (b) RAG settings: all function signatures
(RAG All) and oracle-retrieved signatures ( RAG Oracle ); and (c) finetuned models: using function documentation
(FTDoc), and QA-formatted documentation ( FTDoc−QA), and synthetic problems ( FTSyn−QA). Finetuning with
synthetic problems is preferable to RAG or finetuning with purely documentation-based data.
5 Results
Table 1 shows our results. For base models there is
no clear advantage of code pretraining in HRPLs
when applied directly to Excel, an LRPL.
In the RAG Allsetting all models except for
CodeLlama 13B suffer when given all function
signatures, likely because the long context makes
it harder to identify the most relevant function.
However, optimal retrievals in RAG Oracle improve
most models, showing the upper-bound of RAG as-
sisted by a powerful reasoning model, o1. Still,
compared to Base Models , the boost is marginal.
In the finetuned FTDocsetting, "continued train-
ing" on raw function documentation often harms
performance. However, restructuring this into QA
format ( FTDoc−QA) can be useful. Yet compared
toBase Models the improvement is also marginal.
We achieve a significant performance boost in
nearly all models over Base Models with finetuning
on our synthetically generated data in FTSyn−QA.
The finetuned models even outperform correspond-
ingRAG Oracle settings, where the functions ex-
pected to be used are provided as hints by o1. See
Figure 1 for a showcase.
Most code-specialized models (* -coder ) enjoy a
greater boost from synthetic data finetuning com-
pared to non-specialized counterparts ( w/o -coder ).
This suggests that previously learned coding skills
can aid transfer to new programming languages.
We observe similar trends when evaluating mod-
els on TAT-QA. See Appendix F for details.
5.1 Analysis of Model Generations
We randomly sample 100 problems from the Wik-
iTQ test set and conduct a qualitative analysis of themodel generations produced by Qwen2.5-Coder
3BBase Model andFTSyn−QA.4Table 2 demon-
strates that fine-tuning on synthetic data primarily
reduces errors in table reading (-22) and formula
syntax (-8), while preserving the model’s ability
in planning and function selection. The small in-
crease in planning errors is attributed to the model’s
tendency to prioritize single-function formula gen-
eration. Since the synthetic data focuses on the use
of individual functions, we observe that 64.3% of
the solutions generated by FTSyn−QAinvolve a
single function, compared to 44.4% in Base . See
Appendix G for further details and analysis.
Result Category Base Model FTSyn−QA
Correct 13 34
Error: Plan Logic 25 33
Error: Function Choice 8 9
Error: Table Indexing 35 13
Error: Formula Syntax 9 1
Error: Other 10 10
Table 2: Analysis of Qwen2.5-Coder 3B base/finetuned.
6 Conclusion
We show that generating textbook-quality exam-
ples for a low resource programming language
can rapidly improve LLM generation through fine-
tuning, with benefits over RAG approaches. We
demonstrate on the Excel language, owing to its
popularity in practice, but sparsity of training data.
We posit that further work on individual languages
may benefit similarly, and we encourage more syn-
thetic data generations be included in LLM pretrain-
ing, especially long-tail phenomena like LRPLs.
4This coder model was chosen for its large perf. boost.
4

Limitations
We demonstrated our methods on a single LRPL,
though our findings should apply to any low-
resource programming domain. We also focused
on learning just the function library in our exam-
ple LRPL, a more realistic target for generating
textbook-quality data than the infinite space of pos-
sible programs, so we cannot compare to the chal-
lenges or benefits of generating complex data. Fur-
ther, due to computational costs we only investi-
gated methods for pruning bad synthetic genera-
tions, and not any methods for fixing those bad
generations. This could provide valuable future
work.
References
Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed
Awadallah, Ammar Ahmad Awan, Nguyen Bach,
Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat
Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck,
Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav
Chaudhary, Dong Chen, Dongdong Chen, Weizhu
Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng,
Parul Chopra, Xiyang Dai, Matthew Dixon, Ro-
nen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao,
Min Gao, Amit Garg, Allie Del Giorno, Abhishek
Goswami, Suriya Gunasekar, Emman Haider, Jun-
heng Hao, Russell J. Hewett, Wenxiang Hu, Jamie
Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi,
Xin Jin, Nikos Karampatziakis, Piero Kauffmann,
Mahoud Khademi, Dongwoo Kim, Young Jin Kim,
Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi
Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui
Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu,
Weishung Liu, Xiaodong Liu, Chong Luo, Piyush
Madan, Ali Mahmoudzadeh, David Majercak, Matt
Mazzola, Caio César Teodoro Mendes, Arindam Mi-
tra, Hardik Modi, Anh Nguyen, Brandon Norick,
Barun Patra, Daniel Perez-Becker, Thomas Portet,
Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang
Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy,
Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil
Salim, Michael Santacroce, Shital Shah, Ning Shang,
Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia
Song, Masahiro Tanaka, Andrea Tupini, Praneetha
Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan
Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel
Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia
Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu,
Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang,
Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu,
Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen
Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan
Zhang, and Xiren Zhou. 2024. Phi-3 technical report:
A highly capable language model locally on your
phone. Preprint , arXiv:2404.14219.
Marc Fisher and Gregg Rothermel. 2005. The eu-ses spreadsheet corpus: a shared resource for sup-
porting experimentation with spreadsheet depend-
ability mechanisms. SIGSOFT Softw. Eng. Notes ,
30(4):1–5.
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio
César Teodoro Mendes, Allie Del Giorno, Sivakanth
Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo
de Rosa, Olli Saarikivi, Adil Salim, Shital Shah,
Harkirat Singh Behl, Xin Wang, Sébastien Bubeck,
Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee,
and Yuanzhi Li. 2023. Textbooks are all you need.
Preprint , arXiv:2306.11644.
Or Honovich, Thomas Scialom, Omer Levy, and Timo
Schick. 2023. Unnatural instructions: Tuning lan-
guage models with (almost) no human labor. In
Proceedings of the 61st Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers) , pages 14409–14428, Toronto, Canada.
Association for Computational Linguistics.
Sathvik Joel, Jie JW Wu, and Fatemeh H. Fard.
2024. A survey on llm-based code generation for
low-resource and domain-specific programming lan-
guages. Preprint , arXiv:2410.03981.
Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia
Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine
Jernite, Margaret Mitchell, Sean Hughes, Thomas
Wolf, Dzmitry Bahdanau, Leandro von Werra, and
Harm de Vries. 2022. The stack: 3 tb of permissively
licensed source code. Preprint .
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Fed-
erico Cassano, Joel Lamy-Poirier, Nouamane Tazi,
Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei,
Tianyang Liu, Max Tian, Denis Kocetkov, Arthur
Zucker, Younes Belkada, Zijian Wang, Qian Liu,
Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-
Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue
Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade,
Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su,
Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai,
Niklas Muennighoff, Xiangru Tang, Muhtasham
Oblokulov, Christopher Akiki, Marc Marone, Cheng-
hao Mou, Mayank Mishra, Alex Gu, Binyuan Hui,
Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas
Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten
Scholak, Sebastien Paquet, Jennifer Robinson, Car-
olyn Jane Anderson, Nicolas Chapados, Mostofa Pat-
wary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz
Ferrandis, Lingming Zhang, Sean Hughes, Thomas
Wolf, Arjun Guha, Leandro von Werra, and Harm
de Vries. 2024. Starcoder 2 and the stack v2: The
next generation. Preprint , arXiv:2402.19173.
OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer,
Adam Richardson, Ahmed El-Kishky, Aiden Low,
Alec Helyar, Aleksander Madry, Alex Beutel, Alex
Carney, Alex Iftimie, Alex Karpenko, Alex Tachard
Passos, Alexander Neitz, Alexander Prokofiev,
Alexander Wei, Allison Tam, Ally Bennett, Ananya
Kumar, Andre Saraiva, Andrea Vallone, Andrew Du-
berstein, Andrew Kondrich, Andrey Mishchenko,
5

Andy Applebaum, Angela Jiang, Ashvin Nair, Bar-
ret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin
Sokolowsky, Boaz Barak, Bob McGrew, Borys Mi-
naiev, Botao Hao, Bowen Baker, Brandon Houghton,
Brandon McKinzie, Brydon Eastman, Camillo Lu-
garesi, Cary Bassin, Cary Hudson, Chak Ming Li,
Charles de Bourcy, Chelsea V oss, Chen Shen, Chong
Zhang, Chris Koch, Chris Orsinger, Christopher
Hesse, Claudia Fischer, Clive Chan, Dan Roberts,
Daniel Kappler, Daniel Levy, Daniel Selsam, David
Dohan, David Farhi, David Mely, David Robinson,
Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Free-
man, Eddie Zhang, Edmund Wong, Elizabeth Proehl,
Enoch Cheung, Eric Mitchell, Eric Wallace, Erik
Ritter, Evan Mays, Fan Wang, Felipe Petroski Such,
Filippo Raso, Florencia Leoni, Foivos Tsimpourlas,
Francis Song, Fred von Lohmann, Freddie Sulit,
Geoff Salmon, Giambattista Parascandolo, Gildas
Chabot, Grace Zhao, Greg Brockman, Guillaume
Leclerc, Hadi Salman, Haiming Bao, Hao Sheng,
Hart Andrin, Hessam Bagherinezhad, Hongyu Ren,
Hunter Lightman, Hyung Won Chung, Ian Kivlichan,
Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte,
Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina
Kofman, Jakub Pachocki, James Lennon, Jason Wei,
Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu,
Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero
Candela, Joe Palermo, Joel Parish, Johannes Hei-
decke, John Hallman, John Rizzo, Jonathan Gordon,
Jonathan Uesato, Jonathan Ward, Joost Huizinga,
Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Ka-
rina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood,
Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu,
Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad,
Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho,
Liam Fedus, Lilian Weng, Linden Li, Lindsay Mc-
Callum, Lindsey Held, Lorenz Kuhn, Lukas Kon-
draciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd,
Maja Trebacz, Manas Joglekar, Mark Chen, Marko
Tintor, Mason Meyer, Matt Jones, Matt Kaufer,
Max Schwarzer, Meghan Shah, Mehmet Yatbaz,
Melody Y . Guan, Mengyuan Xu, Mengyuan Yan,
Mia Glaese, Mianna Chen, Michael Lampe, Michael
Malek, Michele Wang, Michelle Fradin, Mike Mc-
Clay, Mikhail Pavlov, Miles Wang, Mingxuan Wang,
Mira Murati, Mo Bavarian, Mostafa Rohaninejad,
Nat McAleese, Neil Chowdhury, Neil Chowdhury,
Nick Ryder, Nikolas Tezak, Noam Brown, Ofir
Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins,
Patrick Chao, Paul Ashbourne, Pavel Izmailov, Pe-
ter Zhokhov, Rachel Dias, Rahul Arora, Randall
Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Mi-
yara, Reimar Leike, Renny Hwang, Rhythm Garg,
Robin Brown, Roshan James, Rui Shu, Ryan Cheu,
Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer,
Sam Toyer, Samuel Miserendino, Sandhini Agarwal,
Santiago Hernandez, Sasha Baker, Scott McKinney,
Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani
Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang,
Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji,
Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan
Clark, Tao Wang, Taylor Gordon, Ted Sanders, Te-
jal Patwardhan, Thibault Sottiaux, Thomas Degry,
Thomas Dimson, Tianhao Zheng, Timur Garipov,Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peter-
son, Tyna Eloundou, Valerie Qi, Vineet Kosaraju,
Vinnie Monaco, Vitchyr Pong, Vlad Fomenko,
Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech
Zaremba, Yann Dubois, Yinghai Lu, Yining Chen,
Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yun-
yun Wang, Zheng Shao, and Zhuohan Li. 2024a.
Openai o1 system card. Preprint , arXiv:2412.16720.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, Ir-
wan Bello, Jake Berdine, Gabriel Bernadett-Shapiro,
Christopher Berner, Lenny Bogdonoff, Oleg Boiko,
Madelaine Boyd, Anna-Luisa Brakman, Greg Brock-
man, Tim Brooks, Miles Brundage, Kevin Button,
Trevor Cai, Rosie Campbell, Andrew Cann, Brittany
Carey, Chelsea Carlson, Rory Carmichael, Brooke
Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully
Chen, Ruby Chen, Jason Chen, Mark Chen, Ben
Chess, Chester Cho, Casey Chu, Hyung Won Chung,
Dave Cummings, Jeremiah Currier, Yunxing Dai,
Cory Decareaux, Thomas Degry, Noah Deutsch,
Damien Deville, Arka Dhar, David Dohan, Steve
Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti,
Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix,
Simón Posada Fishman, Juston Forte, Isabella Ful-
ford, Leo Gao, Elie Georges, Christian Gibson, Vik
Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-
Lopes, Jonathan Gordon, Morgan Grafstein, Scott
Gray, Ryan Greene, Joshua Gross, Shixiang Shane
Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris,
Yuchen He, Mike Heaton, Johannes Heidecke, Chris
Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele,
Brandon Houghton, Kenny Hsu, Shengli Hu, Xin
Hu, Joost Huizinga, Shantanu Jain, Shawn Jain,
Joanne Jang, Angela Jiang, Roger Jiang, Haozhun
Jin, Denny Jin, Shino Jomoto, Billie Jonn, Hee-
woo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Ka-
mali, Ingmar Kanitscheider, Nitish Shirish Keskar,
Tabarak Khan, Logan Kilpatrick, Jong Wook Kim,
Christina Kim, Yongjik Kim, Jan Hendrik Kirch-
ner, Jamie Kiros, Matt Knight, Daniel Kokotajlo,
Łukasz Kondraciuk, Andrew Kondrich, Aris Kon-
stantinidis, Kyle Kosic, Gretchen Krueger, Vishal
Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan
Leike, Jade Leung, Daniel Levy, Chak Ming Li,
Rachel Lim, Molly Lin, Stephanie Lin, Mateusz
Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue,
Anna Makanju, Kim Malfacini, Sam Manning, Todor
Markov, Yaniv Markovski, Bianca Martin, Katie
Mayer, Andrew Mayne, Bob McGrew, Scott Mayer
McKinney, Christine McLeavey, Paul McMillan,
Jake McNeil, David Medina, Aalok Mehta, Jacob
Menick, Luke Metz, Andrey Mishchenko, Pamela
Mishkin, Vinnie Monaco, Evan Morikawa, Daniel
Mossing, Tong Mu, Mira Murati, Oleg Murk, David
Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak,
Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh,
Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex
Paino, Joe Palermo, Ashley Pantuliano, Giambat-
6

tista Parascandolo, Joel Parish, Emy Parparita, Alex
Passos, Mikhail Pavlov, Andrew Peng, Adam Perel-
man, Filipe de Avila Belbute Peres, Michael Petrov,
Henrique Ponde de Oliveira Pinto, Michael, Poko-
rny, Michelle Pokrass, Vitchyr H. Pong, Tolly Pow-
ell, Alethea Power, Boris Power, Elizabeth Proehl,
Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh,
Cameron Raymond, Francis Real, Kendra Rimbach,
Carl Ross, Bob Rotsted, Henri Roussez, Nick Ry-
der, Mario Saltarelli, Ted Sanders, Shibani Santurkar,
Girish Sastry, Heather Schmidt, David Schnurr, John
Schulman, Daniel Selsam, Kyla Sheppard, Toki
Sherbakov, Jessica Shieh, Sarah Shoker, Pranav
Shyam, Szymon Sidor, Eric Sigler, Maddie Simens,
Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin
Sokolowsky, Yang Song, Natalie Staudacher, Fe-
lipe Petroski Such, Natalie Summers, Ilya Sutskever,
Jie Tang, Nikolas Tezak, Madeleine B. Thompson,
Phil Tillet, Amin Tootoonchian, Elizabeth Tseng,
Preston Tuggle, Nick Turley, Jerry Tworek, Juan Fe-
lipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya,
Chelsea V oss, Carroll Wainwright, Justin Jay Wang,
Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei,
CJ Weinmann, Akila Welihinda, Peter Welinder, Ji-
ayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner,
Clemens Winter, Samuel Wolrich, Hannah Wong,
Lauren Workman, Sherwin Wu, Jeff Wu, Michael
Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu,
Qiming Yuan, Wojciech Zaremba, Rowan Zellers,
Chong Zhang, Marvin Zhang, Shengjia Zhao, Tian-
hao Zheng, Juntang Zhuang, William Zhuk, and Bar-
ret Zoph. 2024b. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Panupong Pasupat and Percy Liang. 2015. Compo-
sitional semantic parsing on semi-structured tables.
Preprint , arXiv:1508.00305.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton
Ferrer, Moya Chen, Guillem Cucurull, David Esiobu,
Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, An-
thony Hartshorn, Saghar Hosseini, Rui Hou, Hakan
Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa,
Isabel Kloumann, Artem Korenev, Punit Singh Koura,
Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Di-
ana Liskovich, Yinghai Lu, Yuning Mao, Xavier Mar-
tinet, Todor Mihaylov, Pushkar Mishra, Igor Moly-
bog, Yixin Nie, Andrew Poulton, Jeremy Reizen-
stein, Rashi Rungta, Kalyan Saladi, Alan Schelten,
Ruan Silva, Eric Michael Smith, Ranjan Subrama-
nian, Xiaoqing Ellen Tan, Binh Tang, Ross Tay-
lor, Adina Williams, Jian Xiang Kuan, Puxin Xu,
Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan,
Melanie Kambadur, Sharan Narang, Aurelien Ro-
driguez, Robert Stojnic, Sergey Edunov, and Thomas
Scialom. 2023. Llama 2: Open foundation and fine-
tuned chat models. Preprint , arXiv:2307.09288.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu,
Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng
Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tian-
hao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren,
Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang,
Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zihan Qiu. 2025. Qwen2.5 technical report. Preprint ,
arXiv:2412.15115.
Kang Min Yoo, Dongju Park, Jaewook Kang, Sang-Woo
Lee, and Woomyoung Park. 2021. GPT3Mix: Lever-
aging large-scale language models for text augmen-
tation. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2021 , pages 2225–2239,
Punta Cana, Dominican Republic. Association for
Computational Linguistics.
Yueheng Zhang, Xiaoyuan Liu, Yiyou Sun, Atheer Al-
harbi, Hend Alzahrani, Basel Alomair, and Dawn
Song. 2025. Can llms design good questions based
on context? Preprint , arXiv:2501.03491.
Wei Zhao, Zhitao Hou, Siyuan Wu, Yan Gao, Haoyu
Dong, Yao Wan, Hongyu Zhang, Yulei Sui, and
Haidong Zhang. 2024. NL2Formula: Generating
spreadsheet formulas from natural language queries.
InFindings of the Association for Computational Lin-
guistics: EACL 2024 , pages 2377–2388, St. Julian’s,
Malta. Association for Computational Linguistics.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-
Seng Chua. 2021. TAT-QA: A question answering
benchmark on a hybrid of tabular and textual con-
tent in finance. In Proceedings of the 59th Annual
Meeting of the Association for Computational Lin-
guistics and the 11th International Joint Conference
on Natural Language Processing (Volume 1: Long
Papers) , pages 3277–3287, Online. Association for
Computational Linguistics.
A Sampling Tables
We prompt GPT-4o to choose the best table with
which to demonstrate a particular Excel function
fi. In the prompt, we insert the function, its docu-
mentation di, and 10 randomly selected tables tj
forj∈[1,10]. See Figure 2 for the prompt.
We generate samples in batches. In one batch,
we sample a table for each function in our library
of top 100 most used functions. Then, we generate
a sample for all 100 (function, table) tuples. We
repeat this until we have generated a suitable-sized
curriculum; student models will thus train on a
given function fiacross many randomly sampled
tables.
B Sampling Questions
We prompt the teacher model GPT-4o to generate
textbook-quality tutorials in a QA format, demon-
7

Teacher Prompt: Choose a Table for a Given Function
## General Instruction:
You are a helpful assistant to a data scientist who is learning to use Excel. You are preparing
a tutorial for the data scientist about a specific Excel function. ,→
Below is an Excel function to be demonstrated. In order to demonstrate the function, we need to
find a data table with information that is useful to query with the function. ,→
Given the function, its documentation, and a list of tables, choose the table that would be the
most interesting to use to demonstrate the function. The demonstration will need to show
how the function can be used to query the table as well as how its arguments work.,→
,→
## Output:
First write an explanation of how each table could be used to demonstrate the function.
Then, choose which table is best suited to demonstrate all the functionality of the function
including its arguments. ,→
Last, write the number of the table that you have chosen on its own line.
## Function:
{func}
## Documentation:
{docs}
## Tables:
{tables}
Figure 2: We match a table to each function by querying GPT-4o. First, we instruct the model to generate reasoning
for each table about how it could be suitable, then we instruct it to choose the table with the best reasoning.
strating how to use a given function. A key aspect
of the prompt is that we instruct the teacher to gen-
erate a set of examples, one demonstrating how to
use each argument slot of the target function. See
Figure 3 for the prompt, and Figure 4 for a sam-
ple output for the function MATCH(). We instruct
the teacher to format tutorial information in JSON,
which allows us to compile the training tutorials
according to e.g. a Jinja template. See Figure 5
for an example of a final, compiled tutorial of the
MATCH function.
C Data Validation
The teacher model generates sample problems with
Excel reasoning and solutions. The problems gen-
erated by the teacher model can be considered as tu-
ples in the format (table, query, Excel formula, Ex-
cel execution) (we execute the formula ourselves).
We aim to format these tuples into training data
after filtering them for quality. In this filtering step,
we generate a parallel solution to each problem in
Python, a high resource programming language in
which we expect model solutions to be of high qual-
ity. We give nearly the same prompt as in model
evaluations, but cast each sample as a Python prob-
lem using a Pandas DataFrame instead of an Ex-cel spreadsheet. We collect the model-generated
Python code and execute it so that we may compare
the Python and Excel executed values.
Because Pandas DataFrames have subtle differ-
ences to spreadsheets (e.g. they are 0-indexed
whereas Excel is 1-indexed), we develop a system
of basic rules for determining equality between ex-
ecuted values. We are able to accept around 50%
of synthetic generations after Python-validation.
We also show that this validation is useful. We
finetune a Qwen2.5-Coder 14B model on equal
amounts of validated and un-validated data (6,440
samples in each). We observe that when trained on
unvalidated data, the model scores 52.11% on the
downstream WikiTQ test set, compared to 54.93%
when finetuned on an equal amount of validated
data. This demonstrates (a) unvalidated data has
great potential even with just simple filtering out of
un-executable samples, and (b) parallel generation
of solutions in an HRPL like Python can provide
useful supervision for higher-quality assurance, re-
sulting in downstream task improvement.
D Finetuning Hyperparameters
We finetune all models using the hyperparameters
shown in Table 3. We chose these by doing hy-
8

perparameter search over the given values using
Qwen2.5-coder 3B and choosing the parameters
which yielded the highest score on the heldout vali-
dation dataset. We then used these settings to train
all our models, and applied these finetuned models
to the test sets.
Hyperparameter Value Search Range
Batch size 4 {2, 4, 8, 16, 32}
Learning Rate 5e-5 {1e-5, 5e-5, 1e-4, 5e-4}
LoRA r 64 {32, 64, 128}
LoRA α 1
Max train epochs 6
Early stop patience 3
LR scheduler cosine {linear, cosine}
Warmup ratio 0.5
Table 3: Hyperparameter choices for model training.
EFTDoc−QABaseline
We design this baseline as a control, training on
the information content of function documentation
but reformatted into the QA format of the target
domain. Therefore, improvement over this base-
line may be attributed to content, rather than for-
mat. We use a strong LLM (GPT-4o) to do the
reformatting, which involves restructuring (natural
language) content only, and not synthetically gen-
erating novel content. We use the prompt shown
in Figure 6. Since function documentation often
contains many sections including one for examples,
we add a worked example of a simple function,
ABS, to demonstrate how to reformat questions
and answers found in example tables.
F TAT-QA Results
We show results of evaluating base models and
finetuned models on TAT-QA in Figure 4. Our fine-
tuned models train on validated synthetic problems
in QA format, matching the test-time task format
the same as in experiments on WikiTQ in §4.
We observe similar trends on TAT-QA as we
do on WikiTQ. Large models perform better than
small models, and our finetuning data is able to
improve 7 of 8 models significantly. We notice that
in general, the effect size of finetuning is smaller
when evaluated on TAT-QA, and we look to the
lower base model performances to explain this.
While TAT-QA is a similarly-formatted dataset in
which data tables are queried in natural language,
it is in a very specialized domain. Financial data is
likely to be more scarce in general LLM pretrain-
ing data, so we expect a priori that performancemay be lower on this dataset, and the lower base
model performances confirm this.
We also note that it is difficult to say if
code-specialized models outperform non-code-
specialized models on this task, since they perform
very similarly to each other. We believe this is due
to the same problem as above: code-specialization
would not have improved models’ financial reason-
ing ability, so our further code finetuning in the
general domain may not show particular benefits to
one type of model or another, either, on this dataset.
Base Model FTSyn−QA
GPT-4o 77.78 -
Qwen2.5-coder 3B 5.88 8.06
Qwen2.5-coder 14B 14.37 14.60
Qwen2.5 3B 6.75 11.11
Qwen2.5 14B 15.47 15.03
CodeLlama-Instruct 7B 0.44 2.18
CodeLlama-Instruct 13B 3.49 7.41
Llama2 7B 0.00 3.92
Llama2 13B 1.09 6.54
Table 4: Evaluation results on our subset of TAT-QA.
Execution Match (EM) results are shown.
G Further Analysis of Generations
In our error analysis in §5.1, we use the categoriza-
tion guidelines in Table 5 to analyze the generations
from Qwen models on our WIkiTQ test set.
Result Category Guideline
Correct The model generation executes
to the correct answer
Error: Plan Logic If the CoT is wrong
Error: Function Choice If the CoT is correct, but it de-
cides to use the wrong function,
or is missing some functions
Error: Formula Syntax If the function is correct, but
the way of using the function is
wrong
Error: Table Indexing If the way of using the func-
tion is correct, and we change
the cell/row/column numbers
we would get the correct answer
Error: Other Any other error, such as correct
plan but incorrect formula se-
mantics
Table 5: Annotation guideline for error classifications.
Next, we turn to analyzing model performance
before and after our function finetuning.
We first show the percentage of model-generated
formulas which consist of a single function call, in
9

Improvements Regressions
Samples Percent Samples Percent
Qwen2.5-coder 3B 39/131 29.77 5/46 10.87
Qwen2.5-coder 14B 39/143 27.27 7/98 7.14
Qwen2.5 3B 22/93 23.66 2/55 3.64
Qwen2.5 14B 13/75 17.33 16/73 21.92
Table 6: Improvements and Regressions in same, single-function prediction due to finetuning Qwen models. Models
master individual functions and often improve in generating single-function calls, usually with little regression.
Table 7. We observe that all models increased the
number of single function predictions after tuning,
however this effect is more noticeable in smaller
models.
Base Model FTSyn−QA
Qwen2.5-coder 3B 44.44 64.32
Qwen2.5-coder 14B 46.17 59.00
Qwen2.5 3B 40.22 73.08
Qwen2.5 14B 40.22 40.85
Table 7: Percentage of generated formulas consisting of
a single function.
Of each model’s improvements (incorrect before
tuning, correct after), many generations produce
the same function before and after. This shows
learned mastery of these functions, where finetuned
models produce better formulas using the same
function. We also observe minimal regressions
when predicting the same function (correct usage
before tuning, incorrect after). See Table 6.
10

Teacher Prompt: Generate Examples
## General Instruction:
You are a helpful assistant to a data scientist who is learning to use Excel.
You are tasked with creating a tutorial of examples demonstrating the functionality of F, given F 's reference
documentation, as well as a random data table T taken from Wikipedia.,→
The tutorial should contain at least one example demonstrating each of F 's argument slots, in order to thoroughly describe
how F works.,→
## Task:
First, analyze the documentation of the function F to understand what each argument does. Write a brief explanation of
what each argument is used for, including whether it is required or optional.,→
Format the explanation as markdown:
```markdown
Function: F
- arg1 <required>: explanation of arg1
- arg2 <required>: explanation of arg2
- arg3 <optional>: explanation of arg3
```
Second, write a series of examples demonstrating the use of F on the table T. Each example should contain:
1. The function F
2. The argument A being demonstrated
3. A natural language query Q which requires the use of F and A executed on the table T to compute a solution. Write the
query in a natural and realistic way, as if an interested person were trying to analyze the data table to solve a
problem.,→
,→
Make the query specific so there is only one correct answer. For example, to demonstrate a string manipulation function,
the query Q should specify exactly how to format the output string so that a program can be written to do this.,→
4. A brief explanation of what F does in general (not related to the query Q or table T).
5. A step by step explanation of how to use F and A to solve the query Q given T. When explaining the steps, only use
values mentioned in the query Q or references into the table T. Use the syntax section of the function F 's
documentation to explain how the arguments are used.,→
,→
6. The answer to the query Q. After any reasoning, restate the answer on its own line at the end, e.g. "True", "False",
"5", etc.,→
7. The final Excel formula using F and A to solve the query Q
8. Write the parameter name and required/optional for each of the final arguments given to F as a list, e.g. "param1
<required>", "param2 <optional>", etc.,→
Write examples which demonstrate the required arguments, then examples for each of the optional arguments.
Format the examples as a JSON list according to the following structure:
```json
[
{{
"func": str,
"demo_argument": str <required/optional>,
"query": str,
"func_explanation": str,
"step_by_step": [ str, ... ],
"answer": str,
"formula": str,
"structure": [ str <required/optional>, ... ]
}},
...
]
```
For the Excel formula, use the following format:
"=FUNCTION(ARGUMENTS)"
## Function:
{func}
## Documentation:
{docs}
## Random Table:
In Excel tables, the first row is usually reserved for column headers. The first column is usually reserved for row
headers. For example, the data starts in A2.,→
Larger tables may be excerpted here. If so, the first and last rows of the table will be shown, with an ellipsis (...) in
between representing the hidden middle rows.,→
Remember that NaN values in Excel may be written in the table as "nan".
{table}
## Tutorial:
Figure 3: We instruct the teacher model GPT-4o to generate multiple tutorials for a given function, demonstrating
each function argument. The generation is in JSON, which is then compiled into corresponding textbook demonstra-
tions for the student model using a simple Jinja template. See Figure 4 for a sample JSON output, and Figure 5 for a
compiled tutorial.
11

Sample Teacher Output: MATCH Function Example
[
{
"func":"MATCH",
"demo_argument":"lookup_value <required>",
"query":"What is the position of the team 'Boston Red Sox 'in the list of teams?",
"func_explanation":"The MATCH function searches for a specified item in a range of cells and returns the relative
position of that item in the range.",,→
"step_by_step":[
"Identify the lookup_value, which is 'Boston Red Sox '.",
"Identify the lookup_array, which is the range A2:A8 containing the team names.",
"Use the MATCH function to find the position of 'Boston Red Sox 'in the range A2:A8."
],
"answer":"4",
"formula":"=MATCH(\"Boston Red Sox\", A2:A8, 0)",
"structure":[
"lookup_value <required>",
"lookup_array <required>",
"match_type <optional>"
],
"executed":[
{
"kind":"plain",
"value":4
}
]
},
{
"func":"MATCH",
"demo_argument":"lookup_array <required>",
"query":"Find the position of the team with 87 wins in the list of wins.",
"func_explanation":"The MATCH function searches for a specified item in a range of cells and returns the relative
position of that item in the range.",,→
"step_by_step":[
"Identify the lookup_value, which is 87.",
"Identify the lookup_array, which is the range B2:B8 containing the number of wins.",
"Use the MATCH function to find the position of 87 in the range B2:B8."
],
"answer":"4",
"formula":"=MATCH(87, B2:B8, 0)",
"structure":[
"lookup_value <required>",
"lookup_array <required>",
"match_type <optional>"
],
"executed":[
{
"kind":"plain",
"value":3
}
]
},
{
"func":"MATCH",
"demo_argument":"match_type <optional>",
"query":"What is the position of the team with the closest number of wins less than or equal to 90?",
"func_explanation":"The MATCH function searches for a specified item in a range of cells and returns the relative
position of that item in the range.",,→
"step_by_step":[
"Identify the lookup_value, which is 90.",
"Identify the lookup_array, which is the range B2:B8 containing the number of wins.",
"Set the match_type to 1 to find the largest value less than or equal to 90.",
"Use the MATCH function to find the position of the closest number of wins less than or equal to 90 in the
range B2:B8.",→
],
"answer":"3",
"formula":"=MATCH(90, B2:B8, 1)",
"structure":[
"lookup_value <required>",
"lookup_array <required>",
"match_type <optional>"
],
"executed":[
{
"kind":"plain",
"value":7
}
]
}
]
Figure 4: One query to the teacher model for the MATCH function yields three examples formatted as JSON.
We compile each example into plain text using a Jinja template, for training the student with a simple next-token
prediction objective. See Figure 5 for an example tutorial.12

Student Training Sample: Example of a MATCH Function Tutorial
## General Instruction:
You are a helpful assistant to a data scientist who is learning to use Excel.
Given a table of data and a user query, write a step-by-step explanation of how to use Excel to
solve the query using the table. Produce a final Excel formula that can be executed to
solve the query.,→
,→
## Table:
| | A | B | C | D | E | F |
|----|------|-----------|------|--------|--------|-------|
| 1 | Rank | Nation | Gold | Silver | Bronze | Total |
| 2 | 1 | Brazil | 13 | 18 | 12 | 43 |
| 3 | 2 | Argentina | 7 | 4 | 7 | 18 |
| 4 | 3 | Chile | 7 | 2 | 3 | 12 |
| 5 | 4 | Colombia | 5 | 5 | 4 | 14 |
| 6 | 5 | Venezuela | 4 | 6 | 6 | 16 |
| 7 | 6 | Uruguay | 1 | 1 | 0 | 2 |
| 8 | 7 | Peru | 0 | 1 | 0 | 1 |
| 9 | 8 | Panama | 0 | 0 | 2 | 2 |
| 10 | 8 | Bolivia | 0 | 0 | 2 | 2 |
| 11 | 10 | Paraguay | 0 | 0 | 1 | 1 |
## Query:
What is the position of the nation 'Chile 'in the list of nations?
## Reasoning:
The MATCH function searches for a specified item in a range of cells and returns the relative
position of that item in the range. ,→
Identify the lookup_value, which is 'Chile '.
Identify the lookup_array, which is the range B2:B11 containing the list of nations.
Use the MATCH function to find the position of 'Chile 'in the range B2:B11.
Since we are looking for an exact match, set match_type to 0.
## Formula:
```excel
=MATCH("Chile", B2:B11, 0)
```
Figure 5: A textbook-quality demonstration of how to use the MATCH function to find the position of the string
“Chile” in a list, using an exact match. The teacher’s chain of thought reasoning is included to explain how to gather
information from the table and construct the final formula.
13

Teacher Prompt: Extract Examples and Reformat into QA ( FTDoc−QA)
In the following article about an Excel formula API, there is a section describing examples of using the API (each example
formula looks like "=FORMULA(arguments)"). The section contains one or multiple tables containing the examples.,→
Task:
Extract all examples from the tables and format them like this:
1. First, copy the description of the example formula.
2. Next, if the formula contains contains a cell reference (e.g. A2), then copy the portion of the table containing the
referred data (and not the example rows) so that the formula can be evaluated.,→
3. Then, copy the formula itself into a code block.
4. Last, copy the output of the formula below the code block.
If there are no examples present in the article, write "[No examples provided]".
Demonstration:
This article describes the formula syntax and usage of the ABS function in Microsoft Excel.
Description
Returns the absolute value of a number. The absolute value of a number is the number without its sign.
Syntax
ABS(number)
The ABS function syntax has the following arguments:
- Number <Required>: The real number of which you want the absolute value.
Example
Copy the table below, and paste into cell A1 in Excel. You may need to select any cells that contain formulas and press F2
and then Enter to make the formulas work. You may also want to make the columns wider to make your worksheet easier to
read.,→
,→
| | A | B | C |
|----|----------|----------------------|------------|
| 1 | Data | Unnamed: 1 | Unnamed: 2 |
| 2 | -4 | | |
| 3 | Formula | Description | Result |
| 4 | =ABS(2) | Absolute value of 2 | 2 |
| 5 | =ABS(-2) | Absolute value of -2 | 2 |
| 6 | =ABS(A2) | Absolute value of -4 | 4 |
See Also
Subtract numbers
Multiply and divide numbers in Excel
Calculate percentages
Expected output:
Absolute value of 2
```excel
=ABS(2)
```
>>> 2
-----
Absolute value of -2
```excel
=ABS(-2)
```
>>> 2
-----
| | A | B | C |
|----|----------|----------------------|------------|
| 1 | Data | Unnamed: 1 | Unnamed: 2 |
| 2 | -4 | | |
Absolute value of -4
```excel
=ABS(A2)
```
>>> 4
-----
Now it 's your turn! Please extract the examples from the following article:
{function_docs}
Figure 6: We query the teacher model (GPT-4o) to reformat a documentation page into QA examples by extracting
content and restructuring (not synthesizing new content). We work one example of the ABS function to demonstrate
to the teacher.
14