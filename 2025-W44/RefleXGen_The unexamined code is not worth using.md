# RefleXGen:The unexamined code is not worth using

**Authors**: Bin Wang, Hui Li, AoFan Liu, BoTao Yang, Ao Yang, YiLu Zhong, Weixiang Huang, Yanping Zhang, Runhuai Huang, Weimin Zeng

**Published**: 2025-10-27 05:28:32

**PDF URL**: [http://arxiv.org/pdf/2510.23674v1](http://arxiv.org/pdf/2510.23674v1)

## Abstract
Security in code generation remains a pivotal challenge when applying large
language models (LLMs). This paper introduces RefleXGen, an innovative method
that significantly enhances code security by integrating Retrieval-Augmented
Generation (RAG) techniques with guided self-reflection mechanisms inherent in
LLMs. Unlike traditional approaches that rely on fine-tuning LLMs or developing
specialized secure code datasets - processes that can be resource-intensive -
RefleXGen iteratively optimizes the code generation process through
self-assessment and reflection without the need for extensive resources. Within
this framework, the model continuously accumulates and refines its knowledge
base, thereby progressively improving the security of the generated code.
Experimental results demonstrate that RefleXGen substantially enhances code
security across multiple models, achieving a 13.6% improvement with GPT-3.5
Turbo, a 6.7% improvement with GPT-4o, a 4.5% improvement with CodeQwen, and a
5.8% improvement with Gemini. Our findings highlight that improving the quality
of model self-reflection constitutes an effective and practical strategy for
strengthening the security of AI-generated code.

## Full Text


<!-- PDF content starts -->

REFLEXGEN:THE UNEXAMINED CODE IS NOT WORTH USING
Bin Wang1, Hui Li1, AoFan Liu1, BoTao Yang1, Ao Yang1, YiLu Zhong1,
Weixiang Huang2, Yanping Zhang2, Runhuai Huang3, Weimin Zeng3
1School of Electronic and Computer Engineering,Peking University
2Apability & Platform Business Dept., China Mobile Internet Co.
3China Telecom Cloud Technology Co., Ltd.
ABSTRACT
Security in code generation remains a pivotal challenge
when applying large language models (LLMs). This pa-
per introduces RefleXGen, an innovative method that sig-
nificantly enhances code security by integrating Retrieval-
Augmented Generation (RAG) techniques with guided self-
reflection mechanisms inherent in LLMs. Unlike traditional
approaches that rely on fine-tuning LLMs or developing spe-
cialized secure code datasets—processes that can be resource-
intensive—RefleXGen iteratively optimizes the code gener-
ation process through self-assessment and reflection without
the need for extensive resources. Within this framework,
the model continuously accumulates and refines its knowl-
edge base, thereby progressively improving the security of
the generated code. Experimental results demonstrate that
RefleXGen substantially enhances code security across mul-
tiple models, achieving a 13.6% improvement with GPT-3.5
Turbo, a 6.7% improvement with GPT-4o, a 4.5% improve-
ment with CodeQwen, and a 5.8% improvement with Gemini.
Index Terms—code generation, security, large language
models, RAG
1. INTRODUCTION
Code generation technologies, which enable the creation
of target code via natural language descriptions or minimal
code prompts, significantly lower the barriers to software
development. They allow a broader range of non-experts
to engage in software development and substantially reduce
the workload for developers. Initially, code generation relied
heavily on heuristic rules or expert systems. While effective,
these methods often lacked flexibility and scalability[1]. Sub-
sequently, researchers began using static language models
and neural networks to establish mappings between codes,
which expanded the applications but were still limited in
functionality[2, 3].
THIS WORK IS SUPPORTED BY GUANGDONG PROVINCIAL
KEY LABORATORY OF ULTRA HIGH DEFINITION IMMERSIVE ME-
DIA TECHNOLOGY(GRANT NO. 2024B1212010006)
Corresponding authorWith the advent of LLMs based on the Transformer ar-
chitecture, an increasing number of LLMs have been trained
on extensive code corpora[4, 5, 6, 7, 8, 9]. These models can
generate code without the need for samples and have demon-
strated remarkable success across numerous code generation
tasks. These advanced large language models have signifi-
cantly propelled the evolution of code generation technolo-
gies. They are capable of generating, optimizing, and even
debugging code based on user requirements, thus markedly
enhancing software development efficiency and opening new
programming avenues for non-professional programmers.
According to the 2023 GitHub Annual Report, nearly all
developers (92%) are utilizing or experimenting with AI pro-
gramming tools, which have become powerful aids in acceler-
ating development cycles and boosting productivity[10, 11].
However, large language models for code completion and
generation have shortcomings [11, 6, 8]. Pre-trained on pub-
licly available datasets, the training code is not guaranteed to
be safe or reliable. Consequently, the generated code may
contain defects or vulnerabilities. These issues can cause se-
rious problems for users, such as low-quality code, compi-
lation failures, or security vulnerabilities—problems that are
more direct than hallucinations or errors in dialogue gener-
ation [12, 11]. Therefore, enhancing the ability of language
models to generate reliable and secure code is a significant
challenge in current research.
To effectively address security challenges in code gener-
ation, we have developed an innovative method called Re-
fleXGen. This approach enhances code generation by guid-
ing large language models to engage in self-reflection, cou-
pled with a knowledge base composed of the model’s own
historical thought records and secure code snippets. As a re-
sult, it significantly improves the security of the generated
code. Throughout this process, the model autonomously iden-
tifies and mitigates potential security risks, accumulates prac-
tices in secure coding, and progressively enriches the knowl-
edge base to guide the generation of future secure code. No-
tably, RefleXGen does not require updates to existing training
datasets nor fine-tuning of the model, and can be seamlessly
integrated into existing large-scale models. Verification onarXiv:2510.23674v1  [cs.SE]  27 Oct 2025

multiple proprietary and open-source models has confirmed
RefleXGen’s substantial effectiveness in enhancing the secu-
rity of code generation.
2. RELATED WORK
2.1. Code Generation
Code generation has a long history, traditionally defined as
finding programs within a programming language’s search
space that satisfy task-specific constraints [13]. However,
search-based methods often struggle due to the vastness
of the search space and the lack of formalized constraints.
With advancements in deep learning, new approaches have
emerged that generate programs from informal specifications
such as natural language descriptions, partial code, input-
output examples, or pseudocode [2, 14, 3]. Despite progress,
these methods are typically limited to generating shorter pro-
grams in domain-specific languages or single lines of code in
general-purpose languages.
Recently, transformer-based large language models have
revitalized code generation. Models like Codex [4] demon-
strate exceptional capability in auto-completing Python func-
tions based on function signatures and docstrings. CodeGen
[15] enhances program synthesis quality through multi-turn
interactions that refine user specifications. CodeT5 [8] intro-
duces automatic test case generation to improve code solution
selection. CodeRanker [16] presents a fault-aware ranking
model that predicts program correctness without execution,
effectively addressing code selection challenges.
2.2. Code Generation Security
The security of code generation has become a critical research
area in large language models. Extensive studies have ana-
lyzed and evaluated the security of these models, highlight-
ing vulnerabilities in generated code. Models like StarCoder
[5] and CodeLlama [17] have implemented specific security-
enhancing measures during training. Additionally, works like
SecurityEval [18] and SecuCoGen [19] focus on assessing
models’ ability to generate secure code. Techniques such as
SafeCoder [20], FRANC, and SVEN [11] enhance code gen-
eration security from different perspectives, introducing in-
novative mechanisms and algorithms to improve the safety of
generated code.
3. METHODOLOGY
The RefleXGen method integrates the concept of Self-
refinement with RAG technology, aiming to enhance the
safety of LLMs in code generation without the need for fine-
tuning the model itself. As illustrated in Figure 1, the work-
flow of this method encompasses two key phases and three
core operations. In the first phase, the code generation modelproduces initial code based on specific user requirements.
Subsequently, in the second phase, RefleXGen performs deep
reflection and iterative optimization on this initial code. Once
the code’s safety meets the predetermined standards, the re-
flective insights are updated into the safety knowledge base
to guide subsequent related tasks. The following discussion
will introduce the crucial operations involved.
Step1:①Initial code generation.In the stage, the system
is provided with an input code snippetx, a promptp gen, and
accesses the modelM. The code generation model then pro-
duces the initial outputy 0:
y0=M(p gen∥x)(1)
While this initial output generally meets the basic require-
ments outlined in the input, it may still present issues such as
poor reliability or contain latent security vulnerabilities that
necessitate further refinement.
Step2:Reflection and Optimization.In this step, the sys-
tem initially employs its model to introspect and determine
the presence of any potential defects in the output. Should the
output be defect-free, the system will proceed to display the
results directly. However, if defects are identified, the system
transitions into a phase of reflective iteration. The specific
steps involved in this phase are as follows:
②Knowledge-Driven Security Feedback:In this stage, Re-
fleXGen conducts a RAG query utilizing both the initial code
output and specific input requests, as outlined in Equations 2
and 3. The query is designed to uncover pertinent security
knowledge, including standards for secure coding and histor-
ical feedback. When the query identifies applicable security
practices and knowledge, the system integrates this informa-
tion with the initial input and the defined problem .
r0=Retrieve(x, y 0)(2)
y1=M(p gen∥x∥y 0∥r0)(3)
③Defect Fixing and Knowledge Integration: If the RAG
query fails to provide sufficient security knowledge, the sys-
tem proceeds to a thorough reflection and iterative repair pro-
cess, as outlined in Equation 4. This phase involves a criti-
cal assessment and enhancement of the code based on iden-
tified vulnerabilities and potential improvements. Once the
code fulfills all specified safety requirements, the refined se-
curity knowledge and the enhanced code are systematically
organized and stored within the secure knowledge base (sec.
RAG). Subsequently, the system reinitiates the first step to
verify the output, ensuring that the improvements effectively
address the initial shortcomings.
yt+1=M(p refine∥x∥y 0∥fb 0∥. . .∥y t∥fb t∥rt)(4)
UpdateRAG(x, y t+1)(5)

Fig. 1: The diagram presents the structured workflow of the ReflexGen methodology, segmented into three critical stages:
①Initial Code Generation,②Knowledge-Driven Security Feedback, and③Defect Fixing and Knowledge Integration. The
process initiates with the generation of initial code. If, upon introspection, the model discerns security deficiencies in the code,
it activates Step 2. This stage entails rigorous reflection and optimization to address and rectify vulnerabilities. Subsequently,
through a cyclical process of secure code production, insights derived from this reflective phase are systematically integrated
into the security knowledge base, thus promoting continual enhancements.
By reflecting on the code and incorporating historical
data, RefleXGen readjusts the code, repairs insecure parts,
and even introduces safer coding practices. The optimized
code not only meets the initial functional requirements but
also significantly enhances its security.
4. EXPERIMENT
4.1. Model Selection
Due to the limitations of smaller open-source and specialized
code-completion models in dialogue and reflective knowl-
edge assessment, we selected more comprehensive main-
stream models for our evaluation. These include prominent
commercial models like OpenAI’s GPT-3.5 Turbo and GPT-4,
Google’s Gemini, and the open-source model Qwen. These
models exhibit advanced code generation capabilities and
excel in managing dialogues, aligning well with our testing
criteria.
4.2. Datasets
To evaluate RefleXGen’s improvements in code generation
security and reliability, we selected challenging scenarios
from the most impactful Common Weakness Enumerations(CWEs). We used a dataset validated by He et al.[11], fea-
turing nine scenarios from MITRE’s top 25 most dangerous
software vulnerabilities. Each CWE scenario includes two
to three specific programming environments crafted by He et
al., eliminating low-quality prompts and replicating diverse
daily code completion tasks, making it a robust tool for as-
sessing models’ code security capabilities. These scenarios,
based on incomplete code prompts in C/C++ or Python, chal-
lenge the models to produce appropriate code completions,
highlighting their ability to handle incomplete inputs in real
programming environments.
4.3. Performance of RefleXGen
To ensure a fair comparison, we initially set the RAG con-
tent to empty, allowing RefleXGen to progressively generate
content during testing. We tracked several metrics: Sec. Rate
, Pass Rate, Eff. Total,Sec. Count,Unres. Count. To ob-
tain reliable data, we conducted five repeated experiments for
each model. CodeQL [21] was utilized to perform security
analysis and assessment. In each experiment, every scenario
was subjected to 25 task generations to average the results,
ensuring an objective assessment of each model’s generative
capabilities.
As shown in Table 1, RefleXGen demonstrated outstand-

Table 1: The Pass Rate (Pass Rate) refers to the percentage of correct outputs from valid inputs. The Security Rate (Sec. Rate)
reflects the percentage of successfully compiled tests that also meet security standards. The ”Efficiency Total” (Eff. Total)
indicates the number of successfully compiled tests within a CWE category, while the ”Security Count” (Sec. Count) shows
the number of tests that compiled successfully and met security standards, both out of 25 tests. Finally, the Unresolved Count
(Unres. Count) reflects the figure of tests that failed to compile.
ModelGPT3.5Turbo GPT4o CodeQwen1.5 Gemini1.0Pro
Base +RefleXGen Base +RefleXGen Base +RefleXGen Base +RefleXGen
Sec.Rate 75.5 89.1 (↑13.6) 92.3 99.0 (↑6.7) 83.7 88.2 (↑4.5) 80.2 86.0 (↑5.8)
Pass.Rate 97.6 95.8 (↓1.8) 94.2 100 (↑5.8) 86.7 69.8 (↓16.9) 92.2 83.6 (↓8.6)
Eff.Total 24.5 24.0 (↓0.5) 23.6 25.0 (↑1.4) 21.6 20.4 (↓1.2) 23.1 22.8 (↓0.3)
Sec.Count 19.5 22.3 (↑2.8) 21.9 24.7 (↑2.8) 17.9 19.4 (↑1.5) 19.1 21.2 (↑2.1)
Unres.Count 0.5 1.1 (↑0.6) 1.4 0 (↓1.4) 3.3 3.8 (↑0.5) 1.9 2.1 (↑0.2)
CWE220-py
CWE221-py
CWE7870-py
CWE7871-py
CWE1250-c
CWE1251-c0255075100GPT3.5TurboSec.Rate
CWE220-py
CWE221-py
CWE7870-py
CWE7871-py
CWE1250-c
CWE1251-c0255075100GPT4oSec.Rate
CWE220-py
CWE221-py
CWE7870-py
CWE7871-py
CWE1250-c
CWE1251-c0255075100CodeQwenSec.Rate
CWE220-py
CWE221-py
CWE7870-py
CWE7871-py
CWE1250-c
CWE1251-c0255075100GeminiSec.RateBaseline ReflexGen
Fig. 2: Sec.Rate Difference among Cases of RefleXGen
ing performance across four major models, effectively en-
hancing code security. Specifically, OpenAI’s GPT-3.5 Turbo
showed a 13.6% improvement in code safety, GPT-4 im-
proved by 6.7%, CodeQwen-1.5 by 4.5%, and Gemini-1.0-
pro achieved a 5.8% increase in security. These results indi-
cate that the RefleXGen method significantly reduces the rate
of defects and problematic code generation across different
models.
Furthermore, we conducted a detailed analysis of three
CWE scenarios that are typically concealed yet pose se-
vere risks. As illustrated in Figure 2, under the RefleXGen
method, the code security generated by GPT-3.5 and Cod-
eQwen demonstrated significant improvements in scenarios
prone to triggering high-risk vulnerabilities. In contrast,
Gemini exhibited fluctuations in security enhancements,
while the improvements in GPT-4 were relatively modest,
likely due to its already high baseline of code safety.It is worth noting that, except for GPT-4, the initial com-
pilation success rate for other models declined. This decline
is primarily attributed to the introduction of more restrictive
conditions and code interferences, which added complexity
to the tasks. These changes led to more complex code out-
puts, thereby affecting the compilation success rates. How-
ever, GPT-4, with its robust overall capabilities, was less af-
fected and even showed an improvement in compilation suc-
cess. In contrast, CodeQwen, which has a smaller parameter
size, experienced a greater decline. This phenomenon under-
scores the dependency of RefleXGen’s enhancements on the
models’ capabilities in dialogue and handling complex sce-
narios.
CONCLUSION
In this work, we have introduced RefleXGen, an innovative
method that significantly enhances the security of code gen-
erated by large language models without the need for model
fine-tuning or the creation of specialized security datasets.
Universally applicable to all code generation models and op-
erating independently of external enhancements, RefleXGen
leverages the models’ inherent reflective processes to accu-
mulate security knowledge. By building a dynamic knowl-
edge base, it optimizes prompts for subsequent code gener-
ation cycles. Experimental results demonstrate that RefleX-
Gen substantially improves code generation security across
various models, including GPT-3.5, GPT-4, CodeQwen, and
Gemini, with particularly notable enhancements in models
possessing stronger overall capabilities. This advancement
underscores the potential of self-reflective mechanisms in AI
models to autonomously improve code security, paving the
way for future research in secure code generation without ex-
tensive resource investment.
5. REFERENCES
[1] Xin-Ye Li, Jiang-Tian Xue, Zheng Xie, and Ming Li,
“Think outside the code: Brainstorming boosts large

language models in code generation,”arXiv preprint
arXiv:2305.10679, 2023.
[2] Wang Ling, Edward Grefenstette, Karl Moritz Her-
mann, Tom ´aˇs Ko ˇcisk`y, Andrew Senior, Fumin Wang,
and Phil Blunsom, “Latent predictor networks for code
generation,”arXiv preprint arXiv:1603.06744, 2016.
[3] Veselin Raychev, Pavol Bielik, and Martin Vechev,
“Probabilistic model for code with decision trees,”ACM
SIGPLAN Notices, vol. 51, no. 10, pp. 731–747, 2016.
[4] OpenAI, “Openai codex,” 2021, Accessed: 2024-08-18.
[5] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Fed-
erico Cassano, Joel Lamy-Poirier, Nouamane Tazi,
Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei,
et al., “Starcoder 2 and the stack v2: The next gener-
ation,”arXiv preprint arXiv:2402.19173, 2024.
[6] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebastian
Gehrmann, et al., “Palm: Scaling language modeling
with pathways,”Journal of Machine Learning Research,
vol. 24, no. 240, pp. 1–113, 2023.
[7] Dongling Xiao, Han Zhang, Yukun Li, Yu Sun, Hao
Tian, Hua Wu, and Haifeng Wang, “Ernie-gen: An
enhanced multi-flow pre-training and fine-tuning frame-
work for natural language generation,”arXiv preprint
arXiv:2001.11314, 2020.
[8] Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH
Hoi, “Codet5: Identifier-aware unified pre-trained
encoder-decoder models for code understanding and
generation,”arXiv preprint arXiv:2109.00859, 2021.
[9] Yujia Li, David Choi, Junyoung Chung, Nate Kush-
man, Julian Schrittwieser, R ´emi Leblond, Tom Eccles,
James Keeling, Felix Gimeno, Agustin Dal Lago, et al.,
“Competition-level code generation with alphacode,”
Science, vol. 378, no. 6624, pp. 1092–1097, 2022.
[10] Priyan Vaithilingam, Tianyi Zhang, and Elena L Glass-
man, “Expectation vs. experience: Evaluating the us-
ability of code generation tools powered by large lan-
guage models,” inChi conference on human factors in
computing systems extended abstracts, 2022, pp. 1–7.
[11] Jingxuan He and Martin Vechev, “Large language mod-
els for code: Security hardening and adversarial testing,”
inProceedings of the 2023 ACM SIGSAC Conference
on Computer and Communications Security, 2023, pp.
1865–1879.
[12] Hammond Pearce, Baleegh Ahmad, Benjamin Tan,
Brendan Dolan-Gavitt, and Ramesh Karri, “Asleep atthe keyboard? assessing the security of github copilot’s
code contributions,” in2022 IEEE Symposium on Secu-
rity and Privacy (SP). IEEE, 2022, pp. 754–768.
[13] Cordell Green, “Application of theorem proving to
problem solving,” inReadings in Artificial Intelligence,
pp. 202–222. Elsevier, 1981.
[14] Zeyu Sun, Qihao Zhu, Yingfei Xiong, Yican Sun, Lili
Mou, and Lu Zhang, “Treegen: A tree-based trans-
former architecture for code generation,” inProceedings
of the AAAI conference on artificial intelligence, 2020,
vol. 34, pp. 8984–8991.
[15] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan
Wang, Yingbo Zhou, Silvio Savarese, and Caiming
Xiong, “Codegen: An open large language model for
code with multi-turn program synthesis,”arXiv preprint
arXiv:2203.13474, 2022.
[16] Jeevana Priya Inala, Chenglong Wang, Mei Yang,
Andres Codas, Mark Encarnaci ´on, Shuvendu Lahiri,
Madanlal Musuvathi, and Jianfeng Gao, “Fault-aware
neural code rankers,”Advances in Neural Information
Processing Systems, vol. 35, pp. 13419–13432, 2022.
[17] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu
Liu, Tal Remez, J ´er´emy Rapin, et al., “Code llama:
Open foundation models for code,”arXiv preprint
arXiv:2308.12950, 2023.
[18] Mohammed Latif Siddiq and Joanna CS Santos, “Se-
curityeval dataset: mining vulnerability examples to
evaluate machine learning-based code generation tech-
niques,” inProceedings of the 1st International Work-
shop on Mining Software Repositories Applications for
Privacy and Security, 2022, pp. 29–33.
[19] Jiexin Wang, Liuwen Cao, Xitong Luo, Zhiping Zhou,
Jiayuan Xie, Adam Jatowt, and Yi Cai, “Enhancing
large language models for secure code generation: A
dataset-driven study on vulnerability mitigation,”arXiv
preprint arXiv:2310.16263, 2023.
[20] Jingxuan He, Mark Vero, Gabriela Krasnopolska, and
Martin Vechev, “Instruction tuning for secure code gen-
eration,”arXiv preprint arXiv:2402.09497, 2024.
[21] Tam ´as Szab ´o, “Incrementalizing production codeql
analyses,” inProceedings of the 31st ACM Joint Euro-
pean Software Engineering Conference and Symposium
on the Foundations of Software Engineering, 2023, pp.
1716–1726.