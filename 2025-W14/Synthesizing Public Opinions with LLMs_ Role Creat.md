# Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy

**Authors**: Rabimba Karanjai, Boris Shor, Amanda Austin, Ryan Kennedy, Yang Lu, Lei Xu, Weidong Shi

**Published**: 2025-03-31 21:21:52

**PDF URL**: [http://arxiv.org/pdf/2504.00241v1](http://arxiv.org/pdf/2504.00241v1)

## Abstract
This paper investigates the use of Large Language Models (LLMs) to synthesize
public opinion data, addressing challenges in traditional survey methods like
declining response rates and non-response bias. We introduce a novel technique:
role creation based on knowledge injection, a form of in-context learning that
leverages RAG and specified personality profiles from the HEXACO model and
demographic information, and uses that for dynamically generated prompts. This
method allows LLMs to simulate diverse opinions more accurately than existing
prompt engineering approaches. We compare our results with pre-trained models
with standard few-shot prompts. Experiments using questions from the
Cooperative Election Study (CES) demonstrate that our role-creation approach
significantly improves the alignment of LLM-generated opinions with real-world
human survey responses, increasing answer adherence. In addition, we discuss
challenges, limitations and future research directions.

## Full Text


<!-- PDF content starts -->

Synthesizing Public Opinions with LLMs: Role
Creation, Impacts, and the Future to eDemorcacy
Rabimba Karanjai1†,Boris Shor1,Amanda Austin1,Ryan Kennedy3,Yang Lu1,Lei Xu2,Weidong Shi1
1University Of Houston,2Kent State University,3Ohio State University
†rabimba@cs.uh.edu
This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data,
addressing challenges in traditional survey methods like declining response rates and non-response
bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-
context learning that leverages RAG and specified personality profiles from the HEXACO model and
demographic information, and uses that for dynamically generated prompts. This method allows LLMs
to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare
our results with pre-trained models with standard few-shot prompts. Experiments using questions
from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly
improves the alignment of LLM-generated opinions with real-world human survey responses, increasing
answer adherence. In addition, we discuss challenges, limitations and future research directions.
Date:April 2, 2025
1 Introduction
Public opinion research, a cornerstone of democratic societies, has faced significant challenges in recent decades
Berinsky (2013). One of the most pressing issues is the growing difficulty in obtaining survey data that
accurately represents a population Plewes and Tourangeau (2013). A key problem lies in the steep decline in
response rates for traditional survey methods such as telephone surveys. This trend complicates efforts to
collect data with adequate sample sizes, particularly when analyzing specific subgroups Kennedy and Hartig
(2019). The drop in response rates exacerbates non-response bias, which becomes significantly pronounced
when response patterns vary systematically across demographic or political subpopulations, such as those
defined by age, race, or political affiliation Simmons and Hare (2023). This issue is further compounded
when the bias aligns with unobservable yet critical characteristics, like political ideologies or voting behaviors
Groves and Peytcheva (2008).
Even surveys with large overall sample sizes can struggle with insufficient data points Wang et al. (2015). This
challenge, often called the "curse of dimensionality," undermines the validity of inferences about subpopulations
and poses significant hurdles for studying political behavior and public opinions Bellman (1957); Ornstein
(2020).
In response to these obstacles, social scientists have embraced innovative approaches such as multilevel
regression with poststratification (MRP) Gelman and Little (1997); Park et al. (2006). MRP and its variants
have emerged as essential tools for estimating opinions within subgroups, especially in hierarchical or multilevel
data structures such as regional and demographic categories. These methods improve accuracy by leveraging
information from larger groups to generate more reliable estimates for smaller, under represented subgroups.
However, their effectiveness often depends on assumptions that may not always hold true Little (1993). Despite
these limitations, these persistent issues have prompted researchers to explore alternative data sources and
cutting-edge technologies with the potential to improve the reliability and precision of public opinion research.
Among the technologies gaining significant traction in public opinion research is the generation of synthetic
data by advanced language models, specifically Large Language Models (LLMs). Synthetic data refers to
information created through computational processes that mimic real-world data patterns without direct
observation. LLMs, which are trained on vast and diverse text datasets, have been proposed as a novel method
1arXiv:2504.00241v1  [cs.CL]  31 Mar 2025

for producing synthetic public opinion data Argyle et al. (2023). These sophisticated models identify complex
statistical relationships between demographic variables and the language used in political contexts. Beyond
simple word associations, LLMs capture higher-order interactions during their training phase, optimizing
the likelihood of sequences of words or phrases based on their contextual usage. This capability allows
them to extrapolate from observed patterns, generating survey responses that aim to represent the political
perspectives of various demographic groups. The generation process typically involves guiding the model with
specific parameters that outline the characteristics of a hypothetical respondent, such as their demographic
background or political stance. Conditioning the LLM in this way can produce answers to survey questions
that strive to reflect these specified traits.
The ability of LLMs to produce synthetic data on a large scale has sparked interest among social scientists
as a possible way to overcome the difficulties in gathering representative survey data Argyle et al. (2023);
Qu and Wang (2024). However, there are ongoing concerns about the accuracy of this synthetic data, with
skepticism regarding whether LLMs truly capture real-world public opinions. As a result, researchers are
creating methods to evaluate and measure the quality associated with this process.
This paper systematically describes the different approaches that can customize LLMs for tasks such as
synthesizing public opinions. We evaluated the pros and cons of these approaches. In addition, we present a
novel technique called role creation based on knowledge injection to LLMs for simulating population traits, a
type of in-context learning. Our experimental results show that this new approach can significantly improve
the accuracy of LLM-based public opinion polling. In addition, we discuss the impacts of LLM-simulated
public opinion polling, challenges, and future research directions.
Specifically, we address the following research questions:
•RQ1:Can survey opinion data be simulated using LLMs?
•RQ2:Can our framework inject specific role knowledge into LLMs as part of in-context learning to help
them better mimic human responses?
•RQ3:IfRQ2is satisfied, can this approach be generalized and made model agnostic?
2 Background and Motivation
2.1 The Promises of Synthetic Public Opinions with LLMs
Social scientists recognize the potential of LLMs for generating synthetic samples, primarily because these
models can produce data without the logistical challenges associated with traditional methods. Unlike human
respondents, LLMs can manage longer surveys while maintaining data quality, helping to reduce respondent
fatigue and loss of focus Bail (2024); Messeri and Crockett (2024).
LLMs demonstrate human-like traits when simulating human behavior and psychological processes. For
example, some models have shown they can mimic human moral decisions and behavioral patterns Dillion
et al. (2023). This alignment with human ethical decision-making processes is evident in their capacity to
predict and reflect actual moral decisions Dillion et al. (2023). When the morally correct action is obvious,
LLMs usually opt for sensible answers; however, in uncertain circumstances, they—like humans—show doubt
Scherrer et al. (2024). Moreover, LLMs can predict various social behaviors such as trust, cooperation, and
competitive tendencies Leng and Yuan (2023); Xie et al. (2024); Zhao et al. (2024). Additionally, LLMs can
capture public perceptions of personality traits among notable individuals, showcasing their flexibility and
precision in mimicking human behaviors Cao and Kosinski (2024).
LLMs also exhibit potential for advancing public opinion research by offering novel ways to simulate political
behaviors and preferences. They can evaluate the positions of politicians on key policy issues Wu et al. (2023)
and gauge public views on contentious topics like climate change Lee et al. (2024). These models can also
serve as practical tools for estimating voter choices Qi et al. (2024). Furthermore, generative agents have been
shown to accurately replicate participants’ responses on the General Social Survey, matching how participants
would answer their own questions two weeks later, including on topics such as political party affiliation and
ideology Park et al. (2024). The ability of LLMs to generate synthetic samples suggests their potential value in
2

LLM customized for polling
Pre-training Post -training 
Train from scratch 
with polling dataCustomize 
existing LLM
with polling dataPrompt engineering Simple RAG Enhanced RAG
targeting 
role play and pollingFine -tuningFigure 1Taxonomy of LLM task customization approaches.
estimating public opinion, particularly in contexts where traditional data collection methods are constrained,
such as in non-democratic regimes (though see Qi et al. (2024)). They might even be capable of predicting
public reactions to future political events Wang et al. (2024).
The ability of LLMs to enhance public opinion research extends beyond generating synthetic data. These
models can also play a supportive role in various research stages. For example, LLMs can pre-test new
survey questions and assist in developing item scales Bail (2024). They can serve as substitutes for human
respondents who drop out of longitudinal studies, thus helping to maintain sample integrity. Furthermore,
LLMs can annotate open-ended data collected from human or synthetic samples with minimal supervision,
streamlining the data analysis process Ziems et al. (2024). While social scientists are optimistic about the
potential of LLMs to revolutionize public opinion research, significant challenges remain in ensuring that the
synthetic data generated accurately reflects human public opinion.
2.2 Approaches to Customize LLMs for Public Opinion Research
LLM customization methods can be broadly classified into pre-training and post-training approaches.
2.2.1 Pre-training
General-purpose LLMs are typically trained with open web data and can perform a wide range of tasks,
such as responding to a survey. In the case of pre-training, with domain-specific datasets such as public
opinion polling, an LLM can be trained from scratch for specific polling-related tasks such as role-playing
and answering polling questions. LLMs trained in such a way can outperform general-purpose LLMs. The
downside of this approach is the high cost and the time required to train a new LLM from the ground up.
Recent advances in training, such as DeepSeek DeepSeek-AI et al. (2025), suggest a potential path for
reducing training costs. A more feasible approach for pre-training is to enhance existing general LLMs by
further pre-training them with domain-specific data. Continuous pre-training involves taking pre-trained
general-purpose LLMs and further training them on a new task or refining their ability to understand and
perform within specific knowledge domains, such as public opinion polling and political attitude surveys.
Table 1Comparison of different domain customization approaches.
pre-training prompt engineering RAG Enhancements with role profiles Fine tuning
Special knowledge ! ! ! !
Improved quality ! limited ! ! ,! !
Model change !,! no no no !
Cost extremely high very low low low high
Role customization ! limited ! ! ,! !
Expertise required high low medium medium medium
2.2.2 Post-training
In post-training approaches, a pre-trained LLM can be further refined with fine-tuning. Fine-tuning can tailor
pre-trained models to the specific nuances of a task. Such specialization can significantly enhance the LLM’s
effectiveness in that particular task compared to a general-purpose pre-trained model. Like pre-training,
3

Opinion polling 
adaptation
Post training LLM optimizationprompt engineeringRAG
fine tuningIn-context learning 
with role profiles
Integrated  
approachFigure 2 Adapting LLMs to synthesizing public opinion tasks.
fine-tuning incurs higher costs, requires AI expertise, and takes time. This method contrasts with other
post-training approaches where the model weights are not changed.
The most straightforward post-training approach is prompt engineering, which includes zero-shot and few-shot
learning. In the case of zero-shot learning, a user prompts an LLM without any examples, attempting to
take advantage of the reasoning patterns it has gleaned in a general-purpose LLM. In zero-shot learning, a
user provides an LLM with a prompt without any examples, aiming to leverage the reasoning patterns the
model has acquired during its general-purpose training. Prompt engineering can enhance the performance of
a pre-trained model; however, this improvement is often limited.
Another line of post-training optimization is knowledge injection Lauscher et al. (2020); Chen et al. (2022). By
incorporating knowledge bases such as political affiliation, polling results, political ideology, and demographics,
it is possible to teach a pre-trained model about the specific domain and tasks. Fine-tuning is one way to add
knowledge to a pre-trained model. With fine-tuning, the model builder continues the model training process
and adapts it using task-specific data. By exposing the model to a specific knowledge domain, its weights can
be adapted for the targeted applications. As a result, its performance in particular tasks, such as polling
simulation, can be more relevant to the specialized domains.
Another approach to improve a model’s knowledge base is using in-context learning, such as retrieval augmented
generation (RAG) Fan et al. (2024); Lewis et al. (2020). RAG utilizes information retrieval techniques to
allow general-purpose LLMs to access relevant data from a knowledge source, often stored in vector databases,
and integrate it into the generated text Jing et al. (2024). Post-training knowledge injection can address
limitations for many knowledge-intensive tasks Ovadia et al. (2024). However, general post-training knowledge
injections, such as existing RAG approaches, are not explicitly designed for role-play and opinion polling
tasks.
A promising direction is RAG enhanced with role profiles. RAG—augmented with role profiles—can simulate
the political opinions of specific population groups more accurately than simple RAG. We introduce this
approach to the literature and present experimental results to demonstrate its advantages for polling tasks.
3 Simulating Voter Preferences
Several studies investigate the feasibility and effectiveness of using LLMs to simulate survey responses. A key
study by Argyle et al. (2023) finds that LLMs can provide reasonably accurate simulations of group-level
responses in behavioral science and economics experiments, as well as for political surveys. Horton (2023)
and Aher et al. (2023) further corroborate these findings, demonstrating the potential of LLMs to capture
4

Figure 3 Role generation from attributes.
human-like response patterns in various survey settings.
One crucial aspect of simulating human responses is to capture the diversity of opinions across different
cultural backgrounds. If trained effectively, LLMs have the potential to generate culturally nuanced responses,
facilitate cross-cultural research, and provide insight into how cultural factors influence survey responses.
Furthermore, LLMs can simulate such responses with varying levels of confidence. This capability allows for a
more realistic simulation of human-like response patterns, as survey respondents often express their opinions
with differing degrees of certainty.
However, existing research primarily focuses on using LLMs “out of the box” with prompting strategies.
While prompt engineering can be practical, it may not fully capture the nuances of individual differences
and personal traits that influence survey responses. This paper argues that by dynamically creating specific
personal preferences in LLMs, we can achieve a more realistic and nuanced simulation of survey feedback.
3.1 LLMs as Simulated Opinion Sources
At their core, LLMs function as advanced statistical language models trained on vast datasets of textual
information. These models predict the probability distribution of the next token (word or character) within a
sequence based on preceding tokens. Mathematically, this process can be expressed as:
p(xn|x1, . . . , x n−1),
where xirepresents a token from a predefined vocabulary. This capability extends beyond mere memorization,
as it leverages the model’s ability to capture intricate statistical patterns within training data, enabling novel
and contextually appropriate text generation.
A key feature that renders LLMs particularly adept at simulating diverse opinions is their reliance on
conditioning. Before generating text, the model receives an initial input or context that significantly influences
subsequent output. This conditioning context, represented by tokens x1, . . . , x n−1, plays a crucial role in
guiding the model’s response. By strategically modifying this context, we can exert substantial control over
the direction of text generation. For instance, providing a context that outlines specific demographic traits or
political orientations can alter the probability distribution of subsequent tokens, prompting responses aligned
with specified characteristics.
Table2 provides examples of how the profile creation from the attribute pool works. Figure 3 shows how the
LLM takes these attributes and uses them—alongside political leanings—to create the rolesto be used for
opinion generation.
Once we have the roles created, as shown in Figure 3, we save those in a vector database to be used for
a dynamic RAG for querying our LLM before sending the query to each user. Once a question is asked,
the RAG outputs a section of the relevant profile that matches with the few-shot prompts passed through
the query. Based on this retrieved RAG response, we query the LLM for an opinion/answer to the queried
question, asking the LLM to role-play according to the retrieved profile. This retrieved profile is dynamic and
can be potentially infinite based on the variation of the few shot prompts. We illustrate the full architecture
in Figure 4.
5

Table 2Different Role Creation based on Attributes
HEXACO Political Concise Prompt (Respond as...)
Dimension Leaning
HConservative ...high honesty/humility, conservative values. Integrity,
tradition, honest governance.
Liberal ...high honesty/humility, liberal justice. Equality,
fairness, systemic equity.
Populist ...cynical of power. Low H elites, skeptical, ’common
person’ vs ’establishment’.
EConservative ...low emotionality, national security. Calm, rational,
strong defense, measured concern.
Liberal ...high emotionality, social empathy. Concerned,
empathetic, vulnerable, compassionate solutions.
Populist ...emotional appeals, common frustrations. High E
grievances, overlooked, wronged by elites.
XConservative ...introverted, measured action. Deliberate, reserved,
cautious, behind-scenes influence.
Liberal ...extraverted, public engagement. Lively, engaging,
activist, public discourse, collective action.
Populist ...extraverted, rally base. Energetic, direct, ’common
person’, bypass ’establishment’.
AConservative ...low agreeableness, firm stance. Direct, less consensus,
strong convictions, principled.
Liberal ...high agreeableness, consensus. Cooperative, polite,
common ground, compromise, harmony.
Populist ...low agreeableness vs. elites. Combative, critical,
’people’s will’, conflict if needed.
CConservative ...high conscientiousness, fiscal responsibility.
Organized, rules, disciplined, efficient, responsible gov.
Liberal ...low conscientiousness flexible, urgent needs. Flexible,
responsive, immediate problems, adaptable policy.
Populist ...low conscientiousness anti-bureaucracy. Disregard ’red
tape’, direct action, swift results.
OConservative ...low openness, tradition. Conventional, historical
precedent, cautious change, proven methods.
Liberal ...high openness, progress. Creative, forward-thinking,
innovative, social progress, rethink systems.
Populist ...high openness disruptive style. Reject ’elitist’ norms,
unconventional style, disrupt status quo.
6

Figure 4 Opinion generation using roles.
7

4 Experiments and Evaluation
We evaluate our framework and survey results by comparing them with existing human evaluation results.
For our evaluation, we ask the same questions as our dataset and compare the responses’ similarity with
human-given responses. This dataset has not been used for training, nor were any few-shot examples used for
our framework or the pre-trained LLMs with which we compare.
4.1 Data
Our study centers on 30 issue-related questions from the 2021 Cooperative Election Study (CES), previously
known as the Cooperative Congressional Election Study (CCES). We chose to focus on issue attitudes because
they are the most frequent and influential targets of political polls Morris (2022). Extensive observational and
experimental evidence demonstrates that these issue polls significantly influence political decision-making
Burstein (2003); Butler et al. (2011); Wlezien and Soroka (2016); Morris (2022). Consequently, these polling
responses are those most likely to be sought from synthetic respondents. Among available data sources, we
selected the CCES due to its substantial respondent base (exceeding 17,000) and consistent coverage of a
broad range of issues. We focused on 2021 as the most recent year for which we could reasonably assume that
all the LLMs under consideration had been trained on relevant data.1
4.2 Experimental Setup
We ran all our experiments on a machine running on Ubuntu 24.04 with RTX 4090 as our inference provider.
llama.cpp was the backend with Vulkan API support to run and generate the experiments. We intentionally
generated every response in a fresh state without memory of the previous interactions. All the experiments
were done, keeping the model temperature at 0.
4.3 Result Adherence
We begin by assessing the accuracy of the LLM’s responses in comparison to human responses, focusing
specifically on how well LLMs replicate human responses. A key distinction between our study and previous
ones is that we not only examine aggregate results but also analyze individual opinion levels to determine how
closely they match human responses. We analyze both the respondent-level accuracy as well as topic-level
accuracy. We compare the responses of various LLMs and humans with those generated by our framework to
see if our methodology produces results that closely align with human responses.
4.3.1 Accuracy for Individual Responses
For individual responses, we begin by examining the issues where the LLM’s responses aligned with human
responses, measured by percentage.
Table 3 shows our adherence to different topics with the questions of the CCES questionnaire. We used a
few-shot prompting for the different pre-trained LLMs before getting the results, But we used the template
shown in Figure 4 to generate opinions for the role generation.
5 Discussion
Based on our empirical experiments shown in Section 4, we can observe certain specific characteristics of
the survey response for the models. While we targeted particular questions, we have achieved to show the
adherence trends in Table 3.
If we circle back to our original queries:
1It is crucial to acknowledge that the training data for LLMs typically lags behind the LLM’s publication date by at least a
couple of years. At this time, real-time updating of LLM weights is impractical due to resource constraints.
8

Table 3Simulated Result Adherence
Model Parameters Adherence On Questions(%)
phi3 14b 72.7
deepseek-r1 32b 79.8
gemma2 9b 70.2
gemma2 27b 71.6
llama3.2 3b 67.6
llama3.3 70b 73.1
Role Creation + gemma2 27b 77.9
Role Creation + gemma2 9b 73.3
Role Creation + llama 3.3 70b 84.1
RQ1: Can survey opinion data be simulated using LLMs?
Yes.The results in Table 3, clearly indicate that opinions can be simulated by LLMs. However, the
method by which we can simulate these opinions does vary the achieved adherence performance. For
all the responses, we had to provide the voter demography and profile information either by a few-shot
prompt or through our framework’s role creation.
While we utilize open-weight LLMs with few shot prompting, our framework uses dynamic role creation based
on personality attributes as depicted in Table2 and voter preferences. This method allows us to dynamically
generate multiple roles with granular preferences attached to them, as shown in Figure 3. For our question,
we then use a RAG system to select these roles and—as the LLM—to give opinions on the survey questions
following the flow shown in Figure 4. The text highlighted in blue are the dynamically generated roles from
Figure 3 retrieved from RAG using the question and then sent to the LLM to get a simulated response.
These results bring us to our second research question:
RQ2: Can response adherence be increased for LLMs to mimic humans?
Yes.Our empirical results in Section 4 show that using our framework of dynamically generating roles and
then pairing them up with existing pre-trained LLMs increases their adherence to human responses.
We test our framework with three different LLMs from two different LLM families, and they all show
improvement from their base models, which were prompted with few-shot prompts. One noticeable insight
we glean is that the bigger models show bigger gains with the same technique than the smaller models.
This finding suggests that the ability to understand nuanced instructions likely played a role in how closely
the generated opinions aligned with human responses.
This result, however, raises an interesting question. Can the LLMs predict human responses, or do they
already have these associations as part of their pretraining data? Judging from how all the LLMs crossed 50%
adherence with human responses with just a few-shot prompting, we hypothesize the preference associations
are already present in the LLM’s pretraining data. Making them roleplay with targeted rolesseems to further
encourage more opinionated responses.
That finding brings us to our last research question:
9

RQ3: Can we generalize this framework and make it model agnostic?
Our empirical and experimental results show improvements across three different models. However, a
closer examination reveals inconsistencies in the actual performance gains, highlighting a weakness in
our framework. Since we primarily rely on the LLMs’ ability to understand roles and elicit opinionated
responses, their performance suffers when they fail to grasp complex instructions or nuances.
These observations lead us to consider whether we can embed models’ preferences through fine-tuning and
trigger-based generation. We leave this as an open research question for future work, along with another
question regarding the role of non-English languages in generating similar opinions.
6 Impacts
Using LLMs to synthesize public opinions has the potential to transform the democratic process on a global
scale. For decades, researchers have gathered public opinion data through labor-intensive, costly, and time-
consuming methods such as in-person interviews, phone calls, and survey mailings. Additionally, gathering
accurate public opinions can be challenging in regions with low economic development, and in some countries,
local governments actively control or restrict public opinion surveys. The LLM-based approach offers a
promising solution to these challenges by making the process cheaper, faster, and more accurate. For example,
it can refine survey questions using simulated polls before finalizing them, reducing bias and improving
question quality. Furthermore, LLM role-playing in public opinion simulation opens a new frontier where
campaigns can model citizens’ reactions to different candidate messages. This pre-testing allows campaigns to
experiment with various strategies and messages before implementation. Crucially, this method also enables
the simulation of responses from populations whose opinions might otherwise be marginalized or silenced.
7 Challenges and research directions
7.1 Technology Limitations
The existing literature identifies potential limitations of using LLMs to generate synthetic samples for public
opinion research. These include the risk of training data memorization, where models might reproduce
specific details instead of developing new inferences, and sensitivity to prompt formulations, which can lead
to inconsistent or biased outputs. Variations across different LLMs can also compromise the reliability of
generated samples, and the models’ tendency to generalize may introduce distortions.
LLMs are sensitive to prompts’ precise wording and structure, which can substantially influence their outputs.
When applying linguistic rules and world knowledge, LLMs can be influenced by specific examples and
phrasing, leading to response variations based on subtle changes in prompt formulation Chang and Bergen
(2024). This sensitivity becomes less significant when employing our technique with role creation. A thorough
investigation into how our role creation technique enhances RAG and in-context learning to mitigate the
sensitivity of LLMs to prompt variations would provide valuable insights and is a promising direction for
future research.
Another concern in using LLMs to generate synthetic samples for public opinion research is the inconsistency
across different models. Research demonstrates that different LLMs can exhibit varying traits and performance
levels, potentially leading to output discrepancies. To mitigate this issue, researchers either need a deep
understanding of the relative strengths and weaknesses of different models, or they must be able to identify
high-quality models to include in an ensemble model, which could mitigate inconsistencies.
7.2 Localization and Less Represented Languages
A significant challenge with large language models, especially in political polling tasks and when working
with languages other than English, is the notable gap in available training data. This discrepancy results in
poorer performance for LLMs in less commonly spoken languages due to limited data, imbalances, biases, and
10

cultural nuances. The practical use of LLMs for polling in non-English languages and non-Western cultures
presents significant challenges, making it an important area for future research.
7.3 Data Quality
Knowledge injection and creation of role profiles depend on data quality. Public opinions and policy preferences
shift over time. The quality of the data used in different phases of knowledge injection and fine-tuning, as
well as the timeliness of the data, can affect the accuracy of polling tasks.
7.4 Continuous Updates and Learning
For public opinion polling, the RAG database and role profiles must be regularly updated to capture shifts
in public opinion, demographic changes, and reactions to major political events that can influence views on
policies and political decisions. This requires ongoing updates to the knowledge base, which increases both
labor and financial costs. Finding ways to reduce these costs without compromising the accuracy of the LLM
in polling tasks presents a significant research challenge.
7.5 Compliance
With the wider adoption of LLMs for polling, compliance could become a challenge. In some countries, these
capabilities may be exploited or manipulated to spread false information and deceive the public. Ensuring
public trust in LLM-based polling could be a significant issue in the future.
8 Conclusion
This research demonstrates the potential of LLMs to create synthetic public opinion data and contributes
a novel method that goes beyond a few short prompts to generate more precise responses. Using a RAG
method with user roles, the proposed role-creation framework significantly enhances the accuracy of simulated
opinions, mitigating issues identified with standard prompting or basic RAG implementation. The results
show improved answer adherence between model opinion and existing human dataset. The potential impact of
this work on improving the cost and time for opinion collection, along with democratic processes—particularly
in under-resourced or restrictive environments—is substantial, opening new avenues for eDemocracy.
Acknowledgment
We would like to express our sincere gratitude to the Google Developer Expert program and the Google ML
team for their invaluable support in this research, providing GCP credits that enabled the successful execution
of this work.
References
Gati V Aher, Rosa I Arriaga, and Adam Tauman Kalai. Using large language models to simulate multiple humans and
replicate human subject studies. In International Conference on Machine Learning , pages 337–371. PMLR, 2023.
Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. Out of
one, many: Using language models to simulate human samples. Political Analysis , 31(3):337–351, 2023.
Christopher A. Bail. Can generative ai improve social science? Proceedings of the National Academy of Sciences , 121
(21):e2314021121, 2024.
R. Bellman. Dynamic Programming . Princeton University Press, Princeton, NJ, 1957.
Adam J. Berinsky. Silent voices: Public opinion and political participation in America . Princeton University Press,
2013.
Paul Burstein. The impact of public opinion on public policy: A review and an agenda. Political research quarterly , 56
(1):29–40, 2003.
11

Daniel M Butler, David W Nickerson, et al. Can learning constituency opinion affect how legislators vote? results
from a field experiment. Quarterly Journal of Political Science , 6(1):55–83, 2011.
Xubo Cao and Michal Kosinski. Large language models know how the personality of public figures is perceived by the
general public. Scientific Reports , 14(1):6735, 2024.
Tyler A. Chang and Benjamin K. Bergen. Language model behavior: A comprehensive survey. Computational
Linguistics , 50(1):293–350, 2024.
Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen.
Knowprompt: Knowledge-aware prompt-tuning with synergistic optimization for relation extraction. In Proceedings
of the ACM Web Conference 2022 , WWW ’22, page 2778–2788. ACM, April 2022. doi: 10.1145/3485447.3511998.
URL http://dx.doi.org/10.1145/3485447.3511998 .
DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun
Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng
Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong
Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao
Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu,
Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang,
Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu,
Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu,
Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye,
Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao
Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun
Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang,
Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin
Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan
Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu,
Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan
Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang
Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong,
Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren,
Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou,
Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia
Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2025.
URL https://arxiv.org/abs/2412.19437 .
Danica Dillion, Niket Tandon, Yuling Gu, and Kurt Gray. Can ai language models replace human participants? Trends
in Cognitive Sciences , 27(7):597–600, 2023.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey
on rag meeting llms: Towards retrieval-augmented large language models, 2024. URL https://arxiv.org/abs/
2405.06211 .
A. Gelman and T.C. Little. Poststratification into many categories using hierarchical logistic regression. Survey
Methodology , 23:127, 1997.
Robert M. Groves and Emilia Peytcheva. The impact of nonresponse rates on nonresponse bias: a meta-analysis.
Public opinion quarterly , 72(2):167–189, 2008.
John J. Horton. Large language models as simulated economic agents: What can we learn from homo silicus?, 2023.
Zhi Jing, Yongye Su, and Yikun Han. When large language models meet vector databases: A survey, 2024. URL
https://arxiv.org/abs/2402.01763 .
Courtney Kennedy and Hannah Hartig. Response rates in telephone surveys have resumed their decline, 2019.
Anne Lauscher, Olga Majewska, Leonardo F. R. Ribeiro, Iryna Gurevych, Nikolai Rozanov, and Goran Glavaš.
Common sense or world knowledge? investigating adapter-based knowledge injection into pretrained transformers,
2020. URL https://arxiv.org/abs/2005.11787 .
Sanguk Lee, Tai-Quan Peng, Matthew H. Goldberg, Seth A. Rosenthal, John E. Kotcher, Edward W. Maibach, and
12

Anthony Leiserowitz. Can large language models estimate public opinion about global warming? an empirical
assessment of algorithmic fidelity and bias. PLOS Climate , 3(8):e0000429, 2024.
Yan Leng and Yuan Yuan. Do llm agents exhibit social behavior? arXiv preprint arXiv:2312.15198 , 2023.
Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler,
Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for
knowledge-intensive nlp tasks. ArXiv, abs/2005.11401, 2020. URL https://api.semanticscholar.org/CorpusID:
218869575 .
Roderick JA Little. Post-stratification: a modeler’s perspective. Journal of the American Statistical Association , 88
(423):1001–1012, 1993.
Lisa Messeri and M. J. Crockett. Artificial intelligence and illusions of understanding in scientific research. Nature, 627
(8002):49–58, 2024.
G Elliott Morris. Strength in Numbers: How Polls Work and why We Need Them . WW Norton & Company, 2022.
Joseph T. Ornstein. Stacked regression and poststratification. Political Analysis , 28(2):293–301, 2020.
Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. Fine-tuning or retrieval? comparing knowledge
injection in llms, 2024. URL https://arxiv.org/abs/2312.05934 .
David K. Park, Andrew Gelman, and Joseph Bafumi. State-level opinions from national surveys: Poststratification
using multilevel logistic regression. In Public opinion in state politics , pages 209–228. 2006.
Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb
Willer, Percy Liang, and Michael S. Bernstein. Generative agent simulations of 1,000 people. arXiv preprint
arXiv:2411.10109 , 2024.
Thomas J. Plewes and Roger Tourangeau, editors. Nonresponse in social science surveys: A research agenda . 2013.
Weihong Qi, Hanjia Lyu, and Jiebo Luo. Representation bias in political sample simulations with large language
models.arXiv preprint arXiv:2407.11409 , 2024.
Yao Qu and Jue Wang. Performance and biases of large language models in public opinion simulation. Humanities
and Social Sciences Communications , 11(1):1–13, 2024.
Nino Scherrer, Claudia Shi, Amir Feder, and David Blei. Evaluating the moral beliefs encoded in llms. Advances in
Neural Information Processing Systems , 36, 2024.
Gabriel Simmons and Christopher Hare. Large language models as subpopulation representative models: A review.
arXiv preprint arXiv:2310.17888 , 2023.
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, and Zhiyuan et al. Chen. A survey on
large language model based autonomous agents. Frontiers of Computer Science , 18(6):186345, 2024.
Wei Wang, David Rothschild, Sharad Goel, and Andrew Gelman. Forecasting elections with non-representative polls.
International Journal of Forecasting , 31(3):980–991, 2015.
Christopher Wlezien and Stuart N Soroka. Public opinion and public policy. In Oxford research encyclopedia of politics .
2016.
Patrick Y. Wu, Jonathan Nagler, Joshua A. Tucker, and Solomon Messing. Large language models can be used to
estimate the latent positions of politicians. arXiv preprint arXiv:2303.12057 , 2023.
Chengxing Xie, Canyu Chen, Feiran Jia, Ziyu Ye, Kai Shu, Adel Bibi, Ziniu Hu, Philip Torr, Bernard Ghanem, and
Guohao Li. Can large language model agents simulate human trust behaviors? arXiv preprint arXiv:2402.04559 ,
2024.
Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, and Xing Xie. Competeai: Understanding
the competition dynamics of large language model-based agents. In Forty-first International Conference on Machine
Learning , 2024.
Caleb Ziems, William Held, Omar Shaikh, Jiaao Chen, Zhehao Zhang, and Diyi Yang. Can large language models
transform computational social science? Computational Linguistics , 50(1):237–291, 2024.
13