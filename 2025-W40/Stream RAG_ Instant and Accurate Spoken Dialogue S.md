# Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage

**Authors**: Siddhant Arora, Haidar Khan, Kai Sun, Xin Luna Dong, Sajal Choudhary, Seungwhan Moon, Xinyuan Zhang, Adithya Sagar, Surya Teja Appini, Kaushik Patnaik, Sanat Sharma, Shinji Watanabe, Anuj Kumar, Ahmed Aly, Yue Liu, Florian Metze, Zhaojiang Lin

**Published**: 2025-10-02 14:18:20

**PDF URL**: [http://arxiv.org/pdf/2510.02044v1](http://arxiv.org/pdf/2510.02044v1)

## Abstract
End-to-end speech-in speech-out dialogue systems are emerging as a powerful
alternative to traditional ASR-LLM-TTS pipelines, generating more natural,
expressive responses with significantly lower latency. However, these systems
remain prone to hallucinations due to limited factual grounding. While
text-based dialogue systems address this challenge by integrating tools such as
web search and knowledge graph APIs, we introduce the first approach to extend
tool use directly into speech-in speech-out systems. A key challenge is that
tool integration substantially increases response latency, disrupting
conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented
Generation (Streaming RAG), a novel framework that reduces user-perceived
latency by predicting tool queries in parallel with user speech, even before
the user finishes speaking. Specifically, we develop a post-training pipeline
that teaches the model when to issue tool calls during ongoing speech and how
to generate spoken summaries that fuse audio queries with retrieved text
results, thereby improving both accuracy and responsiveness. To evaluate our
approach, we construct AudioCRAG, a benchmark created by converting queries
from the publicly available CRAG dataset into speech form. Experimental results
demonstrate that our streaming RAG approach increases QA accuracy by up to 200%
relative (from 11.1% to 34.2% absolute) and further enhances user experience by
reducing tool use latency by 20%. Importantly, our streaming RAG approach is
modality-agnostic and can be applied equally to typed input, paving the way for
more agentic, real-time AI assistants.

## Full Text


<!-- PDF content starts -->

Stream RAG: Instant and Accurate Spoken
Dialogue Systems with Streaming Tool Usage
Siddhant Arora1,2,∗,Haidar Khan1,Kai Sun1,Xin Luna Dong1,Sajal Choudhary1,Seungwhan Moon1,
Xinyuan Zhang1,Adithya Sagar1,Surya Teja Appini1,Kaushik Patnaik1,Sanat Sharma1,Shinji Watanabe2,
Anuj Kumar1,Ahmed Aly1,Yue Liu1,Florian Metze1,Zhaojiang Lin1
1Meta,2Carnegie Mellon University
∗Work done at Meta
End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional
ASR–LLM–TTS pipelines, generating more natural, expressive responses with significantly lower
latency. However, these systems remain prone to hallucinations due to limited factual grounding.
While text-based dialogue systems address this challenge by integrating tools such as web search
and knowledge graph APIs, we introduce the first approach to extend tool use directly into speech-
in speech-out systems. A key challenge is that tool integration substantially increases response
latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented
Generation (Streaming RAG), a novel framework that reduces user-perceived latency by predicting
tool queries in parallel with user speech, even before the user finishes speaking. Specifically, we
develop a post-training pipeline that teaches the model when toissue tool callsduring ongoing speech
and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby
improving bothaccuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a
benchmark created by converting queries from the publicly available CRAG dataset into speech form.
Experimental results demonstrate that ourstreaming RAGapproach increases QA accuracy by up
to 200% relative (from 11.1% to 34.2% absolute) and further enhances user experience by reducing
tool use latency by 20%. Importantly, ourstreaming RAGapproach is modality-agnostic and can be
applied equally to typed input, paving the way for more agentic, real-time AI assistants.
Date:October 3, 2025
Correspondence:siddhana@andrew.cmu.edu,zhaojiang@meta.com
1 Introduction
Spoken Dialogue Systems (SDS) are foundational to many everyday technologies, powering intelligent assistants
such as Alexa and Siri, as well as interactive voice response systems in customer service. With the rapid
expansion of SDS capabilities to mobile phones and wearable devices, the need for robust, scalable, and
generalizable solutions has never been greater. Traditionally, SDS have relied on cascaded pipelines composed
of multiple modules—including voice activity detection (VAD), automatic speech recognition (ASR), natural
language understanding (NLU), natural language generation (NLG), and text-to-speech (TTS) synthesis—each
introducing potential points of failure and latency (Glass, 1999; Huang et al., 2024). Recently, end-to-end
(E2E) SDS (Xie and Wu, 2024; Nguyen et al., 2023; Meng et al., 2024; Zhang et al., 2024; Arora et al., 2025b)
have been proposed, which directly generate spoken responses from speech input within a unified architecture.
This E2E approach not only mitigates error propagation across modules but also captures non-phonemic
information more effectively, resulting in significantly lower inference time and computational overhead, and
paving the way for more natural and efficient conversational experiences.
Despite these advances, current E2E SDS are fundamentally constrained by their reliance on internalized
knowledge from static training data, which often results in responses that lack factual grounding or fail
to reflect the most up to date information. This shortcoming is particularly critical for action-oriented or
knowledge-seeking tasks, such as booking hotels or answering questions about current events. In contrast,
text-based conversational assistants have begun to overcome these limitations by integrating external tools
1arXiv:2510.02044v1  [cs.CL]  2 Oct 2025

through Retrieval-Augmented Generation (RAG)(Yang et al., 2024; Chen et al., 2023a,c; Gao et al., 2024a,b),
dynamically retrieving relevant information from sources like web search, knowledge graphs (KG), and real-
time APIs. Yet, the integration of such tool use into E2E SDS remains largely unexplored. A key challenge
is that while external tools can substantially improve factual accuracy, invoking them often introduces
additional latency, leading to awkward silences that disrupt the natural conversational flow. This raises a
research question:How can we trade-off between accuracy and responsiveness for developing SDS that feel
both intelligent and natural?
1
User SpeaksTool QueryHigh LatencyTool ResultsSystem ResponseSpoken Utterance
User Speaks
Tool QueryLow LatencyTool ResultsSystem ResponseTradition RAG: Sequential Tool CallsPreserve Conversation Flow
Streaming RAG: Parallel Tool Calls during SpeechWait for user to ﬁnish speakingSpoken Utterance
Figure 1Comparison of Traditional RAG with proposed
Streaming RAG which fires tool queries in parallel with
user speech.In this paper, we present, to the best of our knowl-
edge, the firstspeech-in, speech-out language model
that seamlessly integrates external tool invocation
with low latency. The key idea is aStreaming
RAGstrategy that generates tool queries in par-
allel with user speech, often times even before the
user has finished speaking (Fig. 1). A naive imple-
mentation of streaming queries, however, faces two
challenges: (1) queries issued from partial speech
may be suboptimal, yielding distracting tool out-
puts and inaccurate responses; and (2) unnecessary
tool calls may be triggered, wasting computational
resources. We introduce effective modeling tech-
niques to address these challenges and make the
following contributions.
Contribution 1: We introduce a formal framework
for tool integration in speech-in speech-out systems and empirically show that leveraging web search and KG
APIs significantly enhances factual question answering. Evaluating three state-of-the-art (SOTA) models,
Qwen-OMNI (Xu et al., 2025), OpusLM (Tian et al., 2025), and Kimi-Audio (Ding et al., 2025), we find that
external tool integration delivers substantial performance gains, boosting accuracy by up to 140% relative
(from 11.1% to 26.3% absolute). However, tool usage also introduces considerable latency, increasing first-token
response time by 2.3x.
Contribution 2: To address this, we proposeStreaming Retrieval-Augmented Generation(Streaming RAG),
thefirstframework that empowers the system to trigger tool queries in parallel with user speech, even before
the user finishes speaking. Within this framework, we introduce two novel approaches: (1)Fixed-Interval
Streaming RAG, which issues tool queries at regular intervals during speech input and carefully examines
quality of retrieval results on the full query to guarantee response quality, and can be incorporated into any
speech-in, speech-out model without post-training; (2)Model-Triggered Streaming RAG, which post-trains
the model to intelligently determine optimal query timing based on the evolving user utterance to save
computation resources. Our results demonstrate that our proposedModel-Triggered Streaming RAGdelivers
over 200% relative improvement in accuracy (from 11.1% to 34.2% absolute in T. 1) compared to the no-tool
baseline, while also reducing tool result generation latency by 20%. Though designed for speech-in speech-out
systems, streaming RAG can also be adapted in cascaded SDS, or even chatbots as users type.
Contribution 3: Finally, we introduce AudioCRAG, a benchmark created by recording spoken queries from
the CRAG (Yang et al., 2024) dataset, enabling robust evaluation of tool usage capabilities in speech-in
speech-out systems. To support future research, we will open source our training code and AudioCRAG-Human
benchmark, supporting future research in tool-integrated voice assistants.
2 Related studies
2.1 Benchmarks for Tool Usage
Recent advances in benchmarking text-based dialogue systems for tool usage (Chen et al., 2023b; Ouyang et al.,
2025; Cheng and Dou, 2025; Cohen et al., 2025; Xiong et al., 2024a) have primarily focused on evaluating factual
question answering and task completion within simulated environments (detailed related work discussion in
2

Appendix A, B). The CRAG benchmark (Yang et al., 2024) is a leading example, featuring 4,409 question-
answer pairs and providing mock APIs for both web and KG search. Recent benchmarks (Meta CRAG-MM
Challenge Organizers, 2025; Ma et al., 2024; Jang et al., 2025) have extended tool-augmented dialogue
evaluation to multimodal input and longer-context scenarios. While these benchmarks have significantly
advanced the evaluation of tool-augmented dialogue systems, they remain largely limited to text-based outputs
and do not fully address the unique challenges presented by speech-in speech-out systems.
2.2 E2E Spoken Dialogue Systems
Several E2E spoken dialogue systems (Xu et al., 2025; Xie and Wu, 2024; Arora et al., 2025a; Nguyen et al.,
2023; Meng et al., 2024; Zhang et al., 2024; Arora et al., 2025b) have recently been introduced, demonstrating
impressive semantic understanding and high audio quality in their responses. However, these systems have not
yet been trained or evaluated for their ability touse external tools. Another research direction (Feng et al.,
2025) explores E2E RAG for direct speech-to-text retrieval, utilizing multimodal embeddings for enabling
speech utterances to directly retrieve relevant text. While this method outperforms models without RAG,
it is primarily limited tospeech-to-textscenarios. Its ability to access KGs and other APIs, critical for
real-world applications, remains unproven. Furthermore, the retrieval scope is limited, as experiments are
conducted with retrieval restricted to just 10 paragraphs, whereas modern RAG benchmarks require searching
across thousands of web pages. Although recent studies (Maben et al., 2025) have developed web interfaces
that integrate tools into speech-inspeech-outscenarios using cascaded pipelines, comprehensive empirical
investigations of E2E speech-in speech-out systems and, crucially, systematic analyses of user-perceived latency,
remain largely unexplored. In this work, we address these gaps by developing a comprehensive framework
for tool integration in E2E speech-in speech-out systems and designing benchmarks to quantitatively assess
the tool usage capabilities of SOTA models. Furthermore, recognizing the importance of latency for natural
conversational flow, we introduce novel streaming RAG methods that not only enhance tool usage performance
but also reduce user-perceived latency.
3 Methodology
A RAG spoken conversation system takes an audio question Qas input and outputs a spoken answer A. Let
the ASR transcript of audio question be Xasrand the ASR transcript of the spoken answer be Xres. Answers
are generated by speech-in speech-out models, leveraging both the model’s internal knowledge and information
retrieved from external sources. To incorporate external information, the model needs to formulate a tool
queryQTto retrieve relevant resultsRfrom an external toolT.
3.1 Tool Integration for Speech-in Speech-Out LLMs
1Tool Results RFinal Audio Response A
User Input Speech QKnowledge GraphWeb SearchAPI resultsRelevant paragraphs from web pagesSpeech-In Speech-Out LMUser Input Speech QStage 1: Query GenerationStage 2: Response Generation<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Tool QueryQT
Figure 2Proposed formulation for integrating tool usage
in E2E speech-in, speech-out dialogue systems using a two-
stage inference approach.Figure 2 illustrates our proposed formulation for
integrating external tools into speech-in speech-
out systems. We introduce a two-stage inference
approach:Query GenerationandResponse Gener-
ation. In the Query Generation stage, the system
processes an audio question and generates queries
for each external tool to retrieve relevant infor-
mation by maximizing the posterior distribution
P(QT|Q)(Examples of generated queries are pro-
vided in T. 7 in the Appendix.). In the Response
Generationstage, theretrievedresults Rfromthese
tools are combined with the original audio ques-
tion and input into the model to generate the final
spoken response by maximizing the posterior distri-
bution P(A|Q, R ). Byconditioningthefinaloutput
generation on the input audio, this formulation not
only provides a simple and effective mechanism for
3

<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Block b-1Block bBlock b+1
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Tool  Query QTb<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Tool Results  after each block RbFinal Audio Response A<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(User Input Speech QTool call has high latency: user has finished speaking! Knowledge GraphWeb SearchAPI resultsRelevant paragraphs from web pages
Reﬂector: Compare intermediate tool query  with ﬁnal tool query QTbQTBSpeech-In Speech-Out LM
User Input Speech QStage 1: Query GenerationStage 2: Response Generation<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Rb*[   ]*B Parallel Threads[   ]*B Parallel Threads(a)Fixed-Interval Streaming RAG.
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Block b-1Block bBlock b+1
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Single thread with tool results   using RQprevBFinal Audio Response A<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(User Input Speech QSpeech-In Speech-Out LM
User Input Speech QStage 1: Query GenerationStage 2: Response Generation<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(
<latexit sha1_base64="DzYDhPmLt/912TPlY2zk+gJpBwM=">AAAB8nicbVBNS8NAEJ3Ur1q/qh69LBbBU0mKqMdSLx4r2A9IQtlsN+nSzSbsboQS+jO8eFDEq7/Gm//GTZuDtj4YeLw3w8y8IOVMadv+tiobm1vbO9Xd2t7+weFR/fikr5JMEtojCU/kMMCKciZoTzPN6TCVFMcBp4Ngelf4gycqFUvEo56l1I9xJFjICNZGcr0OiyKJvBzVRvWG3bQXQOvEKUkDSnRH9S9vnJAspkITjpVyHTvVfo6lZoTTec3LFE0xmeKIuoYKHFPl54uT5+jCKGMUJtKU0Gih/p7IcazULA5MZ4z1RK16hfif52Y6vPVzJtJMU0GWi8KMI52g4n80ZpISzWeGYCKZuRWRCZaYaJNSEYKz+vI66beaznWz9XDVaHfKOKpwBudwCQ7cQBvuoQs9IJDAM7zCm6WtF+vd+li2Vqxy5hT+wPr8AeEwkFc=</latexit>(Knowledge GraphWeb SearchAPI resultsRelevant paragraphs from web pagesTool Call?Model decides when to make tool call 
No reﬂector: Use last tool call results Tool  Query QTbMost recent tool query  (Single Thread)QprevBB*[    ]
R (b)Model-Triggered Streaming RAG.
Figure 3Proposed formulations for streaming tool query generation, referred to asStreaming RAG, to minimize
user-perceived latency in speech-in, speech-out systems. (a) Fixed-Interval Streaming RAG: tool calls are triggered at
fixed intervals and evaluated by a reflector module. (b) Model-Triggered Streaming RAG: the model autonomously
decides when to make tool calls, eliminating the need for a reflector and directly utilizing the most recent tool results
for response generation.
interacting with text-based APIs, but also preserves the key advantages of speech-in speech-out systems,
mitigating error propagation and enabling the model to capture non-phonemic information (such as prosody
and speaker intent) more effectively.
3.2 Streaming RAG
RAG-based systems, as proposed in S. 3.1, can significantly improve factual accuracy by incorporating external
tools. However, these tool calls often introduce substantial latency, which is particularly problematic in
speech-in, speech-out applications where users expect rapid, conversational responses and even brief silences
can disrupt the natural flow of dialogue. One way to address this challenge lies in the nature of audio inputs,
which arrive as a continuous stream. This streaming property enables tool calls to be initiated before the user
has finished speaking, offering a unique opportunity to mitigate latency.
To minimize user-perceived latency, we introduceStreaming RAG: the first framework to generate and issue
tool queries in parallel as audio input is received. This novel approach is built on three key design components:
1Trigger: When to initiate a new tool query; 2Threads: The number of parallel tool query threads; 3
Reflector: The module that determines whether intermediate tool results are sufficient for generating the
final output. By exploring different design choices for each component, we introduce two complementary
approaches for streaming tool query generation in the following subsections: a fixed-interval trigger method
and a model-based trigger method.
3.2.1 Fixed-Interval Streaming RAG
In this approach, thetriggeris set to fire tool calls at fixed chunk intervals during audio input. The input
speech Qis divided into a sequence of Bblocks, Q={Qb|b= 1, . . . , B}, with each block containing Nblock
frames. To approximate P(QT|Q)as described in S. 3.1, we follow a block-wise prediction strategy. In this
approach, after processing each audio block b, the model predicts a tool query ˆQT
bby conditioning on the
input speech accumulated up to block b, specificallyQ 1:b:
ˆQT
b= arg max
QT
bP(QT
b|Q1:b)(1)
4

This strategy results in Bparallel tool call threads running simultaneously (see Figure 3a), where each thread
generates a tool query prediction ˆQT
bfor its corresponding block b∈[1, B]. The tool queries ˆQT
bgenerated
after each block are then stored in cache. Given the high latency of tool calls (See T. 3), users typically
complete their utterances before tool responses are ready. Upon utterance completion, an explicitreflector
module “ reflect ()” evaluates the cached intermediate queries ˆQT
bagainst final query ˆQT
Bto determine whether
an early intermediate tool call provides sufficient information to answer the user’s question Q. The reflector
module systematically evaluates all intermediate queries in the cache and identifies theearliestsufficient tool
callb⋆where the intermediate query ˆQT
bwill give the same result as final query ˆQT
Bas shown:
b⋆= min{b∈[1, B]where reflect( ˆQT
b,ˆQT
B) = True}.(2)
All subsequent parallel tool calls after b⋆are promptly terminated, and the retrieved results Rb⋆from this
intermediate call ˆQT
b⋆are used to generate the final spoken response Aby maximizing the posterior distribution
P(A|Q, R b⋆)(instead of P(A|Q, R )in S. 3.1). We employ a reflector module that uses simple yet effective
heuristics: (a) if the top 5 web documents for an intermediate web query match those of the final web query,
and (b) if the KG results for intermediate and final KG queries are identical. These heuristics ensure that the
information retrieved from an early tool call using ˆQT
b⋆is consistent with what would have been obtained
by waiting for the final tool call using ˆQT
B, thereby providing a strong quality guarantee. Since most tool
call latency arises from chunking and reranking the chunks of web documents, these checks enable significant
latency savings without any compromise in model performance. A key advantage of this strategy is its
plug-and-play nature: it requires no additional post-training for speech-in, speech-out models and can be
directly applied at inference time across a variety of architectures. However, there are important considerations.
First, generating parallel tool calls at every fixed interval (Eq. 1) increases computational overhead, which may
pose challenges for deployment on resource-constrained devices such as wearables devices. Second, reliance
on an external reflector module (Eq. 2) to determine the sufficiency of intermediate tool calls may limit the
extent of achievable latency improvements.
3.2.2 Model-Triggered Streaming RAG
To address the limitations of Fixed-Interval Streaming RAG and further optimize both efficiency and
responsiveness, we propose a more adaptive approach:Model-Triggered Streaming RAG. Here, thetriggeris
learned: the model is trained to autonomously determine the optimal moments to initiate tool queries, issuing
a query only when it encounters new or additional information as illustrated in Figure 3b. In this formulation,
the model receives user input in fixed chunk intervals as before and intelligently determines whether a tool call
is needed after each block b. The model can either: 1Predict NO_QUERY if a new tool query is unnecessary,
or2Generate a new tool query. To make this decision, the model conditions on both the accumulated input
speechQ 1:b(see Eq. 1) as well as most recent tool query ˆQprev
b=ˆQT
max{i<b: ˆQT
i̸=NO_QUERY}as shown:
ˆQT
b= arg max
QT
bP(QT
b|Q1:b,ˆQprev
b)(3)
When a new query ˆQT
b̸=NO_QUERY , the system immediately terminates any ongoing tool calls for the
previous query ˆQprev
b, ensuring that only asingle tool call threadruns at any given time. (Examples of
generated ˆQT
bare provided in T. 8 in the Appendix.) This approach offers several key advantages. First, it
effectively eliminating redundant parallel threads and significantly reducing computational overhead. This is
especially important for deployment on resource-constrained devices. Second, this formulation removes the
need for an externalreflector(Eq. 2) module. The model confidently relies on the results Rfrom the most
recent tool call using ˆQprev
Bto generate the spoken response Aby maximizing P(A|Q, R ), reducing system
complexity.
Post-training: To train the model, we transform text-based tool usage benchmarks into spoken format (See
S. G). Word-level timestamps are computed using a pre-trained ASR model. For each partial ASR transcript
Xasr
bup to block b, we generate corresponding queries for each tool QT
busing an LLM as pseudo ground
truth (GT). To create effective training labels that teach the model when to trigger new queries, we employ a
similarity-based labeling strategy where we compare the current pseudo GT query QT
bwith the most recent
non-empty tool query label ˆQprev
b(see Eq. 3) before block b. Our labeling function assigns the training label
5

Table 1Performance comparison of accuracy, first-token latency, and latency savings (as a percentage of tool use latency,
which is 3.37 seconds as reported in Table 3) for three models—Qwen2.5-7B, OpusLM, and Kimi Audio—across three
settings: Closed Book (without tool usage), Open Book (with tool usage), and Streaming RAG (Model-Triggered
Streaming RAG). We evaluate all models on both the AudioCRAG-Synthetic (Syn.) and AudioCRAG-Human (Hum.).
Streaming RAG is not applied to Kimi-Audio as it can handle only a restricted length of tool result references (S. F).
*: OpusLM currently does not support taking tool result references in speech-out settings in zero-shot.
Setting Ref length Model Accuracy Latency
Syn. Hum. First Token (s) % Savings
Syn. Hum. Syn. Hum.
Closed Book 0 Qwen2.5-7B 11.1 13.1 1.34 1.24✗ ✗
0 OpusLM 18.4 15.5 5.67 7.07✗ ✗
0 Kimi Audio 16.7 16.0 0.85 0.89✗ ✗
Open Book 23K Qwen2.5-7B 26.3 26.9 5.90 5.40✗ ✗
(S. 3.1) 15K OpusLM* 0.0 0.0 9.05 10.44✗ ✗
500 Kimi Audio 21.8 19.6 4.22 4.22✗ ✗
Streaming RAG 23K Qwen2.5-7B 34.2 37.4 5.32 3.60 20.7% 53.4%
(S. 3.2.2) 15K OpusLM 23.6 22.8 8.63 9.04 14.8% 41.5%
ˆQT
b(Eq. 3) for the tool query after block bas follows: 1when the current query is sufficiently similar to the
previous query (as determined by manually defined heuristics f(···)), we assign the special label NO_QUERY
to teach the model that no new tool call is needed. 2When the queries are sufficiently different, we assign
the actual pseudo ground truth query QT
bas the label to teach the model to trigger a new tool call:
ˆQT
b=(
NO_QUERY,iff( QT
b,ˆQprev
b) = True,
QT
b,else.(4)
For KG queries, we assign a NO_QUERY label when the current query exactly matches the previous one.
For web queries, we assign a NO_QUERY label if the top five retrieved documents for the current query
remain unchanged from the previous query.
We employ a multi-task fine-tuning strategy targeting two key capabilities. First, we train the model on
Streaming Tool Query Generation by optimizing P(QT
b|Q1:b,ˆQprev
b)forb∈[1, B], enabling intelligent decisions
about when to trigger tool queries. Second, we fine-tune on Response Generation by optimizing P(A|Q, R )
(S. 3.1) to improve the intelligibility of the speech output.
An important aspect of our post-training is enhancing the model’s ability to recover from errors in intermediate
query predictions. For example, when presented with the audio question, “Who founded Rare Beauty in
2019?”, we observed that an initial misinterpretation of ˆQprev
b, such as “Red Bull founder”, can lead the
model to subsequently generate NO_QUERY labels, effectively halting further attempts to retrieve the
correct information. This issue arises because, during training, the model is always provided with correct
previous query ˆQprev
b, whereas during inference, it may make errors due to partial utterances Q1:bbeing
ambiguous. Thus the model lacks the ability to recover from such mistakes during inference. To overcome
this, we introduce a novel strategy in which we deliberately inject negative samples during post-training by
substituting the previous query ˆQprev
bin Eq. 4 with incorrect ones Qneg
b. Crucially, when we perform negative
sampling, we fall back to the pseudo ground truth query QT
bas the training label:
(ˆQT
b,ˆQprev
b) =(
(ˆQT
b,ˆQprev
b),with probability0.9,
(QT
b, Qneg
b),with probability0.1.(5)
This approach explicitly teaches the model to recover from errors in intermediate query prediction, thereby
maintaining accuracy (see T. 5 for ablation) while achieving latency savings.
6

4 Experiment Setup
4.1 Evaluation Benchmarks
To rigorously evaluate our proposed approach, we construct comprehensive benchmark datasets featuring
spoken queries paired with simulated tool interactions. We begin with the CRAG dataset (Yang et al., 2024),
which form the basis for our spoken version of CRAG, which we termAudioCRAG. It consists of 2 distinct
variants: 1Audio CRAG Synthetic: To generate spoken queries, we use our in-house TTS system. We then
apply a rigorous filtering procedure which results in a high-quality set of 1,862 spoken queries, which we
refer to as theAudioCRAG-Syntheticbenchmark. 2Audio CRAG Human: To further enhance the realism
and diversity of our evaluation, we introduce the AudioCRAG-Human benchmark, which consists of 618
human-recorded spoken queries. We will release Audio CRAG Human benchmark upon acceptance to supports
the development of more natural and reliable voice assistants. Further details on the construction of these
benchmarks are provided in Sec. C. We follow the CRAG setup to incorporate both web and KG-based tools,
and adopt its robust evaluation methodology, as described in Secs. D and E. Additionally, we leverage a
random subset of 16,000 questions from the text-based factual question answering dataset TriviaQA (Joshi
et al., 2017) to post-train our speech-in, speech-out models, as detailed in Sec. G.
4.2 Evaluated SOTA Speech-in Speech-out Models
In this work, we present a comprehensive benchmark of three SOTA speech-in, speech-out conversational
systems: Qwen-OMNI (Xu et al., 2025), Kimi-Audio (Ding et al., 2025) and OpusLM (Tian et al., 2025). We
evaluate them under both tool-augmented and non-tool-augmented conditions. Further details about our
experimental setup are provided in S. F.
We perform an ablation study (referred to as “Tool Integration” in T. 4) where we post-train the model on
sequential query generation (i.e. P(QT|Q)in S. 3.1) and output generation, to assess the impact of streaming
RAG post-training versus standard post-training on final response generation. Additionally, we conduct an
ablation study on open book setting (S. 3.1) using a self-cascade approach with a three-stage inference pipeline:
(1) the audio question is used to generate a tool query QT(corresponding to the “Query Generation” stage
described in S. 3.1); (2) the audio question, and tool results are combined to produce the final text output
Xresby maximizing P(Xres|Q, R ); and (3) the audio question, and final text output are used to generate the
final speech output Aby optimizing P(A|Q, Xres). Since we teacher-force the text output to obtain the final
speech output in stage (3), this self-cascade approach can only be applied to a “thinker-talker” architecture
(eq. Qwen-OMNI) or Chain-of-Thought (CoT) style architectures (eg. OpusLM). The motivation for this
ablation is to investigate whether the inclusion of RAG references Rduring the “Response Generation” stage
(S. 3.1) affects the quality of the generated speechA.
5 Results
5.1 Impact of Tool Integration and Streaming RAG on SOTA Models
Table 1 provides a comprehensive performance comparison of three models, Qwen2.5-7B, OpusLM, and Kimi
Audio, evaluated across three settings: Closed Book, Open Book, and Streaming RAG. All models are assessed
on both the AudioCRAG-Synthetic (Syn.) and AudioCRAG-Human (Hum.). In the Closed Book setting,
where models rely solely on their internal knowledge without access to external tools (reference length = 0),
all models achieve accuracy scores below 20%. These results highlight the inherent limitations of closed-book
approaches in handling complex queries. The Open Book setting, which provides models with access to
external information, demonstrates the clear benefits of tool integration. Qwen2.5-7B and Kimi Audio’s
accuracy rises substantially, underscoring the value of leveraging external context. As expected, latency
increases due to the additional overhead of retrieving information from external tools (See T. 3 for detailed
latency analysis).
Most notably, our post-training approach to buildModel-Triggered Streaming RAG, as described in S. 3.2.2,
delivers significant advancements in both accuracy and efficiency. Qwen2.5-7B and OpusLM achieve significant
7

Table 2Results on AudioCRAG-Synthetic for Qwen2.5-7B, OpusLM, and Kimi Audio comparing text vs. speech
output across Closed Book, Open Book, andModel-Triggered Streaming RAG.
Setting Ref length Output Model Acc.
Closed Book 0 Text Qwen2.5-7B 15.0
0 Speech Qwen2.5-7B 11.1
0 Text OpusLM 20.1
0 Speech OpusLM 18.4
0 Text Kimi Audio 24.2
0 Speech Kimi Audio 16.7
Open Book 23K Text Qwen2.5-7B 39.6
(S. 3.1) 23K Speech Qwen2.5-7B 26.3
23K Speech (self-cascade) Qwen2.5-7B 33.8
15K Text OpusLM 26.3
15K Speech OpusLM 0.0
15K Speech (self-cascade) OpusLM 21.2
5K Text Kimi Audio 45.8
500 Speech Kimi Audio 21.8
Streaming RAG 23K Text Qwen2.5-7B 39.8
(S. 3.2.2) 23K Speech Qwen2.5-7B 34.2
15K Text OpusLM 29.7
15K Speech OpusLM 23.6
Table 3First Token Latency breakdown, showing median (P50) and 90th percentile (P90) timings, for the Qwen2.5-7B
on AudioCRAG-Synthetic.
Model Setting PLatency (sec)
Tool Use LatencyResponse Gen TotalQuery Gen Tool Results Gen
Qwen2.5-7BOpen Book P50 0.59 2.78 2.52 5.90
(S. 3.1) P90 0.85 4.90 3.25 9.00
Streaming RAG P50 0.59 2.20 2.52 5.32
(S. 3.2.2) P90 0.85 4.37 3.25 8.47
accuracy improvements across both benchmarks. While the absolute accuracy scores may appear low, they
are consistent with the evaluation results observed in the CRAG benchmark. Notably, Qwen2.5-7B with
Model-Triggered Streaming RAGachieves accuracy comparable to the open book performance of similarly
sized LLMs reported in the original CRAG paper (i.e., 34.2 for Qwen2.5-7B vs. 32.1 for LLAMA-3 8B Instruct
in (Yang et al., 2024)). Importantly, although post-training is performed exclusively on the synthetic dataset,
we observe consistent and even greater improvements on the human-spoken benchmark, demonstrating the
robustness and strong generalization capabilities of our method. This setting also introduces substantial
latency savings compared to the Open Book configuration, with Qwen2.5-7B and OpusLM achieving 20.7%
and 14.8% reductions in first-token latency on the synthetic benchmark, and even greater savings on the
human benchmark1. These results demonstrate that our streaming RAG approach not only advances the
accuracy of speech-in speech-out systems, but also optimizes response efficiency by enabling earlier and more
effective prediction of tool queries.
8

Figure 4Average latency savings (Mean) by streaming RAG approaches (S. 3.2.)
5.2 Analysis: Modality Gap Between Speech and Text Output
Table 2 provides a comparative evaluation of the three SOTA models in generating either text or speech outputs
from speech inputs, both with and without the integration of external tool results. In the absence of tool
results (Ref length = 0), all models achieve higher accuracy when generating text outputs compared to speech
outputs. The incorporation of tool results generally leads to improved text generation performance, with
Kimi Audio achieving the highest accuracy. In contrast, the accuracy for speech output remains consistently
lower across all models and conditions. The self-cascade approach, in which the model first generates an
intermediate text response before producing the final speech output, provides moderate improvements in
speech output accuracy for both Qwen2.5-7B and OpusLM. However, it still underperforms compared to the
text-out baseline, primarily due to errors in accurately generating answers involving uncommon entity nouns.
Overall, these findings underscore a persistent challenge in direct speech generation, particularly when tool
results are integrated, as this appears to negatively impact the quality of generated speech responses. Our
Model-Triggered Streaming RAGdemonstrates clear advantages: it maintains comparable performance for
text output and delivers substantial improvements for speech output, even outperforming the self-cascade
approach. These results underscore the effectiveness of post-training in overcoming challenges in direct speech
generation in tool-augmented scenarios.
5.3 Analysis of Latency Bottlenecks in Tool-Integrated Speech Dialogue
Table 3 provides a comprehensive breakdown of latency measurements for the Qwen2.5-7B speech-to-speech
model, evaluated in both the Open Book setting and our proposedModel-Triggered Streaming RAGsetting
on AudioCRAG-Synthetic. We report both median (P50) and 90th percentile (P90) values for each stage of
the processing pipeline. The latency is decomposed into three main components: tool query generation, tool
result generation, and speech response synthesis, with measurements provided for both the first token outputs
(We also provide latency measurements for last token output in T. 9). For both tool query and tool result
generation, it is assumed that all tools are accessed in parallel; thus, the reported latency corresponds to the
maximum query or result generation time among all tools. The majority of this latency arises from leveraging
external web pages, which introduces significant delays, most notably, increasing the first token latency by
2.3x in Open Book Setting. Notably, ourModel-Triggered Streaming RAGsetting enables early generation of
tool results, successfully reducing P50 first token latency by 9.8% and tool use latency by 20.7%.
5.4 Ablation Study
Streaming RAG approaches: Figure 4 highlights the substantial latency savings enabled by theModel-Triggered
Streaming RAG(S. 3.2.2), compared to theFixed-Interval Streaming RAG(S. 3.2.1). Three key metrics are
evaluated: overall latency savings, the percentage of queries benefiting from reduced latency, and the number
of parallel threads required. Even without any post-training, theFixed-Interval Streaming RAGapproach
1Our latency calculations on synthetic audio exclude end-point detection latency, which is required in all production systems.
Since our streaming RAG approach enables processing without waiting for end-point detection, including this factor would
further amplify the observed latency savings, as observed in higher latency savings for human-spoken audio, where endpoint
detection errors often introduce trailing silence.
9

Table 4Comparison of speech-to-speech model under different post-training conditions
Post-Training Ref length Post Train Data Model Acc.
Tool Integration (S. 4.2) 15K 16K OpusLM 22.4
Streaming RAG (S. 3.2.2) 15K 16K OpusLM 23.6
Tool Integration (S. 4.2) 23K 16K Qwen2.5-7B 34.9
Streaming RAG (S. 3.2.2) 23K 16K Qwen2.5-7B 34.2
Table 5Ablation Results for Negative Sampling Strategy inModel-Triggered Streaming RAG(S. 3.2.2).
Scenario Ref Length Post Train Data Output Model Acc.
Open Book 15K 0 Text Qwen2.5-7B 39.6
Post-train (S. 3.2.2) 15K 16K Text Qwen2.5-7B 39.8
- Negative sampling 15K 16K Text Qwen2.5-7B 36.5
already reduces tool usage latency (3.37s in T. 3) by 10.7% for Open Book Qwen-OMNI, demonstrating its
flexibility and plug-and-play compatibility with any existing speech-in speech-out model. TheModel-Triggered
Streaming RAGmethod, utilizing Qwen-OMNI, consistently delivers superior performance. It achieves greater
average latency reductions and benefits a higher proportion of queries with improved response times. Notably,
Model-Triggered Streaming RAGrequire only a single parallel thread, representing a significant advancement
in resource efficiency compared to theFixed-Interval Streaming RAGapproach, which demands multiple
parallel threads. These findings highlight the effectiveness ofModel-Triggered Streaming RAGin not only
minimizing latency, validating our intuition from Section 3.2.2, but also in optimizing system efficiency and
resource utilization.
Post-training Strategies: Table 4 presents performance comparison under different post-training conditions.
Incorporating streaming tool query generation during post-training (S. 3.2.2) results in comparable performance
for both models. These results suggest thatModel-Triggered Streaming RAGcan be integrated into post-
training process without negatively impacting model performance.
Ablation Results for Negative Sampling Strategy: Table 5 presents the ablation results evaluating the impact
of deliberately injecting negative samples during post-training (Eq. 5). The findings highlight that, without
negative sampling, streaming tool query generation can reduce final accuracy in text output settings, primarily
due to errors in final query generation, as detailed in S. 3.2.2. In contrast, our negative sampling approach
significantly enhances the model’s robustness, enabling it to recover from intermediate prediction errors. This
leads to consistently high accuracy while also achieving notable latency reductions (Tab. 3).
6 Conclusion and Discussion
In this work, we introduced the first comprehensive approach for integrating external tool usage directly into
E2E speech-in, speech-out dialogue systems. By incorporatingModel-Triggered Streaming RAGpipeline,
we further enhanced the model’s ability to leverage retrieved information and autonomously decide when
to trigger new tool queries, resulting in improved accuracy and responsiveness. Empirical evaluation on the
newly introduced AudioCRAG benchmark demonstrated that tool integration can more than double factual
question answering accuracy compared to closed-book models. Additionally, our streaming RAG approach
achieved a 20% reduction in tool usage latency, thereby preserving natural conversational flow. Overall, our
contributions advance the state of the art in spoken dialogue systems by enabling accurate, real-time, and
tool-augmented voice assistants. In addition, we are committed to open science and will release our training
code and Audio CRAG Human benchmark to support future research and development in this area.
10

References
Siddhant Arora, Kai-Wei Chang, Chung-Ming Chien, Yifan Peng, Haibin Wu, Yossi Adi, Emmanuel Dupoux, Hung-Yi
Lee, Karen Livescu, and Shinji Watanabe. On the landscape of spoken language models: A comprehensive survey,
2025a.https://arxiv.org/abs/2504.08528.
Siddhant Arora, Jinchuan Tian, Hayato Futami, Jee-weon Jung, Jiatong Shi, Yosuke Kashiwagi, Emiru Tsunoo, and
Shinji Watanabe. Chain-of-thought training for open e2e spoken dialogue systems. InInterspeech 2025: Proceedings
of the Annual Conference of the International Speech Communication Association. ISCA, 2025b.
J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking large language models in retrieval-augmented generation.arXiv
preprint arXiv:2309.01431, 2023a.https://arxiv.org/abs/2309.01431.
Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented
generation, 2023b.https://arxiv.org/abs/2309.01431.
Z. Chen, Z. Gu, L. Cao, J. Fan, S. Madden, and N. Tang. Symphony: Towards natural language query answering over
multi-modal data lakes. InProceedings of the Conference on Innovative Data Systems Research (CIDR), 2023c.
Bowen Cheng and Zhicheng Dou. Dailyqa: A benchmark to evaluate web retrieval augmented llms based on capturing
real-world changes.arXiv preprint arXiv:2505.17162, 2025.https://arxiv.org/abs/2505.17162.
Haggai Cohen et al. Wixqa: A multi-dataset benchmark for enterprise retrieval-augmented generation.arXiv preprint
arXiv:2505.08643, 2025.https://arxiv.org/abs/2505.08643.
Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang,
et al. Kimi-audio technical report.arXiv preprint arXiv:2504.18425, 2025.
Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, and Xie Chen. Enhancing speech-to-speech
dialogue modeling with end-to-end retrieval-augmented generation, 2025.https://arxiv.org/abs/2505.00028.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen
Wang. Retrieval-augmented generation for large language models: A survey, March 2024a. https://arxiv.org/abs/
2312.10997. arXiv:2312.10997.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen
Wang. Retrieval-augmented generation for large language models: A survey, March 2024b. https://arxiv.org/abs/
2312.10997. arXiv:2312.10997.
James Glass. Challenges for spoken dialogue systems. InProceedings of the 1999 IEEE ASRU Workshop, volume 696.
MIT Laboratory for Computer Science Cambridge, 1999.
Rongjie Huang, Mingze Li, Dongchao Yang, Jiatong Shi, Xuankai Chang, Zhenhui Ye, Yuning Wu, Zhiqing Hong,
Jiawei Huang, Jinglin Liu, et al. Audiogpt: Understanding and generating speech, music, sound, and talking head.
InProceedings of the AAAI, pages 23802–23804, 2024.
Lawrence Jang, Yinheng Li, Dan Zhao, Charles Ding, Justin Lin, Paul Pu Liang, Rogerio Bonatti, and Kazuhito
Koishida. Videowebarena: Evaluating long context multimodal agents with video understanding web tasks, 2025.
https://arxiv.org/abs/2410.19100.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised
challenge dataset for reading comprehension. InProceedings of the 55th Annual Meeting of the Association for
Computational Linguistics, Vancouver, Canada, July 2017. Association for Computational Linguistics.
Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and
Rongrong Ji. Video-rag: Visually-aligned retrieval-augmented long video comprehension, 2024.https://arxiv.org/
abs/2411.13093.
Zixian Ma, Weikai Huang, Jieyu Zhang, Tanmay Gupta, and Ranjay Krishna.m&m’s: A benchmark to evaluate
tool-use for multi-step multi-modal tasks. InEuropean Conference on Computer Vision (ECCV), 2024. arXiv
preprint arXiv:2403.11085.
Leander Melroy Maben, Gayathri Ganesh Lakshmy, Srijith Radhakrishnan, Siddhant Arora, and Shinji Watanabe.
Aura: Agent for understanding, reasoning, and automated tool use in voice-driven tasks, 2025. https://arxiv.org/
abs/2506.23049.
11

Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. A survey of multimodal retrieval-augmented generation, 2025.
https://arxiv.org/abs/2504.08748.
Ziqiao Meng, Qichao Wang, Wenqian Cui, Yifei Zhang, Bingzhe Wu, Irwin King, Liang Chen, and Peilin Zhao. Parrot:
Autoregressive spoken dialogue language modeling with decoder-only transformers. InAudio Imagination: NeurIPS
2024 Workshop AI-Driven Speech, Music, and Sound Generation, 2024.
Meta CRAG-MM Challenge Organizers. Crag-mm: Comprehensive multi-modal, multi-turn rag benchmark. KDD Cup
2025 Challenge; dataset and description hosted on Hugging Face, 2025. Includes both single-turn and multi-turn
conversational variants using egocentric and public images, spanning 13 domains, with mock search APIs for image
and web retrieval.
Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin
Algayres, Benoît Sagot, Abdelrahman Mohamed, and Emmanuel Dupoux. Generative spoken dialogue language
modeling.Trans. Assoc. Comput. Linguistics, 11:250–266, 2023. doi: 10.1162/TACL\_A\_00545. https://doi.org/
10.1162/tacl_a_00545.
Bo Ni, Zheyuan Liu, Leyao Wang, Yongjia Lei, Yuying Zhao, Xueqi Cheng, Qingkai Zeng, Luna Dong, Yinglong Xia,
Krishnaram Kenthapadi, Ryan Rossi, Franck Dernoncourt, Md Mehrab Tanjim, Nesreen Ahmed, Xiaorui Liu, Wenqi
Fan, Erik Blasch, Yu Wang, Meng Jiang, and Tyler Derr. Towards trustworthy retrieval augmented generation for
large language models: A survey, 2025.https://arxiv.org/abs/2502.06872.
Zhihong Ouyang et al. Hoh: A dynamic benchmark for evaluating the impact of outdated information on rag.arXiv
preprint arXiv:2503.04800, 2025.https://arxiv.org/abs/2503.04800.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph
retrieval-augmented generation: A survey, 2024.https://arxiv.org/abs/2408.08921.
Yifan Peng, Shakeel Muhammad, Yui Sudo, William Chen, Jinchuan Tian, Chyi-Jiunn Lin, and Shinji Watanabe.
OWSM v4: Improving open whisper-style speech models via data scaling and cleaning. InProceedings of the Annual
Conference of the International Speech Communication Association (INTERSPEECH), 2025.
Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech
recognition via large-scale weak supervision. InProc. ICML, 2023.
Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. UTMOS:
UTokyo-SaruLab system for voiceMOS challenge 2022. InInterspeech, pages 4521–4525, 2022. doi: 10.21437/
Interspeech.2022-439.
Miao Su, Zixuan Li, Zhuo Chen, Long Bai, Xiaolong Jin, and Jiafeng Guo. Temporal knowledge graph question
answering: A survey, 2025.https://arxiv.org/abs/2406.14191.
Jinchuan Tian, William Chen, Yifan Peng, Jiatong Shi, Siddhant Arora, Shikhar Bharadwaj, Takashi Maekaku, Yusuke
Shinohara, Keita Goto, Xiang Yue, et al. Opuslm: A family of open unified speech language models.arXiv preprint
arXiv:2506.17611, 2025.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou,
Quoc Le, and Thang Luong. Freshllms: Refreshing large language models with search engine augmentation, 2023.
https://arxiv.org/abs/2310.03214.
Jason Wei, Karina Nguyen, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and
William Fedus. Measuring short-form factuality in large language models, November 2024. https://arxiv.org/abs/
2411.04368. Includes the introduction of the SimpleQA benchmark.
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources to advance general
chinese embedding, 2023.
Zhifei Xie and Changqiao Wu. Mini-omni: Language models can hear, talk while thinking in streaming, 2024.
https://arxiv.org/abs/2408.16725.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. Benchmarking retrieval-augmented generation for medicine.
In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,Findings of the Association for Computational
Linguistics ACL 2024, pages 6233–6251, Bangkok, Thailand and virtual meeting, August 2024a. Association for
Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.372. https://aclanthology.org/2024.findings-acl.372 .
Guanming Xiong, Junwei Bao, and Wen Zhao. Interactive-KBQA: Multi-turn interactions for knowledge base question
answeringwithlargelanguagemodels. InLun-WeiKu, AndreMartins, andVivekSrikumar, editors,Proceedings of the
12

62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10561–10582,
Bangkok, Thailand, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.569.
https://aclanthology.org/2024.acl-long.569/.
Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang,
Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-omni technical report.arXiv preprint, 2025.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui,
Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang,
Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar,
Wen tau Yih, and Xin Luna Dong. Crag – comprehensive rag benchmark. In38th Conference on Neural Information
Processing Systems (NeurIPS 2024) Track on Datasets and Benchmarks, 2024.https://arxiv.org/pdf/2406.04744.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan
Liu, and Maosong Sun. Visrag: Vision-based retrieval-augmented generation on multi-modality documents, 2025.
https://arxiv.org/abs/2410.10594.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Xu Han, Zihang Xu, Yuanwei Xu, Weilin Zhao, Maosong Sun, and
Zhiyuan Liu. Beyond the turn-based game: Enabling real-time conversations with duplex models. InProc. EMNLP,
2024.
Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan
Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous
agents. InInternational Conference on Learning Representations (ICLR), 2024. https://webarena.dev . arXiv
preprint arXiv:2307.13854.
13

Appendix
A Benchmarking Text Dialogue Systems for Tool Usage
Recent advances in benchmarking text-based dialogue systems for tool usage (Chen et al., 2023b; Ouyang
et al., 2025; Cheng and Dou, 2025; Cohen et al., 2025; Xiong et al., 2024a; Vu et al., 2023; Xiong et al.,
2024b; Peng et al., 2024; Su et al., 2025; Ni et al., 2025) have primarily focused on evaluating factual question
answering and task completion within simulated environments. The CRAG benchmark (Yang et al., 2024) is a
leading example, featuring 4,409 question-answer pairs and providing mock APIs for both web and knowledge
graph (KG) search. CRAG supports a range of KG and web retrieval tasks, and highlights key challenges
such as hallucinations in retrieval-augmented generation (RAG) and the importance of leveraging KGs and
search ranking to improve factual accuracy. Evaluation is conducted automatically using two LLM judges.
SimpleQA (Wei et al., 2024) is another widely adopted benchmark, designed to assess language models on
short, fact-seeking questions. With 4,326 adversarially collected questions spanning diverse topics and a
straightforward grading scheme based on single, indisputable answers, SimpleQA provides a robust testbed
for factual accuracy. Moving beyond question answering, WebArena (Zhou et al., 2024) offers a simulated
environment for evaluating dialogue agents on web-based tasks using fully functional websites, enabling
assessment of more complex, action-oriented behaviors. While these benchmarks have significantly advanced
the evaluation of tool-augmented dialogue systems, they remain largely limited to text-based interactions and
do not fully address the unique challenges presented by speech-in speech-out systems.
B Multimodal Benchmarks for Tool Usage
Recent benchmarks Mei et al. (2025); Yu et al. (2025); Luo et al. (2024) have extended tool-augmented
dialogue evaluation to multimodal and longer-context scenarios. The m&m’s benchmark (Ma et al., 2024)
evaluates LLMs on multi-step, multi-modal tasks using a diverse set of 33 tools, including public APIs
and multimodal models such as off-the-shelf automatic speech recognition (ASR) models, highlighting the
potential for developing agents that leverage audio-based tools. CRAG_MM (Meta CRAG-MM Challenge
Organizers, 2025) builds on the original CRAG benchmark by introducing visual question answering (QA)
tasks that combine images and text-based queries, utilizing mock APIs for both image descriptions and
web search. For video understanding and long-context reasoning, the Video Web Arena (Jang et al., 2025)
benchmark evaluates multimodal agents on tasks involving 2,021 manually crafted tutorial videos. While
these benchmarks advance the field by incorporating multimodal tools, they still do not evaluate systems in
speech-in speech-out scenarios.
C Audio CRAG Benchmark
We begin with the CRAG dataset (Yang et al., 2024), which contains 2,706 text queries. Since these queries
are not directly suitable for speech-based evaluation, we first identify those requiring adaptation before TTS
conversion. Through careful manual inspection, we determine that queries containing dates or brackets
benefit from rewriting to ensure naturalness and clarity in spoken form. In total, we identify 569 such queries
and rewrite them using a large language model (LLAMA-4 Maverick). The resulting 569 rewritten queries,
combined with the remaining original queries, form the basis for our spoken version of CRAG, which we
termAudioCRAG. We follow the CRAG setup to incorporate web and KG-based tools and adopt its robust
evaluation setup, as detailed in S. D and E.
Audio CRAG Synthetic: To generate spoken queries, we process all 2,706 queries through our in-house TTS
system. We then apply a rigorous filtering procedure to remove queries with intelligibility issues, specifically
excluding any utterances for which Whisper Radford et al. (2023) hypotheses exhibit a non-zero word error
rate. We also remove utterances with suboptimal audio quality, as determined by UTMOS (Saeki et al.,
2022) scores below 3.5. This results in a high-quality set of 1,862 spoken queries, which we refer to as the
AudioCRAG-Syntheticbenchmark.
14

Audio CRAG Human: To further enhance the realism and diversity of our evaluation, we introduce the
AudioCRAG-Human benchmark, which consists of 618 human-recorded spoken queries. These queries are
recorded by a diverse pool of participants to capture natural variations in speech, accent, and prosody.
The inclusion of human-recorded audio enables a more comprehensive assessment of speech-in speech-out
systems under real-world conditions, providing valuable insights into model robustness and generalization
beyond synthetic speech. This benchmark serves as a critical resource for evaluating the effectiveness of tool
integration in conversational AI systems.
D Tool Usage Setup
To enable effective tool usage, we build upon the CRAG framework by integrating two complementary
information sources: web search, which provides access to fresh and dynamic content, and a knowledge graph,
which offers structured and reliable information. For web search, we aggregate all 100,000 documents from the
CRAG corpus and employ a BGE-based re-ranker2Xiao et al. (2023) to index and retrieve the top 50 most
relevant documents for each query. These documents are then segmented into chunks and re-ranked using
the same BGE model based on their similarity to the query, ensuring highly contextually relevant retrieval.
Meanwhile, queries to the knowledge graph are performed via a simulated API, adhering to the methodology
established in CRAG3.
Table 6Results on AudioCRAG-Synthetic for Qwen2.5-7B, OpusLM, and Kimi Audio comparing text vs. speech
output across Open Book setting showing average rates of accurate, hallucinated, and missing responses, as well as
overall truthfulness scores for each system.
Ref length Output Model Score Acc. Halluc Miss.
0 Text Qwen2.5-7B -13.1 15.0 28.1 56.9
0 Speech Qwen2.5-7B -21.1 11.1 32.3 56.6
0 Text OpusLM -44.3 20.1 64.3 15.6
0 Speech OpusLM -47.9 18.4 66.2 15.4
0 Text Kimi Audio -38.5 24.2 62.7 13.1
0 Speech Kimi Audio -53.5 16.7 70.3 12.9
E Evaluation Setting
Similar to previous work (Yang et al., 2024), we employ model-based automatic evaluation. We use a three-way
scoring system, assigning scores of 1, -1, and 0 for accurate, incorrect, and missing answers, respectively. The
evaluation is conducted using the Llama 4-maverick LLM evaluator. For speech outputs, we first transcribe
the audio using Whisper (Radford et al., 2023) before passing the transcriptions to the LLM evaluator. In
this study, our primary focus is on enhancing system accuracy; therefore, we report average accuracy values
in Tables 1 and 2. For additional context, we also provide the average rates of accurate, hallucinated, and
missing responses, as well as overall truthfulness scores for each system in the Open Book Setting (see Table 6).
Notably, our results indicate that Qwen-OMNI was less likely to generate hallucinated responses compared to
OpusLM and Kimi Audio.
F Experiment Setup of SOTA speech-in speech-out models
Qwen-OMNI(Xu et al., 2025) is a end-to-end multimodal model that seamlessly integrates diverse input
modalities—including text, images, audio, and video—and generates both text and natural speech responses
in a real-time streaming fashion. It leverages an innovative Thinker-Talker architecture, where the Thinker
2https://huggingface.co/BAAI/bge-large-en-v1.5
3https://github.com/facebookresearch/CRAG/tree/main/mock_api
15

module performs high-level reasoning to produce a text response, which is then used by the Talker module,
conditioning on both the text and the Thinker’s hidden representations, to generate streaming speech output.
OpusLM(Tian et al., 2025) is an open-source speech-in, speech-out model post-trained to directly answer
complex semantic and factual questions from raw audio inputs, through Chain-of-Thought reasoning.
Kimi Audio(Ding et al., 2025) is a universal audio foundation model that unifies audio understanding,
generation, and conversational abilities within a single framework. Pre-trained on over 13 million hours of
diverse audio and text data, Kimi Audio achieves state-of-the-art performance across a wide range of audio
benchmarks, including audio understanding and speech conversation tasks.
For tool-augmented scenarios, retrieval results are provided up to each model’s maximum token limit (“Ref
length” in Tables), maintaining a 2:1 ratio of web page to KG results. Specifically, we observe that Kimi-Audio
is currently optimized for handling tool result references up to a certain length. When this limit is exceeded,
an error arises during the audio detokenization process, specifically within the rotary embedding mechanism,
highlighting an architectural constraint in processing longer input sequences or larger reference contexts.
Addressing this limitation presents a valuable opportunity for future model enhancements.
Our evaluation encompasses both speech-in-text-out and speech-in-speech-out scenarios. For the Fixed-Interval
Streaming RAG setting (Section 3.2.1), intermediate tool queries are generated at consistent 1-second intervals.
In the Model-Triggered Streaming RAG setting, the model dynamically determines the need for a tool call
after processing each 500ms block. This approach allows us to utilize a smaller chunk size, as only a single tool
call thread is required for Model-Triggered Streaming RAG, thereby enabling more efficient and responsive
processing.
G Post Training Data Preparation
This subsection details the experimental setup for post-training the pretrained speech-in, speech-out model
to significantly enhance its tool usage capabilities, as outlined in S. 3.2.2. We leverage a random subset of
16,000 questions from the text-based factual question answering dataset TriviaQA (Joshi et al., 2017), which
contains 97,000 questions and 662,659 associated web documents. ForModel-Triggered Streaming RAG, we
further compute word-level timestamps using a pre-trained ASR model, OWSM CTC v4 1B (Peng et al.,
2025), enabling us to generate partial ASR transcripts Xasr
bat 500 ms intervals. Note that if word occurs at
boundary of block b, it is excluded from Xasr
b. For each partial transcript Xasr
b, we generate corresponding
psuedo ground truth queries QT
busing LLAMA-4-Maverick to simulate incremental user input. To simulate
realistic tool usage, we concatenate all documents and employ a web query reranker to retrieve the top 50
most relevant documents for each query. The text questions from this 16k subset are converted into discrete
speech tokens using text-to-speech synthesis with the corresponding pretrained speech-in, speech-out model.
Recognizing that TriviaQA answers are typically single named entities, we further transform the queries into
a conversational style using LLAMA-4-Maverick, making them more suitable for dialogue-based evaluation.
H Prompts used for Factual QA
H.1 Prompt in Closed Book Setting.
PROMPT = """ You are given an Audio Question and the time when it was asked in the Pacific Time Zone
(PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT". Your task
is to answer the question in as few words as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer “invalid question”.
2. If you are uncertain or don’t know the answer, respond with “I don’t know”.
### Question
{query}
### Query Time
{query_time}
16

Table 7Example KG Queries, and Web Queries generated by Qwen-OMNI in Open Book setting
ASR Transcript of QuestionXasrWeb Query ˆQwebKG Query ˆQKG
which of nolan greenwald’s
movies has achieved the highest
level of box office success on a
global scale?NolanGreenwald’shighest-
grossing movie{’domain’: ’movie’,
’movie_name’: "Nolan
Greenwald’s movies",
’movie_aspect’: ’revenue’}
who has played drums for the red
hot chili peppers?Red Hot Chili Peppers
drummers{’domain’: ’music’,
’artist_name’: ’Red Hot
Chili Peppers’, ’artist_as-
pect’: ’member’}
what’s the current stock price of
tortoise midstream energy fund?Tortoise Midstream En-
ergy Fund stock price{’domain’: ’finance’, ’mar-
ket_identifier’: ’Tortoise
Midstream Energy Fund’,
’metric’: ’price’, ’datetime’:
’02/28/2024’}
whatwasthevolumeoftradingin
cabot corporation’s stock on the
most recent day that dividends
were distributed?CABOT Corp stock trad-
ingvolumeondividenddis-
tribution date{’domain’: ’finance’, ’mar-
ket_identifier’: ’Cabot
Corporation’, ’metric’:
’dividend’, ’datetime’:
’02/28/2024’}
which movie won the academy
award for best film in 2020?2020 Academy Award for
Best Picture{’domain’: ’movie’,
’movie_aspect’: ’oscar_-
awards’, ’year’: 2020}
which teams have won against
phoenix suns during 2022-12?Teams that beat Phoenix
Suns in December 2022{’domain’: ’sports’,
’sport_type’: ’basketball’,
’team’: ’Phoenix Suns’,
’datetime’: ’2022-12-15’}
### Answer
"""
H.2 Prompt in Open Book / Streaming RAG Setting.
PROMPT = """ You are given an Audio Question, References and the time when it was asked in the Pacific
Time Zone (PT), referred to as "Query Time". The query time is formatted as "mm/dd/yyyy, hh:mm:ss PT".
The references may or may not help answer the question. Your task is to answer the question in as few words
as possible.
Please follow these guidelines when formulating your answer:
1. If the question contains a false premise or assumption, answer “invalid question”.
2. If you are uncertain or don’t know the answer, respond with “I don’t know”.
### Question
{query}
### Query Time
{query_time}
### References
# web
{web_results}
# knowledge graph
{kg_response}
### Answer
"""
17

H.3 KG Query extraction in Open Book Setting.
PROMPT = """ You are an agent that only outputs JSON. You are given a Query and Query Time. Do the
following:
1) Determine the domain the query is about. The domain should be one of the following: "finance",
"sports", "music", "movie", "encyclopedia". If none of the domains apply, use "other". Use "domain" as the
key in the result json.
2) Extract structured information from the query. Include different keys into the result json depending
on the domains, and put them DIRECTLY in the result json. Here are the rules:
For ‘encyclopedia’ and ‘other’ queries, these are possible keys:
- ‘main_entity’: extract the main entity of the query.
For ‘finance’ queries, these are possible keys:
- ‘market_identifier’: stock identifiers including individual company names, stock symbols.
- ‘metric’: financial metrics that the query is asking about. This must be one of the following: ‘price’,
‘dividend’, ‘P/E ratio’, ‘EPS’, ‘marketCap’, and ‘other’.
- ‘datetime’: time frame that the query asks about. When datetime is not explicitly mentioned, use ‘Query
Time’ as default.
For ‘movie’ queries, these are possible keys:
- ‘movie_name’: name of the movie
- ‘movie_aspect’: if the query is about a movie, which movie aspect the query asks. This must be one of
the following: ‘budget’, ‘genres’, ‘original_language’, ‘original_title’, ‘release_date’, ‘revenue’, ‘title’, ‘cast’,
‘crew’, ‘rating’, ‘length’.
- ‘person’: person name related to moves
- ‘person_aspect’: if the query is about a person, which person aspect the query asks. This must be one of
the following: ‘acted_movies’, ‘directed_movies’, ‘oscar_awards’, ‘birthday’.
- ‘year’: if the query is about movies released in a specific year, extract the year
For ‘music’ queries, these are possible keys:
- ‘artist_name’: name of the artist
- ‘artist_aspect’: if the query is about an artist, extract the aspect of the artist. This must be one of the
following: ‘member’, ‘birth place’, ‘birth date’, ‘lifespan’, ‘artist work’, ‘grammy award count’, ‘grammy award
date’.
- ‘song_name’: name of the song
- ‘song_aspect’: if the query is about a song, extract the aspect of the song. This must be one of the following:
‘author’, ‘grammy award count’, ‘release country’, ‘release date’.
For ‘sports’ queries, these are possible keys:
- ‘sport_type’: one of ‘basketball‘, ‘soccer‘, ‘other‘
- ‘tournament’: NBA, World Cup, Olympic.
- ‘team’: teams that users are interested in.
- ‘datetime’: time frame that the user is interested in. When datetime is not explicitly mentioned, use ‘Query
Time’ as default.
Return the results in a FLAT json.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON!!!*
"""
18

H.4 KG Query extraction in Streaming RAG Setting.
PROMPT = """ You are an agent that only outputs JSON. You are given an Audio Query, Previously
generated JSON result (’Previous Result’) and Query Time. Do the following:
1) Determine the domain the query is about. The domain should be one of the following: ¨finance¨, ¨ sports¨,
¨ music¨, ¨ movie¨, ëncyclopedia¨. If none of the domains apply, use öther¨. Use¨domainäs the key in the result json.
2) Extract structured information from the query. Include different keys into the result json depending on the
domains, and put them DIRECTLY in the result json. Here are the rules:
For ‘encyclopedia’ and ‘other’ queries, these are possible keys: - ‘main_entity’: extract the main entity of the
query.
For ‘finance’ queries, these are possible keys: - ‘market_identifier’: stock identifiers including individual
company names, stock symbols. - ‘metric’: financial metrics that the query is asking about. This must be one
of the following: ‘price’, ‘dividend’, ‘P/E ratio’, ‘EPS’, ‘marketCap’, and ‘other’. - ‘datetime’: time frame
that the query asks about. When datetime is not explicitly mentioned, use ‘Query Time’ as default.
For ‘movie’ queries, these are possible keys: - ‘movie_name’: name of the movie - ‘movie_aspect’: if the
query is about a movie, which movie aspect the query asks. This must be one of the following: ‘budget’,
‘genres’, ‘original_language’, ‘original_title’, ‘release_date’, ‘revenue’, ‘title’, ‘cast’, ‘crew’, ‘rating’, ‘length’.
- ‘person’: person name related to moves - ‘person_aspect’: if the query is about a person, which person
aspect the query asks. This must be one of the following: ‘acted_movies’, ‘directed_movies’, ‘oscar_awards’,
‘birthday’. - ‘year’: if the query is about movies released in a specific year, extract the year
For ‘music’ queries, these are possible keys: - ‘artist_name’: name of the artist - ‘artist_aspect’: if the query
is about an artist, extract the aspect of the artist. This must be one of the following: ‘member’, ‘birth place’,
‘birth date’, ‘lifespan’, ‘artist work’, ‘grammy award count’, ‘grammy award date’. - ‘song_name’: name of
the song - ‘song_aspect’: if the query is about a song, extract the aspect of the song. This must be one of the
following: ‘author’, ‘grammy award count’, ‘release country’, ‘release date’.
For ‘sports’ queries, these are possible keys: - ‘sport_type’: one of ‘basketball‘, ‘soccer‘, ‘other‘ - ‘tournament’:
NBA, World Cup, Olympic. - ‘team’: teams that users are interested in. - ‘datetime’: time frame that the
user is interested in. When datetime is not explicitly mentioned, use ‘Query Time’ as default. Return the
results in a FLAT json.
*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON!!!*
3) Compare your newly generated result to the ’Previous Result’. **If your new result would be exactly the
same as the ’Previous Result’, output only NO_QUERY.** Return the results in a FLAT json.
Previous Result:
{prev_kg_query}
"""
H.5 Web Query extraction in Open Book Setting.
PROMPT = """ You are given an Audio Query and Query Time. Your task is to generate a web query that
can be used to retrieve relevant web pages. Rewrite the following query into a short and succinct form, focusing
on the main topic or domain (e.g. finance, sports, music, movie, encyclopedia), key entities mentioned (e.g.
people, organizations, locations), and specific aspects of those entities (e.g. performance metrics, relationships,
events). Ensure the rewritten query is clear, concise, and easy to understand. Note that simply outputting
the original query is not acceptable. You must rephrase the query to make it more concise and focused on the
key information that will help retrieve relevant web pages.
For ’finance’ queries, focus on: - Company names or stock symbols - Financial metrics (e.g. price, dividend,
P/E ratio, EPS, marketCap) - Specific timeframes or events; if no timeframe is specified, use the Query Time
as default
19

For ’sports’ queries, focus on: - Sports Type (eg. basketball, soccer) - Teams, players - Statistics or performance
metrics (e.g. scores, wins, losses) - Specific events or tournaments (eg. NBA, World Cup, Olympic) - Time
frame that the user is interested in; if no timeframe is specified, use the Query Time as default
For ’music’ queries, focus on: - Artist names or song titles - Specific aspects of artist (eg. band name, birth
place, birth date, lifespan, artist work, grammy award count, grammy award date) - Specific aspects of song
(eg. author, grammy award count, release country, release date) - Music genres or categories - Specific awards
or recognition (e.g. Grammy Awards, Billboard)
For ’movie’ queries, focus on: - Movie titles or celebrity names - Movie genres or other categories like budget,
language, release_date, revenue, cast, crew, rating, length - Specific aspects of celebrity like acted_movies,
directed_movies, oscar_awards, birthday - Specific awards or recognition (e.g. Oscars) For ’other’ queries,
focus on: - Main entity or topic - Specific aspects or attributes of the entity
When rewriting the query, ensure that it captures all important information from the original question that
could impact the retrieval results. Do not omit any crucial details, such as specific dates, locations, or
relationships between entities. Also, do not invent any new details on your own. If necessary, use the Query
Time to provide context for the query. The goal is to create a concise and accurate query that effectively
conveys the user’s intent and retrieves relevant information. *NEVER include ANY EXPLANATION or
NOTE in the output, ONLY OUTPUT THE REWRITTEN QUERY!!!* """
H.6 Web Query extraction in Streaming RAG Setting.
PROMPT = """You are given an Audio Query, previously generated Web query (’Previous Result’) and
Query Time.
Your task is to generate a web query that can be used to retrieve relevant web pages. Rewrite the following
query into a short and succinct form, focusing on the main topic or domain (e.g. finance, sports, music, movie,
encyclopedia), key entities mentioned (e.g. people, organizations, locations), and specific aspects of those
entities (e.g. performance metrics, relationships, events). Ensure the rewritten query is clear, concise, and
easy to understand.
Note that simply outputting the original query is not acceptable. You must rephrase the query to make it
more concise and focused on the key information that will help retrieve relevant web pages.
For ’finance’ queries, focus on: - Company names or stock symbols - Financial metrics (e.g. price, dividend,
P/E ratio, EPS, marketCap) - Specific timeframes or events; if no timeframe is specified, use the Query Time
as default
For ’sports’ queries, focus on: - Sports Type (eg. basketball, soccer) - Teams, players - Statistics or performance
metrics (e.g. scores, wins, losses) - Specific events or tournaments (eg. NBA, World Cup, Olympic) - Time
frame that the user is interested in; if no timeframe is specified, use the Query Time as default
For ’music’ queries, focus on: - Artist names or song titles - Specific aspects of artist (eg. band name, birth
place, birth date, lifespan, artist work, grammy award count, grammy award date) - Specific aspects of song
(eg. author, grammy award count, release country, release date) - Music genres or categories - Specific awards
or recognition (e.g. Grammy Awards, Billboard)
For ’movie’ queries, focus on: - Movie titles or celebrity names - Movie genres or other categories like budget,
language, release_date, revenue, cast, crew, rating, length - Specific aspects of celebrity like acted_movies,
directed_movies, oscar_awards, birthday - Specific awards or recognition (e.g. Oscars)
For ’other’ queries, focus on: - Main entity or topic - Specific aspects or attributes of the entity
When rewriting the query, ensure that it captures all important information from the original question that
could impact the retrieval results. Do not omit any crucial details, such as specific dates, locations, or
relationships between entities. Also, do not invent any new details on your own. If necessary, use the Query
Time to provide context for the query. The goal is to create a concise and accurate query that effectively
conveys the user’s intent and retrieves relevant information. Now, compare the new web query to the previously
generated web query (’Previous Result’).
20

If the new query is similar enough to the previous web query (i.e., it effectively conveys the same user intent
and would retrieve similar relevant information), output only *NO_QUERY*.
Previous Result:
{prev_web_query}
"""
I LLM as judge prompt
PROMPT="""
Assume you are a human expert in grading predictions given by a model. You are given a question and a
model prediction. Judge if the prediction matches the ground truth answer by following these steps: 1: Take
it as granted that the Ground Truth is always correct. 2: If the Prediction indicates it is not sure about the
answer, "score" should be "0"; otherwise, go the next step. 3: If the Prediction exactly matches the Ground
Truth, "score" is 1. 4: If the Prediction does not exactly match the Ground Truth, go through the following
steps and likely give a score as 0. 5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction
gives a number that almost exactly matches the ground truth. 6: If the Prediction is self-contradictory, "score"
must be 0. 7: If the prediction is not answering the question, "score" must be 0. 8: If the prediction is a
concise and correct summary of the ground truth, "score" is 1. 9: If ground truth contains a set of items,
prediction must contain exactly same items for the score to be 1. 10: Otherwise, "score" is 0.
Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score"
field with value 1 or 0.
You should make the judgment based on provided examples. Examples: Question: "which company has
higher eps, btu or cma?" Ground Truth: "cma" Prediction: "it is not possible to determine which company
has a higher eps." Output: "score": 0, "explanation": "The prediction is not sure about the answer."
Question: "who is taller, a or b?" Ground Truth: "a" Prediction: "The answer is a. a is of height 1.75 m and b
is of height 1.82 m. So b is taller." Output: "score": 0, "explanation": "The prediction is self-contradictory."
Question: "who among patricia place and lana clarkson is younger?" Ground Truth: "lana clarkson" Prediction:
"lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)." Output:
"score": 1, "explanation": "The prediction gives the correct answer."
Question: "did lakers win more games than warriors in 2024-4?" Ground Truth: "yes" Prediction: "yes, lakers
won more games than warriors in 2023-2024 season." Output: "score": 0, "explanation": "The prediction is
not answering the question."
Question: "what is the height of the building where citigroup is headquartered?" Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)." Output: "score": 0, "explanation": "The prediction, 151 m, does
not match the ground truth, 279 m."
Question: "who were the last 3 prime ministers of uk?" Ground Truth: "rishi sunak, liz truss, and boris
johnson" Prediction: "1. liz truss, 2. boris johnson, 3.theresa may" Output: "score": 0, "explanation": "The
prediction does not contain item, rishi sunak, that is in the grount truth."
Question: "who authored the taming of the shrew?" Ground Truth: "william shakespeare" Prediction:
"william shakespeare" Output: "score": 1, "explanation": "The prediction exactly matches the ground truth."
Question: "who authored The Taming of the Shrew?" Ground Truth: "william shakespeare" Prediction:
"w shakespeare" Output: "score": 1, "explanation": "The prediction matches the ground truth as w is the
abbreviation of william."
Question: "what is the state bird of california?" Ground Truth: "california quail" Prediction: "california
valley quail" Output: "score": 1, "explanation": "The prediction matches the ground truth as california quail
is also known as california valley quail."
21

Question: "what country is maharashtra metro rail corporation limited located in?" Ground truth: "india"
Prediction: "indianapolis" Output: "score": 0, "explanation": "The prediction doesn’t match the ground
truth answers."
Question: "what was wolves’s record in their last game in eng-premier league?" Ground Truth: "w" Prediction:
"wolves’s record in their last game in the eng-premier League was a 2-1 win over fulham." Output: "score": 1,
"explanation": "The prediction correctly answers the result is a win."
Question: "how many seconds is 3 minutes 15 seconds?" Ground Truth: "195 seconds" Prediction: "3 minutes
15 seconds is 195 seconds." Output: "score": 1, "explanation": "The prediction matches ground truth as it
correctly states the number of seconds asked in the question."
Question: "on which days did am distribute dividends in the last year?" Ground Truth: "2023-01-24, 2023-04-
25, 2023-07-25, 2023-10-24" Prediction: "the company distributed dividends on october 24, 2023 and january
24, 2023." Output: "score": 0, "explanation": "The prediction partially matches the ground truth, but it is
incomplete."
Question: "on which days did xxx distribute dividends in the last year?" Ground Truth: "2023-01-13, 2023-
03-25, 2023-11-21" Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: "score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground
truth."
Question: "who are current band members of maroon 5?" Ground Truth: "adam Levine, jesse carmichael,
james valentine, matt flynn, pj morton, sam farrar" Prediction: "they are jesse carmichael, matt flynn, adam
levine, pj morton, sam farrar, and james valentine." Output: "score": 1, "explanation": "The prediction
exactly matches the ground truth."
Question: "which movies comprise the matrix franchise?" Ground Truth: "the matrix, the matrix reloaded,
the matrix revolutions, the matrix resurrections" Prediction: "the matrix, the matrix reloaded, the matrix
revolutions, the animatrix, and the matrix resurrections." Output: "score": 0, "explanation": "The prediction
covers more items than what are given by the ground truth."
Question: "how deep is the deepest lake of new york?" Ground Truth: "618 ft" Prediction: "the deepest lake
in new york is seneca lake, with a depth of 618.23 feet." Output: "score": 1, "explanation": "The prediction
exactly matches the number in ground truth after rounding."
Question: "what is the closing price of meta yesterday?" Ground Truth: "$310.17" Prediction: "310.2" Output:
"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."
Question: "what is the current market cap of appl?" Ground Truth: "2.81 trillion" Prediction: "2.667 trillion"
Output: "score": 0, "explanation": "The prediction does not match the number in ground truth."
Question: "what is the current pe ratio of appl?" Ground Truth: "28.3" Prediction: "the current pe ratio of
apple is 26.66" Output: "score": 0, "explanation": "The prediction does not match the number in ground
truth."
Question: "how much is tesla’s stock price down from its all-time high?" Ground Truth: "$221.83" Prediction:
"209.52" Output: "score": 0, "explanation": "The prediction does not match the number in ground truth."
Question: "what is the length of amazon river?" Ground Truth: "over 4000 miles" Prediction: "the length of
amazon river is 4,000 miles" Output: "score": 0, "explanation": "The prediction does not say Amazon River
is longer than 4000 miles."
Question: "how many copies x were sold?" Ground Truth: "2 million." Prediction: "it is over 2 million"
Output: "score": 0, "explanation": "The prediction does not match the ground truth."
Question: "what is the population of country x?" Ground Truth: "3,576,873" Prediction: "the population of
country x is 3.3 million." Output: "score": 0, "explanation": "The prediction, 3.3 M, does not match the
number, 3.6 M, in ground truth."
Question: "what is the current market value of stock x?" Ground Truth: "$2,237,578,268" Prediction: "$2.16
billion." Output: "score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in
22

ground truth."
Question: "how old is michael jordan?" Ground Truth: "61" Prediction: "mj was born on February 17, 1963."
Output: "score": 0, "explanation": "The prediction does not explicitly say the age."
Question: "how long did amy teach as a teacher?" Ground Truth: "24 years and 255 days" Prediction: "25
years." Output: "score": 0, "explanation": "The prediction does not exactly match the ground truth."
Question: "at what age did usher perform in the half time show of superbowl?" Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show." Output: "score": 0, "explanation":
"The prediction does not match the ground truth."
Question: "what year did olivia rodrigo win the grammys best new artist award?" Ground Truth: "2022"
Prediction: "sure! here’s an answer in three words or fewer: 2022 grammy awards." Output: "score": 1,
"explanation": "The prediction answers the correct number of the year."
Question: "how many times have zendaya and tom holland starred in a film together?" Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and
spider-man: no way home." Output: "score": 1, "explanation": "The prediction gives the right number and it
is reasonable."
Question: "who is younger, timothee chalamet or tom holland?" Ground Truth: "tom holland" Prediction:
"timothée chalamet is younger than tom holland." Output: "score": 0, "explanation": "The prediction does
not match the ground truth."
Question: "who had more number one hits on the us billboard, a or b?" Ground Truth: "a had more number
one hits on the us billboard than b, with 20 number one hits compared to b’s 15." Prediction: "a" Output:
"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."
Question: "what is xxx’s birthdate?" Ground Truth: "1996-01-01." Prediction: "02/01/1996" Output: "score":
0, "explanation": "The prediction does not match the ground truth."
Question: "what was the worldwide box office haul for movie x?" Ground Truth: "101756123." Prediction:
"102 million" Output: "score": 1, "explanation": "The prediction exactly matches the number in ground
truth after rounding."
Question: "how much has spotify’s user base increased by since 2020 in na?" Ground Truth: "spotify’s user
base increased by 34 million since 2020." Prediction: "spotify’s north american user base increased from 36
million in 2020 to 85 million by 2021" Output: "score": 0, "explanation": "The prediction is not answering
the question as it only gives the increase from 2020 to 2021."
"""
23

Table 8Streaming Tool Queries generated by Qwen-OMNI afterModel-Triggered Streaming RAGpost-training
ASR Tran-
script of
Question
XasrPartial ASR TranscriptXasr
bWeb Query ˆQweb
bKG Query ˆQKG
b
Who
founded
Rare Beauty
in 2019?Who founded Who founded what {"domain": "other", "main_-
entity": "Who"}
Who founded rare Red Bull founder {"domain": "other", "main_-
entity": "Red"}
Who founded rare beauty Who founded Rare Beauty {"domain": "other", "main_-
entity": "Rare Beauty"}
Who founded rare beauty in NO_QUERY NO_QUERY
Who founded rare beauty in
2019?NO_QUERY NO_QUERY
How many
jump shots
did Darius
Miles score
with in the
game on
2000-11-08?how many Number of queries on
03/27/2024{"domain": "other", "main_-
entity": "How many"}
how many jump Number of jump shots in bas-
ketball{"domain": "other", "main_-
entity": "jump shot"}
how many jump shots did Derek Jeter jump shots count {"domain": "sports",
"sport_type": "other",
"person": "Dairy", "date-
time": "03/27/2024, 19:52:43
PT"}
howmanyjumpshotsdidDar-
ius milesDarius jump shots count {"domain": "sports",
"sport_type": "other",
"person": "Darius", "date-
time": "03/27/2024, 19:52:43
PT"}
howmanyjumpshotsdidDar-
ius miles scoreDariusMilesjumpshotscount {"domain": "sports",
"sport_type": "other",
"person": "Darius Miles",
"datetime": "03/27/2024,
19:52:43 PT"}
howmanyjumpshotsdidDar-
ius miles score with inDarius Miles jump shots
scoredNO_QUERY
howmanyjumpshotsdidDar-
ius miles score with in the
game onDarius Miles jump shots
scored in game on 03/27/2024NO_QUERY
howmanyjumpshotsdidDar-
ius miles score with in the
game on NovemberDarius Miles jump shots
scored in game on November
2024{"domain": "sports",
"sport_type": "other",
"person": "Darius Miles",
"datetime": "November"}
howmanyjumpshotsdidDar-
ius miles score with in the
game on November 8Darius Miles jump shots
scored on November 8{"domain": "sports",
"sport_type": "other",
"person": "Darius Miles",
"datetime": "November 8"}
howmanyjumpshotsdidDar-
ius miles score with in the
game on November 8NO_QUERY NO_QUERY
howmanyjumpshotsdidDar-
ius miles score with in the
game on November 8, 2000Darius Miles jump shots
scored on November 8, 2000{"domain": "sports",
"sport_type": "other",
"person": "Darius Miles",
"datetime": "November 8,
2000"}
24

Table 9Last-token Latency breakdown, showing median (P50) and 90th percentile (P90) timings, for the Qwen2.5-7B
in Open Book Setting on AudioCRAG-Synthetic (First Token Latency=5.9 sec in T. 1).
Model Token PLatency (sec)
Tool LatencyResponse Gen TotalQuery Gen Tool Results Gen
Qwen2.5-7B Last TokenP50 0.59 2.78 16.70 20.07
P90 0.85 4.90 42.41 48.16
Table 10Post-training Parameters of OpusLM
Parameter Value
train_micro_batch_size_per_-
gpu1
gradient_accumulation_steps 2
epochs 2
gradient_clipping 1.0
bf16 enabled true
optimizer type Adam
optimizer lr 0.00001
optimizer betas [0.9, 0.95]
optimizer eps 1e-8
optimizer weight_decay 3e-7
optimizer adam_w_mode true
scheduler type WarmupDecayLR
scheduler warmup_type linear
scheduler total_num_steps 21534
scheduler warmup_num_steps 1077
scheduler warmup_min_lr 0
scheduler warmup_max_lr 0.00001
Table 11Post-training Parameters of Qwen-OMNI Thinker
Parameter Value
bf16 True
gradient_accumulation_steps 4
epochs 1
gradient_clipping 1.0
learning_rate 7e-6
lr_scheduler_type cosine
warmup_ratio 0.05
per_device_train_batch_size 1
weight_decay 0.01
25

Table 12Post-training Parameters of Qwen-OMNI Talker
Parameter Value
bf16 True
gradient_accumulation_steps 4
gradient_clipping 1.0
epochs 2
learning_rate 5e-5
per_device_train_batch_size 1
lr_scheduler_type linear
warmup_ratio 0.0
weight_decay 0.01
26