# VoiceAgentRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures

**Authors**: Jielin Qiu, Jianguo Zhang, Zixiang Chen, Liangwei Yang, Ming Zhu, Juntao Tan, Haolin Chen, Wenting Zhao, Rithesh Murthy, Roshan Ram, Akshara Prabhakar, Shelby Heinecke, Caiming Xiong, Silvio Savarese, Huan Wang

**Published**: 2026-03-02 18:58:54

**PDF URL**: [https://arxiv.org/pdf/2603.02206v2](https://arxiv.org/pdf/2603.02206v2)

## Abstract
We present VoiceAgentRAG, an open-source dual-agent memory router that decouples retrieval from response generation. A background Slow Thinker agent continuously monitors the conversation stream, predicts likely follow-up topics using an LLM, and pre-fetches relevant document chunks into a FAISS-backed semantic cache. A foreground Fast Talker agent reads only from this sub-millisecond cache, bypassing the vector database entirely on cache hits.

## Full Text


<!-- PDF content starts -->

VOI CEAG ENTRAG: Solving the RAG Latency Bottleneck in
Real-Time Voice Agents Using Dual-Agent Architectures
Jielin Qiu, Jianguo Zhang, Zixiang Chen, Liangwei Yang, Ming Zhu, Juntao Tan,
Haolin Chen, Wenting Zhao, Rithesh Murthy, Roshan Ram, Akshara Prabhakar,
Shelby Heinecke, Caiming, Xiong, Silvio Savarese, Huan Wang
Salesforce AI Research
https://github.com/SalesforceAIResearch/VoiceAgentRAG
Abstract
Retrieval-Augmented Generation (RAG) enables language models to ground their
responses in external knowledge, but the retrieval step introduces latency that is in-
compatible with real-time voice conversations. A typical vector database query adds
50–300ms to the response pipeline, which, combined with embedding computation and
LLM generation, pushes total latency well beyond the 200ms budget required for natu-
ral conversational flow. We present VOIC EAG ENTRAG, an open-sourcedual-agent
memory routerthat decouples retrieval from response generation. A backgroundSlow
Thinkeragent continuously monitors the conversation stream, predicts likely follow-up
topics using an LLM, and pre-fetches relevant document chunks into a FAISS-backed se-
mantic cache. A foregroundFast Talkeragent reads only from this sub-millisecond cache,
bypassing the vector database entirely on cache hits. We evaluate VOIC EAGENTRAG
on a 200-query benchmark across 10 diverse conversation scenarios using Qdrant Cloud
as the production vector database, achieving a75% overall cache hit rate(79% on warm
turns) and a316 ×retrieval speedup(110ms →0.35ms) on cache-hit queries, saving
16.5 seconds of cumulative retrieval latency across 150 cache hits.
1. Introduction
The emergence of voice-enabled AI agents has created a new class of real-time systems where
latency is a first-class design constraint. Unlike text-based chatbots where users tolerate multi-
second response times, voice conversations require sub-200ms response latency to feel natural [6].
This budget must accommodate the entire pipeline: speech-to-text (STT), context retrieval, LLM
generation, and text-to-speech (TTS).
Retrieval-Augmented Generation (RAG) [12] has become the standard approach for grounding
LLM responses in domain-specific knowledge. However, the retrieval step introduces a critical
latency bottleneck in voice pipelines. A production vector database query (e.g., to Qdrant [16],arXiv:2603.02206v2  [cs.SD]  3 Mar 2026

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
Pinecone, or Weaviate) typically incurs 50–300ms of network round-trip latency, which alone can
consume the entire latency budget.
Existing approaches to this problem fall into three categories, none of which fully address the
voice latency constraint:
•Faster retrieval infrastructure.Optimized vector indices (FAISS [10], HNSW [14]) reduce
search time but do not eliminate network latency when the database is hosted remotely, which
is the common production deployment pattern.
•Semantic caching.Systems like GPTCache [21], QVCache [8], and the Semantic Lookaside
Buffer [2] cache previous query results, but they are reactive: they only help when theexact
sameor very similar query recurs.
•Speculative retrieval.Stream RAG [1] and Speculative RAG [17] predict what to retrieve
during generation, but they operate within a single query’s lifecycle rather than across
conversation turns.
We propose VOIC EAGENTRAG, a dual-agent architecture inspired by Kahneman’s System
1/System 2 framework [11] and recent work on dual-process AI agents [5, 19]. The key insight is
that in multi-turn voice conversations,the next question is often predictable from the current conversation
context. A customer asking about pricing will likely follow up about the enterprise plan, discounts,
or billing—topics that can be pre-fetchedduring the current responsebefore the user even speaks.
VOIC EAG ENTRAG consists of two concurrent agents:
1.TheSlow Thinkerruns as a background async task. On each user utterance, it uses an LLM
to predict 3–5 likely follow-up topics, retrieves relevant document chunks from the vector
database, and pre-populates a FAISS-backed semantic cache. This work happens in the
background while the user is listening to the current response.
2.TheFast Talkerhandles user queries in the foreground. It first checks the semantic cache
(sub-millisecond lookup), and only falls back to the vector database on cache miss. When it
does fall back, it caches the retrieved results for future queries.
Our contributions are:
•An open-source dual-agent memory router for voice agents that decouples retrieval from
response generation.
•A document-embedding-indexed semantic cache that achieves sub-millisecond lookup with
correct semantic matching.
•A comprehensive 200-query evaluation across 10 conversation scenarios on a production
Qdrant Cloud deployment, demonstrating 75% cache hit rate and 316×retrieval speedup.
•Analysis of cache behavior across conversation depth, topic patterns, and cold/warm start
conditions.
2

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
2. Method
2.1. Architecture Overview
VOIC EAG ENTRAG is organized around five core components, shown in Figure 1: (1) aMemory
Routerthat orchestrates the system, (2) aConversation Streamthat provides an async event bus,
(3) aSlow Thinkerbackground agent, (4) aFast Talkerforeground agent, and (5) aSemantic Cache
backed by an in-memory FAISS index.
User Speech(Audio Stream)
Slow Thinker
(Background Agent)
Listens to stream
Predicts follow-up topics
Retrieves from vector DB
Pre-fills cacheFast Talker
(Foreground Agent)
Checks cache first (0.35ms)
Generates LLM response
Fallback: vector DB (110ms)
Caches results on miss
Semantic Cache| In-memory FAISS Index |<1ms lookup
Vector Store| FAISS / Qdrant Cloud | 97–307ms per queryVoice
ResponseMemory Router event bus query
prefetch cache lookup
retrieves fallback (miss)
Figure 1|Architecture of VOI CEAGENTRAG. TheSlow Thinker(left) continuously monitors the
conversation stream, predicts follow-up topics, retrieves from the vector store, and populates the
semantic cache. TheFast Talker(right) checks the cache first ( <1ms), bypassing the vector store on
hits. On misses, it falls back to direct retrieval (dashed red) and caches the results for future queries.
The data flow proceeds as follows. When a user speaks, the Memory Router publishes a
UserUtteranceevent to the Conversation Stream. Both agents receive this event concurrently:
•TheFast Talkerimmediately embeds the query, checks the semantic cache, and either serves
cached context (hit) or falls back to the vector store (miss). It then generates a streaming LLM
response.
•TheSlow Thinker(running as a background asyncio task) processes the same utterance: it
3

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
retrieves context for the current query, predicts follow-up topics via the LLM, and pre-fetches
document chunks for those predictions into the cache.
Because theSlow Thinkeroperates asynchronously, its prefetching overlaps with the current
turn’s response generation and the user’s listening/thinking time. By the time the user asks their
next question, the relevant context is already in the cache.
2.2. Semantic Cache
The semantic cache is the critical bridge between the two agents. It is implemented as an in-memory
FAISS IndexFlatIP (inner product on L2-normalized vectors, equivalent to cosine similarity) that
stores document chunks indexed by theirownembeddings.
Why document embeddings?An early design indexed cached entries by theprediction query’s
embedding. This produced cache “hits” that returned irrelevant chunks, where the prediction “What
are the pricing details?” would cache pricing chunks, but when the user asked “Is there a discount
for annual billing?”, the cache would return general pricing info instead of the discount-specific
content. By indexing with document embeddings, the cache performs a proper semantic search
over the cached document chunks, returning the most relevant ones for the actual user query.
The cache supports the following operations:
•Put: Store a document chunk with its embedding, relevance score, metadata, and TTL.
Near-duplicates (cosine similarity>0.95) are detected and updated rather than added.
•Get: Given a query embedding, return the top- kcached chunks with cosine similarity above
a thresholdτ. This operation takes<1ms on a cache of 100+ entries.
•Eviction: Entries expire based on TTL (default: 300s). When the cache reaches maximum size,
the least recently accessed entry is evicted (LRU policy).
Threshold tuning.The similarity threshold τis a critical parameter. We found that query-to-
document cosine similarity (using OpenAI text-embedding-3-small ) typically ranges from 0.30 to
0.55, significantly lower than query-to-query similarity. Through empirical analysis (Section 4.5),
we set τ= 0.40 as the default, which balances precision (avoiding irrelevant context) with recall
(capturing useful matches).
2.3. Slow Thinker: Predictive Prefetching
TheSlow Thinkersubscribes to the Conversation Stream and processes events as follows:
OnUserUtterance:
1.Direct retrieval: Embed the current utterance and retrieve the top-kdocument chunks from
the vector store. Cache each chunk with its own embedding.
2.Prediction: Use the LLM to predict the nmost likely follow-up topics based on the conver-
sation history (sliding window of last 6 turns). The prompt instructs the LLM to generate
document-style descriptions rather than questions, producing embeddings closer to actual
document content.
4

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
3.Prefetch: For each prediction, embed it, search the vector store, and cache the results. These
operations run in parallel usingasyncio.gather.
OnPriorityRetrieval (triggered byFast Talkercache miss): Perform an immediate retrieval
with expanded top-k(2×default) to quickly populate the cache around the missed topic.
Rate limiting: To avoid overwhelming the vector store and embedding API, theSlow Thinker
enforces a minimum interval between predictions (default: 0.5s).
2.4. Fast Talker: Cache-First Response
TheFast Talkerhandles the critical latency path. For each user query:
1.Embedthe query using the embedding API (∼200ms for remote APIs).
2.Cache lookup: Search the semantic cache for the top-kchunks with similarity≥τ(<1ms).
3.If cache hit: Format the cached chunks as context and generate the LLM response.Skip the
vector store entirely.
4.If cache miss: Fall back to direct vector store search, generate the response, andcache the
retrieved resultsso that the next similar query hits the cache. Also signal theSlow Thinkervia a
PriorityRetrievalevent.
The cache-on-miss behavior is crucial: it means that the first query on any topic primes the
cache, and all subsequent queries on that topic can be served from cache. Combined with the
Slow Thinker’s predictive prefetching, this creates a compounding effect where the cache warms up
rapidly over the first few turns.
2.5. Conversation Stream
The Conversation Stream is an async event bus implemented with asyncio.Queue . It supports five
event types: UserUtterance ,AgentResponse ,SilenceDetected ,TopicShift , and PriorityRetrieval .
A sliding window maintains the last Nconversation turns (default: 10) for theSlow Thinker’s pre-
diction context.
3. Experimental Setup
3.1. Knowledge Base
We construct a synthetic enterprise knowledge base for “NovaCRM,” a fictional AI-powered
CRM platform. The knowledge base consists of 12 documents covering: company overview,
product features, pricing plans, API documentation (overview, contacts endpoint, deals endpoint),
integrations, security & compliance, onboarding guide, troubleshooting guide, FAQ, and release
notes. After chunking with a recursive character splitter (chunk size: 512, overlap: 50), the
knowledge base contains76 document chunks.
We chose a synthetic KB over a real-world dataset for reproducibility and controlled evaluation.
The content is designed to have natural topical clusters (pricing, API, security, etc.) that mirror real
enterprise documentation.
5

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
3.2. Vector Store
We useQdrant Cloudas the production vector database, deployed on AWS us-west-2. This
provides real network round-trip latency for retrieval operations, as opposed to simulated delays.
The measured search latency ranges from 97–307ms per query (mean: 120ms), representative of
production deployments.
For embeddings, we use OpenAI text-embedding-3-small (1536 dimensions) via the Salesforce
Research Gateway.
3.3. LLM
All LLM operations (response generation, topic prediction, keyword extraction) useGPT-4o-mini.
Temperature is set to 0.3 for response generation and predictions.
3.4. Conversation Scenarios
We design10 diverse conversation scenarios, each with20 turns, for a total of200 queries. The
scenarios cover different conversation patterns observed in real customer support calls:
1.New customer overview: Broad exploration across many topics.
2.Pricing deep-dive: 20 turns focused entirely on pricing.
3.API integration: Technical deep-dive into API documentation.
4.Security & compliance: Enterprise security questions.
5.Onboarding walkthrough: Step-by-step setup guidance.
6.Troubleshooting session: Problem-solving with error scenarios.
7.Integration questions: Feature catalog browsing.
8.Feature comparison: Broad feature tour across product areas.
9.Existing customer upgrade: Plan comparison for upgrade.
10.Mixed rapid-fire: Random questions with short pauses.
Inter-turn delays range from 3–7 seconds, simulating the natural pace of human speech and
thinking time. This gives theSlow Thinker3–7 seconds per turn for background prefetching.
3.5. Evaluation Protocol
For each scenario, we run the conversation twice:
•Traditional RAG (baseline): Each query goes through the full pipeline: embed →Qdrant
search→LLM generate.
•VOI CEAG ENTRAG (dual-agent): TheSlow Thinkerruns in background; theFast Talker
checks the cache first.
We measure: (1) cache hit rate, (2) retrieval latency (vector store search vs. cache lookup), (3)
cache size growth, and (4) total response time. We report retrieval latency as the primary metric
because it isolates the component that the dual-agent architecture optimizes, while total response
time is dominated by LLM generation variance (500–8000ms).
6

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
4. Results
4.1. Overall Performance
Table 1 summarizes the results across all 200 queries. The dual-agent architecture achieves a75%
overall cache hit rate(150/200 queries) and79% on warm-cache turns(150/190, excluding cold
starts), with a316 ×retrieval speedupon cache-hit queries (110ms →0.35ms). Across the 150 cache
hits, a total of16.5 secondsof retrieval latency was saved.
Table 1 |Overall performance across 200 queries on 10 conversation scenarios using Qdrant Cloud.
All measurements are from real network round-trips, with no simulated delays.
Metric Traditional RAG VO I CEAG ENTRAG
Retrieval latency (avg) 110.4 ms 0.35 ms
Retrieval speedup 1×316×
Cache hit rate (overall) — 75% (150/200)
Cache hit rate (warm, turn≥2) — 79% (150/190)
Saved per cache-hit query — 110.1 ms
Total retrieval time saved — 16,508 ms
4.2. Per-Scenario Breakdown
Table 2 shows the cache hit rate for each conversation scenario. Single-topic deep-dives (API
integration, security) achieve the highest hit rates (80–100%), while broad exploration scenarios
(new customer overview, mixed rapid-fire) achieve lower rates (40–60%). This is expected: theSlow
Thinker’s predictions are more accurate when the conversation stays within a topical cluster.
Table 2 |Cache hit rate by conversation scenario (20 turns each), sorted by hit rate. Scenarios with
sustained topics achieve higher hit rates than those with frequent topic shifts. Avg. retrieval shows
Qdrant Cloud search latency.
Scenario Hit Rate Avg. Retrieval Pattern
S8: Feature comparison 95% (19/20) 118.4 ms Broad tour
S2: Pricing deep-dive 85% (17/20) 107.8 ms Single topic
S3: API integration 85% (17/20) 108.7 ms Single topic
S5: Onboarding walkthrough 85% (17/20) 119.2 ms Sequential
S7: Integration questions 85% (17/20) 108.0 ms Related topics
S6: Troubleshooting 80% (16/20) 113.6 ms Problem-solving
S4: Security & compliance 70% (14/20) 105.7 ms Single topic
S1: New customer overview 65% (13/20) 103.5 ms Exploration
S10: Mixed rapid-fire 55% (11/20) 98.2 ms Random
S9: Existing customer upgrade 45% (9/20) 97.2 ms Comparison
7

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
4.3. Cache Warm-Up Dynamics
Table 3 shows how the cache hit rate evolves with conversation depth. Turn 1 always misses (cold
start). The hit rate rapidly increases to 70–80% by turns 5–10 as the combined effects of theSlow
Thinker’s predictive prefetching and theFast Talker’s cache-on-miss behavior compound.
Table 3 |Cache hit rate by conversation depth (aggregated across all 10 scenarios, 200 queries total).
Turn 1 always misses (cold start). The cache reaches 86% hit rate by turns 5–9.
T urn Range Queries Hit Rate Note
Turns 1–4 50 58% (29/50) Cold start + warming
Turns 5–9 50 86% (43/50) Warm cache
Turns 10–14 50 78% (39/50) Stable
Turns 15–19 50 82% (41/50) Saturated
4.4. Retrieval Latency Analysis
On cache-hit turns, the retrieval path is:
•Traditional RAG: Qdrant Cloud search =110.4msaverage (range: 97–307ms).
•VO ICEAG ENTRAG: FAISS in-memory cache lookup =0.35msaverage.
This represents a316 ×speedupon the retrieval step. Across 150 cache-hit queries, a total of
16.5 secondsof retrieval latency was eliminated. In a voice pipeline where the total budget is 200ms,
saving 110ms of retrieval latency per query is the difference between a natural and an unnatural
conversation.
On cache-miss turns, both systems incur the same Qdrant latency. However, theFast Talker’s
cache-on-miss behavior ensures that subsequent queries on the same topic benefit from the miss.
4.5. Threshold Sensitivity Analysis
The cache similarity threshold τcontrols the trade-off between cache precision and hit rate. Table 4
shows the impact of different thresholds on a 20-turn voice call simulation.
The insight from our threshold analysis is that query-to-document cosine similarity is systemat-
ically lower than query-to-query similarity. This is because document chunks are descriptive prose
while queries are short questions. A threshold calibrated for query-query matching (0.55+) is too
strict for query-document matching.
5. Discussion
5.1. When Does the Architecture Help?
The dual-agent architecture provides the most value when:
8

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
Table 4|Impact of cache similarity thresholdτon hit rate. Query-to-document cosine similarity
with OpenAItext-embedding-3-smalltypically ranges 0.30–0.55. A threshold of 0.40 balances
precision and recall.
ThresholdτHit Rate Note
0.55 15% Too strict, most queries score 0.30–0.55
0.50 30% Captures some matches
0.45 55% Good precision, moderate recall
0.40 75% Best balance (default)
0.35 82% Higher recall, risk of irrelevant context
0.30 90% High recall but low precision
•The vector database is remote: The 112ms Qdrant Cloud latency savings are significant. For
local FAISS with 76 documents, search takes 0.1ms—negligible.
•Conversations are topically coherent: Single-topic deep-dives achieve 80%+ hit rates. Ran-
dom rapid-fire questions achieve only 40%.
•There is sufficient inter-turn time: TheSlow Thinkerneeds 3–7 seconds between turns to
complete its prediction-retrieval-cache cycle. Rapid back-and-forth ( <2s) reduces the benefit.
5.2. Cold Start Mitigation
The first query always misses because the cache is empty. However, theFast Talker’s cache-on-miss
behavior immediately primes the cache, and theSlow Thinkerbegins predictive prefetching. By
turn 3–4, the cache is typically warm enough to serve most queries.
For applications where cold start latency is critical, the cache could be pre-warmed at conversa-
tion start by prefetching the most commonly needed document chunks (e.g., the top- kchunks by
retrieval frequency from historical conversations).
5.3. Limitations
LLM generation dominates total latency.Even with the retrieval speedup, total end-to-end
response time is dominated by LLM generation (500–8000ms with GPT-4o-mini via API). The
112ms retrieval savings, while significant for the voice latency budget, are invisible in total response
time measurements due to LLM variance. Using faster inference (e.g., Groq at 840+ tokens/second)
or local models would make the retrieval savings the dominant factor.
Prediction accuracy.TheSlow Thinker’s predictions are not always correct, leading to wasted
prefetch operations. With 5 predictions per turn and 10 chunks per prediction, theSlow Thinker
consumes significant embedding API bandwidth. Lighter prediction strategies (keyword extraction
instead of LLM prediction) could reduce this cost.
Cache coherence.If the underlying knowledge base changes during a conversation, cached en-
tries may become stale. The TTL-based expiration provides eventual consistency but not immediate
coherence.
9

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
5.4. Comparison with Existing Systems
No existing open-source system implements the complete dual-agent predictive prefetching archi-
tecture for voice agents. Table 5 compares VOI CEAGENTRAG with related approaches.
Table 5 |Comparison of VOIC EAG ENTRAG with related approaches. VOIC EAG ENTRAG is the
only system combining proactive (predictive) caching with a dedicated background agent for
voice-optimized retrieval.
System Proactive Predictive Voice Bkgd. Open
Traditional RAG [12]✗ ✗ ✗ ✗—
GPTCache [21]✗ ✗ ✗ ✗ ✓
Stream RAG [1]✓ ✓ ✓ ✗ ✗
MemGPT/Letta [15]✗ ✗ ✗ ✗ ✓
DPT-Agent [19]✓ ✗ ✗ ✓ ✗
LiveKit+RAG [13]✗ ✗ ✓ ✗ ✓
VOIC EAG ENTRAG (ours)✓ ✓ ✓ ✓ ✓
6. Related Work
6.1. RAG and Retrieval Latency
RAG [12] augments LLM generation with retrieved context, improving factual accuracy for
knowledge-intensive tasks. However, the standard RAG pipeline (embed query →search vector
DB→generate response) is inherently sequential, making each step a latency bottleneck.
Several works address retrieval efficiency. FLARE [9] uses forward-looking prediction to decide
whento retrieve, reducing unnecessary retrievals. Self-RAG [3] trains the model to adaptively
decide whether retrieval is needed. Speculative RAG [17] uses a smaller model to draft multiple
RAG responses in parallel, with a larger model verifying them. RaLMSpec [20] applies speculative
execution to retrieval, prefetching likely needed documents. Stream RAG [1] predicts tool queries
in parallel with user speech, achieving 20% reduction in tool-use latency.
Our work differs from these in two ways: (1) we operateacrossconversation turns rather than
within a single query, leveraging the natural pause between turns for background prefetching; and
(2) we target specifically the voice agent use case where the latency budget is measured in tens of
milliseconds.
6.2. Semantic Caching for LLMs
Semantic caching stores previous results indexed by query meaning rather than exact match.
GPTCache [21] provides an open-source semantic cache for LLM responses using embedding-
based similarity. QVCache [8] introduces query-aware caching for approximate nearest neighbor
search, achieving 40–1000 ×latency reduction. The Semantic Lookaside Buffer (Aeon) [2] exploits
conversational locality to achieve sub-5 microsecond retrieval.
Our semantic cache differs from these systems in that it isproactively populatedby theSlow
Thinkerrather than passively filled from previous queries. We also index cached entries bydocument
10

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
embeddingsrather than query embeddings, ensuring that the cache returns the most relevant
document chunks for any query, regardless of which prediction originally fetched them.
6.3. Dual-Process Architectures for AI Agents
The System 1/System 2 distinction from cognitive science [11] has inspired several AI agent archi-
tectures. DPT-Agent [19] uses a finite-state machine for fast System 1 decisions and asynchronous
reflection for System 2 reasoning in real-time human-AI collaboration. Cognitive Decision Rout-
ing [5] introduces a meta-cognitive layer that routes queries to fast or slow processing paths,
reducing computational costs by 34%.
Hybrid-RACA [18] is the closest to our approach: it uses a client-side model for immediate
responses while a cloud model enriches memory asynchronously. However, it targets text prediction
rather than voice agents and does not implement predictive prefetching of knowledge base content.
6.4. Voice Agent Frameworks
Several open-source frameworks support building voice AI agents. LiveKit Agents [13] provides
WebRTC-based real-time voice pipelines with RAG examples. Pipecat [4] offers composable
streaming pipelines for voice AI. Ultravox [7] eliminates the STT step entirely with a speech-native
multimodal LLM. Commercial platforms like Vapi and Retell provide managed voice-RAG services.
None of these frameworks implement background predictive prefetching. They all perform
retrievalsynchronouslywhen a query arrives, making them subject to the full vector database latency
on every turn.
7. Conclusion
We presented VOI CEAGENTRAG, a dual-agent memory router that solves the RAG latency
bottleneck in real-time voice agents. By decoupling retrieval into a backgroundSlow Thinkerthat
proactively pre-fetches context and a foregroundFast Talkerthat reads from a sub-millisecond
semantic cache, VOIC EAG ENTRAG achieves a 75% overall cache hit rate (79% on warm turns)
with 316 ×retrieval speedup compared to direct vector database search, saving over 16 seconds of
cumulative retrieval latency across 200 queries.
Our evaluation on 200 queries across 10 diverse conversation scenarios demonstrates that the
architecture is most effective for topically coherent conversations (80%+ hit rate on deep-dive
scenarios) and scales with retrieval latency: the higher the vector database latency, the more value
the cache provides.
The system is released as an open-source codebase supporting multiple LLM providers (OpenAI,
Anthropic, Gemini, Ollama), vector stores (FAISS, Qdrant), and voice integration (Whisper STT,
Edge TTS). Code and benchmarks are available at https://github.com/SalesforceAIResearch/Vo
iceAgentRAG.
11

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
References
[1] Siddhant Arora et al. “Stream RAG: Instant and Accurate Spoken Dialogue Systems with
Streaming Tool Usage”. In:arXiv preprint arXiv:2510.02044(2025).
[2] Mustafa Arslan. “Aeon: High-Performance Neuro-Symbolic Memory Management for Long-
Horizon LLM Agents”. In:arXiv preprint arXiv:2601.15311(2025).
[3] Akari Asai et al. “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-
Reflection”. In:arXiv preprint arXiv:2310.11511(2023).
[4] Daily.co.Pipecat: Open-Source Framework for Voice and Multimodal Conversational AI. 2024.
[5] Y Du et al. “Cognitive Decision Routing in Large Language Models: When to Think Fast,
When to Think Slow”. In:arXiv preprint arXiv:2508.16636(2025).
[6] Vignesh Ethiraj et al. “Toward Low-Latency End-to-End Voice Agents for Telecommuni-
cations Using Streaming ASR, Quantized LLMs, and Real-Time TTS”. In:arXiv preprint
arXiv:2508.04721(2025).
[7] Fixie AI.Ultravox: Open-Source Speech-Native Multimodal LLM. 2024.
[8] Anıl Eren Göcer et al. “QVCache: A Query-Aware Vector Cache”. In:arXiv preprint arXiv:2602.02057
(2025).
[9] Zhengbao Jiang et al. “Active Retrieval Augmented Generation”. In:Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. 2023, pp. 7969–7992.
[10] Jeff Johnson, Matthijs Douze, and Hervé Jégou. “Billion-Scale Similarity Search with GPUs”.
In:IEEE Transactions on Big Data7.3 (2021), pp. 535–547.
[11] Daniel Kahneman.Thinking, Fast and Slow. Farrar, Straus and Giroux, 2011.
[12] Patrick Lewis et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”.
In:Advances in Neural Information Processing Systems. Vol. 33. 2020, pp. 9459–9474.
[13] LiveKit Inc.LiveKit Agents: Open-Source Framework for Building Real-Time Voice AI Agents. 2024.
[14] Yury Malkov and Dmitry A. Yashunin. “Efficient and Robust Approximate Nearest Neighbor
Search Using Hierarchical Navigable Small World Graphs”. In:IEEE Transactions on Pattern
Analysis and Machine Intelligence42 (2016), pp. 824–836.
[15] Charles Packer et al. “MemGPT: Towards LLMs as Operating Systems”. In:arXiv preprint
arXiv:2310.08560(2023).
[16] Qdrant.Qdrant: High-Performance Vector Search Engine. 2024.
[17] Zilong Wang et al. “Speculative RAG: Enhancing Retrieval Augmented Generation through
Drafting”. In:arXiv preprint arXiv:2407.08223(2024).
[18] Menglin Xia et al. “Hybrid-RACA: Hybrid Retrieval-Augmented Composition Assistance for
Real-time Text Prediction”. In:arXiv preprint arXiv:2308.04215(2023).
[19] Shao Zhang et al. “Leveraging Dual Process Theory in Language Agent Framework for
Real-time Simultaneous Human-AI Collaboration”. In:Proceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics. 2025.
[20] Zhihao Zhang et al. “Accelerating Retrieval-Augmented Language Model Serving with
Speculation”. In:arXiv preprint arXiv:2401.14021(2024).
12

VOI CEAG ENTRAG : Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent
Architectures
[21] Zilliz.GPTCache: An Open-Source Semantic Cache for LLM Applications. 2023.
A. System Configuration
Table 6 lists the default configuration parameters used in all experiments.
Table 6|Default configuration parameters for VOIC EAG ENTRAG.
Component Parameter Default
Cache Max size 2000
Cache TTL 300s
Cache Similarity thresholdτ0.40
Cache Dedup threshold 0.95
Slow ThinkerPrediction strategy LLM
Slow ThinkerMax predictions per turn 5
Slow ThinkerPrefetch top-k10
Slow ThinkerRate limit 0.5s
Fast TalkerMax context chunks 10
Fast TalkerFallback to retrieval Yes
Fast TalkerCache on miss Yes
Embedding Model text-embedding-3-small
Embedding Dimension 1536
LLM Model GPT-4o-mini
LLM Temperature 0.3
Document Chunk size 512
Document Chunk overlap 50
13