# Talk with the Things: Integrating LLMs into IoT Networks

**Authors**: Alakesh Kalita

**Published**: 2025-07-23 18:39:59

**PDF URL**: [http://arxiv.org/pdf/2507.17865v1](http://arxiv.org/pdf/2507.17865v1)

## Abstract
The convergence of Large Language Models (LLMs) and Internet of Things (IoT)
networks open new opportunities for building intelligent, responsive, and
user-friendly systems. This work presents an edge-centric framework that
integrates LLMs into IoT architectures to enable natural language-based
control, context-aware decision-making, and enhanced automation. The proposed
modular and lightweight Retrieval Augmented Generation (RAG)-based LLMs are
deployed on edge computing devices connected to IoT gateways, enabling local
processing of user commands and sensor data for reduced latency, improved
privacy, and enhanced inference quality. We validate the framework through a
smart home prototype using LLaMA 3 and Gemma 2B models for controlling smart
devices. Experimental results highlight the trade-offs between model accuracy
and inference time with respect to models size. At last, we also discuss the
potential applications that can use LLM-based IoT systems, and a few key
challenges associated with such systems.

## Full Text


<!-- PDF content starts -->

1
Talk with the Things: Integrating LLMs into IoT
Networks
Alakesh Kalita, Senior Member, IEEE,
Abstract —The convergence of Large Language Models (LLMs)
and Internet of Things (IoT) networks opens new opportuni-
ties for building intelligent, responsive, and user-friendly sys-
tems. This work presents an edge-centric framework that inte-
grates LLMs into IoT architectures to enable natural language-
based control, context-aware decision-making, and enhanced
automation. The proposed modular and lightweight Retrieval-
Augmented Generation (RAG)-based LLMs are deployed on
edge computing devices connected to IoT gateways, enabling
local processing of user commands and sensor data for reduced
latency, improved privacy, and enhanced inference quality. We
validate the framework through a smart home prototype using
LLaMA 3 and Gemma 2B models for controlling smart devices.
Experimental results highlight the trade-offs between model
accuracy and inference time with respect to model size. At last,
we also discuss the potential applications that can use LLM-
based IoT systems, and a few key challenges associated with
such systems.
Index Terms —Large Language Models (LLMs), Internet of
Things (IoT), MQTT, Edge Computing
I. I NTRODUCTION
With the development of digitization, IoT ( Internet
of Things ) technology has become a prominent topic
in the field of networking and communication. IoT aims to
connect billions of physical devices, which can be both living
and non-living entities found on the earth’s surface, with
sensing/actuation and communication capacities. By enabling
seamless connectivity and data exchange, IoT is transforming
how we interact with the world around us [1]. On the other
hand, new deep generative-based Large Language Models
(LLMs) are designed to understand and generate human lan-
guage. Recent advances in generative AI and transformer-
based LLMs have demonstrated remarkable capabilities in
understanding natural language and reasoning. LLMs, such as
GPT-4, GPT-4o, and Llama 3.1, are trained on vast amounts of
text data, enabling them to perform a wide range of language
tasks, including translation, summarization, and conversation.
Notably, traditional IoT networks only work commands and
actions basis in the environment like smart home, industry
etc. However, the integration of LLMs in IoT networks can
make a huge transformation towards more intelligent, efficient,
and responsive IoT systems [2], [3], [4]. This convergence
leverages the natural language understanding capabilities of
LLMs in IoT, allowing devices to communicate in a more
meaningful and context-aware manner [5]. For example, if a
Alakesh Kalita is with the Department of Mathematics and Computing,
Indian Institute of Technology (ISM) Dhanbad, India.user says “ Set up for movie night ”, LLMs can help
toset the Dim lights in the living room ,close the blinds, adjust
the thermostat to a comfortable temperature, turn on the TV
and navigate to the user’s preferred streaming service . All
can be done with a single command, which requires multiple
individual commands for each actuation in the current IoT
systems. In brief, IoT can serve as the “ sensing limbs ” of
an environment, feeding real-time sensor inputs (e.g., temper-
ature, motion, vibrations) to LLMs, while LLMs act as the
“brain ” that analyzes data, draws conclusions, and communi-
cates or acts upon them. Thus, LLM-based communication in
IoT can significantly improve efficiency, bandwidth utilization,
and user interactions.
However, the devices used in IoT networks are resource-
constrained in terms of processing capacity, memory, and
power. Therefore, using LLM in this kind of device is not
practically trivial. Similarly, technologies like cloud computing
have their disadvantages, like higher latency and cost. There-
fore, edge computing can be a suitable approach to use LLM
in IoT applications such as smart homes, smart industries, etc.
Edge computing processes data closer to the source of the data
generators rather than relying on centralized data centres at
higher propagation delay [6]. Edge computing reduces latency,
conserves bandwidth, and enhances real-time data processing
capabilities.
Therefore, in this article, we briefly discuss how LLMs
can be used to establish efficient communication in edge
computing-based IoT systems. In the next section, we present
our proposed framework for integrating LLMs into edge-based
IoT systems. We then describe our experimental setup in a
smart home environment and discuss the results obtained.
Finally, before concluding, we highlight the advantages of
using LLMs in IoT systems, explore additional use cases
beyond smart homes, and outline key challenges that must
be addressed to make such systems more efficient and robust.
II. S YSTEM ARCHITECTURE : HOWLLM SWORK IN AN
IOT N ETWORK
IoT devices are often resource-constrained regarding pro-
cessing capacity, memory, and power. Consequently, running
LLMs directly on these devices is not technically feasible.
To address this challenge, we propose a framework where an
IoT network’s border router (BR) or gateway is connected to
an Edge computing device equipped with sufficient resources
to infer user requests using LLMs. Therefore, in our proposed
framework, LLMs are deployed at the edge of the network, and
the edge device is connected to the gateway or BR of the IoT
network. IoT devices attached to sensors collect informationarXiv:2507.17865v1  [cs.NI]  23 Jul 2025

2
IoT Devices
Gateway
Edge
Computing
Device
Bluetooth 6TiSCH WiFi HaLow
Ethernet WiFi 5G
IoT Data 
Collection
ModuleData 
Processing
ModulePrompt
Creation
ModuleLLM
ModuleLLM Response
Handling
ModuleIoT Actuator
Handling
Module
Command : Set a movie night
"temperature ": 15,
"Kitchen ": "Not occupied"
"Day": "Sunday" ,
"time": "3PM"Prompt  =  Actions needed to set up
a comfortable and enjoyable movie
night when t he room temperature
is {data[ 'temperature ']}, on
{data[ 'Day']} ....etc.,action 1  = "energy_saving_mode"
action 2  = "Prepare living room"
etc.,Switch  OFF kitchen's lights
Switch ON living room AC
Data
StorageHistorical
DataIoT devices using Bluetooth, 6TiSCH, or Wi-Fi HaLow connect wirelessly to a central gateway . These
devices send updates from sensors (like temperature or motion) to the gateway , and receive control
commands (like turning on a light or fan) from it. This communication happens through a simple and
efficient messaging system called MQTT .
EdgePublisher
Payload:  {"power":"on", 
           "volume": 20 }Topic: home/tv1/status SubscriberTopic: home/tv1/command Publisher
Subscriber
The gateway connects to an edge device using Wi-Fi or Ethernet, or it can act as the edge device
itself if powerful enough. This edge device collects data from sensors, creates a clear prompt, uses
an LLM to process it, and sends smart commands back to IoT  devices through the gateway .
Fig. 1. Proposed framework is divided into multiple structured modules that handle data collection, processing, prompt creation, response handling, and
actuator management.
from the environment and transmit their data to the gateway
using multi-hop (mesh topology) or single-hop (star topology)
communication networks. The gateway then forwards the data
to the edge computing device for processing by the LLM
deployed at the edge. Note that some edge devices can also
serve as IoT gateways. Apart from the input provided by the
IoT devices, users can also provide their input directly to the
LLMs in the form of text, video, or voice commands. For
example, in smart home environments, users can send voice
commands to perform different activities. The transmission of
data from the IoT devices to the gateway or edge follows
traditional source and channel encoding.
To simplify our proposed LLM in IoT network framework
to facilitate efficient and intelligent interactions within IoT
ecosystems, we divide it into multiple structured modules that
handle data collection, processing, prompt creation, response
handling, and actuator management. These modules are shown
in Figure 1 and discussed as follows,
IoT Data Collection Module : This module is responsible
for gathering data from various IoT devices, including the
commands received from the users. This module ensuresreal-time data collection from IoT devices and users. For
this, light-weight application layer protocol MQTT (Message
Queuing Telemetry Transport) can be used. For example,
in a smart home environment, both the Smart TV and the
edge device act as publishers and subscribers. The Smart
TV publishes its status updates (e.g., power state, volume)
to a topic like home/tv1/status , which the edge device
subscribes to. Conversely, the edge device sends control com-
mands (e.g., turn on/off ) by publishing to a topic such
ashome/tv1/command , which the Smart TV subscribes to.
This bidirectional communication allows seamless monitoring
and control using the lightweight MQTT protocol. The broker
for the MQTT protocol can be deployed in the edge or gateway
of the IoT network.
Data Processing Module : Once the data is collected, it is
processed to extract relevant information and convert it into
a suitable format for further analysis. This module applies
filtering, aggregation, and transformation techniques to ensure
the data is ready for the next stages.
Storage Module : This module is responsible for main-
taining historical data, which can be fed to the LLM for

3
referencing past information, providing context, and improving
the experience of the end users. In brief, in addition to sensory
input and user input, the LLM model is fed with historical data
for better inference. For instance, when a user issues a voice
command to set up a movie night, our proposed framework
can consider previous historical data from the user to adjust
the temperature.
Strictured Prompt Creation Module : To enable context-
aware reasoning, we incorporate a lightweight Retrieval-
Augmented Generation (RAG) mechanism within our sys-
tem. This module dynamically constructs prompts by re-
trieving relevant information from two sources: (i) real-time
sensor data collected from the connected IoT devices, and (ii)
historical context and domain-specific knowledge stored in the
Storage Module. By combining live environmental inputs with
past observations or rules, the system ensures that the LLM
operates with an up-to-date and situationally grounded context.
These structured prompts allow the LLM to generate responses
that are not only linguistically coherent but also semantically
aligned with the current state of the IoT environment. As
a result, the LLM’s decisions become more accurate, goal-
oriented, and responsive to real-world conditions.
1structured_prompt = (
2 f"{session[’user_command’]}.\n"
3 f"Only consider these devices: {’, ’.join(
session[’devices’])}.\n"
4 f"Current sensor readings: {json.dumps(session[’
current_sensor_values’])}\n"
5 "First, give a 20-word description. Then respond
ONLY in the following JSON format:\n"
6 "{\n"
7 " \"description\": \"<short description>\",\n"
8 " \"commands\": [\n"
9 " {\"device\": \"<device>\", \"action\": \"<
action>\", \"mode\": \"<mode> (optional)\" }\n"
10 " ]\n"
11 "}"
12)
Listing 1. Structured Prompt Generation
LLM Module : The core component of the framework, the
LLM Module, utilizes advanced language models to interpret
prompts and generate coherent, context-aware responses. This
module leverages the capabilities of LLMs to understand and
process natural language queries effectively. Some of the open-
source LLMs are mentioned in Table I.
1def get_llama_response(prompt):
2 url = "http://localhost:11434/api/generate"
3 payload = {
4 "model": "llama3",
5 "prompt": prompt,
6 "stream": False
7 }
Listing 2. An example of using an open-source LLM
LLM Response Handling Module : After generating the
structured prompt, it is sent to the deployed LLM, such as
LLaMA, for inference. The LLM processes the prompt and
returns a textual response that contains the intended actions to
be taken on the IoT devices. This raw response is then parsed
to extract actionable commands in a predefined JSON like
format. The parsing step is essential to ensure the output is
structured correctly and interpretable by downstream modules.TABLE I
COMPARISON OF OPEN-SOURCE LLM S
Model Params Q4 Size Min RAM
TinyLlama-1.1B 1.1B 0.55–0.7 GB 1.5 GB
StableLM-Zephyr-3B 3B 1.8–2.2 GB 4 GB
Phi-2 2.7B 1.5–1.8 GB 3 GB
Gemma-2B 2B 1.3–1.6 GB 3 GB
LLaMA-2-7B 7B 3.8–4.5 GB 6 GB
Mistral-7B 7B 4.0–4.5 GB 6 GB
GPT-J-6B 6B 3.0–3.5 GB 5 GB
DistilGPT-2 82M 0.3–0.5 GB 1 GB
IoT Actuator Handling Module : The final module in
the framework, the IoT Actuator Handling Module, translates
the processed responses into actionable commands for IoT
devices. It ensures that the system’s outputs are effectively
communicated within the IoT network, enabling intelligent and
automated control.
1def control_device(device, action):
2 if device == "fan":
3 publish_fan(action)
4 elif device == "light":
5 ...
Listing 3. Transmitting commands to the IoT devices using MQTT
Note that these modules can be easily integrated to work
with current IoT infrastructures, enhancing their capabilities
without designing the entire IoT system from scratch. For
example, an LLM can interface with existing smart home
assistants and IoT devices, using APIs and standard commu-
nication protocols to understand and execute user commands.
III. E XPERIMENTAL SETUP AND EVALUATION
To validate the feasibility of deploying LLMs at the edge
of IoT networks for natural-language-based control, we de-
veloped a smart home prototype using a Raspberry Pi 5 (8
GB) as the edge computing device. Three appliances i.e.,
a light, a TV and a fan were connected to the edge via
ESP8266 NodeMCU microcontrollers using 5V mechanical
relay modules. Communication between the Raspberry Pi
and the IoT nodes was established over Wi-Fi using the MQTT
protocol.
Each IoT device is subscribed to a unique MQTT topic for
receiving control commands and publishes its current status to
a dedicated topic. The user issued natural language commands
through a local text-based interface, which were converted into
structured prompts and processed by the LLMs. The resulting
response was parsed into JSON format and translated into
MQTT commands, which were then sent to the appropriate
devices. We tested two different LLMs i.e., LLaMA 3 (7B)
andGemma 2B . Each model was evaluated with three repre-
sentative commands related to smart room setup. For each
command, we recorded the LLM’s response and the total
inference time from prompt submission to command dispatch.
The expected device states were pre-defined for each use case.
Table II summarizes the results. LLaMA 3 exhibited high
semantic accuracy, generating responses that closely matched
the expected device states for each user command. However,
its inference time was significantly higher, ranging from 126

4
TABLE II
COMPARISON OF LLAMA 3 AND GEMMA 2B ONSMART HOME ENVIRONMENT
Model Command Expected Output LLM Response Time (s)
LLaMA 31.Set the room for Study Light: on, Fan: on, TV: off Light: on, Fan: on, TV: off 208
2.Set the room for movie night Light: on, Fan: on, TV: on Light: on, Fan: on, TV: on 204
3.I want to sleep now Light: off, Fan: off, TV: off Light: on, Fan: off, TV: off 126
Gemma 2B1.Set the room for Study Light: on, Fan: on, TV: off Light: on, Fan: on, TV: on 28
2.Set the room for movie night Light: on, Fan: on, TV: on Light: on, Fan: on, TV: off 29
3.I want to sleep now Light: off, Fan: off, TV: off Light: off, Fan: off, TV: off 30
to208 seconds . Note that these delays define the execution
times of the LLMs only. We do not consider the other delays
like propagation and transmission delays of the command,
delay for generating structured prompts, and so on. In contrast,
Gemma 2B demonstrated significantly lower latency ( 28–30
seconds ) across all commands. However, it occasionally mis-
interpreted the intent, such as turning on the TV when it
should remain off or failing to activate it when required. This
suggests that while lightweight models are more suitable for
low-latency edge deployment, they may require fine-tuning or
optimized prompt engineering to achieve comparable accuracy
to larger models. Overall, the results indicate that LLMs can
be effectively integrated into edge-based IoT control systems.
LLaMA 3 ensures high accuracy but requires more compute
and response time, whereas Gemma 2B offers faster, local
inferencing at the cost of occasional semantic deviations. This
trade-off between inference quality and latency is crucial when
choosing models for real-time IoT applications. The output
of one user command “ I want to sleep now ” is shown
below.
1# User Input
2Enter your smart home command: I want to sleep now
3
4# Sending to LLaMA...
5Raw Output from LLaMA:
6Here is your sleep-aiding response:
7Description: Prepare your sleep sanctuary with a dim
light and calming TV settings.
8{
9"description": "Prepare your sleep sanctuary with
a dim light and calming TV settings.",
10 "commands": [
11 {"device": "light", "action": "dim"},
12 {"device": "tv", "action": "off"}
13 {"device": "fan", "action": "off"}
14 ]
15}
16# Post processing of LLM output
17Description: Prepare your sleep sanctuary with a dim
light and calming TV settings.
18
19Commands:
20
21Device: light | Desired: dim | Current: on
22No action needed for light
23
24Device: tv | Desired: off | Current: on
25Turning OFF tv
26
27Device: fan | Desired: off | Current: off
28No action needed for fan
29
30# Final Action
31TV = Turn OffIV. A DVANTAGES OF USING LLM S IN IOT N ETWORKS
Improved Decision Making : LLMs enhance decision-
making in IoT systems by understanding user intent and envi-
ronmental context. This allows the system to infer appropriate
actions even from vague or high-level inputs. For example,
a command like “ prepare the room for sleep ” can
lead to multiple coordinated actions such as turning off lights
and reducing fan speed, all without needing the user to specify
each task individually.
Faster Response Time : By processing natural language
commands directly at the edge or via lightweight prompt-based
inference, LLMs reduce the need for large data exchanges
and multi-step rule evaluation. This streamlined processing
pipeline reduces transmission, computation, and queuing de-
lays, leading to quicker system responses, especially in time-
sensitive IoT applications.
Enhanced User Experience : Natural language interfaces
powered by LLMs allow users to interact with IoT systems
more intuitively and efficiently. Users can issue simple voice
or text commands, which are interpreted accurately by the
system. By incorporating historical user data, the system
can also learn preferences over time, enabling personalized
automation and reducing repetitive configurations.
Scalability and Adaptability : LLMs provide a flexible con-
trol interface that scales with the number of IoT devices and
services without requiring complex rule-based programming.
As new devices are added to the system, the LLM can dynami-
cally adapt to new vocabulary and control patterns, simplifying
deployment in large-scale or evolving environments.
V. U SECASES
This section discusses some of the use cases of how LLMs
can be integrated into Edge-based IoT systems:
Industrial IoT : In next-generation Industrial IoT (IIoT)
environments, including smart manufacturing systems, intel-
ligent energy grids, and autonomous industrial infrastructures,
LLMs are expected to play a pivotal role in enhancing
system intelligence, adaptability, and autonomy. Future IIoT
architectures may employ hierarchical LLM-based frameworks
wherein edge-level agents perform real-time sensor fusion
and anomaly detection, while cloud-based LLMs synthesize
contextual information from historical records, sensor data,
and operational knowledge to generate high-level decisions
for predictive maintenance and system optimization. These
models can be used to enable data-driven forecasting of equip-
ment failures by identifying subtle correlations across mul-
timodal sensor streams, thereby supporting condition-based

5
maintenance and minimizing unplanned downtime. LLMs can
also serve as decision-support agents, analyzing continuous
operational data such as throughput, energy consumption, and
fault indicators to dynamically recommend adjustments that
improve process efficiency, safety, and resource allocation.
Moreover, LLMs are expected to facilitate natural language
interfaces for industrial systems, allowing operators to issue
queries or commands in plain language, which the model
can translate into machine-executable actions. This capability
addresses long-standing challenges in human-machine inter-
action by improving accessibility and reducing interface com-
plexity. With advancements in multimodal reasoning, future
LLMs are envisioned to integrate heterogeneous data sources
including time-series signals, visual inputs, textual logs, and
expert annotations into a unified semantic framework, thereby
enhancing situational awareness and decision accuracy. The
development of domain-specialized, resource-efficient LLMs
tailored for IIoT use cases is expected to significantly improve
the robustness, transparency, and operational intelligence of
industrial systems deployed across resource-constrained and
mission-critical environments.
Smart Healthcare : In future Internet of Medical Things
(IoMT) ecosystems comprising of wearable sensors, remote
monitoring devices, smart diagnostics, and hospital infrastruc-
ture, LLMs can be expected to enable intelligent, patient-
centric healthcare delivery through real-time monitoring, per-
sonalized support, and clinical decision assistance. LLMs
deployed at the edge could continuously analyze different
physiological data (e.g., heart rate, oxygen saturation, glucose
levels) to detect early indicators of medical deterioration and
generate context-aware alerts, while accounting for patient-
reported symptoms and behavioral context. Such systems may
leverage LLMs for improving the sensitivity and specificity
of automated monitoring. Additionally, conversational LLM-
based health assistants can be developed to facilitate natural
interaction with patients and caregivers, enabling voice-based
symptom reporting, personalized medication reminders, and
dynamic health education ultimately improving adherence and
reducing clinical workload. Furthermore, by integrating health
records, lifestyle data, and sensor inputs, LLMs can support
predictive analytics and treatment personalization such as
identifying high-risk patterns in diabetic patients or forecasting
readmission risks in post-operative cases. To mitigate clinician
burden and information overload, LLMs can also assist in data
summarization, transforming continuous IoMT streams into
structured reports that highlight actionable insights.
Semantic Communication : With the increasing deploy-
ment of IoT devices and the emergence of data-intensive
applications, traditional communication systems are facing sig-
nificant challenges related to bandwidth efficiency, latency, and
scalability. As current physical-layer technologies approach
Shannon’s limit, semantic communication, which focuses on
transmitting the meaning rather than the raw data has emerged
as a promising paradigm [7]. By integrating LLMs into IoT
systems, particularly at the network edge, intelligent, context-
aware, and bandwidth-efficient communication can be enabled.
For example, the door security cameras in smart homes,
instead of transmitting a full high-resolution image to a centralserver for processing, the edge device can extract semantic
features such as image edges and natural language descriptions
like “ a fair-colored unknown man wearing a blue t-shirt with
a tense face ”. These compact representations consume signif-
icantly less bandwidth and processing power. Upon reception,
the LLM at the receiver end can reconstruct or interpret the
image using the edge-map and description, potentially even re-
generating a high-fidelity approximation of the original scene
using generative models. This approach drastically reduces
transmission load, minimizes network congestion, and lowers
end-to-end latency [8]. Beyond compression, LLMs enable
context-aware summarization, anomaly detection, and event-
triggered communication, ensuring that only meaningful and
prioritized information is transmitted. When deployed at the
edge, such systems can adapt in real-time, leveraging historical
and situational data to improve decision-making accuracy and
optimize communication schedules. Therefore, in the context
of 5G and emerging 6G technologies, it is expected that
LLM-powered semantic communication holds the potential to
revolutionize IoT networking by shifting the focus from syn-
tactic data accuracy to semantic relevance, thereby enhancing
the efficiency, scalability, and responsiveness of future smart
systems.
VI. O PEN CHALLENGES AND ISSUES
Even though LLMs in Edge-based IoT systems can revolu-
tionize the current IoT system, there are a variety of challenges
to be tackled before they can be used in real applications.
This section discusses the two major open issues for future
investigations.
A. Privacy and Security
The integration of LLMs into IoT systems introduces critical
concerns on privacy. Such systems frequently collect highly
sensitive data ranging from in-home camera footage to per-
sonal health metrics and daily behavioral patterns. This raises
the fundamental question: Where is the data processed, and
who has access to it ? The answer is inherently tied to the
architectural design of the system.
When an LLM is executed locally on an edge device, data
processing remains within the user’s trusted boundary, offering
a high degree of privacy preservation [9]. Conversely, when
raw sensor data is transmitted to a cloud-based LLM or a
third-party API, there is a non-trivial risk of data exposure,
unauthorized storage, or misuse. Despite these risks, cloud-
based approaches offer significant advantages in terms of cost-
efficiency and deployment scalability, as they benefit from
access to extensive datasets and substantial computational
resources. While local open-source LLMs may not match the
scale of cloud models, their lightweight nature enables fast,
private execution and supports a broad range of practical tasks.
Therefore, there are trade offs running the LLMs either on
edge (local) device or cloud. However, a promising alternative
can be a hybrid architectures, where preliminary processing of
the data can be done on the edge device, and only encrypted
summaries or non-sensitive features can be transmitted to the

6
cloud for advanced inference. This approach provides a bal-
anced trade-off between data privacy and system performance,
mitigating the limitations of both extremes.
Security is the complementary challenge in the same con-
text. An LLM deployed at the edge can be prone to different
types of attacks. For example, adversaries might attempt
to compromise system integrity through techniques such as
prompt injection , wherein manipulated inputs could lead the
LLM to disclose sensitive information or behave undesirably.
In extreme cases, if the model retains fragments of user input
across sessions, it may inadvertently act as a leakage channel
for private data. Mitigating these risks requires robust input
filtering mechanism which can detect suspicious input, and
sandboxing mechanisms to restrict the model’s access and
operational domain.
In summary, ensuring user trust in LLM-enabled IoT sys-
tems necessitates rigorous attention to privacy and security.
In environments where IoT devices are continuously sensing,
recording, and interpreting user data, it is imperative that the
intelligence driving them adheres to the same ethical and
operational standards expected of any trusted human assistant.
B. Accuracy and Reliability in LLM-Enabled IoT Systems
In safety-critical domains such as industrial automation,
healthcare, and smart infrastructure, the reliability of LLM-
generated outputs is essential. Unlike general-purpose chat-
bot applications, erroneous outputs in IoT settings can have
physical consequences such as triggering false alarms, missing
critical faults, or causing unsafe actuator behavior. For in-
stance, in a smart home environment, if an LLM misinterprets
smoke sensor data as non-threatening, it may delay emergency
response to an actual fire hazard. Conversely, an overly sensi-
tive model might trigger frequent negative actuation, such as
turning off appliances unnecessarily or repeatedly notifying
medical staff for non-critical fluctuations leading to alert
fatigue and reduced trust in the system.
Therefore, ensuring reliable behavior in different environ-
ments requires a multi-layered approach. Fine-tuning LLMs
on domain-specific data (e.g., annotated sensor logs and fault
descriptions) or layer-wise unified compression reduces hal-
lucinations and enhances context-aware accuracy [10], [11].
In addition, hybrid architectures that pair LLMs with de-
terministic rule-based verifiers can flag or override unsafe
critical decisions. For instance, LLM-generated commands
such as shutting down a compressor should be validated
against physical thresholds before execution.
Furthermore, to properly evaluate the performance of LLM
based IoT systems, we need benchmarks that go beyond just
language quality. These benchmarks should also check how
accurately the models handle real-world tasks, how fast they
respond, and how well they deal with unusual or unexpected
situations. In summary, while LLMs bring powerful reasoning
capabilities to IoT, their deployment in critical applications
must be underpinned by domain adaptation, architectural
safeguards, transparent outputs, and rigorous evaluation. With
these measures, LLMs can become dependable agents capable
of responsible decision-making in real-world IoT systems.VII. C ONCLUSION
In this article, we discussed how LLMs can enhance the
efficiency and intelligence of edge-based IoT systems. We
proposed a modular framework that integrates LLMs into IoT
architectures and outlined the role of each component within
the system, from data collection and prompt generation to
response handling and device control. We also explored the
advantages of using LLMs for natural language-based inter-
action, personalized automation, and context-aware decision-
making in IoT environments. Finally, we highlighted a few
use cases of LLM-based IoT systems, key challenges such as
privacy, security, reliability, and model optimization that must
be addressed to enable safe and effective deployment of LLM-
enabled IoT systems in real-world scenarios.
REFERENCES
[1] A. Kalita and M. Khatua, “6TiSCH – IPv6 Enabled Open Stack IoT
Network Formation: A Review,” ACM Trans. Internet Things , vol. 3,
no. 3, jul 2022.
[2] I. Kok, O. Demirci, and S. Ozdemir, “When IoT Meet LLMs:
Applications and Challenges,” 2024. [Online]. Available: https:
//arxiv.org/abs/2411.17722
[3] Y . Gao, Z. Ye, M. Xiao, Y . Xiao, and D. I. Kim, “Guiding
IoT-Based Healthcare Alert Systems with Large Language Models,”
2024. [Online]. Available: https://arxiv.org/abs/2408.13071
[4] Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, “Pushing
Large Language Models to the 6G Edge: Vision, Challenges, and
Opportunities,” 2024. [Online]. Available: https://arxiv.org/abs/2309.
16739
[5] T. An, Y . Zhou, H. Zou, and J. Yang, “IoT-LLM: Enhancing Real-World
IoT Task Reasoning with Large Language Models,” 2025. [Online].
Available: https://arxiv.org/abs/2410.02429
[6] A. Hazra, A. Kalita, and M. Gurusamy, “Meeting the Requirements of
Internet of Things: The Promise of Edge Computing,” IEEE Internet of
Things Journal , vol. 11, no. 5, pp. 7474–7498, 2024.
[7] S. Barbarossa, D. Comminiello, E. Grassucci, F. Pezone, S. Sardellitti,
and P. Di Lorenzo, “Semantic Communications Based on Adaptive
Generative Models and Information Bottleneck,” IEEE Communications
Magazine , vol. 61, no. 11, pp. 36–41, 2023.
[8] A. Kalita, “Large Language Models (LLMs) for Semantic
Communication in Edge-based IoT Networks,” 2024. [Online].
Available: https://arxiv.org/abs/2407.20970
[9] K. Khatiwada, J. Hopper, J. Cheatham, A. Joshi, and S. Baidya,
“Large Language Models in the IoT Ecosystem – A Survey on
Security Challenges and Applications,” 2025. [Online]. Available:
https://arxiv.org/abs/2505.17586
[10] R. Birkmose, N. M. Reece, E. H. Norvin, J. Bjerva, and
M. Zhang, “On-Device LLMs for Home Assistant: Dual Role in
Intent Detection and Response Generation,” 2025. [Online]. Available:
https://arxiv.org/abs/2502.12923
[11] Z. Yu, Z. Wang, Y . Li, H. You, R. Gao, X. Zhou, S. R. Bommu,
Y . K. Zhao, and Y . C. Lin, “EDGE-LLM: Enabling Efficient Large
Language Model Adaptation on Edge Devices via Layerwise Unified
Compression and Adaptive Layer Tuning and V oting,” 2024. [Online].
Available: https://arxiv.org/abs/2406.15758