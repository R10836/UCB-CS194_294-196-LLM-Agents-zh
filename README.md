# UCB-CS194-294-196-LLM-Agents-zh
UC Berkeley CS 194/294-196 (LLM Agents) 的简体中文翻译以及笔记。正在更新中，还有很多很多需要完善，打算同步于课程更新，欢迎您的参与。

# 预备知识

## 英文缩写&术语

| 英语                                  | 简中         | 补充                                                         |
| ------------------------------------- | ------------ | ------------------------------------------------------------ |
| Large Language Model (LLM)            | 大语言模型   |                                                              |
| Artificial General Intelligence (AGI) | 通用人工智能 | 一个远大的目标                                               |
| Agent                                 | 智能体/代理  |                                                              |
| Embody                                | 具身         |                                                              |
| Multi-Agent System (MAS)              | 多智能体系统 |                                                              |
| Token                                 |              | 文本分割后得到的最小语义单位                                 |
| Prompt                                | 提示词       | 我们向AI提出的问题或指令                                     |
| Reason                                | 推理         | 模型根据已有的知识，通过逻辑的推导得出结论                   |
| align                                 | 对齐         | 确保大语言模型的行为与用户的意图或期望一致                   |
| Chain-of-Thought (CoT)                |              | 让LLM通过(intermediate step)解决问题的技术                   |
| decode                                | 解码         | 将模型生成的内部表示转化为人类可读的文本的过程               |
| Universal Self-Consistency (USC)      | 通用自一致性 |                                                              |
| Retrieval-Augmented Generation (RAG)  | 检索增强生成 | 在生成模型中引入检索机制，使得模型能够在生成文本之前从外部知识库中检索相关信息 |
| Reinforcement Learning (RL)           | 强化学习     | 智能体通过与环境进行交互，根据得到的奖励或惩罚来调整自己的行为，最终目标是最大化累计奖励 |
| Human-computer interface (HCI)        | 人机界面     |                                                              |
| Agent-computer interface (ACI)        |              |                                                              |

## 老师背景

| 老师         |      |      |
| ------------ | ---- | ---- |
| Dawn Song    |      |      |
| Xingyun Chen |      |      |
| Denny Zhou   |      |      |
| Shunyu Yao   |      |      |
|              |      |      |

## 提到的项目

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

## 提到的论文

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



# Dawn Song 的开场白

0：00 -- 9：20

## 人员

![image-20241004230657936](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410042307201.png)

这些公司都是在**人工智能（AI）**及**机器学习（ML）**领域的领军者或具有重要影响力的科技公司。它们在AI技术的开发、研究、应用以及商业化方面都发挥着关键作用。接下来分别讲解每家公司及其重点业务：

### 1. **DeepMind**
   - **简介**：DeepMind是一家由英国科学家在2010年创办的AI公司，2014年被谷歌收购。DeepMind以其在**强化学习**（reinforcement learning）和深度学习上的突破性研究而闻名，尤其是在开发能够击败人类顶尖围棋选手的AI系统AlphaGo上取得了巨大成就。
   - **重点**：DeepMind的核心技术主要集中在**强化学习**、**神经网络**、**深度学习**和**健康医疗AI**应用上。
   - **目标**：DeepMind的最终目标是开发出通用人工智能（AGI），也就是具备人类智能水平甚至超越人类的AI系统。

### 2. **OpenAI**
   - **简介**：OpenAI是由伊隆·马斯克（Elon Musk）、Sam Altman等人共同创立的一家美国AI公司，成立于2015年。最初OpenAI定位为非营利组织，后来转为有限盈利公司。它专注于开发安全和有益的人工智能，并且在自然语言处理（NLP）方面尤其突出，推出了非常成功的GPT系列模型，包括GPT-3、GPT-4等。
   - **重点**：OpenAI的核心产品包括GPT（生成式预训练变换器）系列和DALL·E（生成图像模型）。它在大规模**语言模型**（LLMs）和**生成式AI**的研究与应用上处于领先地位。
   - **目标**：推动AGI的安全发展，确保其对全人类有益。

### 3. **LlamaIndex**
   - **简介**：LlamaIndex（原称GPT Index）是一个专注于构建、查询和管理**大语言模型（LLMs）**数据库的开源项目。它简化了LLM与现有知识库系统的整合，允许开发者更方便地处理和管理大规模文本数据。
   - **重点**：LlamaIndex的核心功能是为基于语言模型的应用提供索引和检索机制，特别适合构建智能问答系统或需要处理大量文档和知识的应用场景。
   - **目标**：增强LLM在真实世界应用中的可用性和可扩展性，特别是文本数据的高效管理和查询。

### 4. **Databricks**
   - **简介**：Databricks是一家数据与AI公司，成立于2013年，主要由Apache Spark的创始团队创立。它提供基于云的**数据分析**和**机器学习平台**，为企业提供端到端的数据工程、机器学习和商业智能解决方案。
   - **重点**：Databricks的核心技术是**大数据**处理、**数据湖**和**机器学习平台**。它帮助企业在大规模数据处理和机器学习项目中实现高效协作与生产部署。
   - **目标**：提供统一的数据和AI平台，简化大数据和AI的协作与管理。

### 5. **ServiceNow**
   - **简介**：ServiceNow是一家提供**云计算**解决方案的公司，专注于数字工作流程的自动化，帮助企业提升工作效率。虽然它并非传统意义上的AI公司，但它近年来也在通过AI来增强其工作流程自动化解决方案，帮助企业优化IT、客户服务和员工管理。
   - **重点**：ServiceNow主要应用AI技术来改善**工作流自动化**，特别是通过AI驱动的**IT服务管理**（ITSM）和**企业资源规划**（ERP）系统提高企业效率。
   - **目标**：通过AI和自动化技术，帮助企业优化数字化流程管理。

### 6. **Meta AI**
   - **简介**：Meta AI是Meta（前Facebook）旗下的AI研究部门，致力于推动AI技术的发展，尤其是在NLP、计算机视觉和强化学习等领域。Meta AI还支持开源AI项目，包括PyTorch深度学习框架的开发。
   - **重点**：Meta AI在**大规模语言模型**、**计算机视觉**、**强化学习**、**生成式AI**和**社会AI伦理**等方面都有重要研究和进展。Meta AI希望通过AI来改进社交媒体、元宇宙等平台的体验。
   - **目标**：推动AI研究前沿发展，同时探索AI在社交、虚拟现实和增强现实等领域的创新应用。

### 7. **NVIDIA**
   - **简介**：NVIDIA是全球领先的**图形处理器（GPU）**制造商，近年来在AI和深度学习领域的硬件支持上扮演着不可或缺的角色。它开发的GPU成为了AI训练和推理计算的核心设备。
   - **重点**：NVIDIA的核心业务集中在**GPU硬件**、**深度学习**、**高性能计算**和**自动驾驶**等领域。NVIDIA的GPU和CUDA技术是AI研究和应用的重要基石。
   - **目标**：提供全球领先的AI计算基础设施，推动各行业的AI应用。

### 8. **Anthropic**
   - **简介**：Anthropic是由OpenAI的前高层创立的一家AI研究公司，成立于2021年。它致力于开发更安全、更可靠的AI系统，特别是在确保AI的安全性和伦理性方面做出了大量研究。
   - **重点**：Anthropic的核心关注点是**AI安全**、**公平性**和**透明性**，以确保AI的开发符合道德标准，并避免潜在的社会负面影响。
   - **目标**：推动AI的安全发展，并确保AI技术在全球范围内得到负责任的应用。



## 引言

### Accelerated development of large language models

大语言模型近期的加速发展

![image-20241004232043301](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410042320386.png)

### LLM agents: enabling LLMs to interact with the environments

大预言模型智能体：使大预言模型能与环境交互

![image-20241004232221907](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410042322999.png)

**Agent（智能体）: LLM + Memory: **大语言模型（LLM）负责进行推理和规划（Reasoning & Planning）。Memory 保存智能体的记忆，用于帮助其在行动时参考过去的经验和数据。

**Tool use（工具使用）+ Retrieval（检索）**：Agent 使用外部工具来执行任务或辅助计算。或从外部数据源中检索信息，以增强其推理和决策能力。

**Action（行动）**：智能体根据推理和规划采取的实际行为，作用于环境中。

**Feedback（反馈）**：智能体从环境中接收到的反馈信息，用于调整和优化后续行动。

#### **详细解释：**

1. **LLM（Reasoning & Planning）**

- **大语言模型（LLM）\**位于智能体的核心，负责进行\**推理**和**规划**。LLM通过分析输入信息（如文本、对话等）来推测下一步的行动。它不仅能通过语言理解问题，还能生成解决方案或做出复杂的规划决策。其推理能力依赖于模型的训练数据以及内部的算法设计。

2. **Memory（记忆）**

- **记忆**在这里指的是智能体能够保存之前的交互或信息，用于在后续的任务中参考。这种记忆机制帮助LLM保持**上下文**的连贯性，并增强智能体的长期决策能力。记忆可能包含外部存储的内容，如数据库、先前执行的任务结果、与用户的历史交互等。

3. **Tool use（工具使用）**

- **工具使用**意味着智能体能够调用外部的工具来完成任务，这些工具可能包括**计算工具**、**API**、**爬虫**或其他自动化工具。工具的使用使得智能体超越了其语言模型的局限性，可以处理更多类型的数据，执行计算密集型操作，或进行系统级别的任务。

4. **Retrieval（检索）**

- **检索**是指智能体能够从外部数据库或信息源中获取相关数据。这一机制帮助LLM补充其有限的知识，因为LLM的训练数据不一定包含实时或具体的细节。通过检索，智能体能够获得最新或特定的信息，以做出更好的决策。

5. **Action（行动）**

- **行动**是智能体根据推理结果在外部环境中执行的具体行为。这可能包括生成文本回答、操作系统任务、与外部API交互等。行动是智能体与外界互动的体现。

6. **Environment（环境）**

- **环境**是智能体所作用的外部世界，它可以是一个虚拟环境（如软件系统）、物理环境（如机器人执行任务的现实世界）或者社会环境（如智能体与用户的对话场景）。环境的状态会受到智能体行动的影响。

7. **Feedback（反馈）**

- **反馈**是智能体从环境中收到的回馈信息。这些反馈帮助智能体**调整行动策略**，例如，环境的反馈可以用来判断智能体的某个行为是否成功，从而帮助智能体改进后续的推理和决策。



## LLM Agents in Diverse Environments

大语言模型智能体可用于多种环境

![image-20241004234418561](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410042344720.png)

这张图描绘了**多智能体系统（Multi-agent Systems）**中的一个**语言智能体（Language Agent）**如何与外部环境互动，并通过**化身（Embodiment）**与物理世界进行交互。图中包括了语言智能体的内部组成（推理与规划、工作记忆、长期记忆）以及与环境的双向信息交流。以下是这幅图的详细翻译和解释：

---

### **翻译：**

#### 左侧部分：
- **Language Agent（语言智能体）**
  - **Reasoning & Planning（推理与规划）**：负责决策和解决问题。
  - **Working Memory（工作记忆）**：处理短期任务和信息的记忆模块。
  - **Long-term Memory（长期记忆）**：存储长期信息和经验。
  - **多智能体系统（Multi-Agent System，MAS）** 是由多个智能体（agent）组成的系统，这些智能体可以在一个共享的环境中相互作用、通信、协作或竞争，以完成共同的任务或解决问题。
  
- **Embodiment（化身/体现）**
  - **Sensors（传感器）**：RGB摄像头、声音、距离、接触、光线等传感器，用于感知环境。
  - **Effectors（执行器）**：如机械臂、腿、轮子、扬声器等，用于与物理世界进行物理交互。
  
#### 右侧部分：
- **Environment（环境）**
  - **Humans（人类）**：与人类进行交互。
  - **Databases（数据库）**：访问和使用数据库中的数据。
  - **Web（网页）**：从网络中获取信息。
  - **APIs（应用程序接口）**：通过API与其他系统或服务进行通信。
  - **Knowledge Bases（知识库）**：使用结构化的知识库作为信息源。
  - **Apps（应用程序）**：与各种应用程序进行集成和交互。
  - **Physical World（物理世界）**：与物理环境直接互动。

#### 双向箭头解释：
- **Grounding（落地/基础化）**：语言智能体通过与人类交互和使用工具等方式，增强其对现实世界的理解。
- **Human Interaction（人机交互）**：智能体与人类之间的互动，包括对话、任务协作等。
- **Tool Augmentation（工具增强）**：智能体通过使用外部工具，增强其功能和能力。
- **Memory Update（记忆更新）**：智能体通过与环境的互动，不断更新其工作记忆和长期记忆。

---

### **详细解释：**

#### 1. **Language Agent（语言智能体）**
   - **语言智能体**是图中的核心部分，主要依赖大语言模型（LLM）进行**推理与规划（Reasoning & Planning）**。它通过接收输入、处理信息来生成输出。智能体的工作记忆用于管理短期任务信息，长期记忆则用于存储历史数据和经验，帮助其做出更为准确和上下文相关的决策。

#### 2. **Embodiment（化身/体现）**
   - **化身**指的是智能体的物理体现方式。通过传感器，智能体能够感知外部世界，包括颜色、声音、距离等多种信息。这些传感器可以帮助智能体获得丰富的环境感知数据。执行器则是智能体用来影响物理世界的“工具”，比如机械臂、轮子等，它们可以让智能体做出物理行为。

#### 3. **Environment（环境）**
   - **环境**是智能体操作和交互的外部空间，涵盖了人与数字世界的多个层面。智能体可以从数据库中获取数据，从网页上抓取信息，使用API与外部系统进行通信。它还能与知识库、应用程序（如手机APP或软件）以及物理世界进行互动。智能体在与这些外部资源的交互中不断更新其内部记忆和状态。

#### 4. **Grounding（落地/基础化）与 Human Interaction（人机交互）**
   - **Grounding**是指智能体通过与人类交互和工具使用，巩固其对现实世界的理解。这包括从用户反馈中学习，使用工具辅助任务完成等。
   - **人机交互**是智能体与人类之间的互动，如自然语言对话、协作完成任务等。这种互动帮助智能体提升任务处理的精确性。

#### 5. **Tool Augmentation（工具增强）与 Memory Update（记忆更新）**
   - **工具增强**意味着智能体可以通过外部工具（如API或其他软件工具）扩展其能力，做出更复杂的决策或执行任务。
   - **记忆更新**指的是智能体通过与外界的互动，持续更新其工作记忆和长期记忆，这能让智能体从过去的经验中学习，提升后续的任务表现。



## Multi-agent collaboration: division of labor for complex tasks

多智能体协作：复杂任务的分工

![image-20241005000705629](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410050007805.png)

这张图展示了多智能体系统中的**可定制智能体（Customizable Agents）**和**多智能体对话（Multi-Agent Conversations）**的概念。主要分为三个部分：**可对话智能体（Conversable Agent）**、**智能体定制（Agent Customization）**以及**灵活的对话模式（Flexible Conversation Patterns）**。以下是这幅图的详细翻译与解释。

---

### **翻译：**

#### 图的标题：
- **Specialized agents for different subtasks（为不同子任务定制的专门智能体）**
  
  - 示例包括：**Autogen、CrewAI、CAMEL、Mixture-of-Agents**等。
  
  ### Autogen
  
  - **微软出品：** Autogen 是微软推出的一款工具，旨在让大型语言模型（LLM）能够创建下一代应用程序。
  - **核心功能：** 它能将 LLM 变成一个个可以对话、执行任务的智能体。这些智能体可以互相协作，共同完成复杂任务。
  - **应用场景：** 从生成文本、翻译语言，到更复杂的任务如编写代码、创建数字内容，Autogen 的应用潜力巨大。
  
  ### CrewAI
  
  - **协同工作：** CrewAI 的理念是让多个 AI 智能体像团队一样协同工作。
  - **分工合作：** 每个智能体都有自己的角色，比如研究员、作家、规划师。它们可以共同完成一项任务，就像一个真正的团队一样。
  - **灵活性强：** CrewAI 的灵活性很高，你可以自定义智能体，让它们适应不同的任务。
  
  ### CAMEL
  
  - **开源社区：** CAMEL 是一个开源社区，专注于探索多智能体系统的扩展规律。
  - **LLM 多智能体框架：** CAMEL 提供了一个框架，可以将多个 LLM 结合起来，形成一个多智能体系统。
  - **研究导向：** 这个项目更偏向于研究，旨在深入了解多智能体系统的本质。
  
  ### Mixture-of-Agents
  
  - **模型聚合：** Mixture-of-Agents（MoA）是一种通过聚合多个 LLM 的能力来增强性能的方法。
  - **优势互补：** 不同的 LLM 有不同的擅长领域，MoA 可以将它们的优势结合起来，从而获得更好的效果。
  - **应用前景：** MoA 在自然语言处理、机器翻译等领域有广阔的应用前景。

#### 左上角：
- **Conversable Agent（可对话智能体）**：代表可以与用户或其他智能体进行对话的基础智能体。

#### 左下角：
- **Agent Customization（智能体定制）**：展示了可以对智能体进行定制。图中分为两种定制后的智能体：
  1. 一个蓝色的智能体（表示一般的对话智能体），可能用于常规的对话任务。
  2. 一个绿色的智能体，带有Python标志，表示此智能体具有编程能力（Python）并且与用户协作完成任务。

#### 右上角：
- **Multi-Agent Conversations（多智能体对话）**：展示了多个智能体之间的对话。
  1. 蓝色智能体与绿色智能体进行双向对话，表示这些智能体可以协作完成任务。
  
#### 右下角：
- **Flexible Conversation Patterns（灵活的对话模式）**：
  - **Joint chat（联合对话）**：多个智能体可以同时进行对话，协同工作，解决某个任务或问题。
  - **Hierarchical chat（层级对话）**：在层级结构中，不同智能体承担不同角色，可能通过分层次的管理进行合作。

---

### **详细解释：**

#### 1. **Conversable Agent（可对话智能体）**
   - 这是一个可以通过自然语言与人类或其他智能体交互的基础模块。在这个系统中，智能体可以接收指令，进行对话，处理任务。

#### 2. **Agent Customization（智能体定制）**
   - **智能体定制**展示了如何通过扩展功能和工具来增强智能体的能力。
     - 图中的蓝色智能体代表一般的对话能力，适用于常规交互。
     - 绿色智能体则表示具备特定的技术能力（如Python编程）和合作特性，能够与用户协作编程或执行复杂任务。这展示了智能体可以根据需求定制，适应不同的任务环境。

#### 3. **Multi-Agent Conversations（多智能体对话）**
   - **多智能体对话**展示了**多个智能体之间**的互动，可以通过合作完成复杂的任务。例如，一个智能体可能擅长语言处理，另一个可能专注于编程任务，两者可以互相协作。

#### 4. **Flexible Conversation Patterns（灵活的对话模式）**
   - **联合对话（Joint chat）**：多个智能体平等参与同一个对话，可以同时响应同一任务。这种模式适合需要多方协作的任务，如团队讨论。
   - **层级对话（Hierarchical chat）**：智能体之间存在某种层级关系，一个智能体可能管理或指导其他智能体。这种模式可以用于复杂任务的管理和分配，类似于组织中的管理层级。

#### 5. **Specialized agents for different subtasks（为不同子任务定制的专门智能体）**
   - 图中列举的Autogen、CrewAI、CAMEL、Mixture-of-Agents等，都是专为不同子任务设计的智能体系统。这些系统可以根据不同任务需求进行智能体的定制或组合，以提高任务的处理效率。

---





![image-20241005000723821](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410050007032.png)

Emergence of social behaviors with role-play LLMs

角色扮演大预言模型的社会行为的出现

Generative agents, Project Sid, ...



## Why empowering LLMs with the agent framework

幻灯片的原文如下：

- **Solving real-world tasks typically involves a trial-and-error process**
  
- **Leveraging external tools and retrieving from external knowledge expand LLM’s capabilities**

- **Agent workflow facilitates complex tasks**
  - **Task decomposition**
  - **Allocation of subtasks to specialized modules**
  - **Division of labor for project collaboration**
  - **Multi-agent generation inspires better responses**

---

### **翻译：**

- **解决现实世界任务通常涉及一个试错过程**
  
- **利用外部工具和从外部知识中检索信息扩展了LLM的能力**

- **智能体的工作流程促进了复杂任务的处理**
  - **任务分解**
  - **将子任务分配给专门的模块**
  - **通过分工协作完成项目**
  - **多智能体的生成启发更好的响应**



## Challenges for LLM agent deployment in the wild

这张幻灯片的原文如下：

- **Reasoning and planning**
  - LLM agents tend to make mistakes when performing complex tasks end-to-end

- **Embodiment and learning from environment feedback**
  - LLM agents are not yet efficient at recovering from mistakes for long-horizon tasks
  - Continuous learning, self-improvement
  - Multimodal understanding, grounding and world models

- **Multi-agent learning, theory of mind**

- **Safety and privacy**
  - LLMs are susceptible to adversarial attacks, can emit harmful messages and leak private data

- **Human-agent interaction, ethics**
  - How to effectively control the LLM agent behavior, and design the interaction mode between humans and LLM agents

---

翻译：

- **推理与计划**
  - 大型语言模型（LLM）代理在执行端到端的复杂任务时容易犯错

- **具身性与从环境反馈中学习**
  - LLM代理还未能高效地从长时间任务中的错误中恢复
  - 持续学习、自我提升
  - 多模态理解、基础认知和世界模型

- **多智能体学习、心智理论**

- **安全与隐私**
  - 大型语言模型容易受到对抗性攻击，可能会发送有害信息并泄露私人数据

- **人机互动与伦理**
  - 如何有效控制LLM代理的行为，并设计人类与LLM代理的交互模式



## Topics covered in this course

这张幻灯片的原文如下：

- **Topics covered in this course**
  - **Model core capabilities**
    - Reasoning
    - Planning
    - Multimodal understanding
  - **LLM agent frameworks**
    - Workflow design
    - Tool use
    - Retrieval-augmented generation
    - Multi-agent systems
  - **Applications**
    - Software development
    - Workflow automation
    - Multimodal applications
    - Enterprise applications
  - **Safety and ethics**

---

**翻译：**

- **本课程涵盖的主题**
  - **模型核心能力**
    - 推理
    - 计划
    - 多模态理解
  - **大型语言模型代理框架**
    - 工作流程设计
    - 工具使用
    - 检索增强生成
    - 多智能体系统
  - **应用**
    - 软件开发
    - 工作流程自动化
    - 多模态应用
    - 企业应用
  - **安全与伦理**





# Lecture 1, Denny Zhou

**LLM Reasoning: Key Ideas and Limitations**

**大预言模型推理：关键idea和局限性**



LLM is a "transformer" model trained to predict the next word.





![image-20241005235851560](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410052358690.png)

Few-shot prompting -> reasoning process

![image-20241005235943051](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410052359212.png)



Key Idea: Derive the Final Answer through

Intermediate Steps



Regardless of training, fine-tuning, or prompting, when provided with examples that include intermediate steps, LLMs will generate responses that also include intermediate steps.

无论经过训练、微调还是提示，只要提供包含中间步骤的示例，大型语言模型（LLM）都会生成也包含中间步骤的响应。



In addition to intermediate steps, is it helpful to introduce reasoning strategies in demonstration examples?

除了中间步骤之外，在示例中引入推理策略是否有帮助？

demonstration examples: 示例



Least-to-Most Prompting

Enable easy-to-hard generalization by decomposition

从易到难提示：通过分解实现简单到复杂的泛化

**泛化** 在机器学习中，特别是自然语言处理领域，指的是模型在训练数据上学习到的知识，能够应用到未曾见过的新数据上的能力。简单来说，就是模型能够举一反三，不仅能解决训练时遇到的问题，还能处理新的、相似的问题。



SCAN (Compositional Generalization)

组合泛化

在大型语言模型（LLM）领域，组合泛化指的是模型能够将学到的知识组合起来，以应对从未见过的、由已知组件组成的复杂任务的能力。换句话说，就是模型能够通过将已知的知识模块进行组合，来解决新的问题。



CFQ (Compositional Generalization): Text-to-Code



Why interediate steps are helpful?

Transformer generating intermediate steps can solve any inherently serial problem as long as its depth exceeds a constant threshold

Transformer 生成中间步骤可以解决任何固有的串行问题，只要其深度超过一个常数阈值

Transformer generating direct answers either requires a huge depth to solve or cannot solve at all

Transformer 生成直接答案要么需要巨大的深度才能解决，要么根本无法解决



How to trigger step by step reasoning without using demonstration examples?

如何在不使用演示示例的情况下触发逐步推理？



LLMs as Analogical Reasoners

类比推理者



Adaptive generate relevant examples and knowledge, rather than using a fix set of examples.

自适应生成相关示例和知识，而不是使用固定的示例集。



Is it possible to trigger step by step reasoning even without using any prompt like "let's think step by step"?

是否可以在不使用“让我们一步一步思考”这样的提示的情况下，触发模型进行逐步推理？

—— Chain-of-Thought Reasoning without Prompting

无需提示的思维链推理



Key observations:

1: Pre-trained LLMs have had responses with step-by-step reasoning among the generations started with the top-k tokens.

2: Higher confidence in decoding the final answer when a step-by-step reasoning path is present.

主要观察结果：

1、预训练的大型语言模型（LLMs）在生成过程中，已经出现了包含逐步推理的响应。

2、当存在逐步推理路径时，对最终答案的解码置信度更高。



Generating intermediate steps are helpful, but...



Any concern on generating intermediate steps instead of direct answers?

是否担心生成中间步骤而不是直接答案？

Always keep mind that LLMs are probabilistic models(概率模型) of generating next tokens. They are not humans.



![image-20241006143414694](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410061434870.png)

sampling：采样是指从模型生成的概率分布中随机选择一个元素的过程。这个概率分布代表了模型认为每个可能的下一个词（或其他token）出现的可能性。

![image-20241006144236045](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410061442230.png)



Self-Consistency

Greatly improves step-by-step reasoning

自我一致性 显著增强了 逐步推理能力

**自一致性** (Self-Consistency) 在大型语言模型 (LLM) 领域，指的是模型在处理同一问题或相似问题时，能够产生一致的、可靠的答案。换句话说，模型不会出现前后矛盾、自相矛盾的情况。

**逐步推理** (Step-by-Step Reasoning) 则是指模型通过一系列中间步骤来解决问题，而不是直接给出最终答案。这类似于人类思考问题的方式，我们通常会将复杂问题分解成更小的子问题，逐一解决。



How about free-from answers? 开放式答案？

Universal Self-Consistency (USC) 通用自一致性

Ask LLMs to self-select the most consistant answer 让大型语言模型自行选择最一致的答案



**Limitations**

LLMs Can Be Easily Distracted by Irreverent Context

LLMs Cannot Self-Correct Reasoning Yet

Premise Order Matters in LLM Reasoning 前提的顺序在大型语言模型进行推理时会产生影响



**oracle:** 作为神谕（即权威的判断标准）

Oracle: Let LLMs self correct only when the answer is wrong

当大型语言模型的答案错误时，由一个权威系统来引导模型进行自我纠正。



Multi-LLM debate? Worse than self-consistency!

多个大型语言模型之间的辩论比模型自身的自洽性问题更严重。



Oracle feedback needed for LLM to self-correct

大型语言模型需要来自权威且可靠的外部反馈才能有效地进行自我纠错。



Self-debug naturally leverage unit tests as oracle

自我调试 自然地利用单元测试 作为权威的判断标准



![image-20241006154158609](https://raw.githubusercontent.com/LukiRyan/TyporaImageBox/main/img/202410061541755.png)

幻灯片的原文如下：

- **Summary**
  - Generating intermediate steps improves LLM performance
    - Training / finetuning / prompting with intermediate steps
    - Zero-shot, analogical reasoning, special decoding
  - Self-consistency greatly improves step-by-step reasoning
  - Limitation: irrelevant context, self-correction, premise order

---

**翻译：**

- **总结**
  - 生成中间步骤可以提高LLM的性能
    - 通过中间步骤进行训练/微调/提示
    - 零样本推理、类比推理、特殊解码
  - 自我一致性极大地改善了逐步推理的能力
  - 限制：无关上下文、自我纠错、前提顺序

---

**解释：**

这张幻灯片总结了通过生成中间步骤可以大幅提升大型语言模型（LLM）的性能，尤其是在训练、微调和提示的过程中，通过分步骤完成任务能够使模型更有效地解决复杂问题。零样本推理、类比推理等技术也是提升模型性能的关键。自我一致性，即模型在重复解答中保持一致的推理路径，能进一步提升其逐步推理的能力。不过，模型在处理一些问题时仍然存在局限性，例如无法区分无关的上下文、自我纠正中的错误和前提顺序问题。



What are next?

1: Define a right problem to work on

2: Solve it from the first principles

1：定义一个正确的问题来解决

2：从第一原则入手解决

**第一性原理**（First Principles）是一个哲学和科学方法论的概念，它强调从最基本的、不可再分的真理出发，通过逻辑推理和演绎，构建起复杂的知识体系。换句话说，就是将问题分解到最基本的组成部分，然后从这些基础部分重新构建起整个问题。



# Lecture 2, Shunyu Yao

**Brief History and Overview**

What is LLM agents?

A brief history of LLM agents

On the future of LLM agents



https://www.youtube.com/watch?v=RM6ZArd2nVc&list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&index=4

中文字幕较为准确

**评论区有人总结：**

Chapter 1: Introduction (00:00 - 00:49)
- Presenter: *Shunyu Yao* introduces LLM agents.
- **Goals**: Discuss what LLM agents are, their history, and future directions.
- **Field complexity**: The area is evolving, dynamic, and hard to fully define.

Chapter 2: What is an LLM Agent? (00:50 - 03:55)
- Definition of **Agent**: An intelligent system interacting with an environment, physical or digital (robots, video games, chatbots).
- LLM agents: Agents that use *LLMs to interact and reason* within a text-based environment.
- **Three agent categories**: 
  1. Text agents (interact via language),
  2. LLM agents (use LLMs for action),
  3. Reasoning agents (LLMs reasoning before acting).

Chapter 3: Early Text Agents and Limitations (03:56 - 05:36)
- Early *rule-based agents* (e.g., ELIZA, 1960s) were *domain-specific* and limited.
- Later, *reinforcement learning (RL)* text agents emerged, but required *extensive training* and specific rewards.

Chapter 4: LLMs and Their Potential (05:37 - 07:48)
- **LLMs**, trained via **next-token prediction**, generalize across tasks with minimal task-specific training.
- *GPT-3 (2020)* marked the start of exploring LLMs in various reasoning and action tasks.

Chapter 5: LLM Agents’ History (07:49 - 09:06)
- Historical perspective: Combining *LLMs with reasoning* (symbolic reasoning, acting tasks).
- The field has grown to encompass **web interaction, software engineering**, and **scientific discovery**.

Chapter 6: Question-Answering and Reasoning Challenges (09:07 - 12:58)
- *Challenges* with using LLMs for QA (e.g., outdated knowledge, complex computations).
- Solutions:
  - *Program generation* for complex calculations.
  - *Retrieval-Augmented Generation (RAG)* for real-time knowledge retrieval.
  - **Tool use**: Invoke external tools like calculators or APIs for knowledge gaps.

Chapter 7: ReAct Paradigm (12:59 - 18:52)
- *ReAct**: Combining **reasoning and acting* to improve task-solving by iterating thought and action.
- Example: GPT-4 reasoning about purchasing companies by *searching and calculating* market caps.
- **Human-like reasoning**: ReAct enables agents to adapt and improve reasoning in real-time.

Chapter 8: Limitations of Text-to-Action Mapping (18:53 - 23:00)
- **Challenges in video games**: Mapping text observations directly to actions without thinking can lead to failure (e.g., imitating instead of reasoning).
- *ReAct’s advantage**: Adding a **thinking action* allows for **planning, reflection**, and adaptive strategies.

Chapter 9: Long-Term Memory and Reflexion (23:01 - 33:22)
- *Short-term memory limits* LLM agents (context window constraints).
- *Long-term memory**: Reflexion introduces a way for agents to **reflect on failures* and improve over time.
- Examples: *Coding tasks* with unit test feedback allow agents to persist knowledge across tasks.

Chapter 10: Broader Use of LLM Agents (33:23 - 37:53)
- *Applications beyond QA**: LLM agents are being applied to **real-world tasks* like *online shopping (WebShop)* and **software engineering (SWE-Bench)**.
- **ChemCrow example**: LLM agents propose chemical discoveries, extending their impact into the **physical realm**.

Chapter 11: Theoretical Insights on Agents (37:54 - 43:41)
- Traditional agents have a fixed action space (e.g., *Atari* agents).
- *LLM agents’ augmented action space**: Reasoning allows for an **infinite range of thoughts* before acting, offering a more *human-like* approach.

Chapter 12: Simplicity and Abstraction in Research (43:42 - 54:23)
- *Simplicity**: Simple concepts like **chain of thought* and *ReAct* are powerful because they are **generalizable**.
- Importance of *abstraction**: Successful research involves both **deep understanding of tasks* and **high-level abstraction**.

Chapter 13: Future Directions (54:24 - 1:07:38)
- *Training**: Models should be trained specifically for **agent tasks* to improve their performance in complex environments.
- **Interface**: Optimizing the agent’s environment (e.g., file search commands) enhances performance.
- **Robustness**: Agents must consistently solve tasks, not just occasionally succeed.
- **Human interaction**: Agents need to work reliably with humans in **real-world scenarios**.
- **Benchmarking**: Developing practical, scalable benchmarks for evaluating agents in **real-life tasks**.

Chapter 14: Summary and Lessons (1:07:39 - 1:08:38)
- *Key insights**: LLM agents are **transforming tasks* across many domains. 
- The future of LLM agents involves tackling robustness, human collaboration, and expanding their applications into **physical spaces**.



## What is "agent"?

![image-20241006232204160](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410062322343.png)

幻灯片的原文如下：

- **What is "agent"?**
  - An "intelligent" system that interacts with some "environment"
    - Physical environments: robot, autonomous car, ...
    - Digital environments: DQN for Atari, Siri, AlphaGo, ...
    - Humans as environments: chatbot
  - Define "agent" by defining "intelligent" and "environment"
    - It changes over time!
    - Exercise question: how would you define "intelligent"?

---

**翻译：**

- **什么是“智能体（agent）”？**
  - 一个与某种“环境”进行交互的“智能”系统
    - 物理环境：机器人、自动驾驶汽车等
    - 数字环境：用于Atari的DQN、Siri、AlphaGo等
    - 人类作为环境：聊天机器人
  - 通过定义“智能”和“环境”来定义“智能体”
    - 它随着时间而变化！
    - 练习问题：你会如何定义“智能”？



## What is "LLM agent"?

![image-20241006232545169](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410062325468.png)

幻灯片的原文如下：

- **What is "LLM agent"?**
  - Level 1: Text agent
    - Uses text action and observation
    - Examples: ELIZA, LSTM-DQN
  - Level 2: LLM agent
    - Uses LLM to act
    - Examples: SayCan, Language Planner
  - Level 3: Reasoning agent
    - Uses LLM to reason to act
    - Examples: ReAct, AutoGPT
    - **The key focus of the field and the talk**

---

**翻译：**

- **什么是“LLM智能体”？**
  - 第一层次：文本智能体
    - 使用文本行动和观察
    - 示例：ELIZA, LSTM-DQN
  - 第二层次：LLM智能体
    - 使用大语言模型（LLM）来执行操作
    - 示例：SayCan, 语言规划器
  - 第三层次：推理智能体
    - 使用大语言模型推理并行动
    - 示例：ReAct, AutoGPT
    - **本领域和讨论的重点**



## Text agent

* Domain specific!

* Requires scalar reward signals

* Require extensive training

**文本代理（Text Agent）**：

- **定义：** 一种能够自主执行文本任务的 AI 模型。它可以理解、生成、并根据给定的目标或任务与用户或环境进行交互。
- 功能：
    - **理解自然语言：** 准确地理解用户指令或环境中的文本信息。
    - **生成文本：** 根据理解的内容，生成符合要求的文本回复、文章、代码等。
    - **执行任务：** 在文本环境中完成特定的任务，例如信息检索、对话、翻译等。

**领域专业化（Domain Specific）**：

- **含义：** 文本代理在特定领域（如医疗、法律、金融等）经过大量数据训练，使其具备该领域专业知识和技能。
- 优势：
    - **准确性高：** 能够更准确地理解和生成领域相关的文本。
    - **效率高：** 通过预训练，可以快速适应新的任务。
    - **专业性强：** 能够提供更专业、更深入的回答。

**标量奖励信号（Scalar Reward Signals）**：

- **定义：** 用于评估文本代理行为的单一数值。
- 作用：
    - **强化学习：** 通过不断调整模型参数，最大化累积奖励，从而实现模型的优化。
    - **反馈机制：** 提供模型行为的直接反馈，引导模型向目标方向发展。
- 示例：
    - 对话系统：对话流畅度、信息准确性、用户满意度等。
    - 文本摘要：摘要质量、信息覆盖率等。

**大量训练（Extensive Training）**：

- **必要性：** 文本代理需要大量高质量的训练数据，才能学习到复杂的语言模式和领域知识。
- **数据类型：** 包括文本、代码、结构化数据等。
- **训练方式：** 通常采用大规模预训练模型（如 GPT-3）进行微调，或者使用强化学习进行端到端的训练。



## The promise of LLMs

Training: next-token prediction on massive text corpora

Inference: (few-shot) prompting for various tasks!



## A brief history of LLM agent

LLM agent -> Reasoning agent

![image-20241007112425346](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071124556.png)

幻灯片的原文如下：

- **A brief history of LLM agents**
  - **Reasoning**
    - CoT
    - Zero-shot CoT
    - Self-consistency
    - ...
  - **Acting**
    - Grounding, tool use, etc.
    - Game
    - Robotics
    - RAG
    - ...
  - **LLM agent (but not reasoning agent)** 
  - **ReAct**
  - **New applications/tasks/benchmarks**
    - Web browsing
    - Software engineering
    - Scientific discovery
    - ...
  - **New methods**
    - Memory, learning, planning, multi-agent
    - ...

---

**翻译：**

- **LLM智能体的简史**
  - **推理**
    - Chain of Thought (CoT, 思维链)
    - Zero-shot CoT（零次训练思维链）
    - 自一致性
    - ...
  - **行动**
    - 基础、工具使用等
    - 游戏
    - 机器人
    - RAG（Retrieval-Augmented Generation, 检索增强生成）
    - ...
  - **LLM智能体（但不是推理智能体）**
  - **ReAct**
  - **新应用/任务/基准**
    - 网页浏览
    - 软件工程
    - 科学发现
    - ...
  - **新方法**
    - 记忆、学习、规划、多智能体
    - ...

---

**解释：**

这张幻灯片提供了大语言模型（LLM）智能体的发展历史和未来趋势。重点分为推理和行动两个主要方面：

1. **推理（Reasoning）**：包括思维链（CoT）、零次训练的思维链（Zero-shot CoT）以及自一致性等概念。这些技术帮助LLM智能体进行更复杂的推理任务。
   
2. **行动（Acting）**：这一部分专注于智能体如何使用工具，执行游戏任务，或用于机器人和检索增强生成（RAG）等应用。此类智能体虽然有行动能力，但不具备高深度推理能力。

3. **ReAct**：这是一个新出现的框架，它结合了推理和行动的能力，将LLM智能体推向了一个新的发展阶段。

4. **新应用和任务**：幻灯片还列出了LLM智能体未来可能应用的领域，如网页浏览、软件工程、科学发现等。这表明LLM智能体正在扩展到更多实际应用领域。

5. **新方法**：新出现的研究方向包括如何让智能体具备记忆、学习、规划和多智能体协作能力，以进一步提高其智能水平。



## RAG for knowledge

![image-20241007113647864](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071136004.png)

**Retrieval-Augmented Generation**，是一种结合了信息检索和生成模型的新型方法。它通过在生成模型中引入检索机制，使得模型能够在生成文本之前从外部知识库中检索相关信息，从而提升生成文本的质量、准确性和多样性。

**核心思想：**

- **检索相关信息：** 当模型接收到一个问题或任务时，它会先从外部知识库（如维基百科、公司内部文档等）中检索与之相关的文本片段。
- **结合上下文生成：** 将检索到的相关信息与原始输入一同输入到生成模型中，生成模型在生成文本时会参考这些额外的信息，从而生成更符合上下文、更准确的文本。

**优势：**

- **提高生成质量：** 通过引入外部知识，模型能够生成更准确、更相关、更具信息量的文本。
- **增强模型泛化能力：** 模型不再仅仅依赖于训练数据，能够处理更多样的任务和问题。
- **解决幻觉问题：** 减少模型生成虚假或不真实信息的概率。
- **提高模型可解释性：** 通过展示检索到的信息，可以更好地理解模型的生成过程。

**应用场景：**

- **问答系统：** 提高回答的准确性和全面性。
- **文本摘要：** 生成更精炼、更准确的摘要。
- **对话系统：** 增强对话的连贯性和信息量。
- **内容生成：** 自动生成新闻报道、产品描述等。

**与传统生成模型的区别：**

| 特点     | 传统生成模型                 | RAG模型                          |
| -------- | ---------------------------- | -------------------------------- |
| 知识来源 | 仅依赖训练数据               | 训练数据 + 外部知识库            |
| 生成过程 | 基于训练数据中的模式进行生成 | 在生成过程中引入检索到的相关信息 |
| 优势     | 生成多样性                   | 生成质量、准确性、可解释性       |



## Tool use

Special tokens to invoke(调用) tool calls for:

​	Search engine, calculator, etc.

​	Task-specific models (translation)

​	APIs

**特殊标记用于调用工具：**

- 搜索引擎、计算器等
- 任务特定模型（翻译）
- API



**工具使用（Tool Use）**：

- **直译：** 工具使用
- **意译：** 工具调用、工具辅助、工具集成
- **含义：** 在大语言模型中，将外部工具（如搜索引擎、计算器、翻译模型等）集成进来，让模型能够在生成文本时调用这些工具，从而获取更准确、更全面的信息，提升生成结果的质量。

**特殊标记（Special Tokens）**：

- **直译：** 特殊标记
- **意译：** 特殊符号、控制符
- **含义：** 在文本中插入一些特殊的符号或序列，用于触发模型的特定行为，比如调用某个工具、切换到不同的模式等。这些标记通常是预定义的，模型在训练过程中学习到它们的含义。

**模型调用（Model Calls for: Search engine, calculator, etc.）**：

- **直译：** 模型调用（用于搜索引擎、计算器等）
- **意译：** 模型调用外部工具、模型集成
- **含义：** 当模型在生成文本时遇到需要计算、搜索或翻译等任务时，它会通过特定的标记或机制调用相应的工具或模型来完成这些任务。



举例：

### Search Engine

**Prompt:** "What is the capital of Australia?" 

**Special Token:** `[SEARCH]` 

**Model:** Recognizes the `[SEARCH]` token and calls a search engine API. The API returns "Canberra" as the capital of Australia. 

**Response:** "The capital of Australia is Canberra."

### Calculator

**Prompt:** "How much is 500 divided by 7?" 

**Special Token:** `[CALC]` 

**Model:** Recognizes the `[CALC]` token and calls a calculator API. The API calculates the result. 

**Response:** "500 divided by 7 equals approximately 71.43."

### Task-Specific Model (Translation)

**Prompt:** "Translate 'Hello, how are you?' into French." 

**Special Token:** `[TRANSLATE]` 

**Model:** Recognizes the `[TRANSLATE]` token and calls a translation model. The model translates the text. 

**Response:** "Bonjour, comment allez-vous?"

### API

**Prompt:** "Get me the current weather in Tokyo." 

**Special Token:** `[WEATHER]` 

**Model:** Recognizes the `[WEATHER]` token and calls a weather API. The API retrieves the current weather data for Tokyo. 

**Response:** "The current weather in Tokyo is sunny with a temperature of 25 degrees Celsius."

**Note:** The specific special tokens used may vary depending on the implementation. These examples illustrate the general concept of using special tokens to trigger tool calls within a large language model.

步骤：

1. 自然语言输入
2. 问题类型识别
3. 特殊标记匹配
4. 任务执行
5. 结果输出



## What if both knowledge and reasoning are needed?

![image-20241007115631755](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071156904.png)

幻灯片的原文如下：

- **QA**
  - **Knowledge-intensive QA**
    - Tool use
    - RAG
    - WebGPT
  - **Symbolic reasoning**
      - CoT
  - **Mathematical reasoning**
    - Tool use
    - PoT
  - **Commonsense QA**
    - RAG
  - **Multi-hop knowledge-intensive QA**
    - Self-ask
    - IRCOT



### Knowledge-intensive 知识密集型

- **含义：** 要求模型具备丰富的背景知识才能回答的问题。
- **例子：** “为什么地球是圆的？”、“谁发明了电灯？”
- **Tool use 工具使用**：指模型在回答问题时，可以调用外部工具（如搜索引擎、计算器）来获取信息。
- **RAG Retrieval-Augmented Generation：检索增强生成**，即模型先从外部知识库中检索相关信息，然后结合这些信息生成答案。
- **WebGPT：OpenAI开发的一个大语言模型**，能够通过搜索引擎获取实时信息，并将其整合到回答中。

### Symbolic reasoning 符号推理

- **含义：** 模型通过符号和逻辑规则进行推理，以解决问题。
- **例子：** 推导数学定理、解决逻辑谜题。
- **CoT Chain-of-Thought：思维链**，即模型在推理过程中，会逐步生成中间步骤，最终得出结论。

### Mathematical reasoning 数学推理

- **含义：** 模型能够进行数学计算和证明。
- **例子：** 解方程、证明几何定理。
- **Tool use 工具使用**：与Knowledge-intensive QA类似，模型可以调用计算器等工具进行数学计算。
- **PoT Proof-of-the-Theorem：定理证明**，指模型能够证明数学定理。

### Commonsense 常识

- **含义：** 人类在日常生活中习得的、不言自明的知识。
- **例子：** “鸟会飞”、“水往低处流”。
- **RAG**

### Multi-hop knowledge-intensive 多跳知识密集型

- **含义：** 需要通过多个信息来源或推理步骤才能回答的问题。
- **例子：** “谁是诺贝尔文学奖得主中最年轻的人？”（需要先找到诺贝尔文学奖得主名单，再比较他们的年龄。）
- **Self-ask 自问自答**：模型在回答问题时，会主动提出一些子问题，并尝试回答这些子问题，以最终得出答案。
- **IRCOT Iterative Retrieval and Chain-of-Thought：迭代检索和思维链**，结合了RAG和CoT的优点，通过多次迭代检索和推理来回答复杂问题。



## abstract of Reasoning or Acting

![image-20241007121250762](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071212961.png)



ReAct = Reasoning + Acting

![image-20241007121705459](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071217638.png)



## ReAct is simple and intuitive to use

intuitive易懂的



**流程：**

人类输入：task

模型生成：thought + action

action被模型输入到外部环境中产生：observation

thought + action + observation 附加到模型的上下文：thought_2 + action_2

循环...



**例子：**

![image-20241007122452868](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071224076.png)

![image-20241007122514915](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071225114.png)

![image-20241007122729278](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071227473.png)

![image-20241007123157207](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071231450.png)

![image-20241007123306977](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071233266.png)

![image-20241007123433792](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071234030.png)



## Many tasks can be turned into text games

一个伟大的idea：将ReAct范式可以应用到其他领域

![image-20241007123735985](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071237146.png)

![image-20241007123751055](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071237268.png)

![image-20241007123806555](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071238750.png)

![image-20241007123823184](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071238298.png)

![image-20241007123908174](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071239289.png)

![image-20241007123922534](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071239661.png)

![image-20241007123935356](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071239467.png)

![image-20241007123946552](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071239684.png)

![image-20241007124004827](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071240958.png)

![image-20241007124022718](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071240871.png)

![image-20241007124039908](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071240065.png)



## ReAct Enables Systematic Exploration

![image-20241007124618915](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071246047.png)

![image-20241007124632071](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071246190.png)

![image-20241007124910384](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071249621.png)



## Reasoning agent: reasoning is an internal action for agents

![image-20241007125039088](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071250288.png)

![image-20241007125058962](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071250125.png)

![image-20241007125114936](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071251074.png)

![image-20241007125129554](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071251667.png)



## long-term memory

几个项目：

Reflexion: Language Agents with Verbal Reinforcement Learning

Also check: Self-refine, Self-debugging, etc.

VOYAGER: An Open-Ended Embodied Agent with Large Language Models

Generative Agent: Interactive Simulacra of Human Behavior

​	Generative Agent

​	Episodic memory of experience

​	Semantic memory of (reflective) knowledge

Cognitive architrctures for language agrnts (CoALA)

![image-20241007125730846](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071257006.png)

![image-20241007130429898](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071304073.png)

![image-20241007134456293](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071344466.png)

幻灯片的原文如下：

**Long-term memory**
- Read and write
- Stores experience, knowledge, skills, ...
- Persist over new experience

**Short-term memory**

**Code-based controller**
- Learn
- Retrieve
- Reason
- Action

**Cognitive architectures for language agents (CoALA)**
- Memory
- Action space
- Decision making

**Exercise questions**
- What distinguishes external environment vs internal memory?
- What distinguishes long vs short term memory?

---

**翻译：**

**长期记忆**
- 读写能力
- 储存经验、知识、技能等
- 在新体验中持续保存

**短期记忆**

**基于代码的控制器**
- 学习
- 检索
- 推理
- 行动

**语言代理的认知架构（CoALA）**
- 记忆
- 行动空间
- 决策制定

**练习问题**
- 外部环境与内部记忆的区别是什么？
- 长期记忆和短期记忆的区别是什么？

---

**解释：**

这张幻灯片描述了大语言模型（LLM）智能体的认知架构和其核心组件。

1. **长期记忆（Long-term memory）**：类似于人类的大脑，长期记忆部分能够读取和写入信息，储存模型的经验、知识和技能，并且在新的体验中持续保存。这意味着模型不仅能够记住过去的经验，还能在未来的任务中加以利用。

2. **短期记忆（Short-term memory）**：这是模型在短时间内处理信息和执行任务的能力。短期记忆一般用于处理当前任务的步骤和推理过程，如接受指令、思考、执行行动并观察结果。

3. **基于代码的控制器（Code-based controller）**：这是模型的核心逻辑控制单元，它管理学习、检索和推理过程，并且执行任务中的具体行动。通过观察外部环境和获取反馈，控制器能够不断调整模型的行为。

4. **语言代理的认知架构（CoALA）**：这是一个专门针对语言智能体设计的认知架构，主要包括记忆、行动空间和决策制定等模块。记忆模块存储信息，行动空间决定智能体能够执行的动作，而决策制定则是在不同选择之间进行权衡，决定智能体的行为。

5. **练习问题（Exercise questions）**：两个提出的问题用于帮助进一步思考语言智能体的认知架构。
   - 第一个问题是关于外部环境与内部记忆的区分，即智能体如何处理来自外部的感知数据与其内部储存的信息。
   - 第二个问题涉及长期记忆与短期记忆的差异，即如何在处理当前任务时有效利用长期存储的经验和短期存储的即时信息。



## Some lessons for research

* Simplicity and generality
* You need both...
    * Thinking in abstraction
    * Familiarity with tasks (not tasks-specific methods!)
* Learning history and other subjects helps!

* 简单性和通用性
* 你两者都需要。。。
    * 抽象思维
    * 熟悉任务（不是特定任务的方法！）
* 学习历史和其他科目会有所帮助！



## What's next

55:47

![image-20241007144501608](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071445779.png)

![image-20241007141335577](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071413728.png)

![image-20241007141347031](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071413155.png)

![image-20241007141402636](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071414763.png)

![image-20241007141417835](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071414977.png)

### Training

FireAct: Training LLM for agents

如何训练用于agents的模型？

在哪里获得数据？

现在，研究LLM和研究Agent的人仍然是分开的，这是不对的，模型需要针对Agent进行训练。就像GPU和DL，模型和agent之间的协同作用应该加强。

![image-20241007142635424](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071426680.png)



### Interface

Human-computer interface (HCI)

Agent-computer interface (ACI)

如何为我们的LLM agents构建一个环境？

idea: 如果无法优化agent，则可以优化环境

![image-20241007143440394](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410071434596.png)



### Robustness

如何确作品在现实世界中有效？



### Human

如何确保作品在现实世界中与人类工作？



### Benchmark

**基准测试**（benchmarking）是一种通过运行一系列标准测试和试验，来评估计算机程序、一组程序或其他操作的相对性能的方法。简单来说，就是用一个统一的标准去测量不同事物（如电脑、软件、算法）的性能，看看谁更好。

怎样建立良好的基准测试？

现有的基准测试和现实中人们真正关心的东西之间存在很大差异

还需要将更多现实世界的元素纳入基准测试中，且需要新的指标



# Lecture 3, Chi Wang and Jerry Liu

全篇提到的论文：

| 论文                                                         |      |      |
| :----------------------------------------------------------- | ---- | ---- |
| Zaharia et al. 2024. The Shift from Models to Compound AI Systems |      |      |
|                                                              |      |      |
|                                                              |      |      |

全篇提到的项目：

| 项目 |      |      |
| :--- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |



Agent AI Frameworks & AutoGen

智能体人工智能框架与自动生成



Agenda

Agentic AI Frameworks

AutoGen



What are future AI applications like?  未来的AI应用是什么样的？

How da we empower every developer to build them?  如何赋予每个开发人员构建它们的能力？



## What are future AI applications like? 

Genertive -> Agentic

Generate content like text & image -> Execute complex tasks on behalf of human

生成内容如文本和图像 -> 代表人类执行复杂任务
