# UCB-CS194/294-196-LLM-Agents-zh

这是 UC Berkeley CS 194/294-196 (LLM Agents) 的简体中文翻译以及笔记。正在更新中，还有很多很多需要完善，打算与课程同步更新，欢迎您的阅读。希望能够得到您宝贵的建议，谢谢。

同时也会在 [CSDN 专栏](https://blog.csdn.net/bbqbro/category_12800996.html)进行更新。

# 目录

- [预备知识](#预备知识)
- [Dawn Song 的开场白](#dawn-song-的开场白)
- [Lecture 1, Denny Zhou](#lecture-1-denny-zhou)
- [Lecture 2, Shunyu Yao](#lecture-2-shunyu-yao)
- [Lecture 3, Chi Wang and Jerry Liu](#lecture-3-chi-wang-and-jerry-liu)
- [Lecture 4, Burak Gokturk](#lecture-4-burak-gokturk)
- [Lecture 5, Omar Khattab](#lecture-5-omar-khattab)
- [Lecture 6, Graham Neubig](#lecture-6-graham-neubig)
- [未完待续](#未完待续)

# 预备知识

## 英文缩写&术语

| 英语                                                         | 简中                            | 补充                                                         |
| ------------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------ |
| Large Language Model (LLM)                                   | 大语言模型                      |                                                              |
| Artificial General Intelligence (AGI)                        | 通用人工智能                    | 一个远大的目标                                               |
| Agent                                                        | 智能体/代理                     |                                                              |
| Embody                                                       | 具身                            |                                                              |
| Multi-Agent System (MAS)                                     | 多智能体系统                    |                                                              |
| Token                                                        |                                 | 文本分割后得到的最小语义单位                                 |
| Prompt                                                       | 提示词                          | 我们向AI提出的问题或指令                                     |
| Reason                                                       | 推理                            | 模型根据已有的知识，通过逻辑的推导得出结论                   |
| align                                                        | 对齐                            | 确保大语言模型的行为与用户的意图或期望一致                   |
| Chain-of-Thought (CoT)                                       | 思维链                          | 让LLM通过(intermediate step)解决问题的技术                   |
| decode                                                       | 解码                            | 将模型生成的内部表示转化为人类可读的文本的过程               |
| Universal Self-Consistency (USC)                             | 通用自一致性                    |                                                              |
| Retrieval-Augmented Generation (RAG)                         | 检索增强生成                    | 在生成模型中引入检索机制，使得模型能够在生成文本之前从外部知识库中检索相关信息 |
| Reinforcement Learning (RL)                                  | 强化学习                        | 智能体通过与环境进行交互，根据得到的奖励或惩罚来调整自己的行为，最终目标是最大化累计奖励 |
| Supervised Fine-Tuning (SFT)                                 | 监督微调                        |                                                              |
| Bidirectional Encoder Representations from Transformers (BERT) | 基于Transformer的双向编码器表示 | 是由Google AI团队开发的一种预训练语言模型。双向编码器是一种在自然语言处理（NLP）中广泛使用的技术，它能够同时考虑一个词语在其上下文中前后两个方向的信息。 |
| Generative Pre-trained Transformer (GPT)                     | 生成式预训练 Transformer        |                                                              |
| Parameter-efficient Fine Tuning (PEFT)                       | 参数高效微调                    | 对模型参数进行精细的调整，而不是对所有参数进行大规模的更新   |
| Low-Rank Adaptation of Large Language Models (LoRA)          | 大语言模型的低秩适应            |                                                              |
| Natural Language Inference (NLI)                             | 自然语言推理                    | 根据给定的句子推断出逻辑关系                                 |



# Dawn Song 的开场白

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

| 非项目论文                                                   |      |      |
| :----------------------------------------------------------- | ---- | ---- |
| Zaharia et al. 2024. The Shift from Models to Compound AI Systems |      |      |
| Initial Prototype Flexible multi-agent conversation framework Code/function execution |      |      |
| StateFlow - Build State-Driven Workflows with Customized Speaker Selection in GroupChat |      |      |
|                                                              |      |      |



项目：

| **项目**         |      |      |
| ---------------- | ---- | ---- |
| AutoGen          |      |      |
| OpenAI Assistant |      |      |
| LlamaIndex       |      |      |
| LangChain        |      |      |
| Langraph         |      |      |
| CrewAI           |      |      |
| SciAgents        |      |      |
| Agent-E          |      |      |
| MemGPT           |      |      |
| AutoBuild        |      |      |
| LlamaParse       |      |      |



Agent AI Frameworks & AutoGen

智能体人工智能框架与自动生成



Agenda

1. Agentic AI Frameworks

2. AutoGen



What are future AI applications like?  未来的AI应用是什么样的？

How da we empower every developer to build them?  如何赋予每个开发人员构建它们的能力？



## What are future AI applications like? 

Genertive -> Agentic

Generate content like text & image -> Execute complex tasks on behalf of human

生成内容如文本和图像 -> 代表人类执行复杂任务



## Examples of Agentic AI

Personal assistants

Autonomous robots

Gameing agents

Science agents

Web agents

Software agents



## Key Benefits of Agentic AI

Use Interface

​	Natural interaction with human agency

Strong Capability

​	Operate with minimal human intervention

Useful Architecture

​	Intuitive programming paradigm

使用界面

​	与人类机构的自然互动

强大的能力

​	以最少的人为干预进行操作

有用的架构

​	直观的编程范式



![image-20241008160544103](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410081605398.png)

1. User -> Commander: User Quesion
2. Commander -> Writer: Question
3. Writer -> Commander: Code
4. Commander -> Safeguard: Code
5. Safeguard -> Commander: Clearance
6. Commander -> Writer: Log
7. Writer -> Commander: Answer
8. Commander -> User

4~6: Repeat until answering the user’s question or timeout



### 流程分析

1. **用户提问（User Question）**
   - **描述**：用户向系统提交一个问题。这个问题是交互的起点，系统需要对此作出反应。
   - **作用**：这是整个交互流程的第一步，决定了接下来的处理步骤。

2. **指挥者（Commander）**
   - **描述**：指挥者是整个流程的核心，负责协调各个组件之间的交互。
   - **作用**：它接收用户问题并管理后续的信息流动。

3. **将问题传递给编写者（Question）**
   - **描述**：指挥者把用户提出的问题传递给编写者。
   - **作用**：此步骤确保编写者能够获得问题的上下文，以便进行适当的处理。

4. **编写者生成代码（Code）**
   - **描述**：编写者根据接收到的问题生成相应的代码。
   - **作用**：这一步是关键，因为生成的代码将影响后续的安全性和准确性。

5. **指挥者审查代码（Code）**
   - **描述**：指挥者将生成的代码传递给安全保障系统进行审查。
   - **作用**：这一步确保代码在执行之前经过安全性检查，避免潜在风险。

6. **安全保障确认（Clearance）**
   - **描述**：安全保障系统检查代码，确保其无误并安全后，向指挥者发出许可。
   - **作用**：这一环节提供了一个安全网，以防止不安全或有害的代码被执行。

7. **记录日志（Log）**
   - **描述**：指挥者将信息记录在日志中，以便后续审查或调试。
   - **作用**：日志记录有助于追踪问题的来源，并为未来的优化提供参考。

8. **编写者返回答案（Answer）**
   - **描述**：编写者根据用户的问题和生成的代码，形成最终的答案。
   - **作用**：这是响应用户的关键环节，确保用户得到所需的信息。

9. **指挥者返回最终答案给用户（Final Answer）**
   - **描述**：指挥者将编写者生成的最终答案传回给用户。
   - **作用**：这是整个流程的结束，用户得到了他们所询问的信息。

这个流程通过多个组件的协作，确保用户的提问能够被准确、安全地处理和回答。指挥者在其中起到了协调者的角色，而编写者和安全保障系统则分别负责内容生成和安全审查。这样的设计不仅提高了系统的准确性，还增强了其安全性，避免了可能的风险。



## Agentic Programming

* Handle more complex tasks / Improve response quality
    * Improve over natural iteration
    * Divide & conquer
    * Grounding & validation
* Easy to understand, maintain, extend
    * Modular composition
    * Natural human participion
    * Fast & creative experimentation

**处理更复杂的任务 / 提升响应质量**
- 通过自然迭代进行改进
- 分而治之
- 以实际情况为基础进行验证

**易于理解、维护和扩展**
- 模块化组合
- 自然的人为参与
- 快速且具有创意的实验



![image-20241008212656599](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410082126740.png)

![image-20241008212717301](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410082127529.png)



## Agentic AI Framework Desiderata

- **Agentic AI Framework Desiderata**
- Intuitive unified agentic abstraction
- Flexible multi-agent orchestration
- Effective implementation of agentic design patterns
- Support diverse application needs

- **代理型AI框架的必要条件**
- 直观统一的代理抽象
- 灵活的多代理编排
- 有效实现代理设计模式
- 支持多样化的应用需求



这张图表描述了设计代理型AI框架时的关键需求（Desiderata），即满足该框架的一系列理想特性：

1. **直观统一的代理抽象（Intuitive unified agentic abstraction）**：
   - 该框架应该提供简洁、统一的抽象模型，便于开发者理解和使用代理系统。
   
2. **灵活的多代理编排（Flexible multi-agent orchestration）**：
   - 框架需能够协调多个代理的交互，具有高度的灵活性，以适应复杂的场景和任务。
   
3. **有效实现代理设计模式（Effective implementation of agentic design patterns）**：
   - 框架应能够有效地实现常见的代理设计模式，帮助开发者以模块化和结构化的方式构建系统。
   
4. **支持多样化的应用需求（Support diverse application needs）**：
   - 该框架应具备通用性，能够支持不同领域的应用需求，包括不同规模和复杂度的代理系统。

这几项需求确保代理型AI系统既能满足开发的灵活性和扩展性，又能提供适合不同场景的解决方案。



## Multi-Agent Orchestration



![image-20241008215025636](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410082150855.png)

**多智能体编排**是一种在人工智能领域，尤其是分布式系统中，协调多个智能体（Agent）共同完成复杂任务的技术。这些智能体可以是软件程序、机器人、虚拟角色等，它们具有自主性，能够感知环境、做出决策并执行行动。

**智能体（Agent）**：系统中的基本单位，具有感知、决策和行动能力。

**编排**：对多个智能体的行为进行协调和控制，以实现共同目标。



- **Multi-Agent Orchestration**
  - Static/dynamic
  - NL/PL
  - Context sharing/isolation
  - Cooperation/competition
  - Centralized/decentralized
  - Intervention/automation

- **多智能体编排**
  - 静态/动态
  - 自然语言/编程语言
  - 上下文共享/隔离
  - 协作/竞争
  - 集中化/去中心化
  - 干预/自动化



这张图表描述了多代理系统的各种编排方式，阐述了在设计多代理系统时可能遇到的不同选择和设计模式：

1. **静态/动态 (Static/dynamic)**：
   - 静态代理系统指的是角色和交互模式预先确定的系统，而动态系统可以根据环境或输入进行调整和改变。

2. **自然语言/编程语言 (NL/PL)**：
   - 多智能体系统可以使用自然语言（NL）与用户或其他代理进行交互，或者使用编程语言（PL）进行更结构化的通信。

3. **上下文共享/隔离 (Context sharing/isolation)**：
   - 智能体之间可以共享上下文信息（例如共享环境状态或任务信息），也可以保持隔离以确保任务独立性或信息隐私。

4. **协作/竞争 (Cooperation/competition)**：
   - 代理可以通过合作方式共同完成任务，或者通过竞争方式争夺资源或达成目标。

5. **集中化/去中心化 (Centralized/decentralized)**：
   - 集中化系统中存在一个中心控制代理来协调其他代理的行为，而去中心化系统则依赖于各代理的独立决策和行动。

6. **干预/自动化 (Intervention/automation)**：
   - 系统可以设计成允许人工干预，在必要时进行手动调整，也可以完全自动化，代理自主完成所有任务。

#### 图解部分：
- **上半部分**：
   - 显示了一个具有指挥者（Commander）的多代理系统，指挥者与多个代理互动，协调他们的任务和行为。
   
- **下半部分**：
   - 描绘了一个带有“经理（Manager）”的场景，经理通过广播与多个代理互动，并选择其中一个代理进行更直接的“对话（Speak）”。



## Agent Design Patterns

Conversation

Prompting & reasoning

Tool use

Planing

Integrating multiple models, modalities and memories

**Agent 设计模式**

对话

提示与推理

工具使用

规划

整合多种模型、模式和记忆



**Agent 设计模式**是一套用于设计和实现智能体的可复用解决方案。这些模式基于人工智能领域的最佳实践和既定原则，提供了一种结构化的方式来构建能够有效感知环境、推理并从经验中学习的智能体。

**第一性原理**，简单来说，就是从最基本的、不可再分的原理出发，通过逻辑推理和数学计算，来理解和预测复杂系统的行为。这种方法不依赖于经验公式或模型，而是直接从系统的基本构成要素和物理定律出发，逐层推导。



## Example of Agentic AI Frameworks

* AutoGen
    * Multi-agent conversation programming
        * Comprehensive & flexible
        * Integrable with other frameworks like OpenAI Assistant, LlamaIndex, LangChain
* Langchain-based
    * Langraph
        * Graph-based control flow
    * CrewAI
        * High-level static agent-task workflow

- **AutoGen**
  - 多代理对话编程
    - 全面且灵活
    - 可与其他框架集成，如OpenAI Assistant、LlamaIndex、LangChain
- **基于LangChain**
  - **Langraph**
    - 基于图的控制流
  - **CrewAI**
    - 高级静态代理任务工作流程



1. **AutoGen**
   - **多代理对话编程**：AutoGen提供了一种全面且灵活的方法来实现多代理系统中的对话编程。通过这种编程方法，多个代理可以通过对话协作来完成复杂任务，适用于需要多方合作或互动的应用场景。
   - **与其他框架集成**：AutoGen的一个显著特点是它能够与其他流行的AI框架集成，如OpenAI Assistant、LlamaIndex和LangChain。这种兼容性意味着开发者可以将AutoGen纳入现有系统中，实现更强大的多代理对话系统。

2. **基于LangChain**
   - **Langraph**：Langraph是一种基于图的控制流模型。这意味着任务或操作可以通过图结构来管理和控制，允许开发者定义任务的依赖关系和执行顺序。这种图结构的设计通常用于处理复杂的工作流或任务编排，尤其是在多步骤和多路径的任务处理中。
   - **CrewAI**：CrewAI提供了一个高级的静态代理任务工作流系统。这种工作流是静态的，意味着代理的任务分配和执行顺序是在开发时就预先设定好的，不会根据动态变化进行调整。CrewAI特别适合需要明确任务分配和执行次序的场景，确保代理在系统中的任务是有序且可控的。



以下是对 AutoGen、OpenAI Assistant、LlamaIndex、LangChain、Langraph 和 CrewAI 的解释：

### 1. **AutoGen**
AutoGen 是一种多代理对话编程框架，专注于通过协调多个人工智能代理来完成复杂任务。它支持多代理系统的构建，使多个代理可以协作、共享信息，并基于上下文来做出智能决策。AutoGen 特点是灵活性和广泛的兼容性，能够与其他 AI 框架（如 OpenAI Assistant、LlamaIndex 和 LangChain）集成，从而增强多代理的交互能力。

### 2. **OpenAI Assistant**
OpenAI Assistant 是 OpenAI 提供的智能对话系统，基于其强大的自然语言处理模型（如 GPT 系列）。它支持与人类进行自然对话，并且可以执行任务、回答问题、生成文本内容等。OpenAI Assistant 可以用于广泛的应用场景，如虚拟助手、客服、教育等。

### 3. **LlamaIndex**
LlamaIndex 是一个工具，旨在从大规模数据集（如文档、数据库或知识库）中构建索引，便于快速检索信息。它结合了自然语言处理和搜索技术，允许用户在庞大信息库中高效查询。LlamaIndex 的核心在于将结构化和非结构化数据转化为易于检索的格式，提升搜索速度和准确度。

### 4. **LangChain**
LangChain 是一个用于构建和管理语言模型的框架。它的主要目标是简化与大型语言模型（如 GPT）集成的开发过程，帮助开发者轻松创建支持复杂任务的语言应用。LangChain 特别侧重于语言模型与外部系统（如数据库、API）的交互，通过提供工具来控制任务流，处理多步骤工作流程。

### 5. **Langraph**
Langraph 是基于 **LangChain** 的扩展，采用图结构来管理任务流和控制流。在多步骤的复杂任务中，Langraph 使用图形化的方式表示不同任务节点之间的依赖关系，使开发者可以直观地设计任务的执行顺序和路径。它适用于需要并行执行和分支控制的复杂应用场景。

### 6. **CrewAI**
CrewAI 是一种专注于静态任务工作流程的高层次代理任务管理框架。它为代理任务提供了一个预定义的静态工作流，确定了任务分配和执行的固定路径。CrewAI 在任务管理和执行时更注重确定性，适用于那些不需要动态调整的多代理系统。通过这种框架，开发者可以确保任务执行的顺序和分配是可预测且稳定的。



## simply two steps

1、Define agents: Conversable & Customizable (Conversable agent)

2、Get them to talk: Conversation Programming

![image-20241009100956740](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091010068.png)

![image-20241009101755181](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091017505.png)

![image-20241009101847619](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091018851.png)

![image-20241009102431169](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091024411.png)

![image-20241009104329096](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091043294.png)

1. **Evaluation (评估)**

- **Agent-based evaluation tools (基于代理的评估工具)**
    - **Examples: AgentEval, AutoDefense, Observability**
    - **中文翻译**: 基于代理的评估工具
        - 示例：AgentEval、AutoDefense、Observability
    - **解释**：此部分描述了用于评估多代理系统的工具。这些工具主要用于衡量和评估不同代理的性能、安全性和可观察性。例如，AgentEval 可能是一个用于评估代理执行效果的系统，而 AutoDefense 则专注于自动化的防御策略，Observability 提供对系统行为的可见性和监控能力。

2. **Interface (界面)**

- **Lower the barrier of programming (降低编程的门槛)**
    - **Examples: AutoBuild, Composable Actor Platform**
    - **中文翻译**: 降低编程的门槛
        - 示例：AutoBuild、Composable Actor Platform
    - **解释**：这部分强调了如何降低开发人员使用和创建多代理系统的难度。AutoBuild 可能是一个自动化构建工具，使用户能够快速开发和部署多代理系统，而 Composable Actor Platform 则可能是一种模块化平台，允许用户通过简单的组件组合来创建代理，极大地降低了编程和开发的复杂性。

3. **Learning/Teaching/Optimization (学习/教学/优化)**

- **Agents made smarter (使代理更智能)**
    - **Examples: AgentOptimizer, EcoAssistant, Learn to Cooperate**
    - **中文翻译**: 使代理更智能
        - 示例：AgentOptimizer、EcoAssistant、Learn to Cooperate
    - **解释**：这一部分集中在如何通过学习、教学和优化来提升代理的智能水平。例如，AgentOptimizer 可能是用于优化代理策略的工具，而 EcoAssistant 则可能用于帮助代理优化在复杂环境中的资源使用。Learn to Cooperate 可能涉及多个代理之间的协作学习，使它们在合作任务中表现得更好。



这个图片展示了多代理系统的三个重要方面：**评估**、**界面** 和 **学习/教学/优化**。每个方面都列出了相关的工具示例，帮助开发者和研究人员更好地创建、评估和优化多代理系统。这些工具和平台共同作用，可以降低开发复杂性、提升系统性能，并使代理能够更智能地适应复杂的任务和环境。



![image-20241009105210863](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091052075.png)

#### 1. **How to design optimal multi-agent topology? (如何设计最优的多代理拓扑结构？)**
   - **Quality (质量)**
   - **Monetary Cost (货币成本)**
   - **Latency (延迟)**
   - **Manual Effort (人工投入)**
   - **中文翻译**：如何设计最优的多代理拓扑结构？
      - 质量、货币成本、延迟、人工投入
   - **解释**：这一部分提到了在设计多代理系统时需要考虑的几个关键因素。要设计一个最优的多代理拓扑结构，需要平衡系统的质量、经济成本、延迟（即代理之间的通信或执行任务的时间延迟）以及需要的人力投入。这些因素是评估系统效率和可行性的关键。

#### 2. **How to create highly capable agents? (如何创建高能力的代理？)**
   - **Reasoning (推理能力)**
   - **Planning (规划能力)**
   - **Modality (多模态处理)**
   - **Learning (学习能力)**
   - **中文翻译**：如何创建高能力的代理？
      - 推理、规划、多模态处理、学习
   - **解释**：为了打造强大的智能代理，系统需要具备一些核心能力。推理能力意味着代理能够理解和处理复杂的信息。规划能力是指代理能有效规划其任务和行动。多模态处理则意味着代理能够处理不同类型的数据和信息源（如图像、文本等）。学习能力意味着代理可以通过过去的经验或数据不断自我提升。

#### 3. **How to enable scale, safety and human agency? (如何实现规模化、安全性以及人类代理？)**
   - **Parallelization (并行化)**
   - **Resilience (弹性)**
   - **Intent (意图识别)**
   - **Teaching (教学能力)**
   - **中文翻译**：如何实现规模化、安全性和人类代理？
      - 并行化、弹性、意图识别、教学能力
   - **解释**：当涉及大规模多代理系统时，如何扩展并保持系统的安全性至关重要。并行化允许多个代理同时执行任务，从而提升系统的效率。弹性是系统在面对错误或故障时依然能够正常运行的能力。意图识别是系统能够理解和识别人类或其他代理意图的能力。而教学能力则指系统能够与人类或其他代理互动并传授其知识或技能的能力，这对于人机协作至关重要。



## Jerry Liu: Building an Multimodal Knowledge Assistant

LlamaIndex:

Build Production LLM APPsover Enterprise Data

LlamaIndex helps any developer build context-augmented LLM apps from prototype to production.

基于企业数据构建生产LLM应用程序

LlamaIdex帮助任何开发人员构建从原型到生产的上下文增强LLM应用程序。



## Build a Knowledge Assistant

Goal: Build an interface that can take in any tasks as input and give back an output.

Input forms: simple questions, complex questions, research tasks

Output forms: short answer, structured output, research report

目标：构建一个可以接收任何任务作为输入并返回输出的界面。

输入形式：简单问题、复杂问题、研究任务

输出形式：简答题、结构化输出、研究报告



## Knowledge Assistant with Basic RAG

![image-20241009111949750](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410091119038.png)



## Can we do more?

除了基本的RAG流程管道外，还能做什么？

There's many questions/tasks that naive RAG cant't give an answer to

* huallucinations
* Limited time savings
* Limited decision-making enhancement



## A Better Knowledge Assistant

1. High-quality Multimodel RAG
2. Complex output generation
3. Agentic reasoning over complex inputs
4. Towards a scalable, full-stack application

**成为更好的知识助理的四个重要要素**

1. 高质量多模态RAG
2. 可生成复杂输出
3. 能对复杂输入进行智能体推理
4. 能够成为可扩展的全栈APP



1 High-quality Multimodel RAG：

Advanced Data and Retrieval   高级数据和检索

Data -> Data Processing -> Index -> Agent -> Response   数据 -> 数据处理 -> 索引 -> 代理 -> 响应



2 Complex output generation:

Report Generation, Data Analysis, Action-Taking   报告生成、数据分析、行动采取。



3 Agentic reasoning over complex inputs:

Other Tools

Advanced RAG and Retrieval Tool -> Agent (Tool Use, Memory, Query Planning, Reflection) -> Response

高级RAG和检索工具 -> 代理（工具使用、记忆、查询规划、反思） -> 响应

在多模态RAG pipeline中加入Agent推理层



4 Towards a scalable, full-stack application

不重要



# Lecture 4, Burak Gokturk

LLM Agents

Enterprise Trends for Generative AI   生成式 AI 领域的企业动向

Key Blocks to buid successful agents   成果构建智能体的关键模块

What's next?



## 提到的项目

| 项目               |      |      |
| ------------------ | ---- | ---- |
| Gemini             |      |      |
| Claude - Anthropic |      |      |
| glean              |      |      |
| Vertex AI          |      |      |



## Key Trends in Generative AI

- **Some observations**
   - In recent years, ML has completely changed our expectations of what is possible with computers
   - Increasing scale (compute, data, model size) delivers better results
   - The kinds of computations we want to run and the hardware on which we run them is changing dramatically

- **一些观察**
   - 近年来，机器学习彻底改变了我们对计算机能力的预期
   - 增加规模（计算能力、数据、模型大小）可以带来更好的结果
   - 我们希望运行的计算类型以及运行它们的硬件正在发生巨大变化



1. **机器学习改变预期**：近几年，随着机器学习的迅猛发展，计算机可以完成的任务范围和能力已经超出了以往的预期。机器学习使得计算机不仅限于传统任务（如存储、基本运算），而是能够处理复杂的模式识别、预测分析等任务，从而在很多领域取得突破。

2. **扩大规模带来更好结果**：随着计算能力的提升、数据量的增加以及模型规模的扩大，机器学习模型的性能表现显著提高。这意味着更强大的计算设备、更大规模的数据集和更复杂的模型结构可以带来更加精确和有效的结果。

3. **计算类型和硬件的变化**：随着计算需求的改变，所需的计算类型和支持这些计算的硬件也在快速演变。传统的计算模型和硬件可能无法适应现代机器学习的需求，因此新的硬件架构（如GPU、TPU等）和更高效的计算方式（如并行计算）正在逐渐成为主流。

这些观察强调了机器学习在现代计算中的重要性以及为实现更好结果所需的基础设施和资源的快速变化。



## Transformers + autoregressive training + massive

![image-20241012131320972](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410121313260.png)

- **Transformers + autoregressive training + massive data**
    - **Backbone architecture**: Diagram of a transformer model structure with multiple layers of multi-head attention and feed-forward layers.
    - **Next token prediction**: Example showing how the foundation model predicts the next word in a sentence, starting with "I went to the..." and predicting "conference".
    - **Pre-training on trillions of tokens**: Visualization showing sentences like "The cat sat on the mat", "The teacher read a book", and "I love to dance", illustrating how models are pre-trained on massive datasets.

- **Transformer模型 + 自回归训练 + 海量数据**
    - **基础架构**：展示了一个Transformer模型的结构图，包含多层的多头注意力机制和前馈神经网络层。
    - **下一个词的预测**：通过例子展示了基础模型如何预测句子的下一个词，从 "I went to the..." 预测到 "conference"。
    - **在数万亿的词标上进行预训练**：通过可视化展示了句子，如 "The cat sat on the mat"（猫坐在垫子上），"The teacher read a book"（老师读了一本书），以及 "I love to dance"（我喜欢跳舞），表明模型是在大量数据集上进行预训练的。



1. **Transformer模型**：Transformer是一种深度学习模型结构，特别适用于自然语言处理任务。图中的架构展示了Transformer模型的典型组件，包括多头注意力机制和前馈神经网络层。这种结构允许模型在不同的输入位置之间创建复杂的依赖关系，使得其在处理长序列时表现非常优越。

2. **自回归训练**：自回归模型是一种序列模型，它根据前一个时间步或前一段文本的输出，来预测下一个时间步或下一个单词。在图片中，模型根据 "I went to the..." 来预测下一个单词 "conference"，说明其利用了上下文信息来进行预测。

3. **海量数据预训练**：模型的性能在很大程度上取决于预训练阶段的数据量。图片中显示了模型在海量语料库上进行训练的过程（数万亿个词标）。通过在如此大规模的数据上进行训练，模型可以捕获语言中的复杂模式和关联，提高其在各种自然语言任务中的表现。

总结来看，这张图片展示了Transformer模型的基础架构、自回归训练方式，以及如何利用大规模数据进行预训练，从而使模型能够在自然语言处理中进行准确的预测。



## Towards useful AI agents

![image-20241012190847392](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410121908242.png)

更实用的是 RLHF



- **Towards useful AI agents**
  - **Supervised Fine-Tuning (SFT)**: 
    - **Prompt**: "I have pains in my lower back while sleeping. What could be causing this?"
    - **Output (expert response)**: "You might be experiencing a lower back strain, typically caused by lifting heavy objects or abrupt movements. This can lead to sharp pain in the lower back, especially when moving or lifting..."
  - **Reinforcement Learning from Human Feedback (RLHF)**:
    - Diagram showing two paths: 
      - One through the **SFT model**
      - Another through the **RL-tuned model**, then **reward model**, and finally reaching the **RL objective**.

- **朝着有用的AI代理迈进**
  - **监督微调 (SFT)**:
    - **提示**: “我在睡觉时感到下背部疼痛，可能是什么原因导致的？”
    - **输出 (专家回应)**: “您可能经历了下背部拉伤，通常由提举重物或突然的动作引起。这可能会导致下背部的剧烈疼痛，尤其是在移动或提举时……”
  - **基于人类反馈的强化学习 (RLHF)**：
    - 图示展示了两条路径：
      - 一条经过 **SFT模型**
      - 另一条经过 **RL微调模型**，接着是 **奖励模型**，最终到达 **强化学习目标 (RL objective)**。

1. **监督微调 (SFT)**：
   - SFT 是一种监督学习的方法，模型通过标注好的数据进行训练，提供具有专家水平的响应。图中以下背部疼痛为例，展示了如何根据给定的提示生成一个专家级别的输出。SFT模型会直接生成与问题相关的回答，基于之前训练的数据集的知识。
   
2. **基于人类反馈的强化学习 (RLHF)**：
   - RLHF 是一种将强化学习与人类反馈相结合的模型优化方法。在RLHF中，模型先通过SFT模型生成初步响应，然后通过“RL微调模型”和“奖励模型”来评估生成的回答是否符合预期。**奖励模型根据人类反馈为生成的输出分配奖励或惩罚**，并最终通过强化学习目标进行优化，使模型的表现更加符合人类期望。

3. **两者的比较**：
   - SFT 专注于通过静态数据提供直接的、预先训练好的专家答案。
   - RLHF 则允许模型通过人类反馈进行进一步优化，动态调整模型的输出，使其更加符合使用者的需求。RLHF模型在RL目标优化的基础上，能够生成更符合预期的输出结果，并提升其适用性。

总结来看，这张图片展示了两种AI模型训练方法：监督微调 (SFT) 和基于人类反馈的强化学习 (RLHF)。SFT提供了直接的响应，而RLHF通过奖励机制进一步优化了模型的表现，向更有用的AI代理迈进。



### SFT（Supervised Fine-Tuning）是什么？

**监督微调**（SFT）是机器学习中一种常见的方法，属于**监督学习**的范畴。具体来说，SFT是在预训练模型的基础上，使用标注的高质量数据进行进一步训练，以便让模型在特定的任务上表现更好。其主要特点包括：

1. **监督学习**：SFT使用的是带标签的数据集。也就是说，训练数据中每一个输入都有明确的目标输出（即答案）。通过这种方式，模型可以“学习”如何在相同类型的任务中做出准确的预测或生成合理的响应。
  
2. **微调**：与从头开始训练模型不同，SFT通常是在已经经过大规模预训练的模型基础上进行。这种方法利用了模型已经学到的一般知识，并通过额外的训练使其能够在特定领域或任务上表现得更好。

3. **举例**：在自然语言处理（NLP）领域，预训练的大语言模型（如GPT）可以通过SFT被微调，来回答医学问题或法律问题等特定领域的查询。例如，给定一个医疗问题，微调后的模型能生成符合领域知识的回答。

### RLHF（Reinforcement Learning from Human Feedback）是什么？

**基于人类反馈的强化学习**（Reinforcement Learning from Human Feedback, RLHF）是结合了**强化学习**（Reinforcement Learning, RL）和人类反馈的一种机器学习方法。它的核心思想是通过人类提供的反馈来引导模型的优化，使其表现更加符合人类的期望。

1. **强化学习**：不同于监督学习，强化学习的目标是通过试错过程来优化模型的策略。模型在给定任务中做出一系列决策，然后通过一个“奖励机制”来衡量这些决策的好坏。RL的目标是最大化长期的奖励。
  
2. **人类反馈**：在RLHF中，人类反馈取代了传统强化学习中的自动奖励机制。具体来说，人类评估模型的输出，并根据输出的好坏给出“奖励”或“惩罚”。通过这些反馈，模型能够学习生成更符合人类期望的输出。

3. **RLHF的过程**：
   - 首先，模型通过监督微调（SFT）完成初步训练。
   - 然后，模型生成的输出会被人类评估，反馈用于训练一个奖励模型。
   - 最后，通过强化学习算法，模型会根据奖励模型的反馈不断调整，以生成更优质的输出。

4. **举例**：在对话系统或生成式AI中，RLHF可以用来改进模型的生成结果。如果模型生成了不合适或偏离预期的回答，人类反馈可以帮助模型纠正这种偏差，逐步生成更符合实际需求的回答。

### 总结

- **SFT**：主要用于在预训练模型基础上，通过带标签的高质量数据进一步优化模型的性能，适合任务是预定义、明确答案的场景。
  
- **RLHF**：通过引入人类的反馈来指导模型的优化，特别适合处理开放性任务或模型输出不确定的情况。通过强化学习的方式，模型能够逐步学习符合人类偏好的输出。



## Enterprise Trends: Trend 1 - The obvious: AI is moving so much faster

Why?

* The amount of data needed has come down
* Anyone can develop AI



## Trend 2 - Technical Treads

![image-20241012202029952](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410122020271.png)

![image-20241012202531572](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410122025835.png)

![image-20241012203107068](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410122031771.png)

Where are we headed?



Separate models for different tasks --> Single model that can generalize across millions of tasks、推理能力

针对不同任务的独立模型 --> 能够泛化处理数百万种任务的单一模型



Dense models --> Efficient sparse models

稠密模型 --> 高效稀疏模型

**Dense models:** 稠密模型是指模型中的所有参数都是非零的。

**Efficient sparse models:** 高效稀疏模型是指模型中只有部分参数是非零的，这可以减少模型的计算量和存储空间，提高模型的效率。



Single modality models --> Models that deal with many modalities

单模态 --> 多模态



## Trend 3 - It is the choice of the platform that matters

重要的是平台的选择



LMSys Chat Leaderbord Ranks

LMSys Chat Leaderbord Ranks，也就是LMSys聊天机器人竞技场排行榜，是一个用于评估大型语言模型（LLM）性能的众包开放平台。



## Key success factor for generative AI

![image-20241013221526212](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410132215672.png)

**Key success factors for generative AI**
1. **Access to a broad set of models**
   - …so you can find the best model for their use case and budget
2. **Ability to customize models with your data**
   - …so you can improve the quality, latency, and performance of the model
3. **A platform for managing models in production**
   - …so you can deploy and manage models in a scalable way
4. **Choice and flexibility at every level**
   - …so you can avoid vendor lock-in and use the best tools for the job

**生成式 AI 成功的关键因素**

1. **访问广泛的模型集**
   - …这样你可以根据具体的用例和预算找到最合适的模型
2. **使用你的数据定制模型的能力**
   - …这样你可以提高模型的质量、延迟和性能
3. **用于生产环境中管理模型的平台**
   - …这样你可以以可扩展的方式部署和管理模型
4. **每个层级的选择和灵活性**
   - …这样你可以避免供应商锁定，并使用最适合任务的工具



这张图片描述了生成式AI应用中取得成功的四个关键因素：

1. **访问广泛的模型集**：这意味着企业或开发者需要能够访问各种各样的AI模型，以便根据其特定的需求、用例和预算选择最合适的模型。这可以帮助用户优化资源使用，并确保他们选择的模型是最佳解决方案。

2. **使用数据定制模型的能力**：这表明，生成式AI模型在实际应用中表现出色的一个关键点是，能够根据特定的数据进行定制。定制可以提升模型的准确性、响应速度（延迟）和整体性能，这对于应用到特定场景中的生成式AI尤为重要。

3. **生产环境中管理模型的平台**：生成式AI不仅需要在实验室中运行良好，还需要在实际生产环境中进行管理和部署。一个好的平台可以帮助团队高效地管理模型的生命周期，包括从训练到部署再到维护，且能够支持模型的可扩展性。

4. **每个层级的选择和灵活性**：避免被某个供应商或特定工具锁定，意味着企业可以自由选择最适合特定任务的工具或模型。这种灵活性可以确保企业在技术演变或新需求出现时迅速适应和调整，从而保持竞争优势。



## Trend 4 - Cost of API calls is approaching 0

API调用的成本接近0



## Trend 5 - Search

Another big realization

LLM and Search need to come together.



## Trend 6 - Enterprise Search/Assistant

Enterprise Learnings with wheir AI investments   企业通过AI投资过程学习



全球芯片依然短缺



![image-20241013233759539](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410132337785.png)



## Agumentation tools to enhance capabilities of foundation models

增强基础模型能力的扩展工具

![image-20241013234604311](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410132346711.png)

**Augmentation tools to enhance capabilities of foundation models**
1. **Tuning/Distillation**
   - Customize based on specific data and use case
   - Create a smaller model for cost/latency purposes
2. **Grounding**
   - Combine with search to make it factual
3. **Extensions/Function Calling**
   - Function calling to be able to make LLMs on areas where they perform poorly

**增强基础模型能力的扩展工具**

1. **微调/蒸馏**
   - 基于特定数据和用例进行定制
   - 创建一个较小的模型以降低成本/延迟
2. **基础对接**
   - 与搜索结合，使其更加事实化
3. **扩展/函数调用**
   - 使用函数调用来增强大型语言模型（LLMs）在其表现不佳的领域中的能力



这张图片介绍了三种用于增强基础模型能力的工具：

1. **微调/蒸馏**：这是通过使用特定的数据和用例来定制模型的过程。通过这样的定制，可以生成一个体积较小的模型，达到降低计算成本和延迟的目的。蒸馏是一种模型压缩技术，将大模型的知识转移到较小的模型中，从而保持高性能的同时减少资源消耗。

2. **基础对接**：这是通过将生成模型与搜索功能相结合，使生成的内容更加符合事实。这个方法有助于避免生成虚假的或不准确的信息，确保输出的可靠性和准确性。

3. **扩展/函数调用**：这是通过添加函数调用功能，提升大型语言模型在其弱项领域的表现。通过调用特定的函数或工具，可以弥补模型在某些任务中的不足，比如执行复杂计算或访问外部数据库的能力。

这些工具帮助提升基础模型的性能，使其更加高效、精确和灵活。



**模型蒸馏**（Model Distillation）是一种机器学习技术，它的核心思想是将一个复杂的大模型（教师模型）的知识“蒸馏”到一个更小、更简单的模型（学生模型）中。就好比一位经验丰富的老师在教导一位初学者一样，教师模型将自己学到的知识和经验传授给学生模型，让学生模型也能具备较强的预测能力。

为什么需要模型蒸馏？

- **模型压缩：** 大模型通常参数量巨大，占用大量计算资源。通过蒸馏，我们可以获得一个更小的模型，部署到资源有限的设备上，例如移动端或嵌入式系统。
- **加速推理：** 小模型的计算量更小，推理速度更快，可以提高模型的实时响应能力。
- **提升泛化能力：** 在某些情况下，小模型在泛化能力方面甚至可以超越教师模型。

模型蒸馏的原理

1. **教师模型训练：** 首先，我们训练一个大型的、性能优异的教师模型。
2. 知识蒸馏：
    - **软标签：** 将教师模型的输出概率分布（软标签）作为学生模型的学习目标。相比于传统的硬标签（one-hot编码），软标签包含了更多的信息，可以帮助学生模型更好地学习。
    - **损失函数：** 除了传统的交叉熵损失函数，我们还可以引入额外的损失函数来约束学生模型，使其更好地模仿教师模型的行为。
3. **学生模型训练：** 使用教师模型的软标签和额外的损失函数来训练学生模型。



**模型微调（Fine-tuning）** 就像是对一个已经训练好的大模型进行个性化定制的过程。想象一下，你买了一件成衣，虽然款式不错，但总觉得不太合身。于是，你找裁缝进行一些修改，让衣服更贴合你的身材。模型微调也是如此，它让预训练的大模型能够更好地适应你的特定任务和数据。

为什么需要模型微调？

- **特定任务适配：** 预训练的大模型通常是在海量通用数据上训练的，而你的任务可能非常具体，比如医疗诊断、金融预测等。微调可以帮助模型更好地理解和处理这些特定领域的数据。
- **提升性能：** 通过微调，模型可以在特定任务上取得更好的性能，例如更高的准确率、更低的误差。
- **减少训练时间：** 从头开始训练一个大模型需要大量的计算资源和时间。而微调只需要在预训练模型的基础上进行少量调整，可以大大缩短训练时间。

模型微调的原理

1. **预训练模型：** 首先，我们需要一个在海量数据上预训练好的大模型。这个模型已经具备了强大的语言理解能力或图像识别能力。
2. **特定数据集：** 然后，我们准备一个与我们的任务相关的小数据集。这个数据集应该包含大量的标注数据，用来指导模型进行微调。
3. **微调过程：** 将预训练模型的最后一层或几层替换或调整，使其适应新的任务。然后，使用特定数据集对模型进行训练，更新模型的参数。



## Key Components of Customization Agent Builder *

![image-20241014000416337](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410140004629.png)

- **Title:** Key Components of Customization Agent Builder
- **Four components listed:**
  1. Fine Tuning
  2. Distillation
  3. Grounding
  4. Function Calling

- **标题:** 定制代理构建器的关键组件
- **四个列出的组件:**
  1. 微调
  2. 蒸馏
  3. 基础
  4. 函数调用



1. **Fine Tuning (微调):**
   
   微调指的是使用较小的特定领域数据集对预训练模型进行调整。通过这种方式，模型能够在保留广泛训练能力的同时，专注于某些特定任务。
   
2. **Distillation (蒸馏):**
   
   蒸馏是一种方法，通过让一个较小的模型模仿较大的模型，提取知识。这有助于创建更高效的模型，同时尽量保持精度。
   
3. **Grounding (基础):**
   
   基础指的是将AI模型的输出与现实世界的知识或外部资源连接起来，确保模型的回答在实际应用中准确且相关。
   
4. **Function Calling (函数调用):**
   
   函数调用指的是模型调用外部函数或API获取信息或执行操作的能力，使AI能够更具动态性和处理复杂操作的能力。



## Tune and customize with your data

![image-20241014000747036](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410140007357.png)

这张图片呈现了一个大模型微调（fine-tuning）的连续谱，从简单且成本效益高的方式到复杂且成本更高的方式。它将微调方法分为四个主要阶段：

- **Prompt design（提示设计）**：这是最简单的方式，通过设计合适的提示词来引导模型生成你想要的输出。
- **Supervised tuning with distillation（监督微调与蒸馏）**：在这个阶段，模型通过监督学习的方式在特定数据集上进行微调，同时利用蒸馏技术来加速训练过程。
- **Reinforcement learning with human feedback（基于人类反馈的强化学习）**：这个阶段引入人类反馈，通过强化学习的方式来优化模型，使其更符合人类的期望。
- **Full fine tuning（全参数微调）**：这是最复杂也是成本最高的方式，对模型的所有参数进行微调，以获得最佳性能。



| 英文原文                                   | 中文翻译               |
| ------------------------------------------ | ---------------------- |
| Tune and customize with your data          | 用你的数据微调和定制   |
| Simple, cost efficient                     | 简单，成本效益高       |
| Complex, more expensive                    | 复杂，成本更高         |
| Prompt design                              | 提示设计               |
| Supervised tuning                          | 监督微调               |
| Distillation                               | 蒸馏                   |
| Reinforcement learning with human feedback | 基于人类反馈的强化学习 |
| Full fine tuning                           | 全参数微调             |



- **提示设计（Prompt design）**：通过精心设计的提示词，可以引导模型生成特定类型的文本。这种方法简单易行，但对提示词的质量要求较高。
- **监督微调（Supervised tuning）**：通过提供大量的标注数据，让模型学习到输入和输出之间的映射关系。蒸馏技术可以加速这个过程，通过将一个大型模型的知识迁移到一个更小的模型上。
- **基于人类反馈的强化学习（Reinforcement learning with human feedback）**：这种方法将人类的反馈作为奖励信号，通过强化学习的方式来优化模型。这种方法可以使模型更符合人类的价值观和偏好。
- **全参数微调（Full fine tuning）**：对模型的所有参数进行调整，以获得最佳性能。这种方法通常需要大量的计算资源和数据，但可以取得最好的效果。



## Popular model adaptation approaches and model goals

![image-20241016181713643](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410161817997.png)

**Popular model adaptation approaches and goals**

1. **Fine Tuning**
   - Pre-trained Model: Tunable
   - Input Text
2. **Prompt Design**
   - Pre-trained Model: Frozen
   - Engineered Prompt
   - Input Text
3. **Prompt Tuning**
   - Pre-trained Model: Frozen
   - Tunable Soft Prompt
   - Input Text
4. **Distillation**
   - Pre-trained Model: Prompt Tuned Teacher
   - Student

Table details:

- **How does it work?**
   - Fine-tuning: Regular full model fine-tuning.
   - Prompt design: No training, just craft the input.
   - Prompt tuning: Train only 1-100 soft tokens.
   - Distillation: Distill a large model to a smaller student model.
   
- **Training data**
   - Fine-tuning: 10k-100k.
   - Prompt design: 1-10 examples (2048 tokens).
   - Prompt tuning: 100-1k examples.
   - Distillation: 100-1k labeled, 100k-100M unlabeled.
   
- **Training cost**
   - Fine-tuning: Prohibitively expensive.
   - Prompt design: Zero.
   - Prompt tuning: Average cost.
   - Distillation: Expensive.
   
- **Training time**
   - Fine-tuning: Weeks/Months.
   - Prompt design: Zero.
   - Prompt tuning: Tens of minutes.
   - Distillation: Days.
   
- **Quality**
   - Fine-tuning: Very high.
   - Prompt design: High.
   - Prompt tuning: Very high.
   - Distillation: Very high (slightly lower than teacher).
   
- **Inference cost**
   - Fine-tuning: High.
   - Prompt design: Zero-shot/Few-shot higher (2x to 5x higher).
   - Prompt tuning: High (slightly higher than fine-tuned model).
   - Distillation: Very low.
   
- **LLM size**
   - Fine-tuning: All sizes; large models (>62B).
   - Prompt design: Large models (62B+).
   - Prompt tuning: Moderate size (e.g., T5-3B).
   - Distillation: Large to small models (e.g., 64B to 1B or smaller).



**流行的模型自适应方法及目标**
1. **微调**
   - 预训练模型：可调
   - 输入文本
2. **提示设计**
   - 预训练模型：冻结
   - 工程化提示
   - 输入文本
3. **提示微调**
   - 预训练模型：冻结
   - 可调软提示
   - 输入文本
4. **蒸馏**
   - 预训练模型：提示微调的教师模型
   - 学生模型

表格细节：

- **如何工作？**
   - 微调：常规的全模型微调。
   - 提示设计：无需训练，只需设计输入。
   - 提示微调：只训练1-100个软标记。
   - 蒸馏：将大模型蒸馏成小模型。

- **训练数据**
   - 微调：10k-100k。
   - 提示设计：1-10个例子（2048个token）。
   - 提示微调：100-1k个例子。
   - 蒸馏：100-1k个标注数据，100k-1亿个未标注数据。

- **训练成本**
   - 微调：成本极高。
   - 提示设计：零。
   - 提示微调：平均成本。
   - 蒸馏：昂贵。

- **训练时间**
   - 微调：数周/月。
   - 提示设计：零。
   - 提示微调：数分钟。
   - 蒸馏：数天。

- **质量**
   - 微调：非常高。
   - 提示设计：高。
   - 提示微调：非常高。
   - 蒸馏：非常高（略低于教师模型）。

- **推理成本**
   - 微调：高。
   - 提示设计：零样本/少样本情况下推理成本更高（2到5倍）。
   - 提示微调：高（略高于微调模型）。
   - 蒸馏：非常低。

- **大语言模型（LLM）大小**
   - 微调：适用于所有模型，尤其是大模型（大于62B参数）。
   - 提示设计：大模型（62B+）。
   - 提示微调：中等规模（例如T5-3B）。
   - 蒸馏：从大模型蒸馏成小模型（如64B到1B或更小）。



这张图片展示了四种模型自适应的方法：

1. **微调**：通过使用大量数据对整个模型进行全量微调。这种方法适合大型模型，但训练成本高、时间长，推理时的成本也较高。然而，这种方法的输出质量通常很高，适用于需要极高准确性的场景。

2. **提示设计**：不需要训练模型，而是通过工程化的提示设计来输入特定的例子。这种方法非常适合大模型，成本低，但依赖于高质量的提示设计。

3. **提示微调**：只调整模型中的软提示部分，而保持模型的其他部分冻结。相比于全模型微调，它的训练成本较低，时间短，但仍然能达到非常高的性能，适合中等规模的模型。

4. **蒸馏**：将一个经过提示微调的大模型的知识蒸馏到一个较小的学生模型中。这样可以降低推理成本，并将大模型的能力压缩到一个更小的、性能较低但成本更低的模型中。

这些方法的选择取决于应用场景、资源限制和需要的模型规模。



## Conventional Fine-tuning

传统微调

![image-20241016182019669](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410161820064.png)

**Conventional Fine-tuning**
- **Basic steps**:
  - Get a pretrained model checkpoint (e.g., BERT)
  - Have a new dataset/task
  - Do supervised learning on new dataset and update the weights of the new model
- **Requires**:
  - Modest amount of compute (e.g., xxx chips for a few days for a 340B model)
  - In-depth knowledge of the model architecture

The diagram shows an example of fine-tuning using BERT, where specific datasets (like MNLI, NER, SQuAD) are used for tasks, and model weights are adjusted accordingly to perform well on those tasks.



**传统微调**
- **基本步骤**：
  - 获取一个预训练模型的检查点（例如 BERT）
  - 使用一个新的数据集/任务
  - 在新的数据集上进行监督学习，并更新新模型的权重
- **需求**：
  - 适量的计算资源（例如，使用一些芯片来运行几天以训练一个340B的模型）
  - 对模型架构的深入了解

该图展示了一个使用BERT进行微调的例子，具体数据集（如MNLI、NER、SQuAD）用于不同任务，并通过调整模型权重使其在这些任务上表现良好。



这张图片展示了传统微调（Fine-tuning）的流程。传统微调的主要目标是利用已经经过预训练的模型（如BERT）来处理新的任务或数据集。预训练模型已经在大量的通用数据上进行了初步训练，它掌握了一些基本的语言模式和知识。通过微调，模型能够通过监督学习在一个特定任务（如自然语言推理、命名实体识别、问答任务等）上进行专门训练，适应具体的应用场景。

微调的基本流程可以分为三个步骤：
1. 获取预训练模型的检查点，作为微调的基础。
2. 使用新的数据集或任务进行训练。
3. 通过监督学习，更新模型的权重，使其能够在新任务上表现得更好。

微调通常需要一定量的计算资源和对模型架构的深入了解。模型的规模越大，训练所需的计算资源越多。例如，使用具有数十亿参数的模型可能需要数天的时间来完成训练。

微调后的模型能够专注于特定任务，并显著提高在这些任务上的表现。



**微调** 是在深度学习中，特别是自然语言处理领域，一种非常常见且有效的技术。它指的是在预训练模型的基础上，使用特定任务的数据集进行进一步的训练，以使模型更好地适应新的任务。



BERT是**Bidirectional Encoder Representations from Transformers**的缩写，中文译为“基于Transformer的双向编码器表示”。它是由Google AI团队开发的一种预训练语言模型，在自然语言处理（NLP）领域掀起了巨大的波澜。

‌**BERT和GPT在自然语言处理领域各有优势，目前并没有被淘汰。**‌ BERT（Bidirectional Encoder Representations from Transformers）是由‌[谷歌](https://www.baidu.com/s?wd=谷歌&usm=4&ie=utf-8&rsv_pq=e6d27da600471cb7&oq=bert被gpt淘汰了吗&rsv_t=4df69TCIYsO6WZjoEiaK8Z2yVkFyBNj86EDB5%2FH%2FDrAbBhHSV8Yrh%2B0rLIw&sa=re_dqa_generate)开发的一种预训练模型，它通过双向训练的方式，能够更好地理解上下文信息，因此在理解文本的语义和含义方面表现出色。BERT在处理需要深入理解文本内容的任务时，如情感分析、文本分类等，表现出色。GPT（Generative Pre-trained Transformer）则是由‌[OpenAI](https://www.baidu.com/s?wd=OpenAI&usm=4&ie=utf-8&rsv_pq=e6d27da600471cb7&oq=bert被gpt淘汰了吗&rsv_t=899124ZhA0PTDVWq6rtUHSOQds6WXFnGXMvLE5AQzdbnIiSwc%2FgGc0B6IzM&sa=re_dqa_generate)开发的一种生成式预训练模型，它通过单向训练的方式，能够生成连贯的文本。GPT在生成式任务中表现出色，如文本生成、对话系统等。由于其强大的生成能力，GPT在需要创造性输出的任务中非常有用。



**双向编码器**是一种在自然语言处理（NLP）中广泛使用的技术，它能够同时考虑一个词语在其上下文中前后两个方向的信息。简单来说，就是让模型在处理某个词语时，不仅能看到它前面的词语，还能看到它后面的词语，从而更准确地理解这个词语的含义。



## Concentional Prompt Tuning

传统的提示调优

![image-20241016183022732](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410161830994.png)

**Conventional Prompt Tuning**

- **Basic steps**:
  - Freeze the backbone model.
  - Prepend a soft prompt (e.g., learnable) to the input.
  - Only optimize the prompt to adapt to downstream tasks.

In the diagram, different task prompts are combined into a mixed-task batch, and only the prompt parameters (with 82K parameters each) are tuned while the large pre-trained model (with 11 billion parameters) remains frozen.



**传统的提示调优**

- **基本步骤**：
  - 冻结主干模型。
  - 在输入前添加一个软提示（例如，可学习的提示）。
  - 仅优化提示，以适应下游任务。

在图表中，不同的任务提示被组合成一个混合任务批次，只有提示参数（每个提示有82K个参数）被调整，而大型预训练模型（拥有110亿个参数）保持冻结状态。



这张图片展示了提示调优（Prompt Tuning）的基本流程。在提示调优中，预训练的大型语言模型（例如，包含110亿参数的模型）不需要更新其权重，而是通过引入一个软提示进行微调。这个软提示会作为输入的一部分，在与任务相关的数据集上进行训练。

提示调优的关键步骤如下：
1. 冻结预训练模型的所有参数，使其保持原有状态，不再参与训练。
2. 为每个具体任务创建一个软提示（Soft Prompt），并将其附加在模型输入之前。这个提示是可学习的，意味着可以在训练过程中进行调整。
3. 仅对软提示的参数进行优化，以便模型能够更好地执行特定的下游任务（如分类、生成等任务）。

提示调优的优势在于，它极大地减少了需要训练的参数数量（只需调整提示的82K参数，而不是整个模型的110亿参数），因此计算资源和时间成本较低。这种方法在处理不同任务时能够提供高效的调整方式，而不需要对整个模型进行重新训练。



**提示调优**，简单来说就是通过精心设计输入的提示（Prompt），来引导语言模型生成更符合我们期望的输出。它有点像在和语言模型玩一个“猜谜游戏”，你给它一个提示，它根据提示来生成答案。

特征：		提示调优	微调

修改对象：	提示		模型参数

**软提示** 是一种通过在输入中添加可学习的连续向量来引导大型语言模型（LLM）生成特定输出的技术。简单来说，就是给模型提供一些额外的信息，让它知道我们想要它生成什么样的文本。

硬提示：通常是人工设计的、离散的文本片段，例如：“翻译成中文：”，或者“写一首关于爱情的诗”。硬提示更直观，但灵活性较差。

软提示：是一组连续的向量，这些向量是模型的一部分，可以随着模型的训练而更新。软提示更灵活，可以通过学习适应不同的任务和数据。

工作原理：

1. 初始化： 在模型的输入部分添加一个可学习的向量序列，作为软提示。
2. 训练： 在训练过程中，软提示的向量会随着模型的训练不断更新，从而更好地引导模型生成我们想要的输出。
3. 推理： 在推理阶段，将软提示与输入文本一起输入到模型中，模型就会根据软提示和输入文本生成相应的输出。



## Overview of Prompting

**提示学习概述**

![image-20241016185052052](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410161850351.png)



**Overview of Prompting**

- **In-context learning**: an LLM (Large Language Model) is given a prompt including a few training examples and a test instance as input.
- By leveraging its **autoregressive power**, LLMs understand and learn from the in-context examples to generate the output for the test instance directly, without any update to its parameters.
- Perform in a **zero-shot/few-shot** manner.
- Context learning is one of the most common ways to use giant LLMs.

**Advantages**:
- No training, only inference
- Enables real-time interaction (e.g., ChatGPT)
- Lowered requirements on training dataset size or label cost: only a few examples (<10) are enough
- Better generalization and less likely to overfit on the training data

In the diagram, there are examples showing **Zero-Shot** and **One-Shot** prompting scenarios:
- **Zero-Shot**: The model receives the following input: 
  - "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"
  - The model gives the wrong output: 27.
- **One-Shot**: The model is first given an example with a different question and the correct answer:
  - "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? Answer: 11."
  - Then, the model is asked the same question about apples again. This time, it gives the correct answer: 9.



**提示学习概述**

- **上下文学习**：LLM（大型语言模型）在提示中接收一些训练示例和一个测试实例作为输入。
- 通过利用其**自回归能力**，LLM能够从上下文中的示例中理解和学习，并为测试实例直接生成输出，而无需更新其参数。
- 以**零样本/少样本**的方式执行。
- 上下文学习是使用大型LLM的最常见方式之一。

**优势**：
- 无需训练，仅推理
- 支持实时交互（例如，ChatGPT）
- 对训练数据集大小或标签成本的要求降低：只需要几个示例（少于10个）即可
- 更好的泛化能力，且不容易在训练数据上过拟合

在图示中，展示了**零样本**和**单样本**提示学习场景：
- **零样本**：模型接收到如下输入：
  - “餐厅有23个苹果。如果他们用20个来做午餐并买了6个，那么他们还有多少苹果？”
  - 模型给出了错误答案：27。
- **单样本**：首先模型获得了一个不同问题的示例及正确答案：
  - “罗杰有5个网球。他买了2罐网球，每罐有3个网球。他现在有多少个网球？答案：11。”
  - 接着，模型再次被问及关于苹果的问题。这次，模型给出了正确答案：9。



这张图片展示了如何通过提示学习使用大型语言模型（LLM）。**提示学习**主要指的是模型在不改变内部参数的前提下，依靠提示信息中的示例进行学习和推理。这种方式非常适合大型预训练模型，因为可以避免重新训练模型，直接在推理阶段进行使用。

通过提示学习，模型可以根据少量甚至是零示例的情况，直接进行推理，通常被称为**零样本**（Zero-Shot）或**少样本**（Few-Shot）学习。这样的方法显著降低了对大规模标注数据的需求，并减少了训练时间。

在图片的例子中，展示了模型在零样本和单样本下的表现对比。在零样本提示下，模型没有示例的指导，容易给出错误答案；但在单样本提示中，模型通过一个示例进行学习，能够更好地理解问题，并给出正确答案。



**上下文学习（In-Context Learning）**，简单来说，就是通过在模型输入中提供一些示例或提示，让模型能够根据这些示例，快速适应新的任务或领域。这种学习方式不需要对模型进行额外的参数更新，而是通过模型本身的泛化能力来完成。



**自回归** 在大模型领域，特别是在自然语言处理中，是一个非常重要的概念。它描述了一种模型生成文本的方式：模型在生成下一个词时，会参考之前已经生成的词。



在传统机器学习中，模型通常需要大量带标签的数据进行训练。然而，在大模型时代，**零样本学习** 和 **少样本学习** 提供了一种全新的思路，使得模型能够在极少甚至没有标注数据的情况下，完成新的任务。

**零样本学习 (Zero-Shot Learning)**

- **定义：** 模型从未见过训练集中的任何关于该类的样本，却能够对该类进行分类或预测。
- **原理：** 模型通常会学习一个潜在的语义空间，将不同类别的样本映射到这个空间中。通过学习已知类别的属性和关系，模型可以推断出未见过的类别的属性，从而实现对未见过的类别的分类。
- **举例：** 假设模型训练时只见过猫和狗的图片，现在给模型一张从未见过的老虎的图片，模型也能判断出这是一只猫科动物。

**少样本学习 (Few-Shot Learning)**

- **定义：** 模型仅通过少量标注样本就能快速适应新的任务或类别。
- **原理：** 少样本学习通常会使用元学习（Meta-Learning）的思想，通过学习如何在少量样本上快速学习的能力，来适应新的任务。
- **举例：** 给模型提供几个手写数字的样本，模型就能学会识别新的手写数字。

假设我们有一个训练好的模型，它已经学习了猫和狗的各种属性。当我们给它一张老虎的图片时：

1. 模型会提取老虎的图像特征，比如条纹、尖耳朵、大眼睛等。
2. 模型会将这些特征映射到语义空间中。
3. 模型会发现老虎的特征与猫的特征有很多相似之处，比如都有尖耳朵、都是肉食动物等。
4. 因此，模型会推断老虎可能属于猫科动物。



**语义空间**，简单来说，就是一种将语言中的词语或句子映射到一个数学空间的方法。在这个空间中，相似的词语或句子会彼此靠近，而意思相差较远的词语或句子则会离得较远。



## What is parameter-efficient fine turning

![image-20241016193115784](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410161931065.png)



## Why do we need PEFT? Pros and Cons

Parameter-efficient Fine Tuning (PEFT).

**参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）** 是一种在大型预训练模型上进行微调的优化技术。它的核心思想是，通过对模型参数进行精细的调整，而不是对所有参数进行大规模的更新，来达到适应下游任务的目的。



PEFT主要有以下几种实现方式：

- **LoRA（Low-Rank Adaptation of Large Language Models）：** 在模型的每一层中插入可训练的低秩矩阵，通过对这些矩阵进行更新来实现对模型的微调。
- **Prefix Tuning：** 在输入序列的前面添加可学习的提示词，通过调整这些提示词来实现对模型的微调。
- **P-Tuning v2：** 对Prefix Tuning的改进，通过引入连续的提示表示，提高了模型的性能。
- **Adapter Tuning：** 在模型的每一层中插入一个适配器模块，通过对适配器模块的参数进行更新来实现对模型的微调。



## LoRA

LoRA的细节、有点、以及被大量使用这一趋势



Distillation：

## Memory & Compute Efficient LLM

![image-20241016203331149](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162033472.png)

**Memory & Compute Efficient LLM**

- **Quantization**: Speedup 6B LLM inference by 2x
- **Retrieval Enhancement**: Model size reduction by 50x in text generation
- **Side Tuning**: Avoid back-propagating through the backbone model
- **Distillation**: Model size reduction & performance improvement

**内存与计算高效的大型语言模型（LLM）**

- **量化**：使60亿参数的LLM推理速度提高2倍
- **检索增强**：在文本生成中将模型大小减少50倍
- **侧向微调**：避免对主干模型进行反向传播
- **蒸馏**：减少模型大小并提升性能



这张图展示了在大型语言模型（LLM）中实现内存和计算高效的几种技术：

1. **量化（Quantization）**：量化技术通过将模型的权重从高精度浮点数（如32位）减少为低精度浮点数或整数（如8位），从而大幅减少模型计算所需的内存和计算资源。在这里，它将60亿参数的模型推理速度提高了2倍。

2. **检索增强（Retrieval Enhancement）**：这一方法通过结合外部的检索模块，减少了模型生成文本时的计算负担，显著降低了模型的参数量。在图中提到，它将文本生成模型的大小减少了50倍。

3. **侧向微调（Side Tuning）**：与传统的微调方式不同，侧向微调不需要对主干模型进行反向传播更新，而是通过添加新的模块来适应新的任务，从而减少训练时间和资源消耗。

4. **蒸馏（Distillation）**：模型蒸馏是一种压缩模型的技术，通过将一个大型模型的知识“蒸馏”到一个较小的模型中，同时保持较高的性能表现。这种技术可以显著减小模型的体积，并提升推理性能。

这些方法的共同点是通过不同的技术手段提高大型模型的计算效率，减少推理时间和内存占用，使得大型模型可以在资源有限的环境下更高效地运行。



## Key Terms: Teacher and Student Model

"教师"模型可以生成训练标签，因此“学生”模型不需要创建很多标签

![image-20241016205946759](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162059053.png)



Grounding：

## Some shortcoming of LLMs when it comes to factuality

大语言模型在事实性方面的短板

LLMs produce output that we can check that it is factually wrong

大语言模型生成的输出可能存在事实错误。



* LLMs are trained in the past
    * LLMs data is frozen in past. So it will not know about recent developments or facts.
* LLMs hallucinate
    * LLMs are creative by nature. This might result in hallucinations at a significant rate.
* LLMs cant't cite sources
    * LLMs are good in reasoning, but less so on quoting which sources they used to come to a conclusion...

大语言模型的训练数据是过时的

- 大语言模型的训练数据是固定的，因此它对最近的发展或事实可能并不了解。

大语言模型会产生幻觉

- 大语言模型具有创造性，这可能会导致幻觉的产生。

大语言模型无法引用来源

- 大语言模型擅长推理，但不太擅长引用用于得出结论的来源。



**Minimizing hallucinations boils down to solving 3 problems**

* Right context
* Better models
* User experience

**减少幻觉归结为解决三个问题**

- 正确的上下文（可以让网络搜索与LLM并行、搜索引擎与广告商的私人数据结合）
- 更好的模型
- 用户体验



**幻觉** 在大语言模型中指的是模型生成的内容与实际情况不符，或者说，模型“编造”了不存在的事实。为了减少这种现象，研究人员认为主要有三个方面需要改进：

1. **正确的上下文：**
    - **问题：** 幻觉常常发生在模型对问题的理解不够准确的情况下。
    - **解决方案：** 需要为模型提供更准确、更完整的上下文信息，以便它能更好地理解用户的问题，从而生成更准确的答案。例如，提供问题的背景信息、限定问题的范围等。
2. **更好的模型：**
    - **问题：** 模型本身的局限性可能导致幻觉。
    - **解决方案：** 需要开发更强大的模型架构，采用更好的训练方法，并使用高质量的训练数据。例如，引入更多的知识图谱、提高模型的推理能力等。
3. **用户体验：**
    - **问题：** 用户的提问方式、表达习惯等都会影响模型的理解和生成。
    - **解决方案：** 需要设计更好的用户交互界面，引导用户更准确地表达需求。同时，可以提供一些反馈机制，让用户能够对模型生成的答案进行评价，从而帮助模型不断改进。



## Right context

![image-20241016212652846](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162126203.png)

**私有文档：** 指的是模型在训练时未曾接触过的、具有私密性的文档。

**新鲜内容：** 指的是来自网络的最新、最及时的数据。

**权威内容：** 指的是来自权威机构或可靠来源的数据。



![image-20241016212925111](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162129372.png)

**Retrieve and Generate (RAG)**

- **Retrieval Service**
    - 1: Retrieve (search results)
- **Retrieve Augment Generate (RAG)** 
    - 2: Augment
    - 3: Generate
- **LLM** (Large Language Model)
- **Prompt**
- **Response**



**检索与生成 (RAG)**

- **检索服务**
    - 1: 检索（搜索结果）
- **检索增强生成 (RAG)**
    - 2: 增强
    - 3: 生成
- **大型语言模型 (LLM)**
- **提示**
- **响应**



这张图展示了 RAG（检索-增强-生成）系统的工作流程。

1. **Prompt（提示）**：用户输入提示或问题，作为初始查询。
  
2. **Retrieval Service（检索服务）**：系统通过检索模块从外部数据源（如文档、知识库或网络）中获取与提示相关的信息。这一步称为“检索”，会返回相关的搜索结果。

3. **Retrieve Augment Generate（检索-增强-生成）**：
   - **Retrieve（检索）**：先获取外部的相关信息（如搜索结果）。
   - **Augment（增强）**：将检索到的信息与提示结合，增强模型的输入，以提高生成质量。
   - **Generate（生成）**：大型语言模型（LLM）结合增强后的信息生成最终的答案或响应。

4. **LLM（大型语言模型）**：在此过程中，LLM利用其语言理解和生成能力，结合检索到的信息来创建合理的响应。

5. **Response（响应）**：最终，系统生成的结果作为对用户问题的回答或回应。

**解释：**
RAG是结合检索和生成的一种架构。与单纯依赖生成模型不同，它通过在生成前先检索相关信息来增强生成的准确性。这种方法特别适用于需要引用外部知识库进行回答的复杂问题。例如，它在需要实时更新的信息（如新闻或学术文献）或回答涉及外部知识库的情况下非常有效。



## Typical RAG/NLI based grounding architecture in a nutshell

![image-20241016214109906](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162141229.png)

**Typical RAG/NLI based grounding architecture in a nutshell**

- **Pre-hoc**  
    - Prompt → Prompt2Query → Retrieval & Ranking → Extractive answer (snippets) → LLM → Response
- **Post-hoc**
    - Grounding Detection using NLI
    - Report:  
        sent 1, score, citations  
        ...  
        sent n, score, citations

---

**[Pre-hoc] Response generation:**
- Augment input with search results ("Retrieve Augment Generate" approach)
- LORA/fine-tune model to ground in pre-hoc input and generate citations

**[Post-hoc] Response corroboration:**
- Use Natural Language Inference to corroborate against corpus and generate/validate citations



**典型的基于RAG/NLI的基础架构概述**

- **预处理阶段 (Pre-hoc)**  
    - 提示 → 提示转查询 → 检索与排序 → 抽取式回答（片段）→ 大型语言模型 (LLM) → 响应
- **后处理阶段 (Post-hoc)**  
    - 使用自然语言推理 (NLI) 进行基础验证
    - 报告：  
        句子1，得分，引用  
        ...  
        句子n，得分，引用

**[预处理] 响应生成：**
- 使用搜索结果增强输入（"检索-增强-生成" 方法）
- 通过 LORA 或微调模型在预处理输入上进行基础验证并生成引用

**[后处理] 响应验证：**
- 使用自然语言推理 (NLI) 根据语料库进行验证并生成/验证引用



这个图展示了典型的基于 **RAG（Retrieve Augment Generate，检索增强生成）** 和 **NLI（Natural Language Inference，自然语言推理）** 的基础架构，用于生成和验证响应。架构分为两个主要阶段：**Pre-hoc**（预处理）和 **Post-hoc**（后处理）。

1. **Pre-hoc（预处理阶段）**：
   - **Prompt2Query（提示转查询）**：用户输入的提示首先被转换成查询。
   - **Retrieval & Ranking（检索与排序）**：查询通过检索模块获取相关的搜索结果，并根据相关性进行排序。
   - **Extractive Answer（抽取式回答）**：从检索结果中提取出相关的内容片段，作为后续生成响应的基础。
   - **LLM（大型语言模型）**：语言模型基于提取到的片段生成最终的响应。

2. **Post-hoc（后处理阶段）**：
   - **Grounding Detection using NLI（使用自然语言推理进行基础验证）**：生成的响应通过自然语言推理模型进行验证，确保响应中的信息有据可依，并生成验证报告。
   - **Report（报告）**：每个句子都会有一个验证得分，并且列出该句子的引用来源。

3. **LORA/微调模型**：LORA（Low-Rank Adaptation of Large Language Models）用于微调模型，确保模型在生成响应时能够进行准确的引用和验证。

4. **自然语言推理（NLI）**：在后处理阶段，NLI用于对生成的响应进行验证，确保它们的准确性，并生成相应的引用。

这套架构的目的是在生成响应的过程中，确保响应不仅仅是生成语言的结果，还能够在信息的来源和可靠性上进行验证，特别适合于需要高可信度回答的场景。



## Better models

**Better models**

How did we do this?

- Reinforcement learning with human and AI feedback
- Design reward model to highly punish ungrounded responses in addition to valuing helpfulness
- Create realistic high-quality training data

Impact: -81% hallucinations vs GPT40

**更好的模型**

我们是如何做到的？

- 人工智能反馈强化学习
- 设计奖励模型，除了重视帮助性之外，还对无依据的回答进行严厉惩罚
- 创建逼真的高质量训练数据

效果：与GPT-40相比，幻觉减少了81%



## User Experience



## How Function Calling works

1. Define your functions
2. Wrap functions in a tool
3. Call Gemini with a tools argument

1. 定义你的函数

2. 将函数封装到一个工具中
3. 使用`tools`参数调用Gemini



## A day in the life of a Function Call

![image-20241016215642093](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410162156416.png)

**A day in the life of a Function Call**

1. **Send prompt**
    - Prompt → Gemini → Response (User interaction)
   
2. **Gemini selects function and parameters**
    - Tool  
    - Function(s)  

3. **API calls to external systems**
    - REST APIs  
    - Databases  
    - Document Repositories  
    - Customer management systems  

4. **Respond to user**

**一次函数调用的一天**

1. **发送提示**
    - 提示 → Gemini → 响应（用户交互）
   
2. **Gemini 选择函数和参数**
    - 工具  
    - 函数（多个）

3. **API 调用外部系统**
    - REST APIs  
    - 数据库  
    - 文档存储库  
    - 客户管理系统  

4. **响应用户**



这个图片展示了一个典型的函数调用流程，特别是在 **Gemini** 系统中，如何通过多个步骤处理用户输入并与外部系统交互，最终生成响应。

1. **用户输入（Send prompt）**：
   - 用户通过提示输入需求，Gemini 作为中间层接收提示，并最终给出相应的反馈或响应。Gemini 在这个过程中充当了与系统和用户交互的接口。
   
2. **Gemini 选择函数和参数**：
   - 在接收到提示后，Gemini 根据需求选择适当的工具和函数来处理输入。这个选择过程可以包括多个函数的调用，以完成复杂的任务。
   
3. **与外部系统的API调用**：
   - 当选择的函数需要获取外部数据时，Gemini 通过API调用来连接各种外部系统，如 **REST APIs**、数据库、文档存储库，以及客户管理系统。这些系统提供了响应所需的数据或服务。
   
4. **响应用户**：
   - 最终，处理过的数据和外部资源整合后，系统返回一个完整的响应给用户。

这个流程展示了一个基于函数调用架构的系统如何协调内部工具和外部系统，来满足用户的需求。在现实中，这种架构适用于复杂任务，如自动化工作流程、数据处理，以及客户关系管理系统中的多步骤交互。  



## Function calling is all about developer control and flexibility

* Structured outputs
* Real-time information retrieval
* Employee search tool
* Customer support agent
* Autonomous workflows

函数调用关乎开发者对模型的控制力和灵活性

- 结构化输出
- 实时信息检索
- 员工搜索工具
- 客户支持代理
- 自主工作流



最后讲大模型评估也是很重要的工作

AI是很大的风口

学习要保持创造力是很重要的




# Lecture 5, Omar Khattab

**Compound AI Systems & DSPy**

Compound AI Systems **复合式人工智能系统**

**Compound AI Systems** 是一种结合多个 AI 模型和模块的架构，旨在分解复杂任务并提高系统整体的性能。与单一模型不同，Compound AI 系统将不同能力的模块化组件组合在一起，以实现更强大的信息处理和生成能力。这种系统的基本原理是“分而治之”，即将一个复杂的任务拆分为若干子任务，每个子任务由专门的模型模块处理，最后将各模块的输出进行整合，得到最终结果。例如，在复杂的问答系统中，可以采用一个模块进行检索，另一个模块进行信息过滤，再由生成模块生成最终的答案。这样的架构可以提高响应的准确性和信息的相关性，同时避免单一模型可能带来的局限性。

Compound AI 系统的优势在于其高度的灵活性和可扩展性，可以根据任务需求和实际应用场景进行不同模块的组合。此外，它还支持跨领域的任务处理，因为各个模块可以独立更新和优化，从而适应不同的任务需求。

DSPy

**DSPy** 是斯坦福大学推出的一个编程框架，旨在简化和优化大语言模型（LLM）的调用过程。DSPy 的核心是采用**声明式编程（Declarative Programming）**，通过定义任务目标和优化指标，而不是手工编写提示（prompts），实现对 LLM 的控制。DSPy 编译这些声明性的语言模型调用，并将其转化为自我优化的流水线，使得 LLM 的行为更加稳定和可预测。





大致主题（主讲DSPy）：

It's never been easier to build realy impressive AI demos.

现在构建令人印象深刻的AI演示从未如此简单。

Turning monolithic LMs into reliable systems remains challenging.

将单一的大型语言模型转化为可靠的系统依然具有挑战性。



## 提到的

| 项目        | 相关                                                         |
| ----------- | ------------------------------------------------------------ |
| DSPy        | 很多：https://github.com/stanfordnlp/dspy                    |
| DrQA        | Reading Wikipedia to Answer Open-Domain Questions            |
| ORQA        | Latent Retrieval for Weakly Supervised Open Domain QA        |
| RAG         | Retrieval-Augmented Generation for Knowledge-Intensive NLP   |
| ColBERT-QA  | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT |
| GoldEn      | Answering Complex Open-Domain Questions Through Iterative Query Generation |
| DecompRC    | Multi-hop Reading Comprehension through Question Decomposition and Rescoring |
| MDR         | Multi-Hop Dense Retrieval for Open-Domain Question Answering |
| Baleen      | Baleen: Robust Multi-Hop Retrieval Augmented Generation      |
| STORM       | Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models |
| AlphaCodium | Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering |
| DIN-SQL     | DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction |
| RARR        | RARR: Researching and Revising What Language Models Say, Using Language Models |
| MIPRO       | Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs |
|             | Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together |
| IReRa       | In-Context Learning for Extreme Multi-Label Classification   |
| STORM       | Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models |
| ECG-Chat    | ECG-Chat: A Large ECG-Language Model for Cardiac Disease Diagnosis |



## Every AI system will make mistakes.

But the **monlithic nature** of LMs makes them hard to control, debug, and improve.

语言模型的整体性

但大型语言模型的**单一性**使得它们难以控制、调试和改进。



To tackle this, AI researchers increasingly **build Compound AI Systems**(modular programs that use LMs as specialized components)

复合人工智能系统  语言模型在其中扮演模块化角色

为了解决这个问题，AI研究人员越来越多地**构建复合AI系统**（一种将大型语言模型作为专业组件使用的模块化程序）。



## Compound AI Systems：

复合AI系统

“复合AI系统”是指一种将多个独立的AI模块组合在一起工作的系统结构，每个模块负责特定的任务。这些模块通常包括大型语言模型（如GPT-4）等，但还可以结合其他类型的模型或工具，比如计算模块、知识库、自然语言处理单元等。通过这样的模块化设计，复合AI系统能够分工明确、更加灵活地应对复杂任务。

在复合AI系统中，大型语言模型（LM）通常作为其中的“组件”之一，只处理特定的生成或分析任务，而不再作为唯一的决策来源。这种方式允许系统在遇到不同类型的需求时调用合适的模块进行处理，从而有效地提升了系统的可控性和可靠性。



Transparency、Efficiency、Control、Quality、Inference-time Scaling

透明性、效率、控制、质量、推理时间扩展



### 例1

例子：RAG:  Retrieval-Augmented Generation 检索增强生成：是一种结合了信息检索（Retrieval）和生成模型（Generation）的方法，用于提升大语言模型在特定任务上的准确性和知识性。其核心思想是将生成模型与信息检索系统结合，以生成更可靠的答案。具体来说，RAG先通过检索模块从外部知识库（如Wikipedia、数据库、文档集合等）中提取相关信息，再将这些信息作为上下文输入到生成模型中，帮助生成模型回答问题或完成任务。这种方法使模型能够更好地处理需要精确信息的任务，减少生成虚假信息（hallucination）的概率。



![image-20241026181615032](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410261816379.png)

1. **Transparency**: can debug traces & offer user-facing attribution  
2. **Efficiency**: can use smaller LMs, offloading knowledge & control flow  

3. **透明性**：可以调试跟踪记录并提供面向用户的归因信息  
4. **效率**：可以使用更小的语言模型，分担知识和控制流程  



1. **透明性（Transparency）**：
    - 在复合AI系统中，各模块（如检索器和语言模型）各自承担不同的任务，使得调试过程更加清晰。通过跟踪和分析各模块的行为，开发人员可以更容易地识别问题来源，调试过程也更加透明。
    - 此外，系统能够提供“面向用户的归因”，即解释答案的来源或过程。这使用户可以理解系统是如何得出结论的，从而增加了信任度。

2. **效率（Efficiency）**：
    - 复合AI系统可以使用更小的语言模型，因为部分知识处理任务已被分担到其他模块（如知识检索器）上。这样，系统不需要一个大型、通用的模型来解决所有问题，而是通过协作完成任务。
    - 这种方法能够减少计算资源的需求，提高系统的运行效率，同时更容易进行控制流程管理，便于在复杂环境中实施和优化系统。



### 例2

例子：Multi-Hop Retrieval-Augmented Generation **多跳检索增强生成**：则是RAG的一种增强形式，专注于多跳检索任务。在多跳任务中，模型需要多次检索信息，并将各次检索的结果串联，才能找到完整的答案。例如，若要回答“哪位美国总统签署了某项特定法案？”这类问题，模型可能首先需要检索出该法案的内容，然后再检索与该法案有关的历史信息，以找出签署总统的名字。因此，Multi-Hop RAG中的检索过程是分步进行的，模型依次执行多个检索步骤，逐层构建信息链，以得到更全面、上下文完整的回答。

![image-20241026182924637](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410261829013.png)

1. **输入**：表示在检索系统的第一步输入的初始查询或数据，标记为 \( t = 1 \)。
2. **查询**：最初的查询被处理并转换为 \( Q_{t-1} \)，用于在后续步骤中检索相关信息的形式。
3. **FLIPR检索器**：一个使用 **FLIPR索引** 的检索模型，根据查询 \( Q_{t-1} \) 搜索相关段落或文档。
4. **Top-K段落**：FLIPR检索器根据查询返回与之最相关的 \( K \) 个段落。
5. **两阶段凝缩器**：该组件从 Top-K 段落中提取关键信息，生成 **凝缩事实**。
6. **更新查询 \( Q_t \)**：基于凝缩的事实，对查询进行更新，以进一步优化信息搜索或朝最终答案更接近。
7. **完成？**：该决策点用于判断是否已经达到最终的迭代次数 \( t = T \)。如果没有，则循环回去进一步优化查询。
8. **任务特定读取器**：一旦查询处理完成，任务特定读取器解读优化后的信息并生成 **预测结果**。

该图示描述了一种基于迭代查询更新的多跳检索（multi-hop retrieval）模型流程，用于在庞大的信息库中获取精准答案。这种系统的核心是通过不断更新和优化查询，使得模型在每次迭代中能找到更准确的相关信息。具体来说：

- 初始查询进入系统，通过 FLIPR 检索器查询 FLIPR 索引库，得到 Top-K 最相关的段落。
- 接着，两阶段凝缩器对这些段落进行信息提炼，去除冗余，保留关键事实。这一阶段生成的凝缩事实用于更新查询。
- 然后，系统判断是否已经达到了预设的迭代次数，未达到则循环以更新的查询再进行检索。
- 最终，当迭代完成后，任务特定读取器对最终获得的信息进行分析，并生成预测结果，作为系统输出。

这种方法在需要多次信息检索和整合的任务中尤其有效，如问答系统中的复杂问题解析，通过多次检索和信息整合提高准确性和答案的全面性。



**Control**: can iteratively improve the system & ground it via tools

**控制**：可以通过工具迭代性地改进系统并将其扎根



### 例3

例子：Compositional Report Generation

1. **Retrieval-Augmented Generation（RAG，检索增强生成）**

Retrieval-Augmented Generation 是一种结合信息检索和文本生成的技术框架。基本思路是先从外部知识库中检索到与输入查询相关的文本片段（即“检索”），然后在这些检索到的内容基础上生成回答或完成指定的任务（即“生成”）。这种方法特别适用于需要外部知识支持的生成任务，比如回答事实性问题、生成内容密集的文本等。

* **应用优势**：相较于仅依靠模型内部记忆的生成方法，RAG通过实时检索，能够动态访问更新的外部信息库，确保输出的准确性和实时性。典型的应用场景包括问答系统、文本摘要、客户支持等需要外部知识支撑的任务。

2. **Multi-Hop Retrieval-Augmented Generation（多跳检索增强生成）**

Multi-Hop Retrieval-Augmented Generation 是对RAG的进一步扩展，涉及**多次（多跳）检索和生成**。与单次检索不同，多跳检索能够处理复杂的、多步推理问题。例如，一个查询可能需要先找到第一层相关信息，再利用这些信息检索出更深层次的关联内容，从而多次迭代检索，最终生成综合性的答案。

* **应用场景**：多跳检索增强生成适用于需要跨越多个信息来源，或通过多个推理步骤才能得到最终答案的问题。这在复杂问答、学术搜索和多层推理的知识图谱生成等领域有广泛应用。

3. **Compositional Report Generation（组合式报告生成）**

Compositional Report Generation 是一种面向任务的生成技术，旨在**生成结构化的报告或文档**。它通常将生成过程分为多个组成部分，每个部分独立生成特定的信息块，最后组合成完整的报告。例如，在生成一份医疗报告时，可以先生成病人的背景信息、接着生成诊断结果、然后生成治疗建议，最后将所有模块组合成完整报告。

* **技术优势**：这种生成方式能够更好地控制生成内容的结构和信息质量，避免传统生成模型可能带来的逻辑不连贯和内容缺失的问题。尤其在对信息完整性和逻辑清晰度要求较高的场景中（如金融报告、医学报告、法律文件生成等），组合式生成可以确保报告的专业性和条理性。



**RAG** 更强调通过检索增强生成的知识基础；

**Multi-Hop RAG** 在RAG的基础上引入多步推理，适应复杂逻辑问题；

**Compositional Report Generation** 则通过模块化生成确保文本的结构和逻辑性，特别适合生成高度结构化的文本。



![image-20241026185710137](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410261857467.png)

1. **主题 \( t \) 调查**：通过 **ColBERT 检索器** 开始对主题 \( t \) 的调查，收集相关文章。
2. **识别观点**：识别多个观点 \( P \)，以覆盖主题的不同角度。
3. **阅读与提问**：Wikipedia Writer 机器人与这些观点互动，通过阅读和提问来添加具体的视角。
4. **专家咨询**：Expert 机器人接收问题 \( q \)，将其分解为子问题，进行搜索和筛选，并将信息综合为答案 \( a \)。
5. **添加可信来源**：从专家的发现中收集可信来源和参考资料。
6. **草稿大纲 \( O_D \)**：基于收集到的信息和观点直接生成草稿大纲。
7. **完善**：通过多个对话 \( \{ C_0, ..., C_N \} \) 对草稿进行完善，并整合可信来源以最终完成大纲 \( O \)。
8. **最终输出**：生成包含参考资料的组织良好的最终大纲。

该流程图描述了一个基于多模块和多代理系统的知识生成过程，特别适用于创建复杂主题的维基百科式内容。系统通过以下步骤实现信息的高效组织与生成：

1. **ColBERT 检索器**：这是一个嵌入检索模型，用于初步筛选与主题相关的内容。这一步确保了信息来源的广泛性和多样性。
2. **观点识别与添加**：为了确保信息的全面性，系统会识别并添加多种不同视角，以减少偏见。
3. **专家咨询与问题分解**：专家模块进一步细化问题，将复杂问题分解为多个子问题，分别查找答案并综合，形成更完整的回答。这种多跳检索和生成过程能深入处理复杂的跨领域问题。
4. **可信来源整合**：系统从多层次的信息中提取可信来源，最终呈现可靠的参考资料。
5. **草稿生成与完善**：生成一个初步的大纲后，通过多次对话反复优化，确保信息的逻辑性和流畅性。最终输出的内容具有条理清晰、信息全面的特点，适合用于生成正式的百科类文章。

这个系统的设计展示了如何利用不同模块（如检索、专家、生成器）协同工作，以实现高效的多来源信息整合和生成，在自动内容创建和知识管理领域具有广泛的应用潜力。



**Quality**: more reliable composition of better-scoped LM capabilities

**质量**：更可靠地组合具有更佳范围限定的语言模型能力

这句话强调了在大语言模型（LM）设计和应用中的一个核心目标：**通过更精准的功能范围定义，实现语言模型能力的可靠组合**。其核心思想是将大模型的能力模块化，将每个模块的作用范围或应用场景进行清晰定义，从而在执行具体任务时能够更加精准和高效。

这种更优范围限定的能力组合能够有效减少因模型泛化能力过强导致的误差和偏差。例如，在复杂任务的处理过程中，可以将模型划分为专门负责检索、分析和生成的模块，每个模块都有明确的职责范围，通过精确调控模型的行为来确保输出的稳定性和一致性。这一过程与 DSPy 等框架中的模块化优化思路一致，使得最终的模型在多步骤、多场景任务中表现出更高的质量和可靠性。



### 例4

AlphaCodium

https://github.com/Codium-ai/AlphaCodium



![image-20241026212456015](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410262124420.png)

**Pre-processing**

1. **Input - Problem Description + Public Tests**: Start with the problem description and public tests.
2. **Problem Reflection**: Reflect on the problem to understand requirements.
3. **Generate Possible Solutions**: Generate potential solutions based on initial insights.
4. **Public Tests Reasoning**: Use public tests to reason about the solution.
5. **Generate Additional AI Tests**: Create additional tests to ensure robustness.
6. **Rank Solutions**: Evaluate and rank the generated solutions based on performance.

**Code Iterations**

1. **Initial Code Solution**: Begin with an initial solution.
2. **Iterate on Public Tests**: Refine the solution based on feedback from public tests.
3. **Iterate on AI Tests**: Further refine the solution with AI-generated tests.
4. **Final Solution**: Arrive at the final, optimized solution.

**SQL Generation**

1. **Schema Linking**: Link relevant database schema for understanding context.
2. **Classification & Decomposition**: Classify the query as Easy, Nested Complex, or Non-Nested Complex for appropriate handling.
3. **Self-correction**: Generate the final SQL query and self-correct for accuracy.

**Task-Agnostic Prompting Strategies**

- Strategies include **Best-of-N**, **Chain of Thought**, **Program of Thought**, **ReAct**, **Reflexion**, **Archon**.



**预处理**

1. **输入 - 问题描述 + 公共测试**：从问题描述和公共测试开始。
2. **问题反思**：对问题进行反思，理解需求。
3. **生成可能的解决方案**：基于初步见解生成潜在的解决方案。
4. **公共测试推理**：利用公共测试对方案进行推理分析。
5. **生成额外的AI测试**：创建额外的测试以确保方案的稳健性。
6. **解决方案排序**：基于性能对生成的解决方案进行排序。

**代码迭代**

1. **初始代码方案**：从初始方案开始。
2. **基于公共测试迭代**：根据公共测试的反馈优化方案。
3. **基于AI测试迭代**：使用AI生成的测试进一步优化方案。
4. **最终方案**：得出最终优化的方案。

**SQL生成**

1. **模式链接**：链接相关的数据库模式，以理解上下文。
2. **分类和分解**：将查询分类为简单、嵌套复杂或非嵌套复杂，以便适当处理。
3. **自我校正**：生成最终的SQL查询，并进行自我校正以确保准确性。

**任务无关的提示策略**

- 包括 **Best-of-N**、**思维链**（Chain of Thought）、**思维程序**（Program of Thought）、**ReAct**、**Reflexion** 和 **Archon** 等策略。



这张图展示了一个基于 AI 系统的多阶段问题解决框架，包含从预处理到代码迭代和SQL生成的完整过程。这种框架不仅适用于代码生成，还可以适应不同类型的任务，如数据库查询和自动化问题解决。整体流程分为以下几个部分：

1. **预处理**：从问题描述出发，逐步进行问题反思和初步解决方案的生成。这一阶段通过公共测试和额外的 AI 测试来验证解决方案的可行性，并根据测试结果对方案进行排序，确保初步方案具备较高的可靠性。

2. **代码迭代**：在此阶段，基于公共测试和 AI 测试对初始方案进行多次迭代和优化。最终的解决方案是经过多次反馈校正的结果，力求满足问题的各项需求。

3. **SQL生成**：该部分针对数据库查询任务，分为模式链接、分类分解和自我校正三个步骤，通过理解数据库结构和分解复杂查询，最终生成精确的SQL语句，并进行自我校正以确保输出的准确性。

4. **任务无关的提示策略**：该部分列出了常用的提示策略，包括“最佳N次选择”（Best-of-N）、“思维链”（Chain of Thought）和“ReAct”等。它们可以广泛应用于不同任务中，提高生成内容的质量和准确性。这些策略对复杂任务的解决尤为有用，例如多步骤推理和复杂问题分解。

这个框架展示了如何在复杂任务中结合多种 AI 模块和策略，以实现更加精确和有效的自动化解决方案。



1. **Best-of-N**

Best-of-N 是一种简单的生成策略，指在执行同一任务时，生成 N 个不同的候选输出，然后从中选择质量最好的一个。通常用于语言模型的生成任务中，通过多次生成避免了单一输出可能带来的质量问题。这种方法的优势在于能够增加生成的多样性，同时确保最终选择的输出更符合任务要求。

2. **思维链（Chain of Thought, CoT）**

思维链（Chain of Thought, CoT）是一种引导语言模型进行多步推理的提示策略。其原理是让模型在回答复杂问题时，逐步展示其推理过程，而不是直接生成最终答案。例如，给出问题后，提示模型按照步骤分解每个逻辑环节，最终生成完整的回答。这种策略适用于需要逻辑推理的任务，如数学问题解答和复杂问答任务，通过清晰的步骤化回答，提升了模型的解释性和准确性。

3. **思维程序（Program of Thought, PoT）**

思维程序（Program of Thought, PoT）是一种扩展的思维链策略，它将复杂任务结构化为更细化的子任务，让模型在多个步骤中解决问题。与思维链不同，PoT 侧重将每一步明确地模块化，类似于编程中的函数调用，使得模型能够通过执行子任务逐步解决整个问题。PoT 适用于结构化较强的任务，例如编程辅助或流程化的操作步骤。

4. **ReAct**

ReAct 是一种结合“反应”（React）和“行动”（Act）的策略，用于任务规划和推理。ReAct 不仅让模型思考下一步该如何回答（反应），还让模型采取行动，例如查询额外信息或调用子模块来辅助完成任务。它在复杂的问答系统中特别有效，能够灵活地根据问题采取不同的行动，从而提高回答的准确性和任务完成度。

5. **Reflexion**

Reflexion 是一种反馈导向的策略，允许模型根据自身生成的初始回答进行自我检查和反思。模型会生成初始答案，然后对其进行自我评估，并通过内置反馈机制判断是否需要修改或完善答案。这种策略在需要高准确性的任务中非常有用，例如诊断类或分析类任务，因为它允许模型自我纠正并迭代改进回答。

6. **Archon**

Archon 是一种结合多种策略的复合框架，允许模型灵活运用多种生成和提示策略来处理任务。例如，Archon 可以在任务开始时使用思维链策略，然后切换到 ReAct 以进行动态调整，最终通过 Reflexion 来完成自我评估。这种组合式的策略框架适用于高度复杂和动态的任务，因为它能够在不同阶段灵活应用最合适的策略，从而提高最终结果的精确度和可靠性。

这些提示策略为大语言模型提供了更灵活和精确的生成方法，尤其在复杂任务、多步骤推理和动态信息需求的场景下表现优异。



**Inference-time Scaling**: systematically searching for better outputs

**推理时间扩展**：系统地搜索更优输出

“推理时间扩展”指的是在模型推理阶段通过优化方法系统性地探索并生成更高质量的输出。这一方法常用于大语言模型（LLM）或其他深度学习模型中，以在不更改模型结构或参数的情况下，通过不同的推理技巧或调优手段来提高模型的输出质量。这种方法通常包括对不同的生成路径、温度参数调整、采样策略等进行探索，以确保获得最佳的推理结果。

例如，推理时间扩展可能涉及使用**多样化采样**（如调节采样温度和引入随机性），或**选择性筛选**生成的多个候选答案，以挑选出最符合任务需求的输出。通过在推理阶段进行多次生成并评估，这种方法可以显著提升复杂任务的完成度和生成质量。





## Unfortunately, LMs are highly sensitive to how they are instructed to solve tasks, so under the hood...

不幸的是，语言模型对任务指令的形式高度敏感，因此在底层的实现上，...



该句强调了语言模型（LMs）的一个关键特性，即它们对输入指令的格式和措辞极其敏感。这意味着，即使是相似的任务，如果指令表达稍有不同，模型的输出结果可能会有显著差异。这种高度敏感性使得在实际应用中，需要对提示（prompt）进行精细调试和优化，以确保模型能够准确理解并执行任务。

这种现象的根本原因在于模型的预训练过程中，对特定语言模式的偏好会影响它在不同指令下的响应，因此在复杂任务中，指令设计显得尤为重要。



![image-20241026215522391](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410262155722.png)

**Each "prompt" couples five very different roles:**

1. The core **input → output** behavior, a **Signature**.
2. The computation specializing an inference-time strategy to the signature, a **Predictor**.
3. The computation formatting the signature’s inputs and parses its typed outputs, an **Adapter**.
4. The computations defining objectives and constraints on behavior, **Metrics and Assertions**.
5. The strings that instruct (or weights that adapt) the LM for desired behavior, an **Optimizer**.

**Existing Compound AI Systems are modular in principle, but are too "stringly-typed": they couple the fundamental system architecture with incidental choices not portable to new LMs, objectives, or pipelines.**

**每个“提示”结合了五种截然不同的角色：**

1. 核心的 **输入→输出** 行为，称为 **签名**。
2. 专门针对签名的推理时间策略的计算，称为 **预测器**。
3. 对签名的输入进行格式化并解析其类型化输出的计算，称为 **适配器**。
4. 定义行为目标和约束的计算，称为 **指标和断言**。
5. 指导语言模型实现所需行为的指令（或调整权重），称为 **优化器**。

**现有的复合式人工智能系统在原则上是模块化的，但“过于依赖字符串类型”：它们将系统的基础架构与偶然性的选择耦合在一起，难以迁移到新的语言模型、目标或流水线中。**



该图表描述了一个**复合提示系统**的设计结构，强调每个“提示”在与模型的交互中充当的不同角色。这种结构基于多层次的模块化设计，每个模块承担特定的任务，有助于在复杂 AI 系统中实现更为灵活、可控的模型行为：

1. **Signature（签名）**：代表模型的核心行为，即如何将输入映射到输出。签名是任务的主要定义，它决定了模型的总体目标。

2. **Predictor（预测器）**：这是一个推理模块，在推理阶段选择合适的策略来实现签名的目标。它负责确定如何在实际推理中使用模型的能力。

3. **Adapter（适配器）**：适配器负责格式化输入和输出，将它们转化为模型能够理解和使用的类型化数据。这有助于确保不同类型的数据在进入模型之前被正确地格式化。

4. **Metrics and Assertions（指标和断言）**：这一层定义了对模型行为的目标和约束。通过指标和断言，可以确保模型的行为符合预期的性能要求和质量标准。

5. **Optimizer（优化器）**：优化器的作用是微调提示文本或调整权重，以确保模型按照所期望的方式工作。它通过反馈机制不断调整提示，提高生成结果的质量和准确性。

这种分层设计有助于提高复合 AI 系统的可移植性和模块化，但也指出了当前系统的不足：许多系统过于依赖于具体的提示文本（即“stringly-typed”），导致难以适应不同的模型或任务需求。在实现通用性和灵活性方面，还需要进一步的优化和改进。



## We know how to build control systems & improve them modularly.

我们知道如何构建控制系统并以模块化方式改进它们。

That is called **programming.**



What if we could abstract Compound AI Systems as **programs** with fuzzy **natural-language-typed modules** that **learn their behavior**?

如果我们可以将复合式人工智能系统抽象为带有模糊自然语言类型模块的程序，这些模块能够学习它们的行为，该会怎样？ 

这句话设想了一种新的设计方法，探讨是否可以将复合式人工智能系统（Compound AI Systems）视作由**模糊自然语言类型模块**组成的程序。这些模块不仅接受自然语言的描述，还可以通过“学习”来自数据的行为，逐步适应和改进自己的功能。该设想试图通过让各模块在模糊语言描述下自我学习，来提升系统的灵活性和可适应性，最终减少对固定规则或预定义结构的依赖，使得系统更容易适应不同任务和场景的需求。



## DSPy

data science python

自然语言 -> 自然语言



## As an example, let’s say we wanted to build this pipeline for multi-hop retrieval-augmented generation.

![image-20241027100152255](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410271001552.png)



**As an example, let’s say we wanted to build this pipeline for multi-hop retrieval-augmented generation.**

作为一个示例，假设我们想构建这个多跳检索增强生成的流水线。

```python
def multihop_qa(question: str) -> str:
    for i in range(2):
        query = (question, context)
        context = (query)
    return (question, context)
```

图示代码片段定义了一个名为 `multihop_qa` 的函数，用于实现多跳检索增强生成。代码通过一个简单的循环进行两轮迭代，每次迭代中，`query` 会基于当前 `question` 和 `context` 进行更新，而 `context` 也基于 `query` 进行调整，最终返回生成的 `answer`。



这个图示展示了一个**多跳检索增强生成（Multi-hop Retrieval-Augmented Generation）**的示例实现，用于处理复杂的问答任务。主要原理包括以下几个步骤：

1. **初始化**：函数接受一个 `question` 参数作为初始查询，同时初始化 `context`，用于存储多跳信息。
2. **多跳迭代**：在循环中，每一跳都会基于 `question` 和 `context` 生成一个新的 `query`，并使用此 `query` 更新 `context`，以便在下一次跳跃中包含之前的知识。
3. **返回结果**：在完成预定次数的跳跃后，函数返回最终的 `context`，作为对问题的综合回答。

这种多跳检索方法在复杂任务中十分有效，尤其是需要层层递进地获取信息时。通过递归迭代，系统能够逐渐收集并整合所需信息，从而提供更加全面和精准的回答。



1. **Question（问题）**：`question` 是用户输入的初始问题，通常是多跳检索系统的核心驱动因素。例如，“谁是目前联合国秘书长的前任？”这种问题会触发系统寻找相关信息。`question` 在整个过程中保持不变，是系统回答最终问题的主要依据。

2. **Query（查询）**：`query` 是系统在每个跳跃阶段生成的查询语句。`query` 可以理解为根据当前已知信息生成的“子问题”或“跟进问题”。每一跳中，系统会将 `question` 和当前 `context` 的信息结合起来生成一个新的 `query`，以便在下一次跳跃中查找更具体的信息。例如，在第一跳中，系统可能会查询“联合国秘书长的前任是谁？”，而在接下来的跳跃中，可能会进一步查询这个前任的其他信息。

3. **Context（上下文）**：`context` 是从每次检索中返回的信息，它在每一跳之间传递并更新。`context` 包含从检索模块返回的文本段落或事实信息，为系统的下一步推理和生成提供支持。在每一跳中，`context` 都会结合 `query` 来丰富系统对原始 `question` 的理解，并帮助生成更为准确的回答。通过不断累积 `context`，系统能够在多跳中获取更深层次的信息，以逐步靠近最终答案。



为什么循环

**递进式信息获取**：复杂问题往往无法通过单次检索直接获得完整答案。第一轮检索可以获取初步的信息，将问题分解为子问题；第二轮则在第一轮获取的上下文基础上进行更深入的查询，从而帮助模型获得更加详尽的背景信息。

**多步推理**：多跳检索的核心目的是模拟人类的多步推理过程。通过每次迭代，系统能够逐步理解问题的深层次含义，并在初步检索的基础上深入挖掘。两次循环可以让系统在首次获取的 `context` 基础上进一步生成新的 `query`，最终得到更精准的答案。

**信息累积与上下文增强**：在多跳系统中，后续的检索可以利用前一轮的 `context` 累积信息，使模型的理解逐步丰富。例如，回答一个需要背景知识的问题时，第一次检索可能仅提供相关人物或事件的初步信息，第二次检索则可以基于这些信息获取更细节的内容，帮助回答更精确的子问题。



## Anatomy of an LM program in DSPy

DSPy 中语言模型程序的组成剖析



风格模仿PyTorch



## How can we translate these into high-quality prompts?

我们该如何将这些内容转化为高质量的提示语？



First, modules are **translated into basic prompt** using Adapters and Predictors.

首先，模块通过适配器和预测器**转换为基本提示**。



DSPy’s **Optimizers** can then tune this prompt ... jointly along with all other prompts in your program.

DSPy 的**优化器**随后可以与程序中的其他所有提示一起联合调整该提示。



Instead of tweaking brittle prompts

不再调整脆弱的提示



## DSPy Optimizers vary in how they tune the prompts & weights in your program, but at a high level they typically...

DSPy 优化器在调整程序中的提示和权重方面有多种方法，但总体上通常会..

![image-20241027100114087](https://raw.githubusercontent.com/R10836/TyporaImageBox/main/img/202410271001417.png)

*DSPy Optimizers vary in how they tune the prompts & weights in your program, but at a high level they typically...*

1. Construct an **initial prompt** from each module via an **Adapter**
2. **Generate examples** of every module via rejection sampling
3. Use the examples to **update the program’s modules**  
    - a. Automatic few-shot prompting: `dspy.BootstrapFewShotWithRandomSearch`
    - b. Induction of instructions: `dspy.MIPROv2`
    - c. Multi-stage fine-tuning: `dspy.BootstrapFinetune`

*DSPy 优化器在调整程序中的提示和权重方面有多种方法，但总体上通常会...*

1. 通过**适配器**从每个模块构建**初始提示**
2. 使用拒绝采样为每个模块**生成示例**
3. 使用这些示例来**更新程序的模块**  
    - a. 自动少样本提示：`dspy.BootstrapFewShotWithRandomSearch`
    - b. 指令归纳：`dspy.MIPROv2`
    - c. 多阶段微调：`dspy.BootstrapFinetune`



这张图片展示了 **DSPy 框架**中优化器（Optimizers）如何在高层次上调整提示（prompts）和权重的流程。该流程主要包含以下步骤：

1. **构建初始提示**：每个模块通过适配器（Adapter）生成一个初始提示，作为任务的起点。这一步是将模块的输入与提示格式进行标准化的关键。

2. **生成示例**：通过拒绝采样（rejection sampling）为每个模块生成一组样本。拒绝采样是一种生成数据的统计方法，用于确保生成的样本符合特定标准，这样可以有效提高模型的精度。

3. **更新模块**：使用生成的样本来对各模块进行更新。这里列出了三种更新策略：
    - **自动少样本提示**（Automatic few-shot prompting）：通过随机搜索的方法生成少量示例，提升模型在少样本条件下的泛化能力。
    - **指令归纳**（Induction of instructions）：通过 `dspy.MIPROv2` 进行指令归纳，使得模型能够在学习过程中推理出适当的指令。
    - **多阶段微调**（Multi-stage fine-tuning）：通过 `dspy.BootstrapFinetune` 进行多阶段的微调，使得模型能够在不同阶段逐步提升性能。

这些步骤的目的在于使模型更好地适应特定任务和输入数据，提高生成输出的准确性和一致性。这种模块化和多层次的优化方法，使得 DSPy 框架在复杂任务中表现更出色，同时也更易于迁移到不同任务和模型。



## MIPRO

works well in practice & has enable many SoTA systems



接下来是讲论文：[Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/pdf/2406.11695)



# Lecture 6, Graham Neubig

Agent for Software Development



# 未完待续

敬请期待。。。

