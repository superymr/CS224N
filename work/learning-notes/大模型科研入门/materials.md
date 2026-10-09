# 大模型科研入门

> 分类：人工智能 / 大模型 / 科研入门
> 状态：持续收集

## 学习目标

- 建立大模型领域的术语地图。
- 理解模型、数据、训练、推理、对齐、评测和应用之间的关系。
- 为后续阅读论文、复现实验和提出研究问题准备词汇基础。

## 当前材料

- 已确认课程分区：大模型学习 / 斯坦福 CS224N。课程材料单独维护于 `CS224N/materials.md`，按讲次收集概念、推导、代码、作业与疑问。

- 本轮将介绍大模型科研入门的核心术语，作为后续学习笔记的总目录。

## 已确认收录：第一批核心术语

### 1. Token 与 Embedding

- `Token` 是模型处理文本的基本单位，不一定等于汉字或单词。
- `Tokenizer` 将文本转换为 Token ID；Token 数量会影响上下文长度和推理成本。
- `Embedding` 将离散的 Token ID 转换为连续向量，供模型进行数学运算。
- 需区分模型内部的 Token Embedding 与用于语义检索的文本 Embedding。

### 2. Transformer 与 Attention

- `Transformer` 是当前主流大语言模型的核心架构，主要包含 Self-Attention 与 Feed-Forward Network。
- `Attention` 根据上下文动态决定当前 Token 应重点关注哪些 Token。
- Attention 的基本形式：`Attention(Q, K, V) = softmax(QKᵀ / √d) V`。
- `Q` 表示查询，`K` 表示可匹配的线索，`V` 表示实际聚合的信息。
- 生成式模型常使用 `Causal Self-Attention`，当前位置只能看到自己及之前的 Token。

### 3. Pretraining 与 SFT

- `Pretraining` 使用海量文本训练模型预测下一个 Token，使模型获得通用语言、知识和模式能力。
- `SFT`（Supervised Fine-Tuning）使用指令—回答样本继续训练，使模型学会遵循指令、对话和指定格式。
- 核心区别：预训练主要让模型学会语言，SFT 主要让模型学会按要求完成任务。

### 4. LoRA、RLHF 与 DPO

- `LoRA` 冻结原始模型，只训练低秩增量参数，属于参数高效微调方法。
- `RLHF` 通过人类偏好数据训练奖励模型，再用强化学习优化语言模型。
- `DPO` 直接利用“更好回答/较差回答”的偏好数据进行优化，通常比完整 RLHF 流程更简单。
- 训练链路可粗略理解为：`预训练 → SFT/LoRA → RLHF 或 DPO`。

### 5. Inference

- `Inference` 是使用训练好的模型生成结果的过程。
- 模型通常逐个 Token 生成文本。
- `temperature` 控制随机性；`top-k` 和 `top-p` 控制候选 Token 范围；`max_new_tokens` 控制最大生成长度。

### 6. RAG

- `RAG`（Retrieval-Augmented Generation）先检索外部文档，再把相关内容放入 Prompt，让模型据此回答。
- 典型流程：文档切分 → 文本向量化 → 向量检索 → 重排序 → 拼接 Prompt → 生成回答。
- RAG 可以接入私有数据和新知识，但不能自动保证检索结果或最终回答正确。

### 7. Benchmark 与 Hallucination

- `Benchmark` 是用于比较模型能力的标准化数据集或测试集合。
- 应关注评测任务、指标、数据污染、测试集泄漏和是否使用额外提示词。
- `Hallucination` 指模型生成看似合理但实际错误、虚构或缺乏证据的内容。
- 减少幻觉的常见方法包括 RAG、工具查询、事实核验、证据引用和不确定性表达。

### 8. 关键关系

```text
文本 → Tokenizer → Token ID → Embedding → Transformer/Attention
     → 预测下一个 Token → Inference → 生成回答
```

```text
预训练模型 → SFT/LoRA → RLHF/DPO → RAG/工具调用 → 推理与评测
```

### 9. 本节重点记忆

1. `Token` 是输入单位，`Embedding` 是向量表示。
2. `Attention` 是 Transformer 中用于建模上下文关系的核心机制。
3. 预训练获得通用能力，SFT 学习指令遵循。
4. LoRA 是低成本微调方法，不是新的基础模型。
5. RAG 是外部知识检索增强，不等于重新训练模型。

## RAG 与语义搜索

### 语义搜索

语义搜索不是只匹配字面关键词，而是将问题和文档转换为向量，再根据语义相似度检索相关内容。

基本流程：

```text
文本 → Embedding 向量 → 相似度计算 → 返回相关内容
```

### RAG

RAG（Retrieval-Augmented Generation，检索增强生成）将语义搜索与大语言模型结合：

```text
用户问题 → 检索相关文档 → 将文档放入 Prompt → 大模型生成回答
```

语义搜索负责“找到相关资料”，RAG 负责“让大模型利用这些资料回答”。RAG 可接入最新知识、企业内部资料和个人知识库，并减少重新训练模型的需求，但不能自动保证检索内容或最终回答一定正确。

### RAG 典型组成

- `Chunking`：文档切分
- `Embedding Model`：文本向量模型
- `Vector Database`：向量数据库
- `Retriever`：检索器
- `Reranker`：重排序模型
- `Context`：提供给大模型的上下文

典型完整流程：

```text
文档 → 文档切分 → Embedding → 向量数据库 → 语义检索 → 重排序 → 大模型生成答案
```

## LoRA 的通俗理解

LoRA 更像“专业技能插件”，而不是知识库或传统操作指南。

- `RAG` 更像知识库：回答时检索外部资料，适合提供可更新、可追溯的事实。
- `Prompt` 更像操作指南：告诉模型当前任务应该如何完成。
- `SFT` 更像系统培训：用大量示例教模型完成任务。
- `LoRA` 更像技能插件：在冻结原始模型的基础上，训练少量新增参数，使模型长期形成某种领域能力、语言风格或行为倾向。

LoRA 可以让模型记住一部分专业知识，但不适合作为可靠知识库，因为知识更新需要重新训练，内容可能记错，也不方便追溯来源。

实际系统中常将三者结合：

```text
基础模型 + LoRA 专业能力 + RAG 最新知识
```

例如，LoRA 负责让模型用医疗专业方式回答，RAG 负责提供最新医学资料。

## Token 与 Embedding

### 1. Token 的含义

Token 是文本经过 Tokenizer 拆分后得到的离散语言片段。它可能是：

- 一个汉字；
- 一个中文词语或短语；
- 一个英文单词或子词；
- 一个单词的一部分；
- 代码中的关键字、符号或其他片段。

Token 不等于汉字，也不等于完整单词。具体切分方式由 Tokenizer、词表和预处理规则决定。

### 2. Token、Token ID 与向量的区别

三者关系如下：

```text
人类文本
  ↓ Tokenizer
Token 片段
  ↓ 查词表
Token ID
  ↓ Embedding 查表
向量
```

- **Token**：文本中的离散片段。
- **Token ID**：词表中该 Token 对应的整数编号。
- **Embedding 向量**：模型通过 Embedding 矩阵查找到的连续数值表示。

Token 本身不是向量；Token ID 也不是有语义的数值。向量是 Token 进入神经网络后进行计算的表示。

### 3. Token ID 是否固定

在同一个模型中，如果使用相同的 Tokenizer、词表、预处理规则和特殊 Token 配置，那么词表中某个 Token 对应的 Token ID 通常是固定的。

例如：

```text
机器 → 1256
```

但一段文本具体如何切分并不一定固定为预期形式。例如“机器”可能被切分为：

```text
[机器]
```

也可能被切分为：

```text
[机, 器]
```

因此需要区分：

- 词表中某个 Token 的 ID 是固定的；
- 文本被切分成哪些 Token，可能受到 Tokenizer 规则和上下文影响。

不同模型之间的 Token ID 通常不可直接互换。即使两个模型出现相同的 ID，也不能据此判断它们代表相同的 Token；必须确认词表和 Tokenizer 是否兼容。

### 4. Token 的语言倾向性

Token 不是一种独立的自然语言，也不是完整的“模型语言”。它更像是经过编码的文本碎片。

抽象意义上的 Token 可以跨语言使用，但具体 Tokenizer 和词表会受到训练语料、语言结构和编码设计影响，因此具有语言倾向性。例如：

- 中文可能按单字、词语或短语切分；
- 英文单词可能被拆成多个子词；
- 代码中的符号、关键字和缩进可能分别成为 Token；
- 同一段中文、英文或代码在不同模型中可能产生不同数量的 Token。

更准确的表述是：**Token 是跨语言的编码单位，但具体 Tokenizer 和词表具有语言倾向性。**

### 5. Embedding 的作用

Token ID 只是编号，没有大小或距离意义。例如：

```text
猫 → 100
狗 → 101
飞机 → 102
```

这些编号并不表示“猫”和“狗”比“猫”和“飞机”更相似。模型不能直接依靠这些整数理解语言关系。

Embedding 通常是一个可训练矩阵，形状可以表示为：

```text
词表大小 × 向量维度
```

Token ID 可以看作矩阵的行号。模型根据 Token ID 取出对应行，得到一个向量：

```text
Token ID 582
  ↓
Embedding 矩阵第 582 行
  ↓
[0.12, -0.45, 0.88, ...]
```

这些向量可以参与矩阵运算、相似度计算，并通过梯度下降进行学习和更新。训练后，语义或语法关系相近的 Token 往往会形成有用的向量结构。

### 6. Token Embedding 与上下文表示

Token Embedding 是 Token 的初始向量。例如“银行”在不同语境中可能有不同含义：

```text
我去银行取钱。
河流的银行很陡。
```

这两个句子中的“银行”可能使用相同的初始 Token Embedding，但经过 Transformer 和 Attention 结合上下文处理后，会得到不同的隐藏状态。

需要区分：

- **Token Embedding**：Token 的初始向量；
- **Hidden State**：结合上下文后的动态语义表示；
- **Output Logits**：模型对下一个 Token 的预测分数。

### 7. 核心总结

> Token 是文本切分后的离散片段；Token ID 是词表中的编号；Embedding 将 Token ID 映射为可训练的向量；Transformer 再利用上下文将这些初始向量转换为动态语义表示。
