# 1. 任务目标
使用LangChain在HuggingFace文档上构建RAG，实现特定知识库的问答。

# 2. 具体流程
## 2.1 知识库准备
- 构建知识库

`ds = datasets.load_dataset("m-ric/huggingface_doc",split="train")`
- 使用LangchainDocument封装内容（字段：page_content、metadata）

`RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"],metadata={"source":doc["source"]}) for doc in tqdm(ds)
]`

- 对文档作初步分块和tokenize处理用于统计文档tokens长度，绘制直方图
- 文档分块处理流程：选用`EMBEDDING_MODEL_NAME = "thenlper/gte-small"
`，对RAW_KNOWLEDGE_BASE作文档分块操作。具体地，使用RecursiveCharacterTextSplitter对多个文档作tokenize和分块处理，最后删除重复的内容。

## 2.2 构建向量数据库
作用：将文档分块经过embedding后存入向量数据库，用户的一个query经过同一embedding再通过相似性搜索返回数据库中最接近的文档。关键：搜索算法、距离度量

- 搜索算法，使用Facebook的FAISS
- 距离度量。通常有3种：余弦相似度、点积、欧氏距离。这里使用余弦相似度。
- （可选）使用pacmap和plotly可视化embedding向量。

## 2.3 LLM
将用户的query和检索出的文本合并形成一个prompt，输入到LLM生成答案。这里使用`"HuggingFaceH4/zephyr-7b-beta"`。prompt方面需要提供一个LLM的聊天模板。

## 2.4 重排Rerank（补充）
通常，当query在数据库中检索出多个文档后，还需要对检索出的文档进行重排，从而得到更精确的文档。这里使用

`from ragatouille import RAGPretrainedModel`

`RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")`

作为重排模型。

# 3. 工作流
准备好文档知识库，并通过embedding模型，构建向量数据库。

query->检索数据库->重排->(query+documents)->prompt->LLM->answer
