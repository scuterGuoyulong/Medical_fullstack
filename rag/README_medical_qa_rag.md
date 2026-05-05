# 医学 QA RAG 使用说明

本方案面向 `instruction/input/output` JSONL，每条 QA 作为一个 chunk 全量入库。

## 依赖

```bash
pip install faiss-cpu sentence-transformers
```

## 1. 构建 FAISS 知识库

```bash
cd /home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl

export HF_ENDPOINT=https://hf-mirror.com

python medical_fullstack/rag/build_medical_qa_faiss.py \
  --data_jsonl DataSets/medical/finetune/train_zh_0_sample100_rewrite_llm.jsonl \
  --out_dir medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss \
  --embedding_model BAAI/bge-m3 \
  --device cuda:0
```

输出目录包含：

- `index.faiss`：FAISS `IndexFlatIP`，向量已归一化。
- `docs.jsonl`：每条 QA chunk 及 metadata。
- `config.json`：模型、chunk 策略、维度等配置。

## 2. 检索并生成 RAG Prompt

```bash
python medical_fullstack/rag/medical_qa_rag.py \
  --index_dir medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss \
  --query "宫颈糜烂用什么手术治疗比较好？" \
  --top_k 8 \
  --rerank_top_n 3 \
  --device cuda:0 \
  --reranker_device cuda:0 \
  --print_prompt_only
```

流程为：

1. `BAAI/bge-m3` 对 query 向量化。
2. FAISS 取 `top_k=8`。
3. `BAAI/bge-reranker-v2-m3` 重排。
4. 取 Top3 拼入医学安全 Prompt。
5. Prompt 要求仅基于参考资料回答，并在末尾标注“参考依据：片段1、片段2”。

如需临时关闭 reranker：

```bash
python medical_fullstack/rag/medical_qa_rag.py \
  --index_dir medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss \
  --query "低 T3 综合征的并发症是什么？" \
  --no_rerank
```

## 3. 检索评测

```bash
python medical_fullstack/rag/eval_medical_qa_rag.py \
  --index_dir medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss \
  --data_jsonl DataSets/medical/finetune/train_zh_0_sample100_rewrite_llm.jsonl \
  --max_k 8 \
  --device cuda:0 \
  --reranker_device cuda:0
```

评测方式：

- query：`instruction`
- gold：同一行 QA 的 `doc_id`
- 指标：`Recall@1`、`Recall@3`、`Recall@5`

## 4. 可选：回答规则审计

如果你有生成结果 JSONL，每行包含 `question/query/instruction` 和 `answer/text/prediction`，可以做规则审计：

```bash
python medical_fullstack/rag/eval_medical_qa_rag.py \
  --index_dir medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss \
  --answers_jsonl output/rag_answers.jsonl
```

审计规则包括：

- 回答和检索上下文覆盖度过低，标记为可能不忠实。
- 绝对化表述，如“确诊为”“一定是”“无需检查”。
- 涉及用药、剂量、停药、换药，但没有“遵医嘱/医生指导”。
- 涉及胸痛、呼吸困难、意识障碍、大出血、孕产急症等，但没有及时就医提示。

## 5. FastAPI 服务接入

新的 FAISS QA RAG 使用 `RAG_INDEX_DIR`：

```bash
export VLM_MODEL_PATH=/path/to/your/qwen-vl-model
export RAG_INDEX_DIR=/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/medical_fullstack/rag/indexes/medical_qa_bge_m3_faiss
export RAG_EMBED_DEVICE=cuda:0
export RAG_RERANKER_DEVICE=cuda:0

bash medical_fullstack/serve/run_vlm_rag_server.sh
```

旧的 `RAG_INDEX_PKL` 仍然保留；如果同时设置，服务优先使用新的 `RAG_INDEX_DIR`。
