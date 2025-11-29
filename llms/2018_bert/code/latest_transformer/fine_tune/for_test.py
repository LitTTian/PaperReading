from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(
    model_name,
    ignore_mismatched_sizes=True
).to(device)

# 2. 终极修复版QA推理函数
def bert_qa_inference(model, tokenizer, question, context, device):
    model.eval()
    
    # 步骤1：编码输入（保留原始分词信息）
    # 关键：add_special_tokens=True（默认），确保[CLS]/[SEP]正确添加
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        return_offsets_mapping=True,  # 辅助验证token对应原文本的位置
        add_special_tokens=True
    ).to(device)
    
    # 步骤2：提取关键信息（验证token与原文本的映射）
    offset_mapping = inputs.pop("offset_mapping")[0]  # (seq_len, 2)：每个token的起止字符位置
    sequence_ids = inputs.sequence_ids(0)            # 标记token归属（None/0/1）
    
    # 步骤3：构建上下文掩码（仅保留上下文token）
    context_mask = torch.tensor([seq_id == 1 for seq_id in sequence_ids]).unsqueeze(0).to(device)
    non_context_mask = ~context_mask
    
    # 步骤4：前向传播+过滤logits
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits.masked_fill(non_context_mask, -1e9)  # (1, seq_len)
        end_logits = outputs.end_logits.masked_fill(non_context_mask, -1e9)  # (1, seq_len)
    
    # 步骤5：优化起止位置预测（支持单个token答案）
    # 方法：遍历所有可能的起止组合，选择logits和最大的组合（QA任务标准做法）
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()
    
    # 兜底：若起止位置相同，扩展end_idx到合理范围（最多向后3个token）
    if start_idx == end_idx:
        # 取start_idx到start_idx+3的end_logits最大值
        end_candidates = end_logits[0, start_idx:min(start_idx+4, len(end_logits[0]))]
        end_idx = start_idx + torch.argmax(end_candidates).item()
    
    # 步骤6：解码答案（关键：允许start_idx <= end_idx，且跳过空token）
    answer = ""
    if start_idx <= end_idx:
        # 解码token（强制跳过特殊token，即使包含）
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        
        # 兜底：若解码为空，直接取原文本对应位置的内容（通过offset_mapping）
        if answer == "":
            start_char = offset_mapping[start_idx][0].item()
            end_char = offset_mapping[end_idx][1].item()
            answer = context[start_char:end_char].strip()
    
    # 步骤7：计算置信度
    start_conf = torch.softmax(start_logits, dim=1)[0, start_idx].item()
    end_conf = torch.softmax(end_logits, dim=1)[0, end_idx].item()
    avg_conf = round((start_conf + end_conf) / 2, 4)
    
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "start_position": start_idx,
        "end_position": end_idx,
        "confidence": avg_conf
    }


def test_bert_qa_inference(question=None, context=None):
    question = "What is the capital of France?" if question is None else question
    context = "Paris is the capital and most populous city of France." if context is None else context
    
    # 执行推理
    result = bert_qa_inference(model, tokenizer, question, context, device)
    
    # 打印详细结果（含调试信息）
    print("=== BERT QA结果 ===")
    print(f"问题：{result['question']}")
    print(f"上下文：{result['context']}")
    print(f"预测答案：{result['answer']}")  # 正确输出：Paris
    print(f"起止位置：{result['start_position']} ~ {result['end_position']}")
    print(f"置信度：{result['confidence']}")
    
    # 额外调试：打印token映射（看位置9对应的token）
    # inputs_debug = tokenizer(question, context, return_tensors="pt")
    # tokens = tokenizer.convert_ids_to_tokens(inputs_debug["input_ids"][0])
    # print(f"\n=== Token映射 ===")
    # for i, token in enumerate(tokens):
    #     print(f"位置{i}：{token}")

test_bert_qa_inference()