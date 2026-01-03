import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,  # 复用图文处理器（简化数据预处理）
    AutoModelForVisionAndLanguageGeneration,  # 也可手动拆分组件
    CLIPVisionModel,  # 视觉编码器（CLIP ViT）
    AutoModelForCausalLM,  # 语言模型（Qwen2/LLaMA）
    CLIPImageProcessor,
    AutoTokenizer
)

class CustomLLaVA(nn.Module):
    def __init__(
        self,
        vision_encoder_path: str,  # 视觉编码器路径（如 openai/clip-vit-large-patch14）
        llm_path: str,  # 语言模型路径（如 Qwen/Qwen2-7B-Instruct）
        vision_hidden_dim: int = 768,  # 视觉编码器输出维度（CLIP ViT-L 为 768）
        llm_hidden_dim: int = 4096,  # 语言模型输入维度（Qwen2-7B 为 4096）
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device

        # 1. 加载视觉编码器（提取图片特征）
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder_path).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_path)

        # 2. 定义投影层（映射视觉特征到语言模型维度）
        self.vision_language_projector = nn.Sequential(
            nn.Linear(vision_hidden_dim, llm_hidden_dim),  # 维度映射
            nn.GELU(),  # 激活函数
            nn.Linear(llm_hidden_dim, llm_hidden_dim)  # 特征增强
        ).to(self.device)

        # 3. 加载语言模型（因果生成核心）
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

        # 4. 定义图片标记（<image>）：用于文本序列中占位
        self.image_token = "<image>"
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        self.llm.resize_token_embeddings(len(self.tokenizer))  # 扩展词表

    def extract_image_features(self, images):
        """步骤1：提取图片特征并映射维度"""
        # 图片预处理
        pixel_values = self.image_processor(
            images, return_tensors="pt", padding=True
        ).pixel_values.to(self.device)

        # 视觉编码器提取特征（取 CLS 或 patch 特征，此处取最后一层隐藏状态）
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            # vision_hidden_states: [batch_size, num_patches, vision_hidden_dim]
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # 取 CLS 特征 [batch_size, vision_hidden_dim]

        # 投影层映射到语言模型维度
        projected_vision_features = self.vision_language_projector(vision_features)  # [batch_size, llm_hidden_dim]
        return projected_vision_features

    def merge_text_image_features(self, text_input: str, images):
        """步骤2：融合文本特征和视觉特征"""
        # 1. 提取图片特征
        image_features = self.extract_image_features(images)  # [1, llm_hidden_dim]（单张图片）

        # 2. 文本编码（包含 <image> 占位符）
        input_text = f"{self.image_token} {text_input}"
        text_inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # 3. 获取文本嵌入
        text_embeddings = self.llm.get_input_embeddings()(text_inputs.input_ids)  # [1, seq_len, llm_hidden_dim]

        # 4. 替换 <image> 占位符为实际视觉特征
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        image_token_positions = (text_inputs.input_ids == image_token_id).nonzero(as_tuple=True)[1]

        # 替换对应位置的嵌入
        for pos in image_token_positions:
            text_embeddings[:, pos, :] = image_features

        return text_embeddings, text_inputs.attention_mask

    def generate(self, text_input: str, images, max_new_tokens: int = 40):
        """步骤3：图文融合生成回复（模拟你的推理逻辑）"""
        self.eval()  # 评估模式
        with torch.no_grad():
            # 融合图文特征
            input_embeddings, attention_mask = self.merge_text_image_features(text_input, images)

            # 语言模型生成
            outputs = self.llm.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                torch_dtype=torch.float16
            )

            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 去除输入部分，只保留生成内容
            input_text = f"{self.image_token} {text_input}"
            generated_response = generated_text.replace(input_text, "").strip()
            return generated_response