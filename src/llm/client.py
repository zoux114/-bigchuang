"""
LLM API 客户端
支持 OpenAI 兼容 API (DeepSeek, 智谱, 通义千问等)
"""
import os
from typing import Optional, List, Dict

from config.settings import LLM_CONFIG, PROMPT_TEMPLATE


class LLMClient:
    """LLM API 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API Key
            base_url: API Base URL
            model: 模型名称
        """
        self.api_key = api_key or LLM_CONFIG["api_key"]
        self.base_url = base_url or LLM_CONFIG["base_url"]
        self.model = model or LLM_CONFIG["model"]
        self.temperature = LLM_CONFIG["temperature"]
        self.max_tokens = LLM_CONFIG["max_tokens"]

        if not self.api_key:
            raise ValueError(
                "请设置 API Key。可以在 config/.env 文件中设置 LLM_API_KEY"
            )

        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")

        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 生成温度
            max_tokens: 最大生成 token 数

        Returns:
            模型回复
        """
        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        return response.choices[0].message.content

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        单轮对话

        Args:
            prompt: 用户输入
            system_prompt: 系统提示

        Returns:
            模型回复
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages)

    def rag_query(
        self,
        question: str,
        context: str,
        prompt_template: Optional[str] = None,
    ) -> str:
        """
        RAG 查询

        Args:
            question: 用户问题
            context: 检索到的上下文
            prompt_template: Prompt 模板

        Returns:
            模型回复
        """
        template = prompt_template or PROMPT_TEMPLATE
        prompt = template.format(context=context, question=question)

        return self.complete(prompt)


def test_llm_client():
    """测试 LLM 客户端"""
    try:
        client = LLMClient()

        # 简单测试
        response = client.complete("你好，请用一句话介绍自己。")
        print(f"回复: {response}")

    except ValueError as e:
        print(f"配置错误: {e}")
    except Exception as e:
        print(f"请求失败: {e}")


if __name__ == "__main__":
    test_llm_client()
