import os
from pathlib import Path
from typing import Optional, List
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# 加载 .env 文件
load_dotenv()


# 项目根路径
PROJECT_PATH = Path("C:/Users/Administrator/Desktop/llm_app/rag")

# 数据目录
DATA_DIR = PROJECT_PATH / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"


BAAI_MODEL_DIR = DATA_DIR / "llm" / "BAAI"

# 注意：本地实际目录名是 bge-large-zh-v1___5（下划线），需要与这里保持一致
AVAILABLE_MODELS = ['bge-large-zh-v1___5']
MODEL_NAME = AVAILABLE_MODELS[0]
MODEL_PATH = BAAI_MODEL_DIR / MODEL_NAME
# Chroma集合
CHROMA_COLLECTION_NAME = "law_documents"

print("=" * 60)
print("路径配置验证")
print("=" * 60)
print(f"项目路径: {PROJECT_PATH}")
print(f"数据目录: {DATA_DIR}")
print(f"向量数据库目录: {CHROMA_PERSIST_DIR}")
print(f"BAAI模型目录: {BAAI_MODEL_DIR}")
print(f"当前模型路径: {MODEL_PATH}")
print(f"模型路径是否存在: {MODEL_PATH.exists()}")

# 如果模型路径不存在，创建目录
if not MODEL_PATH.exists():
    print(f"\n模型路径不存在，正在创建目录...")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"已创建目录: {MODEL_PATH}")
    print(f"请将模型文件放入此目录")
else:
    print(f"模型路径已存在")
    
    # 检查模型文件
    model_files = list(MODEL_PATH.glob("*"))

print("=" * 60)
print()  

class DeepSeekChatModel(ChatOpenAI):
    """DeepSeek 专用聊天模型类（基于 ChatOpenAI 封装）"""

    def __init__(self, **kwargs):
        # 从.env或环境变量读取配置
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model_name = kwargs.pop("model", None) or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 模型参数
        temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "4096"))
        timeout = int(os.getenv("DEEPSEEK_TIMEOUT", "60"))
        max_retries = int(os.getenv("DEEPSEEK_MAX_RETRIES", "3"))
        streaming = os.getenv("DEEPSEEK_STREAMING", "false").lower() == "true"
        
        # 额外参数
        top_p = float(os.getenv("DEEPSEEK_TOP_P", "0.95"))
        frequency_penalty = float(os.getenv("DEEPSEEK_FREQUENCY_PENALTY", "0.0"))
        presence_penalty = float(os.getenv("DEEPSEEK_PRESENCE_PENALTY", "0.0"))

        print(f"初始化 DeepSeek 对话模型: {model_name}")

        super().__init__(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            streaming=streaming,
            model_kwargs={
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            },
            **kwargs
        )


class BGELocalEmbeddings(Embeddings):
    """
    使用本地 BGE 模型的 Embeddings 类
    从指定的本地路径加载模型
    """

    def __init__(self, model_path: Optional[Path] = None, device: str = "cpu"):
        """
        Args:
            model_path: BGE 模型本地路径
            device: 运行设备，"cpu" 或 "cuda"
        """
        # 使用传入的模型路径，默认为上面配置的路径
        self.model_path = model_path or MODEL_PATH
        
        # 检查模型路径是否存在且非空
        if not self.model_path.exists():
            raise FileNotFoundError(f"BGE 模型路径不存在: {self.model_path}")
        if not any(self.model_path.iterdir()):
            raise FileNotFoundError(
                f"BGE 模型目录已创建但为空: {self.model_path}。"
                f" 请从 HuggingFace 下载 'BAAI/bge-large-zh-v1.5' "
                f"并保存到该目录，或者在 Python 中运行：\n"
                f"from sentence_transformers import SentenceTransformer\n"
                f"model = SentenceTransformer('BAAI/bge-large-zh-v1.5')\n"
                f"model.save(r'{self.model_path}')"
            )
        
        self.device = device
        self.batch_size = int(os.getenv("BGE_BATCH_SIZE", "32"))
        
        print(f"正在从本地加载 BGE 模型: {self.model_path}")
        print(f"设备: {self.device}, 批处理大小: {self.batch_size}")
        
        # 加载模型
        try:
            self.model = SentenceTransformer(str(self.model_path), device=self.device)
            # 获取向量维度
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"本地模型加载成功！向量维度: {self.dimension}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            time.sleep(100)
            raise

        
        # 添加缓存
        self._cache = {}
        self._call_count = 0
        self._total_texts = 0

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """使用本地 BGE 模型获取 embeddings"""
        if not texts:
            return []
        
        self._call_count += 1
        self._total_texts += len(texts)
        
        # 去重处理
        unique_texts = list(set(texts))
        
        # 检查缓存
        uncached_texts = []
        cached_embeddings = {}
        
        for text in unique_texts:
            if text in self._cache:
                cached_embeddings[text] = self._cache[text]
            else:
                uncached_texts.append(text)
        
        # 计算未缓存的文本
        if uncached_texts:
            try:
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                
                for i, text in enumerate(uncached_texts):
                    embedding = embeddings[i].tolist()
                    self._cache[text] = embedding
                    cached_embeddings[text] = embedding
                    
            except Exception as e:
                print(f"BGE 模型计算失败: {e}")
                raise
        
        # 按原始顺序返回
        result = []
        for text in texts:
            if text in cached_embeddings:
                result.append(cached_embeddings[text])
            else:
                result.append([0.0] * self.dimension)
        
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        if not texts:
            return []
        
        if len(texts) <= self.batch_size:
            return self._embed(texts)
        
        results = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._embed(batch)
            results.extend(batch_results)
            print(f"处理批次 {i//self.batch_size + 1}/{total_batches}")
        
        return results

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        if not text:
            return []
        result = self._embed([text])
        return result[0] if result else []
    
    def clear_cache(self):
        """清空缓存"""
        cache_size = len(self._cache)
        self._cache.clear()
        print(f"已清空 embeddings 缓存，共 {cache_size} 条")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "dimension": self.dimension,
            "cache_size": len(self._cache),
            "call_count": self._call_count,
            "total_texts_processed": self._total_texts,
        }


class LLMManager:
    """大语言模型管理器"""
    
    _llm_instance = None
    _embeddings_instance = None

    @classmethod
    def get_deepseek_model(cls, model_name: Optional[str] = None, force_recreate: bool = False) -> DeepSeekChatModel:
        """获取 DeepSeek 对话模型"""
        if cls._llm_instance is not None and not force_recreate:
            return cls._llm_instance
        
        try:
            model = DeepSeekChatModel(model=model_name) if model_name else DeepSeekChatModel()
            cls._llm_instance = model
            return model
        except Exception as e:
            print(f"初始化 DeepSeek 模型失败: {e}")
            raise

    @classmethod
    def get_bge_embeddings(cls, force_recreate: bool = False) -> BGELocalEmbeddings:
        """获取 BGE 本地嵌入模型"""
        if cls._embeddings_instance is not None and not force_recreate:
            return cls._embeddings_instance
        
        try:
            embeddings = BGELocalEmbeddings()
            cls._embeddings_instance = embeddings
            return embeddings
        except Exception as e:
            print(f"初始化 BGE Embeddings 失败: {e}")
            raise


def get_llm(model_name: Optional[str] = None, force_recreate: bool = False) -> DeepSeekChatModel:
    """获取 DeepSeek 聊天模型"""
    return LLMManager.get_deepseek_model(model_name, force_recreate)


def get_embeddings(force_recreate: bool = False) -> BGELocalEmbeddings:
    """获取 BGE 本地嵌入模型"""
    return LLMManager.get_bge_embeddings(force_recreate)


if __name__ == "__main__":
    """测试模块"""
    print("=" * 60)
    print("LLM 模块测试")
    print("=" * 60)
    
    # 测试1：DeepSeek 对话模型
    print("\n1. 测试 DeepSeek 模型加载...")
    try:
        chat_model = get_llm()
        print(f"对话模型: {chat_model.model_name}")
    except Exception as e:
        print(f"失败: {e}")

    # 测试2：BGE 本地嵌入模型
    print("\n2. 测试 BGE 本地嵌入模型加载...")
    try:
        embedding_model = get_embeddings()
        print(f"嵌入模型加载成功")
        print(f"模型路径: {embedding_model.model_path}")
        print(f"向量维度: {embedding_model.dimension}")
        
        # 测试嵌入
        test_text = "测试文本"
        embedding = embedding_model.embed_query(test_text)
        print(f"测试嵌入维度: {len(embedding)}")
        
        # 显示模型统计信息
        stats = embedding_model.get_stats()
        print(f"\n模型统计信息:")
        print(f"缓存大小: {stats['cache_size']}")
        print(f"调用次数: {stats['call_count']}")
        
        time.sleep(100)
    except Exception as e:
        print(f"失败: {e}")
        time.sleep(100)