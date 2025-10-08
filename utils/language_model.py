import copy
import os
import threading
import time
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from contextlib import contextmanager

import numpy as np
import requests
import vllm
import torch
import re
import regex
from tqdm import tqdm
import json
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.read_utils import load_json, markdown_code_pattern
from utils.concurrency_manager import (
    ConcurrencyManager, EnhancedConnectionPool, 
    create_enhanced_llm_wrapper, get_global_concurrency_manager
)


class RequestQueue:
    """Thread-safe request queue with priority support"""
    def __init__(self, max_size=1000):
        self._queue = queue.PriorityQueue(maxsize=max_size)
        self._lock = RLock()
        
    def put(self, item, priority=0, timeout=None):
        """Put item in queue with priority (lower number = higher priority)"""
        with self._lock:
            self._queue.put((priority, time.time(), item), timeout=timeout)
    
    def get(self, timeout=None):
        """Get item from queue"""
        with self._lock:
            priority, timestamp, item = self._queue.get(timeout=timeout)
            return item
    
    def empty(self):
        with self._lock:
            return self._queue.empty()
    
    def qsize(self):
        with self._lock:
            return self._queue.qsize()


class ConnectionPool:
    """Connection pool manager for vLLM clients"""
    def __init__(self, base_url, pool_size=5, max_retries=3, timeout=30):
        self.base_url = base_url
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._pool = queue.Queue(maxsize=pool_size)
        self._lock = RLock()
        self._created_clients = 0
        self._healthy_clients = set()
        
        # Initialize pool with clients
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            try:
                client = self._create_client()
                if client:
                    self._pool.put(client)
                    self._healthy_clients.add(id(client))
            except Exception as e:
                print(f"Failed to create client: {e}")
    
    def _create_client(self):
        """Create a new OpenAI client"""
        try:
            client = OpenAI(base_url=self.base_url, api_key="EMPTY", timeout=self.timeout)
            self._created_clients += 1
            return client
        except Exception as e:
            print(f"Error creating client: {e}")
            return None
    
    @contextmanager
    def get_client(self):
        """Get client from pool with context manager"""
        client = None
        try:
            # Try to get client from pool
            try:
                client = self._pool.get(timeout=1.0)
                if id(client) not in self._healthy_clients:
                    # Client is marked as unhealthy, create new one
                    client = self._create_client()
            except queue.Empty:
                # Pool is empty, create new client
                client = self._create_client()
            
            if client is None:
                raise Exception("Failed to get healthy client")
                
            yield client
            
        except Exception as e:
            # Mark client as unhealthy
            if client and id(client) in self._healthy_clients:
                self._healthy_clients.discard(id(client))
            # Create replacement client for pool
            replacement = self._create_client()
            if replacement:
                self._healthy_clients.add(id(replacement))
                client = replacement
            raise e
        finally:
            # Return client to pool if healthy
            if client and id(client) in self._healthy_clients:
                try:
                    self._pool.put_nowait(client)
                except queue.Full:
                    pass  # Pool is full, client will be garbage collected
    
    def health_check(self):
        """Check health of connections"""
        healthy_count = 0
        total_clients = []
        
        # Collect all clients from pool
        while not self._pool.empty():
            try:
                client = self._pool.get_nowait()
                total_clients.append(client)
            except queue.Empty:
                break
        
        # Test each client
        for client in total_clients:
            try:
                # Simple health check - try to get models
                response = client.models.list()
                if response:
                    healthy_count += 1
                    self._healthy_clients.add(id(client))
                    self._pool.put(client)
                else:
                    self._healthy_clients.discard(id(client))
            except Exception:
                self._healthy_clients.discard(id(client))
        
        return healthy_count, len(total_clients)


class LLM(object):
    def __init__(self, llm_path, batch_size=16, top_k=50, top_p=1.0, temperature=0.1, max_tokens=8192, memory_size=3, task_info=None, use_reflection=True, gpu_ids=None, tensor_parallel_size=1, gpu_memory_utilization=0.7, agent_id="default"):
        self.use_reflection = use_reflection
        self.gpu_ids = gpu_ids  # 指定使用的GPU ID列表
        self.tensor_parallel_size = tensor_parallel_size
        self.agent_id = agent_id  # Agent identifier for concurrency tracking
        
        # Enhanced concurrency management
        self.concurrency_manager = get_global_concurrency_manager()
        self.request_queue = RequestQueue(max_size=1000)
        self.connection_pool = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_worker")
        self._shutdown = False
        # 串行化 vLLM chat 调用，避免 EngineCore 路由器协议错乱
        self._vllm_call_lock = RLock()
        self.gpu_memory_utilization = gpu_memory_utilization
        # GPU内存管理 - 计数器（累计计数，非并发态）
        self.current_inference_count = 0
        # 并发安全：活动中的推理请求数量（用于热重载等待）
        # 注意：与上面的累计计数不同，这个计数在每次推理开始/结束时增减
        self.active_inference_count = 0
        # 模型热重载同步：在重载期间阻塞新的推理，等待在途推理完成
        try:
            import threading  # 确保可用
            self._reload_condition = threading.Condition()
            self._reload_in_progress = False
        except Exception:
            self._reload_condition = None
            self._reload_in_progress = False
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.token_usage_log = []
        self.tokenizer, self.model, self.generation_kwargs, self.use_api = self.initialize_llm(llm_path, top_k, top_p, temperature, max_tokens)
        
        # Handle different model path formats
        if "qwen" in llm_path.lower() or "dashscope" in llm_path.lower():
            # For qwen models, use the model name directly
            self.llm_name = llm_path
            self.institute_name = "alibaba"
            self.provider_name = "dashscope"
        elif "/" in llm_path:
            # For other models with path format like "org/model"
            llm_name = llm_path.split("/")[-1]
            self.institute_name = llm_path.split("/")[-2] if len(llm_path.split("/")) > 1 else "unknown"
            self.provider_name = llm_path.split("/")[0]
            self.llm_name = llm_name
        else:
            # Simple model name
            self.llm_name = llm_path
            self.institute_name = "unknown"
            self.provider_name = "unknown"
            
        self.batch_size = batch_size
        self.task_info = task_info

        # memory initialization
        self.memory, self.memory_count, self.memory_size = self.initialize_memory(memory_size)

        # prompt template
        (self.system_prompt, self.overall_template, self.data_analysis_type_descriptions,
         self.data_analysis_type_selection_template, self.data_analysis_template, self.decision_making_template,
         self.self_reflection_template, self.memory_update_template) = self.initialize_prompt_template()
        
        # Multi-agent prompt templates
        self.regional_coordination_template = None
        self.global_macro_guidance_template = None
        self.macro_planning_template = None
        self.inter_agent_communication_template = None
        self.hybrid_decision_template = None
        self._initialize_multi_agent_templates()

        # data analysis type initialization
        self.data_analysis_types = None

    def initialize_llm(self, llm_path, top_k, top_p, temperature, max_tokens):
        # init LLM
        use_api = False
        generation_kwargs = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # 判断是否为本地模型路径（优先检查）
        if os.path.exists(llm_path):
            # 本地模型路径存在，使用vLLM加载
            print(f"检测到本地模型路径: {llm_path}")
            print(f"正在使用vLLM加载本地模型: {llm_path}")
            
            # 配置vLLM参数以支持指定GPU
            # 限制max_model_len以节省KV cache内存，同时保持在10000以上
            # Reduce effective max length to handle long prompts
            effective_max_len = max(min(max_tokens, 8192), 6144)  # 在6144-8192之间，更紧凑以处理长prompt
            
            vllm_kwargs = {
                "model": llm_path,
                "gpu_memory_utilization": max(0.85, self.gpu_memory_utilization - 0.05),  # 降低内存利用率防止CUDA错误
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": effective_max_len,  # 限制序列长度以节省KV cache
                "enforce_eager": True,
                "trust_remote_code": True,
                "swap_space": 4,  # 增加swap space到6GB缓解内存压力
                "disable_log_stats": True,  # 减少日志开销
                "max_num_seqs": 128,  # 限制并发序列数减少内存占用
                "block_size": 16  # 减小block size节约内存
            }
            
            print(f"调整序列长度: {max_tokens} -> {effective_max_len} (A800优化)")
            
            # 使用CUDA_VISIBLE_DEVICES环境变量来指定GPU
            if self.gpu_ids is not None:
                if isinstance(self.gpu_ids, (list, tuple)):
                    if len(self.gpu_ids) == 1:
                        # 单GPU模式，设置CUDA_VISIBLE_DEVICES
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids[0])
                        vllm_kwargs["tensor_parallel_size"] = 1
                        print(f"A800配置: 使用GPU {self.gpu_ids[0]} (单卡模式)")
                    else:
                        # 多GPU模式，使用tensor parallel
                        gpu_ids_str = ','.join(map(str, self.gpu_ids))
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
                        vllm_kwargs["tensor_parallel_size"] = len(self.gpu_ids)
                        print(f"A800配置: 使用GPUs {gpu_ids_str} (tensor parallel)")
                else:
                    # 单个GPU ID
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids)
                    vllm_kwargs["tensor_parallel_size"] = 1
                    print(f"A800配置: 使用GPU {self.gpu_ids}")
            else:
                # 使用默认GPU
                vllm_kwargs["tensor_parallel_size"] = 1
                print("A800配置: 使用默认GPU")
            
            print(f"A800优化vLLM配置: {vllm_kwargs}")
            print("正在初始化vLLM引擎(A800优化)...这可能需要几分钟")
            
            try:
                llm_model = vllm.LLM(**vllm_kwargs)
                print(f"vLLM模型加载成功！使用GPU: {self.gpu_ids if self.gpu_ids else 'auto'}")
                use_api = False
                # Create SamplingParams object for local model - THIS FIXES THE ERROR
                generation_kwargs = vllm.SamplingParams(**generation_kwargs)
            except Exception as e:
                print(f"vLLM模型加载失败: {e}")
                raise
        elif llm_path.startswith(('http://', 'https://')):
            # URL形式，使用增强连接池管理
            self.connection_pool = EnhancedConnectionPool(
                llm_path, 
                pool_size=3, 
                timeout=60,  # Reduced from 6000 to reasonable timeout
                concurrency_manager=self.concurrency_manager
            )
            llm_model = None  # 使用连接池而不是固定客户端
            use_api = True
            print(f"初始化增强连接池管理器: {llm_path}")
        elif "o4-mini" in llm_path.lower() or "o4mini" in llm_path.lower():
            # OpenAI o4-mini via OpenAI-Hub (国内镜像站)
            llm_model = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://api.openai-hub.com/v1"
            )
            use_api = True
            print(f"初始化OpenAI o4-mini API (OpenAI-Hub镜像): {llm_path}")
        elif "openai" in llm_path.lower() or "siliconflow" in llm_path.lower():
            llm_model = OpenAI()
            use_api = True
        elif "qwen-" in llm_path.lower() or "dashscope" in llm_path.lower():
            # 通义千问API (OpenAI兼容模式) - 使用qwen-而不是qwen避免路径误判
            llm_model = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            use_api = True
        elif "deepseek" in llm_path.lower():
            # DeepSeek API (OpenAI兼容模式)
            llm_model = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            use_api = True
            print(f"初始化DeepSeek API: {llm_path}")
        else:
            # 其他情况（模型名称等），尝试作为远程模型使用vLLM加载
            print(f"尝试使用vLLM加载模型: {llm_path}")
            # 限制max_model_len以节省KV cache内存，同时保持在10000以上
            # Reduce effective max length to handle long prompts
            effective_max_len = max(min(max_tokens, 8192), 6144)  # 在6144-8192之间，更紧凑以处理长prompt
            
            vllm_kwargs = {
                "model": llm_path,
                "gpu_memory_utilization": max(0.85, self.gpu_memory_utilization - 0.05),  # 降低内存利用率防止CUDA错误
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": effective_max_len,  # 限制序列长度以节省KV cache
                "enforce_eager": True,
                "trust_remote_code": True,
                "swap_space": 4,  # 增加swap space到6GB缓解内存压力
                "disable_log_stats": True,  # 减少日志开销
                "max_num_seqs": 128,  # 限制并发序列数减少内存占用
                "block_size": 16  # 减小block size节约内存
            }
            
            print(f"调整序列长度: {max_tokens} -> {effective_max_len} (A800优化)")
            
            # 使用CUDA_VISIBLE_DEVICES环境变量来指定GPU
            if self.gpu_ids is not None:
                if isinstance(self.gpu_ids, (list, tuple)):
                    if len(self.gpu_ids) == 1:
                        # 单GPU模式，设置CUDA_VISIBLE_DEVICES
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids[0])
                        vllm_kwargs["tensor_parallel_size"] = 1
                        print(f"A800配置: 使用GPU {self.gpu_ids[0]} (单卡模式)")
                    else:
                        # 多GPU模式，使用tensor parallel
                        gpu_ids_str = ','.join(map(str, self.gpu_ids))
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
                        vllm_kwargs["tensor_parallel_size"] = len(self.gpu_ids)
                        print(f"A800配置: 使用GPUs {gpu_ids_str} (tensor parallel)")
                else:
                    # 单个GPU ID
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_ids)
                    vllm_kwargs["tensor_parallel_size"] = 1
                    print(f"A800配置: 使用GPU {self.gpu_ids}")
            else:
                # 使用默认GPU
                vllm_kwargs["tensor_parallel_size"] = 1
                print("A800配置: 使用默认GPU")
            
            print(f"A800优化vLLM配置: {vllm_kwargs}")
            print("正在初始化vLLM引擎(A800优化)...这可能需要几分钟")
            
            try:
                llm_model = vllm.LLM(**vllm_kwargs)
                print(f"vLLM模型加载成功！使用GPU: {self.gpu_ids if self.gpu_ids else 'auto'}")
                use_api = False
                # Create SamplingParams object for remote/fallback model - THIS FIXES THE ERROR
                generation_kwargs = vllm.SamplingParams(**generation_kwargs)
            except Exception as e:
                print(f"vLLM模型加载失败: {e}")
                raise

        return None, llm_model, generation_kwargs, use_api

    def initialize_prompt_template(self):
        system_prompt = load_json("./prompts/system_prompt.json")["template"]

        if not self.task_info:
            return system_prompt, None, None, None, None, None, None, None

        # Overall
        overall_template = load_json("./prompts/agent_prompt_template.json")["template"]
        overall_template = overall_template.replace("<task_description>", self.task_info["task_description"])
        overall_template = overall_template.replace("<data_schema>", self.task_info["data_schema"])
        overall_template = overall_template.replace("<domain_knowledge>", self.task_info["domain_knowledge"])

        # Analysis Type Descriptions
        data_analysis_type_descriptions = load_json("./prompts/data_analysis_type_descriptions.json")

        # Data analysis type selection
        data_analysis_type_selection_template = load_json("./prompts/data_analysis_type_selection_template.json")["template"]

        # Data analysis
        data_analysis_template = load_json("./prompts/data_analysis_template.json")["template"]

        # Decision-making
        decision_making_template = load_json("./prompts/decision_making_template.json")["template"]
        decision_making_template = decision_making_template.replace("<task_target>", self.task_info["task_target"])

        # self-reflection
        self_reflection_template = load_json("./prompts/self_reflection_template.json")["template"]
        self_reflection_template = self_reflection_template.replace("<task_target>", self.task_info["task_target"])
        self_reflection_template = self_reflection_template.replace("<task_output_type>", self.task_info["task_output_type"])

        # memory update
        memory_update_template = load_json("./prompts/memory_update_template.json")["template"]
        memory_update_template = memory_update_template.replace("<memory_num>", str(self.memory_size))

        return (system_prompt, overall_template, data_analysis_type_descriptions, data_analysis_type_selection_template,
                data_analysis_template, decision_making_template, self_reflection_template, memory_update_template)


    def initialize_data_analysis_types(self, data_analysis_types):
        self.data_analysis_types = data_analysis_types

    def initialize_memory(self, memory_size):
        memory = list()
        memory_count = 0

        return memory, memory_count, memory_size
    
    def save_token_usage(self, output_file="token_usage.json"):
        """保存token usage统计到JSON文件"""
        usage_data = {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens,
            'agent_id': self.agent_id,
            'model_name': self.llm_name,
            'detailed_log': self.token_usage_log
        }
        
        with open(output_file, 'w') as f:
            json.dump(usage_data, f, indent=2)
        
        print(f"\n=== Token Usage Summary ===")
        print(f"Agent ID: {self.agent_id}")
        print(f"Model: {self.llm_name}")
        print(f"Total Prompt Tokens: {self.total_prompt_tokens}")
        print(f"Total Completion Tokens: {self.total_completion_tokens}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Token usage saved to: {output_file}")
        print(f"=========================\n")
        
        return usage_data
    
    def _compress_memory(self):
        """记忆系统优化 - 压缩和限制记忆长度"""
        if len(self.memory) > self.memory_size:
            # 保留最重要的记忆，压缩旧记忆
            self.memory = self.memory[-self.memory_size:]
    
    def _get_compressed_memory_text(self):
        """获取压缩的记忆文本，限制到200字符以内"""
        if not self.memory:
            return "N/A"
        
        memory_text = ""
        for exp in self.memory:
            memory_text += f"- {exp}\n"
        memory_text = memory_text[:-1] if memory_text else "N/A"
        
        # 限制到200字符以内
        if len(memory_text) > 200:
            memory_text = memory_text[:197] + "..."
        
        return memory_text
    
    def _build_sampling_params(self):
        """确保返回合法的 vLLM SamplingParams，修正类型与可选项，避免协议不一致。
        - 统一显式字段：top_k(int), top_p(float), temperature(float), max_tokens(int)
        - 避免潜在的 stop/stop_token_ids 单值类型引起的协议报文不一致
        """
        try:
            import vllm as _vllm
        except Exception:
            return self.generation_kwargs
        # 已经是 SamplingParams
        if isinstance(self.generation_kwargs, getattr(_vllm, 'SamplingParams', object)):
            return self.generation_kwargs
        # 字典或可映射对象时，提取并规范类型
        params_dict = {}
        try:
            params_dict = dict(self.generation_kwargs)
        except Exception:
            # 回退：按属性访问
            for key in ("top_k", "top_p", "temperature", "max_tokens", "stop", "stop_token_ids"):
                if hasattr(self.generation_kwargs, key):
                    params_dict[key] = getattr(self.generation_kwargs, key)
        top_k = int(params_dict.get("top_k", 50) or 50)
        top_p = float(params_dict.get("top_p", 1.0) or 1.0)
        temperature = float(params_dict.get("temperature", 0.1) or 0.1)
        max_tokens = int(params_dict.get("max_tokens", 512) or 512)
        # 严格规范潜在问题字段
        stop = params_dict.get("stop", None)
        stop_token_ids = params_dict.get("stop_token_ids", None)
        if isinstance(stop, (str, int)):
            # 单值一律转为列表[str]
            stop = [str(stop)] if isinstance(stop, (str, int)) else None
        elif stop is not None:
            # 其他类型统一转字符串列表
            try:
                stop = [str(x) for x in stop]
            except Exception:
                stop = None
        if isinstance(stop_token_ids, int):
            stop_token_ids = [int(stop_token_ids)]
        elif stop_token_ids is not None:
            try:
                stop_token_ids = [int(x) for x in stop_token_ids]
            except Exception:
                stop_token_ids = None
        try:
            return _vllm.SamplingParams(
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                stop_token_ids=stop_token_ids,
                n=1,
                best_of=1,
            )
        except Exception:
            # 最后的回退：直接返回原对象
            return self.generation_kwargs

    def _sanitize_messages(self, messages):
        """将 chat messages 规范为 vLLM 期望的 [[{role, content}, ...], ...]，并保证 content 为字符串。
        任何非字符串 content 将以 JSON 序列化为字符串，None -> 空串。
        """
        try:
            sanitized = []
            for conv in messages:
                new_conv = []
                for msg in conv:
                    role = str(msg.get("role", "user"))
                    content = msg.get("content", "")
                    if content is None:
                        content = ""
                    elif not isinstance(content, str):
                        try:
                            content = json.dumps(content, ensure_ascii=False)
                        except Exception:
                            content = str(content)
                    new_conv.append({"role": role, "content": content})
                sanitized.append(new_conv)
            return sanitized
        except Exception:
            # 失败则原样返回
            return messages

    def _validate_and_clean_decision(self, decision_dict, default_answer="1"):
        """
        Validate and clean LLM decision response to ensure required fields are present and valid.
        Enhanced with better error handling and fallback mechanisms.
        """
        # Handle None or empty inputs
        if decision_dict is None:
            return {"answer": default_answer, "summary": "Empty response, using default"}
            
        # Handle non-dict inputs  
        if not isinstance(decision_dict, dict):
            try:
                # Try to parse as JSON if it's a string
                if isinstance(decision_dict, str):
                    decision_dict = json.loads(decision_dict.strip())
                else:
                    return {"answer": default_answer, "summary": "Invalid response format, using default"}
            except:
                return {"answer": default_answer, "summary": "Cannot parse response, using default"}
        
        # Validate and clean 'answer' field
        answer = decision_dict.get("answer")
        if answer is None:
            answer = default_answer
        elif not isinstance(answer, (str, int, float)):
            answer = str(answer) if answer is not None else default_answer
        else:
            answer = str(answer).strip()
            if not answer:
                answer = default_answer
        
        # Validate and clean 'summary' field  
        summary = decision_dict.get("summary")
        if summary is None or not isinstance(summary, str) or not summary.strip():
            summary = f"Selected option {answer} (auto-generated summary)"
        else:
            summary = str(summary).strip()
            # Limit summary length to prevent memory issues
            if len(summary) > 500:
                summary = summary[:497] + "..."
        
        # Preserve other fields that might be useful
        result = {
            "answer": answer,
            "summary": summary
        }
        
        # Add other common fields if they exist and are valid
        for field in ["reasoning", "confidence", "data_analysis", "coordination_strategy"]:
            if field in decision_dict and decision_dict[field] is not None:
                try:
                    field_value = str(decision_dict[field]).strip()
                    if field_value and len(field_value) <= 1000:  # Reasonable length limit
                        result[field] = field_value
                except:
                    pass  # Skip invalid fields
        
        return result

    def update_memory(self, sample_info):
        if not self.use_reflection:
            return
        
        # 记忆系统优化 - 更新前先压缩记忆
        self._compress_memory()

        old_experience = ""
        for exp in self.memory:
            old_experience += f"- {exp}\n"
        old_experience = old_experience[:-1]

        new_experience = ""
        for s in sample_info:
            data_text, is_correct, experience = s
            new_experience += f"- {experience}\n"
        new_experience = new_experience[:-1]

        query = copy.copy(self.overall_template)

        # construct prompt
        query = query.replace("<data_text>", sample_info[0][0])
        query = query.replace("<step_instruction>", self.memory_update_template)
        query = query.replace("<memory_size>", str(self.memory_size))
        query = query.replace("<old_experience>", old_experience)
        query = query.replace("<new_experience>", new_experience)

        # replace memory
        retry_count = 0
        while retry_count < 3:
            try:
                response = self.inference(query)
                if response is None:
                    return

                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) > 0:
                    try:
                        self.memory = json.loads(possible_answer[-1])[:self.memory_size]
                    except (json.JSONDecodeError, IndexError) as e:
                        print(f"Memory update JSON parse error: {e}")
                        return
                else:
                    try:
                        self.memory = json.loads(response)[:self.memory_size]
                    except (json.JSONDecodeError, IndexError) as e:
                        print(f"Memory update fallback JSON parse error: {e}")
                        return

                return
            except Exception as e:
                print(f"Error in update memory: {e}\nTry again...")
                # print(f"=================================\n{response}")
                retry_count += 1

    def inference(self, query, system_prompt=None):
        """Enhanced inference with concurrency management and improved error handling"""
        try:
            # Use concurrency manager for controlled access
            with self.concurrency_manager.managed_request(self.agent_id) as request_id:
                return self._internal_inference(query, system_prompt, request_id)
        except Exception as e:
            print(f"Concurrency managed inference failed for {self.agent_id}: {e}")
            # Return fallback response instead of None
            return '{"answer": "1", "summary": "Concurrency management failed, using fallback"}'
    
    def _internal_inference(self, query, system_prompt=None, request_id=None):
        """Internal inference method with original logic but enhanced error handling"""
        # 在热重载期间阻塞新推理，并登记活动推理计数
        def _enter_inference():
            if getattr(self, '_reload_condition', None) is None:
                self.active_inference_count = max(0, getattr(self, 'active_inference_count', 0)) + 1
                return
            with self._reload_condition:
                while getattr(self, '_reload_in_progress', False):
                    self._reload_condition.wait(timeout=0.1)
                self.active_inference_count += 1
        def _exit_inference():
            try:
                if getattr(self, '_reload_condition', None) is None:
                    self.active_inference_count = max(0, getattr(self, 'active_inference_count', 1) - 1)
                    return
                with self._reload_condition:
                    self.active_inference_count = max(0, self.active_inference_count - 1)
                    if self.active_inference_count == 0:
                        # 通知可能等待的热重载线程
                        self._reload_condition.notify_all()
            except Exception:
                pass

        _enter_inference()
        # GPU内存管理 - 增加累计计数并定期清理
        self.current_inference_count += 1
        
        message = [
            {
                "role": "system",
                "content": system_prompt if system_prompt is not None else self.system_prompt,
            },
            {
                "role": "user",
                "content": query
            }
        ]

        if self.use_api:
            # DeepSeek API使用模型名称本身，不需要添加前缀
            # 其他API可能需要特殊处理
            llm_name = self.llm_name

            retry_count = 0
            response = None
            
            # Use enhanced connection pool if available
            if self.connection_pool is not None:
                while retry_count < 3:
                    try:
                        with self.connection_pool.get_client(self.agent_id) as client:
                            if getattr(self, 'provider_name', '') == 'dashscope':
                                api_response = client.chat.completions.create(
                                    model=llm_name,
                                    messages=message,
                                    temperature=self.generation_kwargs['temperature'],
                                    max_tokens=self.generation_kwargs['max_tokens'],
                                    extra_body={"enable_thinking": False}
                                )
                            else:
                                api_response = client.chat.completions.create(
                                    model=llm_name,
                                    messages=message,
                                    temperature=self.generation_kwargs['temperature'],
                                    max_tokens=self.generation_kwargs['max_tokens']
                                )
                            
                            response = api_response.choices[0].message.content
                            
                            # Record token usage if available
                            if hasattr(api_response, 'usage') and api_response.usage:
                                self.total_prompt_tokens += api_response.usage.prompt_tokens
                                self.total_completion_tokens += api_response.usage.completion_tokens
                                self.total_tokens += api_response.usage.total_tokens
                                self.token_usage_log.append({
                                    'timestamp': time.time(),
                                    'agent_id': self.agent_id,
                                    'prompt_tokens': api_response.usage.prompt_tokens,
                                    'completion_tokens': api_response.usage.completion_tokens,
                                    'total_tokens': api_response.usage.total_tokens
                                })
                        
                        # Validate response
                        if response is None or not response.strip():
                            raise Exception("Empty response from API")
                        
                        break
                    except Exception as e:
                        print(f"Connection pool inference error (attempt {retry_count + 1}): {e}")
                        retry_count += 1
                        if retry_count < 3:
                            # Adaptive backoff
                            backoff_delay = self.concurrency_manager.backoff.get_delay(self.agent_id)
                            time.sleep(min(backoff_delay, 10))
            else:
                # Original logic for non-pooled connections with enhanced error handling
                while retry_count < 3:
                    try:
                        if getattr(self, 'provider_name', '') == 'dashscope':
                            api_response = self.model.chat.completions.create(
                                model=llm_name,
                                messages=message,
                                temperature=self.generation_kwargs['temperature'],
                                max_tokens=self.generation_kwargs['max_tokens'],
                                extra_body={"enable_thinking": False}
                            )
                        else:
                            api_response = self.model.chat.completions.create(
                                model=llm_name,
                                messages=message,
                                temperature=self.generation_kwargs['temperature'],
                                max_tokens=self.generation_kwargs['max_tokens'],
                            )
                        
                        response = api_response.choices[0].message.content
                        
                        # Record token usage if available
                        if hasattr(api_response, 'usage') and api_response.usage:
                            self.total_prompt_tokens += api_response.usage.prompt_tokens
                            self.total_completion_tokens += api_response.usage.completion_tokens
                            self.total_tokens += api_response.usage.total_tokens
                            self.token_usage_log.append({
                                'timestamp': time.time(),
                                'agent_id': self.agent_id,
                                'prompt_tokens': api_response.usage.prompt_tokens,
                                'completion_tokens': api_response.usage.completion_tokens,
                                'total_tokens': api_response.usage.total_tokens
                            })
                        
                        # Validate response
                        if response is None or not response.strip():
                            raise Exception("Empty response from API")
                        
                        break
                    except Exception as e:
                        print(f"API inference error (attempt {retry_count + 1}): {e}")
                        retry_count += 1
                        if retry_count < 3:
                            backoff_delay = self.concurrency_manager.backoff.get_delay(self.agent_id)
                            time.sleep(min(backoff_delay, 10))
        else:
            # Enhanced vLLM handling with strict concurrency control
            retry_count = 0
            while retry_count < 2:
                try:
                    # Ensure vLLM engine availability
                    if not self._ensure_vllm_model_ready():
                        print(f"ERROR: vLLM engine unavailable after recovery attempts")
                        return '{"answer": "1", "summary": "Model not available"}'
                    
                    # 使用带超时与崩溃恢复的安全调用
                    # 统一规范消息，避免 content 非字符串
                    safe_message_list = self._sanitize_messages([message])
                    responses_gen = self._vllm_chat_with_timeout(safe_message_list, timeout_s=300)
                    
                    # Validate vLLM response structure
                    if not responses_gen or len(responses_gen) == 0:
                        raise Exception("Empty response list from vLLM")
                    
                    if not responses_gen[0].outputs or len(responses_gen[0].outputs) == 0:
                        raise Exception("Empty outputs from vLLM response")
                    
                    response = responses_gen[0].outputs[0].text
                    
                    # Additional validation
                    if response is None or not response.strip():
                        raise Exception("Empty text content in vLLM response")
                    
                    # GPU内存管理 - 每30次推理清理一次GPU内存（降低频率）
                    if self.current_inference_count % 30 == 0:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                print(f"GPU_MEMORY_CLEANUP: Cleared GPU cache after {self.current_inference_count} inferences")
                        except Exception as cleanup_error:
                            print(f"GPU_MEMORY_CLEANUP_ERROR: {cleanup_error}")
                    
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    if "CUDA error" in error_msg or "illegal memory access" in error_msg:
                        print(f"CUDA_ERROR_RECOVERY: Detected CUDA error in inference: {error_msg}")
                        # Try to clear CUDA cache
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                        except:
                            pass
                        time.sleep(2)  # Wait for GPU to stabilize
                        retry_count += 1
                        if retry_count >= 2:
                            print(f"CUDA_ERROR_FATAL: Multiple CUDA errors, returning fallback")
                            return '{"answer": "1", "summary": "CUDA errors encountered, using fallback"}'
                    else:
                        print(f"LLM_INFERENCE_ERROR: {error_msg}")
                        retry_count += 1
                        if retry_count >= 2:
                            return '{"answer": "1", "summary": "LLM inference failed, using fallback"}'
        
        # 确保活动推理计数回退
        _exit_inference()

        # Final validation
        if response is None or not response.strip():
            print(f"WARNING: Got empty response for {self.agent_id}, using fallback")
            return '{"answer": "1", "summary": "Empty response received, using fallback"}'
        
        return response

    def batch_inference(self, queries, system_prompt=None):
        """Enhanced batch inference with intelligent concurrency control"""
        if not queries:
            return []
        
        all_responses = []
        
        # A800优化：更大的批处理大小提高吞吐量
        optimal_batch_size = 12  # A800可以处理更大的batch size
        
        for batch_start in range(0, len(queries), optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]
            
            # Prepare messages for this batch
            messages = []
            for q in batch_queries:
                messages.append([
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt is not None else self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": q
                    }
                ])
            
            try:
                if self.use_api:
                    # API-based inference with enhanced connection pool
                    batch_responses = self._batch_api_inference(messages)
                else:
                    # vLLM-based inference with concurrency control
                    # 统一规范消息，避免 content 非字符串
                    safe_messages = self._sanitize_messages(messages)
                    batch_responses = self._batch_vllm_inference(safe_messages)
                
                all_responses.extend(batch_responses)
                
            except Exception as e:
                print(f"Batch inference error for {self.agent_id}: {e}")
                # Provide fallback responses for failed batch
                fallback_responses = [
                    '{"answer": "1", "summary": "Batch inference failed, using fallback"}'
                    for _ in batch_queries
                ]
                all_responses.extend(fallback_responses)
            
            # Inter-batch delay to prevent overwhelming the server
            if batch_end < len(queries):
                time.sleep(0.2)  # Small delay between batches
        
        return all_responses
    
    def _batch_api_inference(self, messages):
        """Handle API-based batch inference with controlled concurrency"""
        if self.provider_name == 'siliconflow':
            llm_name = f"{self.institute_name}/{self.llm_name}"
        else:
            llm_name = self.llm_name
        
        responses = [None for _ in range(len(messages))]
        
        if self.connection_pool is not None:
            # A800优化：API推理的并发连接数
            max_workers = min(len(messages), 4)  # A800可以支持更多并发连接
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="enhanced_batch") as executor:
                futures = []
                
                for j, message in enumerate(messages):
                    future = executor.submit(self._enhanced_threading_inference, llm_name, message, responses, j)
                    futures.append(future)
                
                # Wait for all requests with timeout
                for future in futures:
                    try:
                        future.result(timeout=45)  # Reduced timeout
                    except Exception as e:
                        print(f"Batch request timeout or error: {e}")
        else:
            # Fallback to original logic but with delays
            threads = []
            for j, message in enumerate(messages):
                thread = threading.Thread(
                    target=self._enhanced_threading_inference, 
                    args=(llm_name, message, responses, j)
                )
                threads.append(thread)
                thread.start()
                
                # Stagger thread starts to reduce load
                if j < len(messages) - 1:
                    time.sleep(0.1)
            
            for thread in threads:
                thread.join(timeout=45)
        
        # Validate and fix None responses
        for i, response in enumerate(responses):
            if response is None or not response.strip():
                responses[i] = '{"answer": "1", "summary": "Individual request failed, using fallback"}'
        
        return responses
    
    def _batch_vllm_inference(self, messages):
        """Handle vLLM batch inference with enhanced error checking"""
        try:
            # 在热重载期间阻塞新批推理
            entered = False
            if getattr(self, '_reload_condition', None) is None:
                self.active_inference_count = max(0, getattr(self, 'active_inference_count', 0)) + 1
                entered = True
            else:
                with self._reload_condition:
                    while getattr(self, '_reload_in_progress', False):
                        self._reload_condition.wait(timeout=0.1)
                    self.active_inference_count += 1
                    entered = True

            with self.concurrency_manager.managed_request(f"{self.agent_id}_vllm_batch") as request_id:
                # Ensure vLLM engine availability for batch path
                if not self._ensure_vllm_model_ready():
                    print(f"ERROR: vLLM model unavailable for {self.agent_id} (batch) after recovery")
                    return [
                        '{"answer": "1", "summary": "vLLM model unavailable"}'
                        for _ in messages
                    ]
                
                # 使用带超时与崩溃恢复的安全调用
                responses_gen = self._vllm_chat_with_timeout(messages, timeout_s=300)
                
                # Enhanced validation for batch responses
                if not responses_gen:
                    raise Exception("Empty response generator from vLLM")
                
                responses = []
                for i, res in enumerate(responses_gen):
                    try:
                        if not res.outputs or len(res.outputs) == 0:
                            response_text = '{"answer": "1", "summary": "Empty vLLM output"}'
                        else:
                            response_text = res.outputs[0].text
                            if not response_text or not response_text.strip():
                                response_text = '{"answer": "1", "summary": "Empty vLLM text"}'
                        responses.append(response_text)
                    except Exception as e:
                        print(f"Error processing vLLM response {i}: {e}")
                        responses.append('{"answer": "1", "summary": "vLLM response processing error"}')
                
                return responses
        except Exception as e:
            print(f"vLLM batch inference error for {self.agent_id}: {e}")
            # Try to proactively recover engine for subsequent calls
            try:
                self._ensure_vllm_model_ready()
            except Exception:
                pass
            return [
                '{"answer": "1", "summary": "vLLM batch inference failed"}'
                for _ in messages
            ]
        finally:
            # 退出活动推理计数
            try:
                if getattr(self, '_reload_condition', None) is None:
                    self.active_inference_count = max(0, getattr(self, 'active_inference_count', 1) - 1)
                else:
                    with self._reload_condition:
                        self.active_inference_count = max(0, self.active_inference_count - 1)
                        if self.active_inference_count == 0:
                            self._reload_condition.notify_all()
            except Exception:
                pass

    def _vllm_chat_with_timeout(self, messages, timeout_s=120):
        """对 self.model.chat 做超时与崩溃恢复包装，避免 EngineCore 挂死或协议错误导致阻塞"""
        try:
            import threading
            from queue import Queue
        except Exception:
            # 如果导入失败，回退为直接调用
            # 统一规范消息与采样参数，避免协议类型不一致，并串行化调用
            sampling_params = self._build_sampling_params()
            safe_messages = self._sanitize_messages(messages)
            with self._vllm_call_lock:
                return self.model.chat(safe_messages, use_tqdm=False, sampling_params=sampling_params)
        
        result_queue = Queue(maxsize=1)
        
        def _call_chat():
            try:
                sampling_params = self._build_sampling_params()
                safe_messages = self._sanitize_messages(messages)
                with self._vllm_call_lock:
                    res = self.model.chat(safe_messages, use_tqdm=False, sampling_params=sampling_params)
                result_queue.put((True, res))
            except Exception as e:
                result_queue.put((False, e))
        
        # 确保模型实例可用，避免 NoneType.chat
        try:
            if getattr(self, 'model', None) is None:
                self._ensure_vllm_model_ready()
        except Exception:
            pass
        if getattr(self, 'model', None) is None:
            raise RuntimeError("vLLM model is not available")

        t = threading.Thread(target=_call_chat, daemon=True)
        t.start()
        
        try:
            ok, payload = result_queue.get(timeout=timeout_s)
        except Exception:
            # 超时：尝试快速重启引擎并抛出异常让上层提供回退
            print(f"[ERROR] vLLM chat timeout after {timeout_s}s, restarting engine…")
            self._restart_vllm_engine_quick()
            raise TimeoutError(f"vLLM chat timeout after {timeout_s}s")
        
        if ok:
            return payload
        else:
            err = payload
            msg = str(err)
            # 捕捉 vLLM v1 协议/解码异常与 ZeroMQ Router 异常，触发快速重启
            if (
                "EngineCoreRequestType" in msg
                or "b'\\x00\\x00'" in msg
                or "ValidationError" in msg
                or "Expected `array`, got `int`" in msg
                or "router.cpp:166" in msg
                or "EngineArgs.__init__()" in msg
            ):
                # 引擎通信协议异常：重启以恢复
                print(f"[ERROR] vLLM EngineCore protocol error detected: {msg}. Restarting engine…")
                self._restart_vllm_engine_quick()
            raise err

    def _restart_vllm_engine_quick(self):
        """快速重启本地 vLLM 引擎；保守参数，尽量维持当前设备。失败则保持原状。"""
        try:
            import torch
            import os
            # 优雅关闭旧引擎
            try:
                if getattr(self, 'model', None) is not None and hasattr(self.model, 'shutdown'):
                    self.model.shutdown()
            except Exception as _sd:
                print(f"[WARN] vLLM.shutdown() failed: {_sd}")
            try:
                if getattr(self, 'model', None) is not None:
                    del self.model
            except Exception:
                pass
            self.model = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            
            # 选择设备
            device_id = None
            try:
                if isinstance(getattr(self, 'gpu_ids', None), (list, tuple)) and len(self.gpu_ids) > 0:
                    device_id = int(self.gpu_ids[0])
                elif isinstance(getattr(self, 'gpu_ids', None), int):
                    device_id = int(self.gpu_ids)
            except Exception:
                device_id = None
            
            vllm_kwargs = {
                "model": getattr(self, 'llm_name', None) or getattr(self, 'model_path', None),
                "gpu_memory_utilization": getattr(self, 'gpu_memory_utilization', 0.85),
                "tensor_parallel_size": getattr(self, 'tensor_parallel_size', 1),
                "trust_remote_code": True,
            }
            if device_id is not None:
                # 确保可见
                try:
                    curr = os.environ.get('CUDA_VISIBLE_DEVICES')
                    if curr:
                        curr_set = set([x.strip() for x in curr.split(',') if x.strip() != ''])
                        if str(device_id) not in curr_set:
                            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(curr_set | {str(device_id)}, key=lambda x: int(x)))
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
                except Exception:
                    pass
                # vLLM EngineArgs 不支持 device 参数；仅通过 CUDA_VISIBLE_DEVICES 约束
            
            import vllm
            new_model = vllm.LLM(**vllm_kwargs)
            self.model = new_model
            print(f"[INFO] vLLM engine restarted on device {vllm_kwargs.get('device', 'auto')}")
        except Exception as e:
            print(f"[WARN] Failed to restart vLLM engine: {e}")
    
    def _ensure_vllm_model_ready(self) -> bool:
        """确保本地 vLLM 引擎可用；若丢失则尝试快速重启与完整重建。
        返回 True 表示可用，False 表示仍不可用（上层应走回退）。"""
        # API 模式无需保证本地引擎
        if getattr(self, 'use_api', False):
            return True
        # 已经可用
        if getattr(self, 'model', None) is not None:
            return True
        # 通过条件变量串行化恢复流程，避免并发重启
        gated = False
        try:
            if getattr(self, '_reload_condition', None) is not None:
                with self._reload_condition:
                    if getattr(self, '_reload_in_progress', False):
                        # 其他线程正在重启；等待一小段时间
                        self._reload_condition.wait(timeout=2.0)
                        return getattr(self, 'model', None) is not None
                    self._reload_in_progress = True
                    gated = True
        except Exception:
            gated = False
        try:
            # 尝试快速重启
            self._restart_vllm_engine_quick()
            if getattr(self, 'model', None) is not None:
                # 确保采样参数对象有效
                try:
                    import vllm as _vllm
                    if not hasattr(self.generation_kwargs, 'temperature'):
                        # 兼容 dict -> SamplingParams
                        self.generation_kwargs = _vllm.SamplingParams(**dict(self.generation_kwargs))
                except Exception:
                    pass
                return True
            # 快速重启失败，尝试完整重建
            try:
                import vllm as _vllm
                import os as _os
                vllm_kwargs = {
                    "model": getattr(self, 'llm_name', None) or getattr(self, 'model_path', None),
                    "gpu_memory_utilization": getattr(self, 'gpu_memory_utilization', 0.85),
                    "tensor_parallel_size": getattr(self, 'tensor_parallel_size', 1),
                    "trust_remote_code": True,
                }
                # 固定设备（若配置了 gpu_ids）
                device_id = None
                try:
                    if isinstance(getattr(self, 'gpu_ids', None), (list, tuple)) and len(self.gpu_ids) > 0:
                        device_id = int(self.gpu_ids[0])
                    elif isinstance(getattr(self, 'gpu_ids', None), int):
                        device_id = int(self.gpu_ids)
                except Exception:
                    device_id = None
                if device_id is not None:
                    try:
                        curr = _os.environ.get('CUDA_VISIBLE_DEVICES')
                        if curr:
                            curr_set = set([x.strip() for x in curr.split(',') if x.strip() != ''])
                            if str(device_id) not in curr_set:
                                _os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(sorted(curr_set | {str(device_id)}, key=lambda x: int(x)))
                        else:
                            _os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
                    except Exception:
                        pass
                    # vLLM EngineArgs 不支持 device 参数；仅通过 CUDA_VISIBLE_DEVICES 约束
                new_model = _vllm.LLM(**vllm_kwargs)
                self.model = new_model
                try:
                    if not hasattr(self.generation_kwargs, 'temperature'):
                        self.generation_kwargs = _vllm.SamplingParams(**dict(self.generation_kwargs))
                except Exception:
                    pass
                print(f"[INFO] vLLM engine recreated on device {vllm_kwargs.get('device', 'auto')}")
                return True
            except Exception as _e:
                print(f"[WARN] Failed to recreate vLLM engine: {_e}")
                return False
        finally:
            if gated:
                try:
                    with self._reload_condition:
                        self._reload_in_progress = False
                        self._reload_condition.notify_all()
                except Exception:
                    pass
    
    def _enhanced_threading_inference(self, llm_name, message, response_list, m_id):
        """Enhanced threading inference with better error handling"""
        retry_count = 0
        response_list[m_id] = ""
        
        while retry_count < 2:  # Reduced retry count
            try:
                # Apply backoff delay
                if retry_count > 0:
                    backoff_delay = self.concurrency_manager.backoff.get_delay(self.agent_id)
                    time.sleep(min(backoff_delay, 5))
                
                # Use connection pool if available
                client = None
                if self.connection_pool is not None:
                    client_context = self.connection_pool.get_client(self.agent_id)
                    client = client_context.__enter__()
                else:
                    client = self.model
                
                try:
                    thinking_enabled = False
                    if "openai" == self.provider_name:
                        stream = client.chat.completions.create(
                            model=llm_name,
                            messages=message,
                            max_completion_tokens=self.generation_kwargs['max_tokens'],
                            stream=True
                        )
                    else:
                        # DashScope/OpenAI兼容流式：关闭思考模式
                        if getattr(self, 'provider_name', '') == 'dashscope':
                            stream = client.chat.completions.create(
                                model=llm_name,
                                messages=message,
                                max_tokens=self.generation_kwargs['max_tokens'] if "glm-z1-9b" not in llm_name.lower() else 8000,
                                stream=True,
                                extra_body={"enable_thinking": False}
                            )
                        else:
                            stream = client.chat.completions.create(
                                model=llm_name,
                                messages=message,
                                max_tokens=self.generation_kwargs['max_tokens'] if "glm-z1-9b" not in llm_name.lower() else 8000,
                                stream=True
                            )
                finally:
                    if self.connection_pool is not None and client is not None:
                        client_context.__exit__(None, None, None)

                collected_response = ""
                reasoning_finish_flag = False
                
                for chunk in stream:
                    try:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            token = chunk.choices[0].delta.content
                            collected_response += token
                        elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            # 关闭思考模式时忽略 reasoning_content
                            if thinking_enabled:
                                token = chunk.choices[0].delta.reasoning_content
                                collected_response += token
                    except Exception as chunk_error:
                        print(f"Error processing chunk: {chunk_error}")
                        continue
                
                # Validate collected response
                if not collected_response.strip():
                    raise Exception("Empty collected response")
                
                response_list[m_id] = collected_response
                print(f"Success [{m_id}] for {self.agent_id}")
                break

            except Exception as e:
                retry_count += 1
                print(f"Threading inference error (attempt {retry_count}) for {self.agent_id}: {e}")
                if retry_count >= 2:
                    # Provide fallback response
                    response_list[m_id] = '{"answer": "1", "summary": "Threading inference failed, using fallback"}'

    def batch_evaluation(self, llm_name, queries, system_prompt=None):
        messages = list()

        for i, q in enumerate(queries):
            messages.append(([
                {
                    "role": "system",
                    "content": system_prompt if system_prompt is not None else self.system_prompt,
                },
                {
                    "role": "user",
                    "content": q['instruction']
                },
                {
                    "role": "assistant",
                    "content": q['response']
                },
                {
                    "role": "user",
                    "content": f"The correct answer is: {q['answer']}\n\n"
                               f"Please evaluate the response and give me a score from 0 to 10 within the XML tag like: <Score>7<Score>."
                }
            ], i))

        retry_count = 0
        valid_responses = [dict() for _ in range(len(messages))]
        while retry_count < 3:
            retry_messages = []
            threads = []
            eval_responses = []
            responses = [None for _ in range(len(messages))]

            for j, message in enumerate(messages):
                thread = threading.Thread(target=self.threading_inference, args=(llm_name, message[0], responses, j,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            eval_responses.extend(responses)
            for i, res in enumerate(eval_responses):
                score_pattern = r'<Score>(.*?)</Score>'
                scores = re.findall(score_pattern, res)
                if len(scores) >= 1:
                    score = scores[-1]
                    if int(score) < 6:
                        return None
                    else:
                        valid_responses[messages[i][1]] = {
                            "instruction": messages[i][0][1]['content'],
                            "response": messages[i][0][2]['content'],
                            "answer": messages[i][0][3]['content'],
                        }
                else:
                    retry_messages.append(messages[i])

            if len(retry_messages) == 0:
                break
            else:
                messages = retry_messages
                retry_count += 1

        return valid_responses


    def data_analysis_type_selection(self, data_text):
        query = copy.copy(self.overall_template)

        query = query.replace("<data_text>", data_text)
        query = query.replace("<step_instruction>", self.data_analysis_type_selection_template)

        success_flag = False
        while not success_flag:
            responses = []
            try:
                responses = self.inference(query)

                possible_answer = regex.findall(markdown_code_pattern, responses)[-1]
                data_analysis_types = json.loads(possible_answer)
                self.initialize_data_analysis_types(data_analysis_types)

                return data_analysis_types

            except Exception as e:
                print(f"Error: {e}\nTrying again...")
                # print(f"=================================\n{responses}")

    def decision_making_pipeline(self, data_texts, data_analysis_types, answer_option_form):
        data_analysis_results = []
        for data_text in data_texts:
            # data analysis
            data_analysis_samples = list()
            for analysis_type in data_analysis_types:
                predefined_type = None
                for a_type in self.data_analysis_type_descriptions:
                    if analysis_type in a_type or a_type in analysis_type:
                        predefined_type = a_type
                analysis_description = self.data_analysis_type_descriptions[predefined_type] if predefined_type else ""
                analysis_reason = data_analysis_types[analysis_type]
                data_analysis_samples.append([
                    data_text,
                    analysis_type,
                    analysis_description,
                    analysis_reason
                ])

            data_analysis_sample_results = self.data_analysis(data_analysis_samples)
            data_analysis_text = ""
            for i, result in enumerate(data_analysis_sample_results):
                data_analysis_text += f"- {data_analysis_samples[i][1]}: {result['summary']}\n"
            data_analysis_results.append(data_analysis_text)

        # decision-making
        decision_making_samples = list()
        for i, data_text in enumerate(data_texts):
            decision_making_samples.append([
                data_text,
                data_analysis_results[i]
            ])
        decision_making_results = self.decision_making(decision_making_samples, answer_option_form)

        return decision_making_results

    def hybrid_decision_making_pipeline(self, data_texts, answer_option_form):
        # decision-making
        decision_making_samples = list()
        for i, data_text in enumerate(data_texts):
            decision_making_samples.append(data_text)
        decision_making_results = self.decision_making(decision_making_samples, answer_option_form)

        return decision_making_results

    def self_reflection_pipeline(self, data_texts, data_analyses, decisions, reasons, env_changes):
        # self-reflection
        self_reflection_samples = []
        for i, data_text in enumerate(data_texts):
            self_reflection_samples.append([data_texts[i], data_analyses[i],
                                            decisions[i], reasons[i],
                                            env_changes[i]])

        self_reflections = self.self_reflection(self_reflection_samples)

        # update memory
        memory_update_samples = []
        for i, self_reflection in enumerate(self_reflections):
            memory_update_samples.append([self_reflection["data_text"],
                                          self_reflection["is_correct"],
                                          self_reflection["experience"]])
        self.update_memory(memory_update_samples)

        return self_reflections

    def hybrid_self_reflection_pipeline(self, data_texts, decisions, reasons, env_changes, answer_option_form):
        # self-reflection
        self_reflection_samples = []
        for i, data_text in enumerate(data_texts):
            self_reflection_samples.append([data_texts[i], decisions[i],
                                            reasons[i], env_changes[i]])

        self_reflections = self.self_reflection(self_reflection_samples, answer_option_form)

        # update memory
        memory_update_samples = []
        for i, self_reflection in enumerate(self_reflections):
            memory_update_samples.append([self_reflection["data_text"],
                                          self_reflection["is_correct"],
                                          self_reflection["experience"]])
        self.update_memory(memory_update_samples)

        return self_reflections

    def data_analysis(self, sample_info):
        queries = list()

        for s in sample_info:
            data_text, analysis_type, analysis_description, analysis_reason = s
            query = copy.copy(self.overall_template)

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.data_analysis_template)

            # data analysis template
            query = query.replace("<analysis_type>", analysis_type)
            query = query.replace("<analysis_type>", analysis_type)
            query = query.replace("<analysis_description>", analysis_description)
            query = query.replace("<analysis_reason>", analysis_reason)

            queries.append(query)

        retry_count = 0
        while retry_count < 3:
            unsuccessful_count = 0
            failed_responses = list()
            responses = self.batch_inference(queries)

            data_analysis_results = list()
            for res in responses:
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)[-1]
                    data_analysis = json.loads(possible_answer)
                except Exception as e:
                    unsuccessful_count += 1
                    data_analysis = {"summary": "N/A"}
                    failed_responses.append(res)

                if "summary" not in data_analysis:
                    unsuccessful_count += 1
                    data_analysis = {"summary": "N/A"}
                    failed_responses.append(res)

                data_analysis_results.append(data_analysis)

            if unsuccessful_count / len(queries) <= 0.2:
                return data_analysis_results
            else:
                retry_count += 1
                print(f"Error in data analysis: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))

        return [{"summary": "N/A"} for _ in range(len(queries))]

    def decision_making(self, sample_info, answer_option_form):
        queries = list()

        for i, s in enumerate(sample_info):
            query = copy.copy(self.overall_template)

            if len(s) == 2:
                # data analysis
                data_text, data_analysis = s
                query = query.replace("<data_analysis>", data_analysis)
            else:
                data_text = s
                query = query.replace("<data_analysis>", "N/A")

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.decision_making_template)
            query = query.replace("<answer_option_form>", answer_option_form[i])

            # add memory - 使用压缩的记忆文本
            memory_text = self._get_compressed_memory_text()
            query = query.replace("<experience>", memory_text)

            queries.append((i, query))

        retry_count = 0
        decision_making_results = [{
            "answer": None,
            "summary": "N/A",
            "data_text": sample_info[i][0] if isinstance(sample_info[i], (list, tuple)) and len(sample_info[i]) >= 1 else sample_info[i],
            "data_analysis": sample_info[i][1] if isinstance(sample_info[i], (list, tuple)) and len(sample_info[i]) == 2 else "N/A"}
            for i in range(len(queries))
        ]

        # Fast path: single-query uses single inference to avoid vLLM batch path
        if len(queries) == 1:
            idx, single_query = queries[0]
            res = self.inference(single_query)
            try:
                possible_answer = regex.findall(markdown_code_pattern, res)
                if len(possible_answer) <= 0:
                    decision = json.loads(res)
                else:
                    decision = json.loads(possible_answer[-1])
            except Exception:
                decision = {}
            # Validate and clean
            decision = self._validate_and_clean_decision(decision, default_answer="1")
            if decision and decision.get("answer") and decision.get("summary"):
                decision.update({
                    "data_text": sample_info[idx][0] if isinstance(sample_info[idx], (list, tuple)) and len(sample_info[idx]) == 2 else sample_info[idx],
                    "data_analysis": sample_info[idx][1] if isinstance(sample_info[idx], (list, tuple)) and len(sample_info[idx]) == 2 else "N/A"
                })
                decision_making_results[idx].update(decision)
            return decision_making_results
        while retry_count < 3:
            retry_queries = []
            failed_responses = list()
            responses = self.batch_inference([q for _, q in queries])

            for i, res in enumerate(responses):
                ori_query_index = queries[i][0]

                # API Failure
                if res is None:
                    decision_making_results[ori_query_index].update({
                        "data_text": sample_info[ori_query_index][0] if isinstance(sample_info[ori_query_index], (list, tuple)) and len(sample_info[ori_query_index]) == 2 else sample_info[ori_query_index],
                        "data_analysis": sample_info[ori_query_index][1] if isinstance(sample_info[ori_query_index], (list, tuple)) and len(sample_info[ori_query_index]) == 2 else "N/A"
                    })
                    continue

                # Answer Failure
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)
                    if len(possible_answer) <= 0:
                        decision = json.loads(res)
                    else:
                        decision = json.loads(possible_answer[-1])
                except Exception as e:
                    decision = {}
                    failed_responses.append(res)

                # Use enhanced validation and cleaning
                decision = self._validate_and_clean_decision(decision, default_answer="1")
                
                # Check if we still have issues after validation
                if not decision or not decision.get("answer") or not decision.get("summary"):
                    retry_queries.append(queries[i])
                    decision = {}
                    failed_responses.append(res)
                else:
                    # Add additional context fields
                    decision.update({
                        "data_text": sample_info[ori_query_index][0] if isinstance(sample_info[ori_query_index], (list, tuple)) and len(sample_info[ori_query_index]) == 2 else sample_info[ori_query_index],
                        "data_analysis": sample_info[ori_query_index][1] if isinstance(sample_info[ori_query_index], (list, tuple)) and len(sample_info[ori_query_index]) == 2 else "N/A"
                    })
                    decision_making_results[ori_query_index].update(decision)

            if retry_queries:
                retry_count += 1
                queries = retry_queries
                print(f"Error in decision-making: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))
            else:
                return decision_making_results

        return decision_making_results

    def self_reflection(self, sample_info, answer_option_form):
        queries = list()

        for i, s in enumerate(sample_info):
            query = copy.copy(self.overall_template)
            if len(s) == 5:
                # data analysis
                data_text, data_analysis, decision, reason, env_changes = s
                query = query.replace("<data_analysis>", data_analysis)
            else:
                data_text, decision, reason, env_changes = s
                query = query.replace("<data_analysis>", "N/A")

            query = query.replace("<data_text>", data_text)
            query = query.replace("<step_instruction>", self.self_reflection_template)
            query = query.replace("<answer_option_form>", answer_option_form[i])

            # add memory - 使用压缩的记忆文本
            memory_text = self._get_compressed_memory_text()
            query = query.replace("<experience>", memory_text)

            # decision and reason
            query = query.replace("<decision_or_prediction>", str(decision))
            query = query.replace("<decision_or_prediction_summary>", str(reason))

            # env feedback
            query = query.replace("<env_changes>", env_changes)

            queries.append((i, query))

        retry_count = 0
        self_reflection_results = [{
            "is_correct": "YES",
            "answer": None,
            "experience": "N/A",
            "data_text": sample_info[i][0]}
            for i in range(len(queries))
        ]
        if not self.use_reflection:
            return self_reflection_results

        while retry_count < 3:
            retry_queries = list()
            failed_responses = list()
            responses = self.batch_inference([q for _, q in queries])
            for i, res in enumerate(responses):
                ori_query_index = queries[i][0]

                # API Failure
                if res is None:
                    self_reflection_results[ori_query_index].update({
                        "is_correct": "YES",
                        "data_text": sample_info[ori_query_index][0]
                    })
                    continue

                # Paser the response to extract the JSON object
                try:
                    possible_answer = regex.findall(markdown_code_pattern, res)
                    if len(possible_answer) > 0:
                        reflection = json.loads(possible_answer[-1])
                    else:
                        reflection = json.loads(res)
                except (json.JSONDecodeError, IndexError, Exception) as e:
                    print(f"Self reflection JSON parse error: {e}")
                    reflection = {}
                    failed_responses.append(res)

                if "is_correct" not in reflection or "answer" not in reflection or "experience" not in reflection:
                    retry_queries.append(queries[i])
                    reflection = {}
                    failed_responses.append(res)

                if reflection:
                    reflection.update({"data_text": sample_info[ori_query_index][0]})
                    self_reflection_results[ori_query_index].update(reflection)

            if retry_queries:
                retry_count += 1
                queries = retry_queries
                print(f"Error in self reflection: key missing\nTry again...")
                # print("=================================\n".join([res for res in failed_responses]))
            else:
                return self_reflection_results

        return self_reflection_results

    def evaluate(self, samples, task):
        answered_questions = copy.copy(samples)
        for s in answered_questions:
            s.update({"reasoning": None, "decision": None, "is_correct": False})
        queries = []
        spatial_temporal_results = {}
        correct_count = 0
        all_question_num = len(samples)

        for i, s in enumerate(tqdm(samples)):
            query = s["question"] if "prompt" not in s else f"{s['prompt']}\n\n{s['test_query']}"
            queries.append((len(queries), query))

            if (i + 1) % self.batch_size == 0 or i == len(samples) - 1:
                retry_count = 0
                batch_whole_query_num = len(queries)
                while retry_count < 3:
                    retry_queries = []
                    responses = self.batch_inference([q for _, q in queries])
                    for j, res in enumerate(responses):
                        ori_index = i+1-batch_whole_query_num+queries[j][0]
                        if res is None:
                            answered_questions[ori_index].update({
                                "reasoning": None,
                                "decision": None,
                                "is_correct": False
                            })
                            continue
                        if task == 'st_understanding':
                            answer_pattern = r'<Answer>(.*?)</Answer>'
                            possible_answers = re.findall(answer_pattern, res)
                            if len(possible_answers) > 0:
                                model_answer = possible_answers[-1]
                                answered_questions[ori_index].update({
                                    "reasoning": res,
                                    "decision": model_answer,
                                    "is_correct": True if model_answer == samples[ori_index]['answer'] else False
                                })
                            else:
                                retry_queries.append(queries[j])
                        else:
                            try:
                                possible_answers = regex.findall(markdown_code_pattern, res)
                                if len(possible_answers) <= 0:
                                    answer_dict = json.loads(res)
                                else:
                                    answer_dict = json.loads(possible_answers[-1])
                                answered_questions[ori_index].update({
                                    "reasoning": res,
                                    "summary": answer_dict["summary"],
                                    "decision": answer_dict['answer'],
                                    "is_correct": True if answer_dict['answer'] == samples[ori_index]['answer'] else False
                                })
                            except:
                                retry_queries.append(queries[j])

                    if retry_queries:
                        retry_count += 1
                        queries = retry_queries
                        print(f"Retrying {len(queries)} times...")
                    else:
                        break
                queries = []

        # Spatial-temporal relation results
        if task == "st_understanding":
            for sample in answered_questions:
                st_type = sample['spatial_temporal_relation']
                if st_type in spatial_temporal_results:
                    spatial_temporal_results[st_type]['num'] += 1
                    if sample['decision'] == sample['answer']:
                        spatial_temporal_results[st_type]['correct_num'] += 1
                        correct_count += 1
                else:
                    spatial_temporal_results[st_type] = {}
                    spatial_temporal_results[st_type]['num'] = 1
                    if sample['decision'] == sample['answer']:
                        spatial_temporal_results[st_type]['correct_num'] = 1
                        correct_count += 1
                    else:
                        spatial_temporal_results[st_type]['correct_num'] = 0

            for st_type in spatial_temporal_results:
                spatial_temporal_results[st_type]["accuracy"] = (spatial_temporal_results[st_type]['correct_num'] /
                                                                 spatial_temporal_results[st_type]['num'])
        else:
            for sample in answered_questions:
                correct_count += sample['is_correct']

        overall_accuracy = correct_count / all_question_num
        return answered_questions, overall_accuracy, spatial_temporal_results

    def _initialize_multi_agent_templates(self):
        """Initialize multi-agent prompt templates."""
        try:
            # Regional coordination template
            self.regional_coordination_template = load_json("./prompts/regional_coordination_template.json")["template"]
            
            # Macro planning template
            self.macro_planning_template = load_json("./prompts/macro_planning_template.json")["template"]
            
            # Global macro guidance template
            try:
                self.global_macro_guidance_template = load_json("./prompts/global_macro_guidance_template.json")["template"]
            except Exception:
                self.global_macro_guidance_template = self.decision_making_template
            
            # Inter-agent communication template
            self.inter_agent_communication_template = load_json("./prompts/inter_agent_communication_template.json")["template"]
            
            # Enhanced hybrid decision template
            self.hybrid_decision_template = load_json("./prompts/hybrid_decision_making_template.json")["template"]
            
        except Exception as e:
            print(f"Warning: Could not load multi-agent templates: {e}")
            # Use fallback templates
            self.regional_coordination_template = self.decision_making_template
            self.macro_planning_template = self.decision_making_template
            self.global_macro_guidance_template = self.decision_making_template
            self.inter_agent_communication_template = self.decision_making_template
            self.hybrid_decision_template = self.decision_making_template

    def regional_coordination_decision(self, regional_context, vehicles_data, boundary_status, 
                                     coordination_messages, traffic_predictions, route_options, region_id):
        """Make coordinated decisions for vehicles within a region."""
        try:
            # Prepare the regional coordination query
            query = copy.copy(self.overall_template)
            
            # Replace template variables with compact data - PROMPT COMPRESSION  
            regional_template = copy.copy(self.regional_coordination_template)
            regional_template = regional_template.replace("<region_id>", str(region_id))
            regional_template = regional_template.replace("<regional_context>", self._compress_data(regional_context))
            regional_template = regional_template.replace("<vehicles_data>", self._compress_data(vehicles_data))
            regional_template = regional_template.replace("<boundary_status>", self._compress_data(boundary_status))
            regional_template = regional_template.replace("<coordination_messages>", self._compress_data(coordination_messages))
            regional_template = regional_template.replace("<traffic_predictions>", self._compress_data(traffic_predictions))
            regional_template = regional_template.replace("<route_options>", self._compress_data(route_options))
            
            query = query.replace("<step_instruction>", regional_template)
            query = query.replace("<data_text>", f"Regional coordination for Region {region_id}")
            
            # Add memory context
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)
            
            # Make the decision
            response = self.inference(query)
            
            # Parse the response
            try:
                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) > 0:
                    decision_result = json.loads(possible_answer[-1])
                else:
                    decision_result = json.loads(response)
                return decision_result
            except (json.JSONDecodeError, IndexError, Exception) as e:
                print(f"Regional coordination JSON parse error: {e}")
                # Fallback parsing
                return {
                    "vehicle_decisions": [],
                    "regional_summary": "Failed to parse LLM response",
                    "boundary_load_balancing": "Unknown",
                    "inter_region_communication": "Communication failed"
                }
                
        except Exception as e:
            print(f"Regional coordination decision failed: {e}")
            return {
                "vehicle_decisions": [],
                "regional_summary": f"Error: {e}",
                "boundary_load_balancing": "Error in processing",
                "inter_region_communication": "Communication error"
            }

    def macro_route_planning(self, global_state, route_requests, regional_conditions, 
                           boundary_analysis, flow_predictions, coordination_needs, region_routes):
        """Plan macro routes between regions."""
        try:
            # Prepare the macro planning query
            query = copy.copy(self.overall_template)
            
            # Replace template variables with compact data - PROMPT COMPRESSION
            macro_template = copy.copy(self.macro_planning_template)
            macro_template = macro_template.replace("<global_state>", self._compress_data(global_state))
            macro_template = macro_template.replace("<route_requests>", self._compress_data(route_requests))
            macro_template = macro_template.replace("<regional_conditions>", self._compress_data(regional_conditions))
            macro_template = macro_template.replace("<boundary_analysis>", self._compress_data(boundary_analysis))
            macro_template = macro_template.replace("<flow_predictions>", self._compress_data(flow_predictions))
            macro_template = macro_template.replace("<coordination_needs>", self._compress_data(coordination_needs))
            macro_template = macro_template.replace("<region_routes>", self._compress_data(region_routes))
            
            query = query.replace("<step_instruction>", macro_template)
            query = query.replace("<data_text>", "Macro route planning across regions")
            
            # Add memory context
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)
            
            # Make the decision
            response = self.inference(query)
            
            # Parse the response
            try:
                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) > 0:
                    decision_result = json.loads(possible_answer[-1])
                else:
                    decision_result = json.loads(response)
                return decision_result
            except (json.JSONDecodeError, IndexError, Exception) as e:
                print(f"Macro route planning JSON parse error: {e}")
                # Fallback parsing
                return {
                    "macro_routes": [],
                    "system_optimization": "Failed to parse LLM response",
                    "load_balancing": "Unknown",
                    "conflict_resolution": "Resolution failed",
                    "regional_coordination_messages": {}
                }
                
        except Exception as e:
            print(f"Macro route planning failed: {e}")
            return {
                "macro_routes": [],
                "system_optimization": f"Error: {e}",
                "load_balancing": "Error in processing",
                "conflict_resolution": "Error in planning",
                "regional_coordination_messages": {}
            }

    def global_macro_guidance(self, global_state, regional_report, hotspots, flow_targets):
        """Generate global macro guidance for this timestamp.
        Returns a strict JSON dict with keys: priority_goals, avoid_regions, avoid_edges, reroute_suggestions, message, ttl
        """
        try:
            query = copy.copy(self.overall_template)
            guidance_template = copy.copy(self.global_macro_guidance_template)
            guidance_template = guidance_template.replace("<global_state>", self._compress_data(global_state))
            guidance_template = guidance_template.replace("<regional_report>", self._compress_data(regional_report))
            guidance_template = guidance_template.replace("<hotspots>", self._compress_data(hotspots))
            guidance_template = guidance_template.replace("<flow_targets>", self._compress_data(flow_targets))
            query = query.replace("<step_instruction>", guidance_template)
            query = query.replace("<data_text>", "Global macro guidance for current timestamp")

            # Add memory context
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)

            response = self.inference(query)
            try:
                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) > 0:
                    decision_result = json.loads(possible_answer[-1])
                else:
                    decision_result = json.loads(response)
            except (json.JSONDecodeError, IndexError, Exception):
                # Fallback minimal structure
                decision_result = {
                    "priority_goals": ["avoid congestion"],
                    "avoid_regions": [],
                    "avoid_edges": [],
                    "reroute_suggestions": [],
                    "message": "Use caution; avoid severe bottlenecks",
                    "ttl": 60
                }
            # Normalize types
            try:
                decision_result["avoid_regions"] = [int(r) for r in decision_result.get("avoid_regions", []) if str(r).strip() != ""]
            except Exception:
                decision_result["avoid_regions"] = []
            try:
                decision_result["avoid_edges"] = [str(e) for e in decision_result.get("avoid_edges", []) if str(e).strip() != ""]
            except Exception:
                decision_result["avoid_edges"] = []
            if not isinstance(decision_result.get("priority_goals", []), list):
                decision_result["priority_goals"] = [str(decision_result.get("priority_goals", "avoid congestion"))]
            if not isinstance(decision_result.get("reroute_suggestions", []), list):
                decision_result["reroute_suggestions"] = [str(decision_result.get("reroute_suggestions", ""))]
            try:
                decision_result["ttl"] = int(decision_result.get("ttl", 60))
            except Exception:
                decision_result["ttl"] = 60
            if not isinstance(decision_result.get("message", ""), str):
                decision_result["message"] = "Global macro guidance"
            return decision_result
        except Exception as e:
            print(f"Global macro guidance failed: {e}")
            return {
                "priority_goals": ["avoid congestion"],
                "avoid_regions": [],
                "avoid_edges": [],
                "reroute_suggestions": [],
                "message": f"Error: {e}",
                "ttl": 60
            }

    def inter_agent_communication(self, communication_context, sender_info, recipient_info, 
                                message_content, system_context):
        """Facilitate communication between agents."""
        try:
            # Prepare the communication query
            query = copy.copy(self.overall_template)
            
            # Replace template variables with actual data
            comm_template = copy.copy(self.inter_agent_communication_template)
            comm_template = comm_template.replace("<communication_context>", str(communication_context))
            comm_template = comm_template.replace("<sender_info>", str(sender_info))
            comm_template = comm_template.replace("<recipient_info>", str(recipient_info))
            comm_template = comm_template.replace("<message_content>", str(message_content))
            comm_template = comm_template.replace("<system_context>", str(system_context))
            
            query = query.replace("<step_instruction>", comm_template)
            query = query.replace("<data_text>", "Inter-agent communication coordination")
            
            # Add memory context
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
            query = query.replace("<experience>", memory_text)
            
            # Make the decision
            response = self.inference(query)
            
            # Parse the response
            try:
                possible_answer = regex.findall(markdown_code_pattern, response)
                if len(possible_answer) > 0:
                    decision_result = json.loads(possible_answer[-1])
                else:
                    decision_result = json.loads(response)
                return decision_result
            except (json.JSONDecodeError, IndexError, Exception) as e:
                print(f"Inter-agent communication JSON parse error: {e}")
                # Fallback parsing
                return {
                    "message_interpretation": "Failed to parse LLM response",
                    "coordination_opportunities": [],
                    "conflict_resolution": {"conflicts_identified": [], "resolution_strategy": "", "trade_offs": ""},
                    "response_messages": {"to_sender": "", "to_other_agents": {}},
                    "system_impact": "Unknown",
                    "follow_up_actions": []
                }
                
        except Exception as e:
            print(f"Inter-agent communication failed: {e}")
            return {
                "message_interpretation": f"Error: {e}",
                "coordination_opportunities": [],
                "conflict_resolution": {"conflicts_identified": [], "resolution_strategy": "", "trade_offs": ""},
                "response_messages": {"to_sender": "", "to_other_agents": {}},
                "system_impact": "Communication error",
                "follow_up_actions": []
            }

    def enhanced_hybrid_decision_making_pipeline(self, data_texts, answer_option_forms, decision_type="regional",
                                               decision_context=None, system_state=None, agent_communication=None,
                                               regional_coordination=None, traffic_predictions=None):
        """Enhanced hybrid decision making for multi-agent coordination."""
        try:
            decision_making_samples = []
            
            for i, data_text in enumerate(data_texts):
                # Prepare the enhanced hybrid query
                query = copy.copy(self.overall_template)
                
                # Replace template variables with actual data
                hybrid_template = copy.copy(self.hybrid_decision_template)
                hybrid_template = hybrid_template.replace("<decision_type>", decision_type)
                hybrid_template = hybrid_template.replace("<decision_context>", str(decision_context) if decision_context else "Standard traffic coordination")
                hybrid_template = hybrid_template.replace("<system_state>", str(system_state) if system_state else "Current system status")
                hybrid_template = hybrid_template.replace("<agent_communication>", str(agent_communication) if agent_communication else "No current communications")
                hybrid_template = hybrid_template.replace("<regional_coordination>", str(regional_coordination) if regional_coordination else "Standard regional coordination")
                hybrid_template = hybrid_template.replace("<traffic_predictions>", str(traffic_predictions) if traffic_predictions else "No predictions available")
                
                query = query.replace("<step_instruction>", hybrid_template)
                query = query.replace("<data_text>", data_text)
                query = query.replace("<answer_option_form>", answer_option_forms[i] if i < len(answer_option_forms) else "")
                
                # Add memory context
                memory_text = ""
                for exp in self.memory:
                    memory_text += f"- {exp}\n"
                memory_text = memory_text[:-1] if memory_text else "N/A"
                query = query.replace("<experience>", memory_text)
                
                decision_making_samples.append((i, query))
            
            # Batch process the decisions
            decision_making_results = []
            
            retry_count = 0
            while retry_count < 3:
                retry_queries = []
                responses = self.batch_inference([q for _, q in decision_making_samples])
                
                for i, res in enumerate(responses):
                    ori_query_index = decision_making_samples[i][0]
                    
                    if res is None:
                        decision_making_results.append({
                            "answer": None,
                            "summary": "API failure",
                            "data_analysis": "N/A",
                            "coordination_strategy": "Failed to process",
                            "system_impact": "Unknown",
                            "confidence": "LOW"
                        })
                        continue
                    
                    # Parse the response
                    try:
                        possible_answer = regex.findall(markdown_code_pattern, res)
                        if len(possible_answer) > 0:
                            decision = json.loads(possible_answer[-1])
                        else:
                            decision = json.loads(res)
                    except (json.JSONDecodeError, IndexError, Exception) as e:
                        print(f"Enhanced hybrid decision JSON parse error: {e}")
                        decision = {}
                        retry_queries.append(decision_making_samples[i])
                    
                    # Validate required fields
                    required_fields = ["answer", "summary"]
                    if not all(field in decision for field in required_fields):
                        retry_queries.append(decision_making_samples[i])
                        decision = {}
                    
                    if decision:
                        # Ensure all expected fields are present
                        decision.setdefault("data_analysis", "N/A")
                        decision.setdefault("coordination_strategy", "Standard coordination")
                        decision.setdefault("system_impact", "Local optimization")
                        decision.setdefault("confidence", "MEDIUM")
                        decision_making_results.append(decision)
                
                if retry_queries:
                    retry_count += 1
                    decision_making_samples = retry_queries
                    print(f"Enhanced hybrid decision making retry {retry_count}: {len(retry_queries)} queries")
                else:
                    break
            
            return decision_making_results
            
        except Exception as e:
            print(f"Enhanced hybrid decision making failed: {e}")
            return [{
                "answer": None,
                "summary": f"Error: {e}",
                "data_analysis": "Processing failed",
                "coordination_strategy": "Error in coordination",
                "system_impact": "Unknown impact",
                "confidence": "LOW"
            } for _ in data_texts]

    def _compress_data(self, data):
        """
        Compress data for prompt efficiency without losing decision-critical information.
        Removes verbose natural language while preserving key decision factors.
        """
        if data is None:
            return "N/A"
        
        if isinstance(data, dict):
            if not data:
                return "{}"
            
            # For dictionaries, extract only essential key-value pairs
            compressed_items = []
            for k, v in data.items():
                # Compress key names
                key = str(k).replace("_", "").replace(" ", "")[:8]
                
                # Compress values based on type
                if isinstance(v, (int, float)):
                    value = f"{v:.2f}" if isinstance(v, float) else str(v)
                elif isinstance(v, str):
                    value = v[:20]  # Truncate long strings
                elif isinstance(v, (list, tuple)):
                    value = f"[{len(v)}items]" if len(v) > 3 else str(v)
                else:
                    value = str(v)[:15]
                
                compressed_items.append(f"{key}:{value}")
            
            return "{" + ",".join(compressed_items) + "}"
        
        elif isinstance(data, (list, tuple)):
            if not data:
                return "[]"
            
            # For lists, show structure and key elements
            if len(data) <= 3:
                return str(data)
            else:
                # Show first and last elements with count
                return f"[{data[0]}...{data[-1]}({len(data)}total)]"
        
        elif isinstance(data, str):
            if len(data) <= 50:
                return data
            else:
                # Keep essential parts, remove filler words
                compressed = data.replace(" the ", " ").replace(" and ", "&").replace(" with ", "w/")
                return compressed[:50] + "..." if len(compressed) > 50 else compressed
        
        else:
            # For other types, convert to string and truncate
            str_data = str(data)
            return str_data[:30] + "..." if len(str_data) > 30 else str_data


class LocalLLMManager:
    """
    管理本地共享LLM实例的管理器
    用于创建和管理两个共享LLM：traffic_llm和regional_llm
    """
    
    def __init__(self, model_path: str, task_info=None):
        self.model_path = model_path
        self.task_info = task_info
        self.traffic_llm = None
        self.regional_llm = None
        self.traffic_llm_raw = None
        self.regional_llm_raw = None
        
        print(f"\n=== 初始化本地LLM管理器 ===")
        print(f"模型路径: {model_path}")
        
        # 自动注册到全局注册表 - 修复热重载访问问题
        self._register_to_global_registry()
    
    def _register_to_global_registry(self):
        """注册到全局LLM管理器注册表 - 增强版本"""
        try:
            import sys
            import os
            
            # 确保multi_agent_env模块可以被导入
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            print(f"[DEBUG] 尝试导入multi_agent_env模块，当前路径: {parent_dir}")
            
            # 尝试多种方法导入和注册
            registration_success = False
            
            # 方法1: 直接导入multi_agent_env
            try:
                import multi_agent_env
                print(f"[DEBUG] 成功导入multi_agent_env模块")
                
                # 检查全局注册表是否存在
                if hasattr(multi_agent_env, '_global_llm_manager_registry'):
                    print(f"[DEBUG] 发现现有注册表，当前键: {list(multi_agent_env._global_llm_manager_registry.keys())}")
                    multi_agent_env._global_llm_manager_registry["current"] = self
                    registration_success = True
                    print(f"✓ LocalLLMManager已注册到全局注册表 (键: 'current')")
                else:
                    print(f"[DEBUG] 创建新的全局注册表")
                    multi_agent_env._global_llm_manager_registry = {"current": self}
                    registration_success = True
                    print(f"✓ 创建并注册到全局LLM管理器注册表")
                
                # 验证注册是否成功
                if registration_success:
                    test_manager = multi_agent_env._global_llm_manager_registry.get("current")
                    if test_manager is self:
                        print(f"✓ 注册验证成功，管理器对象匹配")
                    else:
                        print(f"✗ 注册验证失败，对象不匹配")
                        registration_success = False
                        
            except ImportError as import_error:
                print(f"[DEBUG] 导入multi_agent_env失败: {import_error}")
                registration_success = False
            except Exception as reg_error:
                print(f"[DEBUG] 注册过程中发生错误: {reg_error}")
                registration_success = False
            
            # 方法2: 如果直接导入失败，尝试通过sys.modules访问
            if not registration_success:
                print(f"[DEBUG] 尝试通过sys.modules访问multi_agent_env")
                if 'multi_agent_env' in sys.modules:
                    multi_agent_env_module = sys.modules['multi_agent_env']
                    if hasattr(multi_agent_env_module, '_global_llm_manager_registry'):
                        multi_agent_env_module._global_llm_manager_registry["current"] = self
                        registration_success = True
                        print(f"✓ 通过sys.modules注册成功")
                    else:
                        multi_agent_env_module._global_llm_manager_registry = {"current": self}
                        registration_success = True
                        print(f"✓ 通过sys.modules创建并注册成功")
                else:
                    print(f"[DEBUG] multi_agent_env不在sys.modules中")
            
            # 方法3: 创建本地fallback注册表
            if not registration_success:
                print(f"[DEBUG] 创建本地fallback注册表")
                globals()['_local_llm_manager_instance'] = self
                globals()['_global_llm_manager_registry'] = {"current": self}
                print(f"✓ 创建本地注册表作为fallback")
                registration_success = True
            
            if registration_success:
                print(f"🎉 LocalLLMManager注册完成！")
                return True
            else:
                print(f"✗ 所有注册方法都失败了")
                return False
                
        except Exception as e:
            print(f"✗ 注册过程发生严重错误: {e}")
            # 最后的fallback
            globals()['_local_llm_manager_instance'] = self
            print(f"✓ 使用紧急fallback注册")
            return False
        
    def initialize_llms(self):
        """初始化两个共享LLM实例，集成并发控制"""
        print("\n=== 初始化共享LLM实例（增强版） ===")
        
        # 设置CUDA内存管理环境变量，防止多vLLM实例冲突
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # 正常模式下设为0，调试时可设为1
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"  # 使用FlashAttention
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # 设置CUDA架构避免编译问题
        # 热重载暂存设备与训练/推理卡对齐：GPU 2,3（若不可见则回退CPU）
        os.environ.setdefault("LLM_HOT_RELOAD_STAGING", "gpu")
        os.environ.setdefault("LLM_HOT_RELOAD_GPUS", "2,3")
        print("[INFO] 已设置CUDA内存管理和attention backend环境变量")
        
        # Initialize global concurrency manager
        concurrency_manager = get_global_concurrency_manager()
        print(f"[INFO] 全局并发管理器已初始化 (最大并发: {concurrency_manager.max_concurrent_requests})")
        
        # 初始化Traffic LLM (使用GPU 2)
        print("正在初始化 Traffic LLM (GPU 2)...")
        traffic_llm_raw = LLM(
            llm_path=self.model_path,
            batch_size=4,  # Reduced batch size for better control
            top_k=50,
            top_p=1.0,
            temperature=0.1,
            max_tokens=8192,  # Reduced to match vLLM config
            memory_size=3,
            task_info=self.task_info,
            use_reflection=True,
            gpu_ids=[2],  # 使用GPU 2
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            agent_id="traffic_llm"  # Add agent identifier
        )
        
        # Store both raw and wrapped instances
        self.traffic_llm_raw = traffic_llm_raw
        self.traffic_llm = create_enhanced_llm_wrapper(traffic_llm_raw, "traffic_llm", concurrency_manager)
        print("[SUCCESS] Traffic LLM 初始化完成（已集成并发控制）")
        
        # 初始化Regional LLM (使用GPU 3)
        print("正在初始化 Regional LLM (GPU 3)...")
        regional_llm_raw = LLM(
            llm_path=self.model_path,
            batch_size=4,  # Reduced batch size for better control
            top_k=50,
            top_p=1.0,
            temperature=0.1,
            max_tokens=8192,  # Reduced to match vLLM config
            memory_size=3,
            task_info=self.task_info,
            use_reflection=True,
            gpu_ids=[3],  # 使用GPU 3
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            agent_id="regional_llm"  # Add agent identifier
        )
        
        # Store both raw and wrapped instances
        self.regional_llm_raw = regional_llm_raw
        self.regional_llm = create_enhanced_llm_wrapper(regional_llm_raw, "regional_llm", concurrency_manager)
        print("[SUCCESS] Regional LLM 初始化完成（已集成并发控制）")
        
        print("\n=== A800双GPU LLM实例初始化完成 ===")
        print("- Traffic LLM: A800 GPU 2 (85%内存利用率)")
        print("- Regional LLM: A800 GPU 3 (85%内存利用率)")
        print("- 优化特性: FlashInfer + 分块预填充 + 前缀缓存")
        print("- 批处理大小: 4 (针对A800优化)")
        print(f"- 并发管理器: 最大{concurrency_manager.max_concurrent_requests}并发")
        
        # 训练使用的GPU应在主进程设置，这里不再覆盖，避免与推理配置混淆
        print(f"[INFO] 推理GPU固定: Traffic->GPU2, Regional->GPU3; 训练GPU由主进程环境变量控制")
        
        # Print concurrency manager status
        concurrency_manager.print_status()
        
        return self.traffic_llm, self.regional_llm
    
    def get_traffic_llm(self):
        """获取Traffic LLM实例"""
        if self.traffic_llm is None:
            raise ValueError("Traffic LLM 尚未初始化，请先调用 initialize_llms()")
        return self.traffic_llm
    
    def get_regional_llm(self):
        """获取Regional LLM实例"""
        if self.regional_llm is None:
            raise ValueError("Regional LLM 尚未初始化，请先调用 initialize_llms()")
        return self.regional_llm
    
    def get_traffic_llm_raw(self):
        """获取Traffic LLM原始实例（无并发控制）"""
        if self.traffic_llm_raw is None:
            raise ValueError("Traffic LLM 原始实例尚未初始化，请先调用 initialize_llms()")
        return self.traffic_llm_raw
    
    def get_regional_llm_raw(self):
        """获取Regional LLM原始实例（无并发控制）"""
        if self.regional_llm_raw is None:
            raise ValueError("Regional LLM 原始实例尚未初始化，请先调用 initialize_llms()")
        return self.regional_llm_raw
    
    def get_gpu_status(self):
        """获取GPU使用状态（修复了可见性问题）"""
        try:
            import torch
            import subprocess
            
            # 使用nvidia-smi获取真实的GPU信息
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_info = {
                        'total_gpus': 0,
                        'gpu_status': []
                    }
                    
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 4:
                                gpu_id = int(parts[0])
                                name = parts[1]
                                memory_used = float(parts[2]) / 1024  # 转换为GB
                                memory_total = float(parts[3]) / 1024  # 转换为GB
                                
                                status = {
                                    'gpu_id': gpu_id,
                                    'name': name,
                                    'memory_allocated': f"{memory_used:.2f}GB",
                                    'memory_total': f"{memory_total:.2f}GB",
                                    'usage': f"{memory_used/memory_total*100:.1f}%"
                                }
                                
                                # 分配标签
                                if gpu_id == 0:
                                    status['assignment'] = 'Traffic LLM'
                                elif gpu_id == 1:
                                    status['assignment'] = 'Regional LLM'
                                else:
                                    status['assignment'] = '推理加速'
                                    
                                gpu_info['gpu_status'].append(status)
                                gpu_info['total_gpus'] += 1
                    
                    return gpu_info
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # 如果nvidia-smi失败，使用PyTorch方法（可能不准确）
            gpu_info = {
                'total_gpus': torch.cuda.device_count(),
                'gpu_status': []
            }
            
            for i in range(torch.cuda.device_count()):
                try:
                    device = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_total = device.total_memory / 1024**3  # GB
                    
                    status = {
                        'gpu_id': i,
                        'name': device.name,
                        'memory_allocated': f"{memory_allocated:.2f}GB",
                        'memory_total': f"{memory_total:.2f}GB",
                        'usage': f"{memory_allocated/memory_total*100:.1f}%"
                    }
                    
                    if i == 0:
                        status['assignment'] = 'Traffic LLM'
                    elif i == 1:
                        status['assignment'] = 'Regional LLM'
                    else:
                        status['assignment'] = '推理加速'
                        
                    gpu_info['gpu_status'].append(status)
                except Exception:
                    continue
            
            return gpu_info
            
        except Exception as e:
            return {'error': f'无法获取GPU状态: {e}'}
    
    def print_gpu_status(self):
        """打印GPU使用状态"""
        status = self.get_gpu_status()
        
        if 'error' in status:
            print(f"错误: {status['error']}")
            return
            
        print("\n=== GPU 使用状态 (真实情况) ===")
        print(f"总 GPU 数量: {status['total_gpus']}")
        print("-" * 80)
        
        for gpu in status['gpu_status']:
            print(f"GPU {gpu['gpu_id']}: {gpu['name']}")
            print(f"  分配: {gpu['assignment']}")
            print(f"  内存: {gpu['memory_allocated']} / {gpu['memory_total']} ({gpu['usage']})")
            print()
        
        if status['total_gpus'] >= 2:
            gpu0_usage = float(status['gpu_status'][0]['usage'].replace('%', ''))
            gpu1_usage = float(status['gpu_status'][1]['usage'].replace('%', ''))
            if gpu0_usage > 50 and gpu1_usage > 50:
                print("[SUCCESS] 两个LLM均已正常加载到不同GPU上")
            else:
                print("[WARNING] GPU内存使用率较低，可能存在问题")

    # ===== PROGRESSIVE TRAINING: LoRA ADAPTER MANAGEMENT =====
    
    def initialize_lora_management(self):
        """Initialize LoRA adapter management system for progressive training."""
        try:
            import threading
            from queue import Queue
            
            # LoRA adapter management state
            self.lora_adapters = {
                'traffic': {'current': None, 'pending': None, 'loaded': False},
                'regional': {'current': None, 'pending': None, 'loaded': False}
            }
            
            # Adapter update queues (thread-safe)
            self.adapter_update_queue = Queue()
            self.update_lock = threading.Lock()
            
            # Current adapter version tracking
            self.adapter_versions = {'traffic': 0, 'regional': 0}
            
            print("\n=== LoRA适配器管理系统初始化完成 ===")
            return True
            
        except Exception as e:
            print(f"LoRA适配器管理初始化失败: {e}")
            return False
    
    def load_lora_adapter_direct(self, llm_type: str, adapter_path: str) -> bool:
        """
        Direct LoRA adapter loading for progressive training hot-reload.
        
        Args:
            llm_type: 'traffic' or 'regional'
            adapter_path: Path to the LoRA adapter directory
            
        Returns:
            bool: Success status
        """
        try:
            if not hasattr(self, 'update_lock'):
                self.initialize_lora_management()
            
            with self.update_lock:
                print(f"\n=== 直接加载LoRA适配器 ===")
                print(f"LLM类型: {llm_type}")
                print(f"适配器路径: {adapter_path}")
                
                # Validate inputs
                if llm_type not in ['traffic', 'regional']:
                    print(f"错误: 无效的LLM类型 '{llm_type}'")
                    return False
                
                if not os.path.exists(adapter_path):
                    print(f"错误: 适配器路径不存在 '{adapter_path}'")
                    return False
                
                # Get the target LLM instance
                target_llm = self.traffic_llm if llm_type == 'traffic' else self.regional_llm
                
                if target_llm is None:
                    print(f"错误: {llm_type} LLM实例未初始化")
                    return False
                
                # Check if target LLM uses vLLM (local model)
                if target_llm.use_api:
                    print(f"错误: {llm_type} LLM使用API模式，不支持LoRA适配器")
                    return False
                
                # SAFETY: 避免在推理进行中重载；等待在途请求完成并阻塞新请求
                try:
                    if hasattr(target_llm, '_reload_condition') and target_llm._reload_condition is not None:
                        import time as _time
                        with target_llm._reload_condition:
                            # 开始重载，阻塞新推理进入
                            target_llm._reload_in_progress = True
                            start_ts = _time.time()
                            timeout_s = 120.0
                            # 额外等待并发管理器中该 Agent 的在途请求清空
                            while True:
                                remaining = timeout_s - (_time.time() - start_ts)
                                if remaining <= 0:
                                    print(f"[WARN] 等待在途推理清空超时，放弃本次直接加载，加入队列")
                                    target_llm._reload_in_progress = False
                                    target_llm._reload_condition.notify_all()
                                    try:
                                        self.queue_adapter_update(llm_type, adapter_path)
                                    except Exception:
                                        pass
                                    return False
                                # 计算该 LLM 的在途数量：本地计数 + 并发管理器活跃请求
                                local_active = int(getattr(target_llm, 'active_inference_count', 0) or 0)
                                cm_active = 0
                                try:
                                    cm = getattr(target_llm, 'concurrency_manager', None)
                                    agent_id = str(getattr(target_llm, 'agent_id', llm_type))
                                    if cm is not None and hasattr(cm, 'active_requests'):
                                        cm_active = sum(1 for _rid in list(cm.active_requests) if str(_rid).startswith(agent_id))
                                except Exception:
                                    cm_active = 0
                                if local_active <= 0 and cm_active <= 0:
                                    break
                                target_llm._reload_condition.wait(timeout=0.5)
                    else:
                        # 条件变量不可用时，保守起见进入队列
                        print(f"[INFO] {llm_type} 缺少同步条件，改为队列等待应用适配器")
                        try:
                            self.queue_adapter_update(llm_type, adapter_path)
                        except Exception:
                            pass
                        return False
                except Exception as _wait_err:
                    print(f"[WARN] 等待在途推理清空时异常: {_wait_err}")
                    try:
                        if hasattr(target_llm, '_reload_condition') and target_llm._reload_condition is not None:
                            with target_llm._reload_condition:
                                target_llm._reload_in_progress = False
                                target_llm._reload_condition.notify_all()
                    except Exception:
                        pass
                    return False
                
                # Attempt to load adapter using PEFT
                success = self._load_adapter_to_vllm_model(target_llm, adapter_path, llm_type)
                
                if success:
                    # Update tracking information
                    self.lora_adapters[llm_type]['current'] = adapter_path
                    self.lora_adapters[llm_type]['loaded'] = True
                    self.adapter_versions[llm_type] += 1
                    
                    print(f"✓ {llm_type} LoRA适配器加载成功")
                    print(f"版本: {self.adapter_versions[llm_type]}")
                    return True
                else:
                    print(f"✗ {llm_type} LoRA适配器加载失败")
                    # Strategy 3: Queue adapter for next model initialization (last resort)
                    print(f"将适配器加入待处理队列...")
                    self.lora_adapters[llm_type]['pending'] = adapter_path
                    print(f"⚠ 适配器已排队，将在下次模型初始化时应用")
                    return False
                    
        except Exception as e:
            print(f"直接LoRA适配器加载错误: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 结束热重载阻塞，允许新推理继续
            try:
                if 'target_llm' in locals() and hasattr(target_llm, '_reload_condition') and target_llm._reload_condition is not None:
                    with target_llm._reload_condition:
                        target_llm._reload_in_progress = False
                        target_llm._reload_condition.notify_all()
            except Exception:
                pass
    
    def _load_adapter_to_vllm_model(self, llm_instance, adapter_path: str, llm_type: str) -> bool:
        """
        Internal method to load LoRA adapter to vLLM model.
        
        Note: This is a workaround since vLLM doesn't support runtime LoRA loading.
        We'll implement a model replacement strategy instead.
        """
        try:
            print(f"尝试为{llm_type}模型应用LoRA适配器...")
            
            # Strategy 1: Try to use vLLM's built-in LoRA support if available
            if hasattr(llm_instance.model, 'load_lora_adapter'):
                try:
                    llm_instance.model.load_lora_adapter(adapter_path)
                    print(f"✓ 使用vLLM内置LoRA加载功能")
                    return True
                except Exception as e:
                    print(f"vLLM内置LoRA加载失败: {e}")
            
            # Strategy 2: Model re-initialization with LoRA (fallback)
            print(f"使用模型重初始化策略...")
            success = self._reinitialize_model_with_lora(llm_instance, adapter_path, llm_type)
            
            if success:
                print(f"✓ 模型重初始化策略成功")
                return True
            else:
                print(f"✗ 模型重初始化策略失败")
                
            # Strategy 3: Queue adapter for next model initialization (last resort)
            print(f"将适配器加入待处理队列...")
            self.lora_adapters[llm_type]['pending'] = adapter_path
            print(f"⚠ 适配器已排队，将在下次模型初始化时应用")
            return False  # 返回False表示本次加载失败，仅完成排队
            
        except Exception as e:
            print(f"LoRA适配器应用过程发生错误: {e}")
            return False
    
    def _reinitialize_model_with_lora(self, llm_instance, adapter_path: str, llm_type: str) -> bool:
        """
        Reinitialize the model with LoRA adapter.
        This is a heavy operation but necessary for vLLM compatibility.
        """
        try:
            import torch
            from peft import PeftModel
            import tempfile
            import shutil
            import os
            import time as _time
            
            print(f"开始重初始化{llm_type}模型...")
            
            # Save current model configuration (operate on underlying original LLM if wrapped)
            base_llm = getattr(llm_instance, 'original_llm', llm_instance)
            model_path = base_llm.llm_name if hasattr(base_llm, 'llm_name') else self.model_path
            gpu_ids = getattr(base_llm, 'gpu_ids', [0] if llm_type == 'traffic' else [1])
            
            # Step 1: Clear current model from memory (if exists)
            try:
                if hasattr(base_llm, 'model') and base_llm.model is not None:
                    # 优先调用vLLM的优雅关闭接口，释放引擎进程占用的显存
                    try:
                        if hasattr(base_llm.model, 'shutdown'):
                            base_llm.model.shutdown()
                        elif hasattr(base_llm.model, 'llm_engine') and hasattr(base_llm.model.llm_engine, 'shutdown'):
                            base_llm.model.llm_engine.shutdown()
                        print("已调用 vLLM.shutdown()，等待释放显存…")
                        _time.sleep(2.0)
                    except Exception as _sd_err:
                        print(f"[WARN] vLLM.shutdown() 调用异常: {_sd_err}")
                    try:
                        del base_llm.model
                        import gc as _gc
                        _gc.collect()
                    except Exception:
                        pass
                    base_llm.model = None
                    torch.cuda.empty_cache()
                    print(f"✓ 清理旧模型内存")
                    
                    # 等待目标GPU出现可用显存（最多60秒），按目标利用率判断
                    try:
                        target_gpu_id = gpu_ids[0] if gpu_ids else (0 if llm_type == 'traffic' else 1)
                        wait_deadline = _time.time() + 60.0
                        # 读取目标利用率配置
                        try:
                            _env_util = os.environ.get('LLM_GPU_MEMORY_UTILIZATION', '').strip()
                            _wait_util = float(_env_util) if _env_util else None
                        except Exception:
                            _wait_util = None
                        if not isinstance(_wait_util, float):
                            _wait_util = float(getattr(base_llm, 'gpu_memory_utilization', getattr(self, 'gpu_memory_utilization', 0.85)))
                        while _time.time() < wait_deadline:
                            status = self.get_gpu_status()
                            gpu_list = status.get('gpu_status', [])
                            free_gb = None
                            total_gb = None
                            for g in gpu_list:
                                if int(g.get('gpu_id', -1)) == int(target_gpu_id):
                                    try:
                                        total_gb = float(str(g.get('memory_total', '0GB')).replace('GB',''))
                                        alloc = float(str(g.get('memory_allocated', '0GB')).replace('GB',''))
                                        free_gb = max(total_gb - alloc, 0.0)
                                    except Exception:
                                        free_gb = None
                                    break
                            # 至少需要 total * utilization - 1GB 的可用空间
                            if free_gb is not None and total_gb is not None:
                                required_gb = max(0.0, total_gb * _wait_util - 1.0)
                                if free_gb >= required_gb:
                                    break
                            # 若无法获取准确信息，退化阈值到16GB
                            if free_gb is not None and total_gb is None and free_gb >= 16.0:
                                break
                            _time.sleep(0.5)
                    except Exception:
                        pass
            except Exception as del_err:
                print(f"[WARN] 清理旧模型时出现问题: {del_err}")
            
            # Step 2: Load base model with transformers first
            print(f"加载基础模型: {model_path}")
            from transformers import AutoModelForCausalLM
            
            # 选择热重载的暂存设备（默认使用GPU，按照 LLM_HOT_RELOAD_GPUS=2,3 顺序挑选）
            staging_device = None
            try:
                staging_policy = os.environ.get('LLM_HOT_RELOAD_STAGING', 'gpu').lower()
                if staging_policy != 'gpu':
                    raise RuntimeError('staging policy not gpu')
                preferred = os.environ.get('LLM_HOT_RELOAD_GPUS')
                preferred_list = []
                if preferred:
                    try:
                        preferred_list = [int(x.strip()) for x in preferred.split(',') if x.strip() != '']
                    except Exception:
                        preferred_list = []
                # 若无环境变量，优先尝试与当前不同的GPU（2/3），且剩余显存>16GB
                if not preferred_list:
                    preferred_list = [2, 3, 1, 0]
                status = self.get_gpu_status()
                gpu_list = status.get('gpu_status', [])
                for gid in preferred_list:
                    if gpu_ids and gid == gpu_ids[0]:
                        continue
                    for g in gpu_list:
                        if int(g.get('gpu_id', -1)) == int(gid):
                            try:
                                total = float(str(g.get('memory_total', '0GB')).replace('GB',''))
                                alloc = float(str(g.get('memory_allocated', '0GB')).replace('GB',''))
                                free_gb = max(total - alloc, 0.0)
                                if free_gb >= 16.0:
                                    staging_device = f"cuda:{gid}"
                            except Exception:
                                pass
                            break
                    if staging_device is not None:
                        break
            except Exception:
                staging_device = None
            
            # 校验 staging_device 是否对当前进程可见，否则回退CPU
            try:
                if staging_device and staging_device.startswith("cuda:"):
                    import re as _re
                    m2 = _re.search(r"cuda:(\d+)", staging_device)
                    if m2:
                        _gid2 = int(m2.group(1))
                        import torch as _torch
                        if (not _torch.cuda.is_available()) or (_gid2 >= _torch.cuda.device_count()):
                            print(f"[INFO] 目标暂存GPU {_gid2} 对当前进程不可见，回退到 CPU 暂存")
                            staging_device = None
            except Exception:
                staging_device = None
            
            if staging_device is None:
                print("[INFO] 暂存设备选择为 CPU（无足够空闲GPU）")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                print(f"[INFO] 暂存设备选择: {staging_device}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=staging_device,
                    trust_remote_code=True
                )
            
            # Step 3: Apply LoRA adapter
            print(f"应用LoRA适配器: {adapter_path}")
            model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)
            
            # Step 4: Merge LoRA weights (optional but recommended)
            print(f"合并LoRA权重...")
            try:
                model_with_lora = model_with_lora.merge_and_unload()
            except Exception as merge_err:
                print(f"[WARN] 合并LoRA失败，将以PEFT形式加载: {merge_err}")
            
            # Step 5: Save merged (or PEFT) model to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"保存合并模型到临时目录...")
                model_with_lora.save_pretrained(temp_dir)
                
                # Also save tokenizer to ensure model and tokenizer are in same directory
                print(f"保存tokenizer到临时目录...")
                tokenizer_path = temp_dir  # Default to temp_dir
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    tokenizer.save_pretrained(temp_dir)
                    print(f"✓ tokenizer已保存到临时目录")
                except Exception as e:
                    print(f"[WARN] 保存tokenizer失败: {e}, 将使用原始路径")
                    tokenizer_path = model_path  # Fallback to original path
                
                # Step 6: Reinitialize vLLM with new model
                print(f"使用合并模型重初始化vLLM...")
                # 先释放合并阶段占用（GPU/CPU），确保显存充足
                try:
                    del model_with_lora
                except Exception:
                    pass
                try:
                    del base_model
                except Exception:
                    pass
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        _torch.cuda.synchronize()
                except Exception:
                    pass

                # 读取实例或环境中的显存占用比例配置，默认0.85
                try:
                    _env_util = os.environ.get('LLM_GPU_MEMORY_UTILIZATION', '').strip()
                    _env_util_val = float(_env_util) if _env_util else None
                except Exception:
                    _env_util_val = None
                _instance_util = getattr(base_llm, 'gpu_memory_utilization', None)
                _self_util = getattr(self, 'gpu_memory_utilization', None)
                _target_util = (
                    _env_util_val if isinstance(_env_util_val, float) else (
                        _instance_util if isinstance(_instance_util, (int, float)) else (
                            _self_util if isinstance(_self_util, (int, float)) else 0.85
                        )
                    )
                )

                vllm_kwargs = {
                    "model": temp_dir,
                    "tokenizer": tokenizer_path,  # Use same path as model, or fallback to original
                    "gpu_memory_utilization": float(_target_util),
                    "tensor_parallel_size": 1,
                    "max_model_len": 8192,
                    "enforce_eager": True,
                    "trust_remote_code": True,
                    "swap_space": 4,
                    "disable_log_stats": True,
                    "max_num_seqs": 128,
                    "block_size": 16
                }
                # 锁定到原始GPU，不自动切换；如需允许切换，可设置 LLM_HOT_RELOAD_ALLOW_SWITCH=1
                final_gpu_id = gpu_ids[0] if gpu_ids else (0 if llm_type == 'traffic' else 1)
                try:
                    if os.environ.get('LLM_HOT_RELOAD_ALLOW_SWITCH', '0') == '1':
                        import torch as _torch
                        visible_ids_env = os.environ.get('CUDA_VISIBLE_DEVICES')
                        if visible_ids_env and visible_ids_env.strip():
                            visible_physical = [int(x.strip()) for x in visible_ids_env.split(',') if x.strip() != '']
                        else:
                            visible_physical = list(range(_torch.cuda.device_count()))
                        status2 = self.get_gpu_status()
                        gpu_list2 = status2.get('gpu_status', [])
                        def _free_gb_for(phy_id: int):
                            for g in gpu_list2:
                                if int(g.get('gpu_id', -1)) == int(phy_id):
                                    try:
                                        total = float(str(g.get('memory_total', '0GB')).replace('GB',''))
                                        alloc = float(str(g.get('memory_allocated', '0GB')).replace('GB',''))
                                        return max(total - alloc, 0.0)
                                    except Exception:
                                        return None
                            return None
                        free_final = _free_gb_for(final_gpu_id)
                        if (free_final is None or free_final < 8.0):
                            candidates = [pid for pid in visible_physical if pid != final_gpu_id]
                            best = None
                            best_free = -1.0
                            for pid in candidates:
                                fg = _free_gb_for(pid)
                                if fg is not None and fg > best_free:
                                    best = pid
                                    best_free = fg
                            if best is not None and best_free >= 16.0:
                                print(f"[INFO] 目标GPU显存不足，切换到可见备用GPU {best} 进行vLLM重启")
                                final_gpu_id = best
                                setattr(llm_instance, 'gpu_ids', [final_gpu_id])
                except Exception:
                    pass
                
                # 在当前可见GPU集合内，严格绑定到目标GPU：
                # 临时覆写 CUDA_VISIBLE_DEVICES，确保vLLM子进程落到与微调前相同的物理GPU
                _prev_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
                try:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(final_gpu_id)
                    new_model = vllm.LLM(**vllm_kwargs)
                finally:
                    if _prev_cuda_visible is None:
                        try:
                            del os.environ['CUDA_VISIBLE_DEVICES']
                        except Exception:
                            pass
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = _prev_cuda_visible
                
                # Replace the model in the LLM instance safely
                # 将新引擎绑定到底层 LLM 对象，并让包装器读取到底层
                setattr(base_llm, 'model', new_model)
                try:
                    # 对包装器场景：如果 llm_instance 是包装器，确保它透传 original_llm
                    if getattr(llm_instance, 'original_llm', None) is base_llm:
                        pass
                except Exception:
                    pass
                print(f"✓ {llm_type}模型重初始化完成")
                return True
                
        except Exception as e:
            print(f"模型重初始化失败: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to recover original model state markers instead of raising
            try:
                if hasattr(llm_instance, 'model') and llm_instance.model is None:
                    # leave as None; wrapper将继续工作但会触发下次初始化加载
                    pass
            except Exception:
                pass
            
            return False
    
    def _recover_original_model(self, llm_instance, llm_type: str):
        """Recover original model if LoRA loading fails."""
        try:
            print(f"尝试恢复{llm_type}的原始模型...")
            
            # This would require reloading the original model
            # For now, we'll just clear the corrupted state
            if hasattr(llm_instance, 'model'):
                llm_instance.model = None
                
            # Mark adapter as failed
            if hasattr(self, 'lora_adapters'):
                self.lora_adapters[llm_type]['loaded'] = False
                self.lora_adapters[llm_type]['current'] = None
                
            print(f"⚠ {llm_type}模型状态已重置，需要重新初始化")
            
        except Exception as e:
            print(f"模型恢复失败: {e}")
    
    def queue_adapter_update(self, llm_type: str, adapter_path: str):
        """Queue an adapter update for safe application."""
        try:
            if not hasattr(self, 'adapter_update_queue'):
                self.initialize_lora_management()
            
            update_request = {
                'type': 'load_adapter',
                'llm_type': llm_type,
                'adapter_path': adapter_path,
                'timestamp': time.time()
            }
            
            self.adapter_update_queue.put(update_request)
            print(f"适配器更新请求已排队: {llm_type} -> {os.path.basename(adapter_path)}")
            return True
            
        except Exception as e:
            print(f"适配器更新排队失败: {e}")
            return False
    
    def process_adapter_updates(self) -> bool:
        """Process queued adapter updates safely."""
        try:
            if not hasattr(self, 'adapter_update_queue') or self.adapter_update_queue.empty():
                return True  # No updates to process
            
            processed_count = 0
            
            while not self.adapter_update_queue.empty():
                try:
                    update_request = self.adapter_update_queue.get(timeout=1)
                    
                    if update_request['type'] == 'load_adapter':
                        success = self.load_lora_adapter_direct(
                            update_request['llm_type'], 
                            update_request['adapter_path']
                        )
                        
                        if success:
                            processed_count += 1
                            print(f"✓ 处理适配器更新: {update_request['llm_type']}")
                        else:
                            print(f"✗ 适配器更新失败: {update_request['llm_type']}")
                    
                except Exception as e:
                    print(f"处理适配器更新时出错: {e}")
                    continue
            
            if processed_count > 0:
                print(f"成功处理 {processed_count} 个适配器更新")
            
            return True
            
        except Exception as e:
            print(f"处理适配器更新队列时出错: {e}")
            return False
    
    def get_lora_status(self) -> dict:
        """Get current LoRA adapter status."""
        try:
            if not hasattr(self, 'lora_adapters'):
                return {'initialized': False}
            
            status = {
                'initialized': True,
                'traffic': {
                    'loaded': self.lora_adapters['traffic']['loaded'],
                    'current_adapter': os.path.basename(self.lora_adapters['traffic']['current']) if self.lora_adapters['traffic']['current'] else None,
                    'version': self.adapter_versions['traffic'],
                    'pending': os.path.basename(self.lora_adapters['traffic']['pending']) if self.lora_adapters['traffic']['pending'] else None
                },
                'regional': {
                    'loaded': self.lora_adapters['regional']['loaded'],
                    'current_adapter': os.path.basename(self.lora_adapters['regional']['current']) if self.lora_adapters['regional']['current'] else None,
                    'version': self.adapter_versions['regional'],
                    'pending': os.path.basename(self.lora_adapters['regional']['pending']) if self.lora_adapters['regional']['pending'] else None
                },
                'queue_size': self.adapter_update_queue.qsize() if hasattr(self, 'adapter_update_queue') else 0
            }
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def print_lora_status(self):
        """Print current LoRA adapter status."""
        status = self.get_lora_status()
        
        print("\n=== LoRA适配器状态 ===")
        
        if not status.get('initialized'):
            print("LoRA管理系统未初始化")
            return
        
        if 'error' in status:
            print(f"获取状态时出错: {status['error']}")
            return
        
        print(f"Traffic LLM:")
        print(f"  - 已加载: {'是' if status['traffic']['loaded'] else '否'}")
        print(f"  - 当前适配器: {status['traffic']['current_adapter'] or '无'}")
        print(f"  - 版本: {status['traffic']['version']}")
        print(f"  - 待处理: {status['traffic']['pending'] or '无'}")
        
        print(f"Regional LLM:")
        print(f"  - 已加载: {'是' if status['regional']['loaded'] else '否'}")
        print(f"  - 当前适配器: {status['regional']['current_adapter'] or '无'}")
        print(f"  - 版本: {status['regional']['version']}")
        print(f"  - 待处理: {status['regional']['pending'] or '无'}")
        
        print(f"更新队列大小: {status['queue_size']}")
        print("=" * 40)
    
    # ===== TIME-SLICED TRAINING: MODEL LIFECYCLE MANAGEMENT =====
    
    def release_inference_models(self) -> bool:
        """释放推理模型以释放GPU内存用于训练"""
        try:
            print("\n=== 释放推理模型 ===")
            
            # 保存当前适配器状态
            current_adapters = {
                'traffic': self.lora_adapters['traffic']['current'] if hasattr(self, 'lora_adapters') else None,
                'regional': self.lora_adapters['regional']['current'] if hasattr(self, 'lora_adapters') else None
            }
            
            # 释放Traffic LLM
            if self.traffic_llm is not None:
                print("释放Traffic LLM...")
                if hasattr(self.traffic_llm, 'model') and self.traffic_llm.model is not None:
                    del self.traffic_llm.model
                del self.traffic_llm
                self.traffic_llm = None
            
            # 释放Regional LLM
            if self.regional_llm is not None:
                print("释放Regional LLM...")
                if hasattr(self.regional_llm, 'model') and self.regional_llm.model is not None:
                    del self.regional_llm.model
                del self.regional_llm
                self.regional_llm = None
            
            # 清理GPU内存
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 保存适配器状态供后续恢复
            if not hasattr(self, '_released_state'):
                self._released_state = {}
            self._released_state['adapters'] = current_adapters
            self._released_state['model_path'] = self.model_path
            self._released_state['task_info'] = self.task_info
            
            print("推理模型已释放，GPU内存已清理")
            return True
            
        except Exception as e:
            print(f"释放推理模型失败: {e}")
            return False
    
    def restore_inference_models(self, new_adapters: dict = None) -> bool:
        """恢复推理模型，可选择加载新的适配器"""
        try:
            print("\n=== 恢复推理模型 ===")
            
            if not hasattr(self, '_released_state'):
                print("错误: 没有找到释放状态信息")
                return False
            
            # 重新初始化LLM实例
            traffic_llm, regional_llm = self.initialize_llms()
            
            # 加载新适配器或恢复原适配器
            adapters_to_load = new_adapters or self._released_state.get('adapters', {})
            
            if adapters_to_load:
                print("加载适配器...")
                for llm_type, adapter_path in adapters_to_load.items():
                    if adapter_path and os.path.exists(adapter_path):
                        print(f"加载{llm_type}适配器: {adapter_path}")
                        self.load_lora_adapter_direct(llm_type, adapter_path)
            
            # 清理释放状态
            if hasattr(self, '_released_state'):
                delattr(self, '_released_state')
            
            print("推理模型已恢复")
            return True
            
        except Exception as e:
            print(f"恢复推理模型失败: {e}")
            return False
    
    def get_memory_status(self) -> dict:
        """获取GPU内存使用状态"""
        try:
            import torch
            if not torch.cuda.is_available():
                return {'error': 'CUDA不可用'}
            
            status = {'gpus': []}
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                gpu_status = {
                    'gpu_id': i,
                    'allocated_gb': round(memory_allocated, 2),
                    'reserved_gb': round(memory_reserved, 2),
                    'total_gb': round(memory_total, 2),
                    'utilization_percent': round(memory_allocated / memory_total * 100, 1)
                }
                status['gpus'].append(gpu_status)
            
            return status
            
        except Exception as e:
            return {'error': str(e)}
    
    def is_inference_models_loaded(self) -> bool:
        """检查推理模型是否已加载"""
        return (self.traffic_llm is not None and 
                self.regional_llm is not None and
                hasattr(self.traffic_llm, 'model') and 
                hasattr(self.regional_llm, 'model'))
