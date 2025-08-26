import copy
import os
import threading
import time

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

class LLM(object):
    def __init__(self, llm_path, batch_size=16, top_k=50, top_p=1.0, temperature=0.1, max_tokens=8192, memory_size=3, task_info=None, use_reflection=True, gpu_ids=None, tensor_parallel_size=1, gpu_memory_utilization=0.7):
        self.use_reflection = use_reflection
        self.gpu_ids = gpu_ids  # 指定使用的GPU ID列表
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
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
            effective_max_len = max(min(max_tokens, 12288), 10240)  # 在10240-12288之间
            
            vllm_kwargs = {
                "model": llm_path,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": effective_max_len,  # 限制序列长度以节省KV cache
                "enforce_eager": True,
                "trust_remote_code": True,
                "swap_space": 4,  # 4GB swap space
                "disable_log_stats": True  # 减少日志开销
            }
            
            print(f"调整序列长度: {max_tokens} -> {effective_max_len} (节省KV cache内存)")
            
            # 使用device参数而不是环境变量来指定GPU
            if self.gpu_ids is not None:
                if isinstance(self.gpu_ids, (list, tuple)):
                    if len(self.gpu_ids) == 1:
                        # 单GPU模式，直接指定设备
                        vllm_kwargs["device"] = f"cuda:{self.gpu_ids[0]}"
                        vllm_kwargs["tensor_parallel_size"] = 1
                        print(f"LLM initialization: Using GPU {self.gpu_ids[0]}")
                    else:
                        # 多GPU模式，使用tensor parallel
                        gpu_ids_str = ','.join(map(str, self.gpu_ids))
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
                        vllm_kwargs["tensor_parallel_size"] = len(self.gpu_ids)
                        print(f"LLM initialization: Using GPUs {gpu_ids_str} with tensor parallel")
                else:
                    # 单个GPU ID
                    vllm_kwargs["device"] = f"cuda:{self.gpu_ids}"
                    vllm_kwargs["tensor_parallel_size"] = 1
                    print(f"LLM initialization: Using GPU {self.gpu_ids}")
            else:
                # 使用默认GPU
                vllm_kwargs["tensor_parallel_size"] = 1
                print("LLM initialization: Using default GPU")
            
            print(f"vLLM配置: {vllm_kwargs}")
            print("正在初始化vLLM引擎...这可能需要几分钟")
            
            try:
                llm_model = vllm.LLM(**vllm_kwargs)
                print(f"vLLM模型加载成功！使用GPU: {self.gpu_ids if self.gpu_ids else 'auto'}")
                use_api = False
            except Exception as e:
                print(f"vLLM模型加载失败: {e}")
                raise
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
        else:
            # 其他情况（模型名称等），尝试作为远程模型使用vLLM加载
            print(f"尝试使用vLLM加载模型: {llm_path}")
            # 限制max_model_len以节省KV cache内存，同时保持在10000以上
            effective_max_len = max(min(max_tokens, 12288), 10240)  # 在10240-12288之间
            
            vllm_kwargs = {
                "model": llm_path,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size,
                "max_model_len": effective_max_len,  # 限制序列长度以节省KV cache
                "enforce_eager": True,
                "trust_remote_code": True,
                "swap_space": 4,  # 4GB swap space
                "disable_log_stats": True  # 减少日志开销
            }
            
            print(f"调整序列长度: {max_tokens} -> {effective_max_len} (节省KV cache内存)")
            
            # 使用device参数而不是环境变量来指定GPU
            if self.gpu_ids is not None:
                if isinstance(self.gpu_ids, (list, tuple)):
                    if len(self.gpu_ids) == 1:
                        # 单GPU模式，直接指定设备
                        vllm_kwargs["device"] = f"cuda:{self.gpu_ids[0]}"
                        vllm_kwargs["tensor_parallel_size"] = 1
                        print(f"LLM initialization: Using GPU {self.gpu_ids[0]}")
                    else:
                        # 多GPU模式，使用tensor parallel
                        gpu_ids_str = ','.join(map(str, self.gpu_ids))
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
                        vllm_kwargs["tensor_parallel_size"] = len(self.gpu_ids)
                        print(f"LLM initialization: Using GPUs {gpu_ids_str} with tensor parallel")
                else:
                    # 单个GPU ID
                    vllm_kwargs["device"] = f"cuda:{self.gpu_ids}"
                    vllm_kwargs["tensor_parallel_size"] = 1
                    print(f"LLM initialization: Using GPU {self.gpu_ids}")
            else:
                # 使用默认GPU
                vllm_kwargs["tensor_parallel_size"] = 1
                print("LLM initialization: Using default GPU")
            
            print(f"vLLM配置: {vllm_kwargs}")
            print("正在初始化vLLM引擎...这可能需要几分钟")
            
            try:
                llm_model = vllm.LLM(**vllm_kwargs)
                print(f"vLLM模型加载成功！使用GPU: {self.gpu_ids if self.gpu_ids else 'auto'}")
                use_api = False
            except Exception as e:
                print(f"vLLM模型加载失败: {e}")
                raise
            generation_kwargs = vllm.SamplingParams(**generation_kwargs)

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

    def update_memory(self, sample_info):
        if not self.use_reflection:
            return

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
                if len(possible_answer) != 0:
                    self.memory = json.loads(possible_answer[-1])[:self.memory_size]
                else:
                    self.memory = json.loads(response)[:self.memory_size]

                return
            except Exception as e:
                print(f"Error in update memory: {e}\nTry again...")
                # print(f"=================================\n{response}")
                retry_count += 1

    def inference(self, query, system_prompt=None):
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
            if "deepseek" in self.llm_name.lower():
                llm_name = "deepseek-ai/" + self.llm_name
            else:
                llm_name = self.llm_name

            retry_count = 0
            response = None
            while retry_count < 3:
                try:
                    response = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        temperature=self.generation_kwargs['temperature'],
                        max_tokens=self.generation_kwargs['max_tokens'],
                        # timeout=120
                    ).choices[0].message.content
                    break
                except:
                    time.sleep(5)
                    retry_count += 1
        else:
            responses_gen = self.model.chat([message], use_tqdm=False, sampling_params=self.generation_kwargs)
            response = responses_gen[0].outputs[0].text

        return response

    def batch_inference(self, queries, system_prompt=None):
        all_responses = []
        messages = list()

        for i, q in enumerate(queries):
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

            if len(messages) == self.batch_size or i == len(queries) - 1:
                if self.use_api:
                    if self.provider_name == 'siliconflow':
                        llm_name = f"{self.institute_name}/{self.llm_name}"
                    else:
                        llm_name = self.llm_name
                    threads = []
                    responses = [None for _ in range(len(messages))]

                    for j, message in enumerate(messages):
                        thread = threading.Thread(target=self.threading_inference, args=(llm_name, message, responses, j, ))
                        threads.append(thread)
                        thread.start()

                    for thread in threads:
                        thread.join()

                    all_responses.extend(responses)
                else:
                    responses_gen = self.model.chat(messages, use_tqdm=False, sampling_params=self.generation_kwargs)
                    responses = [res.outputs[0].text for res in responses_gen]
                    all_responses.extend(responses)
                    messages = list()

        return all_responses

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

    def threading_inference(self, llm_name, message, response_list, m_id):
        retry_count = 0
        response_list[m_id] = ""
        while retry_count < 2:
            time.sleep(5)
            try:
                if "openai" == self.provider_name:
                    stream = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        # temperature=self.generation_kwargs['temperature'],
                        max_completion_tokens=self.generation_kwargs['max_tokens'],
                        stream=True
                    )
                else:
                    stream = self.model.chat.completions.create(
                        model=llm_name,
                        messages=message,
                        max_tokens=self.generation_kwargs['max_tokens'] if "glm-z1-9b" not in llm_name.lower() else 8000,
                        stream=True
                    )

                collected_response = "<think>\n"
                reasoning_finish_flag = False
                for chunk in stream:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        if not reasoning_finish_flag:
                            collected_response += "</think>\n"
                            reasoning_finish_flag = True
                        token = chunk.choices[0].delta.content
                        collected_response += token
                    elif hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        token = chunk.choices[0].delta.reasoning_content
                        collected_response += token
                response_list[m_id] = collected_response
                print(f"\nSuccess [{m_id}].")
                break

            except Exception as e:
                retry_count += 1
                print(e)

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

            # add memory
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
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

                if "answer" not in decision or "summary" not in decision:
                    retry_queries.append(queries[i])
                    decision = {}
                    failed_responses.append(res)

                if decision:
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

            # add memory
            memory_text = ""
            for exp in self.memory:
                memory_text += f"- {exp}\n"
            memory_text = memory_text[:-1] if memory_text else "N/A"
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
                    if len(possible_answer) <= 0:
                        reflection = json.loads(res)
                    else:
                        reflection = json.loads(possible_answer[-1])
                except Exception as e:
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
            
            # Inter-agent communication template
            self.inter_agent_communication_template = load_json("./prompts/inter_agent_communication_template.json")["template"]
            
            # Enhanced hybrid decision template
            self.hybrid_decision_template = load_json("./prompts/hybrid_decision_making_template.json")["template"]
            
        except Exception as e:
            print(f"Warning: Could not load multi-agent templates: {e}")
            # Use fallback templates
            self.regional_coordination_template = self.decision_making_template
            self.macro_planning_template = self.decision_making_template
            self.inter_agent_communication_template = self.decision_making_template
            self.hybrid_decision_template = self.decision_making_template

    def regional_coordination_decision(self, regional_context, vehicles_data, boundary_status, 
                                     coordination_messages, traffic_predictions, route_options, region_id):
        """Make coordinated decisions for vehicles within a region."""
        try:
            # Prepare the regional coordination query
            query = copy.copy(self.overall_template)
            
            # Replace template variables with actual data
            regional_template = copy.copy(self.regional_coordination_template)
            regional_template = regional_template.replace("<region_id>", str(region_id))
            regional_template = regional_template.replace("<regional_context>", str(regional_context))
            regional_template = regional_template.replace("<vehicles_data>", str(vehicles_data))
            regional_template = regional_template.replace("<boundary_status>", str(boundary_status))
            regional_template = regional_template.replace("<coordination_messages>", str(coordination_messages))
            regional_template = regional_template.replace("<traffic_predictions>", str(traffic_predictions))
            regional_template = regional_template.replace("<route_options>", str(route_options))
            
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
            except:
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
            
            # Replace template variables with actual data
            macro_template = copy.copy(self.macro_planning_template)
            macro_template = macro_template.replace("<global_state>", str(global_state))
            macro_template = macro_template.replace("<route_requests>", str(route_requests))
            macro_template = macro_template.replace("<regional_conditions>", str(regional_conditions))
            macro_template = macro_template.replace("<boundary_analysis>", str(boundary_analysis))
            macro_template = macro_template.replace("<flow_predictions>", str(flow_predictions))
            macro_template = macro_template.replace("<coordination_needs>", str(coordination_needs))
            macro_template = macro_template.replace("<region_routes>", str(region_routes))
            
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
            except:
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
            except:
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
                    except:
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
        
        print(f"\n=== 初始化本地LLM管理器 ===")
        print(f"模型路径: {model_path}")
        
    def initialize_llms(self):
        """初始化两个共享LLM实例"""
        print("\n=== 初始化共享LLM实例 ===")
        
        # 初始化Traffic LLM (使用GPU 0)
        print("正在初始化 Traffic LLM (GPU 0)...")
        self.traffic_llm = LLM(
            llm_path=self.model_path,
            batch_size=16,
            top_k=50,
            top_p=1.0,
            temperature=0.1,
            max_tokens=10240,  # 保持在10k以上，但不会太高
            memory_size=3,
            task_info=self.task_info,
            use_reflection=True,
            gpu_ids=[0],  # 使用GPU 0
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85  # 稍微增加内存利用率
        )
        print("[SUCCESS] Traffic LLM 初始化完成")
        
        # 初始化Regional LLM (使用GPU 1)
        print("正在初始化 Regional LLM (GPU 1)...")
        self.regional_llm = LLM(
            llm_path=self.model_path,
            batch_size=16,
            top_k=50,
            top_p=1.0,
            temperature=0.1,
            max_tokens=10240,  # 保持在10k以上，但不会太高
            memory_size=3,
            task_info=self.task_info,
            use_reflection=True,
            gpu_ids=[1],  # 使用GPU 1
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85  # 稍微增加内存利用率
        )
        print("[SUCCESS] Regional LLM 初始化完成")
        
        print("\n=== 所有LLM实例初始化完成 ===")
        print("- Traffic LLM: GPU 0")
        print("- Regional LLM: GPU 1")
        print("- GPU 2-3: 用于推理加速")
        
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
