#!/usr/bin/env python3
"""
Context Length Tester Skill
===========================

测试 LLM 代理服务（OpenAI 兼容 API）的实际上下文长度。

支持两种测试模式：
1. 二分查找法：快速定位 API 能接受的最大 token 数（不验证模型是否真读到）
2. 大海捞针法：通过 needle in haystack 测试模型实际能处理的上下文长度

用法：
    # 快速探测 API 上限（二分查找）
    uv run context_length_tester.py --model deepseek-v3 --mode probe
    
    # 大海捞针测试（实测上下文能力）
    uv run context_length_tester.py --model deepseek-v3 --mode needle --start 100000 --step 20000
    
    # 指定自定义 API 地址
    uv run context_length_tester.py --model deepseek-v3 --mode needle --api-url http://localhost:6001/v1/chat/completions

环境变量：
    OPENAI_API_KEY - API 密钥（默认: sk-test）
    OPENAI_API_BASE - API 基础地址（默认: http://localhost:6001/v1）
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import requests
import tiktoken


class ContextLengthTester:
    """测试模型上下文长度的工具类。"""
    
    def __init__(self, api_url, api_key, model):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.tokenizer = self._get_tokenizer()
    
    def _get_tokenizer(self):
        """获取适合模型的 tokenizer。"""
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        """计算文本 token 数。"""
        return len(self.tokenizer.encode(text))
    
    def build_haystack(self, target_tokens):
        """构造长度约为 target_tokens 的填充文本。"""
        base_sentences = [
            "人工智能正在改变我们的生活方式。",
            "机器学习模型需要大量的训练数据。",
            "深度学习在自然语言处理领域取得了突破性进展。",
            "神经网络的结构设计对模型性能有重要影响。",
            "数据预处理是机器学习流程中的关键步骤。",
            "Transformer架构已经成为现代NLP的基础。",
            "大语言模型展现出了惊人的理解和生成能力。",
            "强化学习让智能体能够在环境中自主学习策略。",
            "计算机视觉技术使机器能够理解和分析图像内容。",
            "多模态学习结合了文本、图像和音频等多种信息源。",
            "迁移学习允许模型将在一个任务上学到的知识应用到另一个任务。",
            "注意力机制帮助模型聚焦于输入序列中的重要部分。",
            "生成对抗网络在图像生成领域取得了令人瞩目的成果。",
            "自监督学习减少了对人工标注数据的依赖。",
            "图神经网络能够处理非欧几里得结构的数据。",
        ]

        haystack_parts = []
        current_tokens = 0
        paragraph_idx = 0
        
        while current_tokens < target_tokens:
            paragraph = " ".join(random.sample(base_sentences, k=random.randint(3, 5)))
            paragraph = f"第{paragraph_idx + 1}段：{paragraph}\n"
            paragraph_tokens = len(self.tokenizer.encode(paragraph))
            
            if current_tokens + paragraph_tokens > target_tokens + 50:
                remaining = target_tokens - current_tokens
                if remaining > 10:
                    truncated = self.tokenizer.decode(self.tokenizer.encode(paragraph)[:remaining])
                    haystack_parts.append(truncated)
                    current_tokens += remaining
                break
            
            haystack_parts.append(paragraph)
            current_tokens += paragraph_tokens
            paragraph_idx += 1
        
        haystack = "".join(haystack_parts)
        actual_tokens = len(self.tokenizer.encode(haystack))
        return haystack, actual_tokens
    
    def send_request(self, messages, max_tokens=100, timeout=180):
        """发送请求到 API。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.0,
        }
        
        try:
            resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout)
            return resp
        except Exception as e:
            return type('obj', (object,), {'status_code': -1, 'text': str(e)})
    
    def probe_context_size(self, low=1000, high=200000):
        """
        二分查找 API 能接受的最大上下文长度。
        注意：这只能测出 API 的上限，不能验证模型是否真读到了内容。
        """
        print(f"开始二分查找 API 上限 - 模型: {self.model}")
        print(f"范围: {low} ~ {high} tokens\n")
        
        best = low
        filler = "The quick brown fox jumps over the lazy dog. "
        
        while low <= high:
            mid = (low + high) // 2
            tok_per_filler = self.count_tokens(filler)
            repeat_times = mid // tok_per_filler + 1
            prompt = (filler * repeat_times)[:mid*4]
            
            tokens = self.count_tokens(prompt)
            if tokens > mid:
                prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:mid])
            
            messages = [{"role": "user", "content": prompt + "\nSummarize the above in one word."}]
            
            resp = self.send_request(messages, max_tokens=10, timeout=60)
            
            if resp.status_code == 200:
                best = mid
                low = mid + 1
                print(f"✓ {mid} tokens OK")
            elif "context" in resp.text.lower() or resp.status_code in [400, 413, 414]:
                print(f"✗ {mid} tokens EXCEEDED: {resp.status_code}")
                high = mid - 1
            else:
                print(f"? {mid} tokens UNKNOWN: {resp.text[:100]}")
                break
            
            time.sleep(0.5)
        
        print(f"\nAPI 能接受的最大上下文长度约为: {best} tokens")
        return best
    
    def needle_test(self, start_tokens=100000, max_tokens=200000, 
                    step_tokens=20000, depth_ratios=(0.0, 0.5, 1.0)):
        """
        大海捞针测试：验证模型实际能处理的上下文长度。
        
        在文本不同深度插入关键信息，测试模型能否准确检索。
        """
        print(f"=" * 80)
        print(f"大海捞针测试 - 模型: {self.model}")
        print(f"范围: {start_tokens} ~ {max_tokens} tokens, 步长: {step_tokens}")
        print(f"测试深度: {depth_ratios}")
        print(f"=" * 80)
        
        results = []
        current_tokens = start_tokens
        
        while current_tokens <= max_tokens:
            # 每次使用随机密钥，避免缓存
            secret = f"KEY-{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
            needle = f"【重要信息】本次测试的密钥是：{secret}。请记住这个密钥，后续会询问。"
            question = "文本中提到的密钥是什么？"
            
            print(f"\n{'='*60}")
            print(f"当前测试: {current_tokens} tokens | 密钥: {secret}")
            print(f"{'='*60}")
            
            haystack, actual_tokens = self.build_haystack(current_tokens)
            print(f"实际长度: {actual_tokens} tokens")
            
            depth_results = {}
            all_passed = True
            
            for depth in depth_ratios:
                print(f"  深度 {depth*100:>5.1f}%... ", end="", flush=True)
                
                # 插入 needle
                tokens = self.tokenizer.encode(haystack)
                insert_pos = int(len(tokens) * depth)
                needle_tokens = self.tokenizer.encode(needle)
                new_tokens = tokens[:insert_pos] + needle_tokens + tokens[insert_pos:]
                text_with_needle = self.tokenizer.decode(new_tokens)
                
                messages = [{
                    "role": "user",
                    "content": f"请仔细阅读以下文本，然后回答问题。\n\n{text_with_needle}\n\n问题：{question}\n请直接给出答案，不要解释。"
                }]
                
                start_time = time.time()
                resp = self.send_request(messages, max_tokens=100)
                elapsed = time.time() - start_time
                
                if resp.status_code != 200:
                    print(f"✗ FAIL ({elapsed:.1f}s) - HTTP {resp.status_code}")
                    depth_results[depth] = {"status": "FAIL", "response": f"HTTP {resp.status_code}", "time": elapsed}
                    all_passed = False
                    continue
                
                data = resp.json()
                if "choices" not in data or not data["choices"]:
                    print(f"✗ FAIL ({elapsed:.1f}s) - Invalid response")
                    depth_results[depth] = {"status": "FAIL", "response": "Invalid", "time": elapsed}
                    all_passed = False
                    continue
                
                answer = data["choices"][0]["message"]["content"].strip()
                success = secret.lower() in answer.lower()
                
                if success:
                    print(f"✓ PASS ({elapsed:.1f}s) - {answer}")
                    depth_results[depth] = {"status": "PASS", "response": answer, "time": elapsed}
                else:
                    print(f"✗ FAIL ({elapsed:.1f}s) - {answer}")
                    depth_results[depth] = {"status": "FAIL", "response": answer, "time": elapsed}
                    all_passed = False
                
                time.sleep(1)
            
            results.append({
                "tokens": actual_tokens,
                "all_passed": all_passed,
                "depths": depth_results,
            })
            
            if not all_passed:
                print(f"\n{'='*60}")
                print(f"模型在 {actual_tokens} tokens 时开始丢失信息")
                print(f"{'='*60}")
                
                # 精细测试
                if step_tokens > 5000:
                    print("\n进行精细测试...")
                    prev_tokens = current_tokens - step_tokens
                    fine_step = max(step_tokens // 5, 5000)
                    fine_tokens = prev_tokens + fine_step
                    
                    while fine_tokens < current_tokens:
                        secret_fine = f"KEY-{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
                        needle_fine = f"【重要信息】本次测试的密钥是：{secret_fine}。请记住这个密钥，后续会询问。"
                        
                        print(f"\n精细测试: {fine_tokens} tokens | 密钥: {secret_fine}")
                        haystack_fine, actual_fine = self.build_haystack(fine_tokens)
                        
                        fine_passed = True
                        for depth in depth_ratios:
                            print(f"  深度 {depth*100:>5.1f}%... ", end="", flush=True)
                            
                            tokens = self.tokenizer.encode(haystack_fine)
                            insert_pos = int(len(tokens) * depth)
                            needle_tokens = self.tokenizer.encode(needle_fine)
                            new_tokens = tokens[:insert_pos] + needle_tokens + tokens[insert_pos:]
                            text_with_needle = self.tokenizer.decode(new_tokens)
                            
                            messages = [{
                                "role": "user",
                                "content": f"请仔细阅读以下文本，然后回答问题。\n\n{text_with_needle}\n\n问题：{question}\n请直接给出答案，不要解释。"
                            }]
                            
                            resp = self.send_request(messages, max_tokens=100)
                            if resp.status_code == 200:
                                data = resp.json()
                                if "choices" in data and data["choices"]:
                                    answer = data["choices"][0]["message"]["content"].strip()
                                    if secret_fine.lower() in answer.lower():
                                        print(f"✓ PASS - {answer}")
                                    else:
                                        print(f"✗ FAIL - {answer}")
                                        fine_passed = False
                                else:
                                    print(f"✗ FAIL - Invalid response")
                                    fine_passed = False
                            else:
                                print(f"✗ FAIL - HTTP {resp.status_code}")
                                fine_passed = False
                            
                            time.sleep(1)
                        
                        if not fine_passed:
                            print(f"\n精确上限约为: {fine_tokens - fine_step} ~ {fine_tokens} tokens")
                            break
                        
                        fine_tokens += fine_step
                
                break
            
            current_tokens += step_tokens
        
        # 总结
        print(f"\n{'='*80}")
        print("测试总结")
        print(f"{'='*80}")
        
        max_passed = 0
        for r in results:
            status = "✓ 全部通过" if r["all_passed"] else "✗ 部分失败"
            print(f"{r['tokens']:>8} tokens - {status}")
            if r["all_passed"]:
                max_passed = max(max_passed, r["tokens"])
        
        print(f"\n{'='*80}")
        print(f"模型 {self.model} 的实测上下文长度约为: {max_passed} tokens")
        print(f"{'='*80}")
        
        return max_passed


def main():
    parser = argparse.ArgumentParser(description="测试 LLM 模型上下文长度")
    parser.add_argument("--api-url", type=str, 
                        default=os.getenv("OPENAI_API_BASE", "http://localhost:6001/v1/chat/completions"),
                        help="API 端点地址")
    parser.add_argument("--api-key", type=str,
                        default=os.getenv("OPENAI_API_KEY", "sk-test"),
                        help="API 密钥")
    parser.add_argument("--model", type=str, required=True, help="模型名称")
    parser.add_argument("--mode", type=str, choices=["probe", "needle"], default="needle",
                        help="测试模式: probe=二分查找API上限, needle=大海捞针实测")
    parser.add_argument("--start", type=int, default=100000, help="起始 token 数 (needle模式)")
    parser.add_argument("--max", type=int, default=200000, help="最大 token 数")
    parser.add_argument("--step", type=int, default=20000, help="步长 (needle模式)")
    parser.add_argument("--depths", type=str, default="0.0,0.5,1.0",
                        help="测试深度，逗号分隔 (needle模式)")
    
    args = parser.parse_args()
    
    tester = ContextLengthTester(args.api_url, args.api_key, args.model)
    
    if args.mode == "probe":
        tester.probe_context_size(low=1000, high=args.max)
    else:
        depth_ratios = tuple(float(d) for d in args.depths.split(","))
        tester.needle_test(args.start, args.max, args.step, depth_ratios)


if __name__ == "__main__":
    main()
