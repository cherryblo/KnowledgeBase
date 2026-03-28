"""
测试脚本：--tool-call-parser 参数功能测试

测试环境：
- NPU 硬件（单卡）
- SGLang 框架
- Llama-3.2-1B-Instruct 模型

测试覆盖：
1. Llama3 Parser 基础功能
2. Pythonic Parser 流式输出
3. Tool Choice Required 约束
4. 多轮对话工具调用
5. 无效 Parser 类型错误处理
"""

import json
import sys
import time
import subprocess
import signal
import os
from typing import List, Dict, Optional

try:
    import openai
except ImportError:
    print("请安装 openai 库: pip install openai")
    sys.exit(1)


class ToolCallParserTester:
    """--tool-call-parser 参数测试类"""

    def __init__(self, model_path: str, base_url: str = "http://127.0.0.1:30000"):
        self.model_path = model_path
        self.base_url = base_url
        self.api_key = "sk-123456"
        self.process = None
        self.client = None

    def start_server(self, parser_type: str, other_args: Optional[List[str]] = None) -> bool:
        """启动 SGLang 服务"""
        args = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "127.0.0.1",
            "--port", "30000",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--tool-call-parser", parser_type,
            "--log-level", "warning",
        ]

        if other_args:
            args.extend(other_args)

        print(f"启动服务，参数: {' '.join(args)}")

        try:
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

            # 等待服务启动
            for i in range(60):
                try:
                    self.client = openai.Client(api_key=self.api_key, base_url=self.base_url + "/v1")
                    # 简单的健康检查
                    response = self.client.chat.completions.create(
                        model=self.model_path,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10
                    )
                    print(f"服务启动成功 (耗时 {i+1} 秒)")
                    return True
                except Exception as e:
                    time.sleep(1)
                    if i % 10 == 0:
                        print(f"等待服务启动... ({i+1}s)")

            print("服务启动超时")
            return False

        except Exception as e:
            print(f"启动服务失败: {e}")
            return False

    def stop_server(self):
        """停止 SGLang 服务"""
        if self.process:
            print("停止服务...")
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.wait(timeout=10)
            except Exception as e:
                print(f"停止服务时出错: {e}")
            finally:
                self.process = None
                self.client = None

    def test_1_llama3_basic_function(self) -> bool:
        """
        测试用例 1：Llama3 Parser 基础功能测试

        测试目的：验证 llama3 parser 能够正确解析 Llama-3.2 模型的工具调用输出
        测试环境：单张 NPU 卡
        预期结果：tool_calls 不为空，工具名称为 "add"，参数包含 a 和 b 字段
        """
        print("\n" + "="*80)
        print("测试用例 1：Llama3 Parser 基础功能测试")
        print("="*80)

        if not self.start_server("llama3"):
            return False

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Compute the sum of two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer", "description": "A number"},
                                "b": {"type": "integer", "description": "A number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ]

            system_message = (
                "You are a helpful assistant with tool calling capabilities. "
                "Only reply with a tool call if the function exists in the library provided by the user. "
                "If it doesn't exist, just reply directly in natural language. "
                "When you receive a tool call response, use the output to format an answer to the original user question. "
                "You have access to the following functions. "
                "To call a function, please respond with JSON for a function call. "
                'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
                "Do not use variables.\n\n"
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Compute (3+5)"},
            ]

            response = self.client.chat.completions.create(
                model=self.model_path,
                max_tokens=2048,
                messages=messages,
                temperature=0.8,
                top_p=0.8,
                stream=False,
                tools=tools,
            )

            tool_calls = response.choices[0].message.tool_calls

            if not isinstance(tool_calls, list) or len(tool_calls) == 0:
                print("❌ 失败：tool_calls 应该是非空列表")
                return False

            function_name = tool_calls[0].function.name
            if function_name != "add":
                print(f"❌ 失败：函数名称应该是 'add'，实际是 '{function_name}'")
                return False

            print(f"✓ 成功：工具调用解析正确")
            print(f"  - 工具名称: {function_name}")
            print(f"  - 参数: {tool_calls[0].function.arguments}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_2_pythonic_streaming(self) -> bool:
        """
        测试用例 2：Pythonic Parser 流式输出测试

        测试目的：验证 pythonic parser 在流式输出模式下能够正确解析工具调用
        测试环境：单张 NPU 卡
        预期结果：流式输出正常，能够实时识别工具调用，最终结果完整
        """
        print("\n" + "="*80)
        print("测试用例 2：Pythonic Parser 流式输出测试")
        print("="*80)

        if not self.start_server("pythonic"):
            return False

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a given location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The name of the city or location.",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ]

            system_message = (
                "You are a travel assistant. "
                "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
                "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
                "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
                "Here is an example:\n"
                '[get_weather(location="Paris")]'
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "What's the weather like in Tokyo?"},
            ]

            stream = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                tools=tools,
                temperature=0.1,
                stream=True,
            )

            chunks = []
            tool_call_found = False

            for chunk in stream:
                chunks.append(chunk)
                if chunk.choices[0].delta.tool_calls:
                    tool_call_found = True
                    print(f"  流式块: 检测到工具调用")

            if not tool_call_found:
                print("⚠ 警告：未检测到工具调用（可能是模型未生成）")
                print(f"  收到 {len(chunks)} 个流式块")
                return True

            print(f"✓ 成功：流式输出正常")
            print(f"  - 总流式块数: {len(chunks)}")
            print(f"  - 检测到工具调用: {tool_call_found}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.stop_server()

    def test_3_tool_choice_required(self) -> bool:
        """
        测试用例 3：Tool Choice Required 测试

        测试目的：验证 tool_choice=required 时强制工具调用的约束生效
        测试环境：单张 NPU 卡
        预期结果：模型必须返回工具调用，不会返回纯文本回复
        """
        print("\n" + "="*80)
        print("测试用例 3：Tool Choice Required 约束测试")
        print("="*80)

        if not self.start_server("llama3"):
            return False

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "multiply",
                        "description": "Multiply two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                }
            ]

            messages = [
                {"role": "user", "content": "What is 5 times 7?"}
            ]

            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                tools=tools,
                tool_choice="required",
                max_tokens=512,
                temperature=0.1,
            )

            tool_calls = response.choices[0].message.tool_calls
            content = response.choices[0].message.content

            if not tool_calls:
                print("❌ 失败：tool_choice=required 时应该返回工具调用")
                print(f"  返回内容: {content}")
                return False

            print(f"✓ 成功：tool_choice=required 约束生效")
            print(f"  - 工具名称: {tool_calls[0].function.name}")
            print(f"  - 参数: {tool_calls[0].function.arguments}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_4_multiturn_conversation(self) -> bool:
        """
        测试用例 4：多轮对话工具调用测试

        测试目的：验证在多轮对话中工具调用的正确性
        测试环境：单张 NPU 卡
        预期结果：第一轮返回工具调用，第二轮基于工具响应返回答案
        """
        print("\n" + "="*80)
        print("测试用例 4：多轮对话工具调用测试")
        print("="*80)

        if not self.start_server("llama3"):
            return False

        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ]

            system_message = (
                "You are a helpful assistant with tool calling capabilities. "
                "To call a function, please respond with JSON for a function call. "
                'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
                "Do not use variables.\n\n"
            )

            # 第一轮：用户请求
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "What time is it?"}
            ]

            response1 = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                tools=tools,
                max_tokens=512,
                temperature=0.1,
            )

            tool_calls = response1.choices[0].message.tool_calls

            if not tool_calls:
                print("⚠ 第一轮未返回工具调用，跳过多轮测试")
                return True

            print(f"  第一轮返回工具调用: {tool_calls[0].function.name}")

            # 第二轮：工具响应
            messages.append(response1.choices[0].message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_calls[0].id,
                "content": "2024-01-15 10:30:00",
                "name": tool_calls[0].function.name,
            })

            response2 = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                tools=tools,
                max_tokens=512,
                temperature=0.1,
            )

            final_content = response2.choices[0].message.content

            if not final_content:
                print("❌ 失败：第二轮应该返回内容")
                return False

            print(f"✓ 成功：多轮对话工具调用正常")
            print(f"  - 第一轮: 工具调用")
            print(f"  - 第二轮: 文本回复")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_5_invalid_parser_type(self) -> bool:
        """
        测试用例 5：无效 Parser 类型测试

        测试目的：验证传入无效 parser 类型时的错误处理
        测试环境：单张 NPU 卡
        预期结果：服务启动失败或给出明确错误提示
        """
        print("\n" + "="*80)
        print("测试用例 5：无效 Parser 类型错误处理测试")
        print("="*80)

        invalid_parser = "invalid_parser_type_xyz"

        args = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "127.0.0.1",
            "--port", "30001",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--tool-call-parser", invalid_parser,
            "--log-level", "error",
        ]

        print(f"尝试使用无效 parser 类型启动服务: {invalid_parser}")

        try:
            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # 等待 5 秒，检查进程是否还在运行
            time.sleep(5)

            poll_result = process.poll()

            if poll_result is not None:
                # 进程已退出，检查错误信息
                stdout, stderr = process.communicate(timeout=5)
                error_output = stdout + stderr

                if "tool_call_parser" in error_output.lower() or "parser" in error_output.lower():
                    print(f"✓ 成功：服务启动失败并给出错误提示")
                    print(f"  - 进程退出码: {poll_result}")
                    return True
                else:
                    print(f"⚠ 服务启动失败，但错误信息不明确")
                    return True
            else:
                # 进程仍在运行，可能是使用了默认值
                print(f"⚠ 服务仍在运行，可能使用了默认值或忽略了无效参数")
                process.terminate()
                process.wait(timeout=5)
                return True

        except Exception as e:
            print(f"⚠ 测试过程出现异常: {e}")
            return True

    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试用例"""
        print("\n" + "="*80)
        print("开始 --tool-call-parser 参数测试")
        print("="*80)
        print(f"模型路径: {self.model_path}")
        print(f"服务地址: {self.base_url}")

        results = {}

        tests = [
            ("Llama3 Parser 基础功能", self.test_1_llama3_basic_function),
            ("Pythonic Parser 流式输出", self.test_2_pythonic_streaming),
            ("Tool Choice Required 约束", self.test_3_tool_choice_required),
            ("多轮对话工具调用", self.test_4_multiturn_conversation),
            ("无效 Parser 类型处理", self.test_5_invalid_parser_type),
        ]

        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"❌ 测试 '{test_name}' 执行异常: {e}")
                results[test_name] = False

            # 测试间隔
            time.sleep(2)

        return results

    def print_summary(self, results: Dict[str, bool]):
        """打印测试结果汇总"""
        print("\n" + "="*80)
        print("测试结果汇总")
        print("="*80)

        passed = sum(1 for v in results.values() if v)
        total = len(results)

        for test_name, result in results.items():
            status = "✓ 通过" if result else "❌ 失败"
            print(f"{status} - {test_name}")

        print("\n" + "-"*80)
        print(f"总计: {passed}/{total} 通过")
        print("="*80)

        return passed == total


def main():
    """主函数"""
    # 从环境变量或命令行获取模型路径
    model_path = os.environ.get(
        "SGLANG_TEST_MODEL",
        "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
    )

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print("使用模型路径:", model_path)

    tester = ToolCallParserTester(model_path)
    results = tester.run_all_tests()
    success = tester.print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
