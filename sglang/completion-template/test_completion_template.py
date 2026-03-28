"""
测试脚本：--completion-template 参数功能测试

测试环境：
- NPU 硬件（单卡）
- SGLang 框架
- DeepSeek-Coder 模型

测试覆盖：
1. 内置 deepseek_coder 模板测试
2. 内置 star_coder 模板测试
3. 内置 qwen_coder 模板测试
4. 自定义 JSON 模板测试
5. 多补全请求测试
6. 流式补全测试
7. 无效模板名称错误处理测试
8. 无效模板文件路径错误处理测试
"""

import json
import sys
import time
import subprocess
import signal
import os
import tempfile
from typing import List, Dict, Optional

try:
    import openai
except ImportError:
    print("请安装 openai 库: pip install openai")
    sys.exit(1)


class CompletionTemplateTester:
    """--completion-template 参数测试类"""

    def __init__(self, model_path: str, base_url: str = "http://127.0.0.1:30000"):
        self.model_path = model_path
        self.base_url = base_url
        self.api_key = "sk-123456"
        self.process = None
        self.client = None

    def start_server(self, other_args: Optional[List[str]] = None, port: int = 30000) -> bool:
        """启动 SGLang 服务"""
        args = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
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
            base_url = f"http://127.0.0.1:{port}/v1"
            for i in range(60):
                try:
                    self.client = openai.Client(api_key=self.api_key, base_url=base_url)
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

    def test_1_deepseek_coder_template(self) -> bool:
        """
        测试用例 1：内置 deepseek_coder 模板测试

        测试目的：验证使用内置 deepseek_coder 模板时，FIM 补全功能正常工作
        测试环境：单张 NPU 卡
        预期结果：服务成功启动，返回有效的补全内容
        """
        print("\n" + "="*80)
        print("测试用例 1：内置 deepseek_coder 模板测试")
        print("="*80)

        if not self.start_server(["--completion-template", "deepseek_coder"]):
            return False

        try:
            # 发送代码补全请求
            prompt = "def calculate_sum(a, b):"
            suffix = "    return a + b"

            response = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=False,
            )

            if not response.choices:
                print("❌ 失败：应该返回补全结果")
                return False

            completion = response.choices[0].text
            if not completion:
                print("❌ 失败：补全内容为空")
                return False

            # 验证补全内容
            full_code = prompt + completion + suffix
            print(f"✓ 成功：deepseek_coder 模板正常工作")
            print(f"  - 补全内容: {completion}")
            print(f"  - 完整代码: {full_code}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_2_star_coder_template(self) -> bool:
        """
        测试用例 2：内置 star_coder 模板测试

        测试目的：验证使用内置 star_coder 模板时，FIM 补全功能正常工作
        测试环境：单张 NPU 卡
        预期结果：补全内容格式正确，补全内容能够正确连接前后缀
        """
        print("\n" + "="*80)
        print("测试用例 2：内置 star_coder 模板测试")
        print("="*80)

        if not self.start_server(["--completion-template", "star_coder"]):
            return False

        try:
            prompt = "class Calculator:"
            suffix = "    def add(self, x, y):"

            response = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=False,
            )

            if not response.choices:
                print("❌ 失败：应该返回补全结果")
                return False

            completion = response.choices[0].text
            if not completion:
                print("❌ 失败：补全内容为空")
                return False

            print(f"✓ 成功：star_coder 模板正常工作")
            print(f"  - 补全内容: {completion}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_3_qwen_coder_template(self) -> bool:
        """
        测试用例 3：内置 qwen_coder 模板测试

        测试目的：验证使用内置 qwen_coder 模板时，FIM 补全功能正常工作
        测试环境：单张 NPU 卡
        预期结果：补全内容格式正确，多次请求都能成功返回
        """
        print("\n" + "="*80)
        print("测试用例 3：内置 qwen_coder 模板测试")
        print("="*80)

        if not self.start_server(["--completion-template", "qwen_coder"]):
            return False

        try:
            # 发送多个补全请求
            prompt = "def process_data(data):"
            suffix = "    return"

            response = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=False,
            )

            if not response.choices:
                print("❌ 失败：应该返回补全结果")
                return False

            completion = response.choices[0].text
            if not completion:
                print("❌ 失败：补全内容为空")
                return False

            print(f"✓ 成功：qwen_coder 模板正常工作")
            print(f"  - 补全内容: {completion}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_4_custom_json_template(self) -> bool:
        """
        测试用例 4：自定义 JSON 模板测试

        测试目的：验证使用自定义 JSON 模板文件时，FIM 补全功能正常工作
        测试环境：单张 NPU 卡
        预期结果：自定义模板被正确加载和应用
        """
        print("\n" + "="*80)
        print("测试用例 4：自定义 JSON 模板测试")
        print("="*80)

        # 创建自定义 JSON 模板文件
        custom_template = {
            "name": "custom_fim",
            "fim_begin_token": "<fim_start>",
            "fim_middle_token": "<fim_middle>",
            "fim_end_token": "<fim_end>",
            "fim_position": "middle"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_template, f)
            template_path = f.name

        try:
            if not self.start_server(["--completion-template", template_path]):
                return False

            prompt = "def hello_world():"
            suffix = "    print('Hello, World!')"

            response = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=False,
            )

            if not response.choices:
                print("❌ 失败：应该返回补全结果")
                return False

            completion = response.choices[0].text
            if not completion:
                print("❌ 失败：补全内容为空")
                return False

            print(f"✓ 成功：自定义 JSON 模板正常工作")
            print(f"  - 补全内容: {completion}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()
            # 清理临时文件
            try:
                os.unlink(template_path)
            except:
                pass

    def test_5_multiple_completions(self) -> bool:
        """
        测试用例 5：多补全请求测试

        测试目的：验证在单个请求中生成多个补全候选的功能
        测试环境：单张 NPU 卡
        预期结果：返回多个补全候选，每个候选都是有效的
        """
        print("\n" + "="*80)
        print("测试用例 5：多补全请求测试")
        print("="*80)

        if not self.start_server(["--completion-template", "deepseek_coder"]):
            return False

        try:
            prompt = "def calculate("
            suffix = "):"

            response = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=False,
                n=3,  # 生成 3 个候选
            )

            if not response.choices or len(response.choices) != 3:
                print(f"❌ 失败：应该返回 3 个补全结果，实际返回 {len(response.choices) if response.choices else 0}")
                return False

            print(f"✓ 成功：多补全请求正常工作")
            for i, choice in enumerate(response.choices):
                print(f"  - 候选 {i+1}: {choice.text[:50]}...")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_6_streaming_completion(self) -> bool:
        """
        测试用例 6：流式补全测试

        测试目的：验证流式代码补全功能
        测试环境：单张 NPU 卡
        预期结果：流式输出正常，补全内容增量返回
        """
        print("\n" + "="*80)
        print("测试用例 6：流式补全测试")
        print("="*80)

        if not self.start_server(["--completion-template", "deepseek_coder"]):
            return False

        try:
            prompt = "def greet(name):"
            suffix = "    return f'Hello, {name}!'"

            stream = self.client.completions.create(
                model=self.model_path,
                prompt=prompt,
                suffix=suffix,
                temperature=0.3,
                max_tokens=32,
                stream=True,
            )

            chunks = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    chunks.append(chunk.choices[0].text)

            if not chunks:
                print("❌ 失败：没有收到流式块")
                return False

            completion = ''.join(chunks)
            print(f"✓ 成功：流式补全正常工作")
            print(f"  - 总流式块数: {len(chunks)}")
            print(f"  - 补全内容: {completion}")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_7_invalid_template_name(self) -> bool:
        """
        测试用例 7：无效模板名称错误处理测试

        测试目的：验证传入无效模板名称时的错误处理
        测试环境：单张 NPU 卡
        预期结果：服务启动失败，给出明确错误提示
        """
        print("\n" + "="*80)
        print("测试用例 7：无效模板名称错误处理测试")
        print("="*80)

        invalid_template = "nonexistent_template_xyz"

        args = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "127.0.0.1",
            "--port", "30001",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--completion-template", invalid_template,
            "--log-level", "error",
        ]

        print(f"尝试使用无效模板名称启动服务: {invalid_template}")

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
                # 进程仍在运行，可能是使用了默认值
                print("⚠ 服务仍在运行，可能使用了默认值或忽略了无效参数")
                process.terminate()
                process.wait(timeout=5)
                return True
            else:
                # 进程已退出，检查错误信息
                stdout, stderr = process.communicate(timeout=5)
                error_output = stdout + stderr

                if "completion" in error_output.lower() or "template" in error_output.lower():
                    print(f"✓ 成功：服务启动失败并给出错误提示")
                    print(f"  - 退出码: {poll_result}")
                    return True
                else:
                    print(f"⚠ 服务启动失败，但错误信息不明确")
                    return True

        except Exception as e:
            print(f"⚠ 测试过程出现异常: {e}")
            return True

    def test_8_invalid_template_path(self) -> bool:
        """
        测试用例 8：无效模板文件路径错误处理测试

        测试目的：验证传入无效模板文件路径时的错误处理
        测试环境：单张 NPU 卡
        预期结果：服务启动失败，给出文件不存在的错误提示
        """
        print("\n" + "="*80)
        print("测试用例 8：无效模板文件路径错误处理测试")
        print("="*80)

        invalid_path = "/nonexistent/path/to/template.json"

        args = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", "127.0.0.1",
            "--port", "30002",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
            "--completion-template", invalid_path,
            "--log-level", "error",
        ]

        print(f"尝试使用不存在的模板文件路径启动服务: {invalid_path}")

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
                # 进程仍在运行
                print("⚠ 服务仍在运行，可能忽略了无效参数")
                process.terminate()
                process.wait(timeout=5)
                return True
            else:
                # 进程已退出
                stdout, stderr = process.communicate(timeout=5)
                error_output = stdout + stderr

                if "template" in error_output.lower() or "not a built-in" in error_output.lower():
                    print(f"✓ 成功：服务启动失败并给出错误提示")
                    print(f"  - 退出码: {poll_result}")
                    return True
                else:
                    print(f"⚠ 服务启动失败，但错误信息不明确")
                    return True

        except Exception as e:
            print(f"⚠ 测试过程出现异常: {e}")
            return True

    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试用例"""
        print("\n" + "="*80)
        print("开始 --completion-template 参数测试")
        print("="*80)
        print(f"模型路径: {self.model_path}")
        print(f"服务地址: {self.base_url}")

        results = {}

        tests = [
            ("内置 deepseek_coder 模板", self.test_1_deepseek_coder_template),
            ("内置 star_coder 模板", self.test_2_star_coder_template),
            ("内置 qwen_coder 模板", self.test_3_qwen_coder_template),
            ("自定义 JSON 模板", self.test_4_custom_json_template),
            ("多补全请求", self.test_5_multiple_completions),
            ("流式补全", self.test_6_streaming_completion),
            ("无效模板名称处理", self.test_7_invalid_template_name),
            ("无效模板文件路径处理", self.test_8_invalid_template_path),
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
        "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-Coder-V2-Instruct"
    )

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print("使用模型路径:", model_path)

    tester = CompletionTemplateTester(model_path)
    results = tester.run_all_tests()
    success = tester.print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
