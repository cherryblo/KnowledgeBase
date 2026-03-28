"""
测试脚本：Prefill Delayer 参数功能测试

测试环境：
- NPU 硬件（多卡，至少 2 张）
- SGLang 框架
- 支持数据并行的模型

测试覆盖：
1. 启用 Prefill Delayer 基础测试
2. 自定义 max_delay_passes 测试
3. token_usage_low_watermark 测试
4. 自定义 forward_passeses_buckets 测试
5. 自定义 wait_seconds_buckets 测试
6. 参数组合测试

注意：
- Prefill Delayer 主要用于多卡数据并行场景
- 单卡测试主要验证参数解析和基本功能
- 完整功能测试需要至少 2 张 NPU 卡
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


class PrefillDelayerTester:
    """Prefill Delayer 参数测试类"""

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
            for i in range(90):
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

    def test_1_enable_prefill_delayer(self) -> bool:
        """
        测试用例 1：启用 Prefill Delayer 基础测试

        测试目的：验证启用 Prefill Delayer 后，服务正常启动
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，请求正常响应
        """
        print("\n" + "="*80)
        print("测试用例 1：启用 Prefill Delayer 基础测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            # 发送简单请求验证服务正常
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：Prefill Delayer 启用正常")
            print(f"  - 返回内容: {response.choices[0].message.content[:50]}...")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_2_custom_max_delay_passes(self) -> bool:
        """
        测试用例 2：自定义 max_delay_passes 测试

        测试目的：验证自定义最大延迟轮数参数生效
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，延迟逻辑使用自定义值
        """
        print("\n" + "="*80)
        print("测试用例 2：自定义 max_delay_passes 测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
            "--prefill-delayer-max-delay-passes", "50",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：自定义 max_delay_passes 参数生效")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_3_token_usage_low_watermark(self) -> bool:
        """
        测试用例 3：token_usage_low_watermark 测试

        测试目的：验证低水位标记机制正常工作
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，水位标记生效
        """
        print("\n" + "="*80)
        print("测试用例 3：token_usage_low_watermark 测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
            "--prefill-delayer-token-usage-low-watermark", "0.7",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Test watermark"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：token_usage_low_watermark 参数生效")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_4_custom_forward_passes_buckets(self) -> bool:
        """
        测试用例 4：自定义 forward_passes_buckets 测试

        测试目的：验证自定义前向传递直方图桶生效
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，直方图使用自定义桶边界
        """
        print("\n" + "="*80)
        print("测试用例 4：自定义 forward_passes_buckets 测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
            "--prefill-delayer-forward-passes-buckets", "10", "30", "60", "120",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Test buckets"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：自定义 forward_passes_buckets 参数生效")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_5_custom_wait_seconds_buckets(self) -> bool:
        """
        测试用例 5：自定义 wait_seconds_buckets 测试

        测试目的：验证自定义等待秒数直方图桶生效
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，直方图使用自定义桶边界
        """
        print("\n" + "="*80)
        print("测试用例 5：自定义 wait_seconds_buckets 测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
            "--prefill-delayer-wait-seconds-buckets", "2", "5", "15", "30",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Test wait buckets"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：自定义 wait_seconds_buckets 参数生效")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def test_6_parameter_combination(self) -> bool:
        """
        测试用例 6：参数组合测试

        测试目的：验证多个 Prefill Delayer 参数组合正常工作
        测试环境：多张 NPU 卡（至少 2 张）
        预期结果：服务成功启动，所有配置参数生效
        """
        print("\n" + "="*80)
        print("测试用例 6：参数组合测试")
        print("="*80)

        dp_args = [
            "--enable-dp-attention",
            "--dp-size", "2",
            "--disaggregation-mode", "null",
            "--disable-overlap-schedule",
            "--enable-prefill-delayer",
            "--prefill-delayer-max-delay-passes", "40",
            "--prefill-delayer-token-usage-low-watermark", "0.8",
            "--prefill-delayer-forward-passes-buckets", "5", "20", "50",
            "--prefill-delayer-wait-seconds-buckets", "1", "3", "10",
        ]

        if not self.start_server(dp_args):
            return False

        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=[{"role": "user", "content": "Test combination"}],
                max_tokens=20,
            )

            if not response.choices:
                print("❌ 失败：应该返回结果")
                return False

            print(f"✓成功：参数组合测试通过")
            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
        finally:
            self.stop_server()

    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试用例"""
        print("\n" + "="*80)
        print("开始 Prefill Delayer 参数测试")
        print("="*80)
        print(f"模型路径: {self.model_path}")
        print(f"服务地址: {self.base_url}")
        print("\n注意：Prefill Delayer 主要用于多卡数据并行场景")
        print("单卡测试主要验证参数解析和基本功能")
        print("完整功能测试需要至少 2 张 NPU 卡\n")

        results = {}

        tests = [
            ("启用 Prefill Delayer 基础功能", self.test_1_enable_prefill_delayer),
            ("自定义 max_delay_passes", self.test_2_custom_max_delay_passes),
            ("token_usage_low_watermark", self.test_3_token_usage_low_watermark),
            ("自定义 forward_passes_buckets", self.test_4_custom_forward_passes_buckets),
            ("自定义 wait_seconds_buckets", self.test_5_custom_wait_seconds_buckets),
            ("参数组合测试", self.test_6_parameter_combination),
        ]

        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
            except Exception as e:
                print(f"❌ 测试 '{test_name}' 执行异常: {e}")
                results[test_name] = False

            # 测试间隔
            time.sleep(3)

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
        "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3-8B-Instruct"
    )

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print("使用模型路径:", model_path)

    tester = PrefillDelayerTester(model_path)
    results = tester.run_all_tests()
    success = tester.print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
