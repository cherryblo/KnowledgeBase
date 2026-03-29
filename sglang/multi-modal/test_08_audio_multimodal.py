"""
用例8：音频多模态推理测试

测试目的：验证Multi-Modal特性在NPU环境下能处理音频+文本输入
测试环境：使用1张NPU卡，单机部署
"""

import base64
import json
import os
import time
import unittest
from pathlib import Path

import requests


class TestAudioMultimodal(unittest.TestCase):
    """音频多模态推理测试"""

    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
        cls.model_path = os.getenv(
            "MODEL_PATH", "Qwen/Qwen2-Audio-7B-Instruct"
        )
        cls.timeout = 300

        # 创建测试音频
        cls.test_audio_path = cls._create_test_audio()

    @staticmethod
    def _create_test_audio():
        """创建一个简单的测试音频文件"""
        import numpy as np

        # 生成一个简单的正弦波音频
        sample_rate = 16000
        duration = 2  # 秒
        frequency = 440  # Hz (A4音符)

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = np.sin(2 * np.pi * frequency * t)

        # 转换为16位PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # 保存为WAV文件
        audio_path = Path("/tmp/test_audio.wav")
        import wave

        with wave.open(str(audio_path), "w") as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return str(audio_path)

    def _encode_audio(self, audio_path):
        """将音频编码为base64"""
        with open(audio_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def test_01_server_health_check(self):
        """测试1：服务器健康检查"""
        print("\n=== 测试1：服务器健康检查 ===")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(response.status_code, 200)
            print("✓ 服务器健康检查通过")
        except Exception as e:
            self.fail(f"服务器健康检查失败: {e}")

    def test_02_model_info(self):
        """测试2：获取模型信息"""
        print("\n=== 测试2：获取模型信息 ===")

        try:
            response = requests.get(f"{self.base_url}/get_model_info", timeout=10)
            self.assertEqual(response.status_code, 200)
            model_info = response.json()
            print(f"✓ 模型信息: {model_info.get('model_path', 'N/A')}")
        except Exception as e:
            self.fail(f"获取模型信息失败: {e}")

    def test_03_audio_transcription(self):
        """测试3：音频转录"""
        print("\n=== 测试3：音频转录 ===")

        audio_base64 = self._encode_audio(self.test_audio_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_base64}"
                            },
                        },
                        {"type": "text", "text": "Transcribe this audio."},
                    ],
                }
            ],
            "max_tokens": 100,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(response.status_code, 200)
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 音频转录成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 转录结果: {content}")

        except Exception as e:
            self.fail(f"音频转录失败: {e}")

    def test_04_audio_question_answering(self):
        """测试4：音频问答"""
        print("\n=== 测试4：音频问答 ===")

        audio_base64 = self._encode_audio(self.test_audio_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is the dominant frequency in this audio?",
                        },
                    ],
                }
            ],
            "max_tokens": 50,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            elapsed_time = time.time() - start_time

            self.assertEqual(response.status_code, 200)
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 音频问答成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 模型回答: {content}")

        except Exception as e:
            self.fail(f"音频问答失败: {e}")

    def test_05_audio_classification(self):
        """测试5：音频分类"""
        print("\n=== 测试5：音频分类 ===")

        audio_base64 = self._encode_audio(self.test_audio_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Classify this audio. Is it speech, music, or noise?",
                        },
                    ],
                }
            ],
            "max_tokens": 50,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            stream_time = time.time() - start_time

            self.assertEqual(response.status_code, 200)
            result = response.json()

            content = result["choices"][0]["message"]["content"]
            print(f"✓ 音频分类成功，耗时: {elapsed_time:.2f}秒")
            print(f"✓ 分类结果: {content}")

        except Exception as e:
            self.fail(f"音频分类失败: {e}")

    def test_06_streaming_audio_inference(self):
        """测试6：流式音频推理"""
        print("\n=== 测试6：流式音频推理 ===")

        audio_base64 = self._encode_audio(self.test_audio_path)

        payload = {
            "model": self.model_path,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:audio/wav;base64,{audio_base64}"
                            },
                        },
                        {"type": "text", "text": "Describe this audio."},
                    ],
                }
            ],
            "max_tokens": 100,
            "stream": True,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            self.assertEqual(response.status_code, 200)

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                full_content += content
                        except json.JSONDecodeError:
                            pass

            print(f"✓ 流式推理成功")
            print(f"✓ 完整回答: {full_content}")
            self.assertTrue(len(full_content) > 0, "流式推理未返回任何内容")

        except Exception as e:
            self.fail(f"流式推理失败: {e}")

    @classmethod
    def tearDownClass(cls):
        """清理测试文件"""
        try:
            if os.path.exists(cls.test_audio_path):
                os.remove(cls.test_audio_path)
        except Exception as e:
            print(f"清理测试文件时出错: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
