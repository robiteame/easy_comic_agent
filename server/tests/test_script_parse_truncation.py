"""长剧本解析与模型输出截断的验收测试。

覆盖用户报告的「上传长剧本后一直显示正在解析、进度停在 0%」问题的完整修复链：

1. 普通长度剧本一次解析成功；
2. 超长剧本按场次分段解析、合并去重并通过统一 schema 校验，场次与角色不丢失；
3. ``finish_reason=length`` 被识别为输出截断；
4. ``output_tokens == max_tokens`` 被保守识别为可能截断；
5. 截断后系统改变策略（分段/细分）而不是同配置重试；
6. 单次调用截断后自动降级分段解析并成功合并；
7. 连续截断（分段也救不了）时图运行终止为 failed，不再继续调用 LLM；
8. 失败任务收敛到 failed 后预算预留被释放，background_jobs 的错误码为
   llm_output_truncated（前端据此停止 loading 并展示专门文案）。
"""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from agent import decision, graph  # noqa: E402
from agent.contracts import FailureKind, RecoveryStrategy  # noqa: E402
from agent.nodes import script_parser  # noqa: E402
from agent.output_schemas import parse_script_output  # noqa: E402
from config import settings  # noqa: E402
from db import SessionLocal, init_db  # noqa: E402
from models import BackgroundJob, BudgetReservation, Project  # noqa: E402
from services import llm_service as llm_service_module  # noqa: E402
from services import task_registry  # noqa: E402
from services.job_types import ERROR_CODE_LLM_OUTPUT_TRUNCATED, classify_error_code  # noqa: E402
from services.llm_service import LLMOutputTruncatedError, LLMService, large_json_max_tokens  # noqa: E402
from services.providers.endpoint import EndpointConfig  # noqa: E402
from tests.support.test_environment import TEST_ROOT  # noqa: F401,E402


def chat_response(
    content: str, *, finish_reason: str = "stop", completion_tokens: int | None = 50, prompt_tokens: int = 120
) -> SimpleNamespace:
    """构造最小化的 ChatCompletion 形状（llm_service 只读这些属性）。"""

    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    message = SimpleNamespace(content=content, reasoning_content=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class FakeAdapter:
    """记录请求并按脚本返回预设响应的假 openai-chat 适配器。"""

    protocol = "openai-chat"

    def __init__(self, endpoint: EndpointConfig, handler) -> None:
        self.endpoint = endpoint
        self.handler = handler
        self.requests: list[dict] = []

    async def complete_json(self, *, messages, model, temperature, max_tokens):
        self.requests.append(
            {
                "messages": messages,
                "user_prompt": messages[-1]["content"] if messages else "",
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        result = self.handler(self.requests[-1])
        if isinstance(result, Exception):
            raise result
        return result

    async def complete(self, *, messages, model, temperature, max_tokens):
        return await self.complete_json(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)


def _install_fake_llm(handler) -> tuple[LLMService, FakeAdapter]:
    service = LLMService()
    endpoint = EndpointConfig(
        protocol="openai-chat", base_url="https://fake.test/v1", api_key="test-key", model="fake-model"
    )
    service._endpoint = endpoint
    service._fallback_endpoint = None
    service.model = endpoint.model
    adapter = FakeAdapter(endpoint, handler)
    original = service._adapter_for
    service._adapter_for = lambda ep: adapter  # type: ignore[method-assign]
    service._original_adapter_for = original  # type: ignore[attr-defined]
    return service, adapter


def _scene(number: int, line: str, location: str = "青山村") -> dict:
    return {
        "scene_number": number,
        "location": location,
        "characters_in_scene": ["林夏"],
        "actions": f"第 {number} 场剧情",
        "dialogue": [{"character": "林夏", "line": line, "emotion": "neutral"}],
        "emotion": "neutral",
        "camera_suggestion": "medium",
    }


class LLMTruncationDetectionTests(unittest.TestCase):
    """验收 3/4：finish_reason=length 与 output_tokens==max_tokens 都判截断。"""

    def test_finish_reason_length_raises_truncation_with_diagnostics(self) -> None:
        service, adapter = _install_fake_llm(
            lambda req: chat_response('{"ok": true}', finish_reason="length", completion_tokens=4096)
        )
        with self.assertRaises(LLMOutputTruncatedError) as raised:
            asyncio.run(service.call_json("sys", "user", max_tokens=4096))
        error = raised.exception
        self.assertEqual(error.finish_reason, "length")
        self.assertEqual(error.output_tokens, 4096)
        self.assertEqual(error.max_tokens, 4096)
        self.assertIn("请增加输出额度或按场次分段解析", str(error))
        self.assertIn("finish_reason=length", str(error))
        self.assertFalse(error.suspected)

    def test_output_tokens_equal_to_max_tokens_is_suspected_truncation(self) -> None:
        service, _ = _install_fake_llm(
            lambda req: chat_response('{"ok": true}', finish_reason="stop", completion_tokens=4096)
        )
        with self.assertRaises(LLMOutputTruncatedError) as raised:
            asyncio.run(service.call_json("sys", "user", max_tokens=4096))
        self.assertTrue(raised.exception.suspected)
        self.assertEqual(raised.exception.output_tokens, 4096)

    def test_truncation_is_never_retried_with_same_config(self) -> None:
        service, adapter = _install_fake_llm(
            lambda req: chat_response('{"ok": true}', finish_reason="length", completion_tokens=4096)
        )
        with self.assertRaises(LLMOutputTruncatedError):
            asyncio.run(service.call_json("sys", "user", max_retries=2, max_tokens=4096))
        self.assertEqual(len(adapter.requests), 1, "截断是确定性失败，不得用相同配置重试")

    def test_normal_output_passes_through_and_parses(self) -> None:
        service, adapter = _install_fake_llm(
            lambda req: chat_response('{"title": "青山"}', finish_reason="stop", completion_tokens=120)
        )
        result = asyncio.run(service.call_json("sys", "user", max_tokens=4096))
        self.assertEqual(result["title"], "青山")
        self.assertEqual(len(adapter.requests), 1)

    def test_plain_text_call_detects_truncation_too(self) -> None:
        service, _ = _install_fake_llm(
            lambda req: chat_response("很长的剧本……", finish_reason="length", completion_tokens=4096)
        )
        with self.assertRaises(LLMOutputTruncatedError):
            asyncio.run(service.call("sys", "user", max_tokens=4096))


class LargeJsonBudgetTests(unittest.TestCase):
    """解析输出额度：默认 16384，端点声明更小上限时收紧到端点值。"""

    def test_default_budget_is_settings_target(self) -> None:
        original = llm_service_module.get_endpoint
        llm_service_module.get_endpoint = lambda capability: EndpointConfig(
            protocol="openai-chat", base_url="https://x", model="m"
        )
        try:
            self.assertEqual(large_json_max_tokens(), settings.LLM_LARGE_JSON_MAX_TOKENS)
        finally:
            llm_service_module.get_endpoint = original

    def test_endpoint_declared_cap_wins_when_smaller(self) -> None:
        original = llm_service_module.get_endpoint
        llm_service_module.get_endpoint = lambda capability: EndpointConfig(
            protocol="openai-chat", base_url="https://x", model="m", params={"max_output_tokens": 8192}
        )
        try:
            self.assertEqual(large_json_max_tokens(), 8192)
        finally:
            llm_service_module.get_endpoint = original


class _RecordingLLM:
    """按调用顺序返回预设结果的 LLM stub，记录每次调用的 prompt 与参数。"""

    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list[dict] = []
        self.available = True

    async def call_json(self, system_prompt, user_prompt, temperature=0.3, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt, "kwargs": kwargs})
        if not self.results:
            raise AssertionError("意外的额外 LLM 调用")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class LongScriptSegmentationTests(unittest.TestCase):
    """验收 1/2/5/6：短剧本一次成功；长剧本分段、合并去重、截断降级。"""

    def setUp(self) -> None:
        self.progress: list[str] = []
        self._original_progress = script_parser._report_progress
        self._original_memory = script_parser.project_memory

        async def record_progress(project_id, message, progress=None):
            self.progress.append(message)

        script_parser._report_progress = record_progress
        script_parser.project_memory = SimpleNamespace(
            save_characters=lambda *a, **k: None,
            save_narrative_context=lambda *a, **k: None,
        )

    def tearDown(self) -> None:
        script_parser._report_progress = self._original_progress
        script_parser.project_memory = self._original_memory

    def _long_script(self, scenes: int = 6, chars_per_scene: int = 1800) -> str:
        filler = "山风吹过，林夏望着远处的青山出神。" * (chars_per_scene // 16)
        return "\n\n".join(f"第{i}场\n{filler}\n林夏：这是第{i}场的台词。" for i in range(1, scenes + 1))

    def _run_parser(self, stub: _RecordingLLM, user_input: str, state: dict | None = None) -> dict:
        original = script_parser.llm_service
        script_parser.llm_service = stub
        try:
            base = {"project_id": "parse-long-project", "user_input": user_input, "style": "anime"}
            base.update(state or {})
            return asyncio.run(script_parser.run(base))
        finally:
            script_parser.llm_service = original

    def test_short_script_parses_in_single_call(self) -> None:
        stub = _RecordingLLM(
            [
                {
                    "title": "青山",
                    "genre": "乡村",
                    "style_suggestion": "anime",
                    "characters": [{"name": "林夏", "appearance": {"hair": "黑长直"}, "voice_type": "少女"}],
                    "script_scenes": [_scene(1, "早上好")],
                    "logic_issues": [],
                }
            ]
        )
        result = self._run_parser(stub, "林夏走进教室，和同学打招呼。")
        self.assertEqual(len(stub.calls), 1, "普通长度剧本必须一次解析成功")
        self.assertEqual(result["script_title"], "青山")
        self.assertEqual(len(result["script_scenes"]), 1)
        self.assertEqual(stub.calls[0]["kwargs"]["max_tokens"], settings.LLM_LARGE_JSON_MAX_TOKENS)

    def test_long_script_is_segmented_merged_and_schema_valid(self) -> None:
        script = self._long_script(scenes=6, chars_per_scene=2400)
        self.assertGreater(len(script), settings.LLM_SCRIPT_PARSE_WHOLE_INPUT_CHARS)
        self.assertEqual(
            len(script_parser.segment_script(script, limit=settings.LLM_SCRIPT_PARSE_SEGMENT_CHARS)),
            3,
            "测试剧本按场次边界切成 3 段（每段 2 场）",
        )
        segment_payloads = [
            {
                "title": "",
                "genre": "",
                "characters": [{"name": "林夏", "appearance": {"hair": "黑长直"}, "voice_type": "少女"}],
                "script_scenes": [_scene(1, "第一段台词"), _scene(2, "第二段台词")],
            },
            {
                "title": "",
                "genre": "",
                # 同名角色再次出现：合并时必须去重，不产生第二个林夏。
                "characters": [{"name": "林夏", "appearance": {}, "voice_type": ""}],
                "script_scenes": [_scene(3, "第三段台词"), _scene(4, "第四段台词")],
            },
            {
                "title": "",
                "genre": "",
                "characters": [{"name": "陈默", "appearance": {}, "voice_type": "少年"}],
                "script_scenes": [_scene(5, "第五段台词"), _scene(6, "第六段台词")],
            },
        ]
        stub = _RecordingLLM(segment_payloads)
        result = self._run_parser(stub, script)

        self.assertEqual(len(stub.calls), 3)
        for index, call in enumerate(stub.calls, start=1):
            self.assertIn(f"第 {index}/3 段", call["user"], "分段 prompt 必须携带分段上下文")
        self.assertIn("正在解析剧本（第 1/3 段）", self.progress)
        self.assertIn("正在解析剧本（第 3/3 段）", self.progress)

        # 合并结果重新通过统一 schema 校验（run 内部已校验，这里独立复核一遍）。
        reparsed = parse_script_output(
            {
                "characters": [
                    {"name": item["name"], "appearance": item["appearance"]} for item in result["characters"]
                ],
                "script_scenes": [
                    {
                        "scene_number": scene["scene_number"],
                        "location": scene["location"],
                        "characters_in_scene": scene["characters_in_scene"],
                        "actions": scene["actions"],
                        "dialogue": scene["dialogue"],
                    }
                    for scene in result["script_scenes"]
                ],
            },
            fallback_style="anime",
        )
        self.assertEqual(len(reparsed.script_scenes), 6, "场次不能丢失")
        # 角色去重：林夏在两段中重复出现，只保留一份；陈默保留。
        names = [item["name"] for item in result["characters"]]
        self.assertEqual(names.count("林夏"), 1)
        self.assertEqual(names.count("陈默"), 1)
        # 场次重新连续编号且保持原顺序。
        numbers = [scene["scene_number"] for scene in result["script_scenes"]]
        self.assertEqual(numbers, [1, 2, 3, 4, 5, 6])
        lines = [scene["dialogue"][0]["line"] for scene in result["script_scenes"]]
        self.assertEqual(lines, [f"第{order}段台词" for order in ["一", "二", "三", "四", "五", "六"]])

    def test_whole_call_truncation_falls_back_to_segments(self) -> None:
        script = "第一场\n" + ("林夏在田间劳作，汗水滑落。\n" * 200)  # > 分段下限但 < 单次阈值
        self.assertGreater(len(script), settings.LLM_SCRIPT_PARSE_SEGMENT_MIN_CHARS)
        self.assertLessEqual(len(script), settings.LLM_SCRIPT_PARSE_WHOLE_INPUT_CHARS)
        truncation = LLMOutputTruncatedError(
            finish_reason="length", output_tokens=4096, max_tokens=16384, provider="openai-chat", model="mimo"
        )
        segment_payloads = [
            truncation,  # 第一次：整本单次调用被截断
            {  # 之后：按场次边界分成的两段
                "title": "",
                "genre": "",
                "characters": [{"name": "林夏", "voice_type": "少女"}],
                "script_scenes": [_scene(1, "分段后的台词")],
            },
            {
                "title": "",
                "genre": "",
                "characters": [{"name": "林夏", "voice_type": "少女"}],
                "script_scenes": [_scene(2, "第二段台词")],
            },
        ]
        stub = _RecordingLLM(segment_payloads)
        result = self._run_parser(stub, script)

        self.assertGreaterEqual(len(stub.calls), 2, "截断后必须改变策略（分段），而不是报错或同配置重试")
        self.assertNotIn("第 1/1 段", stub.calls[1]["user"])
        self.assertIn("这是一部长剧本的第 1/2 段", stub.calls[1]["user"])
        self.assertEqual(
            [scene["dialogue"][0]["line"] for scene in result["script_scenes"]], ["分段后的台词", "第二段台词"]
        )

    def test_unfixable_truncation_raises_deterministic_error(self) -> None:
        script = "第一场\n林夏说了一句话。"
        truncation = LLMOutputTruncatedError(
            finish_reason="length", output_tokens=16384, max_tokens=16384, provider="openai-chat", model="mimo"
        )
        stub = _RecordingLLM([truncation, truncation, truncation, truncation])
        with self.assertRaises(RuntimeError) as raised:
            self._run_parser(stub, script)
        message = str(raised.exception)
        self.assertIn("被截断", message)
        self.assertIn("请增加输出额度或按场次分段解析", message)
        self.assertLessEqual(len(stub.calls), 4, "细分到下限后必须停止调用 LLM")

    def test_provider_switch_selects_fallback_endpoint(self) -> None:
        stub = _RecordingLLM(
            [
                {
                    "title": "青山",
                    "genre": "乡村",
                    "characters": [{"name": "林夏", "voice_type": "少女"}],
                    "script_scenes": [_scene(1, "切换后的台词")],
                }
            ]
        )
        self._run_parser(
            stub,
            "林夏走进教室。",
            state={"provider_switch": {"director_planning": "script_fallback"}},
        )
        self.assertTrue(stub.calls[0]["kwargs"].get("prefer_fallback"), "恢复决策切换 Provider 后解析必须改走备端点")


class SegmentationUnitTests(unittest.TestCase):
    """分段与合并的纯函数行为：不丢字、不重复、去重角色/场景。"""

    def test_segments_cover_all_text_in_order(self) -> None:
        text = "\n".join(f"第{i}场\n" + ("剧情推进。" * 60) for i in range(1, 9))
        segments = script_parser.segment_script(text, limit=1200)
        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segments), text, "分段拼接必须等于原文，不允许丢字或重复")
        for segment in segments:
            self.assertLessEqual(len(segment), 1200 + 200, "分段长度受上限约束（允许一行溢出）")

    def test_merge_dedupes_characters_scenes_and_keeps_dialogue_order(self) -> None:
        def output(characters: list[str], scenes: list[dict]):
            return parse_script_output(
                {
                    "characters": [{"name": name} for name in characters],
                    "script_scenes": scenes,
                }
            )

        first = output(["林夏", "陈默"], [_scene(1, "一"), _scene(2, "二")])
        second = output(["陈默", "林夏", "林夏"], [_scene(1, "一"), _scene(3, "三")])  # 场景 1 重复出现
        merged = script_parser.merge_script_outputs([first, second], fallback_style="anime")

        names = [item.name for item in merged.characters]
        self.assertEqual(names, ["林夏", "陈默"], "角色按首次出现顺序去重")
        self.assertEqual(
            [(scene.scene_number, scene.dialogue[0].line) for scene in merged.script_scenes],
            [(1, "一"), (2, "二"), (3, "三")],
            "重复场景被丢弃，剩余场次按原顺序重新编号",
        )


class TruncationRecoveryDecisionTests(unittest.TestCase):
    """验收 7（决策层）：截断只允许换 Provider；无端点可换时立即终止。"""

    def test_truncation_classifies_before_invalid_output(self) -> None:
        message = "Mimo 剧本解析失败: 模型输出超过最大长度并被截断（finish_reason=length）；Expecting ',' delimiter"
        record = decision.classify_failure(stage="director_planning", message=message)
        self.assertEqual(record.kind, FailureKind.LLM_OUTPUT_TRUNCATED)
        self.assertEqual(decision.primary_strategy_for(record.kind), RecoveryStrategy.SWITCH_PROVIDER)

    def test_truncation_never_proposes_prompt_revision(self) -> None:
        failure = decision.classify_failure(
            stage="director_planning", kind=FailureKind.LLM_OUTPUT_TRUNCATED, message="输出被截断"
        )
        candidates = decision.recovery_candidates(
            failure,
            provider_profiles_by_capability={"script": []},
            retries_remaining=2,
        )
        strategies = {item.strategy for item in candidates}
        self.assertNotIn(RecoveryStrategy.REVISE_PROMPT, strategies)
        self.assertNotIn(RecoveryStrategy.RETRY, strategies)
        selected = [item for item in candidates if item.strategy is RecoveryStrategy.SWITCH_PROVIDER]
        self.assertTrue(
            selected and not selected[0].provider_capability_ok, "没有可换端点时 switch_provider 必须不可行"
        )

    def test_switch_targets_fallback_endpoint_not_current(self) -> None:
        from agent.contracts import ProviderProfile

        profiles = [
            ProviderProfile(capability="script", provider="openai-chat", available=True, is_current=True),
            ProviderProfile(capability="script", provider="script_fallback", available=True),
        ]
        target = decision._switch_provider_target(
            profiles, failing_provider="", failure_kind=FailureKind.LLM_OUTPUT_TRUNCATED
        )
        self.assertEqual(target, "script_fallback")

    def test_job_error_code_for_truncation(self) -> None:
        text = "自动流程失败: [auto_abort] 模型输出超过最大长度并被截断（finish_reason=length，输出 4096/16384 tokens）"
        self.assertEqual(classify_error_code(text), ERROR_CODE_LLM_OUTPUT_TRUNCATED)


class TruncationGraphTerminationTests(unittest.TestCase):
    """验收 7（图级）：连续截断且无 Provider 可换时任务终止，不再调用 LLM。"""

    def test_graph_fails_fast_without_extra_llm_calls(self) -> None:
        init_db()
        calls = {"count": 0}

        async def failing_run(state):
            calls["count"] += 1
            raise RuntimeError(
                "Mimo 剧本解析失败: 模型输出超过最大长度并被截断"
                "（finish_reason=length，输出 4096/16384 tokens）；请增加输出额度或按场次分段解析"
            )

        original_run = script_parser.run
        original_profiles = decision.provider_profiles
        script_parser.run = failing_run
        # 测试环境没有真实 script 端点密钥：保证「无可用切换目标」这一前提稳定。
        decision.provider_profiles = lambda capability, reference_required=False: []
        try:
            result = asyncio.run(
                graph.get_graph().ainvoke(
                    {
                        "project_id": "graph-truncation-project",
                        "mode": "auto",
                        "initial_state": {"project_id": "graph-truncation-project", "user_input": "第一场 林夏"},
                        "quality_profile": "standard",
                        "run_id": "auto",
                        "current_step": "",
                        "errors": [],
                    },
                    config={"recursion_limit": 40},
                )
            )
        finally:
            script_parser.run = original_run
            decision.provider_profiles = original_profiles

        self.assertEqual(result.get("run_status"), "failed")
        self.assertTrue(result.get("errors"))
        self.assertIn("被截断", "\n".join(result["errors"]), "终止原因必须保留截断诊断信息")
        self.assertLessEqual(calls["count"], 1, "无 Provider 可换时不得再次调用 LLM 解析")


class FailedJobBudgetReleaseTests(unittest.TestCase):
    """验收 6/8：失败任务收敛 failed、错误码正确、预算预留释放。"""

    def test_failed_pipeline_releases_reservation_and_sets_error_code(self) -> None:
        init_db()
        job_key = "project:budget-release-project:pipeline:auto"
        db = SessionLocal()
        try:
            # 只清理本测试自己的行：全量套件里其它测试的项目行可能被子表引用，
            # 整表 DELETE 会触发外键约束。
            db.query(BackgroundJob).filter(
                (BackgroundJob.idempotency_key == job_key) | (BackgroundJob.idempotency_key.like(f"{job_key}#%"))
            ).delete(synchronize_session=False)
            db.query(BudgetReservation).filter(BudgetReservation.reservation_key == job_key).delete(
                synchronize_session=False
            )
            db.query(Project).filter(Project.id == "budget-release-project").delete(synchronize_session=False)
            db.commit()
            db.add(Project(id="budget-release-project", title="预算释放"))
            db.commit()
        finally:
            db.close()
        claim = task_registry.claim_job(
            job_key,
            "project:budget-release-project",
            job_type="pipeline",
            project_id="budget-release-project",
            current_step="parse_script",
            message="已排队，正在解析剧本",
        )
        self.assertTrue(claim.claimed)

        db = SessionLocal()
        try:
            reservation = db.query(BudgetReservation).filter(BudgetReservation.reservation_key == job_key).first()
            self.assertIsNotNone(reservation, "任务启动时必须建立预算预留")
        finally:
            db.close()

        async def failing_pipeline():
            raise RuntimeError(
                "自动流程失败: [auto_abort] 模型输出超过最大长度并被截断"
                "（finish_reason=length，输出 4096/16384 tokens）；请增加输出额度或按场次分段解析"
            )

        async def scenario():
            task = task_registry.start(job_key, failing_pipeline())
            try:
                await task
            except RuntimeError:
                pass  # 失败协程的异常由任务持有；终态收敛在 done_callback 里完成。
            # 完成回调在事件循环的下一轮执行；等待其收敛。
            await asyncio.sleep(0.05)

        asyncio.run(scenario())

        db = SessionLocal()
        try:
            job = db.query(BackgroundJob).filter(BackgroundJob.idempotency_key == job_key).first()
            self.assertIsNotNone(job)
            self.assertEqual(job.status, "failed")
            self.assertEqual(job.error_code, ERROR_CODE_LLM_OUTPUT_TRUNCATED)
            self.assertIn("被截断", job.error_message or job.error or "")
            reservations = db.query(BudgetReservation).filter(BudgetReservation.reservation_key == job_key).all()
            self.assertTrue(reservations, "任务启动时必须建立预算预留")
            active = [row for row in reservations if row.status == "active"]
            self.assertEqual(active, [], "任务失败后预算预留必须被释放")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
