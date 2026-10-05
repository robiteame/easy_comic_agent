"""P2: 配置与风格模板原子写入的回归测试。

覆盖：同目录临时文件 + fsync + os.replace、写入失败保留原文件、并发保存不互相
覆盖、损坏 JSON 记录警告并备份为 .corrupt。
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from services import atomic_json, skill_config_service, style_templates  # noqa: E402
from test_environment import TEST_ROOT  # noqa: F401,E402


def _temp_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.startswith("."))


class AtomicWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TEST_ROOT / "atomic" / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)

    def test_write_json_keeps_encoding_and_formatting(self) -> None:
        target = self.directory / "config.json"
        payload = {"名称": "默认方案", "items": [1, 2, 3]}
        atomic_json.atomic_write_json(target, payload)

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), payload)
        self.assertEqual(target.read_text(encoding="utf-8"), json.dumps(payload, ensure_ascii=False, indent=2))
        self.assertEqual(_temp_files(self.directory), [])

    def test_failed_replace_keeps_the_original_file(self) -> None:
        target = self.directory / "config.json"
        atomic_json.atomic_write_json(target, {"version": 1})

        with patch.object(atomic_json.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                atomic_json.atomic_write_json(target, {"version": 2})

        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"version": 1})
        self.assertEqual(_temp_files(self.directory), [])

    def test_missing_and_empty_files_use_the_default(self) -> None:
        target = self.directory / "config.json"
        self.assertEqual(atomic_json.read_json_file(target, default={}), {})
        target.write_text("   ", encoding="utf-8")
        self.assertEqual(atomic_json.read_json_file(target, default={"fallback": True}), {"fallback": True})

    def test_corrupt_json_is_logged_backed_up_and_defaulted(self) -> None:
        target = self.directory / "config.json"
        target.write_text("{ 不是合法 JSON", encoding="utf-8")

        with self.assertLogs("services.atomic_json", level=logging.WARNING) as captured:
            result = atomic_json.read_json_file(target, default={"fallback": True})

        self.assertEqual(result, {"fallback": True})
        self.assertTrue(any("损坏" in line for line in captured.output))
        backup = target.with_name(target.name + ".corrupt")
        self.assertTrue(backup.exists())
        self.assertIn("不是合法 JSON", backup.read_text(encoding="utf-8"))
        self.assertFalse(target.exists())

    def test_corrupt_backup_does_not_overwrite_an_existing_backup(self) -> None:
        target = self.directory / "config.json"
        existing = target.with_name(target.name + ".corrupt")
        existing.write_text("previous", encoding="utf-8")
        target.write_text("broken", encoding="utf-8")

        with self.assertLogs("services.atomic_json", level=logging.WARNING):
            atomic_json.read_json_file(target, default={})

        self.assertEqual(existing.read_text(encoding="utf-8"), "previous")
        backups = [path.name for path in self.directory.glob("config.json.*.corrupt")]
        self.assertEqual(len(backups), 1)
        self.assertEqual(_temp_files(self.directory), [])

    def test_path_lock_is_shared_per_resolved_path(self) -> None:
        first = self.directory / "config.json"
        second = self.directory / "." / "config.json"
        self.assertIs(atomic_json.path_lock(first), atomic_json.path_lock(second))
        self.assertIsNot(atomic_json.path_lock(first), atomic_json.path_lock(self.directory / "other.json"))


class SkillConfigAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = skill_config_service._store_path()
        self._reset_store()

    def tearDown(self) -> None:
        self._reset_store()

    def _reset_store(self) -> None:
        for path in self.store.parent.glob(self.store.name + "*"):
            path.unlink(missing_ok=True)

    def test_concurrent_saves_do_not_lose_updates(self) -> None:
        template_ids = [f"并发方案{index}" for index in range(12)]
        barrier = threading.Barrier(len(template_ids))

        def save(name: str) -> None:
            barrier.wait(timeout=10)
            skill_config_service.save_skill_template({"name": name})

        with ThreadPoolExecutor(max_workers=len(template_ids)) as pool:
            list(pool.map(save, template_ids))

        templates = skill_config_service.list_skill_templates()["templates"]
        saved_names = {template["name"] for template in templates}
        for name in template_ids:
            self.assertIn(name, saved_names)
        # 文件仍是合法 JSON（没有交错写入产生的半截内容）。
        json.loads(self.store.read_text(encoding="utf-8"))
        self.assertEqual(_temp_files(self.store.parent), [])

    def test_concurrent_bindings_and_templates_stay_consistent(self) -> None:
        barrier = threading.Barrier(6)

        def save_template(index: int) -> None:
            barrier.wait(timeout=10)
            skill_config_service.save_skill_template({"name": f"方案{index}"})

        def save_bindings(index: int) -> None:
            barrier.wait(timeout=10)
            try:
                skill_config_service.set_skill_bindings({"project_bindings": {f"project-{index}": "default"}})
            except ValueError:
                pass

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(save_template, index) for index in range(3)]
            futures += [pool.submit(save_bindings, index) for index in range(3)]
            for future in futures:
                future.result()

        payload = json.loads(self.store.read_text(encoding="utf-8"))
        self.assertIn("templates", payload)
        self.assertIn("default", payload["templates"])

    def test_corrupt_store_falls_back_to_defaults_with_a_backup(self) -> None:
        self.store.write_text("{ 坏掉的配置", encoding="utf-8")

        with self.assertLogs("services.atomic_json", level=logging.WARNING):
            config = skill_config_service.list_skill_templates()

        self.assertIn("default", {template["id"] for template in config["templates"]})
        self.assertTrue(self.store.with_name(self.store.name + ".corrupt").exists())
        # 读取后会用默认配置重建文件，并可继续保存。
        self.assertTrue(self.store.exists())
        saved = skill_config_service.save_skill_template({"name": "恢复后的方案"})
        self.assertEqual(saved["name"], "恢复后的方案")


class StyleTemplateAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = style_templates._custom_template_path()
        self._reset_store()

    def tearDown(self) -> None:
        self._reset_store()

    def _reset_store(self) -> None:
        for path in self.store.parent.glob(self.store.name + "*"):
            path.unlink(missing_ok=True)

    def test_concurrent_style_creates_keep_every_template(self) -> None:
        keys = [f"custom_probe_{index}" for index in range(10)]
        barrier = threading.Barrier(len(keys))

        def create(key: str) -> None:
            barrier.wait(timeout=10)
            style_templates.create_custom_style_template(key, key, f"{key} keywords")

        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            list(pool.map(create, keys))

        stored = json.loads(self.store.read_text(encoding="utf-8"))
        for key in keys:
            self.assertIn(key, stored)
        self.assertEqual(_temp_files(self.store.parent), [])

    def test_corrupt_custom_templates_still_return_builtins(self) -> None:
        self.store.write_text("不是 JSON", encoding="utf-8")

        with self.assertLogs("services.atomic_json", level=logging.WARNING):
            options = style_templates.style_options()

        values = {option["value"] for option in options}
        self.assertIn("anime", values)
        self.assertIn("realistic", values)
        self.assertTrue(self.store.with_name(self.store.name + ".corrupt").exists())
        # 损坏文件被隔离后仍可重新创建自定义风格。
        created = style_templates.create_custom_style_template("custom_after_corrupt", "恢复", "keywords")
        self.assertTrue(created["custom"])
        self.assertIn("custom_after_corrupt", json.loads(self.store.read_text(encoding="utf-8")))

    def test_write_failure_preserves_existing_custom_templates(self) -> None:
        style_templates.create_custom_style_template("custom_keep", "保留", "keywords")
        original = self.store.read_text(encoding="utf-8")

        with patch.object(atomic_json.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                style_templates.create_custom_style_template("custom_lost", "丢失", "keywords")

        self.assertEqual(self.store.read_text(encoding="utf-8"), original)
        self.assertEqual(_temp_files(self.store.parent), [])


if __name__ == "__main__":
    unittest.main()
