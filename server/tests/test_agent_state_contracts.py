"""统一 Agent 状态机、结构化契约和检查点生命周期验收。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel, ValidationError

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from agent.checkpoints import CheckpointStore, fingerprint  # noqa: E402
from agent.contracts import (  # noqa: E402
    QUALITY_STRATEGIES,
    STAGE_CONTRACTS,
    CheckpointRecord,
    CritiqueReport,
    DecisionTrace,
    FailureKind,
    HumanInterventionPolicy,
    QualityProfileName,
    QualityStrategy,
    RecoveryCandidate,
    RecoveryStrategy,
    ShotArtifact,
    StageContract,
    StageInput,
    StageName,
    StageOutput,
    StageStatus,
    ensure_stage_status_transition,
    ensure_stage_transition,
    stage_status_transition_allowed,
    stage_transition_allowed,
)
from agent.state import ExecutionIdentity  # noqa: E402


class StructuredContractTests(unittest.TestCase):
    def test_required_structured_objects_and_stage_contracts_are_complete(self) -> None:
        structured_objects = (
            StageName,
            StageStatus,
            FailureKind,
            RecoveryStrategy,
            StageContract,
            ShotArtifact,
            CritiqueReport,
            DecisionTrace,
            RecoveryCandidate,
            CheckpointRecord,
            QualityStrategy,
        )
        self.assertEqual(len(structured_objects), 11)
        self.assertEqual(set(STAGE_CONTRACTS), set(StageName))
        for contract in STAGE_CONTRACTS.values():
            self.assertTrue(issubclass(contract.input_model, BaseModel))
            self.assertTrue(issubclass(contract.output_model, BaseModel))
            self.assertTrue(contract.quality_metrics)
            self.assertTrue(contract.failure_classes)
            self.assertTrue(contract.recoverable)
            self.assertTrue(contract.checkpoint_key)
            self.assertTrue(contract.allowed_recovery)
            self.assertLessEqual(set(contract.allowed_recovery), set(RecoveryStrategy))

    def test_automatic_quality_strategies_disable_human_intervention(self) -> None:
        self.assertEqual(set(QUALITY_STRATEGIES), set(QualityProfileName))
        required_fields = {
            "candidate_count",
            "max_recovery_attempts",
            "quality_threshold",
            "cost_multiplier",
            "expected_duration",
            "publish_policy",
            "human_intervention",
        }
        for name, strategy in QUALITY_STRATEGIES.items():
            self.assertEqual(strategy.name.value, name.value)
            self.assertEqual(strategy.human_intervention, HumanInterventionPolicy.DISABLED)
            self.assertGreater(strategy.expected_duration, 0)
            self.assertTrue(strategy.publish_policy.value)
            self.assertTrue(required_fields.issubset(set(QualityStrategy.model_fields)))
        self.assertEqual({item.value for item in QualityProfileName}, {"draft", "standard", "finishing"})

        invalid = QUALITY_STRATEGIES[QualityProfileName.DRAFT].model_dump(mode="json")
        invalid["human_intervention"] = HumanInterventionPolicy.REQUIRED.value
        with self.assertRaises(ValidationError):
            QualityStrategy.model_validate(invalid)

    def test_execution_identity_requires_stable_state_fields(self) -> None:
        self.assertEqual(
            set(ExecutionIdentity.__required_keys__),
            {"project_id", "shot_version", "run_id", "input_fingerprint"},
        )
        self.assertIn("shot_version", StageInput.model_fields)
        self.assertIn("shot_version", StageOutput.model_fields)
        self.assertIn("shot_version", CheckpointRecord.model_fields)
        self.assertIn("shot_version", ShotArtifact.model_fields)


class StateTransitionTests(unittest.TestCase):
    def test_stage_machine_rejects_illegal_jump_and_terminal_reopen(self) -> None:
        self.assertTrue(stage_transition_allowed(StageName.DIRECTOR_PLANNING, StageName.STORYBOARD_DESIGN))
        self.assertFalse(stage_transition_allowed(StageName.DIRECTOR_PLANNING, StageName.VIDEO_GENERATION))
        self.assertFalse(stage_transition_allowed(StageName.FINAL_REVIEW, StageName.IMAGE_GENERATION))
        self.assertTrue(
            stage_transition_allowed(
                StageName.QUALITY_REVIEW,
                StageName.IMAGE_GENERATION,
                recovery=True,
            )
        )
        with self.assertRaises(ValueError):
            ensure_stage_transition(StageName.DIRECTOR_PLANNING, StageName.VIDEO_GENERATION)

    def test_stage_status_machine_rejects_completed_restart(self) -> None:
        self.assertTrue(stage_status_transition_allowed(StageStatus.PENDING, StageStatus.RUNNING))
        self.assertTrue(stage_status_transition_allowed(StageStatus.RUNNING, StageStatus.RECOVERING))
        self.assertFalse(stage_status_transition_allowed(StageStatus.SUCCEEDED, StageStatus.RUNNING))
        with self.assertRaises(ValueError):
            ensure_stage_status_transition(StageStatus.FAILED, StageStatus.SUCCEEDED)


class CheckpointLifecycleTests(unittest.TestCase):
    def test_checkpoint_save_read_reuse_and_invalidate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("checkpoint-lifecycle", "run-1", root=Path(root))
            record = CheckpointRecord(
                key="image",
                checkpoint_key="image",
                kind="stage",
                project_id="checkpoint-lifecycle",
                shot_version=1,
                run_id="run-1",
                stage=StageName.IMAGE_GENERATION,
                status=StageStatus.SUCCEEDED,
                input_fingerprint="input-1",
                output_fingerprint="output-1",
                payload={"path": "image.png"},
            )
            saved = store.save_record(record)
            self.assertEqual(saved["key"], "image")
            self.assertEqual(store.read_record("image").input_fingerprint, "input-1")
            self.assertIsNotNone(store.reuse_record("image", input_fingerprint="input-1"))
            self.assertTrue(store.invalidate_record("image", reason="test_invalidate"))
            self.assertIsNone(store.reuse_record("image", input_fingerprint="input-1"))

    def test_checkpoint_reuse_and_version_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("version-conflict", "run-1", root=Path(root))
            store.save_shot_artifact(
                "shot-1",
                StageName.IMAGE_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                path="shot.png",
                input_fingerprint="input-1",
                output_fingerprint="output-1",
            )
            self.assertIsNotNone(
                store.reusable_shot_artifact(
                    "shot-1",
                    StageName.IMAGE_GENERATION.value,
                    shot_version=1,
                    input_fingerprint="input-1",
                )
            )
            self.assertIsNone(
                store.reusable_shot_artifact(
                    "shot-1",
                    StageName.IMAGE_GENERATION.value,
                    shot_version=2,
                    input_fingerprint="input-1",
                )
            )
            with self.assertRaises(RuntimeError):
                store.assert_shot_version("shot-1", StageName.IMAGE_GENERATION.value, 2)

    def test_input_fingerprint_change_invalidates_saved_state(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("fingerprint-change", "run-1", root=Path(root))
            store.set_input_fingerprint("input-1")
            store.save_stage(
                StageName.DIRECTOR_PLANNING.value,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="director-output-1",
            )
            store.save_shot_artifact(
                "shot-1",
                StageName.IMAGE_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="image-output-1",
            )
            store.set_input_fingerprint("input-2")
            self.assertEqual(store.stage(StageName.DIRECTOR_PLANNING.value)["status"], "invalidated")
            self.assertEqual(store.shot_artifact("shot-1", StageName.IMAGE_GENERATION.value)["status"], "invalidated")
            self.assertFalse(store.stage_is_reusable(StageName.DIRECTOR_PLANNING.value, "input-1"))

    def test_output_fingerprint_change_invalidates_downstream_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("output-change", "run-1", root=Path(root))
            store.save_stage(
                StageName.IMAGE_GENERATION.value,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="image-output-1",
            )
            store.save_stage(
                StageName.QUALITY_REVIEW.value,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="quality-input-1",
                output_fingerprint="quality-output-1",
            )
            store.save_stage(
                StageName.IMAGE_GENERATION.value,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="image-output-2",
            )
            self.assertTrue(store.stage_is_reusable(StageName.IMAGE_GENERATION.value, "input-1"))
            self.assertEqual(store.stage(StageName.QUALITY_REVIEW.value)["invalidated_reason"], "output_fingerprint_changed")
            self.assertFalse(store.stage_is_reusable(StageName.QUALITY_REVIEW.value, "quality-input-1"))

    def test_shot_version_change_invalidates_related_shot_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store = CheckpointStore("shot-version-change", "run-1", root=Path(root))
            store.save_shot_artifact(
                "shot-1",
                StageName.IMAGE_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="image-output-1",
            )
            store.save_shot_artifact(
                "shot-1",
                StageName.VIDEO_GENERATION.value,
                shot_version=1,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-1",
                output_fingerprint="video-output-1",
            )
            store.save_shot_artifact(
                "shot-1",
                StageName.IMAGE_GENERATION.value,
                shot_version=2,
                status=StageStatus.SUCCEEDED.value,
                input_fingerprint="input-2",
                output_fingerprint="image-output-2",
            )
            self.assertIsNone(store.reusable_shot_artifact("shot-1", StageName.VIDEO_GENERATION.value, shot_version=1))
            self.assertEqual(store.shot_artifact("shot-1", StageName.VIDEO_GENERATION.value)["invalidated_reason"], "shot_version_changed")

    def test_fingerprint_is_stable_for_unordered_sets_and_changes_with_payload(self) -> None:
        self.assertEqual(fingerprint({"values": {1, 2, 3}}), fingerprint({"values": {3, 2, 1}}))
        self.assertNotEqual(fingerprint({"value": "before"}), fingerprint({"value": "after"}))


if __name__ == "__main__":
    unittest.main()
