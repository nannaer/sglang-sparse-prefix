import json
import tempfile
import unittest
from pathlib import Path

from benchmark.agentic_replay.replay_harbor_multi_agent import (
    build_cache_warmup,
    load_trace,
    rewrite_replayed_outputs,
    shape_recorded_cache_prefix,
)


def make_row(
    turn: int,
    request_id: str,
    input_ids: list[int],
    output_ids: list[int],
    started_at: float,
    cached_tokens: int,
) -> dict:
    return {
        "trace_format": "harbor_model_relay_turn_v2",
        "traj_id": "traj-a",
        "turn": turn,
        "request_id": request_id,
        "status_code": 200,
        "request_started_at": started_at,
        "request_finished_at": started_at + 1,
        "input_tokens": len(input_ids),
        "output_tokens": len(output_ids),
        "recorded_cached_tokens": cached_tokens,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "token_ids_complete": True,
    }


class HarborMultiAgentReplayTest(unittest.TestCase):
    def test_load_trace_groups_global_turns_by_agent(self) -> None:
        rows = [
            make_row(0, "main-0", [1], [2], 10, 0),
            make_row(1, "child-0", [7], [8], 12, 0),
            make_row(2, "main-1", [1, 2], [3], 13, 1),
        ]
        calls = [
            {"turn": 0, "request_id": "main-0", "agent_id": "main"},
            {"turn": 1, "request_id": "child-0", "agent_id": "child"},
            {"turn": 2, "request_id": "main-1", "agent_id": "main"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir)
            (trace_dir / "model_relay_trace.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (trace_dir / "multi_agent_trace.json").write_text(
                json.dumps({"model_calls": calls}), encoding="utf-8"
            )

            trace = load_trace(trace_dir)

        self.assertEqual(
            [(agent.agent_id, len(agent.rows)) for agent in trace.agents],
            [("main", 2), ("child", 1)],
        )
        self.assertEqual(trace.agents[1].start_offset_seconds, 2)

    def test_rewrite_uses_only_same_agent_history(self) -> None:
        rows = [
            make_row(0, "main-0", [1, 2], [3, 4], 0, 0),
            make_row(2, "main-1", [1, 2, 3, 4, 9], [5], 1, 4),
        ]

        replay_input, spans = rewrite_replayed_outputs(rows, 1, [[30, 40]])

        self.assertEqual(replay_input, [1, 2, 30, 40, 9])
        self.assertEqual(spans[0]["source_global_turn"], 0)

    def test_cache_shape_breaks_at_recorded_boundary(self) -> None:
        row = make_row(1, "main-1", [1, 2, 3, 4, 5], [6], 1, 3)

        shaped, adjustments = shape_recorded_cache_prefix(
            row["input_ids"], row, [1, 2, 3, 4, 9]
        )

        self.assertEqual(shaped[:3], [1, 2, 3])
        self.assertNotEqual(shaped[3], 4)
        self.assertEqual(adjustments, 1)

    def test_cache_shape_avoids_every_older_matching_branch(self) -> None:
        row = make_row(3, "main-3", [1, 2, 8, 9, 10], [11], 3, 2)
        previous = [1, 2, 3, 4, 5]
        older_branches = [
            [1, 2, 8, 20],
            [1, 2, 6, 30],
        ]

        shaped, adjustments = shape_recorded_cache_prefix(
            row["input_ids"],
            row,
            previous,
            cached_branches=older_branches,
        )

        self.assertEqual(shaped[:2], [1, 2])
        self.assertNotIn(shaped[2], {3, 8, 6})
        self.assertEqual(adjustments, 1)

    def test_first_request_warmup_has_exact_branch_boundary(self) -> None:
        row = make_row(0, "main-0", [10, 11, 12, 13], [14], 0, 3)

        warmup = build_cache_warmup(row)

        self.assertEqual(warmup[:3], [10, 11, 12])
        self.assertNotEqual(warmup[3], 13)


if __name__ == "__main__":
    unittest.main()
