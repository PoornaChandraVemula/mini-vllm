"""CPU regressions for paged KV bookkeeping and scheduling.

Small four-token pages make boundary cases readable. These tests do not load
model weights or claim to exercise CUDA kernels; GPU generation is a separate
integration check.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from minivllm.engine.block_manager import BlockManager
from minivllm.engine.scheduler import Scheduler
from minivllm.engine.sequence import Sequence, SequenceStatus
from minivllm.sampling_params import SamplingParams


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        block_size = patch.object(Sequence, "block_size", 4)
        block_size.start()
        self.addCleanup(block_size.stop)

    def sequence(self, tokens, **sampling):
        return Sequence(list(tokens), SamplingParams(**sampling))

    def scheduler(self, **overrides):
        values = dict(
            max_num_seqs=4,
            max_num_batched_tokens=32,
            max_model_len=32,
            eos=(1, 107),
            kvcache_block_size=4,
            num_kvcache_blocks=16,
        )
        values.update(overrides)
        return Scheduler(SimpleNamespace(**values))

    def cache_prompt(self, manager, seq):
        cached = manager.can_allocate(seq)
        self.assertGreaterEqual(cached, 0)
        manager.allocate(seq, cached)
        seq.num_scheduled_tokens = len(seq) - seq.num_cached_tokens
        manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0

    def assert_cache_consistent(self, manager, active_sequences=()):
        free = list(manager.free_block_ids)
        used = manager.used_block_ids
        self.assertEqual(len(free), len(set(free)))
        self.assertFalse(set(free) & used)
        self.assertEqual(set(free) | used, set(range(len(manager.blocks))))
        expected_refs = [0] * len(manager.blocks)
        for seq in active_sequences:
            for block_id in seq.block_table:
                expected_refs[block_id] += 1
        for block in manager.blocks:
            self.assertEqual(block.ref_count, expected_refs[block.block_id])
            self.assertEqual(block.block_id in used, block.ref_count > 0)


class BlockManagerTests(EngineTestCase):
    def test_shared_prefix_reference_counts_and_release(self):
        manager = BlockManager(6, 4)
        first = self.sequence(range(10, 19))
        second = self.sequence([*range(10, 18), 99])
        self.cache_prompt(manager, first)

        self.assertEqual(manager.can_allocate(second), 2)
        manager.allocate(second, 2)
        self.assertEqual(second.num_cached_tokens, 8)
        self.assertEqual(first.block_table[:2], second.block_table[:2])
        self.assertNotEqual(first.block_table[-1], second.block_table[-1])
        self.assert_cache_consistent(manager, [first, second])

        manager.deallocate(first)
        self.assertEqual(first.block_table, [])
        self.assertEqual(first.num_cached_tokens, 0)
        self.assert_cache_consistent(manager, [second])
        manager.deallocate(second)
        self.assert_cache_consistent(manager)

    def test_released_prefix_is_reused_before_it_is_evicted(self):
        manager = BlockManager(4, 4)
        first = self.sequence(range(10, 15))
        self.cache_prompt(manager, first)
        prefix_block = first.block_table[0]
        manager.deallocate(first)

        second = self.sequence([10, 11, 12, 13, 88])
        self.assertEqual(manager.can_allocate(second), 1)
        manager.allocate(second, 1)
        self.assertEqual(second.block_table[0], prefix_block)
        self.assertEqual(manager.blocks[prefix_block].ref_count, 1)
        self.assert_cache_consistent(manager, [second])

    def test_eviction_removes_stale_hash_mapping(self):
        manager = BlockManager(2, 4)
        first = self.sequence(range(10, 15))
        self.cache_prompt(manager, first)
        old_hash = manager.blocks[first.block_table[0]].hash
        manager.deallocate(first)

        replacement = self.sequence(range(20, 25))
        self.cache_prompt(manager, replacement)
        self.assertNotIn(old_hash, manager.hash_to_block_id)
        manager.deallocate(replacement)
        self.assertEqual(manager.can_allocate(first), 0)
        self.assert_cache_consistent(manager)

    def test_hash_depends_on_preceding_tokens(self):
        manager = BlockManager(6, 4)
        first = self.sequence([10, 11, 12, 13, 20, 21, 22, 23, 30])
        second = self.sequence([40, 41, 42, 43, 20, 21, 22, 23, 30])
        self.cache_prompt(manager, first)
        self.assertEqual(manager.can_allocate(second), 0)
        self.cache_prompt(manager, second)
        self.assertNotEqual(
            manager.blocks[first.block_table[1]].hash,
            manager.blocks[second.block_table[1]].hash,
        )
        self.assert_cache_consistent(manager, [first, second])

    def test_identical_prompt_still_computes_last_block_for_logits(self):
        manager = BlockManager(4, 4)
        first = self.sequence(range(10, 18))
        self.cache_prompt(manager, first)
        second = self.sequence(first.token_ids)
        self.assertEqual(manager.can_allocate(second), 1)
        manager.allocate(second, 1)
        self.assertEqual(len(second) - second.num_cached_tokens, 4)
        self.assert_cache_consistent(manager, [first, second])

    def test_uncomputed_prompt_blocks_cannot_be_shared(self):
        manager = BlockManager(6, 4)
        first = self.sequence(range(10, 19))
        manager.allocate(first, 0)
        first.num_scheduled_tokens = 3
        manager.hash_blocks(first)
        first.num_cached_tokens = 3
        second = self.sequence(first.token_ids)
        self.assertEqual(manager.can_allocate(second), 0)

        first.num_scheduled_tokens = 3
        manager.hash_blocks(first)
        first.num_cached_tokens = 6
        self.assertEqual(manager.can_allocate(second), 1)
        self.assertEqual(manager.blocks[first.block_table[1]].hash, -1)


class SchedulerTests(EngineTestCase):
    def test_chunked_prefill_appends_only_after_final_chunk(self):
        scheduler = self.scheduler(max_num_batched_tokens=3)
        seq = self.sequence(range(10, 19), max_tokens=3)
        original = seq.token_ids.copy()
        scheduler.add(seq)

        for expected_cached in (3, 6):
            batch, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(batch, [seq])
            self.assertEqual(seq.num_scheduled_tokens, 3)
            scheduler.postprocess(batch, [107], is_prefill)
            self.assertEqual(seq.token_ids, original)
            self.assertEqual(seq.num_cached_tokens, expected_cached)
            self.assertEqual(seq.num_scheduled_tokens, 0)
            self.assertEqual(seq.status, SequenceStatus.WAITING)

        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [42], is_prefill)
        self.assertEqual(seq.token_ids, original + [42])
        self.assertEqual(seq.num_cached_tokens, 9)
        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertFalse(scheduler.waiting)
        self.assert_cache_consistent(scheduler.block_manager, [seq])

    def test_decode_allocates_page_when_first_token_crosses_boundary(self):
        scheduler = self.scheduler()
        seq = self.sequence([10, 11, 12, 13], max_tokens=3)
        scheduler.add(seq)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [20], is_prefill)
        self.assertEqual(len(seq), 5)
        self.assertEqual(len(seq.block_table), 1)

        batch, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(len(seq.block_table), 2)
        self.assertEqual(seq.num_scheduled_tokens, 1)
        scheduler.postprocess(batch, [21], is_prefill)
        self.assertEqual(seq.num_cached_tokens, 5)
        self.assertEqual(seq.completion_token_ids, [20, 21])
        self.assert_cache_consistent(scheduler.block_manager, [seq])

    def test_memory_pressure_preempts_then_recomputes_without_losing_output(self):
        scheduler = self.scheduler(num_kvcache_blocks=2, max_model_len=8)
        first = self.sequence([10, 11, 12, 13], max_tokens=2)
        second = self.sequence([20, 21, 22, 23], max_tokens=3)
        scheduler.add(first)
        scheduler.add(second)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [30, 40], is_prefill)

        batch, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(batch, [first])
        self.assertEqual(second.status, SequenceStatus.WAITING)
        self.assertTrue(second.is_prefill)
        self.assertEqual(second.block_table, [])
        self.assertEqual(second.num_cached_tokens, 0)
        self.assertEqual(second.completion_token_ids, [40])
        scheduler.postprocess(batch, [31], is_prefill)
        self.assertTrue(first.is_finished)

        batch, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(batch, [second])
        self.assertEqual(second.num_scheduled_tokens, 5)
        scheduler.postprocess(batch, [41], is_prefill)
        self.assertEqual(second.prompt_token_ids, [20, 21, 22, 23])
        self.assertEqual(second.completion_token_ids, [40, 41])
        self.assert_cache_consistent(scheduler.block_manager, [second])

        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [42], is_prefill)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(second.completion_token_ids, [40, 41, 42])
        self.assert_cache_consistent(scheduler.block_manager)

    def test_each_stop_token_finishes_and_releases_cache(self):
        for stop_token in (1, 107):
            with self.subTest(stop_token=stop_token):
                scheduler = self.scheduler()
                seq = self.sequence([10, 11], max_tokens=5)
                scheduler.add(seq)
                batch, is_prefill = scheduler.schedule()
                scheduler.postprocess(batch, [stop_token], is_prefill)
                self.assertTrue(seq.is_finished)
                self.assertEqual(seq.completion_token_ids, [stop_token])
                self.assertTrue(scheduler.is_finished())
                self.assert_cache_consistent(scheduler.block_manager)

    def test_scalar_eos_is_supported(self):
        scheduler = self.scheduler(eos=1)
        seq = self.sequence([10, 11], max_tokens=5)
        scheduler.add(seq)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [1], is_prefill)
        self.assertTrue(seq.is_finished)

    def test_decode_batch_respects_token_budget(self):
        scheduler = self.scheduler(max_num_batched_tokens=2, max_num_seqs=4)
        sequences = [self.sequence([10 + i], max_tokens=4) for i in range(4)]
        for seq in sequences:
            scheduler.add(seq)
        for token_ids in ([20, 21], [22, 23]):
            batch, is_prefill = scheduler.schedule()
            self.assertTrue(is_prefill)
            self.assertEqual(len(batch), 2)
            scheduler.postprocess(batch, token_ids, is_prefill)

        self.assertEqual(len(scheduler.running), 4)
        batch, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(len(batch), 2)
        self.assertEqual(sum(seq.num_scheduled_tokens for seq in batch), 2)
        scheduler.postprocess(batch, [30, 31], is_prefill)
        self.assert_cache_consistent(scheduler.block_manager, sequences)

    def test_self_preemption_reports_impossible_cache_capacity(self):
        # A normal runner clamps context length to cache capacity. Exercise the
        # scheduler's defensive path with an inconsistent standalone config.
        scheduler = self.scheduler(num_kvcache_blocks=1, max_model_len=12)
        seq = self.sequence([10, 11, 12, 13], max_tokens=5)
        scheduler.add(seq)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [20], is_prefill)

        with self.assertRaisesRegex(RuntimeError, "KV cache capacity"):
            scheduler.schedule()
        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assertEqual(seq.completion_token_ids, [20])
        self.assertEqual(seq.block_table, [])
        self.assert_cache_consistent(scheduler.block_manager)

    def test_ignore_eos_still_obeys_output_limit(self):
        scheduler = self.scheduler()
        seq = self.sequence([10, 11], max_tokens=2, ignore_eos=True)
        scheduler.add(seq)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [107], is_prefill)
        self.assertFalse(seq.is_finished)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [1], is_prefill)
        self.assertTrue(seq.is_finished)
        self.assertEqual(seq.completion_token_ids, [107, 1])
        self.assert_cache_consistent(scheduler.block_manager)

    def test_context_limit_stops_at_last_legal_output(self):
        scheduler = self.scheduler(max_model_len=4)
        seq = self.sequence([10, 11], max_tokens=20, ignore_eos=True)
        scheduler.add(seq)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [20], is_prefill)
        self.assertFalse(seq.is_finished)
        batch, is_prefill = scheduler.schedule()
        scheduler.postprocess(batch, [21], is_prefill)
        self.assertTrue(seq.is_finished)
        self.assertEqual(len(seq), 4)
        self.assertEqual(seq.completion_token_ids, [20, 21])
        self.assertTrue(scheduler.is_finished())
        self.assert_cache_consistent(scheduler.block_manager)


if __name__ == "__main__":
    unittest.main()
