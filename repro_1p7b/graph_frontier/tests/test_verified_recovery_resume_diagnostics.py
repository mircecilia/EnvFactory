import unittest
from pathlib import Path

from repro_1p7b.graph_frontier.verified_recovery_resume_diagnostics import first_difference, observe_collect


class ResumeDiagnosticsTest(unittest.TestCase):
    def test_strict_scalar_and_order(self):
        self.assertIsNone(first_difference({'a': [1, 'x']}, {'a': [1, 'x']}))
        self.assertEqual(first_difference(1, True)['reason'], 'TYPE')
        self.assertEqual(first_difference([1, 2], [2, 1])['path'], '$[0]')

    def test_exact_exception_difference_retained(self):
        a = [{'exception': {'message': 'address 0xabc'}}]
        b = [{'exception': {'message': 'address 0xdef'}}]
        diff = first_difference(a, b)
        self.assertEqual(diff['path'], '$[0].exception.message')
        self.assertEqual(diff['left'], 'address 0xabc')
        self.assertEqual(diff['right'], 'address 0xdef')

    def test_observer_changes_no_result(self):
        result = {'steps': [], 'environment_calls': 11}
        writes = []
        async def original(*args):
            return result
        observed = observe_collect(original, Path('not_written'), lambda p, v: writes.append((p, v)), 'hash')
        # Both coroutines do no I/O; drive directly without platform loop policy.
        coroutine = observed('task', {}, {}, {}, [], None, {}, 'FD')
        with self.assertRaises(StopIteration) as completed:
            coroutine.send(None)
        actual = completed.exception.value
        self.assertIs(actual, result)
        self.assertEqual(len(writes), 1)
        self.assertIs(writes[0][1]['result'], result)
        self.assertEqual(actual['environment_calls'], 11)


if __name__ == '__main__':
    unittest.main()
