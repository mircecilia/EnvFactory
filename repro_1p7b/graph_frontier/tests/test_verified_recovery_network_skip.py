import unittest
from repro_1p7b.graph_frontier.verified_recovery_network_skip import is_network_error, network_events, failure_is_network


class NetworkSkipTest(unittest.TestCase):
    def test_real_network_error_not_validation_or_http404(self):
        self.assertTrue(is_network_error({'type': 'ToolError', 'message': 'NameResolutionError: Failed to resolve host'}))
        self.assertTrue(is_network_error({'type': 'ToolError', 'message': '[Errno 101] Network is unreachable'}))
        self.assertFalse(is_network_error({'type': 'ToolError', 'message': '404 Client Error'}))
        self.assertFalse(is_network_error({'type': 'ValueError', 'message': 'unknown entity id'}))
        self.assertFalse(is_network_error(None))

    def test_no_model_or_state_drift_excused(self):
        records = [{'success': False, 'exception': {'message': 'Network is unreachable'}, 'role': 'STUDENT'}]
        self.assertTrue(failure_is_network(AssertionError('paired response drift'), records))
        self.assertFalse(failure_is_network(AssertionError('paired greedy output drift'), records))
        self.assertFalse(failure_is_network(AssertionError('paired state drift'), records))
        self.assertFalse(failure_is_network(RuntimeError('independent same-prestate gold target validation failed'), records))
        records[0]['role'] = 'TARGET_VALIDATION'
        self.assertTrue(failure_is_network(RuntimeError('independent same-prestate gold target validation failed'), records))

    def test_query_or_success_text_does_not_trigger_skip(self):
        episode = {'arms': {'FD': {'steps': [{'typed_events': [
            {'tool_name': 'x', 'execution_success': True, 'exception': None, 'tool_response': 'Network is unreachable'}
        ]}], 'targets': []}}}
        self.assertEqual(network_events(episode), [])
        episode['arms']['FD']['steps'][0]['typed_events'][0].update(execution_success=False, exception={'message': 'Network is unreachable'})
        self.assertEqual(len(network_events(episode)), 1)


if __name__ == '__main__': unittest.main()
