# Static Graph-Frontier capability comparison

| Metric | Base | Original SFT | Parameter-Aware | SFT-Base | PA-SFT |
|---|---:|---:|---:|---:|---:|
| Task success | 0.0% (0/100; smooth 1.0%) | 29.0% (29/100; smooth 29.4%) | 20.0% (20/100; smooth 20.6%) | +0.290 | -0.090 |
| Final-state success | 29.0% (29/100; smooth 29.4%) | 41.0% (41/100; smooth 41.2%) | 40.0% (40/100; smooth 40.2%) | +0.120 | -0.010 |
| Edge success | 2.8% (6/213; smooth 3.3%) | 45.5% (97/213; smooth 45.6%) | 17.8% (38/213; smooth 18.1%) | +0.427 | -0.277 |
| Internal param propagation | 11.3% (6/53; smooth 12.7%) | 61.0% (97/159; smooth 60.9%) | 25.5% (38/149; smooth 25.8%) | +0.497 | -0.355 |
| Depth-1 success | 4.8% (6/124; smooth 5.6%) | 57.3% (71/124; smooth 57.1%) | 30.6% (38/124; smooth 31.0%) | +0.524 | -0.266 |
| Depth-2 success | 0.0% (0/77; smooth 1.3%) | 26.0% (20/77; smooth 26.6%) | 0.0% (0/77; smooth 1.3%) | +0.260 | -0.260 |
| Depth-3+ success | 0.0% (0/12; smooth 7.1%) | 50.0% (6/12; smooth 50.0%) | 0.0% (0/12; smooth 7.1%) | +0.500 | -0.500 |
| Wrong propagated value | 19 | 32 | 50 | +13 | +18 |
| Redundant calls | 42 | 15 | 202 | -27 | +187 |

## Paired exact McNemar

{
  "base_vs_original_sft": {
    "edge_complete": {
      "discordant": 51,
      "exact_two_sided_p": 1.8323628836469652e-08,
      "paired_support": 100,
      "right_losses": 6,
      "right_wins": 45
    },
    "internal_param_complete": {
      "discordant": 51,
      "exact_two_sided_p": 1.8323628836469652e-08,
      "paired_support": 100,
      "right_losses": 6,
      "right_wins": 45
    },
    "task_success": {
      "discordant": 29,
      "exact_two_sided_p": 3.725290298461914e-09,
      "paired_support": 100,
      "right_losses": 0,
      "right_wins": 29
    }
  },
  "base_vs_parameter_aware": {
    "edge_complete": {
      "discordant": 24,
      "exact_two_sided_p": 0.001543879508972168,
      "paired_support": 100,
      "right_losses": 4,
      "right_wins": 20
    },
    "internal_param_complete": {
      "discordant": 24,
      "exact_two_sided_p": 0.001543879508972168,
      "paired_support": 100,
      "right_losses": 4,
      "right_wins": 20
    },
    "task_success": {
      "discordant": 20,
      "exact_two_sided_p": 1.9073486328125e-06,
      "paired_support": 100,
      "right_losses": 0,
      "right_wins": 20
    }
  },
  "original_sft_vs_parameter_aware": {
    "edge_complete": {
      "discordant": 27,
      "exact_two_sided_p": 5.647540092468262e-06,
      "paired_support": 100,
      "right_losses": 25,
      "right_wins": 2
    },
    "internal_param_complete": {
      "discordant": 27,
      "exact_two_sided_p": 5.647540092468262e-06,
      "paired_support": 100,
      "right_losses": 25,
      "right_wins": 2
    },
    "task_success": {
      "discordant": 9,
      "exact_two_sided_p": 0.00390625,
      "paired_support": 100,
      "right_losses": 9,
      "right_wins": 0
    }
  }
}

## Failure distributions

### Base

- Edge status: {'producer_not_executed': 159, 'success': 6, 'wrong_propagated_value': 19, 'dependency_failure': 9, 'consumer_not_executed': 10, 'producer_failed': 10}
- Root failure: {'final_state_failure': 51, 'wrong_propagated_value': 19, 'producer_not_executed': 24, 'missing_producer': 6}
- First failed depth: {'1': 94}
- System errors: {}

### Original SFT

- Edge status: {'producer_not_executed': 37, 'consumer_not_executed': 30, 'success': 97, 'wrong_propagated_value': 32, 'producer_failed': 13, 'dependency_failure': 4}
- Root failure: {'missing_producer': 14, 'final_state_failure': 24, 'wrong_propagated_value': 26, 'producer_not_executed': 6, 'consumer_not_executed': 1}
- First failed depth: {'1': 52, '2': 3}
- System errors: {}

### Parameter-Aware

- Edge status: {'producer_not_executed': 31, 'dependency_failure': 15, 'wrong_propagated_value': 50, 'consumer_not_executed': 43, 'producer_failed': 36, 'success': 38}
- Root failure: {'missing_producer': 14, 'wrong_propagated_value': 44, 'final_state_failure': 6, 'producer_not_executed': 6, 'consumer_not_executed': 10}
- First failed depth: {'1': 76, '2': 2}
- System errors: {}

## Limitations

This is a 100-task signal-validation pilot. Edges within one task are clustered; no edge-level significance claim is made.
