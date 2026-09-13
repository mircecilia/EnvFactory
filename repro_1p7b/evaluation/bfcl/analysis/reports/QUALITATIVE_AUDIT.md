# Qualitative Paired Audit

## Sampling

A deterministic sample was drawn with seed 20260913 from the paired outcome sets:

- 5 FT cases: Base wrong, SFT correct
- 5 TF cases: Base correct, SFT wrong
- 5 FF cases: both wrong

The audit read each real query, offered tools, complete Base/SFT trajectory, official possible answer, and scorer reason. This is an observation audit, not a failure taxonomy.

## Sample IDs

### FT

- multi_turn_base_12
- multi_turn_miss_func_169
- multi_turn_miss_param_164
- multi_turn_base_136
- multi_turn_base_166

### TF

- multi_turn_long_context_32
- multi_turn_long_context_64
- multi_turn_miss_func_102
- multi_turn_long_context_182
- multi_turn_base_127

### FF

- multi_turn_miss_func_148
- multi_turn_miss_func_183
- multi_turn_long_context_157
- multi_turn_miss_param_16
- multi_turn_base_81

## FT observations

- multi_turn_base_12: Base repeatedly alternated touch, echo, and rm and left summary.txt empty, producing instance_state_mismatch. SFT still used extra inspection/writes, but ended with the expected file content and word count.
- multi_turn_miss_func_169: Base first tried an invalid city-derived airport code and later booked after resolving RMS, but omitted retrieve_invoice and repeated support calls. SFT explicitly resolved the airport, booked, retrieved the invoice, and completed support; the scorer accepted it.
- multi_turn_miss_param_164: Base's first travel-cost call used unresolved location information and missed the expected response. SFT resolved the airport code before the cost call and completed exchange, budget, booking, and invoice operations.
- multi_turn_base_136: Base placed the ZETA order at 150.25 instead of the returned 150.75 and failed state comparison. SFT first resolved the symbol, used the returned current price, and completed order/status/cancel/account operations, though it emitted some duplicate later calls.
- multi_turn_base_166: Base made three booking calls and mutated balance/booking state beyond ground truth. SFT completed a single accepted booking and the subsequent invoice/support sequence, although support calls were repeated.

These fixes are substantive trajectory/parameter/state improvements, not merely XML formatting changes.

## TF observations

- multi_turn_long_context_32: Base completed file read/count plus logarithm and result-file update. SFT stopped after the logarithm/echo sequence without the required final state, yielding instance_state_mismatch.
- multi_turn_long_context_64: Base completed the lock, brake, engine-start, pressure, and tire-shop sequence. SFT called displayCarStatus and omitted required brake/start/shop actions, yielding instance_state_mismatch.
- multi_turn_miss_func_102: Base completed market update, order lifecycle, account lookup, and support ticket retrieval. SFT repeated create_ticket and then produced an empty-turn failure.
- multi_turn_long_context_182: Both used the expected high-level tool names, but SFT's execution responses did not match the expected budget state. The stored score evidence does not justify a more specific causal claim.
- multi_turn_base_127: Base followed watchlist removal with the requested BDX lookup, purchase, and order status. SFT repeated watchlist operations and diverted into order history/details without placing the requested order.

Regressions are therefore also substantive: missing actions, wrong state transitions, repeated calls, and empty turns occur in the sampled set.

## FF observations

- multi_turn_miss_func_148: Both trajectories miss essential trade lifecycle steps and end with empty-turn behavior; Base also emits a tool-call object without a name.
- multi_turn_miss_func_183: Both verify traveler data and start travel planning, but neither reaches the exact accepted booking/invoice/social sequence; both fail final state.
- multi_turn_long_context_157: Base stops after list_all_airports. SFT progresses through nearest-airport and cost calls but still mismatches the expected execution.
- multi_turn_miss_param_16: Both fail to preserve the requested filesystem path/state. Base performs a short wrong-path sequence; SFT repeatedly searches, creates directories, and copies without reaching the required renamed backup.
- multi_turn_base_81: Both begin with fuel conversion but omit or misorder essential vehicle-state actions and fail instance-state comparison.

The FF sample shows genuine planning, parameter, and environment-state failures rather than a common parser outage.

## Overall observation

SFT sometimes behaves more like a trained tool user: the FT sample contains better prerequisite lookup, use of returned values, and completion of dependent actions. But that behavior is not stable. The TF and FF samples show repeated calls, omitted steps, empty turns, and state drift, while aggregate analysis shows SFT makes substantially more calls and reaches generation limits more often.

None of the 15 sampled outcomes was explained solely by a benchmark-side JSON/XML parser accident. The qualitative evidence supports a case-specific behavioral improvement, not a broad or statistically established gain.
