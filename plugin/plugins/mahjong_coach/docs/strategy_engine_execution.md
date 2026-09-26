# Mahjong Coach strategy engine

## Boundary

The strategy engine consumes only facts reconstructed from screenshots. It does not read Mahjong Soul network traffic, WebSocket frames, process memory, or backend APIs.

策略引擎只消费由截图恢复出的牌局事实，不读取雀魂网络流量、WebSocket、进程内存或后端接口。

## Implemented

### Exact hand efficiency

`strategy/efficiency.py` currently provides:

- standard-hand shanten;
- seven-pairs shanten for closed hands;
- thirteen-orphans shanten for closed hands;
- effective tile types for 3n+1 hands;
- exhaustive discard comparison for 3n+2 hands;
- unseen-copy counts after subtracting the hand, rivers, melds, dora indicators, and other visible tiles;
- source counts and recognition anomalies for diagnostics.

The remaining count is deliberately named `unseen_count`. It is calculated as:

```text
unseen(tile) = max(0, 4 - copies_in_hand - external_visible_copies)
```

It is not the exact number in the wall because opponents' concealed hands are unknown.

### Exact genbutsu

`strategy/risk.py` calculates each riichi player's genbutsu from that player's own river. With multiple riichi players, the universally safe set is the intersection of their genbutsu sets. Missing river data yields no universal-safety claim.

## Runtime flow

```text
screenshot
  -> hand / meld / river / dora-indicator recognition
  -> normalized tile facts
  -> exact shanten and ukeire
  -> exact genbutsu where available
  -> legacy route heuristics as a tie-breaker
  -> overlay and plugin dashboard
```

The plugin dashboard stores the latest structured result in `RoundCoachState.last_efficiency`. The visible card shows shanten, the preferred discard, effective tile types, and unseen copies. Full details remain available in the decision JSON.

## Genbutsu defense / 现物防守

When one or more opponents are confirmed riichi, `strategy/risk.py` compares every held tile with each riichi player's own river. A tile is marked `safe_against_all` only when it is present in every identified threat player's river. Missing river data is reported as unknown and never treated as safe.

对手自己打过的牌属于该对手的现物：舍牌振听使其不能用这类牌荣和。这个结论只防荣和，不阻止对手自摸。`not_proven_safe_against` 只表示“不是已证明的现物”，不等于“一定会放铳”。红五与普通五按同一种等待牌处理。

## TODO

- TODO: recognize and pass dora indicators from the original, unwarped screenshot.
- TODO: aggregate all four players' meld tiles, not only the current self-meld path.
- TODO: retain river order and distinguish each player's river reliably across frames.
- TODO: add round context OCR: scores, dealer, round wind, seat wind, honba, and riichi sticks.
- TODO: add suji and wall features as labeled evidence, separate from exact genbutsu.
- TODO: import or reproduce validated tenpai-rate and deal-in-risk tables.
- TODO: estimate win probability and expected points; these are statistical estimates, not exact rule results.
- TODO: add an attack/defense utility layer using placement and score context.
- TODO: expose confidence and missing-input reasons alongside every statistical estimate.
- TODO: compare the deterministic output against Akagi analysis fixtures before enabling it as the only strategy mode.
