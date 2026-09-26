from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from .models import _valid_live_advice_mode
from .tile_labels import normalize_tile


_PLAYER_LABELS = {
    "self": "自己",
    "left_opponent": "上家",
    "top_opponent": "对家",
    "right_opponent": "下家",
}

_ABSURD_BANTER_CARDS: tuple[tuple[str, str], ...] = (
    ("先别急", "可以说“先别急，我先急一下”，然后立刻回到轻松旁观。"),
    ("合理吗", "可以一本正经地说“这合理吗？这不合理吧……嗯，合理吗？”。"),
    ("节目效果", "可以吐槽“刚才还挺正常的，怎么突然有节目效果了”。"),
    ("大概懂了", "可以说“我大概看懂了。大概。”，保留一点心虚和停顿。"),
    ("很有想法", "可以说“这一下很有想法，具体是什么想法我再想想”。"),
    (
        "反复横跳",
        "可以使用“哦对的对的……哎呀不对不对……对……对吗？”，但整条回复最多完整出现一次。",
    ),
    (
        "卡布奇诺",
        "可以自然地说“先给你倒杯卡布奇诺，这局慢慢看”，像坐在旁边陪着，不要顺带评价手牌或牌运。",
    ),
    ("问题大小", "可以说“问题不大，问题具体大不大另说”，说完就收住。"),
    ("保留意见", "可以说“我先保留意见，主要是意见还没想好”，不要继续解释这个梗。"),
    ("事情有趣", "可以说“事情开始变得有意思了，虽然我还没看懂哪里有意思”。"),
    ("局势解释", "可以说“先让局势自己解释一下，我暂时负责看着”，保持旁观口吻。"),
    ("记小本本", "可以说“这一幕我先记小本本上，等会儿看它怎么圆回来”。"),
)

_ABSURD_RIICHI_CARDS: tuple[tuple[str, str], ...] = (
    ("牌桌加速", "先明确说有人立直，再轻松吐槽“好，牌桌有人按下加速键了”。"),
    ("咖啡放稳", "先明确说有人立直，再说“这杯卡布奇诺先放稳，桌子突然有点紧张”。"),
    ("气氛到位", "先明确说有人立直，再说“立直声一响，气氛一下就到位了”。"),
    ("节目开始", "先明确说有人立直，再吐槽“好，节目从这一声立直正式开始”。"),
)

_ABSURD_MELD_CARDS: tuple[tuple[str, str], ...] = (
    ("动作很快", "先准确说清哪一家吃、碰或杠了什么，再吐槽“这一下动作是真快”。"),
    ("很有想法", "先准确播报副露，再说“这一口很有想法，桌面马上就不一样了”。"),
    ("记小本本", "先准确播报副露，再说“行，这一组我先记小本本上”。"),
    ("节目效果", "先准确播报副露，再轻轻吐槽“桌面突然开始有节目效果了”。"),
)

_ABSURD_BANTER_REPLIES: tuple[str, ...] = (
    "先别急，我先急一下。",
    "这合理吗？这不合理吧……嗯，合理吗？",
    "刚才还挺正常的，怎么突然有节目效果了。",
    "我大概看懂了。大概。",
    "这一下很有想法，具体是什么想法我再想想。",
    "哦对的对的……哎呀不对不对……对……对吗？",
    "先给你倒杯卡布奇诺，这局慢慢看。",
    "问题不大，问题具体大不大另说。",
    "我先保留意见，主要是意见还没想好。",
    "事情开始变得有意思了，虽然我还没看懂哪里有意思。",
    "先让局势自己解释一下，我暂时负责看着。",
    "这一幕我先记小本本上，等会儿看它怎么圆回来。",
)

_ABSURD_RIICHI_REPLIES: tuple[str, ...] = (
    "好，牌桌有人按下加速键了。",
    "这杯卡布奇诺先放稳，桌子突然有点紧张。",
    "立直声一响，气氛一下就到位了。",
    "好，节目从这一声立直正式开始。",
)

_ABSURD_MELD_REPLIES: tuple[str, ...] = (
    "这一下动作是真快。",
    "这一口很有想法，桌面马上就不一样了。",
    "行，这一组我先记小本本上。",
    "桌面突然开始有节目效果了。",
)

_COMPANION_RARE_JOKE_BLOCKED_EVENTS = {
    "win_opportunity",
    "riichi_pressure",
    "opponent_meld",
}
_COMPANION_ABSURD_BLOCKED_EVENTS = {"win_opportunity"}


def build_public_payload(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Create the mode-aware view shared by every external surface.

    The distributed plugin always asks for companion mode. That view carries
    only confirmed table events needed for banter; recommendation fields are
    retained solely in the dormant strategy-library branch for rollback and
    offline tests.
    """
    normalized_mode = _valid_live_advice_mode(mode)
    raw_decision = payload.get("last_decision") if isinstance(payload.get("last_decision"), dict) else payload
    raw_state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else payload.get("coach_state")
    raw_state = raw_state if isinstance(raw_state, dict) else {}
    presentation = _build_presentation(raw_decision, raw_state, normalized_mode)
    public_state = _sanitize_state(raw_state, presentation)
    public_decision = _sanitize_decision(raw_decision, public_state, presentation)
    return {
        "last_decision": public_decision,
        "round_state": public_state,
        "presentation": presentation,
    }


def build_neko_companion_cue(
    payload: dict[str, Any],
    *,
    include_rare_joke: bool = False,
    ambient_event: str = "",
    ambient_signature: str = "",
    opponent_meld_event: dict[str, Any] | None = None,
    absurd_banter_enabled: bool = False,
    absurd_variant: int = 0,
) -> tuple[str, str, str]:
    """Build one hidden N.E.K.O context cue for a meaningful table event.

    The cue contains observations only. It is independent from the panel's
    recommendation mode, is never rendered verbatim, and never includes
    candidate tile identities or operation rankings.
    """
    presentation = payload.get("presentation") if isinstance(payload.get("presentation"), dict) else {}
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else {}
    source = str(presentation.get("source_decision_type") or "")
    action_event = str(presentation.get("action_event") or "")
    riichi_players = [str(item) for item in (state.get("riichi_players") or []) if str(item)]
    if source == "win_window":
        event_kind = "win_opportunity"
        action_label = _action_label(action_event) or "和牌"
        event_label = f"界面出现{action_label}按钮"
        internal_reaction = _win_companion_reaction(action_event, state)
    elif source == "opening_plan":
        event_kind = "opening_observation"
        event_label = "新一局开始了"
        internal_reaction = "新一局开始了；直接和玩家轻松互动或说一个短梗，不要点评起手牌。"
    elif source in {"round_settlement", "settlement_candidate", "awaiting_next_round"}:
        event_kind = "round_boundary"
        event_label = "本局进入结算或等待下一局"
        internal_reaction = "好，这局先到这儿，刚才那阵仗还挺有意思。"
    elif opponent_meld_event:
        event_kind = "opponent_meld"
        event_label, internal_reaction = _opponent_meld_event_summary(opponent_meld_event)
    elif source == "defense_alert" and riichi_players:
        event_kind = "riichi_pressure"
        event_label = "牌桌出现新的立直压力"
        internal_reaction = str(presentation.get("headline") or "牌桌突然紧张起来。")
    elif state.get("last_observed_discard"):
        event_kind = "player_interaction"
        event_label = "玩家刚完成了一次操作"
        internal_reaction = "直接和玩家互动或说一个普通短梗，不要评价这次操作对不对，也不要借机分析手牌。"
    elif source == "coach_checkpoint":
        event_kind = "checkpoint_observation"
        event_label = "牌局仍在继续，当前没有需要立即播报的关键桌面事件"
        internal_reaction = "直接和玩家轻松互动或说一个短梗，不要评论手牌、牌型、牌运或成型前景。"
    elif source == "observe" and ambient_event == "ambient_observation" and riichi_players:
        event_kind = "ambient_observation"
        event_label = "已确认的立直压力仍在持续"
        internal_reaction = "立直压力还在；只挑当前已经确认的一个变化随口说一句，不要重复宣布立直。"
    elif source == "observe" and ambient_event == "idle_heartbeat" and riichi_players:
        event_kind = "idle_heartbeat"
        event_label = "立直后的牌桌安静了一阵"
        internal_reaction = "立直后的场面安静了一会儿；可以轻松评论等待感，但不要重复宣布同一位玩家立直。"
    elif source == "observe" and ambient_event == "casual_chat":
        event_kind = "casual_chat"
        event_label = "当前没有需要立即播报的新牌局事件"
        internal_reaction = "自然地和玩家随口聊一两句轻松内容，不要为了找话题硬分析牌局。"
    else:
        return "", "", ""

    evidence = _companion_speech_evidence(presentation.get("evidence"))
    if event_kind in {
        "opening_observation",
        "checkpoint_observation",
        "player_interaction",
        "casual_chat",
    }:
        evidence = []
    if opponent_meld_event:
        evidence = [*_opponent_meld_event_evidence(opponent_meld_event), *evidence]
        evidence = list(dict.fromkeys(evidence))
    allow_rare_joke = bool(
        include_rare_joke
        and event_kind not in _COMPANION_RARE_JOKE_BLOCKED_EVENTS
    )
    allow_absurd_banter = bool(
        absurd_banter_enabled
        and not allow_rare_joke
        and event_kind not in _COMPANION_ABSURD_BLOCKED_EVENTS
    )
    cue_reaction = (
        "这次只做一个显而易见确认问题的轻松彩蛋，不叠加其他口头禅。"
        if allow_rare_joke
        else "逆天模式已开启；用一个短彩蛋回应当前气氛，不要拼接多个梗。"
        if allow_absurd_banter
        else internal_reaction
    )
    lines = [
        "【雀魂吐槽伙伴事件｜只读观察】",
        f"事件：{event_label}",
        f"角色的即时反应：{cue_reaction}",
    ]
    if evidence:
        lines.extend(["", "已确认或识别到的事实：", *[f"- {item}" for item in evidence[:4]]])
    # Product decision: do not feed hand-quality judgments into companion
    # speech. In live use, the generative layer repeatedly paraphrased both
    # strong and weak hands as "good luck". Strategy surfaces still retain the
    # objective hand analysis; the companion receives event facts and banter
    # instructions only.
    lines.extend(
        [
            "",
            "回复要求：",
            "- 使用当前 N.E.K.O 角色原本的性格、自称和称呼，不要额外添加幼态自称。",
            "- 像坐在玩家旁边看牌时随口说一两句，轻松、口语化，可以有停顿和省略，不要写成分析报告。",
            "- 不要逐条复述事实、风险来源或置信度；从中挑一个最有趣或最明显的点说就够了。",
            "- 不要念出完整手牌、逐张牌名、四家分数或整段牌河；这些只供内部判断。",
            "- 不要播报向听数、有效进张数，或距离普通牌型、役满还差多少张；即使此前对话提到过也不要继续复述。",
            "- 不要评价自己的手牌好坏、牌型强弱、牌运、手气、是否顺、是否僵或和牌前景；普通时刻直接和玩家互动或说一个短梗。",
            "- 当自己没有副露时，不要主动强调“没有副露”“零组副露”或反复说明仍是门清；只有确认发生新的副露动作时才谈副露。",
            "- 这是玩家操作之后的陪伴反应或牌桌气氛评论，不是打牌指导。",
            "- 不得告诉玩家应该打哪张牌，也不得建议立直、吃、碰、杠或弃和。",
            "- 不得输出候选牌、候选排序、风险预算数字或下一步操作。",
            "- 只评论已经确认的牌局动作与牌面；不要评价识别能力、数据完整度，也不要描述自己是否敢判断。",
        ]
    )
    if event_kind == "opponent_meld":
        lines.extend(
            [
                "- 必须明确说出是哪一家完成了什么副露；如果是碰，必须说出碰的是哪张牌。",
                "- 可以顺带概括该玩家当前已经公开的全部副露，但不要把它改写成打牌建议。",
            ]
        )
    if event_kind == "win_opportunity":
        lines.extend(
            [
                "- 立刻对刚出现的和牌机会作出自然反应；保持高兴、意外或松一口气的情绪即可。",
                "- 不要固定复读“总算没白等”或“可算等到了”；围绕本次即时反应自然换一种说法。",
                "- 只庆祝或吐槽已出现的和牌窗口，不要指导玩家点击按钮或继续操作。",
            ]
        )
    if event_kind == "casual_chat":
        lines.extend(
            [
                "- 这次直接进行自然的普通聊天；可以聊当下感受、轻松日常或陪伴感，不必谈麻将。",
                "- 不要说“没有事件”“没什么好说”“等牌局变化”之类的系统状态，也不要编造牌局动作。",
                "- 不要复述上一条回复的核心句式或原封不动重复同一个话题。",
                "- 不要回到此前的牌型距离、向听数字、有效进张或“没有副露”话题。",
            ]
        )
    if event_kind in {"ambient_observation", "idle_heartbeat"}:
        lines.extend(
            [
                "- 不要假装发生未识别到的关键事件或局势逆转。",
                "- 只谈已经发生且确认的事；未发生的事完全不要提及。",
                "- 可以评论立直后的牌局节奏、桌面气氛或等待感，或从已确认事实中挑一个轻松观察。",
                "- 不要说“没什么好说的”，也不要重复上一条回复的核心句式。",
            ]
        )
    if allow_rare_joke:
        obvious_question = _mahjong_obvious_question(state)
        lines.extend(
            [
                "",
                "本次允许使用一次独特彩蛋：",
                f"- 可以突然问“铜须是生物吗？”，或问“{obvious_question}”。",
                "- 这是郑重确认显而易见事实的跑题式玩笑，不代表真的识别失败；不要回答、解释或连续重复。",
                "- 本次只能选择这个独特彩蛋，不得同时使用其他口头禅、其他彩蛋句式或第二个梗。",
            ]
        )
    elif allow_absurd_banter:
        absurd_cards = (
            _ABSURD_RIICHI_CARDS
            if event_kind == "riichi_pressure"
            else _ABSURD_MELD_CARDS
            if event_kind == "opponent_meld"
            else _ABSURD_BANTER_CARDS
        )
        absurd_name, absurd_instruction = absurd_cards[
            max(0, int(absurd_variant)) % len(absurd_cards)
        ]
        lines.extend(
            [
                "",
                "本次启用逆天模式：",
                f"- 本次只使用“{absurd_name}”这一张话术卡：{absurd_instruction}",
                "- 可以结合当前已确认事件改写，不必逐字照搬；核心是一本正经地说一点离谱但无害的话。",
                "- 如果示例句或它的核心句式刚在近期对话中出现过，保留这张卡的语气但换一个新的短句；不要逐字复读。",
                "- 一条回复只允许一个梗，不得再叠加铜须、自我怀疑句式或第二张话术卡。",
                "- 仍然不得编造牌局事实、播报内部数字或给出任何打牌指导。",
                "- 除立直、对手吃碰杠和已出现的和牌窗口外，本次输出只能围绕上面指定的话术卡与玩家互动；禁止补充任何关于手牌、牌型、牌运、手气、好坏、强弱、顺不顺或和牌前景的评价，即使上下文里存在这些信息也必须忽略。",
            ]
        )
        if event_kind == "riichi_pressure":
            lines.append("- 立直是本次回复的主体，必须先明确反应立直，再接梗；不得只说泛泛的日常话。")
        elif event_kind == "opponent_meld":
            lines.append("- 副露是本次回复的主体，必须先准确说清玩家与吃碰杠动作，再接梗。")
    else:
        lines.extend(
            [
                "",
                "本次可选语气：",
                "- 可以把“哦对的对的……哎呀不对不对……对……对吗？”当作轻松的自我怀疑式吐槽：局势转折、玩家操作出乎预期，甚至只是当前气氛适合时都可以使用，不要求真的经历三阶段推理。",
                "- 该句式在一条回复中最多完整出现一次，并尽量避免连续两次回复都使用；不要拆乱顺序，也不要与其他梗混用或提及人物来源。",
            ]
        )
    signature_data: dict[str, Any] = {
        "event": event_kind,
        "round": str(state.get("round_id") or ""),
    }
    if event_kind == "player_interaction":
        signature_data.update(
            {
                "discard": str(state.get("last_observed_discard") or ""),
            }
        )
    elif event_kind == "riichi_pressure":
        signature_data["riichi"] = sorted(riichi_players)
    elif event_kind == "checkpoint_observation":
        signature_data["headline"] = str(presentation.get("headline") or "")
    elif event_kind == "win_opportunity":
        signature_data["action"] = action_event or "win"
    elif event_kind == "opponent_meld":
        signature_data["meld"] = opponent_meld_event
    elif event_kind in {"ambient_observation", "idle_heartbeat", "casual_chat"}:
        signature_data["context"] = str(ambient_signature or "")
    signature = json.dumps(signature_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "\n".join(lines), signature, event_kind


def build_absurd_banter_reply(
    payload: dict[str, Any],
    *,
    event_kind: str,
    absurd_variant: int = 0,
    opponent_meld_event: dict[str, Any] | None = None,
) -> str:
    """Return the final spoken line for absurd mode without LLM rewriting.

    Ordinary absurd-mode speech deliberately bypasses generative expansion.
    Live testing showed that a negative prompt was insufficient: the model
    repeatedly appended hand-luck judgments even when the cue prohibited them.
    Critical win reactions keep their normal character-generated path.
    """

    if event_kind == "win_opportunity":
        return ""
    variant = max(0, int(absurd_variant))
    state = payload.get("round_state") if isinstance(payload.get("round_state"), dict) else {}
    if event_kind == "riichi_pressure":
        players = [
            _PLAYER_LABELS.get(str(player), str(player))
            for player in (state.get("riichi_players") or [])
            if str(player)
        ]
        subject = "、".join(players) if players else "有人"
        joke = _ABSURD_RIICHI_REPLIES[variant % len(_ABSURD_RIICHI_REPLIES)]
        return f"{subject}立直了。{joke}"
    if event_kind == "opponent_meld":
        event_label, _ = _opponent_meld_event_summary(opponent_meld_event or {})
        joke = _ABSURD_MELD_REPLIES[variant % len(_ABSURD_MELD_REPLIES)]
        return f"{event_label}。{joke}"
    return _ABSURD_BANTER_REPLIES[variant % len(_ABSURD_BANTER_REPLIES)]


def _mahjong_obvious_question(state: dict[str, Any]) -> str:
    tiles = state.get("last_hand_tiles") or []
    honors = sorted(
        {
            tile
            for value in tiles
            if (tile := normalize_tile(value)) and tile.endswith("z")
        }
    )
    if honors:
        return f"{_tile_label(honors[0])}是字牌吗？"
    if state.get("riichi_players"):
        return "立直棒是棒吗？"
    return "麻将牌是牌吗？"


def _companion_speech_evidence(raw_evidence: Any) -> list[str]:
    """Keep spoken context short while detailed panels retain full evidence."""
    verbose_prefixes = (
        "完整手牌：",
        "手牌观察：",
        "四家点数：",
        "牌河观察：",
        "役满观察：",
    )
    diagnostic_markers = (
        "证据不足",
        "信息不足",
        "尚无稳定置信度",
        "识别不确定性",
        "按未知风险处理",
    )
    concise: list[str] = []
    for value in raw_evidence if isinstance(raw_evidence, list) else []:
        text = str(value or "").strip()
        if not text or text.startswith(verbose_prefixes):
            continue
        # Keep static diagnostics on the analysis panel. Speaking them every
        # heartbeat turns "0 melds" and route distance into repetitive chat.
        if text.startswith("副露观察：") and "自己0组副露" in text:
            continue
        if "向听" in text or "有效进张" in text or ("距离" in text and "张" in text):
            continue
        if any(marker in text for marker in diagnostic_markers):
            continue
        if "立直" in text and any(
            marker in text
            for marker in (
                "暂未检测到",
                "未检测到",
                "没有检测到",
                "没检测到",
                "没有看到",
                "没看到",
                "无人立直",
                "没人立直",
                "无立直压力",
            )
        ):
            continue
        if len(text) > 80:
            continue
        concise.append(text)
    return concise


def _opponent_meld_event_summary(event: dict[str, Any]) -> tuple[str, str]:
    new_melds = [item for item in (event.get("new_melds") or []) if isinstance(item, dict)]
    if not new_melds:
        return "对手完成了新的副露", "桌面上多了一组已经公开的副露。"
    actions = [_opponent_meld_action_text(item) for item in new_melds]
    summary = "；".join(action for action in actions if action)
    return summary or "对手完成了新的副露", f"{summary}。" if summary else "桌面上多了一组已经公开的副露。"


def _opponent_meld_event_evidence(event: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    new_melds = [item for item in (event.get("new_melds") or []) if isinstance(item, dict)]
    if new_melds:
        actions = "；".join(
            action
            for item in new_melds
            if (action := _opponent_meld_action_text(item))
        )
        if actions:
            lines.append(f"对手动作：{actions}。")
    current = event.get("current_melds")
    current = current if isinstance(current, dict) else {}
    for player in ("left_opponent", "top_opponent", "right_opponent"):
        groups = [item for item in (current.get(player) or []) if isinstance(item, dict)]
        if not groups:
            continue
        group_text = "；".join(_meld_group_text(group) for group in groups)
        lines.append(f"{_PLAYER_LABELS.get(player, player)}当前副露：{group_text}。")
    return lines


def _opponent_meld_action_text(meld: dict[str, Any]) -> str:
    player = str(meld.get("owner") or meld.get("player") or "")
    player_label = _PLAYER_LABELS.get(player, player or "对手")
    kind = str(meld.get("kind") or "").lower()
    tiles = [
        tile
        for value in (meld.get("tiles") or [])
        if (tile := normalize_tile(value))
    ]
    labels = [_tile_label(tile) for tile in tiles]
    if kind == "pon":
        return f"{player_label}碰了{labels[0] if labels else '一张牌'}"
    if kind == "chi":
        return f"{player_label}吃了{'、'.join(labels) if labels else '一组牌'}"
    if kind in {"kan", "minkan", "ankan", "kakan"}:
        return f"{player_label}杠了{labels[0] if labels else '一张牌'}"
    return f"{player_label}完成副露{'：' + '、'.join(labels) if labels else ''}"


def _meld_group_text(meld: dict[str, Any]) -> str:
    kind = str(meld.get("kind") or "").lower()
    tiles = [
        tile
        for value in (meld.get("tiles") or [])
        if (tile := normalize_tile(value))
    ]
    labels = [_tile_label(tile) for tile in tiles]
    tile_text = "、".join(labels) if labels else "牌面待确认"
    if kind == "pon":
        return f"碰{labels[0] if labels else '牌'}（{tile_text}）"
    if kind == "chi":
        return f"吃（{tile_text}）"
    if kind in {"kan", "minkan", "ankan", "kakan"}:
        return f"杠{labels[0] if labels else '牌'}（{tile_text}）"
    return tile_text


def sanitize_round_history(history: Any, *, mode: str) -> list[dict[str, Any]]:
    normalized_mode = _valid_live_advice_mode(mode)
    cleaned: list[dict[str, Any]] = []
    for entry in history if isinstance(history, list) else []:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        archived_state = item.get("state") if isinstance(item.get("state"), dict) else None
        if archived_state is not None:
            item["state"] = build_public_payload(
                {"last_decision": {}, "round_state": archived_state},
                mode=normalized_mode,
            )["round_state"]
        for key in (
            "current_plan",
            "opening_plan",
            "local_direction",
            "local_plan",
            "local_detail",
            "suggestion",
            "discard_priority",
            "top_candidates",
            "candidates",
        ):
            if key in item:
                item[key] = [] if key in {"discard_priority", "top_candidates", "candidates"} else ""
        item["guidance_mode"] = normalized_mode
        cleaned.append(item)
    return cleaned


def _sanitize_decision(
    decision: dict[str, Any],
    public_state: dict[str, Any],
    presentation: dict[str, Any],
) -> dict[str, Any]:
    public = dict(decision)
    is_strategy = presentation["mode"] == "strategy"
    public.update(
        {
            "decision_type": "companion_observation"
            if not is_strategy
            else "risk_observation",
            "action_required": False,
            "priority": 0,
            "summary": presentation["headline"],
            "detail": presentation["detail"],
            "suggestion": str(presentation.get("primary_action") or "") if is_strategy else "",
            "buttons": [],
            "coach_state": public_state,
            "presentation": presentation,
        }
    )
    perception = dict(decision.get("perception") or {})
    raw_strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
    perception["strategy"] = _sanitize_strategy(raw_strategy, presentation)
    raw_action = perception.get("action") if isinstance(perception.get("action"), dict) else {}
    public_action = dict(raw_action)
    public_action.pop("call_recommendation", None)
    if is_strategy and presentation.get("source_decision_type") == "call_window":
        public_action["call_recommendation"] = dict(presentation.get("call_recommendation") or {})
    perception["action"] = public_action
    public["perception"] = perception
    return public


def _sanitize_state(state: dict[str, Any], presentation: dict[str, Any]) -> dict[str, Any]:
    public = dict(state)
    is_strategy = presentation["mode"] == "strategy"
    primary_action = str(presentation.get("primary_action") or "") if is_strategy else ""
    public.update(
        {
            "current_plan": primary_action,
            "opening_plan": "",
            "local_direction": str(presentation.get("strategy_direction") or "") if is_strategy else "",
            "local_plan": primary_action,
            "local_detail": str(presentation.get("primary_reason") or "") if is_strategy else "",
            "attack_defense_bias": str(presentation.get("posture") or "observing") if is_strategy else "observing",
            "defense_posture": str(state.get("defense_posture") or "") if is_strategy else "",
            "defense_risk_budget": float(state.get("defense_risk_budget") or 0.0) if is_strategy else 0.0,
            "target_shapes": list(presentation.get("evidence") or []) if is_strategy else [],
            "caution_points": list(presentation.get("risk_sources") or []) if is_strategy else [],
            "guidance_mode": presentation["mode"],
        }
    )
    return public


def _sanitize_strategy(strategy: dict[str, Any], presentation: dict[str, Any]) -> dict[str, Any]:
    is_strategy = presentation["mode"] == "strategy"
    if not is_strategy:
        return {
            "recommended_discard": "",
            "recommended_discard_label": "",
            "call_recommendation": {},
            "risk_options": [],
            "top_candidates": [],
            "candidates": [],
        }
    public_candidates = list(presentation.get("risk_options") or []) if is_strategy else []
    allowed = {
        "table_context": strategy.get("table_context") or {},
        "posture": str(strategy.get("posture") or ""),
        "risk_budget": float(strategy.get("risk_budget") or 0.0),
        "risk_budget_model": strategy.get("risk_budget_model") or {},
        "risk_budget_calculation": str(strategy.get("risk_budget_calculation") or ""),
        "win_potential": str(strategy.get("win_potential") or ""),
        "safe_shape_candidates": int(strategy.get("safe_shape_candidates") or 0),
        "shape_status": strategy.get("shape_status") or {},
        "risk_scale_note": presentation["risk_scale_note"],
        "risk_scale_legend": presentation["risk_scale_legend"],
        "risk_range": presentation["risk_range"],
        "risk_reference": presentation["risk_reference"],
        "pressure_level": presentation["pressure_level"],
        "hand_viability": presentation["hand_viability"],
        "risk_sources": list(presentation["risk_sources"]),
        "uncertainty": list(presentation["uncertainty"]),
        "recommended_discard": str(presentation.get("primary_discard") or "") if is_strategy else "",
        "recommended_discard_label": str(presentation.get("primary_discard_label") or "") if is_strategy else "",
        "call_recommendation": (
            dict(presentation.get("call_recommendation") or {})
            if is_strategy and presentation.get("source_decision_type") == "call_window"
            else {}
        ),
        "risk_options": public_candidates,
        "top_candidates": public_candidates,
        "candidates": public_candidates,
    }
    return allowed


def _build_presentation(
    decision: dict[str, Any],
    state: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    perception = decision.get("perception") if isinstance(decision.get("perception"), dict) else {}
    strategy = perception.get("strategy") if isinstance(perception.get("strategy"), dict) else {}
    hand = perception.get("hand") if isinstance(perception.get("hand"), dict) else {}
    meld = perception.get("meld") if isinstance(perception.get("meld"), dict) else {}
    river = perception.get("river") if isinstance(perception.get("river"), dict) else {}
    action = perception.get("action") if isinstance(perception.get("action"), dict) else {}
    yakuman = perception.get("yakuman") if isinstance(perception.get("yakuman"), dict) else {}
    source_decision_type = str(decision.get("decision_type") or "")
    call_recommendation = (
        _public_call_recommendation(action.get("call_recommendation"))
        if mode == "strategy" and source_decision_type == "call_window"
        else {}
    )
    action_event = _critical_action_event(decision)
    candidates = [
        item
        for item in (strategy.get("top_candidates") or strategy.get("candidates") or [])
        if isinstance(item, dict)
    ]
    riichi_players = [str(item) for item in (state.get("riichi_players") or []) if str(item)]
    risk_values = [
        max(0.0, min(100.0, float(item.get("defense_risk") or 0.0)))
        for item in candidates
    ]
    risk_range = {
        "min": round(min(risk_values), 1) if risk_values else None,
        "max": round(max(risk_values), 1) if risk_values else None,
    }
    risk_reference = round(float(strategy.get("risk_budget") or state.get("defense_risk_budget") or 0.0), 1)
    risk_options = (
        _public_risk_options(candidates, risk_reference)
        if mode == "strategy"
        else []
    )
    suppress_discard = source_decision_type in {
        "call_window",
        "win_window",
        "round_settlement",
        "settlement_candidate",
        "awaiting_next_round",
        "round_idle",
    }
    if suppress_discard:
        # A settlement/win event must never carry a previous frame's discard
        # card into the new state, even if the raw engine payload still keeps
        # candidates for diagnostics.
        risk_options = []
    primary_option = risk_options[0] if risk_options and not suppress_discard else {}
    primary_discard = str(primary_option.get("tile") or "")
    primary_discard_label = str(primary_option.get("tile_label") or "")
    primary_action = f"主建议：打{primary_discard_label}" if primary_discard_label else ""
    primary_reason = ""
    if primary_option:
        primary_reason = (
            f"首选理由：{str(primary_option.get('risk_summary') or '风险待评估')}，"
            f"{str(primary_option.get('shape_summary') or '牌型待评估')}；"
            f"{str(primary_option.get('tradeoff') or '')}"
        ).rstrip("；")
    if call_recommendation:
        primary_discard = ""
        primary_discard_label = ""
        primary_action = str(call_recommendation.get("primary_action") or "")
        call_reason = str(call_recommendation.get("reason") or "").strip()
        primary_reason = f"鸣牌理由：{call_reason}" if call_reason else ""
    pressure_level = _pressure_level(riichi_players, risk_values)
    hand_viability = {
        "strong": "和牌可能性较强",
        "live": "仍有和牌路线",
        "weak": "和牌路线偏弱",
    }.get(str(strategy.get("win_potential") or ""), "和牌可能性尚未稳定")
    evidence = _evidence_lines(
        state,
        hand,
        riichi_players,
        meld=meld,
        river=river,
        action=action,
        decision=decision,
        yakuman=yakuman,
    )
    risk_sources = _risk_sources(strategy, candidates, riichi_players)
    uncertainty = _uncertainty_lines(state)
    posture = str(strategy.get("posture") or state.get("defense_posture") or "")
    posture_label = {
        "push": "推进评估",
        "mawashi": "保形评估",
        "fold": "高压防守评估",
    }.get(posture, "攻守态势待评估")
    risk_budget_calculation = str(strategy.get("risk_budget_calculation") or "").strip()
    shape_summary = _shape_summary(strategy, state, candidates, hand_viability)
    table_context_summary, table_context_explanation = _table_context_lines(strategy, state)
    strategy_direction, strategy_reason = _public_strategy_direction(
        source_decision_type=source_decision_type,
        strategy=strategy,
        state=state,
        hand_viability=hand_viability,
        shape_summary=shape_summary,
    )
    if mode == "companion":
        if source_decision_type == "win_window":
            headline = _win_companion_reaction(action_event, state)
        else:
            headline = _companion_headline(
                riichi_players,
                hand_viability,
                str(state.get("companion_reaction") or ""),
            )
        # The companion product must not leak dormant strategy calculations
        # through a formatter. Keep only short, confirmed public events here;
        # the hidden N.E.K.O cue applies its own stricter speech filter.
        detail_parts = _companion_panel_event_lines(
            source_decision_type=source_decision_type,
            action_event=action_event,
            riichi_players=riichi_players,
            evidence=evidence,
        )
        evidence = list(detail_parts)
        risk_sources = []
        uncertainty = []
        risk_range = {"min": None, "max": None}
        risk_reference = 0.0
        risk_options = []
        primary_discard = ""
        primary_discard_label = ""
        primary_action = ""
        primary_reason = ""
        call_recommendation = {}
        pressure_level = ""
        hand_viability = ""
        posture = ""
        posture_label = ""
        shape_summary = ""
        strategy_direction = ""
        strategy_reason = ""
        risk_budget_calculation = ""
        table_context_summary = ""
        table_context_explanation = ""
    else:
        if source_decision_type == "win_window":
            headline = f"界面事件：检测到{_action_label(action_event) or '和牌'}按钮"
        elif source_decision_type == "call_window":
            headline = f"鸣牌窗口：{primary_action or '正在核对吃碰收益'}"
        else:
            riichi_label = "无立直压力" if not riichi_players else f"{len(riichi_players)}家立直压力"
            headline = f"风险观察：{riichi_label}，整体{pressure_level}"
        detail_parts = [
            primary_action,
            primary_reason,
            strategy_direction,
            shape_summary,
            strategy_reason,
            _risk_range_text(risk_range, risk_reference),
            *risk_sources[:4],
            f"牌型状态：{hand_viability}。",
            *evidence[:3],
            *uncertainty[:2],
        ]
    detail = "\n".join(dict.fromkeys(part for part in detail_parts if part))
    return {
        "version": 2,
        "mode": mode,
        "mode_label": "吐槽伙伴" if mode == "companion" else "风险策略",
        "source_decision_type": source_decision_type,
        "action_event": action_event,
        "headline": headline,
        "detail": detail,
        "evidence": evidence,
        "risk_sources": risk_sources,
        "uncertainty": uncertainty,
        "risk_range": risk_range,
        "risk_reference": risk_reference,
        "risk_options": risk_options,
        "primary_discard": primary_discard,
        "primary_discard_label": primary_discard_label,
        "primary_action": primary_action,
        "primary_reason": primary_reason,
        "call_recommendation": call_recommendation,
        "pressure_level": pressure_level,
        "hand_viability": hand_viability,
        "posture": posture,
        "posture_label": posture_label,
        "shape_summary": shape_summary,
        "strategy_direction": strategy_direction,
        "strategy_reason": strategy_reason,
        "risk_budget_calculation": risk_budget_calculation,
        "table_context_summary": table_context_summary,
        "table_context_explanation": table_context_explanation,
        "risk_scale_note": "0–100 是规则型相对危险指数，不是放铳概率，也不是操作指令。",
        "risk_scale_legend": "0–9 极低｜10–39 较低｜40–59 中等｜60–79 高｜80–100 很高",
        "non_prescriptive": mode == "companion",
    }


def _public_risk_options(
    candidates: list[dict[str, Any]],
    reference: float,
) -> list[dict[str, Any]]:
    """Convert private candidates into a bounded, actionable strategy view.

    Only the discard identity and the already-computed risk/shape explanation
    are exposed. Raw visibility records and internal scoring details remain
    private. Candidate order is the strategy engine's order, so A is the main
    recommendation and B/C are comparisons.
    """
    raw_options = candidates[:3]
    if not raw_options:
        return []

    risks = [max(0.0, min(100.0, float(item.get("defense_risk") or 0.0))) for item in raw_options]
    effective_counts = [int(item.get("effective_count") or 0) for item in raw_options]
    lowest_risk = min(risks, default=0.0)
    highest_effective = max(effective_counts, default=0)
    rendered: list[dict[str, Any]] = []
    for index, item in enumerate(raw_options):
        tile = normalize_tile(item.get("tile")) or ""
        tile_label = _tile_label(tile) if tile else ""
        risk = risks[index]
        effective_count = effective_counts[index]
        shanten = int(item.get("shanten", 8))
        shape_loss = max(0, int(item.get("shape_loss") or 0))
        effective_delta = int(item.get("effective_count_delta") or 0)
        yaku_viable = bool(item.get("yaku_viable", True))
        yaku_status = str(item.get("yaku_status") or "unknown")
        yaku_label = str(item.get("yaku_label") or "役未确认")
        yaku_reason = str(item.get("yaku_reason") or "役条件尚未完成评估。")
        yaku_routes = [str(route) for route in (item.get("yaku_routes") or []) if str(route).strip()]
        within_reference = bool(item.get("within_risk_budget", risk <= reference))
        risk_delta = round(reference - risk, 1)
        risk_level = _public_risk_level(risk)
        basis = _public_risk_basis(item)
        shanten_label = "听牌" if shanten <= 0 else f"{shanten}向听"

        adjustments: list[str] = []
        components = item.get("risk_components") if isinstance(item.get("risk_components"), dict) else {}
        dora_adjustment = float(components.get("dora_adjustment") or 0.0)
        multi_adjustment = float(components.get("multi_riichi_adjustment") or 0.0)
        if dora_adjustment:
            adjustments.append(f"宝牌因素+{dora_adjustment:.0f}")
        if multi_adjustment:
            adjustments.append(f"多家立直压力+{multi_adjustment:.0f}")
        adjustment_text = "；" + "、".join(adjustments) if adjustments else ""

        risk_compare = risk - lowest_risk
        effective_compare = highest_effective - effective_count
        comparison_parts = [
            "风险为同组三案最低"
            if risk_compare <= 0
            else f"比同组最低风险高{risk_compare:.0f}",
            "有效牌为同组三案最多"
            if effective_compare <= 0
            else f"比同组最多有效牌少{effective_compare}张",
        ]
        if within_reference and shape_loss == 0:
            tradeoff = "在个性化参考线内，同时没有增加向听；可用于观察低风险与保形能否兼得。"
        elif within_reference:
            tradeoff = f"风险仍在参考线内，但牌型损失为{shape_loss}；这是安全性与成型速度的交换。"
        elif effective_count == highest_effective:
            tradeoff = "牌型保留相对较多，但风险超过个性化参考线；收益与暴露同时上升。"
        else:
            tradeoff = "风险超过个性化参考线，且牌型保留并非同组最高；需要重点留意风险来源。"
        if not yaku_viable and yaku_status in {"unknown", "no_yaku"}:
            tradeoff = f"{tradeoff} 但{yaku_label}，不能把进听直接等同于可以和牌。"

        rendered.append(
            {
                "id": "ABC"[index],
                "tile": tile,
                "tile_label": tile_label,
                "recommended": index == 0,
                "label": (
                    f"方案 {'ABC'[index]} · 打{tile_label}"
                    if tile_label
                    else f"方案 {'ABC'[index]} · 牌名待确认"
                ),
                "risk": round(risk, 1),
                "risk_level": risk_level,
                "risk_summary": f"相对风险{risk_level} {risk:.0f}/100",
                "risk_status": "参考线内" if within_reference else "超过参考线",
                "risk_delta": risk_delta,
                "risk_delta_label": (
                    f"低于参考线{risk_delta:.0f}"
                    if risk_delta >= 0
                    else f"高于参考线{abs(risk_delta):.0f}"
                ),
                "risk_basis": basis,
                "safety": f"主要规则依据：{basis}",
                "safety_reason": f"打{tile_label or '这张牌'}时，模型识别到的主要规则依据是“{basis}”{adjustment_text}。",
                "risk_reason": (
                    f"相对危险指数为{risk:.0f}/100，{risk_level}；"
                    f"个性化参考线为{reference:.0f}/100，"
                    f"{'低于参考线' if within_reference else '高于参考线'}{abs(risk_delta):.0f}。"
                ),
                "shanten": shanten,
                "effective_count": effective_count,
                "effective_count_delta": effective_delta,
                "shape_loss": shape_loss,
                "yaku_viable": yaku_viable,
                "yaku_status": yaku_status,
                "yaku_label": yaku_label,
                "yaku_reason": yaku_reason,
                "yaku_routes": yaku_routes,
                "shape_summary": f"{shanten_label} · 有效牌{effective_count}张 · {yaku_label}",
                "shape_reason": (
                    f"牌型结果为{shanten_label}，有效牌{effective_count}张，牌型损失{shape_loss}；"
                    f"相对本批牌效基准变化{effective_delta:+d}张。{yaku_reason}"
                ),
                "comparison_reason": "；".join(comparison_parts) + "。",
                "tradeoff": tradeoff,
                "tone": "primary" if index == 0 else ("safe" if within_reference else "danger"),
                "non_prescriptive": False,
            }
        )
    return rendered


def _public_risk_level(value: float) -> str:
    if value >= 80:
        return "很高"
    if value >= 60:
        return "高"
    if value >= 40:
        return "中等"
    if value >= 10:
        return "较低"
    return "极低"


def _public_risk_basis(candidate: dict[str, Any]) -> str:
    """Return a whitelisted rule label without copying a private tile name."""
    components = candidate.get("risk_components") if isinstance(candidate.get("risk_components"), dict) else {}
    source = " ".join(
        (
            str(candidate.get("safety") or ""),
            str(components.get("base_reason") or ""),
            *(
                str(item)
                for item in (candidate.get("safety_evidence") or [])
                if str(item).strip()
            ),
        )
    )
    # With no active riichi opponent the strategy engine deliberately assigns
    # zero defense risk and ranks by hand efficiency/value.  Treating that
    # state as "river evidence missing" made every normal-play option look like
    # an uncertain defensive fallback even though the river is irrelevant to
    # the ordering.
    if "无立直压力" in source:
        return "当前无人立直，按牌效与役种路线比较"
    for label in (
        "无筋中张",
        "无筋2/8",
        "无筋幺九",
        "字牌已见3枚",
        "字牌已见2枚",
        "字牌已见0–1枚",
        "宝牌周边",
        "宝牌",
        "现物",
        "全见",
        "壁",
        "筋",
    ):
        if label in source:
            return label
    return "牌河证据不足，按未知风险处理"


def _companion_panel_event_lines(
    *,
    source_decision_type: str,
    action_event: str,
    riichi_players: list[str],
    evidence: list[str],
) -> list[str]:
    """Return terse event facts for the companion-only native panel."""

    if source_decision_type == "win_window":
        return [f"已识别到{_action_label(action_event) or '和牌'}界面事件。"]
    if source_decision_type in {"round_settlement", "settlement_candidate", "awaiting_next_round"}:
        return ["本局正在结算，等待下一局开始。"]
    if source_decision_type == "opening_plan":
        return ["新一局已经开始，继续观察公开事件。"]
    if riichi_players:
        labels = "、".join(_PLAYER_LABELS.get(player, player) for player in riichi_players)
        return [f"已确认{labels}立直。"]
    confirmed_prefixes = ("对手副露：", "界面按钮：")
    for line in evidence:
        text = str(line or "").strip()
        if text.startswith(confirmed_prefixes):
            return [text]
    return []


def _companion_headline(
    riichi_players: list[str],
    _hand_viability: str,
    _observed_reaction: str,
) -> str:
    if len(riichi_players) >= 2:
        return "好家伙，两家一起立直，桌上突然就热闹了。"
    if riichi_players:
        return "立直声一响，刚才还好好的，气氛说变就变。"
    return "我在看呢。先不分析手牌，陪你随便聊两句。"


def _critical_action_event(decision: dict[str, Any]) -> str:
    buttons = {
        str(item).strip().lower()
        for item in (decision.get("buttons") or [])
        if str(item).strip()
    }
    for button in ("tsumo", "ron", "riichi", "kan", "pon", "chi"):
        if button in buttons:
            return button
    return ""


def _action_label(action: str) -> str:
    return {
        "tsumo": "自摸",
        "ron": "荣和",
        "riichi": "立直",
        "kan": "杠",
        "pon": "碰",
        "chi": "吃",
        "skip": "跳过",
    }.get(str(action or "").strip().lower(), "")


def _win_companion_reaction(action: str, state: dict[str, Any]) -> str:
    normalized_action = str(action or "").strip().lower()
    if normalized_action == "tsumo":
        variants = (
            "自摸亮了——好嘛，这一下总算没白等。",
            "哦？自己摸进来了，这回牌桌倒是挺给面子。",
            "行啊，关键那张自己送上门了，这一下舒服。",
            "这张摸得正好，前面那点耐心算是留对了。",
            "自摸到了。刚才还在等，它倒真来了。",
            "好家伙，自己把答案摸回来了。",
        )
    else:
        action_label = _action_label(normalized_action) or "和牌"
        variants = (
            f"{action_label}亮了——这一回可算等到了。",
            "哦？对面还真把那张送过来了。",
            "抓到了，这一张来得还挺会挑时候。",
            "可以和了，刚才铺的路总算接上了。",
            "这下等到了，牌桌终于肯配合一次。",
            f"{action_label}到了——行，这一手收得住。",
        )
    seed = "|".join(
        (
            normalized_action,
            str(state.get("round_id") or ""),
            str(state.get("update_count") or ""),
            str(state.get("last_hand_signature") or ""),
        )
    )
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return variants[int.from_bytes(digest[:4], "big") % len(variants)]


def _public_call_recommendation(value: Any) -> dict[str, Any]:
    """Expose only the call decision fields needed by strategy surfaces."""
    if not isinstance(value, dict):
        return {}
    decision = str(value.get("decision") or "").strip().lower()
    action = str(value.get("action") or "").strip().lower()
    primary_action = str(value.get("primary_action") or "").strip()
    if decision not in {"call", "skip"} or action not in {"chi", "pon", "kan", "skip"} or not primary_action:
        return {}
    return {
        "version": 1,
        "decision": decision,
        "action": action,
        "action_label": str(value.get("action_label") or _action_label(action) or "跳过"),
        "primary_action": primary_action,
        "reason": str(value.get("reason") or "").strip(),
        "claimed_tile": str(value.get("claimed_tile") or ""),
        "claimed_tile_label": str(value.get("claimed_tile_label") or ""),
        "claimed_tile_known": bool(value.get("claimed_tile_known")),
        "claimed_tile_inferred": bool(value.get("claimed_tile_inferred")),
        "branch_agreement_without_tile": bool(value.get("branch_agreement_without_tile")),
        "post_discard": str(value.get("post_discard") or ""),
        "post_discard_label": str(value.get("post_discard_label") or ""),
        "baseline_shanten": int(value.get("baseline_shanten") or 0),
        "post_shanten": int(value.get("post_shanten") or 0),
        "effective_count": max(0, int(value.get("effective_count") or 0)),
        "yaku_viable": bool(value.get("yaku_viable")),
        "yaku_status": str(value.get("yaku_status") or "unknown"),
        "yaku_label": str(value.get("yaku_label") or "役未确认"),
        "yaku_reason": str(value.get("yaku_reason") or ""),
        "yaku_routes": [
            str(item)
            for item in (value.get("yaku_routes") or [])
            if str(item).strip()
        ],
        "available_actions": [
            str(item)
            for item in (value.get("available_actions") or [])
            if str(item) in {"chi", "pon", "kan"}
        ],
        "available_action_labels": [
            str(item)
            for item in (value.get("available_action_labels") or [])
            if str(item).strip()
        ],
    }


def _public_strategy_direction(
    *,
    source_decision_type: str,
    strategy: dict[str, Any],
    state: dict[str, Any],
    hand_viability: str,
    shape_summary: str,
) -> tuple[str, str]:
    """Expose one useful direction without leaking a tile ranking or command."""
    preset = str(strategy.get("strategy_preset") or state.get("strategy_preset") or "simple")
    prefix = "简易方向" if preset == "simple" else "攻守方向"
    posture = str(strategy.get("posture") or state.get("defense_posture") or "")
    if source_decision_type == "win_window":
        return "和牌窗口已经出现，当前事件优先于普通牌型分析。", "这里只确认界面事件，不输出点击指令。"
    if source_decision_type == "riichi_window":
        return f"{prefix}：已经进入可立直阶段，重点比较牌型质量、打点与当前场况。", "按钮亮起只代表存在选择，不等于必须改变原有主线。"
    if source_decision_type == "call_window":
        return f"{prefix}：当前有鸣牌选择，只在能明显推进主线时考虑改变门清结构。", "不为单纯鸣牌拆散主要搭子，也不把按钮出现当成必须操作。"
    if posture == "push":
        return f"{prefix}：继续做牌，优先保留主要搭子与有效牌，只回避明显危险。", shape_summary
    if posture == "mawashi":
        return f"{prefix}：边防守边推进，先压低明显风险，同时尽量维持向听和有效牌。", shape_summary
    if posture == "fold":
        return f"{prefix}：当前压力偏高，先收缩风险；有安全余地时仍保留基本成型。", shape_summary
    if source_decision_type == "opening_plan":
        return f"{prefix}：先沿最清晰的成型路线推进，保留相邻搭子、对子与有效进张。", hand_viability
    return f"{prefix}：继续观察成型速度，优先维护有效牌和主线结构，暂不做极端防守。", shape_summary


def _risk_range_text(risk_range: dict[str, float | None], reference: float) -> str:
    low, high = risk_range.get("min"), risk_range.get("max")
    if low is None or high is None:
        return "风险区间：当前信息不足，尚未形成稳定刻度。"
    return (
        f"风险区间：已评估变化的相对危险指数为 {low:.0f}–{high:.0f}/100；"
        f"个性化参考线为 {reference:.0f}/100。该参考线只用于解释模型权重。"
    )


def _pressure_level(riichi_players: list[str], risk_values: list[float]) -> str:
    peak = max(risk_values, default=0.0)
    if len(riichi_players) >= 2 or peak >= 80:
        return "高压"
    if riichi_players or peak >= 40:
        return "中等压力"
    return "低压"


def _evidence_lines(
    state: dict[str, Any],
    hand: dict[str, Any],
    riichi_players: list[str],
    *,
    meld: dict[str, Any],
    river: dict[str, Any],
    action: dict[str, Any],
    decision: dict[str, Any],
    yakuman: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    if riichi_players:
        labels = "、".join(_PLAYER_LABELS.get(player, player) for player in riichi_players)
        lines.append(f"牌桌观察：检测到{labels}立直。")
    else:
        lines.append("牌桌观察：暂未检测到对手立直。")
    tiles = hand.get("hand_tiles") or state.get("last_hand_tiles") or []
    normalized = [tile for value in tiles if (tile := normalize_tile(value))]
    if normalized:
        lines.append("完整手牌：" + "、".join(_tile_label(tile) for tile in normalized) + "。")
    honors = Counter(tile for tile in normalized if tile.endswith("z"))
    if honors:
        exact = "、".join(
            f"{_tile_label(tile)}×{count}" if count > 1 else _tile_label(tile)
            for tile, count in sorted(honors.items())
        )
        lines.append(f"手牌观察：字牌共{sum(honors.values())}张，具体为{exact}。")
    open_melds = int(state.get("last_open_meld_count") or 0)
    meld_tiles = [
        tile
        for value in (meld.get("tiles") or state.get("last_meld_tiles") or [])
        if (tile := normalize_tile(value))
    ]
    meld_suffix = f"，牌为{'、'.join(_tile_label(tile) for tile in meld_tiles)}" if meld_tiles else ""
    lines.append(f"副露观察：当前识别到自己{open_melds}组副露{meld_suffix}。")
    opponent_melds = state.get("last_opponent_melds")
    if not isinstance(opponent_melds, dict) or not opponent_melds:
        opponent_melds = river.get("opponent_melds") if isinstance(river.get("opponent_melds"), dict) else {}
    for player in ("left_opponent", "top_opponent", "right_opponent"):
        groups = [item for item in (opponent_melds.get(player) or []) if isinstance(item, dict)]
        if not groups:
            continue
        group_text: list[str] = []
        for group in groups:
            group_text.append(_meld_group_text(group))
        lines.append(f"对手副露：{_PLAYER_LABELS[player]}{len(groups)}组，{'；'.join(group_text)}。")
    discard_piles = state.get("last_discard_piles")
    if not isinstance(discard_piles, dict) or not discard_piles:
        discard_piles = river.get("discard_piles") if isinstance(river.get("discard_piles"), dict) else {}
    river_parts: list[str] = []
    for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
        entries = [item for item in (discard_piles.get(player) or []) if isinstance(item, dict)]
        if not entries:
            continue
        last_tile = normalize_tile(entries[-1].get("tile"))
        suffix = f"，末张{_tile_label(last_tile)}" if last_tile else ""
        river_parts.append(f"{_PLAYER_LABELS[player]}{len(entries)}张{suffix}")
    if river_parts:
        lines.append("牌河观察：" + "；".join(river_parts) + "。")
    scores = state.get("player_scores") if isinstance(state.get("player_scores"), dict) else {}
    ranks = state.get("player_ranks") if isinstance(state.get("player_ranks"), dict) else {}
    score_parts: list[str] = []
    for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
        if player not in scores:
            continue
        rank = int(ranks.get(player) or 0)
        score_parts.append(
            f"{_PLAYER_LABELS[player]}{int(scores[player]):,}点" + (f"（第{rank}位）" if rank else "")
        )
    if score_parts:
        lines.append("四家点数：" + "；".join(score_parts) + "。")
    if state.get("honba_count") is not None or state.get("table_riichi_stick_count") is not None:
        lines.append(
            f"场棒信息：本场{int(state.get('honba_count') or 0)}，供托{int(state.get('table_riichi_stick_count') or 0)}。"
        )
    buttons = [str(item).strip() for item in (decision.get("buttons") or []) if str(item).strip()]
    if buttons:
        labels = [_action_label(button) or button for button in buttons]
        lines.append("界面按钮：识别到" + "、".join(labels) + "；这里只记录界面事实。")
    elif action:
        lines.append(f"界面按钮：当前扫描来源为{str(action.get('source') or '未发现关键按钮')}。")
    observed_discard = normalize_tile(state.get("last_observed_discard"))
    if observed_discard:
        lines.append(f"行为观察：上一轮识别到你打出了{_tile_label(observed_discard)}。")
    routes = [item for item in (yakuman.get("routes") or []) if isinstance(item, dict)]
    if routes:
        top = routes[0]
        lines.append(
            f"役满观察：最近路线为{str(top.get('label') or top.get('route') or '待确认')}，"
            f"距离约{int(top.get('distance') or 0)}张。"
        )
    return lines


def _shape_summary(
    strategy: dict[str, Any],
    state: dict[str, Any],
    candidates: list[dict[str, Any]],
    hand_viability: str,
) -> str:
    status = strategy.get("shape_status") if isinstance(strategy.get("shape_status"), dict) else {}
    route = str(status.get("route") or state.get("closest_shape_route") or "").strip()
    raw_shanten = status.get("shanten", state.get("current_shanten"))
    effective_count = int(
        status.get("effective_count")
        or state.get("current_effective_count")
        or 0
    )
    if route or raw_shanten is not None:
        shanten = int(raw_shanten if raw_shanten is not None else 8)
        if shanten < 0:
            shanten_label = "已成和牌形"
        elif shanten == 0:
            shanten_label = "听牌"
        else:
            shanten_label = f"{shanten}向听"
        parts = [f"最近牌型：{route or '普通面子手'}", shanten_label]
        if effective_count > 0:
            parts.append(f"有效进张约{effective_count}张")
        return "｜".join(parts)
    if not candidates:
        return hand_viability
    shanten_values = [int(item.get("shanten", 8)) for item in candidates]
    effective_values = [int(item.get("effective_count", 0)) for item in candidates]
    best_shanten = min(shanten_values, default=8)
    best_effective = max(effective_values, default=0)
    shanten_label = "听牌" if best_shanten <= 0 else f"{best_shanten}向听"
    return f"{hand_viability}；候选整体最优为{shanten_label}，有效牌最多{best_effective}张"


def _table_context_lines(strategy: dict[str, Any], state: dict[str, Any]) -> tuple[str, str]:
    context = strategy.get("table_context") if isinstance(strategy.get("table_context"), dict) else {}
    context_scores = context.get("scores") if isinstance(context.get("scores"), dict) else {}
    context_ranks = context.get("ranks") if isinstance(context.get("ranks"), dict) else {}
    scores = context_scores or state.get("player_scores")
    ranks = context_ranks or state.get("player_ranks")
    if not isinstance(scores, dict) or not scores:
        return "四家点数尚未稳定", "点数不会在同一小局反复 OCR；等待本大局的稳定读数。"
    ranks = ranks if isinstance(ranks, dict) else {}
    summary_parts: list[str] = []
    for player in ("self", "left_opponent", "top_opponent", "right_opponent"):
        if player not in scores:
            continue
        rank = int(ranks.get(player) or 0)
        summary_parts.append(
            f"{_PLAYER_LABELS[player]}{int(scores[player]):,}" + (f"（{rank}位）" if rank else "")
        )
    honba = int(context.get("honba_count", state.get("honba_count")) or 0)
    sticks = int(context.get("riichi_stick_count", state.get("table_riichi_stick_count")) or 0)
    self_score = int(scores.get("self") or 0)
    self_rank = int(context.get("self_rank") or ranks.get("self") or 0)
    gap_above = int(context.get("gap_above") or 0)
    lead_below = int(context.get("lead_below") or 0)
    explanation = f"本场{honba}、供托{sticks}。"
    if self_rank == 4 and gap_above:
        explanation += f"自己第4位，距上一位{gap_above:,}点。"
    elif self_rank in {1, 2} and lead_below:
        explanation += f"自己第{self_rank}位，领先下一位{lead_below:,}点。"
    elif self_score:
        explanation += f"自己当前第{self_rank or '?'}位。"
    return "｜".join(summary_parts), explanation


def _risk_sources(
    strategy: dict[str, Any],
    candidates: list[dict[str, Any]],
    riichi_players: list[str],
) -> list[str]:
    sources: list[str] = []
    if not riichi_players:
        if candidates:
            return ["当前无人立直：防守风险不覆盖正常牌效，候选主要比较向听数、有效牌、役种路线与价值保留。"]
        return ["当前无人立直：等待形成稳定候选后，优先按正常牌效与役种路线比较。"]
    if len(riichi_players) >= 2:
        sources.append(f"多家压力：检测到{len(riichi_players)}家立直，来自不同方向的风险会叠加。")
    elif riichi_players:
        label = _PLAYER_LABELS.get(riichi_players[0], riichi_players[0])
        sources.append(f"立直压力：检测到{label}立直，该方向的未确认牌风险上升。")
    evidence_text = " ".join(
        str(value)
        for item in candidates
        for value in (item.get("safety_evidence") or [])
        if str(value).strip()
    )
    patterns = (
        ("现物", "安全信息：已识别到现物线索；它只对对应对手构成最低相对风险。"),
        ("筋", "筋信息：牌河形成筋关系；筋只能降低相对风险，不能视为绝对安全。"),
        ("壁", "壁信息：可见牌形成壁线索，相关组合出现的空间有所缩小。"),
        ("无筋", "主要风险：存在无筋牌，尤其无筋中张在模型中权重较高。"),
        ("宝牌", "打点风险：宝牌或宝牌周边会提高对手成牌价值与相对危险权重。"),
        ("字牌", "字牌风险：字牌危险度依据已见张数变化，未见字牌不会自动视为安全。"),
    )
    for needle, message in patterns:
        if needle in evidence_text and message not in sources:
            sources.append(message)
    if not sources:
        sources.append("风险来源：当前牌河证据不足，模型没有把未知信息当成安全信息。")
    return sources


def _uncertainty_lines(state: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    hand_confidence = float(state.get("last_hand_confidence") or 0.0)
    river_confidence = float(state.get("last_river_confidence") or 0.0)
    table_confidence = float(state.get("table_context_confidence") or 0.0)
    if hand_confidence:
        lines.append(f"识别不确定性：手牌置信度 {hand_confidence:.0%}。")
    if river_confidence:
        lines.append(f"识别不确定性：牌河置信度 {river_confidence:.0%}。")
    if table_confidence:
        lines.append(f"识别不确定性：点数与场况置信度 {table_confidence:.0%}。")
    if not lines:
        lines.append("识别不确定性：当前尚无稳定置信度，结论只作局势说明。")
    return lines


def _tile_label(tile: str) -> str:
    rank, suit = tile[0], tile[1]
    if suit == "z":
        return {"1": "东", "2": "南", "3": "西", "4": "北", "5": "白", "6": "发", "7": "中"}.get(rank, tile)
    suffix = {"m": "万", "p": "筒", "s": "索"}.get(suit, suit)
    return f"{'赤' if rank == '0' else ''}{'5' if rank == '0' else rank}{suffix}"
