from plugin.sdk.plugin import NekoPluginBase, neko_plugin, plugin_entry, lifecycle, Ok
from plugin.sdk.minigame import MiniGamePluginMixin, MiniGameScore, minigame_error


@neko_plugin
class TapDuelPlugin(MiniGamePluginMixin, NekoPluginBase):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.load_minigame()

    @lifecycle(id="startup")
    async def startup(self):
        self.register_minigame_static_ui()
        return Ok({"message": "Tap Duel static UI registered."})

    @plugin_entry(id="tap_duel_status", name="Tap Duel Status", description="Return Tap Duel runtime status.")
    async def status(self):
        return Ok(self.minigame.status())

    @plugin_entry(id="tap_duel_start_session", name="Start Tap Duel Session", description="Start a Tap Duel session.")
    async def start_session(self, session_id: str, lanlan_name: str = ""):
        return Ok(self.minigame.start_session(session_id, lanlan_name=lanlan_name))

    @plugin_entry(id="tap_duel_end_session", name="End Tap Duel Session", description="End a Tap Duel session.")
    async def end_session(self, session_id: str, reason: str = "game_end"):
        return Ok(self.minigame.end_session(session_id, reason=reason))

    @plugin_entry(id="tap_duel_submit_score", name="Submit Tap Duel Score", description="Submit a Tap Duel score.")
    async def submit_score(self, player_id: str, session_id: str, score: int, mode: str = "default"):
        if not self.minigame.manifest.leaderboard:
            return Ok(minigame_error("leaderboard_disabled"))
        return Ok(self.minigame.leaderboard.submit(MiniGameScore(player_id, session_id, int(score), mode=mode)))
