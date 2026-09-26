(function () {
  "use strict";

  function validate(script) {
    if (!script || typeof script !== "object") return ["script must be an object"];
    const errors = [];
    if (!script.title) errors.push("title is required");
    if (!Array.isArray(script.characters) || !script.characters.length) errors.push("characters must be a non-empty array");
    if (!Array.isArray(script.scenes) || !script.scenes.length) errors.push("scenes must be a non-empty array");
    const characterIds = new Set((script.characters || []).map((item) => item && item.id).filter(Boolean));
    (script.scenes || []).forEach((scene, sceneIndex) => {
      if (!scene.id) errors.push(`scenes[${sceneIndex}].id is required`);
      if (!Array.isArray(scene.beats) || !scene.beats.length) errors.push(`scenes[${sceneIndex}].beats must be non-empty`);
      (scene.beats || []).forEach((beat, beatIndex) => {
        const actor = beat.character || beat.actor;
        if (actor && !characterIds.has(actor)) errors.push(`scene ${scene.id || sceneIndex} beat ${beatIndex} has unknown actor ${actor}`);
      });
    });
    return errors;
  }

  function createPlayer(script, handlers) {
    const state = { sceneIndex: 0, beatIndex: 0, done: false };
    const hooks = handlers || {};
    function currentBeat() {
      const scene = (script.scenes || [])[state.sceneIndex];
      return scene && scene.beats ? scene.beats[state.beatIndex] : null;
    }
    async function next() {
      if (state.done) return null;
      const beat = currentBeat();
      if (!beat) {
        state.done = true;
        if (hooks.onEnd) await hooks.onEnd(state);
        return null;
      }
      if (hooks.onBeat) await hooks.onBeat(beat, state);
      state.beatIndex += 1;
      const scene = script.scenes[state.sceneIndex];
      if (scene && state.beatIndex >= scene.beats.length) {
        state.sceneIndex += 1;
        state.beatIndex = 0;
      }
      return beat;
    }
    return { state, next, currentBeat };
  }

  window.NekoTheaterRuntime = { validate, createPlayer };
})();
