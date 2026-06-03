<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/AudioSetup.vue
-->
<template>
  <div class="container">
    <main class="main-content">
      <div class="content-wrapper">
        <h1>Audio Setup</h1>

        <!-- USER NEEDS TO ALLOW MIC -->
        <div v-if="stage === 'prompt-mic'">
          <p>We need your microphone to participate.</p>
          <button :disabled="busy" @click="requestMic">
            {{ busy ? "Requesting..." : "Allow Microphone" }}
          </button>
          <p v-if="error" class="error">{{ error }}</p>
        </div>

        <!-- MIC GRANTED, NOW TEST SPEAKER -->
        <div v-else-if="stage === 'test-speaker'">
          <p>Can you hear this test tone?</p>
          <p>
            If not, please turn up your speaker volume--you'll need to hear for
            the experiment.
          </p>
          <p>(The test tone may take a few seconds to play on Safari.)</p>
          <div class="button-container">
            <button :disabled="busy" @click="playTone">Play Tone</button>
            <button :disabled="!heard" @click="confirmSpeaker">
              Yes, I heard it
            </button>
          </div>
        </div>

        <!-- ALL GOOD -->
        <div v-else-if="stage === 'done'">
          <p>All set! Redirecting…</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { getAudioContext, warmupMicrophone } from "@/audio";

export default {
  name: "AudioSetup",
  data() {
    return {
      stage: "prompt-mic", // 'prompt-mic' | 'test-speaker' | 'done'
      busy: false,
      error: "",
      heard: false,
    };
  },
  methods: {
    async requestMic() {
      this.busy = true;
      this.error = "";
      try {
        await warmupMicrophone();
        this.stage = "test-speaker";
      } catch (e) {
        console.error(e);
        this.error = "You must allow microphone access to continue.";
      } finally {
        this.busy = false;
      }
    },
    async playTone() {
      // 1) Create & unlock the AudioContext right here in the click handler
      const audioCtx = getAudioContext();

      // 2) resume it (synchronously! do NOT await)
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      console.log("AudioContext state after resume:", audioCtx.state); // should be "running"

      // simple Web Audio beep
      const osc = audioCtx.createOscillator();
      osc.frequency.value = 440;

      // use an explicit gain node in case Safari defaults to zero volume
      const gain = audioCtx.createGain();
      gain.gain.setValueAtTime(0.05, audioCtx.currentTime);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start(audioCtx.currentTime);
      osc.stop(audioCtx.currentTime + 1);
      this._lastOsc = osc;
      // assume user hears it

      const ua = navigator.userAgent;
      const isSafari =
        ua.includes("Safari") &&
        !ua.includes("Chrome") &&
        !ua.includes("Chromium");
      const heardDelay = isSafari ? 4000 : 1000;

      setTimeout(() => {
        this.heard = true;
      }, heardDelay);
    },
    confirmSpeaker() {
      // Persist that audio is OK
      localStorage.setItem("audioOK", "1");
      this.stage = "done";
      setTimeout(() => this.$router.push("/pre-lobby"), 500);
    },
  },
};
</script>

<style scoped>
.error {
  color: #c0392b;
  margin-top: 0.5em;
}
</style>
