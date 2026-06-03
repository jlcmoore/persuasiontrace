<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/WelcomeVue.vue
 -->

<template>
  <div class="welcome-container">
    <!-- Main Content: Centered text with white background -->
    <main class="main-content">
      <div class="content-wrapper">
        <h1>Welcome</h1>
        <p>Thank you for participating in our experiment!</p>
        <p>
          You will have one or more short conversations with other participants.
          <span v-if="mayUseAudio">These may be textual or audio based.</span>
        </p>
        <div class="checkbox-group">
          <label v-if="mayUseAudio">
            <input v-model="quietConfirmed" type="checkbox" />
            I confirm I will be in a quiet area for this study and permit audio
            recording.
          </label>
          <label>
            <input v-model="aiConfirmed" type="checkbox" />
            I confirm I will not use any generative AI tools.
          </label>
        </div>
        <div v-if="showTurnstile" class="turnstile-container">
          <div
            class="cf-turnstile"
            :data-sitekey="effectiveTurnstileSiteKey"
            data-appearance="interaction-only"
            data-callback="onTurnstileSuccess"
            data-error-callback="onTurnstileError"
            data-expired-callback="onTurnstileExpired"
          ></div>
        </div>
        <button :disabled="!canContinue" @click="nextPage">Continue</button>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";
import { generateFakeID } from "@/utils";

export default {
  data() {
    return {
      quietConfirmed: false,
      aiConfirmed: false,
      mayUseAudio: api.mayUseAudio(),
      turnstileSiteKey: import.meta.env.VITE_TURNSTILE_SITE_KEY || "",
      turnstileToken: "",
      turnstileTestSiteKey: "1x00000000000000000000AA",
    };
  },
  computed: {
    effectiveTurnstileSiteKey() {
      if (api.isDevelopmentMode() && !this.turnstileSiteKey) {
        return this.turnstileTestSiteKey;
      }
      return this.turnstileSiteKey;
    },
    requiresTurnstile() {
      return Boolean(this.effectiveTurnstileSiteKey);
    },
    showTurnstile() {
      return this.requiresTurnstile;
    },
    canContinue() {
      if (api.isDevelopmentMode()) {
        return true;
      }
      if (api.mayUseAudio()) {
        return (
          this.quietConfirmed &&
          this.aiConfirmed &&
          (!this.requiresTurnstile || this.turnstileToken)
        );
      }
      return (
        this.aiConfirmed && (!this.requiresTurnstile || this.turnstileToken)
      );
    },
  },
  mounted() {
    if (this.requiresTurnstile) {
      this.registerTurnstileCallbacks();
      this.ensureTurnstileScript();
    }
  },
  methods: {
    ensureTurnstileScript() {
      const existing = document.querySelector(
        'script[data-turnstile-loader="true"]',
      );
      if (existing) {
        return;
      }
      const script = document.createElement("script");
      script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js";
      script.async = true;
      script.defer = true;
      script.dataset.turnstileLoader = "true";
      document.head.appendChild(script);
    },
    registerTurnstileCallbacks() {
      window.onTurnstileSuccess = (token) => {
        this.turnstileToken = token;
        localStorage.setItem("turnstileToken", token);
      };
      window.onTurnstileError = () => {
        this.turnstileToken = "";
        localStorage.removeItem("turnstileToken");
      };
      window.onTurnstileExpired = () => {
        this.turnstileToken = "";
        localStorage.removeItem("turnstileToken");
      };
    },
    async nextPage() {
      if (!this.canContinue) return;
      if (this.requiresTurnstile && !this.turnstileToken) {
        alert("Please complete the verification before continuing.");
        return;
      }

      localStorage.clear();
      let prolificId = new URLSearchParams(window.location.search).get(
        "PROLIFIC_PID",
      );
      // Generate an id for the participant if not from prolific
      prolificId = prolificId || generateFakeID();
      console.log("Prolific ID:", prolificId);
      localStorage.setItem("prolificId", prolificId);
      if (this.requiresTurnstile && this.turnstileToken) {
        localStorage.setItem("turnstileToken", this.turnstileToken);
      }
      try {
        // Initialize participant with survey responses
        const response = await api.initializeParticipant(
          prolificId,
          this.turnstileToken,
        );
        console.log("Participant initialized:", response);

        // Store participantId
        let participantId = response.participant_id;
        localStorage.setItem("participantId", participantId);

        // Only show the consent if not in dev mode
        if (api.isDevelopmentMode()) {
          this.$router.push("/pre-lobby");
        } else {
          this.$router.push("/consent");
        }
      } catch (error) {
        console.error("Failed to initialize:", error.message);
        alert("An error occurred while initializing. Please try again.");
      }
    },
  },
};
</script>

<style scoped>
.checkbox-group {
  margin: 1em 0;
}

.checkbox-group label {
  display: block;
  margin-bottom: 0.5em;
  font-weight: normal;
}

.turnstile-container {
  margin: 1em 0;
}
</style>
