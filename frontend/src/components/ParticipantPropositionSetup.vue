<!--
Author: Jared Moore
Date: January, 2026
frontend/src/components/ParticipantPropositionSetup.vue
-->

<template>
  <div class="instructions-container">
    <main class="main-content">
      <div class="content-wrapper">
        <h1>Setup</h1>

        <p v-if="prompt" class="setup-prompt" v-html="prompt"></p>
        <p v-if="enabled" class="setup-progress">
          You have provided {{ completedCount }} of
          {{ requiredCount }} responses.
        </p>

        <div v-if="enabled && remainingCount > 0" class="setup-form">
          <label for="decisionInput" class="setup-label">
            Please enter one response:
          </label>
          <textarea
            id="decisionInput"
            v-model="decisionText"
            class="setup-textarea"
            rows="4"
            :disabled="loading"
          ></textarea>

          <div class="setup-actions">
            <button
              :disabled="loading || !decisionText"
              @click="submitDecision"
            >
              Submit
            </button>
          </div>
        </div>

        <div v-else class="setup-disabled"></div>
      </div>
    </main>

    <Popover
      v-if="popover"
      :show="true"
      :title="popover.title"
      :message="popover.message"
      @close="handlePopoverClose"
    />
  </div>
</template>

<script>
import { api } from "@/api";
import Popover from "@/components/round/PopoverVue.vue";

const MAX_ATTEMPTS = 4;

export default {
  name: "ParticipantPropositionSetup",
  components: { Popover },
  data() {
    return {
      participantId: localStorage.getItem("participantId"),
      enabled: false,
      prompt: "",
      requiredCount: 0,
      completedCount: 0,
      remainingCount: 0,
      decisionText: "",
      popover: null,
      loading: false,
      attempts: Number(
        localStorage.getItem("participantPropositionAttempts") || 0,
      ),
    };
  },
  computed: {},
  mounted() {
    this.fetchStatus();
  },
  methods: {
    async fetchStatus() {
      this.loading = true;
      try {
        console.log(
          "ParticipantPropositionSetup: required flag",
          api.participantPropositionsRequired(),
        );
        if (!api.participantPropositionsRequired()) {
          console.log(
            "ParticipantPropositionSetup: skipping setup, routing to lobby",
          );
          this.goToLobby();
          return;
        }
        const response = await api.participantPropositions(this.participantId);
        this.applyStatus(response);
      } catch (error) {
        console.error("Failed to load proposition status:", error.message);
        this.showError("Failed to load setup status. Please try again.");
      } finally {
        this.loading = false;
      }
    },
    applyStatus(response) {
      this.enabled = Boolean(response.enabled);
      this.prompt = response.prompt || "";
      this.requiredCount = response.required_count || 0;
      this.completedCount = response.completed_count || 0;
      this.remainingCount = response.remaining_count || 0;
      if (!this.enabled || this.remainingCount === 0) {
        this.clearAttempts();
        if (this.remainingCount === 0) {
          localStorage.setItem("participantPropositionsComplete", "true");
        }
        this.goToLobby();
        return;
      }
    },
    async submitDecision() {
      if (this.loading) return;
      if (this.attempts >= MAX_ATTEMPTS) {
        this.endStudy();
        return;
      }
      this.loading = true;
      this.popover = null;
      try {
        const response = await api.participantPropositions(
          this.participantId,
          this.decisionText,
        );
        if (response.status !== "ok") {
          this.recordAttempt();
          if (response.reason) {
            console.warn(
              "ParticipantPropositionSetup: submission rejected:",
              response.reason,
            );
          }
          this.showError(response.reason || "");
        } else {
          this.decisionText = "";
        }
        this.applyStatus(response);
        if (this.attempts >= MAX_ATTEMPTS) {
          this.endStudy();
        }
      } catch (error) {
        console.error("Decision submission failed:", error);
        this.recordAttempt();
        this.showError(
          "An error occurred while submitting your response. Please try again.",
        );
      } finally {
        this.loading = false;
      }
    },
    showError(message) {
      if (!message) {
        return;
      }
      this.popover = {
        title: "Please revise your response",
        message,
      };
    },
    handlePopoverClose() {
      this.popover = null;
    },
    recordAttempt() {
      this.attempts += 1;
      localStorage.setItem(
        "participantPropositionAttempts",
        String(this.attempts),
      );
    },
    clearAttempts() {
      this.attempts = 0;
      localStorage.removeItem("participantPropositionAttempts");
    },
    goToLobby() {
      this.$router.push("/lobby");
    },
    endStudy() {
      this.$router.push("/feedback");
    },
  },
};
</script>

<style scoped>
.setup-prompt {
  margin-bottom: 1rem;
}

.setup-progress {
  margin-bottom: 1rem;
  font-style: italic;
}

.setup-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.setup-label {
  font-weight: 600;
}

.setup-textarea {
  width: 100%;
  resize: vertical;
  padding: 0.75rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  font-size: 1rem;
}

.setup-actions {
  display: flex;
  gap: 0.75rem;
}

.setup-attempts {
  color: #444;
  font-size: 0.95rem;
}
</style>
