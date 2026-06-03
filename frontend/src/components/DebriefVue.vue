<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/DebriefVue.vue
-->

<template>
  <div class="survey-container">
    <main class="main-content">
      <div class="content-wrapper">
        <h1>Debrief</h1>

        <p>You have completed the experiment.</p>

        <p>
          You interacted with {{ humanConversations }} real human participants
          during the experiment. You interacted with
          {{ totalRounds - humanConversations }} non-human participants.
        </p>

        <p>Thank you for your participation.</p>

        <p>
          Click the button below to be redirected to Prolific and verify your
          completion.
        </p>

        <p v-if="loadingRoundsData">Loading completion details...</p>
        <p v-else-if="roundsDataError" class="error-text">{{ roundsDataError }}</p>

        <p v-if="completionCode" class="completion-code">
          Completion code: <strong>{{ completionCode }}</strong>
        </p>
        <p v-if="completionCode">
          If redirect fails, use this link directly:
          <a :href="prolificCompletionUrl">{{ prolificCompletionUrl }}</a>
        </p>

        <button
          :disabled="loadingRoundsData || !completionCode"
          @click="sendToProlific"
        >
          Send to Prolific
        </button>
      </div>
    </main>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  data() {
    return {
      participantId: localStorage.getItem("participantId"), // Get the participant ID from local storage
      humanConversations: null,
      totalRounds: null,
      completionCode: null,
      loadingRoundsData: true,
      roundsDataError: "",
    };
  },
  computed: {
    prolificCompletionUrl() {
      const code = encodeURIComponent(this.completionCode || "");
      return `https://app.prolific.co/submissions/complete?cc=${code}`;
    },
  },
  mounted() {
    this.getRoundsData();
  },
  methods: {
    async sendToProlific() {
      this.roundsDataError = "";
      if (!this.completionCode) {
        await this.getRoundsData();
      }
      if (this.completionCode) {
        window.location.href = this.prolificCompletionUrl;
      } else {
        this.roundsDataError =
          "Could not load completion code. Please refresh and try again. If this keeps happening, message the researcher with your Prolific ID.";
      }
    },
    async getRoundsData() {
      this.loadingRoundsData = true;
      try {
        // Get rounds data
        const completionData = await api.getParticipantRounds(
          this.participantId,
        );
        console.log("Rounds data:", completionData);
        this.humanConversations = completionData.num_human_conversations;
        this.totalRounds = completionData.total_rounds;
        this.completionCode = completionData.completion_code;
        if (!this.completionCode) {
          this.roundsDataError =
            "Completion code is unavailable right now. Please refresh and try again.";
        }
      } catch (error) {
        this.roundsDataError = `Error loading completion details: ${error.message}`;
      } finally {
        this.loadingRoundsData = false;
      }
    },
  },
};
</script>

<style scoped>
.error-text {
  color: #8b0000;
}

.completion-code {
  margin-top: 10px;
}
</style>
