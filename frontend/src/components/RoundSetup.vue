<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/RoundSetup.vue
-->

<template>
  <div class="round-setup-container">
    <main class="main-content">
      <div class="content-wrapper">
        <h1>Customize Your Round</h1>

        <form @submit.prevent="initialize">
          <div>
            <label>
              External (fake Prolific) Participant ID:
              <input v-model="prolificId" type="text" required />
            </label>
          </div>

          <div>
            <label>
              <input v-model="isTarget" type="radio" :value="true" />
              Play as Target
            </label>
            <label>
              <input v-model="isTarget" type="radio" :value="false" />
              Play as Persuader
            </label>
            <label>
              <input v-model="isTarget" type="radio" :value="null" />
              No Preference
            </label>
          </div>

          <div>
            <label>
              LLM Target:
              <input v-model="llmTarget" type="text" />
            </label>
          </div>

          <div>
            <label>
              LLM Persuader:
              <input v-model="llmPersuader" type="text" />
            </label>
          </div>

          <div>
            <label>
              Proposition ID:
              <input v-model="propositionID" type="text" />
            </label>
          </div>

          <div>
            <label>
              The Persuader should Support the Proposition
              <select v-model="persuaderSupportsProposition">
                <option :value="null">--</option>
                <option :value="true">Yes</option>
                <option :value="false">No</option>
              </select>
            </label>
          </div>

          <!-- Add more parameters as needed -->

          <button type="submit">Start Round</button>
        </form>

        <div v-if="errorMessage" class="error">
          {{ errorMessage }}
        </div>
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
      prolificId: null,
      isTarget: null,
      llmTarget: null,
      llmPersuader: null,
      propositionID: null,
      persuaderSupportsProposition: null,
      errorMessage: "",
      participantInitialized: false,
      participantIsReady: false,
    };
  },
  mounted() {
    // Extract query parameters.
    const query = this.$route.query;
    const queryProvided = Object.keys(query).length > 0;
    let autoProceed = false;

    if (query.prolificId) {
      this.prolificId = query.prolificId;
      autoProceed = true;
    }
    if (query.isTarget !== undefined) {
      if (query.isTarget === "true") this.isTarget = true;
      else if (query.isTarget === "false") this.isTarget = false;
      else this.isTarget = null;
      autoProceed = true;
    }

    if (query.llmTarget) {
      this.llmTarget = query.llmTarget;
      autoProceed = true;
    }
    if (query.llmPersuader) {
      this.llmPersuader = query.llmPersuader;
      autoProceed = true;
    }
    if (query.propositionID) {
      this.propositionID = query.propositionID;
      autoProceed = true;
    }

    if (query.persuaderSupportsProposition !== undefined) {
      if (query.persuaderSupportsProposition === "true")
        this.persuaderSupportsProposition = true;
      else if (query.persuaderSupportsProposition === "false")
        this.persuaderSupportsProposition = false;
      else this.persuaderSupportsProposition = null;
      autoProceed = true;
    }

    // If URL parameters were provided but no prolificId then auto generate one.
    if (queryProvided && (!this.prolificId || this.prolificId.trim() === "")) {
      this.prolificId = generateFakeID();
      autoProceed = true;
    }

    // If any relevant URL parameter was provided, use $nextTick to ensure all bindings are updated
    // before automatically calling initialize().
    if (autoProceed) {
      this.$nextTick(() => {
        this.initialize();
      });
    }
  },
  methods: {
    async initialize() {
      // Clear any previous session state
      localStorage.clear();

      // If somehow still missing, generate a new participant id before proceeding.
      if (!this.prolificId || this.prolificId.trim() === "") {
        this.prolificId = generateFakeID();
      }

      // Build the parameters object, ignoring empty values.
      const params = {};
      if (this.isTarget !== null) params.is_target = this.isTarget;
      if (this.llmTarget !== null && this.llmTarget.trim() !== "")
        params.llm_target = this.llmTarget;
      if (this.llmPersuader !== null && this.llmPersuader.trim() !== "")
        params.llm_persuader = this.llmPersuader;
      if (this.propositionID !== null && this.propositionID.trim() !== "")
        params.proposition = this.propositionID;
      if (this.persuaderSupportsProposition !== null)
        params.persuader_supports_proposition =
          this.persuaderSupportsProposition;

      // Store the participant ID and game parameters.
      localStorage.setItem("prolificId", this.prolificId);
      localStorage.setItem("current_round_params", JSON.stringify(params));

      const response = await api.initializeParticipant(this.prolificId);
      console.log("Participant initialized:", response);

      // Store participantId
      this.participantId = response.participant_id;
      localStorage.setItem("participantId", this.participantId);

      // Route to the lobby
      this.$router.push("/lobby");
    },
  },
};
</script>

<style scoped>
.round-setup-container {
  /* Add styling here */
}

.error {
  color: red;
  margin-top: 10px;
}
</style>
