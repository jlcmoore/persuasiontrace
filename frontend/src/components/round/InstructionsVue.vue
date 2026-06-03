<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/round/InstructionsVue.vue
-->

<template>
  <div class="overlay">
    <div class="overlay-content">
      <h2>Instructions</h2>

      <!-- Display the dynamically fetched instructions -->
      <div v-if="instructions" class="instructions" v-html="instructions"></div>
      <div v-else>Loading instructions...</div>

      <div class="button-container">
        <button :disabled="!instructions || !canProceed" @click="goToGame">
          {{ buttonText }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { api } from "@/api";

export default {
  props: {
    instructions: {
      type: String,
      required: true,
    },
  },
  emits: ["close"],
  data() {
    return {
      timeRemaining: 30,
      canProceed: false,
    };
  },
  computed: {
    buttonText() {
      if (!this.instructions) {
        return "Loading...";
      }
      return this.canProceed
        ? "Continue"
        : `Please wait ${this.timeRemaining} seconds`;
    },
  },
  mounted() {
    this.startTimer();
  },
  methods: {
    startTimer() {
      const timer = setInterval(() => {
        if (this.timeRemaining > 0 && !api.isDevelopmentMode()) {
          this.timeRemaining--;
        } else {
          this.canProceed = true;
          clearInterval(timer);
        }
      }, 1000);
    },
    goToGame() {
      if (this.instructions) {
        this.$emit("close");
      } else {
        console.log("Instructions not loaded yet.");
      }
    },
  },
};
</script>

<style scoped>
.instructions {
  text-align: left;
}
</style>
