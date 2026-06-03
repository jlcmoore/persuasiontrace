<template>
  <div v-if="show" class="overlay">
    <div class="overlay-content">
      <div id="result-container">
        <div v-if="status == 'waiting'">
          <h2>Result</h2>
          <p>Waiting for the result...</p>
        </div>
        <div v-else>
          <div v-html="result"></div>
          <button class="continue-btn" @click="nextRound">Continue</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    show: {
      type: Boolean,
      required: true,
    },
    status: {
      type: String,
      required: true,
    },
    result: {
      type: String,
      required: true,
    },
  },
  emits: ["roundOver"],
  methods: {
    nextRound() {
      this.$emit("roundOver");
    },
  },
};
</script>

<style scoped>
#result-container {
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
}

.continue-btn {
  margin-left: auto;
}

.success h2 {
  color: #4caf50;
}

.failure h2 {
  color: #f44336;
}

.results-subtitle {
  font-size: 1.1em;
  margin-bottom: 20px;
  color: #444444;
  font-style: italic;
}
</style>
