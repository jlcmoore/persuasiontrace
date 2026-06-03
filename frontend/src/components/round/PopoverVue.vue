<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/round/PopoverVue.vue
-->

<template>
  <div v-if="show" class="overlay">
    <div class="overlay-content">
      <h2>{{ title }}</h2>

      <div v-if="subtitle">
        <p class="subtitle">{{ subtitle }}</p>
      </div>

      <div v-if="message">
        <p>{{ message }}</p>
      </div>

      <div v-if="showSpinner" class="spinner" aria-label="Loading"></div>

      <!--Close button -->
      <div v-if="!blocking" id="button-container">
        <button @click="closeOverlay">{{ okayText }}</button>
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
    title: {
      type: String,
      required: true,
    },
    subtitle: {
      type: String,
      default: null,
    },
    message: {
      type: String,
      required: true,
    },
    blocking: {
      type: Boolean,
      default: false,
    },
    showSpinner: {
      type: Boolean,
      default: false,
    },
    okayText: {
      type: String,
      default: "Okay",
    },
  },
  emits: ["close"],
  data() {
    return {};
  },
  methods: {
    closeOverlay() {
      this.$emit("close");
    },
  },
};
</script>

<style scoped></style>
