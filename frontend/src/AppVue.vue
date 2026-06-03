<!--
Author: Jared Moore
Date: July, 2025
frontend/src/AppVue.vue
-->

<template>
  <div id="app">
    <header class="header">
      <h1>Conversation</h1>
    </header>
    <router-view />
  </div>
</template>

<script>
export default {
  name: "App",
};
</script>

<style>
body {
  font-family: "Source Sans Pro", sans-serif;
  margin: 0;
  padding: 0;
  font-weight: 300;
  font-size: 20px;
}

/* Basic reset for margin */
* {
  margin: 0;
  box-sizing: border-box;
}

/* Main container takes full screen height */
.container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header styles */
.header {
  width: 100%;
  background-color: #c2e9ec;
  /* Dark blue-gray */
  color: white;
  padding: 10px 0;
  text-align: left;
  padding-left: 25px;
  font-weight: bold;
}

/* Main content area */
.main-content {
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  /* background-color: #f5f5f5; */
  /* Light gray background */
}

.overlay-content {
  text-align: center;
}

/* Content wrapper to center the text and limit max width */
.content-wrapper {
  background-color: white;
  padding: 40px;
  /* border-radius: 8px; */
  /* box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); */
  max-width: 800px;
  width: 90%;
  text-align: center;
}

/* For overlays */

.overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.overlay-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
}

/* Typography */

h1 {
  font-weight: 300;
  color: #13494d;
  font-size: 1.6em;
}

h2 {
  font-weight: 500;
  margin-bottom: 16px;
  font-size: 1.4em;
}

p {
  margin-top: 6px;
  margin-bottom: 6px;
  line-height: 1.6;
  font-size: 1em;
  text-align: justify;
}

/* Button styles */

.button-container {
  display: flex;
  justify-content: center;
  gap: 2em;
  margin-top: 20px;
}

button {
  background-color: #c2e9ec;
  color: #12494d;
  border: none;
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #12494d;
  color: white;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

blockquote {
  padding: 0.5em 1em;
  border-left: 3px solid #666;
  background-color: #fafafa;
  font-size: 0.95em;
  color: #444;
  text-align: justify;
}

/* Waiting pane */

.spinner {
  display: inline-block;
  width: 1rem;
  height: 1rem;
  margin-right: 0.5rem;
  border: 2px solid #ccc;
  border-top-color: #333;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
