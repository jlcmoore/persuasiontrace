// Author: Jared Moore
// Date: July, 2025
// frontend/src/api.js

import axios from "axios";

// Create an instance of axios to centralize common configurations
const apiClient = axios.create({
  baseURL: "/", // Base URL for your API
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 60000, // Set a timeout of 60 seconds
});

// Add an interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response, // Let successful responses pass through
  (error) => {
    // Check if the error is from the API
    if (error.response) {
      console.error(
        `API Error: ${error.response.status} - ${error.response.data.detail || error.response.statusText}`,
      );
      return Promise.reject({
        message:
          error.response.data.detail ||
          "An error occurred while processing the request.",
        status: error.response.status,
      });
    } else if (error.request) {
      // No response received from server
      console.error("Network Error: No response received from the server.");
      return Promise.reject({
        message:
          "No response from server. Please check your network connection.",
      });
    } else {
      // Something else went wrong
      console.error(`Error: ${error.message}`);
      return Promise.reject({
        message: "An unexpected error occurred.",
      });
    }
  },
);

const DEFAULT_MAX_MESSAGE_CHARS = 300;
const DEFAULT_MAX_AUDIO_SECONDS = 30;

export const api = {
  // Initialize a participant
  async initializeParticipant(participantId, turnstileToken = null) {
    try {
      const payload = { id: participantId };
      if (turnstileToken) {
        payload.turnstile_token = turnstileToken;
      }
      const response = await apiClient.post("/participant_init/", payload);
      return response.data;
    } catch (error) {
      console.error("Failed to initialize participant:", error.message);
      throw error; // Re-throw the error so the component can handle it
    }
  },

  // Mark participant as ready
  async participantReady(participantId) {
    console.log("participantReady(participantId=", participantId, ")");
    try {
      await apiClient.post("/participant_ready/", { id: participantId });
    } catch (error) {
      console.error("Failed to mark participant as ready:", error.message);
      throw error;
    }
  },

  // Get the current round information
  async getCurrentRound(participantId, params = {}) {
    try {
      const requestData = { participant_id: participantId, ...params };
      const response = await apiClient.post("/current_round/", requestData);
      return response.data;
    } catch (error) {
      console.error("Failed to get current round:", error.message);
      throw error;
    }
  },

  // get Participant rounds
  async getParticipantRounds(participantId) {
    try {
      const response = await apiClient.post("/participant_rounds/", {
        id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to get participant rounds:", error.message);
      throw error;
    }
  },

  // send feedback
  async sendFeedback(participantId, feedback) {
    if (feedback === undefined) {
      feedback = "";
    }
    try {
      await apiClient.post("/send_feedback/", {
        participant_id: participantId,
        feedback: feedback,
      });
    } catch (error) {
      console.error("Failed to send feedback:", error.message);
      throw error;
    }
  },

  // get high-level instructions
  async getParticipantInstructions(participantId) {
    try {
      const response = await apiClient.post("/participant_instructions/", {
        id: participantId,
      });
      return response.data;
    } catch (error) {
      console.error("Failed to get instructions:", error.message);
      throw error;
    }
  },

  async participantPropositions(participantId, decision = null) {
    try {
      const payload = { participant_id: participantId };
      if (decision !== null) {
        payload.decision = decision;
      }
      const response = await apiClient.post(
        "/participant_propositions/",
        payload,
      );
      return response.data;
    } catch (error) {
      console.error("Failed to submit proposition:", error.message);
      throw error;
    }
  },

  isDevelopmentMode() {
    return Boolean(window.SERVER_CONFIG?.development_mode);
  },

  maxAudioSeconds() {
    return (
      Number(window.SERVER_CONFIG?.max_audio_seconds) ||
      DEFAULT_MAX_AUDIO_SECONDS
    );
  },

  maxMessageChars() {
    return (
      Number(window.SERVER_CONFIG?.max_message_chars) ||
      DEFAULT_MAX_MESSAGE_CHARS
    );
  },

  mayUseAudio() {
    return Boolean(window.SERVER_CONFIG?.may_use_audio);
  },

  postPlayDelay() {
    if (this.isDevelopmentMode()) {
      return Boolean(window.SERVER_CONFIG?.post_play_delay) * 1000;
    } else {
      return Boolean(window.SERVER_CONFIG?.post_play_delay) * 1000;
    }
  },

  redirectMilliseconds() {
    if (this.isDevelopmentMode()) {
      return 0;
    } else {
      return 2000;
    }
  },

  participantPropositionsRequired() {
    return Boolean(window.SERVER_CONFIG?.participant_propositions_required);
  },
};
