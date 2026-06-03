<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/round/RoundVue.vue
-->

<template>
  <div class="round-container">
    <div v-if="!showInstructionsPending">
      <div class="round-info">
        <span>Round: {{ numRounds }} of {{ totalRounds }}</span>
        <span v-if="!unlimitedTurns"
          >Turns remaining: {{ turnsRemaining }}</span
        >
        <span v-if="roundTimeLimit !== null"
          >Time remaining: {{ timeRemaining }}</span
        >
      </div>

      <!-- Always show the prompt -->
      <div
        class="prompt-panel"
        :class="promptPhaseClass"
        aria-live="polite"
        aria-atomic="true"
      >
        <div class="prompt-phase-badge">{{ promptPhaseLabel }}</div>
        <div
          v-if="showSurveyPromptOverride"
          class="prompt-body prompt-body-survey"
        >
          <p class="survey-prompt-intro">
            {{ surveyPromptIntroLine }}
          </p>
          <blockquote>
            <p>{{ surveyPromptText }}</p>
          </blockquote>
        </div>
        <div v-else class="prompt-body" v-html="displayedPrompt"></div>
      </div>
    </div>
    <main class="main-content">
      <div v-if="showChat" id="conversation">
        <div v-if="useAudio" class="audio-buttons">
          <div class="audio-status-slot">
            <div v-if="isRecording" class="recording-indicator">
              <span class="red-dot"></span>
              Recording...
              <span class="recording-timer">{{ recordingTimeRemaining }}</span>
            </div>

            <div v-else-if="waiting" class="waiting-panel">
              <span class="spinner"></span>
              Waiting for the other participant...
            </div>

            <!-- Placeholder to reserve space when neither is visible -->
            <div v-else class="status-placeholder" aria-hidden="true"></div>
          </div>
          <div class="button-container">
            <!-- NB: just use messageHighlightPending to track the messages. assume (and enforce) that it is only true when messageHighlight is as well -->
            <button
              :disabled="
                !isRecording || serialDecisionPending || messageHighlightPending
              "
              @click="stopRecordingAndSend"
            >
              Stop recording & Send
            </button>
          </div>
        </div>

        <!-- 1) Chat area (both roles see this) -->
        <Chat
          v-if="!useAudio || (useAudio && showTranscript)"
          :chat-messages="chatMessages"
          :new-message="newMessage"
          :max-message-chars="maxMessageChars"
          :can-send-message-exterior="canSendMessage"
          :typing="typing && turnsRemaining > 0"
          :is-target="isTarget"
          :continuous-measure-pending="
            serialDecisionPending || messageHighlightPending
          "
          :highlight-enabled="messageHighlights && isTarget"
          :allow-input="allowChatInput"
          @update:new-message="newMessage = $event"
          @send-message="handleSendMessage"
          @add-highlight="handleMessageHighlight"
        />
        <div v-if="showHighlightPrompt" class="highlight-prompt">
          <p v-for="(line, idx) in highlightPromptLines" :key="idx">
            {{ line }}
          </p>
          <div class="highlight-actions">
            <button
              type="button"
              class="highlight-action-button"
              :disabled="!hasAnyHighlights"
              @click="clearCurrentHighlights"
            >
              Clear highlights
            </button>
            <button
              type="button"
              class="highlight-action-button highlight-continue"
              :disabled="continueDisabled"
              @click="continueAfterHighlight"
            >
              Continue
            </button>
          </div>
        </div>
      </div>

      <!-- TODO: make these slider panels components -->
      <div v-if="showDecidePanel" class="decision-panel">
        <div v-if="mouseTrace">
          <!-- Instructions -->
          <div class="mouse-trace-instructions">
            <p>
              (<em
                >Use your cursor to show your agreement with the
                proposition.</em
              >)
            </p>
            <!-- <div v-if="targetEndedRound">
              <p>You have ended the round. The audio will finish playing.</p>
            </div>
            <div v-else aria-hidden="true">
              <p>&nbsp;</p>
            </div> -->
          </div>

          <div>
            <!-- percentage readout -->
            <div class="slider-percent">{{ Math.round(cursorPosition) }}%</div>

            <div class="mouse-trace-wrapper">
              <!-- outside label -->
              <div
                class="mouse-trace-panel"
                :class="{
                  locked:
                    targetEndedRound ||
                    (showDecideStartPrompt && !cursorNearStartAnchor),
                }"
                @mousemove="handleMouseMove"
                @mouseleave="onMouseLeave"
                @mouseenter="onMouseEnter"
                @click="clickMouseTracePanel"
              >
                <span class="trace-label disagree">Disagree</span>
                <template v-if="showDecideStartPrompt">
                  <div
                    class="start-prompt"
                    aria-live="polite"
                    aria-label="Move cursor to your previous belief to start"
                  >
                    <span>Move cursor to your previous belief to start</span>
                    <!-- Dot indicates the segment's start anchor; only for mouse-trace segments before playback -->
                    <span
                      :style="{ left: startAnchorPercent + '%' }"
                      class="cursor-dot"
                      aria-hidden="true"
                    ></span>
                  </div>
                </template>

                <template v-else>
                  <!-- Fake cursor line -->
                  <div
                    class="mouse-trace-cursor"
                    :style="{ left: cursorPosition + '%' }"
                  />
                </template>
                <!-- Right label -->
                <span class="trace-label agree"> Agree </span>
              </div>
            </div>

            <!-- End Round button (same width as slider) -->
            <div
              v-if="
                targetCanEndRound &&
                !(messageHighlights && messageHighlightPending)
              "
              class="mouse-trace-wrapper"
            >
              <button
                class="primary-slider-button slider-width-button"
                :disabled="targetEndedRound || waiting"
                @click="targetEndsRound"
              >
                End Round
              </button>
            </div>
          </div>
        </div>
        <div
          v-else-if="
            targetCanEndRound && !(messageHighlights && messageHighlightPending)
          "
        >
          <button @click="targetEndsRound">End Round</button>
        </div>
      </div>

      <!-- 2) Inline slider panel for the target -->
      <div v-if="showQuestionPanel" class="belief-slider-panel">
        <label>
          On a scale from 0 (completely disagree) to 100 (completely agree):
          <br />
          {{ beliefQuestionPromptLine }}
          <template v-if="beliefQuestionStatement">
            <br />
            <strong>{{ beliefQuestionStatement }}</strong>
          </template>
          <br />
          (<em>
            Click on the slider to proceed
            <template v-if="serialQuestionsSentence">
              to the next sentence
            </template>
            . </em
          >)
        </label>
        <div v-if="beliefSurveyProgress" class="belief-survey-progress">
          {{ beliefSurveyProgress }}
        </div>

        <div class="slider-percent">
          {{ Math.round(beliefCursorPosition) }}%
        </div>

        <div class="mouse-trace-wrapper belief">
          <div
            class="mouse-trace-panel"
            :class="{ locked: beliefSelected !== null }"
            @mousemove="handleBeliefMouseMove"
            @mouseenter="unlockBeliefPanel"
            @click="handleBeliefClick"
          >
            <span class="trace-label disagree">Disagree</span>
            <div
              class="mouse-trace-cursor"
              :style="{ left: beliefCursorPosition + '%' }"
            />
            <span class="trace-label agree">Agree</span>
          </div>
        </div>

        <div class="mouse-trace-wrapper">
          <button
            :disabled="beliefSelected === null"
            class="primary-slider-button slider-width-button"
            @click="confirmBelief"
          >
            <template v-if="initialDecisionPending"
              >Confirm Initial Belief</template
            >
            <template v-else-if="finalDecisionPending"
              >Confirm Final Belief</template
            >
            <template v-else>Confirm Next Belief</template>
          </button>
        </div>
      </div>

      <!-- Popover Component -->
      <Popover
        v-if="popover"
        :show="true"
        :title="popover.title"
        :subtitle="popover.subtitle"
        :message="popover.message"
        :okay-text="popover.okayText"
        @close="handlePopoverClose"
      />

      <Popover
        v-if="showPromptLoading"
        :show="true"
        title="Loading prompt"
        message="Waiting for the task to load..."
        :blocking="true"
        :show-spinner="true"
      />

      <RoundResult
        v-if="showRoundResult"
        :show="showRoundResult"
        :status="resultStatus"
        :result="roundResultMessage"
        @round-over="roundOver"
      />

      <InstructionsVue
        v-if="showInstructionsPending"
        :instructions="instructions"
        @close="okayInstructions"
      />
    </main>
  </div>
</template>

<script>
import Chat from "./ChatVue.vue";
import Popover from "./PopoverVue.vue";
import RoundResult from "./RoundResult.vue";
import InstructionsVue from "./InstructionsVue.vue";

import { api } from "@/api";
import { testAssert, formatMMSS } from "@/utils";
import {
  playBase64Audio,
  getWarmStream,
  getAudioContext,
  audioContextSetup,
} from "@/audio";
import {
  addRangeToList,
  isRangeFullyCovered,
  normalizeHighlight,
  normalizeRange,
  normalizeRangesForLength,
  subtractRangeList,
} from "./highlightUtils";

// Require highlights when the target changes belief by >=5 percentage points.
const ON_REFLECTION_DELTA_THRESHOLD = 0.05;

export default {
  components: { Chat, Popover, RoundResult, InstructionsVue },
  // Return the reactive state tree for the round view.
  data() {
    return {
      participantId: localStorage.getItem("participantId"),
      instructions: localStorage.getItem("instructions"),
      showInstructionsPending: false,

      /* Round info */
      currentRoundId: null,
      numRounds: 1,
      totalRounds: null,
      turnsRemaining: null,
      prompt: "",
      promptDuringRound: "",
      maxMessageChars: api.maxMessageChars(),
      serialQuestions: false,
      serialQuestionsSentence: false,
      mouseTrace: false,
      unlimitedTurns: false,
      showTranscript: false,
      targetCanEndRound: false,

      // on reflection
      onReflection: false,
      onReflectionPending: false,
      onReflectionHighlights: [],
      onReflectionSubmitted: false,

      // Timer
      roundTimeLimit: null,
      timeRemainingSec: null,
      timerInterval: null,
      roundTimedOut: false,

      // Mouse Trace state
      cursorPosition: 50, // percent from left; init at center
      recenterCursor: false, // last location vs. fixed center -- TODO: not tested fully
      mouseTraceBuffer: [], // [{ position: 0–100, timestamp: ms since audio start }, ...]
      mouseTraceStartTime: null, // performance.now() when audio begins

      timeWhenPaused: null, // The timestamp at which the last pause occurred
      sumOfPauses: 0, // The duration of the all of the pauses (for the current trace)

      /* TEXT-REVEAL (variable speed) */
      revealTimer: null, // id returned by setTimeout
      revealMsgObj: null, // currently playing message object
      revealSchedule: [], // [{word, delay}, ...] for that message
      revealIdx: 0, // next entry in revealSchedule
      textPausedByLeave: false,
      textStartedTriggered: false,
      playTextPending: null,
      revealStartMs: null, // when current word-timer was started
      revealDelayMs: 0, // full delay for current word
      revealRemainingMs: 0, // how much time is still left

      /* Role */
      isTarget: false,

      /* Slider state */
      initialDecisionPending: false,
      initialTargetBelief: null,
      initialNodeBeliefs: {},
      finalTargetBelief: null,
      finalNodeBeliefs: {},
      finalDecisionPending: false,
      beliefSurveyEnabled: false,
      beliefSurveyItems: [],
      beliefSurveyPhase: null,
      beliefSurveyPhaseItemsCurrent: [],
      beliefSurveyQueue: [],
      beliefSurveyResponses: {},
      beliefSurveyCurrentItem: null,
      // For the serial questions
      serialTargetBelief: null,
      // This next one just in case the message
      // is flagged
      lastSerialTargetBelief: null,
      serialDecisionPending: false,

      /* message highlight state */
      messageHighlights: false,
      messageHighlightPending: false,
      messageHighlightPendingIndex: null,
      messageHighlightRanges: {},
      messageHighlightQueue: {},
      persistMessageHighlights: false, // Toggle to retain per-message highlights across turns

      serialSentenceQueue: [],
      serialSentenceResponsesPending: [],
      lastSerialSentenceResponses: null,
      currentSerialSentence: null,
      serialSentenceActiveMessageIndex: null,
      serialSentenceFullText: "",
      // Shared state btw all sliders
      beliefCursorPosition: 50, // percentage [0–100]
      beliefSelected: null, // once they click, we store the chosen % here
      beliefLocked: false,

      /* Chat state */
      chatMessages: [],
      newMessage: "",
      waiting: false,
      typing: false,
      ws: null,
      // buffer for response messages that arrive too early
      pendingResponses: [],
      targetEndedRound: false, // has made choice yet?
      roundEnded: false,

      // audio state
      mediaRecorder: null,
      recordedChunks: [],
      isPlaying: false,
      isRecording: false,
      audioPermitted: false,
      useAudio: false,
      // popover state
      popover: null,
      recordingTimeout: null,
      warmupAudioPending: false,

      recordingRemainingSec: null,
      recordingInterval: null,

      // Mouse trace audio state
      lastAudioReceived: null,
      audioStartedTriggered: false, // set once we begin playback
      playAudioPending: null,
      playThreshold: 5, // +-5% around last mouse trace
      audioPausedByLeave: false, // have we paused it on leave?
      // Anchor for next mouse-trace segment; null means use initial belief
      lastTraceAnchorPercent: null,

      waitingForAcknowledgeFailed: false,

      // Round Result
      roundResultReady: false,
      resultStatus: "",
      roundResultMessage: "",
      proposition: "",
      mainPropositionText: "",
      discussionPropositionText: "",
      awaitingControlPromptAcknowledgement: false,
      controlPromptTransitionAcknowledged: false,
    };
  },
  computed: {
    // Initial belief as a percentage [0–100]
    initialBeliefPercent() {
      const val = this.initialTargetBelief;
      if (typeof val === "number" && !Number.isNaN(val)) {
        const clamped = Math.min(1, Math.max(0, val));
        return clamped * 100;
      }
      return 50;
    },

    // Anchor used for gating each mouse-trace segment:
    // - First segment: initial belief
    // - Subsequent segments: last cursor position from previous segment
    startAnchorPercent() {
      if (
        typeof this.lastTraceAnchorPercent === "number" &&
        !Number.isNaN(this.lastTraceAnchorPercent)
      ) {
        return Math.min(100, Math.max(0, this.lastTraceAnchorPercent));
      }
      return this.initialBeliefPercent;
    },

    // Whether the current cursor is within threshold of the start anchor
    cursorNearStartAnchor() {
      return (
        Math.abs(this.cursorPosition - this.startAnchorPercent) <=
        this.playThreshold
      );
    },

    // Report whether audio or text playback is waiting to start.
    msgPending() {
      return (
        (this.playAudioPending && !this.audioStartedTriggered) ||
        (this.playTextPending && !this.textStartedTriggered)
      );
    },

    // Decide if the chat panel should be visible.
    showChat() {
      return (
        !this.showInstructionsPending &&
        !(
          this.isTarget &&
          (this.initialDecisionPending || this.finalDecisionPending)
        )
      );
    },

    // Show the round result only after it becomes ready.
    showRoundResult() {
      return this.roundResultReady && !this.onReflectionPending;
    },

    // Choose which prompt copy should be displayed to players.
    displayedPrompt() {
      // Show the full prompt only for initial or final question,
      // otherwise show the during-round version (fallback to full prompt if null)
      if (this.initialDecisionPending || this.finalDecisionPending) {
        return this.prompt;
      }
      return this.promptDuringRound || this.prompt;
    },

    // Phase of the prompt for clearer UI badges
    promptPhase() {
      if (this.initialDecisionPending) return "initial";
      if (this.finalDecisionPending) return "final";
      // if a distinct during-round prompt exists, call out control dialogue
      if (this.promptDuringRound && this.promptDuringRound !== this.prompt) {
        return "control";
      }
      return "discussion";
    },

    // Map the current prompt phase to a badge label.
    promptPhaseLabel() {
      switch (this.promptPhase) {
        case "initial":
          return "Proposition";
        case "final":
          return "Proposition";
        case "control":
          return "Discussion Proposition";
        default:
          return "Proposition";
      }
    },

    // Build the CSS class flags for the prompt phase badge.
    promptPhaseClass() {
      return {
        "phase-initial": this.promptPhase === "initial",
        "phase-final": this.promptPhase === "final",
        "phase-control": this.promptPhase === "control",
        "phase-discussion": this.promptPhase === "discussion",
        "phase-target": this.isTarget,
      };
    },

    showSurveyPromptOverride() {
      return Boolean(
        (this.initialDecisionPending || this.finalDecisionPending) &&
          this.beliefSurveyEnabled &&
          this.beliefSurveyCurrentItem &&
          this.beliefSurveyCurrentItem.text,
      );
    },

    surveyPromptIntroLine() {
      const phaseItems = this.beliefSurveyPhaseItemsCurrent;
      const total = Array.isArray(phaseItems) ? phaseItems.length : 0;
      if (!total || !this.beliefSurveyCurrentItem) {
        return "Please consider the following proposition:";
      }
      const index = phaseItems.findIndex(
        (item) => item.id === this.beliefSurveyCurrentItem.id,
      );
      const oneBased = index >= 0 ? index + 1 : 1;
      const phaseLabel =
        this.beliefSurveyPhase === "final" ? "Post-survey" : "Pre-survey";
      return `Please consider the following proposition (${phaseLabel} ${oneBased} of ${total}):`;
    },

    surveyPromptText() {
      if (!this.showSurveyPromptOverride || !this.beliefSurveyCurrentItem) {
        return "";
      }
      if (this.beliefSurveyCurrentItem.id === "Target") {
        return (
          this.mainPropositionText || this.beliefSurveyCurrentItem.text || ""
        );
      }
      return this.beliefSurveyCurrentItem.text || "";
    },

    // Display the prompt-loading popover until questions are ready.
    showPromptLoading() {
      return (
        !this.prompt && !this.showInstructionsPending && !this.showQuestionPanel
      );
    },

    // Require highlights when the belief change exceeds the threshold.
    onReflectionHighlightRequired() {
      if (!this.onReflection || !this.isTarget) {
        return false;
      }
      const delta = Math.abs(this.initialTargetBelief - this.finalTargetBelief);
      return delta >= ON_REFLECTION_DELTA_THRESHOLD;
    },

    // Disable the reflection submit button until requirements are met.
    isOnReflectionSubmitDisabled() {
      if (this.onReflectionSubmitted) {
        return true;
      }
      return (
        this.onReflectionHighlightRequired &&
        this.onReflectionHighlights.length === 0
      );
    },

    // Determine whether the belief question slider should be shown.
    showQuestionPanel() {
      return (
        !this.showInstructionsPending &&
        !this.isPlaying &&
        this.isTarget &&
        (this.initialDecisionPending ||
          this.finalDecisionPending ||
          this.serialDecisionPending)
      );
    },

    beliefQuestionPromptLine() {
      if (
        (this.initialDecisionPending || this.finalDecisionPending) &&
        this.beliefSurveyCurrentItem
      ) {
        return "How much do you agree with the proposition shown above?";
      }
      if (this.serialQuestionsSentence && this.currentSerialSentence) {
        return "How much do you agree with the sentence currently shown in chat?";
      }
      return "How much do you agree with the proposition shown above?";
    },

    beliefQuestionStatement() {
      if (this.serialQuestionsSentence && this.currentSerialSentence) {
        return this.currentSerialSentence;
      }
      return "";
    },

    beliefSurveyProgress() {
      if (
        !(this.initialDecisionPending || this.finalDecisionPending) ||
        !this.beliefSurveyEnabled
      ) {
        return "";
      }
      const phaseItems = this.beliefSurveyPhaseItemsCurrent;
      if (!phaseItems.length || !this.beliefSurveyCurrentItem) {
        return "";
      }
      const currentIndex = phaseItems.findIndex(
        (item) => item.id === this.beliefSurveyCurrentItem.id,
      );
      if (currentIndex < 0) {
        return "";
      }
      const phaseLabel =
        this.beliefSurveyPhase === "initial" ? "Pre-survey" : "Post-survey";
      return `${phaseLabel}: ${currentIndex + 1} of ${phaseItems.length}`;
    },

    // TODO: should later have a timer to trigger this only after the ppt
    // has had enough time to read the last message but only for !this.useAudio
    showDecidePanel() {
      if (this.onReflectionPending) {
        return false;
      }
      let show =
        this.isTarget &&
        !this.showQuestionPanel &&
        !this.waiting &&
        !this.showInstructionsPending;
      if (this.mouseTrace) {
        show &= this.isPlaying;
      } else {
        show &= !this.isPlaying;
      }
      return show;
    },

    // Ask the participant to return to the anchor before starting playback.
    showDecideStartPrompt() {
      return (
        (this.playAudioPending &&
          (!this.audioStartedTriggered || this.audioPausedByLeave)) ||
        (this.playTextPending &&
          (!this.textStartedTriggered || this.textPausedByLeave))
      );
    },

    // Track whether the round has entered the wrap-up flow.
    chatEndStarted() {
      // If we have already called chat over
      return this.finalDecisionPending || this.roundResultReady;
    },

    // Gate the ability to send messages based on round state.
    canSendMessage() {
      const highlightUnlocked =
        !this.messageHighlights || !this.messageHighlightPending;
      return (
        !this.chatEndStarted &&
        !(this.showDecidePanel && this.mouseTrace) &&
        !this.showQuestionPanel &&
        !this.showPromptLoading &&
        !this.onReflectionPending &&
        highlightUnlocked &&
        (this.turnsRemaining === null || this.turnsRemaining > 0)
      );
    },

    // Control whether the chat input field should be enabled.
    allowChatInput() {
      return !(
        (this.mouseTrace && !this.canSendMessage) ||
        this.showTranscript ||
        this.serialDecisionPending ||
        this.onReflectionPending ||
        (this.messageHighlights && this.messageHighlightPending)
      );
    },

    showHighlightPrompt() {
      // Render the shared highlight guidance while the participant is in a highlight phase.
      return (
        this.isTarget &&
        (this.messageHighlightPending || this.onReflectionPending)
      );
    },

    highlightPromptLines() {
      // Provide instructional copy (single line during the round, two lines during reflection).
      return [
        "Highlight anything in the conversation that influenced you before continuing, or select Continue if nothing stood out.",
      ];
    },

    hasAnyHighlights() {
      // Determine whether any message currently stores highlight ranges.
      if (!Array.isArray(this.chatMessages)) {
        return false;
      }
      return this.chatMessages.some((message) => {
        if (!message || typeof message !== "object") {
          return false;
        }
        const ranges = Array.isArray(message.highlightRanges)
          ? message.highlightRanges
          : message.highlightRange
            ? [message.highlightRange]
            : [];
        return ranges.length > 0;
      });
    },

    continueDisabled() {
      // During reflection, enforce the minimum highlight requirement before allowing Continue.
      if (this.onReflectionPending) {
        return this.isOnReflectionSubmitDisabled;
      }
      return false;
    },

    // Returns "MM:SS"
    timeRemaining() {
      return formatMMSS(this.timeRemainingSec);
    },

    // Format the remaining recording time for display.
    recordingTimeRemaining() {
      return formatMMSS(this.recordingRemainingSec);
    },
  },

  watch: {
    // Prompt the target when the final decision step begins.
    finalDecisionPending(newVal) {
      if (newVal && this.isTarget) {
        this.startBeliefSurveyPhase("final");
        let finalChoiceMessage =
          "Please answer the same set of propositions again. The order is randomized.";
        if (!this.beliefSurveyEnabled && this.hasControlPromptTransition()) {
          finalChoiceMessage =
            "The discussion proposition has ended. Please answer the original proposition shown above.";
        }
        // Nudge the participant to refocus on the proposition set.
        this.popover = {
          title: "Final Choice",
          message: finalChoiceMessage,
        };
        // Attempt to bring the prompt into view
        this.scrollIntoViewAsync(".prompt-panel", "start");
      }
    },
    onReflectionPending(newVal) {
      // Enter or exit reflection highlight mode when the review step toggles.
      if (newVal && this.isTarget) {
        this.enterReflectionHighlightMode();
      } else if (!newVal && this.persistMessageHighlights) {
        this.exitReflectionHighlightMode();
      }
    },
  },

  // Initialize the round once the component mounts.
  async mounted() {
    await this.initRound();
  },

  // Clean up timers, sockets, and intervals before unmounting.
  beforeUnmount() {
    if (this.ws) {
      this.ws.close();
    }
    if (this.typingTimeout) {
      clearTimeout(this.typingTimeout);
    }
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
    }
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
    }
    if (this.revealTimer) {
      clearTimeout(this.revealTimer);
    }
  },

  methods: {
    /* ===================================================================
     * Round Lifecycle & Flow
     * =================================================================== */

    // Handle round initialization failures by notifying the participant.
    roundError(error) {
      console.error("Error initializing round:", error);
      this.popover = {
        title: "Error",
        message: "Failed to initialize round. Returning to Lobby...",
      };
      setTimeout(() => this.$router.push("/lobby"), api.redirectMilliseconds());
    },
    // Load participant-specific instructions from the server.
    async fetchInstructions() {
      this.showInstructionsPending = true;
      try {
        const instructionsResponse = await api.getParticipantInstructions(
          this.participantId,
        );
        localStorage.setItem("instructions", instructionsResponse);
        // Assuming the instructions are returned as HTML string
        this.instructions = instructionsResponse;
        console.log(instructionsResponse);
        // Start the timer when component is mounted
      } catch (error) {
        console.error("Failed to fetch instructions:", error.message);
      }
    },
    // Fetch round data and prime state for the participant.
    async initRound() {
      this.loading = true;
      this.resetOnReflectionState();
      this.resetMessageHighlightState();
      this.persistMessageHighlights = false;
      try {
        // Get round info, unchanged:
        let params = null;
        if (api.isDevelopmentMode()) {
          const raw = localStorage.getItem("current_round_params");
          params = raw ? JSON.parse(raw) : {};
        }

        let roundData;
        try {
          roundData = await api.getCurrentRound(this.participantId, params);
        } catch {
          // If there is an error in the round send them back to the lobby
          this.$router.push("/lobby");
          return;
        }

        console.log("round data", roundData);
        this.isTarget = roundData.is_target;
        this.prompt = roundData.prompt;
        this.promptDuringRound = roundData.prompt_during_round ?? this.prompt;
        this.mainPropositionText = this.extractPromptPropositionText(
          this.prompt,
        );
        this.discussionPropositionText = this.extractPromptPropositionText(
          this.promptDuringRound,
        );
        this.awaitingControlPromptAcknowledgement = false;
        this.controlPromptTransitionAcknowledged = false;

        this.currentRoundId = roundData.round_id;

        this.onReflection = roundData.on_reflection;

        // Reset continuous measure toggles before reassigning
        this.mouseTrace = false;
        this.serialQuestions = false;
        this.serialQuestionsSentence = false;
        this.messageHighlights = false;

        // Only set the continuous measures if this is the target
        if (this.isTarget) {
          const measure = roundData.continuous_measure;
          this.mouseTrace = measure === "mouse-trace";
          this.serialQuestions = measure === "serial-questions";
          this.serialQuestionsSentence =
            measure === "serial-questions-sentence";
          this.messageHighlights = measure === "message-highlights";

          const enabledMeasures = [
            this.mouseTrace,
            this.serialQuestions,
            this.serialQuestionsSentence,
            this.messageHighlights,
          ].filter(Boolean).length;
          testAssert(
            enabledMeasures <= 1,
            "Cannot have two continuous measures",
          );

          this.resetSerialSentenceState();
          this.resetMessageHighlightState();
        }

        if (!this.serialQuestionsSentence) {
          this.resetSerialSentenceState();
        }
        if (!this.messageHighlights) {
          this.resetMessageHighlightState();
        }

        this.useAudio = roundData.use_audio;
        this.showTranscript = roundData.show_transcript;

        // Initialize state to what the server reports
        this.initialTargetBelief = roundData.target_initial_belief;
        this.finalTargetBelief = roundData.target_final_belief;
        const beliefSurvey = roundData.belief_survey || {};
        const rawSurveyItems = Array.isArray(beliefSurvey.items)
          ? beliefSurvey.items
          : [];
        this.beliefSurveyItems = rawSurveyItems
          .map((item) => ({
            id: String(item?.id || "").trim(),
            text: String(item?.text || "").trim(),
          }))
          .filter((item) => item.id && item.text);
        this.beliefSurveyEnabled = Boolean(
          this.isTarget &&
            beliefSurvey.enabled &&
            Array.isArray(this.beliefSurveyItems) &&
            this.beliefSurveyItems.length > 0,
        );
        this.initialNodeBeliefs =
          beliefSurvey &&
          typeof beliefSurvey.initial_node_beliefs === "object" &&
          beliefSurvey.initial_node_beliefs !== null
            ? { ...beliefSurvey.initial_node_beliefs }
            : {};
        this.finalNodeBeliefs =
          beliefSurvey &&
          typeof beliefSurvey.final_node_beliefs === "object" &&
          beliefSurvey.final_node_beliefs !== null
            ? { ...beliefSurvey.final_node_beliefs }
            : {};
        this.resetBeliefSurveyPhase();
        this.chatMessages = roundData.messages; // does not include the ultimate msg
        this.syncOnReflectionHighlights();

        const last_msg_received = roundData.last_msg_received;
        // Buffer any previously received message
        if (last_msg_received) {
          this.pendingResponses.push(last_msg_received);
        }

        this.turnsRemaining = roundData.turns_left ?? null;
        this.targetCanEndRound = roundData.target_can_end_round;

        this.unlimitedTurns = !Number.isFinite(this.turnsRemaining);
        this.roundTimeLimit = roundData.time_remaining;

        // Rounds metadata
        const roundsData = await api.getParticipantRounds(this.participantId);
        this.numRounds = roundsData.num_rounds;
        this.totalRounds = roundsData.total_rounds;

        // The target first has to anser the pre question
        this.initialDecisionPending =
          this.isTarget &&
          (this.initialTargetBelief === null ||
            this.initialTargetBelief === undefined);
        if (this.initialDecisionPending) {
          this.startBeliefSurveyPhase("initial");
        }

        if (this.turnsRemaining) {
          // Round startup
          if (this.roundTimeLimit !== null) {
            this.startRoundTimer(this.roundTimeLimit);
          }

          this.openWebSocket();

          if (roundData.waiting) {
            this.setWaitingTimeout();
          }

          // Show the instructions if the participant has not seen them
          if (this.instructions && this.useAudio && !audioContextSetup()) {
            // they have seen the instructions
            // We need a user gesture to warm up the audio
            this.warmupAudioPending = true;
            this.popover = {
              title: "Reloading round.",
              message: "",
              onClose: this.warmupAudio,
            };
          } else {
            this.fetchInstructions();
          }
        }
      } catch (err) {
        this.roundError(err);
      } finally {
        this.loading = false;
      }
    },
    // Resume the round after the participant closes instructions.
    okayInstructions() {
      // The ppt has accepted the instructions
      // They are warmed up. If we have the prompt, start the rd
      this.showInstructionsPending = false;
      if (this.prompt) {
        this.startRound();
      }
    },
    // Start the round (after the instructions & prompt have been received)
    startRound() {
      const pending = this.pendingResponses.length > 0;
      if (pending && !this.initialDecisionPending) {
        this.flushPendingResponses();
      }

      if (this.useAudio) {
        const ctx = getAudioContext();
        testAssert(ctx !== null, "We should have the context here");
      }
      if (
        this.useAudio &&
        !this.waiting &&
        !this.playing &&
        !this.initialDecisionPending &&
        !pending &&
        (this.turnsRemaining === null || this.turnsRemaining > 0)
      ) {
        // After a delay start recording the persuader
        setTimeout(() => {
          this.startRecording();
        }, api.redirectMilliseconds());
      }
    },
    // Start the countdown timer for the round.
    startRoundTimer(timeRemaining) {
      // Starts a count down for the time given on the server

      // initialize our remaining‐seconds
      this.timeRemainingSec = timeRemaining;
      this.roundTimedOut = false;

      // clear any old interval
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
      }

      // tick down once per second
      this.timerInterval = setInterval(() => {
        if (this.timeRemainingSec > 0) {
          this.timeRemainingSec--;
        } else {
          // time's up!
          clearInterval(this.timerInterval);
          this.timerInterval = null;
          this.roundTimedOut = true;
          this.roundEnded = true;
          this.turnsRemaining = 0;
          this.pendingResponses = [];
          console.warn("Round timer expired; suppressing new responses.");

          // auto‐end the round
          if (!this.useAudio || (!this.isPlaying && !this.chatEndStarted)) {
            this.chatOver();
          }
        }
      }, 1000);
    },
    // Build the WebSocket URL for this participant and round.
    roundChatURL() {
      testAssert(
        this.currentRoundId !== null,
        "CurrentRoundId must not be null",
      );
      testAssert(this.participantId !== null, "ParticipantId must not be null");

      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const host = window.location.host;
      return `${protocol}://${host}/ws/round/${this.currentRoundId}/participant/${this.participantId}`;
    },
    // Establish and manage the round WebSocket connection.
    openWebSocket() {
      const url = this.roundChatURL();
      this.ws = new WebSocket(url);
      this.ws.onopen = () => {
        console.log("WebSocket connected");
      };
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        console.log("Message received, ", msg);
        switch (msg.type) {
          case "round_started":
            // Show the prompt
            this.prompt = msg.prompt;
            // prompt if prompt_during_round is null
            this.promptDuringRound = msg.prompt_during_round ?? this.prompt;
            this.mainPropositionText = this.extractPromptPropositionText(
              this.prompt,
            );
            this.discussionPropositionText = this.extractPromptPropositionText(
              this.promptDuringRound,
            );
            this.awaitingControlPromptAcknowledgement = false;
            this.controlPromptTransitionAcknowledged = false;
            // If the instructions are not showing, show the prompt...
            if (!this.showInstructionsPending && !this.warmupAudioPending) {
              this.startRound();
            }
            break;
          case "response":
            this.handleIncomingMessage(msg);
            break;
          case "echo":
            if (this.useAudio && msg.text !== null) {
              this.chatMessages.push({ text: msg.text, sender: "You" });
            }
            break;
          case "message":
            // We should not be getting messages back--we only send these.
            console.warn("Message type 'message' received", msg);
            break;
          case "flagged": {
            // your last message was flagged; pop a notice
            this.handleFlaggedMessage(msg.reason);
            break;
          }
          case "round_over":
            // round has ended; show slider or end screen

            // NB: seting this.turnsRemaining to 0 reagardless of the actual state
            this.turnsRemaining = 0;
            this.roundEnded = true;
            this.pendingResponses = [];
            console.warn("Round ended; suppressing new responses.");

            // We need the audio to finish playing before we wrap up the chat
            if (
              (!this.useAudio && !this.mouseTrace) ||
              (!this.isPlaying && !this.chatEndStarted)
            ) {
              this.chatOver();
            }
            break;
          case "round_result":
            this.onRoundResult(msg);
            if (!this.onReflection) {
              this.ws.close();
            }
            break;
          default:
            console.warn("Unknown ws type", msg);
        }
      };
      this.ws.onerror = (err) => {
        console.error("WebSocket error", err);
      };
      this.ws.onclose = () => {
        console.log("WebSocket closed");
        this.waitingForResult();
      };
    },
    // Route incoming responses or buffer them until allowed.
    handleIncomingMessage(msg) {
      // LLM or other participant's message
      if (this.roundEnded || this.roundTimedOut) {
        console.warn(
          "Dropping incoming response after round ended/timed out.",
          msg,
        );
        return;
      }
      if (
        this.initialDecisionPending ||
        this.showInstructionsPending ||
        this.awaitingControlPromptAcknowledgement
      ) {
        this.pendingResponses.push(msg);
      } else {
        this.handleReceivedResponse(msg);
      }
    },
    // replay all buffered responses
    flushPendingResponses() {
      if (this.roundEnded) {
        if (this.pendingResponses.length > 0) {
          console.warn(
            "Dropping buffered responses after round ended.",
            this.pendingResponses,
          );
        }
        this.pendingResponses = [];
        return;
      }
      this.pendingResponses.forEach((msg) => {
        this.handleReceivedResponse(msg);
      });
      this.pendingResponses = [];
    },
    // We have received a response from the other player
    handleReceivedResponse(msg) {
      if (this.roundEnded || this.roundTimedOut) {
        console.warn("Ignoring response after round ended/timed out.", msg);
        return;
      }
      console.log("Processing response, ", msg);
      testAssert(msg.text, "The ws should return text");
      const response = { transcript: msg.transcript, sender: "Agent" };

      this.turnsRemaining = msg.turns_left;
      this.targetCanEndRound = msg.target_can_end_round;

      this.clearWaiting();

      if (this.useAudio && !this.roundEnded) {
        testAssert(msg.audio, "The ws should return audio");
        this.isPlaying = true;
        if (this.mouseTrace) {
          this.playAudioPending = true;
          this.audioStartedTriggered = false;
          this.lastAudioReceived = msg.audio;
        } else {
          // We're not awaiting here so we'll get to the
          // end of the function
          this.playIncomingAudio(msg.audio);
        }
      }

      let serialSentences = null;
      if (
        this.serialQuestionsSentence &&
        (this.turnsRemaining === null || this.turnsRemaining > 1)
      ) {
        testAssert(
          Array.isArray(msg.sentences) && msg.sentences.length > 0,
          "Server must provide sentence splits for serial-questions-sentence.",
        );
        serialSentences = msg.sentences;
      }

      if (this.mouseTrace) {
        /* Store the full text but only show it word-by-word later */
        response.full = msg.text; // keep a copy
        response.text = ""; // nothing visible yet

        this.isPlaying = true;
        this.playTextPending = true;
        this.textStartedTriggered = false;
        // remember the msg object
        this.lastTextReceived = response;
      } else {
        /* plain old immediate reveal */
        response.text = serialSentences ? serialSentences[0] : msg.text;

        // if not mouseTrace
        if (
          this.serialQuestionsSentence &&
          (this.turnsRemaining === null || this.turnsRemaining > 1)
        ) {
          // Queueing occurs after we push the message so we know the index.
        } else if (this.serialQuestions && this.turnsRemaining > 1) {
          this.serialDecisionPending = true;
        }
      }
      this.chatMessages.push(response);

      const newMessageIndex = this.chatMessages.length - 1;

      if (
        this.messageHighlights &&
        this.isTarget &&
        response.sender === "Agent"
      ) {
        this.startMessageHighlightTask(newMessageIndex);
      }

      if (
        serialSentences &&
        this.serialQuestionsSentence &&
        (this.turnsRemaining === null || this.turnsRemaining > 1)
      ) {
        this.queueSerialSentenceQuestions({
          text: msg.text,
          sentences: serialSentences,
          messageIndex: newMessageIndex,
        });
      }
    },
    // Notify the server when the target ends the round.
    targetEndsRound() {
      // To be called when the ppt (as a target) wants to end the round.
      testAssert(this.isTarget && !this.waiting && !this.showQuestionPanel);

      if (this.messageHighlights && this.messageHighlightPending) {
        this.scrollIntoViewAsync(".highlight-prompt", "start");
        return;
      }

      if (this.targetEndedRound) return; // already clicked

      // mark as clicked (disables further clicks/opacity)
      this.targetEndedRound = true;

      // Emit a message to the WS to end the round
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const msg = { type: "target_ends_round" };
        this.ws.send(JSON.stringify(msg));
      } else {
        console.error("WebSocket not open");
      }
    },
    // Transition to the result view while awaiting final data.
    waitingForResult() {
      // On round over
      if (this.resultStatus != "result") {
        if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
          // For whatever reason the ws is closed... So just show an empty result
          this.resultStatus = "result"; // stop the spinner
          this.roundResultMessage = "The round is over."; // blank / unknown result
        } else {
          // Otherwise we are waiting already or not yet waiting
          this.resultStatus = "waiting";
        }
      }
      this.roundResultReady = true;

      // Stop recording
      if (this.mediaRecorder && this.isRecording) {
        // stop the auto‐stop timer
        clearTimeout(this.recordingTimeout);
        this.recordingTimeout = null;

        // stop the recorder
        this.mediaRecorder.stop();
        this.isRecording = false;
        this.clearRecordingCountdown();
      }
    },
    // Store the result message delivered by the server.
    onRoundResult(msg) {
      // On round result
      this.resultStatus = "result";
      this.roundResultMessage = this.isTarget
        ? msg.target_message
        : msg.persuader_message;
    },
    // Navigate participants once the round has concluded.
    roundOver() {
      // To be called after all decisions have been made at the end of a round.
      let next;
      if (this.numRounds < this.totalRounds) {
        next = "/lobby";
      } else {
        next = "/feedback";
      }
      // go there after three seconds
      setTimeout(() => {
        this.$router.push(next);
        this.finalDecisionPending = false;
      }, api.redirectMilliseconds());
    },
    // Finish the chat phase and send any pending continuous data.
    chatOver() {
      // To be called after the chat period of a round.
      testAssert(
        this.turnsRemaining <= 1,
        `Turns remaining should <= 1 if round over, but are ${this.turnsRemaining}`,
      );
      if (this.finalDecisionPending || this.roundResultReady) {
        // We have already received a call to chatOver.
        return;
      }
      this.clearWaiting();
      if (this.isTarget) {
        // Send the final mouse trace and serial belief
        const msg = {};
        if (this.mouseTrace) {
          msg.last_mouse_trace = this.mouseTraceBuffer.slice();
          // clear immediately, rounds off the buffer
          this.mouseTraceBuffer = [];
        }
        if (this.serialQuestions) {
          msg.last_serial_question = this.serialTargetBelief;
          this.serialTargetBelief = null;
        }
        if (
          this.serialQuestionsSentence &&
          Array.isArray(this.serialSentenceResponsesPending) &&
          this.serialSentenceResponsesPending.length > 0
        ) {
          msg.last_serial_question_sentences =
            this.serialSentenceResponsesPending.slice();
          this.serialSentenceResponsesPending = [];
          this.currentSerialSentence = null;
        }
        if (this.messageHighlights) {
          const highlightsPayload = this.collectPendingMessageHighlights(true);
          if (highlightsPayload.length > 0) {
            msg.last_message_highlight = highlightsPayload;
          }
        }
        if (Object.keys(msg).length > 0) {
          msg.type = "final_continuous_measure";
          this.ws.send(JSON.stringify(msg));
        }

        if (this.serialQuestionsSentence) {
          this.resetSerialSentenceState();
        }
        if (this.messageHighlights) {
          this.resetMessageHighlightState();
        }

        if (this.onReflection && !this.onReflectionSubmitted) {
          this.onReflectionPending = true;
        }

        // target now confirms final belief
        // Don't show the next pane too fast.
        if (this.finalTargetBelief === null) {
          setTimeout(() => {
            this.finalDecisionPending = true;
          }, api.redirectMilliseconds());
        }
      } else {
        this.waitingForResult();
      }
    },

    /* ===================================================================
     * Messaging & Chat Utilities
     * =================================================================== */
    // The last message the user sent was flagged
    handleFlaggedMessage(reason) {
      console.warn("Last sent message flagged");
      const lastMsg = this.getLastMessage(/*ours=*/ null);
      console.log(`Last message: ${lastMsg}`);

      if (
        !lastMsg &&
        this.isTarget &&
        reason.toLowerCase().includes("server error")
      ) {
        // The flagged was b/c of a server error on the LLM first rsp
        this.sendInitialChoice();
        return;
      }

      if (this.serialQuestions && this.lastSerialTargetBelief) {
        this.serialTargetBelief = this.lastSerialTargetBelief;
      }

      // Reinstate per-sentence responses so the participant can resend them.
      if (
        this.serialQuestionsSentence &&
        Array.isArray(this.lastSerialSentenceResponses)
      ) {
        this.serialSentenceResponsesPending =
          this.lastSerialSentenceResponses.slice();
      }

      if (!this.useAudio) {
        // We only have a message to take off in the textual domains
        // (In audio we are waiting for the transcript)
        this.chatMessages.pop();
      }

      this.afterSentMessage();
      // Cancel the timer we just set on 'waiting'
      this.clearWaiting();

      this.waitingForAcknowledgeFailed = true;

      this.popover = {
        title: "Failed to Send Message",
        message: reason,
        onClose: this.acknowledgeFailed,
      };
    },
    // Run any popover close callback before clearing it.
    handlePopoverClose() {
      // To be called after the user closes the popover
      const callback = this.popover && this.popover.onClose;
      this.popover = null;
      if (typeof callback === "function") {
        callback();
      }
    },
    // Reset failure state after acknowledging a send failure.
    acknowledgeFailed() {
      // To be called after the user closes the popover
      if (this.useAudio && this.waitingForAcknowledgeFailed) {
        this.startRecording();
      }
      this.waitingForAcknowledgeFailed = false;
    },
    // Adds all of the relevant headers to the web socket message
    // and returns it to be sent
    prepareMessageWS(content) {
      const msg = { type: "message", content: content };
      if (this.serialQuestions && this.turnsRemaining > 1) {
        // TODO: narrow this assert not to fire on the target's first response and on their last
        testAssert(
          this.serialTargetBelief !== null,
          "There should be a last target serial belief.",
        );
        msg.last_serial_question = this.serialTargetBelief;
        this.lastSerialTargetBelief = this.serialTargetBelief;
        this.serialTargetBelief = null;
      }

      if (
        this.serialQuestionsSentence &&
        (this.turnsRemaining === null || this.turnsRemaining > 1)
      ) {
        testAssert(
          Array.isArray(this.serialSentenceResponsesPending) &&
            this.serialSentenceResponsesPending.length > 0,
          "There should be sentence-level serial responses.",
        );
        msg.last_serial_question_sentences =
          this.serialSentenceResponsesPending.slice();
        this.lastSerialSentenceResponses =
          msg.last_serial_question_sentences.slice();
        this.serialSentenceResponsesPending = [];
        this.currentSerialSentence = null;
      }

      if (this.messageHighlights && this.isTarget) {
        const highlightsPayload = this.collectPendingMessageHighlights(true);
        if (highlightsPayload.length > 0) {
          msg.last_message_highlight = highlightsPayload;
        }
      }

      if (this.mouseTrace) {
        // pack up everything we've collected so far
        msg.last_mouse_trace = this.mouseTraceBuffer.slice();
        // clear buffer so next message starts fresh
        this.mouseTraceBuffer = [];
      }

      // TODO: other continuous measures later
      testAssert(
        msg,
        `Message should not be null, undefined, or empty: ${msg}`,
      );
      const out = JSON.stringify(msg);
      return out;
    },
    // Send the participant's chat message via WebSocket.
    async handleSendMessage({ message }) {
      if (!this.canSendMessage) return;

      if (this.useAudio) {
        // audio‐only mode: stop the recording and send
        this.stopRecordingAndSend();
        return;
      }
      // Text mode
      if (!message.trim()) return;
      // Add the user's message to the chat
      this.chatMessages.push({ text: message, sender: "You" });

      // 3) send over WS
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(this.prepareMessageWS(message));
      } else {
        console.error("WebSocket not open");
      }

      // reset any old timers & flags
      this.afterSentMessage();
    },
    // Retrieve the latest chat message from either participant.
    getLastMessage(ours) {
      // Returns the last message of ours or the Agents
      if (this.chatMessages.length > 0) {
        const last = this.chatMessages[this.chatMessages.length - 1];
        if (
          (last.sender == "You" && ours) ||
          (last.sender == "Agent" && !ours)
        ) {
          return last;
        }
      }
      return null;
    },
    // Keep the chat scrolled to the newest content.
    updateChatUI() {
      const chatWindow = document.getElementById("chat-window");
      chatWindow.scrollTop = chatWindow.scrollHeight;
    },
    // Clear waiting and typing indicators.
    clearWaiting() {
      // Resets all waiting fields.
      clearTimeout(this.typingTimeout);
      this.typingTimeout = null;
      this.waiting = false;
      this.typing = false;
    },
    // Reset messaging state after a send action.
    afterSentMessage() {
      // Resets all waiting fields.
      this.clearWaiting();

      // Clear the input field
      this.newMessage = "";
      if (!this.useAudio) {
        this.$nextTick(this.updateChatUI);
      }
      this.setWaitingTimeout();

      if (this.messageHighlights) {
        if (this.persistMessageHighlights) {
          this.messageHighlightPending = false;
          this.messageHighlightPendingIndex = null;
        } else {
          this.resetMessageHighlightState();
        }
      }
    },
    // Schedule when the partner should appear to respond.
    setWaitingTimeout() {
      this.waiting = true;
      const readDelay = 40 * 40;
      this.typingTimeout = setTimeout(
        () => {
          if (this.turnsRemaining > 1 || this.turnsRemaining === null) {
            this.typing = true;
          }
        },
        readDelay + Math.random() * api.redirectMilliseconds(),
      );
    },

    /* ===================================================================
     * On-Reflection Highlighting
     * =================================================================== */
    resetOnReflectionState() {
      this.onReflection = false;
      this.onReflectionPending = false;
      this.onReflectionHighlights = [];
      this.onReflectionSubmitted = false;
    },
    // Construct the payload sent to the server for highlights.
    buildOnReflectionPayload() {
      if (!Array.isArray(this.chatMessages)) {
        return [];
      }
      return this.onReflectionHighlights
        .map((entry) => {
          const fullText = this.getChatMessageText(entry.messageIndex);
          if (!fullText) {
            return null;
          }
          const normalized = normalizeHighlight(entry, fullText);
          if (!normalized) {
            return null;
          }
          const { start, end, text } = normalized;
          return {
            message_index: entry.messageIndex,
            text: text,
            selection_start: start,
            selection_end: end,
            full_text: fullText,
          };
        })
        .filter((entry) => entry !== null);
    },
    // Submit highlights to the server and close the socket.
    completeOnReflection() {
      if (this.onReflectionSubmitted) {
        return;
      }
      this.onReflectionSubmitted = true;
      this.onReflectionPending = false;

      const payload = this.buildOnReflectionPayload();
      const msg = {
        type: "on_reflection_highlights",
        on_reflection_highlights: payload,
      };
      console.log(msg);
      this.ws.send(JSON.stringify(msg));
      this.exitReflectionHighlightMode();
      this.ws.close();
    },
    // Reopen the reflection UI so the target can edit highlights.
    editOnReflectionHighlights() {
      if (!this.onReflection) {
        return;
      }
      this.onReflectionPending = true;
    },
    // Prompt the target to complete the reflection step.
    showOnReflection() {
      testAssert(this.onReflection && this.isTarget);

      if (
        this.onReflectionHighlightRequired &&
        this.onReflectionHighlights.length === 0
      ) {
        this.onReflectionPending = true;
        this.popover = {
          title: "Please add a highlight",
          message:
            "Because your view changed, please highlight at least one of the other participant's messages before finishing.",
          onClose: () => {
            this.onReflectionPending = true;
          },
        };
      }
    },

    /* ===================================================================
     * Mouse Trace & Text Reveal
     * =================================================================== */
    async clickMouseTracePanel() {
      // If a message is pending, only activate when cursor is near the start-anchor dot
      if (this.msgPending && !this.cursorNearStartAnchor) {
        return;
      }

      if (this.playAudioPending && this.lastAudioReceived) {
        // clear the "please start" state
        console.log("play Audio pending");

        this.playAudioPending = false;
        this.audioStartedTriggered = true;
        // actually play it
        this.playIncomingAudio(this.lastAudioReceived);
        this.lastAudioReceived = null;
      }
      // both may be played at once
      if (this.playTextPending && this.lastTextReceived) {
        this.playTextPending = false;
        this.textStartedTriggered = true;
        this.startTextReveal(this.lastTextReceived);
        this.lastTextReceived = null;
      }
      // Otherwise do nothing; ending the round is via the button below the slider.
    },
    // map raw mouse X to [0,100] inside the panel

    handleMouseMove(event) {
      const rect = event.currentTarget.getBoundingClientRect();
      let pct = ((event.clientX - rect.left) / rect.width) * 100;
      // clamp
      pct = Math.min(100, Math.max(0, pct));
      this.cursorPosition = pct;

      if (this.mouseTrace && this.mouseTraceStartTime !== null) {
        const now = performance.now();
        const audio_relative_timestamp =
          now - this.mouseTraceStartTime - this.sumOfPauses;
        // NB: we need to subtract how long the trace was paused on this last time (if at all)
        this.mouseTraceBuffer.push({
          position: pct,
          timestamp: audio_relative_timestamp,
        });
      }

      // if we've not yet triggered play & we're in the "pending" state,
      // check for center entry
      const cursorInThreshold =
        Math.abs(pct - this.startAnchorPercent) <= this.playThreshold;

      // Auto-activate when they move to the initial-belief spot
      if (this.msgPending && cursorInThreshold) {
        this.clickMouseTracePanel();
      }
    },
    // Pause playback and timers when the pointer leaves the trace panel.
    onMouseLeave() {
      if (this.mouseTrace && this.isPlaying && !this.msgPending) {
        if (this.useAudio) {
          getAudioContext().suspend().catch(console.warn);
          this.audioPausedByLeave = true;
        }
        // text reveal
        this.pauseTextReveal();
        this.textPausedByLeave = true;

        this.timeWhenPaused = performance.now();
      }
    },
    // Resume playback and timers when the pointer re-enters.
    onMouseEnter() {
      if (this.mouseTrace && this.isPlaying && !this.msgPending) {
        if (this.useAudio) {
          getAudioContext().resume().catch(console.warn);
          this.audioPausedByLeave = false;
        }

        if (this.textPausedByLeave) {
          this.resumeTextReveal();
          this.textPausedByLeave = false;
        }

        if (this.timeWhenPaused !== null) {
          const pauseDuration = performance.now() - this.timeWhenPaused;
          this.sumOfPauses += pauseDuration;
          this.timeWhenPaused = null;
        }
      }
    },
    /* TEXT-REVEAL BEGIN */
    // Start staged transcript reveal for the agent's most recent message.
    startTextReveal(msgObj) {
      // initialize state */
      this.revealSchedule = msgObj.transcript.map(({ text, duration }) => ({
        token: text,
        delay: duration,
      }));

      console.log(msgObj.transcript);
      console.log("revealSchedule", this.revealSchedule);
      this.revealIdx = 0;
      this.revealMsgObj = msgObj;

      if (!this.useAudio) {
        testAssert(
          this.mouseTraceStartTime === null,
          "start time should not be set",
        );
        this.resetMouseTrace();
      } else {
        testAssert(
          this.mouseTraceStartTime !== null,
          "start time should be determined by audio",
        );
      }

      this.scheduleNextWord();
    },
    // Schedule the timer for revealing the next transcript token.
    scheduleNextWord() {
      if (this.revealIdx >= this.revealSchedule.length) {
        this.finishTextReveal();
      } else {
        const { delay } = this.revealSchedule[this.revealIdx];

        this.revealDelayMs = delay * 1000;
        this.revealRemainingMs = this.revealDelayMs;
        this.revealStartMs = performance.now();

        this.revealTimer = setTimeout(this.stepTextReveal, this.revealDelayMs);
      }
    },
    // Reveal the next word in the transcript playback.
    stepTextReveal() {
      const { token } = this.revealSchedule[this.revealIdx++];
      /* append token to visible text */

      this.revealMsgObj.text += (this.revealIdx === 1 ? "" : " ") + token;

      this.scheduleNextWord();
    },
    // Pause text reveal timers while maintaining progress state.
    pauseTextReveal() {
      if (this.revealTimer) {
        clearTimeout(this.revealTimer);
        this.revealTimer = null;

        const elapsed = performance.now() - this.revealStartMs;
        this.revealRemainingMs = Math.max(0, this.revealDelayMs - elapsed);

        // If the delay already expired while we were away,
        // show the word immediately.
        if (this.revealRemainingMs === 0) {
          this.stepTextReveal();
        }
      }
    },
    // Resume text reveal after a pause.
    resumeTextReveal() {
      if (!this.revealTimer && this.revealMsgObj) {
        this.revealStartMs = performance.now();
        this.revealDelayMs = this.revealRemainingMs;

        this.revealTimer = setTimeout(
          this.stepTextReveal,
          this.revealRemainingMs,
        );
      }
    },
    // Complete the reveal, restoring the full transcript text.
    finishTextReveal() {
      clearTimeout(this.revealTimer);
      this.revealTimer = null;
      // Replace the text with the original text (e.g. with punctuation)
      this.revealMsgObj.text = this.revealMsgObj.full;
      this.revealMsgObj = null;
      this.revealSchedule = [];

      this.revealStartMs = null;
      this.revealDelayMs = 0;
      this.revealRemainingMs = 0;

      if (!this.useAudio) {
        // Let the end of the playing audio handle the logic
        // (Keep the continuous measure going briefly)
        setTimeout(() => {
          this.afterPlayAudioOrText();
        }, api.postPlayDelay());
      }
    },
    /* TEXT-REVEAL END */

    resetMouseTrace() {
      if (this.mouseTrace) {
        // reset buffer and anchor time
        this.mouseTraceBuffer = [];
        this.mouseTraceStartTime = performance.now();
        // recenter cursor if desired
        // also clear any stale pause markers
        this.timeWhenPaused = null;
        this.sumOfPauses = 0;
        if (this.recenterCursor) this.cursorPosition = 50;
      }
    },

    /* ===================================================================
     * Serial sentence functions
     * =================================================================== */
    // Reset all serial-sentence tracking so the next persuader message starts fresh.
    resetSerialSentenceState() {
      this.serialSentenceQueue = [];
      this.serialSentenceResponsesPending = [];
      this.currentSerialSentence = null;
      this.lastSerialSentenceResponses = null;
      this.serialSentenceActiveMessageIndex = null;
      this.serialSentenceFullText = "";
      this.serialDecisionPending = false;
    },

    // Split the persuader message into sentences and prepare sequential follow-up prompts.
    queueSerialSentenceQuestions({ text, sentences, messageIndex }) {
      if (!this.serialQuestionsSentence || !this.isTarget) {
        return;
      }
      this.resetSerialSentenceState();
      testAssert(
        Array.isArray(sentences) && sentences.length > 0,
        "Expected server-provided sentences when queuing serial questions.",
      );
      const extracted = sentences;

      testAssert(
        !this.serialSentenceResponsesPending ||
          this.serialSentenceResponsesPending.length === 0,
        "Expected sentence responses to be cleared before queueing a new message.",
      );

      console.debug("[serial-sentences] queued", extracted);
      this.serialSentenceQueue = extracted.slice();
      this.serialSentenceResponsesPending = [];
      this.currentSerialSentence = null;
      this.lastSerialSentenceResponses = null;
      this.serialSentenceActiveMessageIndex = messageIndex;
      this.serialSentenceFullText = text;

      this.startSerialSentenceQuestionsIfReady();
    },

    // Present the next sentence question if the round state permits it.
    startSerialSentenceQuestionsIfReady() {
      if (this.serialDecisionPending) {
        return;
      }
      if (!this.serialQuestionsSentence) {
        return;
      }
      if (this.turnsRemaining !== null && this.turnsRemaining <= 1) {
        return;
      }
      if (!this.serialSentenceQueue.length) {
        return;
      }

      this.currentSerialSentence = this.serialSentenceQueue[0];
      console.debug(
        "[serial-sentences] prompting",
        this.currentSerialSentence,
        this.serialSentenceQueue.length,
      );
      this.serialDecisionPending = true;
      this.resetBeliefSliderDefaults();
      if (this.serialSentenceActiveMessageIndex !== null) {
        this.updateSerialSentenceDisplay(this.currentSerialSentence, {
          append: true,
        });
      }
      this.scrollIntoViewAsync(".belief-slider-panel");
    },

    // Store the latest response and advance to the next sentence, or finish.
    advanceSerialSentenceAfterResponse() {
      if (!this.serialQuestionsSentence) {
        return;
      }

      this.serialSentenceQueue.shift();
      if (this.serialSentenceQueue.length > 0) {
        this.currentSerialSentence = this.serialSentenceQueue[0];
        console.debug(
          "[serial-sentences] advancing",
          this.currentSerialSentence,
          this.serialSentenceQueue.length,
        );
        this.serialDecisionPending = true;
        this.resetBeliefSliderDefaults();
        if (this.serialSentenceActiveMessageIndex !== null) {
          this.updateSerialSentenceDisplay(this.currentSerialSentence, {
            append: true,
          });
        }
        this.scrollIntoViewAsync(".belief-slider-panel");
      } else {
        console.debug(
          "[serial-sentences] finished",
          this.serialSentenceResponsesPending,
        );
        this.currentSerialSentence = null;
        this.serialDecisionPending = false;
        // Preserve the collected responses so they can be sent with the next message
        // (or during the final continuous-measure upload when the round ends).
        this.lastSerialSentenceResponses = Array.isArray(
          this.serialSentenceResponsesPending,
        )
          ? [...this.serialSentenceResponsesPending]
          : [];
        if (this.serialSentenceActiveMessageIndex !== null) {
          this.updateSerialSentenceDisplay(this.serialSentenceFullText || "", {
            finalize: true,
          });
          this.serialSentenceActiveMessageIndex = null;
          this.serialSentenceFullText = "";
        }
        // If we're still in audio mode, restart recording after a short pause.
        if (this.useAudio) {
          setTimeout(() => this.startRecording(), api.redirectMilliseconds());
        }
      }
    },

    /* ===================================================================
     * Message highlight functions
     * =================================================================== */
    enterReflectionHighlightMode() {
      // Switch the UI into reflection highlight mode so the existing conversation can be re-used.
      this.persistMessageHighlights = true;
      this.messageHighlights = true;
      this.messageHighlightPending = false;
      this.messageHighlightPendingIndex = null;
      this.clearAllMessageHighlightDisplays();
      this.syncOnReflectionHighlights();
      this.onReflectionHighlights = [];
      this.scrollIntoViewAsync(".chat-window", "start");
    },

    exitReflectionHighlightMode() {
      // Leave reflection highlight mode and restore the normal chat behaviour.
      this.persistMessageHighlights = false;
      this.messageHighlights = false;
      this.resetMessageHighlightState();
      this.onReflectionHighlights = [];
    },

    resetMessageHighlightState() {
      // Fully reset the per-message highlight state and clear any in-flight ranges.
      this.messageHighlightPending = false;
      this.messageHighlightPendingIndex = null;
      this.clearAllMessageHighlightDisplays();
    },

    startMessageHighlightTask(messageIndex) {
      // Prepare to collect highlights for the persuader message at the given index.
      if (!this.messageHighlights || !this.isTarget) {
        return;
      }
      if (!this.persistMessageHighlights) {
        this.clearAllMessageHighlightDisplays();
      }
      this.messageHighlightPending = true;
      this.messageHighlightPendingIndex = messageIndex;
      this.messageHighlightRanges = {
        ...this.messageHighlightRanges,
        [messageIndex]: [],
      };
      this.messageHighlightQueue = {
        ...this.messageHighlightQueue,
        [messageIndex]: [],
      };
      this.applyMessageHighlightRanges(messageIndex, []);
      this.scrollIntoViewAsync(".highlight-prompt", "start");
    },

    handleMessageHighlight({ messageIndex, start, end }) {
      // Handle highlight selections from the chat component.
      if (!this.messageHighlights || !this.isTarget) {
        return;
      }
      const message = this.chatMessages[messageIndex];
      if (!message || message.sender !== "Agent") {
        return;
      }
      this.toggleHighlightSelection(messageIndex, start, end, {
        updateQueue: !this.persistMessageHighlights,
      });
    },

    toggleHighlightSelection(
      messageIndex,
      start,
      end,
      { updateQueue = false } = {},
    ) {
      // Toggle the selection for the given message and range.
      const text = this.getChatMessageText(messageIndex);
      if (!text) {
        return;
      }
      const normalized = normalizeRange({ start, end }, text.length);
      if (!normalized) {
        return;
      }
      const currentRanges = this.getMessageHighlightRanges(messageIndex, text);
      const nextRanges = isRangeFullyCovered(currentRanges, normalized)
        ? subtractRangeList(currentRanges, normalized)
        : addRangeToList(currentRanges, normalized, text.length);
      this.applyMessageHighlightRanges(messageIndex, nextRanges);
      if (updateQueue) {
        this.updateMessageHighlightMaps(messageIndex, nextRanges, text);
        this.updateHighlightGateStatus();
      } else if (this.persistMessageHighlights || this.onReflectionPending) {
        this.syncOnReflectionHighlights();
      }
    },

    updateMessageHighlightMaps(messageIndex, ranges, text) {
      // Store or remove ranges for the given message.
      if (ranges.length > 0) {
        this.messageHighlightRanges = {
          ...this.messageHighlightRanges,
          [messageIndex]: ranges,
        };
      } else if (
        Object.prototype.hasOwnProperty.call(
          this.messageHighlightRanges,
          messageIndex,
        )
      ) {
        const rest = { ...this.messageHighlightRanges };
        delete rest[messageIndex];
        this.messageHighlightRanges = rest;
      }

      const entries = ranges
        .map((r) =>
          normalizeHighlight(
            {
              messageIndex,
              start: r.start,
              end: r.end,
            },
            text,
          ),
        )
        .filter(Boolean);

      if (entries.length > 0) {
        this.messageHighlightQueue = {
          ...this.messageHighlightQueue,
          [messageIndex]: entries,
        };
      } else if (
        Object.prototype.hasOwnProperty.call(
          this.messageHighlightQueue,
          messageIndex,
        )
      ) {
        const restQueue = { ...this.messageHighlightQueue };
        delete restQueue[messageIndex];
        this.messageHighlightQueue = restQueue;
      }

      if (this.persistMessageHighlights || this.onReflectionPending) {
        this.syncOnReflectionHighlights();
      }
    },

    updateHighlightGateStatus() {
      // Keep the in-round highlight gate active until the participant presses Continue.
      if (!this.messageHighlights || this.persistMessageHighlights) {
        return;
      }
      this.messageHighlightPending = true;
    },

    clearCurrentHighlights() {
      // Drop the in-progress highlights.
      this.clearAllMessageHighlightDisplays();
      if (!this.persistMessageHighlights) {
        this.messageHighlightPending = true;
      }
      this.updateHighlightGateStatus();
    },

    collectPendingMessageHighlights(clear = false) {
      // Serialize queued highlights to send to the backend.
      const payload = [];
      Object.entries(this.messageHighlightQueue).forEach(([index, entries]) => {
        if (!Array.isArray(entries) || entries.length === 0) {
          return;
        }
        entries.forEach((entry) => {
          if (!entry) {
            return;
          }
          payload.push({
            message_index: Number(index),
            start: entry.start,
            end: entry.end,
            text: entry.text,
          });
        });
      });
      if (clear) {
        this.messageHighlightQueue = {};
      }
      return payload;
    },

    clearAllMessageHighlightDisplays() {
      // Remove highlight decorations from all visible chat messages.
      if (!Array.isArray(this.chatMessages)) {
        return;
      }
      this.chatMessages = this.chatMessages.map((message) => {
        if (!message || typeof message !== "object") {
          return message;
        }
        const updated = { ...message };
        delete updated.highlightRanges;
        delete updated.highlightRange;
        delete updated.highlighted;
        return updated;
      });
      this.messageHighlightRanges = {};
      this.messageHighlightQueue = {};
      this.onReflectionHighlights = [];
      this.messageHighlightPendingIndex = null;
      if (this.persistMessageHighlights || this.onReflectionPending) {
        this.syncOnReflectionHighlights();
      }
    },

    continueAfterHighlight() {
      // Advance past the highlight step, either submitting or skipping as appropriate.
      if (this.onReflectionPending) {
        this.completeOnReflection();
        return;
      }
      if (this.messageHighlightPending) {
        this.messageHighlightPending = false;
        this.messageHighlightPendingIndex = null;
      }
    },

    applyMessageHighlightRanges(messageIndex, ranges) {
      if (
        !Array.isArray(this.chatMessages) ||
        !this.chatMessages[messageIndex]
      ) {
        return;
      }
      const normalized = Array.isArray(ranges) ? [...ranges] : [];
      const updated = { ...this.chatMessages[messageIndex] };
      if (normalized.length > 0) {
        updated.highlightRanges = normalized;
      } else {
        delete updated.highlightRanges;
        delete updated.highlightRange;
        delete updated.highlighted;
      }
      this.chatMessages.splice(messageIndex, 1, updated);
    },

    getMessageHighlightRanges(messageIndex, text = null) {
      // Retrieve normalized highlight ranges for the specified message.
      if (!Array.isArray(this.chatMessages)) {
        return [];
      }
      const message = this.chatMessages[messageIndex];
      if (!message || typeof message !== "object") {
        return [];
      }
      const baseRanges = Array.isArray(message.highlightRanges)
        ? message.highlightRanges
        : message.highlightRange
          ? [message.highlightRange]
          : [];
      const content = text ?? this.getChatMessageText(messageIndex);
      const length = content.length;
      return normalizeRangesForLength(baseRanges, length);
    },

    getChatMessageText(messageIndex) {
      // Return the textual content for the message at the given index.
      if (!Array.isArray(this.chatMessages)) {
        return "";
      }
      const message = this.chatMessages[messageIndex];
      if (!message || typeof message !== "object") {
        return "";
      }
      return message.text ?? message.content ?? "";
    },

    syncOnReflectionHighlights() {
      // Refresh the cached reflection highlight payload from chat messages.
      if (!this.persistMessageHighlights || !Array.isArray(this.chatMessages)) {
        this.onReflectionHighlights = [];
        return;
      }
      const highlights = [];
      this.chatMessages.forEach((message, index) => {
        if (
          !message ||
          typeof message !== "object" ||
          message.sender !== "Agent"
        ) {
          return;
        }
        const text = this.getChatMessageText(index);
        if (!text) {
          return;
        }
        const ranges = this.getMessageHighlightRanges(index, text);
        ranges.forEach((range) => {
          highlights.push({
            messageIndex: index,
            start: range.start,
            end: range.end,
            text: text.slice(range.start, range.end),
          });
        });
      });
      this.onReflectionHighlights = highlights;
    },

    ///////////////////////////
    ///////////////////////////
    ///////////////////////////

    // Extract the main proposition text from the rendered prompt HTML.
    extractPromptPropositionText(promptHtml) {
      if (typeof promptHtml !== "string" || !promptHtml.trim()) {
        return "";
      }
      const container = document.createElement("div");
      container.innerHTML = promptHtml;
      const quote = container.querySelector("blockquote");
      const text = quote ? quote.textContent : container.textContent;
      return String(text || "")
        .replace(/\s+/g, " ")
        .trim();
    },

    // Normalize proposition text before comparing phase prompts.
    normalizePromptTextForComparison(text) {
      if (typeof text !== "string") {
        return "";
      }
      return text.replace(/\s+/g, " ").trim().toLowerCase();
    },

    // Determine whether the round switches to a different discussion proposition.
    hasControlPromptTransition() {
      const main = this.normalizePromptTextForComparison(
        this.mainPropositionText,
      );
      const during = this.normalizePromptTextForComparison(
        this.discussionPropositionText,
      );
      return Boolean(main && during && main !== during);
    },

    // Show a blocking interstitial before the discussion proposition begins.
    showControlPromptTransitionPopover() {
      this.awaitingControlPromptAcknowledgement = true;
      this.popover = {
        title: "Proposition Changed",
        subtitle: this.discussionPropositionText,
        message:
          "The discussion proposition is now different from the belief question proposition. Click Continue to proceed.",
        okayText: "Continue",
        onClose: this.acknowledgeControlPromptTransition,
      };
      this.scrollIntoViewAsync(".prompt-panel", "start");
    },

    // Resume the round after the participant acknowledges the prompt change.
    acknowledgeControlPromptTransition() {
      this.awaitingControlPromptAcknowledgement = false;
      this.controlPromptTransitionAcknowledged = true;
      this.flushPendingResponses();
    },

    // Return a shuffled copy of survey items.
    shuffleSurveyItems(items) {
      const shuffled = Array.isArray(items) ? [...items] : [];
      for (let i = shuffled.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      return shuffled;
    },

    // Build the ordered pre/post survey item list for one phase.
    beliefSurveyPhaseItems(phase) {
      if (
        !this.beliefSurveyEnabled ||
        (phase !== "initial" && phase !== "final")
      ) {
        return [];
      }
      const targetItem = {
        id: "Target",
        text: this.mainPropositionText || "Main proposition",
      };
      return [targetItem, ...this.beliefSurveyItems];
    },

    // Reset belief-survey tracking state.
    resetBeliefSurveyPhase() {
      this.beliefSurveyPhase = null;
      this.beliefSurveyPhaseItemsCurrent = [];
      this.beliefSurveyQueue = [];
      this.beliefSurveyResponses = {};
      this.beliefSurveyCurrentItem = null;
    },

    // Start either the pre-survey or post-survey sequence.
    startBeliefSurveyPhase(phase) {
      if (!this.beliefSurveyEnabled) {
        this.resetBeliefSurveyPhase();
        return;
      }
      const phaseItems = this.beliefSurveyPhaseItems(phase);
      if (!phaseItems.length) {
        this.resetBeliefSurveyPhase();
        return;
      }
      const randomizedItems = this.shuffleSurveyItems(phaseItems);
      this.beliefSurveyPhase = phase;
      this.beliefSurveyPhaseItemsCurrent = randomizedItems;
      this.beliefSurveyQueue = randomizedItems.slice();
      this.beliefSurveyResponses = {};
      this.beliefSurveyCurrentItem = this.beliefSurveyQueue[0] || null;
      this.resetBeliefSliderDefaults();
    },

    // Record one belief-survey response and move to the next item.
    recordBeliefSurveyResponse(value) {
      if (!this.beliefSurveyCurrentItem) {
        return true;
      }
      const item = this.beliefSurveyCurrentItem;
      this.beliefSurveyResponses[item.id] = value;

      if (item.id === "Target") {
        if (this.beliefSurveyPhase === "initial") {
          this.initialTargetBelief = value;
        } else if (this.beliefSurveyPhase === "final") {
          this.finalTargetBelief = value;
        }
      }

      this.beliefSurveyQueue.shift();
      this.beliefSurveyCurrentItem = this.beliefSurveyQueue[0] || null;
      if (this.beliefSurveyCurrentItem) {
        this.resetBeliefSliderDefaults();
        return false;
      }
      return true;
    },

    // Build the node-belief payload from the completed survey responses.
    currentBeliefSurveyNodePayload() {
      const payload = {};
      this.beliefSurveyItems.forEach((item) => {
        const value = this.beliefSurveyResponses[item.id];
        if (typeof value === "number" && !Number.isNaN(value)) {
          payload[item.id] = value;
        }
      });
      return payload;
    },

    // Reset the belief slider to its unlocked, centered state.
    resetBeliefSliderDefaults() {
      this.beliefSelected = null;
      this.beliefCursorPosition = 50;
      this.beliefLocked = false;
    },

    // Smooth-scroll a selector into view on the next tick.
    scrollIntoViewAsync(selector, block = "center") {
      this.$nextTick(() => {
        if (!this.$el || !this.$el.querySelector) {
          return;
        }
        const el = this.$el.querySelector(selector);
        if (el && typeof el.scrollIntoView === "function") {
          el.scrollIntoView({ behavior: "smooth", block });
        }
      });
    },

    // Update the persuader chat bubble with either the current sentence or the full text.
    updateSerialSentenceDisplay(text, options = {}) {
      if (this.serialSentenceActiveMessageIndex === null) {
        return;
      }
      const idx = this.serialSentenceActiveMessageIndex;
      const existing = this.chatMessages[idx];
      if (!existing) {
        return;
      }
      let newText = text;
      if (options.append && existing && typeof existing.text === "string") {
        const trimmed = existing.text.trim();
        const originalFull = (this.serialSentenceFullText || "").trim();
        if (!trimmed || trimmed === originalFull) {
          newText = text;
        } else if (trimmed.endsWith(text.trim())) {
          newText = trimmed;
        } else {
          newText = trimmed ? `${trimmed} ${text}` : text;
        }
      }
      const updated = { ...existing, text: newText };
      if (options.finalize) {
        updated.text = text;
      }
      this.chatMessages.splice(idx, 1, updated);
    },

    /* ===================================================================
     * Audio Capture & Playback
     * =================================================================== */
    warmupAudio() {
      // The ppt has seen the instructions before so we need to
      // warm up the stream
      testAssert(
        this.useAudio,
        "Warm up should only be called when using audio",
      );
      getAudioContext();
      this.warmupAudioPending = false;

      // If we have the prompt, start the round
      if (this.prompt) {
        this.startRound();
      }
    },
    // Play the incoming audio B64, then either ask the serial question or
    // after ~2s start recording
    async playIncomingAudio(base64) {
      this.resetMouseTrace();

      try {
        console.log("Playing audio");
        await playBase64Audio(base64);
      } catch (e) {
        console.warn("Playback failed", e);
      } finally {
        console.log("Audio over");
      }

      // Keep the continuous measure going briefly
      setTimeout(() => {
        this.afterPlayAudioOrText();
      }, api.postPlayDelay());
    },
    // Trigger follow-up actions once audio or text playback ends.
    afterPlayAudioOrText() {
      this.isPlaying = false;
      this.mouseTraceStartTime = null;
      // ensure no stale pause is carried to next segment
      this.timeWhenPaused = null;

      this.audioStartedTriggered = false;
      this.textStartedTriggered = false;

      // For subsequent segments, remember where the participant left off
      if (this.mouseTrace) {
        const pct = Math.min(100, Math.max(0, this.cursorPosition));
        this.lastTraceAnchorPercent = pct;
      }

      if (
        this.turnsRemaining == 0 ||
        (this.targetEndedRound && !this.chatEndStarted) ||
        this.roundTimedOut
      ) {
        // The round is over
        this.chatOver();
      } else if (
        this.serialQuestionsSentence &&
        (this.turnsRemaining === null || this.turnsRemaining > 1)
      ) {
        this.startSerialSentenceQuestionsIfReady();
      } else if (this.serialQuestions && this.turnsRemaining > 1) {
        this.serialDecisionPending = true;
      } else if (this.useAudio && !this.roundTimedOut && !this.roundEnded) {
        // after 2s grace, start recording user response
        setTimeout(() => {
          this.startRecording();
        }, api.redirectMilliseconds());
      }
    },
    // Start recording the participant's microphone input.
    async startRecording() {
      if (!navigator.mediaDevices) {
        console.warn("No mediaDevices API");
        return;
      }
      if (this.isRecording) return;
      this.recordedChunks = [];

      // Prefer the pre-warmed stream if we have one
      const warm = getWarmStream();
      const stream =
        warm || (await navigator.mediaDevices.getUserMedia({ audio: true }));
      // If we're using a warm stream, re-enable its tracks
      if (warm) {
        warm.getTracks().forEach((t) => (t.enabled = true));
      }

      // pick a mimeType the browser supports
      let options = { mimeType: "audio/webm;codecs=opus" };
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: "audio/webm" };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
          options = {}; // let it choose a default
        }
      }

      this.mediaRecorder = new MediaRecorder(stream, options);
      this.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) this.recordedChunks.push(e.data);
      };
      this.mediaRecorder.onstop = this.onRecordingStop;
      this.mediaRecorder.start();

      console.log("Recording started");
      this.isRecording = true;
      this.startRecordingCountdown();

      // start a timer to auto–stop
      this.recordingTimeout = setTimeout(() => {
        if (this.isRecording) {
          console.log("Max recording time reached -- stopping automatically");
          this.stopRecordingAndSend();
        }
      }, api.maxAudioSeconds() * 1000);
    },
    // Begin the countdown shown during recording.
    startRecordingCountdown() {
      // Start the recording state
      // clear any leftover interval
      this.clearRecordingCountdown();
      // convert ms -> seconds (ceil so UI doesn't start at 00:00)
      this.recordingRemainingSec = Math.ceil(api.maxAudioSeconds());
      this.recordingInterval = setInterval(() => {
        if (this.recordingRemainingSec > 0) {
          this.recordingRemainingSec--;
        } else {
          this.clearRecordingCountdown();
        }
      }, 1000);
    },
    // Reset the recording countdown timer.
    clearRecordingCountdown() {
      // Clear the recording state
      if (this.recordingInterval) {
        clearInterval(this.recordingInterval);
        this.recordingInterval = null;
      }
      this.recordingRemainingSec = null;
    },
    // Stop recording and send the captured audio clip.
    async stopRecordingAndSend() {
      if (!this.mediaRecorder || !this.isRecording) return;

      // clear the auto–stop timer
      clearTimeout(this.recordingTimeout);
      this.recordingTimeout = null;

      this.mediaRecorder.stop();
      this.isRecording = false;
      this.clearRecordingCountdown();

      this.afterSentMessage();
      console.log("Recording stopped");
    },
    // Handle the media recorder's stop event and upload audio.
    async onRecordingStop() {
      if (this.roundResultReady) {
        this.recordedChunks = [];
        return;
      }
      if (!this.recordedChunks || this.recordedChunks.length == 0) {
        this.handleFlaggedMessage("Failed to record any audio.");
        return;
      }

      clearTimeout(this.recordingTimeout);
      this.recordingTimeout = null;

      // We need to set the mime type
      const mimeType =
        this.mediaRecorder && this.mediaRecorder.mimeType
          ? this.mediaRecorder.mimeType
          : this.recordedChunks[0].type || "audio/webm";

      const blob = new Blob(this.recordedChunks, { type: mimeType });
      // use FileReader to get a data-URL
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });

      // Close the media recorder
      const stream = this.mediaRecorder.stream;
      if (stream === getWarmStream()) {
        // we only disable warm-up tracks
        stream.getTracks().forEach((t) => (t.enabled = false));
      } else {
        // for on-demand streams we fully stop them
        stream.getTracks().forEach((t) => t.stop());
      }
      this.mediaRecorder = null;

      // send via WS
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        await this.ws.send(this.prepareMessageWS(dataUrl));
      } else {
        console.error("WebSocket not open");
      }
    },

    /* ===================================================================
     * Decision & Belief Controls
     * =================================================================== */
    handleBeliefMouseMove(event) {
      // once a belief is selected, do not allow further movement
      if (this.beliefLocked) return;
      const rect = event.currentTarget.getBoundingClientRect();
      let pct = ((event.clientX - rect.left) / rect.width) * 100;
      pct = Math.min(100, Math.max(0, pct));
      this.beliefCursorPosition = pct;
    },
    // Lock in the selected belief percentage when clicked.
    handleBeliefClick() {
      // record the chosen value as a fraction [0–1]
      this.beliefSelected = this.beliefCursorPosition / 100;
      this.beliefLocked = true;
    },
    // Clear belief selection locks when re-entering the slider.
    unlockBeliefPanel() {
      // clear lock as soon as they exit the slider
      this.beliefLocked = false;
      this.beliefSelected = null;
    },
    // Send the participant's initial belief to the server.
    sendInitialChoice(nodeBeliefs = null) {
      const msg = {
        type: "make_choice",
        initial: true,
        belief: this.initialTargetBelief,
      };
      if (nodeBeliefs && Object.keys(nodeBeliefs).length > 0) {
        msg.node_beliefs = nodeBeliefs;
      }
      this.ws.send(JSON.stringify(msg));
    },
    // Process confirmations for initial, final, or serial beliefs.
    confirmBelief() {
      if (this.initialDecisionPending) {
        testAssert(this.beliefSelected !== null, "No belief selected");
        if (this.beliefSurveyEnabled && this.beliefSurveyCurrentItem) {
          const phaseComplete = this.recordBeliefSurveyResponse(
            this.beliefSelected,
          );
          if (!phaseComplete) {
            this.beliefSelected = null;
            return;
          }
          this.initialNodeBeliefs = this.currentBeliefSurveyNodePayload();
          this.sendInitialChoice(this.initialNodeBeliefs);
          this.resetBeliefSurveyPhase();
          this.popover = {
            title: "Survey Complete",
            message:
              "You will now discuss the main proposition shown at the top of the page.",
          };
        } else {
          this.initialTargetBelief = this.beliefSelected;
          this.sendInitialChoice();
        }

        this.initialDecisionPending = false;
        // Set the initial anchor to the confirmed initial belief (percent)
        const initialBeliefForAnchor =
          typeof this.initialTargetBelief === "number"
            ? this.initialTargetBelief
            : this.beliefSelected;
        this.lastTraceAnchorPercent = Math.min(
          100,
          Math.max(0, initialBeliefForAnchor * 100),
        );
        if (
          this.isTarget &&
          !this.controlPromptTransitionAcknowledged &&
          this.hasControlPromptTransition()
        ) {
          this.showControlPromptTransitionPopover();
        } else {
          // now that initial is set, play back any buffered messages:
          this.flushPendingResponses();
        }
      } else if (this.finalDecisionPending) {
        testAssert(this.beliefSelected !== null, "No belief selected");
        let nodeBeliefs = null;
        if (this.beliefSurveyEnabled && this.beliefSurveyCurrentItem) {
          const phaseComplete = this.recordBeliefSurveyResponse(
            this.beliefSelected,
          );
          if (!phaseComplete) {
            this.beliefSelected = null;
            return;
          }
          this.finalNodeBeliefs = this.currentBeliefSurveyNodePayload();
          nodeBeliefs = this.finalNodeBeliefs;
          this.resetBeliefSurveyPhase();
        } else {
          this.finalTargetBelief = this.beliefSelected;
        }

        const msg = {
          type: "make_choice",
          initial: false,
          belief: this.finalTargetBelief,
        };
        if (nodeBeliefs && Object.keys(nodeBeliefs).length > 0) {
          msg.node_beliefs = nodeBeliefs;
        }
        this.ws.send(JSON.stringify(msg));

        this.finalDecisionPending = false;

        if (this.onReflection && this.isTarget) {
          this.showOnReflection();
        }

        // NB: We don't set this.finalDecisionPending to false here
        // because we don't want them to go back to the chat screen
        this.waitingForResult();
      } else if (this.serialDecisionPending) {
        testAssert(this.beliefSelected !== null, "No belief selected");
        if (this.serialQuestionsSentence) {
          if (!Array.isArray(this.serialSentenceResponsesPending)) {
            this.serialSentenceResponsesPending = [];
          }
          this.serialSentenceResponsesPending.push(this.beliefSelected);
          this.advanceSerialSentenceAfterResponse();
        } else {
          // Hide the panel;
          // the decision gets sent in the web socket.
          this.serialTargetBelief = this.beliefSelected;
          // your existing serial‐questions path bundles this on the next WS send
          this.serialDecisionPending = false;
          // after 2s grace, start recording user response
          if (this.useAudio) {
            setTimeout(() => this.startRecording(), api.redirectMilliseconds());
          }
        }
      }
      // TODO: see the repeated code?
      // reset for the next time
      this.beliefSelected = null;
      this.beliefCursorPosition = 50;
      this.beliefLocked = false;
    },
  },
};
</script>

<style scoped>
:global(:root) {
  --highlight-bg: rgba(126, 189, 255, 0.4);
  --highlight-bg-muted: rgba(126, 189, 255, 0.25);
  --highlight-panel-bg: #e6f0ff;
  --highlight-panel-border: #9bb8ff;
  --highlight-panel-text: #1f2d50;
  --highlight-action-border: #7694ff;
  --highlight-action-hover: #d9e5ff;
  --highlight-action-disabled-bg: rgba(31, 45, 80, 0.4);
  --highlight-action-disabled-text: rgba(255, 255, 255, 0.85);
}

/* Audio styling */
/* container styling if you want to align them nicely */
.audio-panel {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* for the recording and waiting indicators */
.audio-buttons {
  display: flex;
  flex-direction: column;
  padding-bottom: 0.5em;
}

.audio-buttons > * {
  margin: auto;
}

/* Reserve space so the layout doesn’t jump when the status changes */
.audio-status-slot {
  min-height: 2rem; /* enough for one-line indicators */
  display: flex;
  align-items: center;
  justify-content: center;
}

/* Optional: keep exact height if you prefer no growth */
.status-placeholder {
  height: 1rem;
}

/* the red "live" dot */
.recording-indicator {
  display: flex;
  align-items: center;
  font-weight: bold;
  color: #d00;
}
.red-dot {
  width: 0.75rem;
  height: 0.75rem;
  background: red;
  border-radius: 50%;
  margin-right: 0.4rem;
  /* pulsing blink */
  animation: blink 1s infinite;
}
@keyframes blink {
  0%,
  60%,
  100% {
    opacity: 1;
  }
  30% {
    opacity: 0.2;
  }
}

/*  */
.round-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  align-items: center; /* center children horizontally */
}

/* Make all direct children of round-container centered and <= 800px */
.round-container > * {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
}

.prompt-panel {
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  padding-left: 1rem;
  padding-right: 1rem;
  border-bottom: 1px solid #ccc;
  border-left: 4px solid transparent;
  border-right: 1px solid #eee;
  border-top: 1px solid #eee;
  border-radius: 6px;
  margin-top: 0.5rem;
  background: #fff;
}

.prompt-phase-badge {
  display: inline-block;
  float: right;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  margin-bottom: 0.4rem;
  color: #222;
  background: #e9eef7;
}

.prompt-panel.phase-initial,
.prompt-panel.phase-final,
.prompt-panel.phase-discussion {
  border-left-color: #1976d2;
}
.prompt-panel.phase-control {
  border-left-color: #6a1b9a;
}

/* Slight emphasis if the viewer is the target */
.prompt-panel.phase-target {
  box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.03) inset;
}

.prompt-panel :deep(blockquote) {
  font-weight: bold;
  font-size: larger;
}

.prompt-body-survey .survey-prompt-intro {
  margin: 0 0 0.6rem 0;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.belief-slider-panel {
  padding: 1rem;
  text-align: center;
}

.belief-survey-progress {
  margin-top: 0.5rem;
  font-weight: 600;
}

.serial-sentence-progress {
  margin-top: 0.5rem;
  font-weight: 600;
}

.serial-sentence-quote {
  margin: 0.75rem auto;
  max-width: 720px;
  font-style: italic;
  line-height: 1.5;
}

.highlight-prompt {
  margin: 0.5rem 0 0;
  padding: 0.65rem 0.85rem;
  background-color: var(--highlight-panel-bg);
  border: 1px solid var(--highlight-panel-border);
  border-radius: 4px;
  font-size: 0.95rem;
  color: var(--highlight-panel-text);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.highlight-prompt p {
  margin: 0;
}

.highlight-prompt p + p {
  margin-top: 0.35rem;
}

.highlight-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.65rem;
}

.highlight-action-button {
  border: 1px solid var(--highlight-action-border);
  background: var(--highlight-panel-bg);
  color: var(--highlight-panel-text);
  border-radius: 4px;
  padding: 0.35rem 0.75rem;
  font-size: 0.9rem;
  cursor: pointer;
}

.highlight-action-button:hover,
.highlight-action-button:focus {
  background: var(--highlight-action-hover);
  outline: none;
}

.highlight-action-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.highlight-action-button.highlight-continue {
  background: var(--highlight-panel-text);
  border-color: var(--highlight-panel-text);
  color: #fff;
}

.highlight-action-button.highlight-continue:hover,
.highlight-action-button.highlight-continue:focus {
  background: var(--highlight-panel-text);
  color: #fff;
  opacity: 0.9;
}

.highlight-action-button.highlight-continue:disabled {
  background: var(--highlight-action-disabled-bg);
  border-color: var(--highlight-action-disabled-bg);
  color: var(--highlight-action-disabled-text);
}

.round-info {
  display: flex;
  justify-content: center;
  gap: 2em;
}

#conversation {
  width: 800px; /* fixed width */
  max-width: 800px;
}

.secondary-button {
  background: transparent;
  border: none;
  color: #1f6feb;
  cursor: pointer;
  padding: 0;
}

.secondary-button:hover {
  text-decoration: underline;
}

/* mouse-trace */

/* share a width var between slider and buttons */
.mouse-trace-wrapper {
  --slider-width: 400px; /* single source of truth */
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 1rem;
}

.mouse-trace-panel {
  position: relative;
  width: var(--slider-width); /* use shared width */
  height: 3rem;
  margin: 0 auto;
  cursor: pointer;
  user-select: none;
  border: 1px solid black;
  border-radius: 4px;
  box-sizing: border-box;
}

.slider-width-button {
  width: var(--slider-width);
  max-width: 100%;
  box-sizing: border-box;
  display: block;
  margin: 0 auto;
}

/* The thin line showing the current mouse-trace position */
.mouse-trace-cursor {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #333;
  transform: translateX(-50%);
  pointer-events: none; /* let clicks through */
}

/* outside labels */
.trace-label {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 5rem; /* enough room to center your text */
  text-align: center;
  font-weight: bold;
  pointer-events: none; /* so clicks go through to the panel */
}

.trace-label.disagree {
  left: -6rem; /* sit just outside the left edge */
  text-align: right;
}

.trace-label.agree {
  right: -6rem; /* sit just outside the right edge */
  text-align: left;
}

/* hide the native range control over the panel */
.hidden-range {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: ew-resize;
}

/* percentage text */
.slider-percent {
  text-align: center;
  font-size: 1.2rem;
}

.mouse-trace-instructions {
  text-align: center;
  /*min-height: 4.5rem; /* keeps space constant */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.mouse-trace-panel.locked {
  opacity: 0.6;
}

.start-prompt {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column; /* stack dot above text */
  gap: 0.35rem; /* space between dot and text */
  align-items: center;
  justify-content: center;
  font-weight: bold;
  cursor: default;
}

.cursor-dot {
  position: absolute; /* position relative to the panel */
  top: 50%;
  width: 10px;
  height: 10px;
  background: red;
  border-radius: 50%;
  border: 2px solid #fff; /* improve visibility */
  z-index: 2;
  pointer-events: none; /* do not block clicks */
  animation: pulse-center 1.2s ease-in-out infinite;
}

@keyframes pulse-center {
  0%,
  100% {
    transform: scale(0.9);
    opacity: 0.7;
  }
  50% {
    transform: scale(1.1);
    opacity: 1;
  }
}
</style>
