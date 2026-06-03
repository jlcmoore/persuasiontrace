<!-- 
Author: Jared Moore
Date: July, 2025
frontend/src/components/round/ChatVue.vue
-->

<template>
  <section class="chat">
    <div id="chat-window" ref="chatWindow" class="chat-window">
      <div
        v-for="(message, index) in filteredChatMessages"
        :key="index"
        :class="getMessageClass(message)"
        @mouseup="handleMouseUp(index, $event)"
      >
        <span
          v-for="(segment, segmentIndex) in getMessageSegments(message)"
          :key="segmentIndex"
          :class="{ 'highlight-segment': segment.highlighted }"
        >
          {{ segment.text }}
        </span>
      </div>
      <div v-if="typing" class="message message-left">
        <div class="message-text typing">
          <div class="dot"></div>
          <div class="dot"></div>
          <div class="dot"></div>
        </div>
      </div>
    </div>

    <div v-if="allowInput">
      <div class="chat-input" :class="{ disabled: continuousMeasurePending }">
        <!-- Binding the input's value and emitting the update event -->
        <div class="textareas-row">
          <textarea
            id="chat-input"
            ref="chatInput"
            :value="newMessage"
            :disabled="continuousMeasurePending"
            class="single-input"
            placeholder="Type your message here..."
            @input="updateMessage($event.target.value)"
            @keyup.enter="handleEnter"
            @paste.prevent
            @contextmenu.prevent
          />
        </div>
        <div class="row">
          <div class="chat-info">
            <p
              class="character-count"
              :class="{ warning: newMessage.length >= maxMessageChars }"
            >
              {{ newMessage.length }}/{{ maxMessageChars }}
            </p>
          </div>
          <button
            id="send-button"
            :disabled="!canSendMessage"
            @click="sendMessage"
          >
            <img src="@/assets/send.svg" />
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
import { getSelectionOffsets } from "./highlightUtils";

export default {
  props: {
    chatMessages: {
      type: Array,
      default: () => [],
    },
    newMessage: {
      type: String,
      default: "",
    },
    continuousMeasurePending: {
      type: Boolean,
      default: false,
    },
    allowInput: {
      type: Boolean,
      default: false,
    },
    canSendMessageExterior: Boolean,
    typing: Boolean,
    isTarget: Boolean,
    maxMessageChars: {
      type: Number,
      required: true,
    },
    highlightEnabled: {
      type: Boolean,
      default: false,
    },
  },
  emits: ["add-highlight", "update:newMessage", "send-message"],
  data() {
    return {};
  },
  computed: {
    canSendMessage() {
      if (
        this.newMessage.trim().length < 1 ||
        this.newMessage.length >= this.maxMessageChars ||
        !this.canSendMessageExterior
      ) {
        return false;
      }
      if (this.chatMessages.length < 1) {
        if (this.isTarget == false) {
          return true;
        }
        return false;
      }
      const lastMessage = this.chatMessages[this.chatMessages.length - 1];
      return lastMessage.sender !== "You";
    },
    filteredChatMessages() {
      // only messages where hide is undefined or false get shown
      return this.chatMessages.filter((m) => m.hide !== true);
    },
  },
  watch: {
    // Auto-scroll when new messages arrive or visibility changes
    chatMessages: {
      handler() {
        this.$nextTick(this.scrollToBottom);
      },
      deep: true,
    },
    // Also scroll when typing indicator changes (to keep it in view)
    typing() {
      this.$nextTick(this.scrollToBottom);
    },
  },
  mounted() {
    this.$nextTick(this.scrollToBottom);
  },
  methods: {
    // Split a message into highlight-aware segments for rendering.
    getMessageSegments(message) {
      const text = message?.text ?? message?.content ?? "";
      if (!text) {
        return [];
      }
      const ranges = Array.isArray(message?.highlightRanges)
        ? message.highlightRanges
        : message?.highlightRange
          ? [message.highlightRange]
          : [];
      return this.buildSegments(text, ranges);
    },
    // Merge overlapping highlight ranges and generate contiguous text segments.
    buildSegments(text, ranges) {
      if (!Array.isArray(ranges) || ranges.length === 0) {
        return text ? [{ text, highlighted: false }] : [];
      }
      const textLength = text.length;
      const normalized = ranges
        .map((range) => {
          const rawStart = range?.start ?? 0;
          const rawEnd = range?.end ?? textLength;
          const start = Math.max(0, Math.min(rawStart, textLength));
          const end = Math.max(start, Math.min(rawEnd, textLength));
          return end > start ? { start, end } : null;
        })
        .filter((range) => range !== null)
        .sort((a, b) => a.start - b.start);
      if (normalized.length === 0) {
        return [{ text, highlighted: false }];
      }
      const merged = [{ ...normalized[0] }];
      for (let i = 1; i < normalized.length; i += 1) {
        const current = normalized[i];
        const last = merged[merged.length - 1];
        if (current.start <= last.end) {
          last.end = Math.max(last.end, current.end);
        } else {
          merged.push({ ...current });
        }
      }
      const segments = [];
      let cursor = 0;
      merged.forEach((range) => {
        if (range.start > cursor) {
          segments.push({
            text: text.slice(cursor, range.start),
            highlighted: false,
          });
        }
        segments.push({
          text: text.slice(range.start, range.end),
          highlighted: true,
        });
        cursor = range.end;
      });
      if (cursor < text.length) {
        segments.push({
          text: text.slice(cursor),
          highlighted: false,
        });
      }
      return segments.filter((segment) => segment.text.length > 0);
    },
    getMessageClass(message) {
      let msgClass = "message";
      if (message.sender === "You") {
        msgClass += " message-right";
      } else {
        msgClass += " message-left";
      }
      if (message.highlighted) {
        msgClass += " highlighted";
      }
      return msgClass;
    },
    // Convert a mouse selection into highlight offsets and emit them upstream.
    handleMouseUp(index, event) {
      if (!this.highlightEnabled) {
        return;
      }
      const container = event?.currentTarget;
      if (!container) {
        return;
      }
      const selection = window.getSelection();
      if (!selection || selection.isCollapsed) {
        return;
      }
      let range;
      try {
        range = selection.getRangeAt(0);
      } catch {
        return;
      }
      if (!container.contains(range.commonAncestorContainer)) {
        return;
      }
      const { start, end } = this.getOffsets(container, range);
      selection.removeAllRanges();
      if (end <= start) {
        return;
      }
      this.$emit("add-highlight", {
        messageIndex: index,
        start,
        end,
      });
    },
    // Safely compute the selection offsets relative to the chat bubble.
    getOffsets(container, range) {
      try {
        return getSelectionOffsets(container, range);
      } catch (error) {
        console.warn("Failed to compute highlight offsets", error);
        return { start: 0, end: 0 };
      }
    },
    updateMessage(value) {
      this.$emit("update:newMessage", value);

      this.adjustTextareaHeight("chat-input");
    },

    adjustTextareaHeight(textareaId = null) {
      // If no specific textarea ID provided, adjust both
      if (!textareaId) {
        this.adjustTextareaHeight("chat-input");
        return;
      }

      const textarea = this.$refs[textareaId.replace("-", "")];
      if (textarea) {
        textarea.style.height = "auto"; // Reset height to calculate scroll height
        textarea.style.height = `${Math.min(textarea.scrollHeight, 80)}px`; // Max height for 4 lines
      }
    },
    async handleEnter(event) {
      if (!event.shiftKey) {
        event.preventDefault();
        await this.sendMessage();
      }
    },

    scrollToBottom() {
      const el = this.$refs.chatWindow;
      if (el) {
        el.scrollTop = el.scrollHeight;
      }
    },

    async sendMessage() {
      if (this.canSendMessage) {
        // Emit message
        this.$emit("send-message", {
          message: this.newMessage,
        });
        this.$emit("update:newMessage", ""); // Clear the message
        this.resetTextareaHeight(); // Reset the textarea height
        this.$nextTick(this.scrollToBottom);
      } else {
        console.error(
          "You cannot send an empty message or send two messages in a row",
        );
      }
    },
    resetTextareaHeight(textareaId = null) {
      // If no specific textarea ID provided, reset both
      if (!textareaId) {
        this.resetTextareaHeight("chat-input");
        return;
      }

      const textarea = this.$refs[textareaId.replace("-", "")]; // remove hyphen to match ref name
      if (textarea) {
        textarea.style.height = "auto";
      }
    },
  },
};
</script>

<style scoped>
section.chat {
  /* Don't force the chat section to fill the entire column height */
  flex: 0 0 auto;
  padding-top: 20px;
  padding-bottom: 20px;
  display: flex;
  flex-direction: column;
  width: 100%;
  /* height: auto by default */
}

.chat-header {
  display: flex;
  justify-content: space-between;
}

button.decide-button {
  /* height: 20px; */
  padding: 19px 20px;
  height: 30px;
  line-height: 0px;
}

/* Fixed, scrollable chat window */
.chat-window {
  height: var(--chat-window-height, clamp(200px, 35vh, 320px));
  max-height: 50vh; /* extra guard on very small screens */
  overflow-y: auto;
}

.chat-input {
  display: flex;
  align-items: end;
  justify-content: space-between;
  padding: 10px;
}

.chat-input textarea {
  flex: 1;
  height: 40px;
  /*  border-radius: 20px;*/
  background: #373636;
  box-shadow: none;
  border: none;
  color: white;
  resize: none;
  overflow: hidden;
  max-height: 80px;
  line-height: 20px;
  height: auto;
  width: 100%;
  padding: 10px;
  font-family: "Source Sans Pro", sans-serif;
}

.chat-input button {
  height: 40px;
  width: 40px;
  padding: 5px;
  border-radius: 20px;
  border: none;
  cursor: pointer;
}

/* Style for bottom textarea */
#chat-input:not(.single-input) {
  border-bottom-left-radius: 20px;
  border-bottom-right-radius: 20px;
}

/*  a container for the bottom row with send button */
.textareas-row {
  display: flex;
  align-items: center;
  flex-direction: column;
  width: 100%;
  margin: 0 13px;
}

/* Style for when there's only one input */
.single-input {
  border-radius: 20px !important;
}

button img {
  width: 30px;
  height: 30px;
}

#send-button {
  background-color: #007bff;
  color: white;
  padding-left: 7px;
}

#send-button:hover {
  background-color: #0056b3;
}

#send-button:disabled {
  background-color: #5e7a98;
  cursor: not-allowed;
}

button#gavel-button {
  background-color: orange;
}

button#gavel-button:hover {
  background-color: rgb(229, 125, 5);
}

/* Chat Message Styles */

.message {
  max-width: 70%;
  width: fit-content;
  font-weight: 300;
  padding: 8px 16px;
  border-radius: 10px;
  margin-bottom: 20px;
  word-wrap: break-word;
  white-space: pre-wrap;
  line-height: 1.4;
  position: relative;
  font-size: 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.highlight-segment {
  background-color: var(--highlight-bg);
  color: inherit;
}

.message-left {
  background-color: #333;
  color: white;
}

.message-right {
  background-color: #e0f7fa;
  margin-left: auto;
}

.message.highlighted {
  border: 1px solid #ffd166;
  box-shadow: 0 0 0 2px rgba(255, 209, 102, 0.45);
}

.message-left.highlighted {
  background-color: #444a63;
}

.message-right.highlighted {
  background-color: #fff3dc;
}

p.character-count {
  margin: 0;
  font-size: 16px;
}

p.character-count.warning {
  color: red;
}

/* Typing animation */

.typing {
  align-items: center;
  display: flex;
  height: 17px;
  margin-top: 5px;
}

.typing .dot {
  animation: mercuryTypingAnimation 1.8s infinite ease-in-out;
  background-color: #666;
  border-radius: 50%;
  height: 7px;
  margin-right: 4px;
  vertical-align: middle;
  width: 7px;
  display: inline-block;
}

.typing .dot:nth-child(1) {
  animation-delay: 200ms;
}

.typing .dot:nth-child(2) {
  animation-delay: 300ms;
}

.typing .dot:nth-child(3) {
  animation-delay: 400ms;
}

.typing .dot:last-child {
  margin-right: 0;
}

@keyframes mercuryTypingAnimation {
  0% {
    transform: translateY(0px);
    background-color: #666;
  }

  28% {
    transform: translateY(-7px);
    background-color: #888;
  }

  44% {
    transform: translateY(0px);
    background-color: #666;
  }
}

#timeout {
  font-style: italic;
  font-size: small;
}
</style>
