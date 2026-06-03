// tests/unit/RoundVue.spec.js

import { mount } from "@vue/test-utils";
import RoundVue from "@/components/round/RoundVue.vue";

// We stub out WebSocket globally so that any `new WebSocket(...)` in
// openWebSocket() won’t crash the tests in Node.
global.WebSocket = class {
  constructor(url) {
    this.url = url;
    // simulate immediate open
    setTimeout(() => this.onopen && this.onopen(), 0);
  }
  send() {}
  close() {
    this.onclose && this.onclose();
  }
};

// mock our api module
jest.mock("@/api", () => ({
  api: {
    makeChoice: jest.fn(() => Promise.resolve()),
    isDevelopmentMode: jest.fn(() => false),
    useAudio: jest.fn(() => false), // default: audio off
    redirectMilliseconds: jest.fn(() => 0),
    continuousMeasure: jest.fn(),
    getRoundResult: jest.fn(),
    maxMessageChars: jest.fn(() => 300),
    mayUseAudio: jest.fn(() => false),
    getCurrentRound: jest.fn(() =>
      Promise.resolve({
        round_id: 100,
        is_target: false,
        prompt: "<p>Test prompt</p>",
        turns_left: 3,
      }),
    ),
    getParticipantRounds: jest.fn(() =>
      Promise.resolve({ num_rounds: 1, total_rounds: 1 }),
    ),
    getParticipantInstructions: jest.fn(() =>
      Promise.resolve({ instructions: "<p>test instructions</p>" }),
    ),
  },
}));

// Mock audio module
jest.mock("@/audio", () => ({
  playBase64Audio: jest.fn(),
  getWarmStream: jest.fn(),
  getAudioContext: jest.fn(),
  warmupMicrophone: jest.fn(),
}));

const $router = {
  push: jest.fn(),
};

describe("RoundVue.vue", () => {
  let wrapper;

  beforeEach(async () => {
    jest.clearAllMocks();
    localStorage.clear();
    localStorage.setItem("participantId", "42");

    // mount with a stub for Chat and a fake router
    wrapper = mount(RoundVue, {
      global: {
        stubs: {
          Chat: { template: "<div class='chat-stub'></div>" },
          Popover: { template: "<div class='popover-stub'><slot /></div>" },
          RoundResult: { template: "<div class='result-stub'></div>" },
          InstructionsVue: {
            template: "<div class='instructions-stub'></div>",
          },
        },
        mocks: {
          $router,
        },
      },
    });
    // the mounted() call in RoundVue triggers initRound(), which is async.
    // we need to wait for those promises to resolve before each test.
    await new Promise((resolve) => setTimeout(resolve, 0));
    
    // Skip the instructions overlay for most tests
    await wrapper.setData({ showInstructionsPending: false });
  });

  it("renders round-info and prompt and chat by default", () => {
    // default state: isTarget=false, no decision pending -> show chat stub
    const info = wrapper.find(".round-info").text();
    expect(info).toContain("Round: 1 of 1");
    expect(info).toContain("Turns remaining: 3");

    expect(wrapper.find(".prompt-body").html()).toContain("Test prompt");
    expect(wrapper.find(".chat-stub").exists()).toBe(true);
  });

  it("when target and initialDecisionPending is true, shows initial slider and button", async () => {
    await wrapper.setData({
      isTarget: true,
      initialDecisionPending: true,
    });
    // chat stub should NOT show when initial slider is pending
    expect(wrapper.find(".chat-stub").exists()).toBe(false);
    const panel = wrapper.find(".belief-slider-panel");
    expect(panel.exists()).toBe(true);

    // should have the custom slider (mouse-trace-panel)
    const slider = panel.find(".mouse-trace-panel");
    expect(slider.exists()).toBe(true);

    // button Confirm Initial Belief
    const btn = panel.find("button");
    expect(btn.text()).toContain("Confirm Initial Belief");
  });

  it("when target and finalDecisionPending is true, shows final slider and button", async () => {
    await wrapper.setData({
      isTarget: true,
      finalDecisionPending: true,
    });
    // chat stub should NOT show
    expect(wrapper.find(".chat-stub").exists()).toBe(false);
    const panel = wrapper.find(".belief-slider-panel");
    expect(panel.exists()).toBe(true);

    // button Confirm Final Belief
    const btn = panel.find("button");
    expect(btn.text()).toContain("Confirm Final Belief");
  });

  it("does not show slider for non-target even if flags set", async () => {
    await wrapper.setData({
      isTarget: false,
      initialDecisionPending: true,
    });
    expect(wrapper.find(".belief-slider-panel").exists()).toBe(false);
    // chat should still show
    expect(wrapper.find(".chat-stub").exists()).toBe(true);
  });

  test.each([
    ["Initial", { initialDecisionPending: true }],
    ["Final", { finalDecisionPending: true }],
  ])("disables Confirm %s Belief when beliefSelected is null", async (decisionType, stateFlags) => {
    const data = {
      isTarget: true,
      beliefSelected: null,
      ...stateFlags,
    };
    await wrapper.setData(data);
    const panel = wrapper.find(".belief-slider-panel");
    expect(panel.exists()).toBe(true);
    const btn = panel.find("button");
    expect(btn.text()).toContain(`Confirm ${decisionType} Belief`);
    expect(btn.attributes("disabled")).toBeDefined();
  });

  it("rounds and displays the percentage correctly for initial belief", async () => {
    await wrapper.setData({
      isTarget: true,
      initialDecisionPending: true,
      beliefCursorPosition: 25,
    });
    const pct = wrapper.find(".slider-percent");
    expect(pct.text()).toBe("25%");
  });

  it("rounds and displays the percentage correctly for final belief", async () => {
    await wrapper.setData({
      isTarget: true,
      finalDecisionPending: true,
      beliefCursorPosition: 87.6,
    });
    const pct = wrapper.find(".slider-percent");
    expect(pct.text()).toBe("88%");
  });

  it("does not show slider when turnsRemaining is zero but not yet finalDecisionPending", async () => {
    await wrapper.setData({
      isTarget: true,
      turnsRemaining: 0,
      finalDecisionPending: false,
    });
    expect(wrapper.find(".belief-slider-panel").exists()).toBe(false);
    expect(wrapper.find(".chat-stub").exists()).toBe(true);
  });
});
