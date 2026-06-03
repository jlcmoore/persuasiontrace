// frontend/tests/unit/WelcomeVue.spec.js

import { mount, flushPromises } from "@vue/test-utils";
import WelcomeVue from "@/components/WelcomeVue.vue";
import { api } from "@/api";

jest.mock("@/api", () => ({
  api: {
    generateFakeID: jest.fn(() => "FAKEID"),
    initializeParticipant: jest.fn().mockResolvedValue({ participant_id: 42 }),
    isDevelopmentMode: jest.fn(),
    maxMessageChars: jest.fn(() => 300),
    mayUseAudio: jest.fn(() => false),
  },
}));

jest.mock("@/utils", () => ({
  generateFakeID: jest.fn(() => "FAKEID"),
}));

const push = jest.fn();
const $router = { push };

// Mock window.alert
global.alert = jest.fn();

beforeEach(() => {
  localStorage.clear();
  jest.clearAllMocks();
});

it("clears localStorage, sets IDs, and navigates to /lobby in dev", async () => {
  api.isDevelopmentMode.mockReturnValue(true);
  const wrapper = mount(WelcomeVue, { global: { mocks: { $router } } });
  
  // Set required data for canContinue to be true
  await wrapper.setData({ turnstileToken: "dummy-token" });
  
  await wrapper.find("button").trigger("click");
  expect(localStorage.getItem("prolificId")).toBe("FAKEID");
  expect(api.initializeParticipant).toHaveBeenCalledWith("FAKEID", "dummy-token");
  await flushPromises();
  expect(localStorage.getItem("participantId")).toBe("42");
  expect(push).toHaveBeenCalledWith("/pre-lobby");
});

it("navigates to /consent when not in dev", async () => {
  api.isDevelopmentMode.mockReturnValue(false);
  const wrapper = mount(WelcomeVue, { global: { mocks: { $router } } });
  
  // Set required checkboxes and token for non-dev
  await wrapper.setData({
    aiConfirmed: true,
    turnstileToken: "dummy-token"
  });
  
  await wrapper.find("button").trigger("click");
  await flushPromises();
  expect(push).toHaveBeenCalledWith("/consent");
});
