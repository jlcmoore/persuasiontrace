// frontend/tests/unit/FeedbackVue.spec.js

import { mount } from "@vue/test-utils";
import FeedbackVue from "@/components/FeedbackVue.vue";
import { api } from "@/api";

jest.mock("@/api", () => ({
  api: {
    sendFeedback: jest.fn(),
  },
}));

const push = jest.fn();
const $router = { push };

beforeEach(() => {
  localStorage.setItem("participantId", "123");
  jest.clearAllMocks();
});

it("binds textarea, calls sendFeedback and routes to /debrief", async () => {
  const wrapper = mount(FeedbackVue, { global: { mocks: { $router } } });
  const ta = wrapper.find("textarea");
  await ta.setValue("My feedback!");
  await wrapper.find("button").trigger("click");
  expect(api.sendFeedback).toHaveBeenCalledWith("123", "My feedback!");
  // sendToDebrief awaits sendFeedback, then pushes
  expect(push).toHaveBeenCalledWith("/debrief");
});
