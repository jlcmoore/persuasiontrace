// frontend/tests/unit/DebriefVue.spec.js

import { mount, flushPromises } from "@vue/test-utils";
import DebriefVue from "@/components/DebriefVue.vue";
import { api } from "@/api";

jest.mock("@/api", () => ({
  api: {
    getParticipantRounds: jest.fn(),
  },
}));

beforeEach(() => {
  localStorage.setItem("participantId", "55");
  jest.clearAllMocks();
});

it("fetches rounds data and displays stats", async () => {
  api.getParticipantRounds.mockResolvedValue({
    num_human_conversations: 2,
    total_rounds: 5, // Fix: total_rounds instead of rounds_completed
    completion_code: "XYZ123",
  });
  const wrapper = mount(DebriefVue);
  await flushPromises();
  expect(api.getParticipantRounds).toHaveBeenCalledWith("55");
  expect(wrapper.vm.humanConversations).toBe(2);
  expect(wrapper.vm.totalRounds).toBe(5);
  expect(wrapper.vm.completionCode).toBe("XYZ123");
  expect(wrapper.text()).toContain(
    "You interacted with 2 real human participants",
  );
});

it("redirects to Prolific with the code on button click", async () => {
  api.getParticipantRounds.mockResolvedValue({
    num_human_conversations: 0,
    total_rounds: 0,
    completion_code: "ABC",
  });
  delete window.location;
  window.location = { href: "" };
  const wrapper = mount(DebriefVue);
  await flushPromises();
  await wrapper.find("button").trigger("click");
  expect(window.location.href).toBe(
    "https://app.prolific.co/submissions/complete?cc=ABC",
  );
});
