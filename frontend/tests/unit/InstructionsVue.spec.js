// tests/unit/InstructionsVue.spec.js

import { mount, flushPromises } from "@vue/test-utils";
import InstructionsVue from "@/components/round/InstructionsVue.vue";
import { api } from "@/api";

jest.useFakeTimers();

jest.mock("@/api", () => ({
  api: {
    getParticipantInstructions: jest.fn(),
  },
}));

it("renders instructions and emits close when button clicked", async () => {
  const wrapper = mount(InstructionsVue, {
    props: {
      instructions: "<p>test instructions</p>",
    },
  });

  expect(wrapper.text()).toContain("test instructions");

  // Skip the 30s timer
  await wrapper.setData({ canProceed: true });

  await wrapper.find("button").trigger("click");
  expect(wrapper.emitted("close")).toBeTruthy();
});
