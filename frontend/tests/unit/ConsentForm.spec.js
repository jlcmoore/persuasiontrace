// tests/unit/ConsentForm.spec.js

import { mount } from "@vue/test-utils";
import ConsentForm from "@/components/ConsentForm.vue";
import { api } from "@/api";

const push = jest.fn();
const $router = { push };

jest.mock("@/api", () => ({
  api: {
    mayUseAudio: jest.fn(() => false),
  },
}));

it("navigates to /pre-lobby when I Agree clicked", async () => {
  const wrapper = mount(ConsentForm, { global: { mocks: { $router } } });
  await wrapper.find("button.consent-button").trigger("click");
  expect(push).toHaveBeenCalledWith("/pre-lobby");
});
