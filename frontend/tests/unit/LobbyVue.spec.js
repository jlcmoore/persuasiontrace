// frontend/tests/unit/LobbyVue.spec.js

import { mount, shallowMount, flushPromises } from "@vue/test-utils";
import LobbyVue from "@/components/LobbyVue.vue";
import { api } from "@/api";

jest.useFakeTimers();

jest.mock("@/api", () => ({
  api: {
    participantReady: jest.fn(),
    getCurrentRound: jest.fn(),
    isDevelopmentMode: jest.fn(),
    redirectMilliseconds: jest.fn(() => 0),
    useAudio: jest.fn(),
    mayUseAudio: jest.fn(),
    maxMessageChars: jest.fn(() => 300),
  },
}));

describe("LobbyVue", () => {
  let push;
  let $router;

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    push = jest.fn();
    $router = { push };
    // simulate that we have a stored participantId
    localStorage.setItem("participantId", "123");
    // default: not development mode
    api.isDevelopmentMode.mockReturnValue(false);
  });

  it("calls participantReady then polls getCurrentRound when waiting, and finally navigates to /round", async () => {
    // Simulate that the user has already fetched instructions
    localStorage.setItem("instructions", "<p>dummy</p>");
    // 1st call returns a waiting_time -> we stay in lobby
    // 2nd call returns full round data -> we navigate
    api.getCurrentRound
      .mockResolvedValueOnce({ waiting_time: 5 })
      .mockResolvedValueOnce({ round_id: 99, is_target: true, prompt: "P" });

    const wrapper = shallowMount(LobbyVue, {
      global: { mocks: { $router } },
    });

    // let participantReady() + first getCurrentRound() finish
    await flushPromises();

    // participantReady should have been called with our ID
    expect(api.participantReady).toHaveBeenCalledWith("123");
    // getCurrentRound should have been called exactly once so far
    expect(api.getCurrentRound).toHaveBeenCalledTimes(1);

    // the component's data.waitingTime should be set to 5
    expect(wrapper.vm.waitingTime).toBe(5);

    // advance timers by the 2s retry delay
    jest.advanceTimersByTime(2000);
    // let the second getCurrentRound() finish
    await flushPromises();

    // now getCurrentRound has been called again
    expect(api.getCurrentRound).toHaveBeenCalledTimes(2);

    // and after we get real round data, we should navigate to "/round"
    jest.runOnlyPendingTimers();
    // give Vue a tick to actually call push()
    await flushPromises();

    expect(push).toHaveBeenCalledWith("/round");
  });

  it("navigates to /feedback if getCurrentRound throws a 'has waited too long' error", async () => {
    // Mock an Axios‐style 400 error as handled by our api.js interceptor:
    api.getCurrentRound.mockRejectedValueOnce({
      message: "Participant 123 has waited too long.",
      status: 400,
    });

    shallowMount(LobbyVue, {
      global: { mocks: { $router } },
    });

    // flush through participantReady() and the first (failing) getCurrentRound()
    await flushPromises();

    // because the error message matches, we should have pushed to "/feedback"
    expect(push).toHaveBeenCalledWith("/feedback");
  });

  it("in dev mode, on forced-params error redirects to /round-setup", async () => {
    api.isDevelopmentMode.mockReturnValue(true);
    // simulate that there were forced params in storage
    localStorage.setItem(
      "current_round_params",
      JSON.stringify({ foo: "bar" }),
    );
    // have getCurrentRound throw some non‐timeout error
    api.getCurrentRound.mockRejectedValueOnce({
      message: "some development-only error",
      status: 500,
    });

    shallowMount(LobbyVue, {
      global: { mocks: { $router } },
    });

    // flush through participantReady() and the first getCurrentRound()
    await flushPromises();

    // because we are in dev _and_ we had forced params, we should go to "/round-setup"
    expect(push).toHaveBeenCalledWith("/round-setup");
  });

  it("ignores generic failures and keeps polling", async () => {
    // first call fails with a generic error
    api.getCurrentRound
      .mockRejectedValueOnce({ message: "Network down", status: 500 })
      // second call returns a waiting_time
      .mockResolvedValueOnce({ waiting_time: 2 });

    const wrapper = shallowMount(LobbyVue, {
      global: { mocks: { $router } },
    });

    // flush through participantReady() and first failing getCurrentRound()
    await flushPromises();

    // we should _not_ have navigated anywhere yet
    expect(push).not.toHaveBeenCalled();
    // getCurrentRound was called once
    expect(api.getCurrentRound).toHaveBeenCalledTimes(1);

    // advance timers to trigger the retry
    jest.advanceTimersByTime(2000);
    await flushPromises();

    // now it should have tried again

    expect(api.getCurrentRound.mock.calls.length).toBeGreaterThanOrEqual(2);
    // still no navigation (we only saw waiting_time)
    expect(push).not.toHaveBeenCalled();

    // and the component's waitingTime should have been updated
    expect(wrapper.vm.waitingTime).toBe(2);
  });

  describe("audio gating", () => {
    const pushMock = jest.fn();

    it("redirects to /audio-setup when audio is on and no audioOK in localStorage", () => {
      api.mayUseAudio.mockReturnValue(true);
      localStorage.removeItem("audioOK");

      // mount a fresh instance
      mount(LobbyVue, {
        global: {
          mocks: { $router: { push: pushMock } },
        },
      });

      expect(pushMock).toHaveBeenCalledWith("/audio-setup");
    });

    it("does NOT redirect when audioOK flag is present", () => {
      api.mayUseAudio.mockReturnValue(true);
      localStorage.setItem("audioOK", "1");

      // mount a fresh instance
      mount(LobbyVue, {
        global: {
          mocks: { $router: { push: pushMock } },
        },
      });

      expect(pushMock).not.toHaveBeenCalledWith("/audio-setup");
    });
  });
});
