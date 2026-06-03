// tests/unit/RoundSetup.spec.js
import { mount, flushPromises } from "@vue/test-utils";
import RoundSetup from "@/components/RoundSetup.vue";
import { api } from "@/api";
import { generateFakeID } from "@/utils";

jest.mock("@/api", () => ({
  api: {
    initializeParticipant: jest.fn(),
  },
}));

jest.mock("@/utils", () => ({
  generateFakeID: jest.fn(),
}));

describe("RoundSetup.vue", () => {
  let push;

  // helper to mount with a fake $router and $route.query
  function mountWith(query = {}) {
    push = jest.fn();
    return mount(RoundSetup, {
      global: {
        mocks: {
          $router: { push },
          $route: { query },
        },
      },
    });
  }

  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  it("renders inputs and does not auto-init when no query", () => {
    const wrapper = mountWith({});
    expect(api.initializeParticipant).not.toHaveBeenCalled();
    // We expect four text inputs: prolificId, llmTarget, llmPersuader, propositionID
    const textInputs = wrapper.findAll('input[type="text"]');
    expect(textInputs).toHaveLength(4);
    // radio buttons for isTarget
    const radios = wrapper.findAll('input[type="radio"]');
    expect(radios).toHaveLength(3);
    // one select for persuaderSupportsProposition
    expect(wrapper.findAll("select")).toHaveLength(1);
  });

  it("manually fills form, submits, stores params, calls API, and routes to /lobby", async () => {
    api.initializeParticipant.mockResolvedValue({ participant_id: 111 });

    const wrapper = mountWith({});
    const textInputs = wrapper.findAll('input[type="text"]');

    // 1) prolificId
    await textInputs[0].setValue("USER42");
    expect(wrapper.vm.prolificId).toBe("USER42");

    // 2) radio: Play as Target (value="true")
    const radios = wrapper.findAll('input[type="radio"]');
    await radios[0].setValue("true");
    expect(wrapper.vm.isTarget).toBe(true);

    // 3) llmTarget
    await textInputs[1].setValue("M_tgt");
    expect(wrapper.vm.llmTarget).toBe("M_tgt");

    // 4) llmPersuader
    await textInputs[2].setValue("M_persu");
    expect(wrapper.vm.llmPersuader).toBe("M_persu");

    // 5) propositionID
    await textInputs[3].setValue("prop42");
    expect(wrapper.vm.propositionID).toBe("prop42");

    // 6) select support
    const select = wrapper.find("select");
    await select.setValue("true");
    expect(wrapper.vm.persuaderSupportsProposition).toBe(true);

    // submit the form
    await wrapper.find("form").trigger("submit.prevent");
    await flushPromises();

    // Check localStorage
    expect(localStorage.getItem("prolificId")).toBe("USER42");
    const storedParams = JSON.parse(
      localStorage.getItem("current_round_params"),
    );
    expect(storedParams).toEqual({
      is_target: true,
      llm_target: "M_tgt",
      llm_persuader: "M_persu",
      proposition: "prop42",
      persuader_supports_proposition: true,
    });

    // API called and participantId stored
    expect(api.initializeParticipant).toHaveBeenCalledWith("USER42");
    expect(localStorage.getItem("participantId")).toBe("111");

    // Navigated to lobby
    expect(push).toHaveBeenCalledWith("/lobby");
  });

  it("auto-initializes when URL query params exist including blank prolificId", async () => {
    const query = {
      prolificId: "",
      isTarget: "false",
      llmTarget: "T1",
      llmPersuader: "P1",
      propositionID: "propX",
      persuaderSupportsProposition: "false",
    };
    generateFakeID.mockReturnValue("AUTOID");
    api.initializeParticipant.mockResolvedValue({ participant_id: 222 });

    mountWith(query);
    // mounted() schedules initialize() on nextTick -> promise
    await flushPromises();

    // blank prolificId triggers generateFakeID
    expect(generateFakeID).toHaveBeenCalled();
    expect(localStorage.getItem("prolificId")).toBe("AUTOID");

    // Check stored params
    const params = JSON.parse(localStorage.getItem("current_round_params"));
    expect(params).toEqual({
      is_target: false,
      llm_target: "T1",
      llm_persuader: "P1",
      proposition: "propX",
      persuader_supports_proposition: false,
    });

    // API called with the generated id
    expect(api.initializeParticipant).toHaveBeenCalledWith("AUTOID");
    expect(localStorage.getItem("participantId")).toBe("222");
    expect(push).toHaveBeenCalledWith("/lobby");
  });

  it("auto-initializes when URL has non-blank prolificId", async () => {
    const query = { prolificId: "FROMURL", isTarget: "true" };
    api.initializeParticipant.mockResolvedValue({ participant_id: 333 });

    mountWith(query);
    await flushPromises();

    // Should not generate a new ID
    expect(generateFakeID).not.toHaveBeenCalled();
    expect(localStorage.getItem("prolificId")).toBe("FROMURL");

    const params = JSON.parse(localStorage.getItem("current_round_params"));
    expect(params).toEqual({ is_target: true });

    expect(api.initializeParticipant).toHaveBeenCalledWith("FROMURL");
    expect(localStorage.getItem("participantId")).toBe("333");
    expect(push).toHaveBeenCalledWith("/lobby");
  });
});
