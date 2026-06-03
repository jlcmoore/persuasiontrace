// tests/unit/ChatVue.spec.js

import { mount } from "@vue/test-utils";
import Chat from "@/components/round/ChatVue.vue";

describe("ChatVue.vue", () => {
  const MAX_CHARS = 200;
  const factory = (props = {}) => {
    return mount(Chat, {
      props: {
        chatMessages: [],
        newMessage: "",
        typing: false,
        isTarget: false,
        allowInput: true,
        canSendMessageExterior: true,
        maxMessageChars: MAX_CHARS,
        ...props,
      },
    });
  };

  it("renders chat messages with correct classes", () => {
    const messages = [
      { text: "Hello from you", sender: "You" },
      { text: "Hello from agent", sender: "Agent" },
    ];
    const wrapper = factory({ chatMessages: messages });
    const messageEls = wrapper.findAll(".message");
    expect(messageEls).toHaveLength(2);

    const [youEl, agentEl] = messageEls;
    expect(youEl.classes()).toContain("message-right");
    expect(youEl.text()).toBe("Hello from you");
    expect(agentEl.classes()).toContain("message-left");
    expect(agentEl.text()).toBe("Hello from agent");
  });

  describe("canSendMessage computed", () => {
    it("disallows when newMessage is empty or whitespace", async () => {
      const wrapper = factory({ newMessage: "   " });
      expect(wrapper.vm.canSendMessage).toBe(false);
      const button = wrapper.find("#send-button");
      expect(button.attributes("disabled")).toBeDefined();
    });

    it("disallows when newMessage length >= maxMessageChars", () => {
      const long = "x".repeat(MAX_CHARS);
      const wrapper = factory({ newMessage: long });
      expect(wrapper.vm.canSendMessage).toBe(false);
    });

    it("allows first message if not target", () => {
      const wrapper = factory({
        newMessage: "hi",
        chatMessages: [],
        isTarget: false,
        allowInput: true,
        canSendMessageExterior: true,
      });
      expect(wrapper.vm.canSendMessage).toBe(true);
    });

    it("disallows first message if isTarget", () => {
      const wrapper = factory({
        newMessage: "hi",
        chatMessages: [],
        isTarget: true,
      });
      expect(wrapper.vm.canSendMessage).toBe(false);
    });

    it("allows when last message sender != You", () => {
      const msgs = [{ text: "A", sender: "Agent" }];
      const wrapper = factory({
        newMessage: "reply",
        chatMessages: msgs,
        isTarget: false,
        allowInput: true,
        canSendMessageExterior: true,
      });
      expect(wrapper.vm.canSendMessage).toBe(true);
    });

    it("disallows when last message sender = You", () => {
      const msgs = [{ text: "Me", sender: "You" }];
      const wrapper = factory({
        newMessage: "again",
        chatMessages: msgs,
        isTarget: false,
        allowInput: true,
        canSendMessageExterior: true,
      });
      expect(wrapper.vm.canSendMessage).toBe(false);
    });
  });

  it("emits update:newMessage when typing in textarea", async () => {
    const wrapper = factory();
    const textarea = wrapper.find("textarea");
    await textarea.setValue("test input");
    // because :value="newMessage" we manually emit
    wrapper.vm.updateMessage("test input");
    expect(wrapper.emitted("update:newMessage")).toBeTruthy();
    expect(wrapper.emitted("update:newMessage")[0]).toEqual(["test input"]);
  });

  it("emits send-message and clears input on send", async () => {
    const wrapper = factory({
      newMessage: "hello",
      chatMessages: [],
      isTarget: false,
        allowInput: true,
        canSendMessageExterior: true,
    });
    // spy on updateMessage
    const sendButton = wrapper.find("#send-button");
    await sendButton.trigger("click");

    // send-message event
    expect(wrapper.emitted("send-message")).toBeTruthy();
    expect(wrapper.emitted("send-message")[0]).toEqual([{ message: "hello" }]);
    // input cleared
    const updates = wrapper.emitted("update:newMessage") || [];
    expect(updates).toHaveLength(1);
    expect(updates[0]).toEqual([""]);
  });
});
