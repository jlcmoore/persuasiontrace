// Author: Jared Moore
// Date: July, 2025
// frontend/src/router.js

import { createRouter, createWebHistory } from "vue-router";
import WelcomePage from "./components/WelcomeVue.vue";
import ConsentForm from "./components/ConsentForm.vue";
import Round from "./components/round/RoundVue.vue";
import Lobby from "./components/LobbyVue.vue";
import AudioSetup from "./components/AudioSetup.vue";
import Feedback from "./components/FeedbackVue.vue";
import Debrief from "./components/DebriefVue.vue";
import RoundSetup from "./components/RoundSetup.vue";
import ParticipantPropositionSetup from "./components/ParticipantPropositionSetup.vue";
import NotFound from "./components/NotFound.vue";
import { api } from "@/api";

const routes = [
  { path: "/", component: WelcomePage },
  { path: "/consent", component: ConsentForm },
  { path: "/audio-setup", component: AudioSetup },
  { path: "/round", component: Round },
  {
    path: "/lobby",
    component: Lobby,
    beforeEnter: (to, from, next) => {
      if (api.participantPropositionsRequired()) {
        const setupComplete =
          localStorage.getItem("participantPropositionsComplete") === "true";
        if (!setupComplete) {
          return next("/pre-lobby");
        }
      }
      if (api.mayUseAudio()) {
        const audioOK = localStorage.getItem("audioOK");
        if (!audioOK && to.path !== "/audio-setup") {
          return next("/audio-setup");
        }
      }
      return next();
    },
  },
  { path: "/feedback", component: Feedback },
  { path: "/pre-lobby", component: ParticipantPropositionSetup },
  { path: "/debrief", component: Debrief },
  {
    path: "/round-setup",
    component: RoundSetup,
    beforeEnter: (to, from, next) => {
      if (api.isDevelopmentMode()) {
        next();
      } else {
        next("/404");
      }
      return;
    },
  },
  { path: "/404", component: NotFound }, // 404 route
  { path: "/:catchAll(.*)", redirect: "/404" }, // Redirect all undefined routes to 404
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
