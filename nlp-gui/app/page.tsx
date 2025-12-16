"use client";

import React, { useState } from "react";
import LogoImage from "./components/Logo";
import { Button } from "../components/ui/button";

export default function Home() {
  const [isStarted, setIsStarted] = useState(false);

  const tashkil_src = "/assets/Shakely_Shokran.png";
  const no_tashkil_src = "/assets/Shakely_Shokran_No_Tashkil.png";

  return (
    // Outer container: Must handle overflow to prevent scrollbars during animation
    <main className="relative h-screen w-screen overflow-hidden bg-zinc-50 dark:bg-black font-sans">
      {/* -------------------------------------------------------
          SCENE 1: THE MAIN APP / DASHBOARD (Background Layer)
          This sits behind the landing page and is revealed.
         ------------------------------------------------------- */}
      <div className="absolute inset-0 flex items-center justify-center bg-white dark:bg-zinc-900">
        <div className="text-center animate-in fade-in duration-1000 delay-500">
          <h1 className="text-4xl font-bold text-zinc-800 dark:text-zinc-100">
            Main Application Scene
          </h1>
          <p className="mt-4 text-zinc-500">
            The landing page has swiped down.
          </p>
        </div>
      </div>

      {/* -------------------------------------------------------
          SCENE 2: LANDING PAGE (Foreground Layer)
          This covers the screen initially. 
          When isStarted is true, it translates Y by 100% (Downwards).
         ------------------------------------------------------- */}
      <div
        className={`absolute inset-0 z-50 flex items-center justify-center bg-zinc-50 dark:bg-black transition-transform duration-1000 ease-in-out ${
          isStarted ? "-translate-y-full" : "translate-y-0"
        }`}
      >
        <div className="flex flex-col items-center gap-8">
          <LogoImage noTashkilSrc={no_tashkil_src} tashkilSrc={tashkil_src} />

          <Button
            size="lg"
            onClick={() => setIsStarted(true)}
            className="transition-all hover:scale-105 active:scale-95"
          >
            <p className="text-lg">Shakel</p>
          </Button>
        </div>
      </div>
    </main>
  );
}
