"use client";

import { Mic, Video } from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

type Mode = "AT" | "ISL";

interface ModeSelectorProps {
  mode: Mode;
  setMode: (mode: Mode) => void;
  isRecording: boolean;
}

export default function ModeSelector({ mode, setMode, isRecording }: ModeSelectorProps) {
  return (
    <div className="flex bg-black/20 p-1 rounded-2xl border border-white/5 mx-auto w-fit">
      <button
        type="button"
        disabled={isRecording}
        onClick={() => setMode("AT")}
        className={twMerge(
          clsx(
            "flex items-center gap-2 px-6 py-3 rounded-xl transition-all duration-300 font-medium",
            mode === "AT" ? "bg-white/10 text-white shadow-lg" : "text-white/50 hover:text-white/80 hover:bg-white/5",
            isRecording && "opacity-50 cursor-not-allowed"
          )
        )}
      >
        <Mic className="w-5 h-5" />
        <span>Audio Transcription</span>
      </button>

      <button
        type="button"
        disabled={isRecording}
        onClick={() => setMode("ISL")}
        className={twMerge(
          clsx(
            "flex items-center gap-2 px-6 py-3 rounded-xl transition-all duration-300 font-medium",
            mode === "ISL" ? "bg-white/10 text-white shadow-lg" : "text-white/50 hover:text-white/80 hover:bg-white/5",
            isRecording && "opacity-50 cursor-not-allowed"
          )
        )}
      >
        <Video className="w-5 h-5" />
        <span>Sign Language (ISL)</span>
      </button>
    </div>
  );
}
