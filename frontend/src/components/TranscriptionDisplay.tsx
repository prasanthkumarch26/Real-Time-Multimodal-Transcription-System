"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface TranscriptionDisplayProps {
  transcription: string;
}

export default function TranscriptionDisplay({ transcription }: TranscriptionDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as text updates
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [transcription]);

  return (
    <div className="relative w-full h-[300px] md:h-[400px] glass rounded-3xl overflow-hidden flex flex-col">
      <div className="p-4 border-b border-white/10 bg-white/5 flex items-center justify-between z-10">
        <h3 className="text-sm font-semibold tracking-wider text-white/70 uppercase">Live Subtitles</h3>
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <div className="w-3 h-3 rounded-full bg-green-500" />
        </div>
      </div>
      
      <div 
        ref={containerRef}
        className="flex-1 p-6 md:p-8 overflow-y-auto scroll-smooth flex flex-col justify-end"
      >
        <AnimatePresence mode="wait">
          {!transcription ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center text-white/30 italic my-auto"
            >
              Waiting for speech or signs...
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-col gap-2"
            >
              <p className="text-xl md:text-2xl font-semibold leading-relaxed text-white drop-shadow-md">
                {transcription}
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      {/* Soft gradient overlay for aesthetics */}
      <div className="absolute top-14 left-0 w-full h-12 bg-gradient-to-b from-black/20 to-transparent pointer-events-none" />
    </div>
  );
}
