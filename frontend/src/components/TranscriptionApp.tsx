"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ModeSelector from "./ModeSelector";
import TranscriptionDisplay from "./TranscriptionDisplay";
import { Mic, Video, Square } from "lucide-react";

type Mode = "AT" | "ISL";

const AT_WS_URL = "ws://localhost:8000/attranscribe";
const ISL_WS_URL = "ws://localhost:8001/isltranscribe";

// ISL: extract video frame at this rate (fps)
const ISL_FRAME_INTERVAL_MS = 100; // 10 fps

export default function TranscriptionApp() {
  const [mode, setMode] = useState<Mode>("AT");
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [status, setStatus] = useState<"idle" | "connecting" | "streaming" | "error">("idle");

  // Refs for WebSocket
  const wsRef = useRef<WebSocket | null>(null);
  // Refs for AT (AudioWorklet)
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Refs for ISL (Video)
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const islIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ─── WebSocket setup ────────────────────────────────────────────────────────
  const connectWebSocket = useCallback((url: string) => {
    const ws = new WebSocket(url);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log(`[WS] Connected to ${url}`);
      setStatus("streaming");
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        if (data.Transcription) {
          setTranscription((prev) => {
            const sep = prev ? " " : "";
            return prev + sep + data.Transcription;
          });
        }
      } catch {
        console.error("[WS] Failed to parse message", event.data);
      }
    };

    ws.onclose = () => {
      console.log("[WS] Connection closed");
      setStatus("idle");
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
      setStatus("error");
      setErrorMsg("WebSocket connection failed. Is the backend running?");
    };

    wsRef.current = ws;
    return ws;
  }, []);

  // ─── AT Mode: AudioWorklet ───────────────────────────────────────────────
  const startATStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: false },
        video: false,
      });
      streamRef.current = stream;

      // AudioContext at 16kHz to avoid browser resampling issues
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      // Load the AudioWorklet processor script from /public
      await audioContext.audioWorklet.addModule("/audio-processor.js");

      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, "audio-processor", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        processorOptions: {},
      });
      workletNodeRef.current = workletNode;

      // Receive Float32 PCM chunks from worklet and send over WebSocket
      workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
        const chunk: Float32Array = event.data;
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(chunk.buffer);
        }
      };

      source.connect(workletNode);
      workletNode.connect(audioContext.destination);
      // Mute output to avoid feedback
      audioContext.destination.channelInterpretation = "discrete";

    } catch (err: unknown) {
      const error = err as Error;
      console.error("[AT] Error starting audio stream:", error);
      setIsRecording(false);
      if (error.name === "NotAllowedError") {
        setErrorMsg("Microphone permission was denied. Please allow access in your browser settings.");
      } else {
        setErrorMsg(`Failed to start audio: ${error.message}`);
      }
    }
  }, []);

  // ─── ISL Mode: Video + Canvas Frame Extraction ──────────────────────────
  const startISLStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;

      // Attach stream to hidden video element
      if (!videoRef.current) {
        const video = document.createElement("video");
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;
        video.style.display = "none";
        document.body.appendChild(video);
        videoRef.current = video;
      }
      videoRef.current.srcObject = stream;

      if (!canvasRef.current) {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 480;
        canvasRef.current = canvas;
      }

      // Send frames over WebSocket as JPEG data URLs
      islIntervalRef.current = setInterval(() => {
        if (!videoRef.current || !canvasRef.current) return;
        if (wsRef.current?.readyState !== WebSocket.OPEN) return;

        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        const dataUrl = canvasRef.current.toDataURL("image/jpeg", 0.6);
        wsRef.current.send(JSON.stringify({ frame: dataUrl }));
      }, ISL_FRAME_INTERVAL_MS);

    } catch (err: unknown) {
      const error = err as Error;
      console.error("[ISL] Error starting video stream:", error);
      setIsRecording(false);
      if (error.name === "NotAllowedError") {
        setErrorMsg("Camera permission was denied. Please allow access in your browser settings.");
      } else {
        setErrorMsg(`Failed to start video: ${error.message}`);
      }
    }
  }, []);

  // ─── Stop everything ─────────────────────────────────────────────────────
  const stopAllStreams = useCallback(() => {
    // Stop AT
    if (workletNodeRef.current) {
      workletNodeRef.current.port.close();
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close().catch(console.error);
      audioContextRef.current = null;
    }
    // Stop ISL
    if (islIntervalRef.current) {
      clearInterval(islIntervalRef.current);
      islIntervalRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.remove();
      videoRef.current = null;
    }
    canvasRef.current = null;
    // Stop media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    // Close WebSocket
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
      wsRef.current = null;
    }
    setStatus("idle");
  }, []);

  // ─── Recording Lifecycle ──────────────────────────────────────────────────
  useEffect(() => {
    if (!isRecording) {
      stopAllStreams();
      return;
    }

    setStatus("connecting");
    setErrorMsg("");

    const wsUrl = mode === "AT" ? AT_WS_URL : ISL_WS_URL;
    connectWebSocket(wsUrl);

    if (mode === "AT") {
      startATStream();
    } else {
      startISLStream();
    }

    return () => {
      stopAllStreams();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording, mode]);

  const handleToggleRecord = () => {
    if (!isRecording) {
      setTranscription("");
      setErrorMsg("");
      setIsRecording(true);
    } else {
      setIsRecording(false);
    }
  };

  const statusColor = {
    idle: "bg-gray-500",
    connecting: "bg-yellow-400 animate-pulse",
    streaming: "bg-green-400 animate-pulse",
    error: "bg-red-500",
  }[status];

  const statusLabel = {
    idle: "Idle",
    connecting: "Connecting...",
    streaming: "Streaming",
    error: "Error",
  }[status];

  return (
    <div className="space-y-8 flex flex-col items-center w-full">
      <ModeSelector mode={mode} setMode={setMode} isRecording={isRecording} />

      {/* Status indicator */}
      <div className="flex items-center gap-2 text-sm text-white/50">
        <span className={`inline-block w-2.5 h-2.5 rounded-full ${statusColor}`} />
        <span>{statusLabel}</span>
      </div>

      <div className="flex flex-col items-center gap-4">
        <button
          id="record-toggle-btn"
          onClick={handleToggleRecord}
          className={`
            relative group flex items-center justify-center gap-3 px-8 py-4 rounded-full font-bold text-lg text-white transition-all duration-300 select-none
            ${isRecording
              ? "bg-red-500 hover:bg-red-600 shadow-[0_0_40px_-5px_rgba(239,68,68,0.5)]"
              : "bg-indigo-600 hover:bg-indigo-500 shadow-[0_0_40px_-5px_rgba(99,102,241,0.5)]"}
          `}
        >
          {isRecording ? (
            <>
              <Square className="w-5 h-5 fill-current" />
              Stop Streaming
            </>
          ) : (
            <>
              {mode === "AT" ? <Mic className="w-5 h-5" /> : <Video className="w-5 h-5" />}
              Start Streaming
            </>
          )}
        </button>

        {errorMsg && (
          <div className="text-red-400 bg-red-500/10 border border-red-500/20 px-4 py-3 rounded-lg text-sm max-w-md text-center">
            {errorMsg}
          </div>
        )}
      </div>

      <TranscriptionDisplay transcription={transcription} />
    </div>
  );
}
