"use client";

import { useState, useEffect, useRef } from "react";
import ModeSelector from "./ModeSelector";
import TranscriptionDisplay from "./TranscriptionDisplay";
import { Mic, Video, Square, Play } from "lucide-react";

type Mode = "AT" | "ISL";

export default function TranscriptionApp() {
  const [mode, setMode] = useState<Mode>("AT");
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Connect to WebSocket based on mode
  useEffect(() => {
    if (!isRecording) return;

    // We assume backend for AT is at ws://localhost:8000/attranscribe
    // For ISL it will be something similar (placeholder here)
    const wsUrl = mode === "AT" ? "ws://localhost:8000/attranscribe" : "ws://localhost:8000/isltranscribe";
    
    console.log(`Connecting to ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.Transcription) {
          // Append the text block as a continuous subtitle stream
          setTranscription((prev) => {
            const separator = prev ? " " : "";
            return prev + separator + data.Transcription;
          });
        }
      } catch (err) {
        console.error("Failed to parse websocket message", err);
      }
    };

    ws.onclose = () => console.log("WebSocket Disconnected");
    wsRef.current = ws;

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [isRecording, mode]);

  // Handle Recording lifecycle
  useEffect(() => {
    if (!isRecording) {
      // Cleanup
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(console.error);
        audioContextRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      return;
    }

    const startRecording = async () => {
      try {
        setErrorMsg("");
        const constraints = mode === "AT" 
            ? { audio: true, video: false } 
            : { audio: false, video: true };
            
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream;

        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContextClass({ sampleRate: 16000 });
        audioContextRef.current = audioContext;

        const source = audioContext.createMediaStreamSource(stream);
        // Using ScriptProcessorNode for simple PCM extraction in React
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        processorRef.current = processor;

        source.connect(processor);
        
        // Connect processor to destination so it runs, but mute it to prevent feedback
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 0;
        processor.connect(gainNode);
        gainNode.connect(audioContext.destination);

        let audioBuffer: number[] = [];
        
        processor.onaudioprocess = (e) => {
            if (!isRecording || wsRef.current?.readyState !== WebSocket.OPEN) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            for (let i = 0; i < inputData.length; i++) {
                audioBuffer.push(inputData[i]);
            }
            
            // Send exactly 1.0 second of audio at a time (16000 samples)
            if (audioBuffer.length >= 16000) {
                const chunk = audioBuffer.slice(0, 16000);
                audioBuffer = audioBuffer.slice(16000); // keep remainder
                
                const pcmData = new Int16Array(16000);
                for (let i = 0; i < chunk.length; i++) {
                    // Convert float [-1.0, 1.0] to Int16
                    pcmData[i] = Math.max(-1, Math.min(1, chunk[i])) * 0x7FFF;
                }
                
                wsRef.current.send(pcmData.buffer); // Send Int16 array buffer
            }
        };

      } catch (err: any) {
        console.error("Error accessing media devices", err);
        setIsRecording(false);
        if (err.name === "NotAllowedError" || err.message.includes("Permission denied")) {
          setErrorMsg(`Microphone/Camera permission denied. Please allow access in your browser and Windows Privacy settings.`);
        } else {
          setErrorMsg(`Failed to start streaming: ${err.message || 'Unknown error'}`);
        }
      }
    };

    startRecording();
    
    return () => {
      if (processorRef.current) processorRef.current.disconnect();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(console.error);
      }
    }
  }, [isRecording, mode]);

  const handleToggleRecord = () => {
    if (!isRecording) {
      setTranscription(""); // Clear old transcriptions
      setErrorMsg("");
      setIsRecording(true);
    } else {
      setIsRecording(false);
    }
  };

  return (
    <div className="space-y-8 flex flex-col items-center">
      <ModeSelector 
        mode={mode} 
        setMode={setMode} 
        isRecording={isRecording} 
      />

      <div className="flex flex-col items-center gap-4 my-2">
        <button
          onClick={handleToggleRecord}
          className={`
            relative group flex items-center justify-center gap-3 px-8 py-4 rounded-full font-bold text-lg text-white transition-all duration-300
            ${isRecording 
              ? "bg-red-500 hover:bg-red-600 shadow-[0_0_40px_-5px_rgba(239,68,68,0.5)]" 
              : "bg-indigo-600 hover:bg-indigo-500 shadow-[0_0_40px_-5px_rgba(99,102,241,0.5)]"}
          `}
        >
          {isRecording ? (
            <>
              <Square className="w-5 h-5 fill-current" />
              Stop Recording
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
