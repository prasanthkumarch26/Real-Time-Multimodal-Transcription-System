import TranscriptionApp from "@/components/TranscriptionApp";

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-8 flex flex-col items-center justify-center">
      <div className="w-full max-w-4xl mx-auto space-y-8">
        <header className="text-center space-y-4">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight">
            Real-Time <span className="text-indigo-600">Multimodal</span> Transcription
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Experience real-time AI transcription acts like AR glasses subtitles. 
            Switch seamlessly between Audio Translation and Indian Sign Language.
          </p>
        </header>

        <TranscriptionApp />
      </div>
    </main>
  );
}
