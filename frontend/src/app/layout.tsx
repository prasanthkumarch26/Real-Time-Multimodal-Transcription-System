import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Real-Time Transcription',
  description: 'Multimodal translation system acting as AR glasses subtitles',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="bg-ambient text-foreground antialiased min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
