import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NexusAI - Free Open AI Platform | A Project by Osama",
  description: "A free, open AI platform that rivals industry leaders. Build AI agents, access powerful APIs, and create intelligent applications - all for free.",
  keywords: ["NexusAI", "AI Platform", "Free AI", "Open Source AI", "Agent Framework", "API", "Osama"],
  authors: [{ name: "Osama" }],
  openGraph: {
    title: "NexusAI - Free Open AI Platform",
    description: "A free, open AI platform with agent framework, OpenAI-compatible API, and multi-model routing",
    siteName: "NexusAI",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
      </body>
    </html>
  );
}
