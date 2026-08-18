import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { ThemeProvider } from "@/lib/theme";
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
  title: "NexusAI - Free AI Platform",
  description: "A free AI platform with multi-model chat, code execution, web search, streaming responses, and OpenAI-compatible API. Powered by Qwen 2.5.",
  keywords: ["NexusAI", "AI Platform", "Free AI", "Open Source AI", "Chat", "Code Execution", "Web Search", "API"],
  authors: [{ name: "Osama" }],
  openGraph: {
    title: "NexusAI - Free AI Platform",
    description: "Multi-model AI chat, code execution, web search, and streaming - all free.",
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
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
