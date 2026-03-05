import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DSPA Bot — UC Berkeley Data Science Peer Advisor",
  description:
    "AI-powered peer advising for the UC Berkeley Data Science major",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
