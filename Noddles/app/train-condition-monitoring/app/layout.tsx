import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Noddles | Train condition monitoring",
  description: "Local predictive maintenance workspace for rail condition monitoring.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
