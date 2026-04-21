import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "LoRA Compare UI",
  description: "Base vs LoRA applied comparison"
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
