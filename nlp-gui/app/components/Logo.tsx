"use client";
import React, { useEffect, useState } from "react";
import Image from "next/image";

type Props = {
  tashkilSrc: string;
  noTashkilSrc: string;
  alt?: string;
  width?: number;
  height?: number;
  intervalMs?: number;
  transitionMs?: number;
  className?: string;
};

export default function Logo({
  noTashkilSrc,
  tashkilSrc,
  alt = "NLP-GUI Logo",
  width = 500,
  height = 500,
  intervalMs = 3000,
  transitionMs = 1200,
  className,
}: Props) {
  const [showTashkil, setShowTashkil] = useState(true);

  useEffect(() => {
    const id = setInterval(() => setShowTashkil((s) => !s), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  // 1. BOTTOM LAYER (Tashkil) - Retracts to Bottom-Right when hidden
  const tashkilClip = showTashkil
    ? "polygon(-50% -50%, 300% -50%, 300% 300%, -50% 300%)" // Visible
    : "polygon(300% -50%, 300% -50%, 300% 300%, -50% 300%)"; // Hidden (retracted)

  // 2. TOP LAYER (No Tashkil) - Expands from Top-Left when visible
  const noTashkilClip = showTashkil
    ? "polygon(-50% -50%, -50% -50%, -50% 300%, -50% -50%)" // Hidden (retracted)
    : "polygon(-50% -50%, 300% -50%, -50% 300%, -50% -50%)"; // Visible

  const transition = `clip-path ${transitionMs}ms cubic-bezier(0.65, 0, 0.35, 1), -webkit-clip-path ${transitionMs}ms cubic-bezier(0.65, 0, 0.35, 1)`;

  // A subtle paper/metallic sheen gradient for the wipe effect
  const wipeGradient = "linear-gradient(135deg, #f5f7fa 0%, #e4e4e4ff 100%)";

  return (
    <div
      className={`relative inline-block ${className ?? ""}`}
      style={{ width, height, overflow: "hidden", borderRadius: 7 }}
      aria-hidden={false}
    >
      {/* IMAGE 1: NO TASHKIL (Top Layer) */}
      <div
        className="absolute inset-0 flex items-center justify-center"
        style={{
          transition,
          willChange: "clip-path",
          clipPath: tashkilClip,
          WebkitClipPath: tashkilClip,
          zIndex: 1,
          //backgroundColor: "#ffffff",
        }}
      >
        <Image
          src={noTashkilSrc}
          alt={alt + " (tashkil)"}
          width={width}
          height={height}
          priority
        />
      </div>

      {/* IMAGE 2: TASHKIL (Bottom Layer) */}
      <div
        className="absolute inset-0 flex items-center justify-center"
        style={{
          transition,
          willChange: "clip-path",
          clipPath: noTashkilClip,
          WebkitClipPath: noTashkilClip,
          zIndex: 2,
          background: wipeGradient,
        }}
      >
        <Image
          src={tashkilSrc}
          alt={alt + " (no tashkil)"}
          width={width}
          height={height}
          priority
        />
      </div>
    </div>
  );
}
