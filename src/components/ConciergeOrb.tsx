import { motion } from 'motion/react';
import { ConnectionStatus } from '../types';

interface ConciergeOrbProps {
  status: ConnectionStatus;
}

export function ConciergeOrb({ status }: ConciergeOrbProps) {
  const isActive = status === 'listening' || status === 'speaking';
  const isSpeaking = status === 'speaking';

  return (
    <div className="relative flex items-center justify-center w-48 h-48">
      {/* Outer Glow */}
      <motion.div
        animate={{
          scale: isActive ? [1, 1.1, 1] : 1,
          opacity: isActive ? [0.3, 0.6, 0.3] : 0.2,
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        className="absolute inset-0 rounded-full bg-emerald-500/20 blur-3xl"
      />

      {/* Main Orb */}
      <motion.div
        animate={{
          scale: isSpeaking ? [1, 1.05, 1] : 1,
          boxShadow: isSpeaking 
            ? "0 0 40px rgba(16, 185, 129, 0.4)" 
            : "0 0 20px rgba(16, 185, 129, 0.2)",
        }}
        transition={{
          duration: 0.5,
          repeat: isSpeaking ? Infinity : 0,
        }}
        className={`w-32 h-32 rounded-full border border-emerald-500/30 bg-gradient-to-br from-zinc-900 to-zinc-950 flex items-center justify-center overflow-hidden`}
      >
        {/* Inner Wave/Pulse */}
        {isActive && (
          <div className="flex items-center gap-1">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                animate={{
                  height: isSpeaking ? [8, 32, 8] : [4, 12, 4],
                }}
                transition={{
                  duration: 0.6,
                  repeat: Infinity,
                  delay: i * 0.1,
                }}
                className="w-1 bg-emerald-500 rounded-full"
              />
            ))}
          </div>
        )}
      </motion.div>

      {/* Status Ring */}
      <svg className="absolute inset-0 w-full h-full -rotate-90">
        <circle
          cx="50%"
          cy="50%"
          r="48%"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          className={`transition-colors duration-500 ${
            status === 'error' ? 'text-red-500/50' : 
            status === 'connecting' ? 'text-amber-500/50' : 
            isActive ? 'text-emerald-500/50' : 'text-zinc-800'
          }`}
          strokeDasharray="4 4"
        />
      </svg>
    </div>
  );
}
