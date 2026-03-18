import { TranscriptMessage } from '../types';
import { motion, AnimatePresence } from 'motion/react';

interface TranscriptPanelProps {
  messages: TranscriptMessage[];
}

export function TranscriptPanel({ messages }: TranscriptPanelProps) {
  return (
    <div className="flex flex-col gap-4 h-[400px] overflow-y-auto p-4 bg-zinc-50/50 rounded-xl border border-zinc-100">
      <AnimatePresence initial={false}>
        {messages.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-zinc-400 italic text-sm">
            Conversation transcript will appear here...
          </div>
        ) : (
          messages.map((msg, i) => (
            <motion.div
              key={msg.timestamp + i}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <span className="text-[10px] uppercase tracking-widest text-zinc-400 mb-1">
                {msg.role === 'user' ? 'Client' : 'Cam'}
              </span>
              <div className={`max-w-[80%] p-3 rounded-2xl text-sm ${
                msg.role === 'user' 
                  ? 'bg-zinc-900 text-white rounded-tr-none' 
                  : 'bg-white border border-zinc-200 text-zinc-800 rounded-tl-none shadow-sm'
              }`}>
                {msg.text}
              </div>
            </motion.div>
          ))
        )}
      </AnimatePresence>
    </div>
  );
}
