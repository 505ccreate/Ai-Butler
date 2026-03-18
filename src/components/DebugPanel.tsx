import { DebugInfo, ConnectionStatus } from '../types';
import { Shield, Activity, Terminal, AlertCircle } from 'lucide-react';

interface DebugPanelProps {
  debug: DebugInfo;
  status: ConnectionStatus;
}

export function DebugPanel({ debug, status }: DebugPanelProps) {
  return (
    <div className="bg-zinc-950/50 border border-zinc-800 rounded-xl p-4 font-mono text-xs text-zinc-400 space-y-4">
      <div className="flex items-center justify-between border-b border-zinc-800 pb-2 mb-2">
        <span className="text-zinc-500 uppercase tracking-widest font-bold">System Diagnostics</span>
        <span className={`px-2 py-0.5 rounded-full text-[10px] uppercase ${
          status === 'error' ? 'bg-red-500/20 text-red-400' :
          status === 'offline' ? 'bg-zinc-800 text-zinc-500' :
          'bg-emerald-500/20 text-emerald-400'
        }`}>
          {status}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Shield size={14} className={debug.micPermission === 'granted' ? 'text-emerald-500' : 'text-zinc-600'} />
            <span>Mic: {debug.micPermission}</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity size={14} className={debug.wsStatus === 'open' ? 'text-emerald-500' : 'text-zinc-600'} />
            <span>WS: {debug.wsStatus}</span>
          </div>
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Terminal size={14} className="text-zinc-600" />
            <span>Mode: Live Audio</span>
          </div>
        </div>
      </div>

      {debug.lastToolCall && (
        <div className="pt-2 border-t border-zinc-800">
          <div className="text-zinc-500 mb-1 flex items-center gap-1">
            <Activity size={12} /> LAST TOOL CALL
          </div>
          <pre className="bg-black/40 p-2 rounded overflow-x-auto whitespace-pre-wrap">
            {debug.lastToolCall}
          </pre>
        </div>
      )}

      {debug.lastError && (
        <div className="pt-2 border-t border-zinc-800 text-red-400">
          <div className="flex items-center gap-1 mb-1">
            <AlertCircle size={12} /> ERROR LOG
          </div>
          <div className="bg-red-500/10 p-2 rounded border border-red-500/20">
            {debug.lastError}
          </div>
        </div>
      )}
    </div>
  );
}
