import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Mic, MicOff, Phone, X, Info, ChevronRight } from 'lucide-react';
import { useGeminiLive } from './hooks/useGeminiLive';
import { BookingDetails } from './types';
import { ConciergeOrb } from './components/ConciergeOrb';
import { DebugPanel } from './components/DebugPanel';
import { BookingPanel } from './components/BookingPanel';
import { TranscriptPanel } from './components/TranscriptPanel';

export default function App() {
  const [isConciergeOpen, setIsConciergeOpen] = useState(false);
  const [booking, setBooking] = useState<BookingDetails>({});
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleBookingUpdate = useCallback((updates: Partial<BookingDetails>) => {
    setBooking(prev => ({ ...prev, ...updates }));
  }, []);

  const handleBookingSubmit = useCallback(() => {
    setIsSubmitted(true);
    // In a real app, we would send this to a backend
    console.log('Reservation Submitted:', booking);
    setTimeout(() => {
      setIsSubmitted(false);
      setBooking({});
    }, 5000);
  }, [booking]);

  const { status, transcript, debug, connect, disconnect } = useGeminiLive(handleBookingUpdate, handleBookingSubmit);

  const toggleConcierge = () => {
    if (isConciergeOpen) {
      disconnect();
      setIsConciergeOpen(false);
    } else {
      setIsConciergeOpen(true);
      connect();
    }
  };

  return (
    <div className="min-h-screen bg-[#f5f2ed] text-[#1a1a1a] font-sans selection:bg-zinc-900 selection:text-white">
      {/* Navigation */}
      <nav className="border-b border-zinc-200/60 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <h1 className="text-2xl font-serif tracking-tighter font-bold">DAP <span className="font-light italic">Executives</span></h1>
            <div className="hidden md:flex items-center gap-6 text-[11px] uppercase tracking-[0.2em] font-medium text-zinc-500">
              <a href="#" className="hover:text-zinc-900 transition-colors">Services</a>
              <a href="#" className="hover:text-zinc-900 transition-colors">Fleet</a>
              <a href="#" className="hover:text-zinc-900 transition-colors">About</a>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <a href="tel:+18005550199" className="flex items-center gap-2 text-[11px] uppercase tracking-widest font-bold text-zinc-400 hover:text-zinc-900 transition-colors">
              <Phone size={14} />
              <span className="hidden sm:inline">1.800.555.0199</span>
            </a>
            <button 
              onClick={toggleConcierge}
              className={`px-6 py-2.5 rounded-full text-[11px] uppercase tracking-widest font-bold transition-all duration-500 flex items-center gap-2 ${
                isConciergeOpen 
                  ? 'bg-zinc-900 text-white' 
                  : 'bg-white border border-zinc-200 text-zinc-900 hover:border-zinc-900'
              }`}
            >
              {isConciergeOpen ? <MicOff size={14} /> : <Mic size={14} />}
              {isConciergeOpen ? 'End Session' : 'Talk to Cam'}
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
          
            <div className="lg:col-span-7 space-y-12">
              <section>
                <h2 className="text-5xl md:text-7xl font-serif font-light leading-tight mb-6">
                  Redefining the <br />
                  <span className="italic">Executive Journey.</span>
                </h2>
                <p className="text-lg text-zinc-500 max-w-xl leading-relaxed">
                  Experience unparalleled luxury transportation with DAP Executives. 
                  Our bespoke concierge service ensures every detail of your travel is handled with precision and discretion.
                </p>
              </section>

              <AnimatePresence>
                {isSubmitted ? (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="bg-emerald-50 border border-emerald-200 rounded-2xl p-8 text-center space-y-4"
                  >
                    <div className="w-16 h-16 bg-emerald-500 text-white rounded-full flex items-center justify-center mx-auto">
                      <Mic size={32} />
                    </div>
                    <h3 className="text-2xl font-serif italic text-emerald-900">Reservation Confirmed</h3>
                    <p className="text-emerald-700 text-sm">
                      Your executive transportation has been successfully arranged. 
                      A confirmation email and SMS have been sent to your details.
                    </p>
                  </motion.div>
                ) : (
                  <BookingPanel booking={booking} />
                )}
              </AnimatePresence>

              {/* Mobile Fallback */}
            <div className="md:hidden bg-white p-6 rounded-2xl border border-zinc-200 shadow-sm flex items-center justify-between">
              <div>
                <h4 className="font-bold text-sm mb-1">Need immediate help?</h4>
                <p className="text-xs text-zinc-500">Our team is available 24/7.</p>
              </div>
              <a href="tel:+18005550199" className="w-12 h-12 rounded-full bg-zinc-900 text-white flex items-center justify-center">
                <Phone size={20} />
              </a>
            </div>
          </div>

          {/* Right Column: Concierge Interface */}
          <div className="lg:col-span-5 space-y-6">
            <AnimatePresence mode="wait">
              {isConciergeOpen ? (
                <motion.div
                  key="concierge-active"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  className="space-y-6"
                >
                  <div className="bg-white rounded-3xl p-8 border border-zinc-200 shadow-xl relative overflow-hidden">
                    <div className="absolute top-4 right-4 text-zinc-300">
                      <Info size={20} />
                    </div>
                    
                    <div className="flex flex-col items-center text-center space-y-8 py-4">
                      <ConciergeOrb status={status} />
                      <div>
                        <h3 className="text-xl font-serif italic mb-2">
                          {status === 'connecting' ? 'Connecting to Cam...' : 
                           status === 'listening' ? 'Listening to you...' : 
                           status === 'speaking' ? 'Cam is speaking...' : 
                           status === 'error' ? 'Voice System Unavailable' : 'Cam Offline'}
                        </h3>
                        {status === 'error' ? (
                          <div className="space-y-4">
                            <p className="text-sm text-red-500/80 max-w-xs mx-auto">
                              {debug.lastError || "We're experiencing a technical issue with the voice system. Please contact our dispatch team directly."}
                            </p>
                            <div className="flex flex-col sm:flex-row gap-3 justify-center">
                              <button 
                                onClick={connect}
                                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-emerald-500 text-white rounded-full text-xs uppercase tracking-widest font-bold hover:bg-emerald-600 transition-colors"
                              >
                                Try Again
                              </button>
                              <a 
                                href="tel:+18005550199" 
                                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-zinc-900 text-white rounded-full text-xs uppercase tracking-widest font-bold hover:bg-zinc-800 transition-colors"
                              >
                                <Phone size={14} />
                                Call Dispatch
                              </a>
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm text-zinc-400 max-w-xs mx-auto">
                            Speak naturally to provide your reservation details. I'm Cam, and I'm here to assist with your travel logistics.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>

                  <TranscriptPanel messages={transcript} />
                  <DebugPanel debug={debug} status={status} />
                </motion.div>
              ) : (
                <motion.div
                  key="concierge-cta"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-zinc-900 rounded-3xl p-12 text-white flex flex-col items-center text-center justify-center min-h-[400px] space-y-8"
                >
                  <div className="w-20 h-20 rounded-full border border-white/20 flex items-center justify-center">
                    <Mic size={32} className="text-emerald-400" />
                  </div>
                  <div className="space-y-4">
                    <h3 className="text-3xl font-serif font-light">Cam: Voice Concierge</h3>
                    <p className="text-zinc-400 text-sm max-w-xs mx-auto">
                      Experience our AI-powered booking assistant. Simply speak to arrange your next luxury transfer.
                    </p>
                  </div>
                  <button 
                    onClick={toggleConcierge}
                    className="group flex items-center gap-3 px-8 py-4 bg-white text-zinc-900 rounded-full font-bold text-xs uppercase tracking-widest hover:bg-emerald-400 transition-all"
                  >
                    Begin Conversation
                    <ChevronRight size={16} className="group-hover:translate-x-1 transition-transform" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-200 mt-24 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="text-[10px] uppercase tracking-widest font-bold text-zinc-400">
            © 2026 DAP Executives Luxury Transportation
          </div>
          <div className="flex gap-8 text-[10px] uppercase tracking-widest font-bold text-zinc-400">
            <a href="#" className="hover:text-zinc-900">Privacy Policy</a>
            <a href="#" className="hover:text-zinc-900">Terms of Service</a>
            <a href="#" className="hover:text-zinc-900">Fleet Standards</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
