import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from "@google/genai";
import { BookingDetails, ConnectionStatus, DebugInfo, TranscriptMessage } from '../types';

const MODEL = "gemini-2.5-flash-native-audio-preview-09-2025";

export function useGeminiLive(
  onBookingUpdate: (details: Partial<BookingDetails>) => void,
  onBookingSubmit: () => void
) {
  const [status, setStatus] = useState<ConnectionStatus>('offline');
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const [debug, setDebug] = useState<DebugInfo>({
    micPermission: 'prompt',
    wsStatus: 'closed',
  });

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);

  // Initialize Audio Context
  const initAudio = async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000 // Match model output if possible, or just use default
      });
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  };

  // Play audio chunks
  const playNextChunk = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || isPlayingRef.current || !audioContextRef.current) {
      if (audioQueueRef.current.length === 0 && isPlayingRef.current === false) {
        setStatus(prev => prev === 'speaking' ? 'listening' : prev);
      }
      return;
    }

    isPlayingRef.current = true;
    const chunk = audioQueueRef.current.shift()!;
    
    const float32Data = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) {
      float32Data[i] = chunk[i] / 32768.0;
    }

    const buffer = audioContextRef.current.createBuffer(1, float32Data.length, 24000);
    buffer.getChannelData(0).set(float32Data);

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    currentSourceRef.current = source;
    
    source.onended = () => {
      if (currentSourceRef.current === source) {
        currentSourceRef.current = null;
      }
      isPlayingRef.current = false;
      playNextChunk();
    };

    source.start();
    setStatus('speaking');
  }, []);

  // Highly optimized base64 encoding for audio chunks
  const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const chunk = 8192; // Process in larger chunks for speed
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk) as any);
    }
    return window.btoa(binary);
  };

  const connect = async () => {
    console.log("Connecting to Gemini Live...");
    try {
      // 1. Check for Secure Context (Required for Mic)
      if (!window.isSecureContext) {
        throw new Error("The Voice Concierge requires a secure connection (HTTPS or localhost). Please ensure you are accessing the site securely.");
      }

      setStatus('connecting');
      setDebug(prev => ({ ...prev, wsStatus: 'connecting', lastError: undefined }));

      // 2. Start audio and mic requests in parallel for speed
      const audioInitPromise = initAudio();
      
      // Environment-agnostic API Key check
      const apiKey = 
        (import.meta as any).env?.VITE_GEMINI_API_KEY || 
        (import.meta as any).env?.GEMINI_API_KEY ||
        process.env.GEMINI_API_KEY;
      
      if (!apiKey || apiKey === '""' || apiKey === "''") {
        throw new Error("Gemini API Key is missing. Please ensure GEMINI_API_KEY is set in your environment variables.");
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Your browser has blocked microphone access because this site is not secure.");
      }

      const micPromise = navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });

      // Wait for both
      const [_, stream] = await Promise.all([audioInitPromise, micPromise]);
      
      streamRef.current = stream;
      setDebug(prev => ({ ...prev, micPermission: 'granted' }));

      const ai = new GoogleGenAI({ apiKey });

      const session = await ai.live.connect({
        model: MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Charon" } },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: `You are Cam, the lead executive transportation concierge for DAP Executives. 
          Persona: "Smooth Professionalism" (Sophisticated Black American tone, late-night R&B swagger, deep, resonant, calm).
          
          Identity:
          - Name: Cam.
          - Role: AI Executive for DAP Executive Cars.
          - STRICT RULE: NEVER mention "Gemini", "Google", or being a large language model. 
          - Maintain the "magic": you are a dedicated part of the DAP Executives team.
          
          Communication Style:
          - OPENING: When the session starts, greet the user professionally and focus immediately on the reservation. 
            Examples: "Welcome to DAP Executives, I'm Cam. Can I help you book a car today?", "Hello, my name is Cam. Are we making a reservation today?", "Good to have you with us. I'm Cam. Ready to arrange your executive transport?"
            Vary these openings so they don't sound redundant.
          - FOCUS: Do not use open-ended questions like "How can I help you tonight?" that invite off-topic chat. Keep it focused on booking and logistics.
          - PERSONALIZATION: Once you learn the user's name, ALWAYS address them by their name. It makes the service feel more exclusive and personable.
          - Be concise and smooth. Don't over-explain.
          
          Operational Guidelines:
          - Extract all details from user speech at once.
          - Ask only the next logical question.
          - Update the draft live using 'update_reservation'.
          - Summarize and get explicit confirmation before using 'submit_reservation'.
          
          Focus ONLY on transportation logistics.`,
          tools: [{
            functionDeclarations: [
              {
                name: "update_reservation",
                description: "Update the current reservation draft with captured details in real-time.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    service_type: { type: Type.STRING, description: "Type of service (Airport Transfer, Hourly, Point-to-Point)" },
                    vehicle_type: { type: Type.STRING, description: "Preferred vehicle (Sedan, SUV, Sprinter)" },
                    pickup_date: { type: Type.STRING, description: "Date of pickup" },
                    pickup_time: { type: Type.STRING, description: "Time of pickup" },
                    pickup_address: { type: Type.STRING, description: "Full pickup address" },
                    dropoff_address: { type: Type.STRING, description: "Full dropoff address" },
                    passenger_count: { type: Type.INTEGER, description: "Number of passengers" },
                    luggage_count: { type: Type.INTEGER, description: "Number of luggage pieces" },
                    customer_name: { type: Type.STRING, description: "Full name of the customer" },
                    customer_phone: { type: Type.STRING, description: "Phone number" },
                    customer_email: { type: Type.STRING, description: "Email address" },
                    flight_number: { type: Type.STRING, description: "Flight number if applicable" },
                    special_requests: { type: Type.STRING, description: "Any special requests or notes" },
                  }
                }
              },
              {
                name: "submit_reservation",
                description: "Finalize and submit the reservation after user confirmation.",
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    confirmed: { type: Type.BOOLEAN, description: "Whether the user has confirmed the details." }
                  },
                  required: ["confirmed"]
                }
              }
            ]
          }]
        },
        callbacks: {
          onopen: () => {
            setDebug(prev => ({ ...prev, wsStatus: 'open' }));
            setStatus('listening');

            // Trigger initial greeting correctly
            sessionRef.current?.send({
              clientContent: {
                turns: [{
                  role: 'user',
                  parts: [{ text: "Hello Cam. Please introduce yourself and ask if I'm ready to book a reservation." }]
                }]
              }
            });

            // Setup audio processing
            const source = audioContextRef.current!.createMediaStreamSource(stream);
            // Use larger buffer to reduce message frequency and overhead
            const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              if (status === 'offline' || status === 'error') return;
              
              const inputData = e.inputBuffer.getChannelData(0);
              const contextSampleRate = audioContextRef.current!.sampleRate;
              
              // Optimized downsampling
              const ratio = contextSampleRate / 16000;
              const newLength = Math.floor(inputData.length / ratio);
              const pcmData = new Int16Array(newLength);
              
              for (let i = 0; i < newLength; i++) {
                const sample = inputData[Math.floor(i * ratio)] || 0;
                pcmData[i] = sample < 0 ? sample * 32768 : sample * 32767;
              }
              
              const base64Data = arrayBufferToBase64(pcmData.buffer);
              sessionRef.current?.sendRealtimeInput({
                media: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
              });
            };

            source.connect(processor);
            processor.connect(audioContextRef.current!.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            try {
              // Handle Audio Output
              const audioParts = message.serverContent?.modelTurn?.parts?.filter(p => p.inlineData);
              if (audioParts && audioParts.length > 0) {
                audioParts.forEach(part => {
                  if (part.inlineData) {
                    const binaryString = atob(part.inlineData.data);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                      bytes[i] = binaryString.charCodeAt(i);
                    }
                    const pcmData = new Int16Array(bytes.buffer);
                    audioQueueRef.current.push(pcmData);
                  }
                });
                playNextChunk();
              }

              // Handle Interruption
              if (message.serverContent?.interrupted) {
                currentSourceRef.current?.stop();
                currentSourceRef.current = null;
                audioQueueRef.current = [];
                isPlayingRef.current = false;
                setStatus('listening');
              }

              // Handle Tool Calls
              const toolCall = message.toolCall;
              if (toolCall) {
                const call = toolCall.functionCalls[0];
                if (call.name === 'update_reservation') {
                  onBookingUpdate(call.args as Partial<BookingDetails>);
                  setDebug(prev => ({ ...prev, lastToolCall: `update: ${JSON.stringify(call.args)}` }));
                  
                  // Send response back
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      id: call.id,
                      response: { output: { success: true } }
                    }]
                  });
                } else if (call.name === 'submit_reservation') {
                  onBookingSubmit();
                  setDebug(prev => ({ ...prev, lastToolCall: 'submit' }));
                  
                  // Send response back
                  sessionRef.current?.sendToolResponse({
                    functionResponses: [{
                      id: call.id,
                      response: { output: { success: true } }
                    }]
                  });
                }
              }

              // Handle Transcripts
              // User Transcript (from inputAudioTranscription)
              const userTranscript = (message.serverContent as any)?.userTurn?.parts?.find((p: any) => p.text)?.text;
              if (userTranscript) {
                setTranscript(prev => [...prev, { role: 'user', text: userTranscript, timestamp: Date.now() }]);
              }

              // Model Transcript (from outputAudioTranscription)
              const modelTranscript = message.serverContent?.modelTurn?.parts?.find(p => p.text)?.text;
              if (modelTranscript) {
                setTranscript(prev => [...prev, { role: 'model', text: modelTranscript, timestamp: Date.now() }]);
              }
            } catch (msgErr) {
              console.error("Error processing message:", msgErr);
            }
          },
          onclose: () => {
            setDebug(prev => ({ ...prev, wsStatus: 'closed' }));
            setStatus('offline');
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setDebug(prev => ({ ...prev, wsStatus: 'error', lastError: String(err) }));
            setStatus('error');
          }
        }
      });

      sessionRef.current = session;

    } catch (err: any) {
      console.error("Connection Error:", err);
      setStatus('error');
      
      let errorMessage = String(err);
      if (err.name === 'NotAllowedError' || err.message?.includes('Permission denied')) {
        errorMessage = "Microphone access was denied. Please enable it in your browser settings to use the voice concierge.";
        setDebug(prev => ({ ...prev, micPermission: 'denied' }));
      }
      
      setDebug(prev => ({ ...prev, lastError: errorMessage }));
    }
  };

  const disconnect = () => {
    sessionRef.current?.close();
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach(track => track.stop());
    setStatus('offline');
  };

  return { status, transcript, debug, connect, disconnect };
}
