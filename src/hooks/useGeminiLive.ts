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

  const statusRef = useRef<ConnectionStatus>('offline');

  // Update status and ref together
  const updateStatus = (newStatus: ConnectionStatus) => {
    statusRef.current = newStatus;
    setStatus(newStatus);
  };

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
        sampleRate: 24000
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
        if (statusRef.current === 'speaking') updateStatus('listening');
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
    updateStatus('speaking');
  }, []);

  // Highly optimized base64 encoding for audio chunks
  const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    const chunk = 8192;
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
        throw new Error("The Voice Concierge requires a secure connection (HTTPS or localhost).");
      }

      updateStatus('connecting');
      setDebug(prev => ({ ...prev, wsStatus: 'connecting', lastError: undefined }));

      // 2. Start audio and mic requests in parallel
      const audioInitPromise = initAudio();
      
      const apiKey = 
        (import.meta as any).env?.VITE_GEMINI_API_KEY || 
        (import.meta as any).env?.GEMINI_API_KEY ||
        process.env.GEMINI_API_KEY;
      
      if (!apiKey || apiKey === '""' || apiKey === "''") {
        throw new Error("Gemini API Key is missing.");
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Microphone access is not available.");
      }

      const micPromise = navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });

      const [_, stream] = await Promise.all([audioInitPromise, micPromise]);
      
      streamRef.current = stream;
      setDebug(prev => ({ ...prev, micPermission: 'granted' }));

      const ai = new GoogleGenAI({ apiKey });

      // Create the session promise first to avoid race conditions in callbacks
      const sessionPromise = ai.live.connect({
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
            updateStatus('listening');

            // Use the promise to send the initial greeting safely
            sessionPromise.then(session => {
              session.send({
                clientContent: {
                  turns: [{
                    role: 'user',
                    parts: [{ text: "Hello Cam. Please introduce yourself and ask if I'm ready to book a reservation." }]
                  }]
                }
              });
            });

            // Setup audio processing
            const source = audioContextRef.current!.createMediaStreamSource(stream);
            const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              // Use ref to avoid stale closure
              if (statusRef.current === 'offline' || statusRef.current === 'error') return;
              
              const inputData = e.inputBuffer.getChannelData(0);
              const contextSampleRate = audioContextRef.current!.sampleRate;
              
              const ratio = contextSampleRate / 16000;
              const newLength = Math.floor(inputData.length / ratio);
              const pcmData = new Int16Array(newLength);
              
              for (let i = 0; i < newLength; i++) {
                const sample = inputData[Math.floor(i * ratio)] || 0;
                pcmData[i] = sample < 0 ? sample * 32768 : sample * 32767;
              }
              
              const base64Data = arrayBufferToBase64(pcmData.buffer);
              sessionPromise.then(session => {
                session.sendRealtimeInput({
                  media: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
                });
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
                updateStatus('listening');
              }

              // Handle Tool Calls
              const toolCall = message.toolCall;
              if (toolCall) {
                const call = toolCall.functionCalls[0];
                sessionPromise.then(session => {
                  if (call.name === 'update_reservation') {
                    onBookingUpdate(call.args as Partial<BookingDetails>);
                    setDebug(prev => ({ ...prev, lastToolCall: `update: ${JSON.stringify(call.args)}` }));
                    
                    session.sendToolResponse({
                      functionResponses: [{
                        id: call.id,
                        response: { output: { success: true } }
                      }]
                    });
                  } else if (call.name === 'submit_reservation') {
                    onBookingSubmit();
                    setDebug(prev => ({ ...prev, lastToolCall: 'submit' }));
                    
                    session.sendToolResponse({
                      functionResponses: [{
                        id: call.id,
                        response: { output: { success: true } }
                      }]
                    });
                  }
                });
              }

              // Handle Transcripts
              const userTranscript = (message.serverContent as any)?.userTurn?.parts?.find((p: any) => p.text)?.text;
              if (userTranscript) {
                setTranscript(prev => [...prev, { role: 'user', text: userTranscript, timestamp: Date.now() }]);
              }

              const modelTranscript = message.serverContent?.modelTurn?.parts?.find(p => p.text)?.text;
              if (modelTranscript) {
                setTranscript(prev => [...prev, { role: 'model', text: modelTranscript, timestamp: Date.now() }]);
              }
            } catch (msgErr) {
              console.error("Error processing message:", msgErr);
            }
          },
          onclose: () => {
            console.log("Gemini Live connection closed");
            setDebug(prev => ({ ...prev, wsStatus: 'closed' }));
            updateStatus('offline');
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setDebug(prev => ({ ...prev, wsStatus: 'error', lastError: String(err) }));
            updateStatus('error');
          }
        }
      });

      sessionRef.current = await sessionPromise;

    } catch (err: any) {
      console.error("Connection Error:", err);
      updateStatus('error');
      
      let errorMessage = String(err);
      if (err.name === 'NotAllowedError' || err.message?.includes('Permission denied')) {
        errorMessage = "Microphone access was denied.";
        setDebug(prev => ({ ...prev, micPermission: 'denied' }));
      }
      
      setDebug(prev => ({ ...prev, lastError: errorMessage }));
    }
  };

  const disconnect = () => {
    sessionRef.current?.close();
    processorRef.current?.disconnect();
    streamRef.current?.getTracks().forEach(track => track.stop());
    updateStatus('offline');
  };

  return { status, transcript, debug, connect, disconnect };
}
