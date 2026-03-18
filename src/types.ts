/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface BookingDetails {
  service_type?: string;
  vehicle_type?: string;
  pickup_date?: string;
  pickup_time?: string;
  pickup_address?: string;
  dropoff_address?: string;
  passenger_count?: number;
  luggage_count?: number;
  customer_name?: string;
  customer_phone?: string;
  customer_email?: string;
  flight_number?: string;
  special_requests?: string;
}

export type ConnectionStatus = 'offline' | 'connecting' | 'listening' | 'speaking' | 'error';

export interface DebugInfo {
  micPermission: 'prompt' | 'granted' | 'denied';
  wsStatus: 'closed' | 'connecting' | 'open' | 'error';
  lastToolCall?: string;
  lastError?: string;
}

export interface TranscriptMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}
