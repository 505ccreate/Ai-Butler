import { BookingDetails } from '../types';
import { Car, User, MapPin, Calendar, Clock, Briefcase, MessageSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface BookingPanelProps {
  booking: BookingDetails;
}

export function BookingPanel({ booking }: BookingPanelProps) {
  const fields = [
    { id: 'service_type', label: 'Service', value: booking.service_type, icon: Car },
    { id: 'vehicle_type', label: 'Vehicle', value: booking.vehicle_type, icon: Car },
    { id: 'pickup_date', label: 'Date', value: booking.pickup_date, icon: Calendar },
    { id: 'pickup_time', label: 'Time', value: booking.pickup_time, icon: Clock },
    { id: 'pickup_address', label: 'Pickup', value: booking.pickup_address, icon: MapPin },
    { id: 'dropoff_address', label: 'Dropoff', value: booking.dropoff_address, icon: MapPin },
    { id: 'passenger_count', label: 'Passengers', value: booking.passenger_count, icon: User },
    { id: 'luggage_count', label: 'Luggage', value: booking.luggage_count, icon: Briefcase },
    { id: 'customer_name', label: 'Name', value: booking.customer_name, icon: User },
    { id: 'customer_phone', label: 'Phone', value: booking.customer_phone, icon: User },
    { id: 'customer_email', label: 'Email', value: booking.customer_email, icon: User },
    { id: 'flight_number', label: 'Flight', value: booking.flight_number, icon: Car },
    { id: 'special_requests', label: 'Notes', value: booking.special_requests, icon: MessageSquare },
  ];

  return (
    <div className="bg-white border border-zinc-200 rounded-2xl shadow-sm overflow-hidden">
      <div className="bg-zinc-50 px-6 py-4 border-b border-zinc-200 flex items-center justify-between">
        <h3 className="font-serif italic text-zinc-900">Reservation Draft</h3>
        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-[10px] uppercase tracking-widest text-zinc-400 font-bold">Live Sync Active</span>
        </div>
      </div>
      
      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
        {fields.map((field) => (
          <div key={field.id} className={`flex flex-col gap-1.5 ${field.label === 'Notes' ? 'md:col-span-2' : ''}`}>
            <div className="flex items-center gap-2 text-zinc-400">
              <field.icon size={12} />
              <span className="text-[10px] uppercase tracking-widest font-bold">{field.label}</span>
            </div>
            <div className="relative min-h-[20px]">
              <AnimatePresence mode="wait">
                <motion.div
                  key={String(field.value)}
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -5 }}
                  className={`text-sm ${field.value ? 'text-zinc-900 font-medium' : 'text-zinc-300 italic'}`}
                >
                  {field.value || 'Not captured'}
                </motion.div>
              </AnimatePresence>
              {field.value && (
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: [0, 1, 0] }}
                  transition={{ duration: 0.5 }}
                  className="absolute -bottom-1 left-0 right-0 h-0.5 bg-emerald-400 origin-left"
                />
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
