import { useState, useEffect, useRef } from 'react';
import { Camera, Users, Activity, UserPlus, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function App() {
  const [students, setStudents] = useState([]);
  const [systemStatus, setSystemStatus] = useState('connecting');
  const [showRegister, setShowRegister] = useState(false);
  const registerNameRef = useRef(null);

  // Poll backend for student data
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/status');
        if (res.ok) setSystemStatus('online');
        else setSystemStatus('error');
      } catch (e) {
        setSystemStatus('offline');
      }
    };
    
    fetchStatus();
    const statusInterval = setInterval(fetchStatus, 5000);

    const fetchStudents = async () => {
      if (systemStatus === 'offline') return;
      try {
        const res = await fetch('http://localhost:5000/api/students');
        const data = await res.json();
        setStudents(data.students);
      } catch (e) {
        // Silent fail for polling
      }
    };

    const studentsInterval = setInterval(fetchStudents, 500);

    return () => {
      clearInterval(statusInterval);
      clearInterval(studentsInterval);
    };
  }, [systemStatus]);

  const handleRegister = async (e) => {
    e.preventDefault();
    const name = registerNameRef.current.value;
    if (!name) return;

    try {
      const res = await fetch('http://localhost:5000/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });
      const data = await res.json();
      if (data.success) {
        alert(`Success: ${data.message}`);
        setShowRegister(false);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (e) {
      alert("Failed to connect to backend");
    }
  };

  // Compute chart data
  const engagementCounts = students.reduce((acc, s) => {
    const state = s.engagement_state.split(' / ')[0]; // E.g. "Engaged" from "Engaged / Positive"
    acc[state] = (acc[state] || 0) + 1;
    return acc;
  }, {});

  const chartData = Object.entries(engagementCounts).map(([name, count]) => ({
    name, count
  }));

  const COLORS = {
    'Engaged': '#22c55e',       // green
    'Passively Engaged': '#3b82f6', // blue
    'Struggling': '#eab308',    // yellow
    'Disengaged': '#ef4444',    // red
    'Unknown': '#64748b'        // slate
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-6 pb-20">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 pb-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-600/20 text-blue-400 rounded-xl border border-blue-500/30 shadow-[0_0_15px_rgba(59,130,246,0.2)]">
            <Activity size={28} className="animate-pulse" />
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              NeuralSight Classroom
            </h1>
            <p className="text-sm text-slate-500 font-medium">AI Engagement Sensing</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium border ${
            systemStatus === 'online' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 
            systemStatus === 'connecting' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : 
            'bg-rose-500/10 text-rose-400 border-rose-500/20'
          }`}>
            <div className={`w-2 h-2 rounded-full ${systemStatus === 'online' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`}></div>
            {systemStatus === 'online' ? 'System Live' : systemStatus === 'connecting' ? 'Connecting...' : 'API Offline'}
          </div>

          <button 
            onClick={() => setShowRegister(true)}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-lg font-medium shadow-lg shadow-blue-900/40 transition-all hover:scale-105 active:scale-95"
          >
            <UserPlus size={18} />
            <span>Register Face</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Col: Video Feed (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="glass-panel p-1 rounded-2xl overflow-hidden relative group">
            <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1 bg-black/60 backdrop-blur-md text-white/90 text-sm font-medium rounded-full border border-white/10">
              <Camera size={14} className="text-red-400" />
              <span>LIVE</span>
            </div>
            
            {systemStatus === 'online' ? (
              <img 
                src="http://localhost:5000/video_feed" 
                alt="Webcam Feed" 
                className="w-full aspect-video object-cover rounded-xl bg-slate-900"
              />
            ) : (
              <div className="w-full aspect-video flex flex-col items-center justify-center bg-slate-900 rounded-xl text-slate-500 border border-slate-800">
                <AlertTriangle size={48} className="mb-4 text-slate-700" />
                <p>Waiting for video stream...</p>
                <p className="text-sm mt-2 opacity-60">Run <code className="text-rose-400">python backend/app.py</code></p>
              </div>
            )}
            
            <div className="absolute inset-0 shadow-[inset_0_0_50px_rgba(0,0,0,0.5)] rounded-2xl pointer-events-none"></div>
          </div>

          {/* Aggregated Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="glass-panel p-6 rounded-2xl flex flex-col">
              <span className="text-slate-400 text-sm font-medium flex items-center gap-2 mb-2">
                <Users size={16} /> Total Present
              </span>
              <span className="text-4xl font-light text-white">{students.length}</span>
            </div>
            
            <div className="glass-panel p-6 rounded-2xl flex flex-col col-span-2">
              <span className="text-slate-400 text-sm font-medium mb-4">Class Overview</span>
              <div className="h-20 w-full">
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                      <XAxis type="number" hide />
                      <YAxis dataKey="name" type="category" hide />
                      <Tooltip 
                        cursor={{fill: 'rgba(255,255,255,0.05)'}}
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc' }}
                        itemStyle={{ color: '#f8fafc' }}
                      />
                      <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={24}>
                        {chartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[entry.name] || COLORS['Unknown']} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-600 text-sm italic">
                    No data yet
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Student Roster (5 cols) */}
        <div className="lg:col-span-5 flex flex-col">
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 text-slate-200">
            Student Roster
            <span className="px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-400 text-xs font-bold font-mono">
              {students.length}
            </span>
          </h2>
          
          <div className="flex-1 space-y-4 pr-2 overflow-y-auto max-h-[calc(100vh-250px)] custom-scrollbar">
            {students.length === 0 ? (
              <div className="h-32 glass-panel rounded-2xl flex items-center justify-center text-slate-500 border-dashed">
                No students detected in frame
              </div>
            ) : (
              students.map((student, i) => {
                const isEngaged = student.engagement_state.includes("Positive") || student.engagement_state.includes("Passively");
                const isDistressed = student.engagement_state.includes("Distressed") || student.engagement_state === "Disengaged";
                const isStruggling = student.engagement_state.includes("Struggling");
                
                return (
                  <div key={i} className="glass-panel p-5 rounded-xl flex items-center gap-4 transition-all hover:bg-slate-800/80 group">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center text-lg font-bold text-slate-300 border-2 border-slate-700 shadow-inner overflow-hidden">
                        {student.name !== "Unknown" ? student.name.charAt(0).toUpperCase() : "?"}
                      </div>
                      <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-slate-900 ${
                        isEngaged ? 'bg-emerald-500' : 
                        isStruggling ? 'bg-amber-500' : 
                        isDistressed ? 'bg-rose-500' : 'bg-slate-500'
                      }`}></div>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-slate-100 truncate group-hover:text-white transition-colors">
                        {student.name}
                      </h3>
                      <div className="flex items-center gap-2 mt-1">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-md ${
                          isEngaged ? 'bg-emerald-500/10 text-emerald-400' : 
                          isStruggling ? 'bg-amber-500/10 text-amber-400' : 
                          isDistressed ? 'bg-rose-500/10 text-rose-400' : 'bg-slate-800 text-slate-400'
                        }`}>
                          {student.engagement_state}
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex gap-6 items-center">
                      <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Gaze</div>
                        <div className={`text-sm font-medium capitalize ${
                          student.gaze_direction === 'Forward' ? 'text-emerald-400' : 'text-amber-400'
                        }`}>
                          {student.gaze_direction || 'N/A'}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Attention</div>
                        <div className={`text-sm font-bold ${
                          student.attention_score >= 80 ? 'text-emerald-400' : 
                          student.attention_score >= 50 ? 'text-amber-400' : 'text-rose-400'
                        }`}>
                          {student.attention_score != null ? `${student.attention_score.toFixed(0)}%` : 'N/A'}
                        </div>
                      </div>
                      <div className="text-right border-l border-slate-700 pl-4 ml-2">
                        <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Emotion</div>
                        <div className="text-sm font-medium text-slate-300 capitalize">{student.fer_emotion}</div>
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </main>

      {/* Registration Modal Overlay */}
      {showRegister && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm" onClick={() => setShowRegister(false)}></div>
          <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl p-6 w-full max-w-md relative z-10 animate-in fade-in zoom-in duration-200">
            <h3 className="text-xl font-bold mb-2">Register Face</h3>
            <p className="text-slate-400 text-sm mb-6">
              Look directly at the camera. The system will capture the most prominent face and link it to this name.
            </p>
            
            <form onSubmit={handleRegister}>
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-300 mb-2">Student Name</label>
                <input 
                  type="text" 
                  ref={registerNameRef}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="e.g. John Doe"
                  autoFocus
                />
              </div>
              
              <div className="flex justify-end gap-3">
                <button 
                  type="button" 
                  onClick={() => setShowRegister(false)}
                  className="px-4 py-2 rounded-lg text-slate-300 font-medium hover:bg-slate-800 transition-colors"
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium shadow-lg shadow-blue-500/30 transition-all"
                >
                  Capture & Register
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
      
    </div>
  )
}

export default App
