// offline-rag-system/frontend/src/Dashboard.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';

const Dashboard = ({ token }) => {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [citations, setCitations] = useState([]);
  const [file, setFile] = useState(null);
  const [viewContent, setViewContent] = useState('');
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [users, setUsers] = useState([]);
  const [showUsers, setShowUsers] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('Analyst');
  const [deleteUsername, setDeleteUsername] = useState('');
  const [role, setRole] = useState('');
  const [recognition, setRecognition] = useState(null);

  useEffect(() => {
    fetchHistory();
    fetchRole();
    fetchUsers();

    // Setup voice recognition
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const rec = new SpeechRecognition();
      rec.continuous = false;
      rec.interimResults = false;
      rec.lang = 'en-US';
      rec.onresult = (event) => {
        setQuery(event.results[0][0].transcript);
      };
      setRecognition(rec);
    }
  }, []);

  const fetchRole = async () => {
    try {
      const response = await axios.get('http://localhost:8000/users/me', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setRole(response.data.role);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get('http://localhost:8000/memory', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(response.data.history);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchUsers = async () => {
    try {
      const response = await axios.get('http://localhost:8000/users', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUsers(response.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateUser = async () => {
    try {
      await axios.post('http://localhost:8000/users', {
        username: newUsername,
        password: newPassword,
        role: newRole
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchUsers();
      alert('User created');
    } catch (err) {
      console.error(err);
    }
  };

  const handleDeleteUser = async () => {
    try {
      await axios.delete(`http://localhost:8000/users/${deleteUsername}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchUsers();
      alert('User deleted');
    } catch (err) {
      console.error(err);
    }
  };

  const handleQuery = async () => {
    try {
      const response = await axios.post('http://localhost:8000/query', { question: query }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAnswer(response.data.answer);
      setCitations(response.data.citations);
      fetchHistory();
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      await axios.post('http://localhost:8000/upload', formData, {
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'multipart/form-data' }
      });
      alert('Uploaded');
    } catch (err) {
      console.error(err);
    }
  };

  const viewSource = async (fileId) => {
    try {
      const response = await axios.get(`http://localhost:8000/view_source/${fileId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setViewContent(response.data.content);
    } catch (err) {
      console.error(err);
    }
  };

  const startVoice = () => {
    if (recognition) {
      recognition.start();
    } else {
      alert('Voice recognition not supported in this browser.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <h1 className="text-cyan-400 text-3xl mb-4">Dashboard</h1>
      <div className="mb-8">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-2" />
        <button onClick={handleUpload} className="bg-cyan-500 text-white p-2">Upload</button>
      </div>
      <div className="mb-8 flex">
        <input 
          type="text" 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          placeholder="Ask a question" 
          className="bg-gray-700 text-white p-2 flex-grow mb-2 mr-2"
        />
        <button onClick={handleQuery} className="bg-cyan-500 text-white p-2 mr-2">Query</button>
        <button onClick={startVoice} className="bg-cyan-500 text-white p-2">Voice</button>
      </div>
      {answer && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-gray-800 p-4 mb-4">
          <p>{answer}</p>
        </motion.div>
      )}
      {citations.length > 0 && (
        <div>
          <h2 className="text-cyan-300">Citations:</h2>
          {citations.map((cit, i) => (
            <p key={i} onClick={() => viewSource(cit.file_id)} className="cursor-pointer text-blue-400">{cit.text}</p>
          ))}
        </div>
      )}
      {viewContent && <pre className="bg-gray-800 p-4 text-white">{viewContent}</pre>}
      <div className="mt-8">
        <button onClick={() => setShowHistory(!showHistory)} className="bg-cyan-500 text-white p-2 mb-4">Toggle Memory/History</button>
        {showHistory && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-gray-800 p-4 mb-4">
            <h2 className="text-cyan-300 mb-2">Your Persistent Memory</h2>
            {history.map((entry, i) => (
              <div key={i} className="mb-4 border-b border-gray-600">
                <p><strong>Time:</strong> {entry.timestamp}</p>
                <p><strong>Query:</strong> {entry.query}</p>
                <p><strong>Answer:</strong> {entry.answer}</p>
                <p><strong>Citations:</strong> {entry.citations.join(', ')}</p>
              </div>
            ))}
          </motion.div>
        )}
      </div>
      {role === 'Admin' && (
        <div className="mt-8">
          <button onClick={() => setShowUsers(!showUsers)} className="bg-cyan-500 text-white p-2 mb-4">Toggle User Management</button>
          {showUsers && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-gray-800 p-4">
              <h2 className="text-cyan-300 mb-2">Manage Users</h2>
              <ul className="mb-4">
                {users.map((user, i) => (
                  <li key={i} className="text-white">{user.username} - {user.role}</li>
                ))}
              </ul>
              <h3 className="text-cyan-300 mb-2">Create User</h3>
              <input 
                type="text" 
                value={newUsername} 
                onChange={(e) => setNewUsername(e.target.value)} 
                placeholder="New Username" 
                className="bg-gray-700 text-white p-2 mb-2 w-full"
              />
              <input 
                type="password" 
                value={newPassword} 
                onChange={(e) => setNewPassword(e.target.value)} 
                placeholder="New Password" 
                className="bg-gray-700 text-white p-2 mb-2 w-full"
              />
              <select 
                value={newRole} 
                onChange={(e) => setNewRole(e.target.value)} 
                className="bg-gray-700 text-white p-2 mb-2 w-full"
              >
                <option value="Admin">Admin</option>
                <option value="Analyst">Analyst</option>
                <option value="Guest">Guest</option>
              </select>
              <button onClick={handleCreateUser} className="bg-cyan-500 text-white p-2 mb-4">Create User</button>
              <h3 className="text-cyan-300 mb-2">Delete User</h3>
              <input 
                type="text" 
                value={deleteUsername} 
                onChange={(e) => setDeleteUsername(e.target.value)} 
                placeholder="Username to Delete" 
                className="bg-gray-700 text-white p-2 mb-2 w-full"
              />
              <button onClick={handleDeleteUser} className="bg-red-500 text-white p-2">Delete User</button>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
};

export default Dashboard;