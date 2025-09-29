// offline-rag-system/frontend/src/Login.js
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const Login = ({ setToken }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [verifying, setVerifying] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setVerifying(true);
    try {
      const response = await axios.post('http://localhost:8000/token', 
        `username=${username}&password=${password}`, 
        { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
      );
      localStorage.setItem('token', response.data.access_token);
      setToken(response.data.access_token);
      setSuccess(true);
      setTimeout(() => window.location.href = '/dashboard', 2000);
    } catch (err) {
      setError('Invalid credentials');
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <motion.div 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }} 
        className="bg-gray-800 p-8 rounded-lg shadow-lg"
      >
        <h1 className="text-cyan-400 text-4xl mb-4">NTRO</h1>
        <p className="text-cyan-300 mb-8">NTRO Intelligence Agency</p>
        <form onSubmit={handleSubmit}>
          <input 
            type="text" 
            placeholder="USERNAME" 
            value={username} 
            onChange={(e) => setUsername(e.target.value)}
            className="bg-gray-700 text-white p-2 mb-4 w-full"
          />
          <input 
            type="password" 
            placeholder="PASSWORD" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)}
            className="bg-gray-700 text-white p-2 mb-4 w-full"
          />
          <motion.button 
            type="submit"
            className="bg-cyan-500 text-white p-2 w-full"
            whileHover={{ scale: 1.05 }}
          >
            ACCESS
          </motion.button>
        </form>
        {verifying && (
          <motion.p 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }}
            className="text-cyan-300 mt-4"
          >
            Verifying Credentials...
          </motion.p>
        )}
        {success && (
          <motion.div 
            initial={{ scale: 0 }} 
            animate={{ scale: 1 }}
            className="text-green-500 mt-4"
          >
            ✓
          </motion.div>
        )}
        {error && <p className="text-red-500 mt-4">{error}</p>}
      </motion.div>
    </div>
  );
};

export default Login;