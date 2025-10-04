import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode'; // ✅ named import

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [loading, setLoading] = useState(true);

    const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

    const logout = useCallback(() => {
        localStorage.removeItem('access_token');
        delete axios.defaults.headers.common['Authorization'];
        setIsAuthenticated(false);
        setUser(null);
    }, []);

    const checkAuth = useCallback(() => {
        const token = localStorage.getItem('access_token');
        if (token) {
            try {
                const decodedToken = jwtDecode(token); // ✅ named import
                const currentTime = Date.now() / 1000;
                if (decodedToken.exp > currentTime) {
                    setIsAuthenticated(true);
                    setUser({ username: decodedToken.sub, role: decodedToken.role || 'Guest' });
                    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                } else {
                    logout();
                }
            } catch (error) {
                console.error("Failed to decode token:", error);
                logout();
            }
        } else {
            setIsAuthenticated(false);
            setUser(null);
            delete axios.defaults.headers.common['Authorization'];
        }
        setLoading(false);
    }, [logout]);

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    const login = async (username, password) => {
        try {
            const response = await axios.post(`${API_BASE_URL}/token`, new URLSearchParams({
                username,
                password
            }));
            const { access_token } = response.data;
            localStorage.setItem('access_token', access_token);
            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
            checkAuth();
            return true;
        } catch (error) {
            console.error("Login failed:", error.response?.data || error.message);
            throw error;
        }
    };

    return (
        <AuthContext.Provider value={{ user, isAuthenticated, loading, login, logout, checkAuth }}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) throw new Error('useAuth must be used within an AuthProvider');
    return context;
};
