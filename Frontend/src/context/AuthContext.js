import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode'; // Correct import for jwt-decode

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [loading, setLoading] = useState(true);

    const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

    const checkAuth = useCallback(() => {
        const token = localStorage.getItem('access_token');
        if (token) {
            try {
                const decodedToken = jwtDecode(token);
                const currentTime = Date.now() / 1000;
                if (decodedToken.exp > currentTime) {
                    // Token is valid
                    setIsAuthenticated(true);
                    // Assuming role is in token, if not, you'd fetch user details from /users/me
                    setUser({ username: decodedToken.sub, role: decodedToken.role || 'Guest' }); // Default to Guest if no role
                    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                } else {
                    // Token expired
                    logout();
                }
            } catch (error) {
                console.error("Failed to decode token:", error);
                logout(); // Invalid token
            }
        } else {
            setIsAuthenticated(false);
            setUser(null);
            delete axios.defaults.headers.common['Authorization'];
        }
        setLoading(false);
    }, []); // Empty dependency array because checkAuth should not change

    useEffect(() => {
        checkAuth();
    }, [checkAuth]);

    const login = async (username, password) => {
        try {
            // Note: axios.post with URLSearchParams is for x-www-form-urlencoded
            const response = await axios.post(`${API_BASE_URL}/token`, new URLSearchParams({
                username: username,
                password: password
            }));
            const { access_token } = response.data;
            localStorage.setItem('access_token', access_token);
            axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
            checkAuth(); // Re-check authentication status after login to update user state
            return true;
        } catch (error) {
            console.error("Login failed in AuthContext:", error.response?.data || error.message);
            throw error; // Re-throw to be caught by LoginPage
        }
    };

    const logout = useCallback(() => {
        localStorage.removeItem('access_token');
        delete axios.defaults.headers.common['Authorization'];
        setIsAuthenticated(false);
        setUser(null);
        // Do NOT navigate here. Let the component using logout handle navigation.
    }, []); // Empty dependency array because logout should not change

    const authContextValue = {
        user,
        isAuthenticated,
        loading,
        login,
        logout,
        checkAuth,
    };

    return (
        <AuthContext.Provider value={authContextValue}>
            {!loading && children} {/* Only render children after auth check is complete */}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};