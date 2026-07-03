import { useState, useEffect } from 'react';
import './App.css';
import logo from './assets/logo.png';

const API_BASE = 'http://127.0.0.1:8000';

function App() {
  const [roles, setRoles] = useState([]);
  const [selectedRole, setSelectedRole] = useState('');
  const [customRole, setCustomRole] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/roles`)
      .then((res) => res.json())
      .then((data) => setRoles(data.roles))
      .catch(() => setError('Could not load role list. Is the backend running?'));
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError('');
  };

  const handleSubmit = async () => {
    setError('');
    setResult(null);

    if (!file) {
      setError('Please upload a CV file (PDF, PNG, or JPG).');
      return;
    }

    const roleToSend = selectedRole === 'other' ? customRole.trim() : selectedRole;

    if (!roleToSend) {
      setError('Please select a role or type your own.');
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('role', roleToSend);

    try {
      const res = await fetch(`${API_BASE}/rate-cv`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Something went wrong rating your CV.');
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const scoreColor = (score) => {
    if (score >= 8) return '#3D9E96';
    if (score >= 5) return '#D9A441';
    return '#C15050';
  };

  return (
    <div className="app">
 <header className="header">
  <div className="logo-wrap">
    <img src={logo} alt="HireLens logo" className="logo" />
  </div>
  <h1>HireLens by Akk</h1>
  <p>Rate your CV the way Pakistani recruiters actually see it.</p>
</header>
      <div className="card">
       <label className="field-label">Upload your CV</label>
<label className="dropzone" htmlFor="cv-upload">
  <input
    id="cv-upload"
    type="file"
    accept=".pdf,.png,.jpg,.jpeg"
    onChange={handleFileChange}
    className="dropzone-input"
  />
  <div className="dropzone-content">
    <div className="dropzone-icon">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <path d="M12 16V4M12 4L7 9M12 4L17 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M4 16V18C4 19.1046 4.89543 20 6 20H18C19.1046 20 20 19.1046 20 18V16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    </div>
    <div className="dropzone-text">
      {file ? file.name : 'Click to upload your CV'}
    </div>
    <div className="dropzone-hint">PDF, PNG, or JPG</div>
  </div>
</label>
        <label className="field-label" style={{ marginTop: '1.25rem' }}>Target role</label>
        <select value={selectedRole} onChange={(e) => setSelectedRole(e.target.value)}>
          <option value="">Select a role</option>
          {roles.map((role) => (
            <option key={role} value={role}>{role}</option>
          ))}
          <option value="other">Other (type your own)</option>
        </select>

        {selectedRole === 'other' && (
          <input
            type="text"
            placeholder="e.g. backend developer with API experience"
            value={customRole}
            onChange={(e) => setCustomRole(e.target.value)}
            style={{ marginTop: '0.75rem' }}
          />
        )}

        <button onClick={handleSubmit} disabled={loading} style={{ marginTop: '1.5rem' }}>
          {loading ? 'Rating your CV...' : 'Rate my CV'}
        </button>

        {error && <p className="error-text">{error}</p>}
      </div>

      {result && (
        <div className="card results">
          <div className="overall-score" style={{ color: scoreColor(result.overall_score) }}>
            {result.overall_score}<span>/10</span>
          </div>
          <p className="verdict">{result.verdict}</p>

          {result.role_resolution && result.role_resolution.source === 'embedding_match' && (
            <p className="role-note">
              Rated against: <strong>{result.role_resolution.matched_role}</strong>
            </p>
          )}

          <div className="categories-grid">
            {Object.entries(result.categories).map(([key, cat]) => (
              <div className="category-card" key={key}>
                <div className="category-header">
                  <span className="category-name">{key}</span>
                  <span className="category-score" style={{ color: scoreColor(cat.score) }}>
                    {cat.score}/10
                  </span>
                </div>
                <p className="category-comment">{cat.comment}</p>
              </div>
            ))}
          </div>

          <div className="top-fixes">
            <h3>Top fixes</h3>
            <ul>
              {result.top_fixes.map((fix, i) => (
                <li key={i}>{fix}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;