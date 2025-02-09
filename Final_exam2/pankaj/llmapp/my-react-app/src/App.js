import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [file, setFile] = useState(null);
  const [answer, setAnswer] = useState('');
  const [pdfFiles, setPdfFiles] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState('');
  const [topicSummaries, setTopicSummaries] = useState({});
  const [historyByTopic, setHistoryByTopic] = useState([]);

  useEffect(() => {
    fetchHistory();
    fetchPdfFiles();
    fetchHistoryByTopic();
    fetchSummaryByTopic(); // Fetch summaries by topic
  }, []);

  const fetchPdfFiles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/pdf-files');
      setPdfFiles(response.data);
    } catch (error) {
      console.error('Error fetching PDF files:', error);
      setError('Failed to fetch PDF files.');
    }
  };

  const fetchSummaryByTopic = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/summarize_history');
      setTopicSummaries(response.data);
    } catch (error) {
      console.error('Error fetching summaries by topic:', error);
      setError('Failed to fetch summaries by topic.');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistoryByTopic = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/history2');
      console.log(response.data)
      setHistoryByTopic(response.data);
    } catch (error) {
      console.error('Error fetching history by topic:', error);
      setError('Failed to fetch history by topic.');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/history');
      setHistory(response.data);
    } catch (error) {
      console.error('Error fetching history:', error);
      setError('Failed to fetch history.');
    } finally {
      setLoading(false);
    }
  };

  const handleInitialize = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/initialize');
      console.log(response.data.message);
    } catch (error) {
      console.error('Error initializing PDFChatBot:', error);
      setError('Failed to initialize PDFChatBot.');
    } finally {
      setLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) {
      setError('Question cannot be empty.');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/ask', { question });
      setAnswer(response.data.answer);
      fetchHistory();
    } catch (error) {
      console.error('Error asking question:', error);
      setAnswer('Error asking question. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('No file selected.');
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('File uploaded successfully:', response.data);
      fetchPdfFiles();
      setFile(null); // Reset file input
      document.querySelector('input[type="file"]').value = '';
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('Failed to upload file.');
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://localhost:5000/summarize_history');
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error fetching summary:', error);
      setError('Failed to fetch summary.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Chatbot UI</h1>
      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}
      <button onClick={handleInitialize}>Initialize PDFChatBot</button>
      <div>
        <input
          type="text"
          value={question}
          onChange={(e) => {
            setQuestion(e.target.value);
            setError(''); // Clear error message on input change
          }}
          placeholder="Enter your question..."
        />
        <button onClick={handleAskQuestion} disabled={loading}>Ask</button>
      </div>
      <div>
        <input type="file" onChange={handleFileUpload} />
        <button onClick={handleUpload} disabled={loading}>Upload</button>
      </div>
      {answer && (
        <div>
          <strong>Answer:</strong> {answer}
        </div>
      )}

      {/* Summary Section */}
      {Object.keys(topicSummaries).map(topicId => (
        <div key={topicId}>
          <h3>Summary for Topic {topicId}:</h3>
          <p>{topicSummaries[topicId]}</p>
          <h3>Topic {topicId} History:</h3>
          <ul>
            {historyByTopic
              .filter(entry => entry.topicid === topicId)
              .map((entry, index) => (
                <li key={index}>
                  <strong>Query:</strong> {entry.query} <br />
                  <strong>Response:</strong> {entry.response} <br />
                </li>
              ))}
          </ul>
        </div>
      ))}
      {/* End of Summary Section */}
    </div>
  );
}

export default App;

