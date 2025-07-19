const express = require("express");
const router = express.Router();
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");
const stream = require("stream");

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const FASTAPI_BASE = "http://localhost:8000";

router.post("/transcribe", upload.single("audio"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No audio file provided." });
  }

  try {
    const formData = new FormData();

    formData.append("audio", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const response = await axios.post(`${FASTAPI_BASE}/transcribe`, formData, {
      headers: {
        ...formData.getHeaders(),
      },
    });

    res.json(response.data);
  } catch (error) {
    console.error(
      "Error communicating with FastAPI transcribe:",
      error.message
    );
    res.status(500).json({
      error: "FastAPI transcription failed",
      details: error.response?.data || error.message,
    });
  }
});

router.post("/scrape-url", async (req, res) => {
  const { url } = req.body;
  if (!url) return res.status(400).json({ error: "URL is required" });

  try {
    const response = await axios.post(`${FASTAPI_BASE}/scrape`, { url });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with FastAPI scrape:", error.message);
    res.status(500).json({
      error: "FastAPI scraping failed",
      details: error.response?.data || error.message,
    });
  }
});

router.post("/spin", async (req, res) => {
  const { text } = req.body;
  if (!text)
    return res.status(400).json({ error: "Text is required for spin" });

  try {
    const response = await axios.post(`${FASTAPI_BASE}/spin`, { text });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with FastAPI spin:", error.message);
    res.status(500).json({
      error: "FastAPI spin failed",
      details: error.response?.data || error.message,
    });
  }
});

router.post("/review", async (req, res) => {
  const { text } = req.body;
  if (!text)
    return res.status(400).json({ error: "Text is required for review" });

  try {
    const response = await axios.post(`${FASTAPI_BASE}/review`, { text });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with FastAPI review:", error.message);
    res.status(500).json({
      error: "FastAPI review failed",
      details: error.response?.data || error.message,
    });
  }
});

router.post("/feedback", async (req, res) => {
  const {
    spun_text,
    original_text,
    ai_review_score,
    manual_feedback,
    metadata,
  } = req.body;
  if (
    !spun_text ||
    ai_review_score === undefined ||
    manual_feedback === undefined
  )
    return res.status(400).json({ error: "Incomplete feedback data" });

  try {
    const response = await axios.post(`${FASTAPI_BASE}/feedback`, {
      spun_text,
      original_text,
      ai_review_score,
      manual_feedback,
      metadata: metadata || {},
    });
    res.json(response.data);
  } catch (error) {
    console.error("Error communicating with FastAPI feedback:", error.message);
    res.status(500).json({
      error: "FastAPI feedback logging failed",
      details: error.response?.data || error.message,
    });
  }
});

module.exports = router;
