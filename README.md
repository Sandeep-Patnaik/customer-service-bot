# Customer Service Bot

## Overview

This project is an AI-powered Customer Service Chatbot built using Streamlit, LangChain, FAISS, Hugging Face Embeddings, and Google Gemini. The chatbot can answer user queries based on a knowledge base stored in a vector database.

## Features

* Customer service chatbot interface using Streamlit
* Knowledge retrieval using FAISS vector database
* Semantic search using Hugging Face embeddings
* Response generation using Google Gemini
* Dynamic knowledge base expansion
* Automatic vector database updates using a scheduler
* Duplicate file detection using processed file tracking

## Task 1: Dynamic Knowledge Base Expansion

The chatbot's knowledge base can be expanded without rebuilding the entire system.

### Implementation

* New knowledge files are placed in the `NewData` folder.
* The `update_vector_db()` function detects and processes new CSV files.
* Processed files are tracked using `processed_files.txt`.
* New information is added directly to the existing FAISS vector database.
* A scheduler automatically checks for and updates the knowledge base at regular intervals.

### Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Embeddings
* Google Gemini
* Schedule

## Project Structure

dataset/ - Original dataset

NewData/ - New knowledge files

src/ - Source code

requirements.txt - Project dependencies

## Current Status

Task 1 Completed: Dynamic Knowledge Base Expansion
