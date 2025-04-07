
# 📚 Printrado Telegram Bot

## 🌟 Overview
The **Printrado Telegram Bot** is a smart recommendation assistant designed to help users discover and purchase books from the Printrado platform. It leverages advanced search techniques, semantic matching, and user-friendly interactions to provide a seamless book discovery experience. 📖✨

---

## 🗂️ File Structure

### 🔑 Main Files
- **`main.py`**: The core script containing bot logic, commands, and user interaction workflows.
- **`config.py`**: Handles configuration management, including environment variables and database settings.
- **`requirements.txt`**: Lists all Python dependencies required to run the bot.
- **`upload.sh`**: A shell script for deploying the bot to an EC2 instance via SCP.
- **`rds-ca-2019-root.pem`**: SSL certificate for secure database connections to AWS RDS.

### 📦 Data and Logs
- **`data/product_embeddings.pkl`**: Precomputed embeddings for books used in semantic search.
- **`logs/bookbot.log`**: Log file for tracking bot activity, errors, and debugging information.

### 🛠️ Compiled Files
- **`__pycache__/config.cpython-39.pyc`**: Compiled Python bytecode for the `config.py` file.

---

## 🛠️ Installation

### ✅ Prerequisites
- Python 3.8 or higher 🐍
- Telegram Bot API token 🤖
- MySQL database with the required schema 🗄️
- AWS RDS SSL certificate (`rds-ca-2019-root.pem`) 🔒

### 🚀 Steps to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/printrado-telegram-bot.git
   cd printrado-telegram-bot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Bot**:
   - Create a `.env` file with the following variables:
     ```
     API_TOKEN=your_telegram_bot_token
     DB_HOST=your_database_host
     DB_USER=your_database_user
     DB_PASS=your_database_password
     ```

4. **Run the Bot**:
   ```bash
   python main.py
   ```

---

## 🚢 Deployment
To deploy the bot to an EC2 instance, use the `upload.sh` script:
```bash
bash upload.sh
```

---

## ✨ Features
- **📈 Top-Selling Books**: Displays the most popular books based on sales data.
- **🔍 Search by Topic**: Allows users to search for books by providing keywords or topics.
- **📝 Spelling Correction**: Automatically corrects misspelled search terms using SymSpell and fuzzy matching.
- **🔤 Abbreviation Handling**: Expands or interprets common abbreviations (e.g., "AI" → "Artificial Intelligence").
- **🧠 Semantic Search**: Uses Sentence Transformers for semantic similarity-based book recommendations.
- **🖼️ Inline Images**: Displays book covers alongside detailed information and purchase links.
- **📜 Pagination**: Supports browsing through multiple pages of results.
- **⚠️ Error Handling**: Provides user-friendly error messages and recovery options.

---

## 📄 Logs
The bot logs its activity in `logs/bookbot.log`, which includes:
- Initialization details
- Search queries and results
- Errors and cleanup information

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 📧 Contact
For questions or support, please contact **[mohmedessam166202@gmail.com](mailto:mohmedessam166202@gmail.com)**.
```
