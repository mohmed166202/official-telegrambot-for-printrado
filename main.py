from symspellpy import SymSpell, Verbosity
import importlib.resources as pkg_resources
import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import pymysql
from pymysql.cursors import DictCursor
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from config import Config, ConfigurationError
from sentence_transformers import SentenceTransformer, util
import torch
from fuzzywuzzy import process, fuzz
import requests


def download_rds_cert():
    """Download the RDS CA certificate if it doesn't exist."""
    cert_file = 'rds-ca-2019-root.pem'
    if not os.path.exists(cert_file):
        try:
            url = 'https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem'
            response = requests.get(url)
            response.raise_for_status()
            with open(cert_file, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded RDS CA certificate to {cert_file}")
        except Exception as e:
            print(f"Error downloading RDS certificate: {e}")
            raise


def setup_logging() -> logging.Logger:
    """Configure application logging."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "bookbot.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class DatabaseManager:
    """Handles database operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.connection = None
        self._connect()

    def _connect(self):
        """Establish a connection to the database."""
        try:
            print(f"\nAttempting to connect to database:")
            print(f"Host: {self.config.DB_HOST}")
            print(f"User: {self.config.DB_USER}")
            print(f"Database: {self.config.DB_NAME}")
            
            connection_params = {
                'host': self.config.DB_HOST,
                'port': self.config.DB_PORT,
                'user': self.config.DB_USER,
                'password': self.config.DB_PASSWORD,
                'database': self.config.DB_NAME,
                'charset': 'utf8mb4',
                'cursorclass': DictCursor,
                'connect_timeout': self.config.DB_CONNECTION_TIMEOUT,
                'ssl': {'ca': 'rds-ca-2019-root.pem'}
            }
            
            self.connection = pymysql.connect(**connection_params)
            print("Database connection established successfully!")
            
        except pymysql.Error as e:
            logger.error(f"Database connection failed: {e}")
            logger.error(f"Attempted connection to: {self.config.DB_HOST}")
            raise DatabaseError(f"Database connection failed: {e}")

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a database query with automatic reconnection."""
        max_retries = self.config.DB_MAX_RETRIES
        for attempt in range(max_retries):
            try:
                if not self.connection or not self.connection.open:
                    self._connect()
                
                with self.connection.cursor() as cursor:
                    cursor.execute(query, params)
                    self.connection.commit()
                    return cursor.fetchall()
                    
            except (pymysql.Error, AttributeError) as e:
                logger.error(f"Database error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise DatabaseError(f"Database operation failed after {max_retries} attempts: {e}")
                self._connect()

    def get_top_selling_books(self, limit: int = 5, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch top selling books with pagination from updated schema."""
        query = """
            SELECT 
                product_id,
                product_name,
                product_url,
                image_url,
                product_length AS length,
                COALESCE(total_items_sold, 0) AS total_items_sold
            FROM ProductSales
            ORDER BY total_items_sold DESC 
            LIMIT %s OFFSET %s
        """
        return self.execute_query(query, (limit, offset))


class BookBot:
    """Main bot class for user interactions and book recommendations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.user_state: Dict[int, str] = {}
        self.user_context: Dict[int, Dict[str, Any]] = {}
        self.abbreviations = self._initialize_abbreviations()
        self.common_misspellings = self._initialize_misspellings()
        self.sym_spell = self._initialize_symspell()
        self.initialize_bot()

    def _initialize_symspell(self) -> SymSpell:
        """Initialize SymSpell for spelling correction."""
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
        sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        return sym_spell

    def _initialize_abbreviations(self) -> Dict[str, str]:
        """Initialize common abbreviations."""
        return {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'sql': 'structured query language',
            'js': 'javascript',
            'py': 'python',
            'php': 'hypertext preprocessor',
            'db': 'database',
            'oop': 'object oriented programming',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience',
            'html': 'hypertext markup language',
            'css': 'cascading style sheets',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'tcp': 'transmission control protocol',
            'ip': 'internet protocol',
            'dns': 'domain name system',
            'json': 'javascript object notation',
            'xml': 'extensible markup language',
            'csv': 'comma separated values',
            'ram': 'random access memory',
            'rom': 'read only memory',
            'cpu': 'central processing unit',
            'gpu': 'graphics processing unit',
            'ssd': 'solid state drive',
            'hdd': 'hard disk drive',
            'cli': 'command line interface',
            'gui': 'graphical user interface',
            'vpn': 'virtual private network',
            'ide': 'integrated development environment',
            'devops': 'development and operations',
            'qa': 'quality assurance',
            'ci': 'continuous integration',
            'cd': 'continuous deployment',
            'mvc': 'model view controller',
            'jwt': 'json web token',
            'rest': 'representational state transfer',
            'soap': 'simple object access protocol',
            'tls': 'transport layer security',
            'ssl': 'secure sockets layer',
            'nosql': 'not only sql',
            'orm': 'object relational mapping',
            'sdlc': 'software development life cycle',
            'agile': 'adaptive software development methodology',
            'scrum': 'agile framework for project management',
            'kanban': 'visual workflow management method',
            'docker': 'containerization platform',
            'k8s': 'kubernetes, container orchestration system',
            'vm': 'virtual machine',
            'os': 'operating system',
            'bash': 'bourne again shell',
            'ftp': 'file transfer protocol',
            'ipfs': 'interplanetary file system',
            'p2p': 'peer to peer',
            'iot': 'internet of things',
            'bi': 'business intelligence',
            'etl': 'extract transform load',
            'erp': 'enterprise resource planning',
            'crm': 'customer relationship management',
            'rsa': 'rivest shamir adleman encryption',
            'aes': 'advanced encryption standard',
            'sha': 'secure hash algorithm',
            'md5': 'message digest algorithm 5',
            'ipv4': 'internet protocol version 4',
            'ipv6': 'internet protocol version 6',
            'pdo': 'php data objects',
            'laravel': 'php framework for web applications',
            'composer': 'dependency manager for php',
            'cakephp': 'php framework for rapid development',
            'yii': 'php framework for high-performance applications',
            'symfony': 'php framework for web applications',
            'zend': 'php framework formerly known as zend framework',
            'codeigniter': 'lightweight php framework',
            'wp': 'wordpress, php-based content management system'
        }

    def _initialize_misspellings(self) -> Dict[str, str]:
        """Initialize common misspellings."""
        return {
            'machine learninig': 'machine learning',
            'mlachine learning': 'machine learning',
            # Add more common misspellings as needed
        }

    def correct_topic(self, topic: str) -> str:
        """Dynamically correct topics using fuzzy matching, SymSpell, and handle abbreviations."""
        topic = topic.strip()
        
        # Special handling for ".NET"
        if topic.lower() == ".net":
            return ".NET"
        
        # Special handling for "C#" and "C++"
        if topic.lower() == "c#":
            return "C#"
        if topic.lower() == "c++":
            return "C++"
        
        # Special handling for "C" to search for "C++" and "C#"
        if topic.lower() == "c":
            return "C"
        
        # Check for common misspellings
        if topic in self.common_misspellings:
            return self.common_misspellings[topic]
        
        # First check if it's a known abbreviation
        if topic in self.abbreviations:
            return self.abbreviations[topic]
            
        # Use SymSpell to correct misspellings
        suggestions = self.sym_spell.lookup(topic, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_topic = suggestions[0].term
            if corrected_topic != topic:
                topic = corrected_topic

        # Split multi-word topics and check each word
        words = topic.split()
        corrected_words = []

        for word in words:
            # If word length <= 3, check if it's an abbreviation
            if len(word) <= 3 and word in self.abbreviations:
                corrected_words.append(self.abbreviations[word])
                continue
            
            # For longer words, use fuzzy matching against existing book titles
            if len(word) > 3:
                existing_words = {book['product_name'].lower() for book in self.books}
                matches = process.extract(word, existing_words, scorer=fuzz.ratio, limit=1)
                
                if matches and matches[0][1] > 85:  # Increase threshold to 85
                    corrected_words.append(matches[0][0])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def initialize_bot(self):
        """Initialize bot data with updated schema."""
        try:
            query = """
                SELECT 
                    product_id,
                    product_name,
                    product_url,
                    image_url,
                    product_length AS length,
                    COALESCE(total_items_sold, 0) AS total_items_sold
                FROM ProductSales
            """
            self.books = self.db_manager.execute_query(query)
            self.embeddings, self.products_data = self.load_embeddings()
            logger.info(f"Bot initialized successfully with {len(self.books)} books.")
        except Exception as e:
            logger.error(f"Error initializing bot: {e}")
            raise

    def create_embeddings(self) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Create embeddings for products."""
        try:
            products_data = [
                {
                    'product_id': book['product_id'],
                    'text': book['product_name'],
                    'book_data': book
                }
                for book in self.books
            ]

            texts = [item['text'] for item in products_data]
            embeddings = self.model.encode(texts, convert_to_tensor=True)

            embeddings_data = {
                'embeddings': embeddings.cpu().numpy(),
                'products': products_data
            }
            
            os.makedirs('data', exist_ok=True)
            with open('data/product_embeddings.pkl', 'wb') as f:
                pickle.dump(embeddings_data, f)
            
            logger.info(f"Product embeddings created and saved for {len(texts)} books.")
            return embeddings, products_data

        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise

    def load_embeddings(self) -> tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Load saved embeddings or create new ones if not found."""
        try:
            if os.path.exists('data/product_embeddings.pkl'):
                with open('data/product_embeddings.pkl', 'rb') as f:
                    data = pickle.load(f)
                    embeddings = torch.tensor(data['embeddings'])
                    products_data = data['products']
                logger.info("Loaded existing embeddings successfully.")
                return embeddings, products_data
            else:
                logger.info("No existing embeddings found. Creating new embeddings...")
                return self.create_embeddings()
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise

    def search_books(self, topic: str) -> List[Dict[str, Any]]:
        """Enhanced search function for product search with sales ranking."""
        try:
            logger.info(f"Original search topic: {topic}")
            
            # Get both abbreviated and expanded forms
            search_terms = [topic.lower()]
            
            # Check if it's an abbreviation and add expanded form
            if topic.lower() in self.abbreviations:
                search_terms.append(self.abbreviations[topic.lower()])
            # Check if it's an expanded form and add abbreviation
            else:
                for abbr, expanded in self.abbreviations.items():
                    if expanded == topic.lower():
                        search_terms.append(abbr)
                        break
            
            # Special handling for "C" to search for "C++" and "C#"
            if topic.lower() == "c":
                search_terms.extend(["c++", "c#"])
            
            logger.info(f"Search terms: {search_terms}")
            
            all_results = {}
            exact_matches = []
            fuzzy_matches = []

            # Search for each term
            for term in search_terms:
                # Use word boundaries to ensure exact word match
                search_pattern = f"\\b{term}\\b"
                query = """
                    SELECT DISTINCT
                        ps.product_id,
                        ps.product_name,
                        ps.product_url,
                        ps.image_url,
                        ps.product_length AS length,
                        COALESCE(ps.total_items_sold, 0) AS total_items_sold
                    FROM 
                        ProductSales ps
                    WHERE 
                        LOWER(ps.product_name) REGEXP %s OR
                        LOWER(ps.product_categories) REGEXP %s
                    ORDER BY 
                        ps.total_items_sold DESC
                """
                results = self.db_manager.execute_query(query, (search_pattern, search_pattern))
                
                # Process exact matches
                for result in results:
                    product_id = result['product_id']
                    if product_id not in all_results:
                        result['relevance_score'] = 1.0  # Exact match
                        exact_matches.append(result)
                        all_results[product_id] = result

            # If no exact matches, try fuzzy matching for each term
            if not exact_matches:
                query = """
                    SELECT 
                        product_id,
                        product_name,
                        product_url,
                        image_url,
                        product_length AS length,
                        COALESCE(total_items_sold, 0) AS total_items_sold
                    FROM 
                        ProductSales
                """
                all_books = self.db_manager.execute_query(query)
                
                for term in search_terms:
                    for book in all_books:
                        product_id = book['product_id']
                        if product_id not in all_results:
                            relevance_score = fuzz.partial_ratio(term, book['product_name'].lower()) / 100.0
                            if relevance_score > 0.8:  # Higher threshold for better matches
                                book['relevance_score'] = relevance_score
                                fuzzy_matches.append(book)
                                all_results[product_id] = book

            # Combine results
            final_results = exact_matches + fuzzy_matches

            # Add semantic matches for both terms
            if self.embeddings is not None:
                for term in search_terms:
                    query_embedding = self.model.encode(term, convert_to_tensor=True)
                    similarity_scores = util.pytorch_cos_sim(query_embedding, self.embeddings)[0]
                    
                    for score, product in zip(similarity_scores, self.products_data):
                        if score > 0.5:  # Semantic similarity threshold
                            product_id = product['book_data']['product_id']
                            if product_id not in all_results:
                                result = product['book_data']
                                result['semantic_score'] = float(score)
                                result['relevance_score'] = max(float(score), 
                                                              all_results.get(product_id, {}).get('relevance_score', 0))
                                all_results[product_id] = result

            # Sort results by combined relevance and sales
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: (
                x.get('relevance_score', 0),     # Primary sort by relevance
                x.get('total_items_sold', 0)     # Secondary sort by sales
            ), reverse=True)
            
            logger.info(f"Found {len(final_results)} matching books")
            return final_results

        except Exception as e:
            logger.error(f"Error in search_books: {e}")
            raise

    def _format_book_info(self, book: Dict[str, Any], rank: int) -> str:
        """Format book information for display with inline image."""
        # Replace the following URL with the actual URL for adding a book to the cart on Printrado's website
        order_url = f"https://printrado.com/?add-to-cart={book['product_id']}"
        return (
            f"📖 **#{rank} {book['product_name']}**\n"
            f"📑 *Length*: {book['length']} pages\n"
            f"🔗 [Order Now]({order_url})\n\n"
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        try:
            welcome_msg = (
                "🌟 *Welcome to Printrado recommendation assistant!* 🌟\n\n"
                "I'm here to help you find your next read 📚✨\n\n"
                "Here are some current top sellers:\n"
            )
            
            await update.message.reply_text(welcome_msg, parse_mode="Markdown")
            
            # Fetch and display only 3 top-selling books initially
            top_books = self.db_manager.get_top_selling_books(limit=3, offset=0)
            for idx, book in enumerate(top_books, 1):
                caption = self._format_book_info(book, idx)
                try:
                    await update.message.reply_photo(
                        photo=book['image_url'],
                        caption=caption,
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Error sending photo for book {book['product_name']}: {e}")
                    await update.message.reply_text(caption, parse_mode="Markdown")

            # Initialize user context for tracking the offset
            self.user_context[update.effective_user.id] = {
                'current_offset': 3  # Next offset after initial 3
            }

            await update.message.reply_text(
                "\n🔍 *Would you like to see more top-selling books?* (yes/no)",
                parse_mode="Markdown"
            )
            self.user_state[update.effective_user.id] = 'awaiting_more_books'
            
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text(
                "❌ *Sorry, there was an error starting the bot.* Please try again.",
                parse_mode="Markdown"
            )

    async def handle_more_books(self, update: Update):
        """Handle user's request for more top-selling books."""
        user_id = update.effective_user.id
        if user_id not in self.user_state:
            await self.start(update, None)
            return

        user_input = update.message.text.strip().lower()
        if user_input in ['yes', 'y']:
            user_context = self.user_context.get(user_id, {})
            current_offset = user_context.get('current_offset', 0)
            top_books = self.db_manager.get_top_selling_books(limit=5, offset=current_offset)
            
            if not top_books:
                await update.message.reply_text(
                    "❌ *No more top-selling books available.*",
                    parse_mode="Markdown"
                )
                return

            for idx, book in enumerate(top_books, start=current_offset + 1):
                caption = self._format_book_info(book, idx)
                try:
                    await update.message.reply_photo(
                        photo=book['image_url'],
                        caption=caption,
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Error sending photo for book {book['product_name']}: {e}")
                    await update.message.reply_text(caption, parse_mode="Markdown")

            # Update the offset for the next potential query
            user_context['current_offset'] = current_offset + len(top_books)
            self.user_context[user_id] = user_context

            await update.message.reply_text(
                "\n🔍 *Would you like to see more top-selling books?* (yes/no)",
                parse_mode="Markdown"
            )
        elif user_input in ['no', 'n']:
            await update.message.reply_text(
                "🔍 *What topic or subject are you interested in?* Please provide a keyword or subject.",
                parse_mode="Markdown"
            )
            self.user_state[user_id] = 'awaiting_topic'
        else:
            await update.message.reply_text(
                "❌ *Please answer with 'yes' or 'no'.*",
                parse_mode="Markdown"
            )

    async def handle_topic_search(self, update: Update, topic: str):
        """Handle topic search with user feedback."""
        try:
            original_topic = topic
            corrected_topic = self.correct_topic(topic)
            
            # Inform user if significant correction was made
            if corrected_topic != original_topic:
                await update.message.reply_text(
                    f"🔍 Searching for '*{corrected_topic}*' related books...",
                    parse_mode="Markdown"
                )
            
            search_results = self.search_books(corrected_topic)
            
            if not search_results:
                await update.message.reply_text(
                    "❌ *No books found matching your topic.*\n"
                    "🔍 *What other topic or subject are you interested in?* Please provide a keyword or subject.",
                    parse_mode="Markdown"
                )
                self.user_state[update.effective_user.id] = 'awaiting_topic'
                return

            # Store results and display them
            self.user_context[update.effective_user.id] = {
                'results': search_results,
                'topic': corrected_topic,
                'current_index': 0
            }

            # Display initial batch of results
            await self.display_search_results(update, search_results, corrected_topic)

        except Exception as e:
            logger.error(f"Error in handle_topic_search: {e}")
            await update.message.reply_text(
                "❌ *Sorry, there was an error processing your search.* Please try again.",
                parse_mode="Markdown"
            )

    async def display_search_results(self, update: Update, results: List[Dict[str, Any]], topic: str):
        """Display search results with pagination."""
        user_id = update.effective_user.id
        current_index = self.user_context[user_id]['current_index']
        batch_size = 3
        end_index = min(current_index + batch_size, len(results))

        for idx, book in enumerate(results[current_index:end_index], start=current_index + 1):
            caption = self._format_book_info(book, idx)
            try:
                await update.message.reply_photo(
                    photo=book['image_url'],
                    caption=caption,
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Error sending photo for book {book['product_name']}: {e}")
                await update.message.reply_text(caption, parse_mode="Markdown")

        self.user_context[user_id]['current_index'] = end_index

        options = "\n🔍 *What would you like to do next?*\n"
        if end_index < len(results):
            options += "- Type 'more' to see more books on this topic\n"
        options += "- Type 'new' to search for another topic\n"
        options += "- Type 'exit' to end the conversation"

        await update.message.reply_text(options, parse_mode="Markdown")
        self.user_state[user_id] = 'awaiting_next_action'

    async def handle_next_action(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle user's next action choice."""
        try:
            user_id = update.effective_user.id
            action = update.message.text.strip().lower()
            
            if action == 'more':
                if user_id in self.user_context:
                    await self.display_search_results(
                        update,
                        self.user_context[user_id]['results'],
                        self.user_context[user_id]['topic']
                    )
                else:
                    await self.start(update, context)
            elif action == 'new':
                await update.message.reply_text(
                    "🔍 *What topic or subject are you interested in?* Please provide a keyword or subject.",
                    parse_mode="Markdown"
                )
                self.user_state[user_id] = 'awaiting_topic'
            elif action == 'exit':
                await update.message.reply_text(
                    "🌟 *Thank you for using Printrado recommendation assistant* Goodbye! 👋",
                    parse_mode="Markdown"
                )
                self.user_state.pop(user_id, None)
                self.user_context.pop(user_id, None)
            else:
                await update.message.reply_text(
                    "❌ *Please choose 'more', 'new', or 'exit'.*",
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"Error in handle_next_action: {e}")
            await update.message.reply_text(
                "❌ *Sorry, there was an error processing your request.* Please try again.",
                parse_mode="Markdown"
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages based on user state."""
        try:
            user_id = update.effective_user.id
            user_input = update.message.text.strip()

            if user_id not in self.user_state:
                await self.start(update, context)
                return

            current_state = self.user_state[user_id]

            if current_state == 'awaiting_more_books':
                await self.handle_more_books(update)
            elif current_state == 'awaiting_topic':
                await self.handle_topic_search(update, user_input)
            elif current_state == 'awaiting_next_action':
                await self.handle_next_action(update, context)
            elif current_state == 'awaiting_another_topic':
                if user_input.lower() in ['yes', 'y']:
                    await update.message.reply_text(
                        "🔍 *What topic or subject are you interested in?* Please provide a keyword or subject.",
                        parse_mode="Markdown"
                    )
                    self.user_state[user_id] = 'awaiting_topic'
                elif user_input.lower() in ['no', 'n']:
                    await update.message.reply_text(
                        "🌟 *Thank you for using Printrado recommendation assistant* Goodbye! 👋",
                        parse_mode="Markdown"
                    )
                    self.user_state.pop(user_id, None)
                    self.user_context.pop(user_id, None)
                else:
                    await update.message.reply_text(
                        "❌ *Please answer with 'yes' or 'no'.*",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    "❌ *I'm not sure how to respond.* Please use /start to begin a new search.",
                    parse_mode="Markdown"
                )

        except Exception as e:
            logger.error(f"Error in handle_message: {e}")
            await update.message.reply_text(
                "❌ *Sorry, there was an error processing your message.* Please try again.",
                parse_mode="Markdown"
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the telegram bot."""
        logger.error(f"Update {update} caused error {context.error}")
        try:
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ *Sorry, something went wrong.* Please try again or start over with /start",
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"Error in error_handler: {e}")

    async def shutdown(self):
        """Cleanup resources when shutting down."""
        try:
            if hasattr(self, 'db_manager') and hasattr(self.db_manager, 'connection'):
                self.db_manager.connection.close()
            logger.info("Bot resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Main function to start the bot."""
    logger.info("Starting bot...")
    try:
        # Print current working directory and check .env file
        print(f"Current working directory: {os.getcwd()}")
        print(f".env file exists: {os.path.exists('.env')}")

        # Download RDS certificate if needed
        download_rds_cert()

        config = Config()
        bot = BookBot(config)
        application = Application.builder().token(config.API_TOKEN).build()
        application.add_handler(CommandHandler("start", bot.start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
        application.add_error_handler(bot.error_handler)
        logger.info("Bot started successfully!")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise
    finally:
        if 'bot' in locals():
            import asyncio
            asyncio.run(bot.shutdown())

if __name__ == "__main__":
    main()