import mysql.connector
import os
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            user=os.getenv('DB_USER', 'examuser'),
            password=os.getenv('DB_PASSWORD', 'exampass'),
            database=os.getenv('DB_NAME', 'exam_proctoring'),
            autocommit=True
        )
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Database connection error: {e}")
        return None

def store_face_encoding(user_id: str, encoding: list) -> bool:
    """Store face encoding in database"""
    try:
        connection = get_db_connection()
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # Convert encoding to json string then binary
        encoding_str = json.dumps(encoding)
        encoding_bytes = bytes(encoding_str, 'utf-8')
        
        cursor.execute(
            "UPDATE users SET face_embedding = %s WHERE email = %s",
            (encoding_bytes, user_id)
        )
        
        cursor.close()
        connection.close()
        
        return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Error storing face encoding: {e}")
        return False

def get_face_encoding(user_id: int) -> Optional[list]:
    """Get face encoding from database"""
    try:
        connection = get_db_connection()
        if not connection:
            return None
        
        cursor = connection.cursor()
        
        cursor.execute(
            "SELECT face_embedding FROM users WHERE id = %s",
            (user_id,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if result and result[0]:
            # Convert binary back to list
            encoding_str = result[0].decode('utf-8')
            try:
                return json.loads(encoding_str)
            except json.JSONDecodeError:
                # Fallback for old data saved with str()
                import ast
                return ast.literal_eval(encoding_str)
        
        return None
    except Exception as e:
        logger.error(f"Error getting face encoding: {e}")
        return None