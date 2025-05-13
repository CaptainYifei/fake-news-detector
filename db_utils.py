import sqlite3
import os
import hashlib
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 数据库文件路径
DB_PATH = "factcheck.db"

def init_db():
    """初始化数据库，创建所需的表"""
    # 检查数据库文件是否存在
    db_exists = os.path.exists(DB_PATH)
    
    # 连接到数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 如果数据库不存在，创建表
    if not db_exists:
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 创建历史记录表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_text TEXT NOT NULL,
            claim TEXT NOT NULL,
            verdict TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # 创建证据表（关联到历史记录）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            history_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            similarity REAL,
            FOREIGN KEY (history_id) REFERENCES history (id)
        )
        ''')
        
        conn.commit()
    
    conn.close()

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    使用SHA-256哈希密码并加盐
    
    Args:
        password: 明文密码
        salt: 可选的盐值，如果不提供则生成新的
        
    Returns:
        Tuple containing (password_hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)  # 生成32字符（16字节）的随机盐值
    
    # 组合密码和盐值，然后哈希
    password_with_salt = password + salt
    password_hash = hashlib.sha256(password_with_salt.encode()).hexdigest()
    
    return password_hash, salt

def create_user(username: str, password: str) -> bool:
    """
    创建新用户
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        创建是否成功
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查用户名是否已存在
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False  # 用户名已存在
        
        # 哈希密码
        password_hash, salt = hash_password(password)
        
        # 插入新用户
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
            (username, password_hash, salt)
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"创建用户时出错: {str(e)}")
        return False

def verify_user(username: str, password: str) -> Optional[int]:
    """
    验证用户凭据
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        用户ID（如果验证成功），否则为None
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 获取用户记录
        cursor.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?", 
            (username,)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return None  # 用户不存在
        
        user_id, stored_hash, salt = user
        
        # 使用相同的盐值哈希输入的密码
        calculated_hash, _ = hash_password(password, salt)
        
        # 比较哈希值
        if calculated_hash == stored_hash:
            return user_id
        else:
            return None
    except Exception as e:
        print(f"验证用户时出错: {str(e)}")
        return None

def save_fact_check(
    user_id: int, 
    original_text: str, 
    claim: str, 
    verdict: str, 
    reasoning: str,
    evidence_chunks: List[Dict[str, Any]]
) -> int:
    """
    保存事实核查结果到历史记录
    
    Args:
        user_id: 用户ID
        original_text: 原始文本
        claim: 提取的声明
        verdict: 结论（TRUE/FALSE/PARTIALLY TRUE等）
        reasoning: 推理过程
        evidence_chunks: 证据块列表
        
    Returns:
        历史记录ID
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 插入历史记录
        cursor.execute(
            "INSERT INTO history (user_id, original_text, claim, verdict, reasoning) VALUES (?, ?, ?, ?, ?)",
            (user_id, original_text, claim, verdict, reasoning)
        )
        
        # 获取新插入的历史记录ID
        history_id = cursor.lastrowid
        
        # 插入证据
        for chunk in evidence_chunks:
            cursor.execute(
                "INSERT INTO evidence (history_id, text, source, similarity) VALUES (?, ?, ?, ?)",
                (history_id, chunk['text'], chunk['source'], chunk.get('similarity', 0))
            )
        
        conn.commit()
        conn.close()
        return history_id
    except Exception as e:
        print(f"保存事实核查记录时出错: {str(e)}")
        return -1

def get_user_history(user_id: int, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    """
    获取用户的事实核查历史记录
    
    Args:
        user_id: 用户ID
        limit: 返回的最大记录数
        offset: 分页偏移量
        
    Returns:
        历史记录列表
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 启用行工厂，使结果可以像字典一样访问
        cursor = conn.cursor()
        
        # 获取历史记录
        cursor.execute(
            """
            SELECT id, original_text, claim, verdict, reasoning, created_at 
            FROM history 
            WHERE user_id = ? 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """, 
            (user_id, limit, offset)
        )
        
        history_rows = cursor.fetchall()
        history = []
        
        for row in history_rows:
            history_item = dict(row)
            
            # 获取相关的证据
            cursor.execute(
                "SELECT text, source, similarity FROM evidence WHERE history_id = ?",
                (row['id'],)
            )
            
            evidence_rows = cursor.fetchall()
            evidence = [dict(evidence_row) for evidence_row in evidence_rows]
            
            history_item['evidence'] = evidence
            history.append(history_item)
        
        conn.close()
        return history
    except Exception as e:
        print(f"获取用户历史记录时出错: {str(e)}")
        return []

def get_history_by_id(history_id: int) -> Optional[Dict[str, Any]]:
    """
    通过ID获取特定的历史记录
    
    Args:
        history_id: 历史记录ID
        
    Returns:
        历史记录字典，如果未找到则为None
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 获取历史记录
        cursor.execute(
            """
            SELECT id, user_id, original_text, claim, verdict, reasoning, created_at 
            FROM history 
            WHERE id = ?
            """, 
            (history_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        history_item = dict(row)
        
        # 获取相关的证据
        cursor.execute(
            "SELECT text, source, similarity FROM evidence WHERE history_id = ?",
            (history_id,)
        )
        
        evidence_rows = cursor.fetchall()
        evidence = [dict(evidence_row) for evidence_row in evidence_rows]
        
        history_item['evidence'] = evidence
        
        conn.close()
        return history_item
    except Exception as e:
        print(f"通过ID获取历史记录时出错: {str(e)}")
        return None

def count_user_history(user_id: int) -> int:
    """
    计算用户的历史记录总数
    
    Args:
        user_id: 用户ID
        
    Returns:
        历史记录数量
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    except Exception as e:
        print(f"计算用户历史记录数时出错: {str(e)}")
        return 0